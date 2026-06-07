"""자율 주차 알고리즘 클라이언트 — RRT* + Stanley Controller

이 스크립트는 시뮬레이터(`demo_self_parking_sim.py`)가 개방한 TCP 포트에
JSONL 프로토콜로 접속한다. 통신 흐름은 아래와 같다.

0. 연결이 성립하면 시뮬레이터가 정적 맵(`{"map": ...}`)을 먼저 보낸다.
1. 이후 시뮬레이터가 매 스텝마다 관측 패킷(`obs`)을 보낸다. 주요 필드:
   - `t`: 시뮬레이터 시간이 초 단위로 증가
   - `state`: 현재 차량 위치(x, y), 각도(yaw), 속도(v)
   - `target_slot`: 목표 주차 슬롯의 직사각형 좌표(xmin, xmax, ymin, ymax)
   - `limits`: 차량 제한(타임스텝, 휠베이스, 조향/가감속 한계 등)
2. 학생 알고리즘은 이 정보를 이용해 다음 명령(`cmd`)을 계산하고,
   `{"steer", "accel", "brake", "gear"}` 값을 JSON 한 줄로 응답한다.
3. 시뮬레이터는 받은 명령을 차량 모델에 적용하고 다음 관측을 보낸다.

알고리즘 개요:
  경로 계획 — RRT* (Rapidly-exploring Random Tree Star)
    · 무작위 샘플링으로 충돌 없는 경로를 탐색한다.
    · 일반 RRT 대비 rewire 단계로 경로 비용을 지속적으로 개선한다.
    · 목표 근방 편향(goal_bias) 샘플링으로 수렴 속도를 높인다.

  제어 — Stanley Controller (전륜 기준)
    · 경로의 가장 가까운 점까지의 횡방향 오차(cross-track error)와
      헤딩 오차를 동시에 보정해 경로 추종 정확도가 높다.
    · 속도 연동 게인으로 저속 시 안정성을 확보한다.

  후진 진입 — 슬롯 방향 추정 후 목표 yaw를 계산하고,
    진입 전 정렬 포인트를 RRT* 중간 목표로 설정하여 두 단계로 주차한다.
"""

import argparse
import json
import math
import os
import random
import signal
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 차량 / 알고리즘 상수
# ─────────────────────────────────────────────────────────────────────────────
WHEELBASE   = 2.6      # 축간거리 (m)
CAR_HALF_W  = 0.9      # 차폭 절반 + 마진 (m)
CAR_FRONT   = 1.7      # 전방 오버행 + 마진 (m)
CAR_REAR    = 1.5      # 후방 오버행 + 마진 (m)

MAX_STEER_RAD = math.radians(35)

# RRT* 파라미터
RRT_MAX_ITER   = 4000   # 최대 반복 횟수
RRT_STEP       = 2.0    # 한 스텝 확장 거리 (m)
RRT_GOAL_BIAS  = 0.15   # 목표점 직접 샘플링 확률
RRT_GOAL_DIST  = 1.5    # 목표 도달 판정 거리 (m)
RRT_REWIRE_R   = 5.0    # rewire 탐색 반경 (m)
COLLISION_STEP = 0.5    # 충돌 검사 간격 (m)

# Stanley 게인
K_STANLEY    = 1.2      # 횡방향 오차 게인
K_SOFT       = 0.5      # 소프트닝 상수 (저속 발산 방지)
K_HEAD       = 1.0      # 헤딩 오차 게인

# 속도 프로파일
CRUISE_SPEED = 2.5      # 순항 속도 (m/s)
ENTRY_SPEED  = 0.6      # 슬롯 진입 속도 (m/s)
DECEL_DIST   = 5.0      # 목표 전 감속 시작 거리 (m)
GOAL_SPEED   = 0.3      # 최종 목표 속도 (m/s)

# 도착 판정
ARRIVE_DIST  = 0.7      # 목표까지 거리 (m)
ARRIVE_YAW   = math.radians(18)  # 허용 yaw 오차


# ─────────────────────────────────────────────────────────────────────────────
# 수학 유틸
# ─────────────────────────────────────────────────────────────────────────────
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def wrap_angle(a: float) -> float:
    """각도를 [-π, π]로 정규화."""
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def dist2(ax, ay, bx, by) -> float:
    return math.hypot(ax - bx, ay - by)

def lerp(a, b, t):
    return a + (b - a) * t


# ─────────────────────────────────────────────────────────────────────────────
# 충돌 검사 — 차량 박스 vs 장애물 AABB
# ─────────────────────────────────────────────────────────────────────────────
def _car_aabb_rotated(cx, cy, yaw) -> List[Tuple[float, float]]:
    """차량 박스의 4개 꼭짓점을 반환 (회전 적용)."""
    pts = [
        ( CAR_FRONT,  CAR_HALF_W),
        ( CAR_FRONT, -CAR_HALF_W),
        (-CAR_REAR,  -CAR_HALF_W),
        (-CAR_REAR,   CAR_HALF_W),
    ]
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    return [
        (cx + p[0]*cos_y - p[1]*sin_y,
         cy + p[0]*sin_y + p[1]*cos_y)
        for p in pts
    ]

def _poly_aabb(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return min(xs), max(xs), min(ys), max(ys)

def _separating_axis(poly_a, poly_b) -> bool:
    """SAT(분리 축 이론)으로 두 볼록 다각형의 교차 여부를 반환."""
    for poly in (poly_a, poly_b):
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i+1) % n]
            # 법선 벡터
            nx, ny = -(by - ay), (bx - ax)
            min_a = min(nx*p[0] + ny*p[1] for p in poly_a)
            max_a = max(nx*p[0] + ny*p[1] for p in poly_a)
            min_b = min(nx*p[0] + ny*p[1] for p in poly_b)
            max_b = max(nx*p[0] + ny*p[1] for p in poly_b)
            if max_a < min_b or max_b < min_a:
                return False   # 분리 축 발견 → 교차 없음
    return True   # 교차


def _rect_poly(xmin, xmax, ymin, ymax):
    return [(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]


class CollisionChecker:
    """장애물 목록 대비 차량 박스 충돌 검사."""

    def __init__(self):
        self.obstacles: List[Tuple] = []   # (xmin, xmax, ymin, ymax)
        self.extent: Tuple = (0, 100, 0, 100)
        self.target_rect: Optional[Tuple] = None

    def build(self, mp: Dict, target_slot: Optional[Tuple] = None):
        self.obstacles.clear()
        self.target_rect = target_slot
        ext = mp.get("extent", (0, 75, 0, 50))
        self.extent = tuple(map(float, ext))

        # 외벽
        for r in mp.get("walls_rects", []):
            self.obstacles.append(tuple(map(float, r)))

        # 주차선 → 얇은 AABB
        LHW = 0.2
        ts = target_slot
        for seg in mp.get("lines", []):
            x1, y1, x2, y2 = map(float, seg)
            # 목표 슬롯 경계선 제외 (진입 허용)
            if ts:
                if abs(y1-y2) < 1e-3:
                    if (abs(y1-ts[2]) < 0.6 or abs(y1-ts[3]) < 0.6) and \
                       min(x1,x2) <= ts[1]+1 and max(x1,x2) >= ts[0]-1:
                        continue
                if abs(x1-x2) < 1e-3 and ts[0]-0.5 <= x1 <= ts[1]+0.5:
                    continue
            if abs(x1-x2) < 1e-3:
                self.obstacles.append((min(x1,x2)-LHW, max(x1,x2)+LHW,
                                       min(y1,y2), max(y1,y2)))
            else:
                self.obstacles.append((min(x1,x2), max(x1,x2),
                                       min(y1,y2)-LHW, max(y1,y2)+LHW))

        # 점유 슬롯
        slots = mp.get("slots", [])
        occ   = mp.get("occupied_idx", [])
        for sl, is_occ in zip(slots, occ):
            if is_occ:
                self.obstacles.append(tuple(map(float, sl)))

    def is_free(self, x, y, yaw) -> bool:
        """해당 자세가 충돌 없으면 True."""
        xmin, xmax, ymin, ymax = self.extent
        if not (xmin + 0.3 <= x <= xmax - 0.3 and
                ymin + 0.3 <= y <= ymax - 0.3):
            return False

        car_poly = _car_aabb_rotated(x, y, yaw)
        caabb    = _poly_aabb(car_poly)

        # 목표 슬롯 안에 완전히 들어간 경우는 충돌 면제
        if self.target_rect:
            tr = self.target_rect
            if (tr[0] <= caabb[0] and caabb[1] <= tr[1] and
                    tr[2] <= caabb[2] and caabb[3] <= tr[3]):
                return True

        for obs in self.obstacles:
            ox0, ox1, oy0, oy1 = obs
            # 빠른 AABB 사전 필터
            if (caabb[1] < ox0 or caabb[0] > ox1 or
                    caabb[3] < oy0 or caabb[2] > oy1):
                continue
            obs_poly = _rect_poly(ox0, ox1, oy0, oy1)
            if _separating_axis(car_poly, obs_poly):
                return False
        return True

    def is_path_free(self, x0, y0, x1, y1, yaw0, yaw1) -> bool:
        """두 점 사이 직선 경로가 충돌 없으면 True."""
        d = dist2(x0, y0, x1, y1)
        if d < 1e-6:
            return self.is_free(x0, y0, yaw0)
        steps = max(2, int(d / COLLISION_STEP) + 1)
        for i in range(steps + 1):
            t = i / steps
            xi   = lerp(x0, x1, t)
            yi   = lerp(y0, y1, t)
            yawi = lerp(yaw0, yaw1, t)
            if not self.is_free(xi, yi, yawi):
                return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# RRT* 계획기
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RRTNode:
    x:      float
    y:      float
    yaw:    float
    cost:   float = 0.0
    parent: int   = -1   # 부모 인덱스


def _steer(nx, ny, sx, sy, step) -> Tuple[float, float]:
    """(nx,ny)에서 (sx,sy) 방향으로 step 거리만큼 이동한 점."""
    d = dist2(nx, ny, sx, sy)
    if d < 1e-6: return sx, sy
    r = min(step, d) / d
    return nx + (sx - nx) * r, ny + (sy - ny) * r


def rrt_star(
    sx, sy, syaw,
    gx, gy, gyaw,
    checker: CollisionChecker,
    extent:  Tuple,
) -> Optional[List[Tuple[float, float, float]]]:
    """
    RRT* 경로 계획.
    반환: [(x, y, yaw), ...] 순서의 경로, 실패 시 None.
    """
    xmin, xmax, ymin, ymax = extent
    nodes: List[RRTNode] = [RRTNode(sx, sy, syaw, 0.0, -1)]

    best_goal_idx  = -1
    best_goal_cost = math.inf

    for _ in range(RRT_MAX_ITER):
        # ── 샘플링 ─────────────────────────────────────────────────────
        if random.random() < RRT_GOAL_BIAS:
            rx, ry = gx, gy
        else:
            rx = random.uniform(xmin + 0.5, xmax - 0.5)
            ry = random.uniform(ymin + 0.5, ymax - 0.5)

        # ── 가장 가까운 노드 탐색 ────────────────────────────────────────
        dists = [dist2(n.x, n.y, rx, ry) for n in nodes]
        nn_idx = min(range(len(nodes)), key=lambda i: dists[i])
        nn = nodes[nn_idx]

        # ── 스티어링 ─────────────────────────────────────────────────────
        nx, ny = _steer(nn.x, nn.y, rx, ry, RRT_STEP)
        # 선분 방향으로 yaw 근사
        nyaw = math.atan2(ny - nn.y, nx - nn.x)

        if not checker.is_path_free(nn.x, nn.y, nx, ny, nn.yaw, nyaw):
            continue

        # ── 근방 노드 목록 (rewire용) ─────────────────────────────────────
        near_idxs = [
            i for i, n in enumerate(nodes)
            if dist2(n.x, n.y, nx, ny) <= RRT_REWIRE_R
        ]

        # ── 최소 비용 부모 선택 ──────────────────────────────────────────
        best_parent = nn_idx
        best_cost   = nn.cost + dist2(nn.x, nn.y, nx, ny)
        for ni in near_idxs:
            n = nodes[ni]
            c = n.cost + dist2(n.x, n.y, nx, ny)
            if c < best_cost and checker.is_path_free(n.x, n.y, nx, ny, n.yaw, nyaw):
                best_parent = ni
                best_cost   = c

        new_node = RRTNode(nx, ny, nyaw, best_cost, best_parent)
        new_idx  = len(nodes)
        nodes.append(new_node)

        # ── Rewire ───────────────────────────────────────────────────────
        for ni in near_idxs:
            n = nodes[ni]
            potential = best_cost + dist2(nx, ny, n.x, n.y)
            if potential < n.cost and \
               checker.is_path_free(nx, ny, n.x, n.y, nyaw, n.yaw):
                nodes[ni] = RRTNode(n.x, n.y, n.yaw, potential, new_idx)

        # ── 목표 도달 확인 ────────────────────────────────────────────────
        d_goal = dist2(nx, ny, gx, gy)
        if d_goal < RRT_GOAL_DIST:
            total_cost = best_cost + d_goal
            if total_cost < best_goal_cost:
                # 목표 노드 추가
                goal_node = RRTNode(gx, gy, gyaw, total_cost, new_idx)
                goal_idx  = len(nodes)
                nodes.append(goal_node)
                best_goal_idx  = goal_idx
                best_goal_cost = total_cost

    if best_goal_idx < 0:
        return None   # 탐색 실패

    # ── 경로 역추적 ──────────────────────────────────────────────────────
    path = []
    idx  = best_goal_idx
    while idx >= 0:
        n = nodes[idx]
        path.append((n.x, n.y, n.yaw))
        idx = n.parent
    path.reverse()

    # ── 경로 yaw 재보간 (진행 방향 기반) ─────────────────────────────────
    smoothed = _smooth_yaw(path, gyaw)
    return smoothed


def _smooth_yaw(path, final_yaw) -> List[Tuple[float, float, float]]:
    """연속된 경로점의 yaw를 진행 방향 각도로 재계산."""
    if len(path) < 2:
        return path
    result = []
    for i, (x, y, _) in enumerate(path):
        if i < len(path) - 1:
            nx, ny = path[i+1][0], path[i+1][1]
            yaw = math.atan2(ny - y, nx - x)
        else:
            yaw = final_yaw
        result.append((x, y, yaw))
    return result


def _path_speed_profile(
    path: List[Tuple[float, float, float]],
    cruise: float = CRUISE_SPEED,
    goal:   float = GOAL_SPEED,
    decel:  float = DECEL_DIST,
) -> List[float]:
    """각 경로점에 목표 속도를 할당."""
    n = len(path)
    speeds = [cruise] * n
    # 끝에서부터 decel 구간 감속
    cum = 0.0
    for i in range(n-1, 0, -1):
        seg = dist2(path[i][0], path[i][1], path[i-1][0], path[i-1][1])
        cum += seg
        if cum >= decel:
            break
        ratio = cum / decel            # 0(목표) → 1(감속 시작)
        speeds[i] = goal + (cruise - goal) * ratio
    speeds[-1] = goal
    return speeds


# ─────────────────────────────────────────────────────────────────────────────
# Stanley Controller
# ─────────────────────────────────────────────────────────────────────────────
def stanley_steer(
    ex, ey, eyaw, ev,
    path: List[Tuple[float, float, float]],
    path_idx: int,
    wheelbase: float,
    max_steer: float,
    reverse: bool = False,
) -> Tuple[float, int]:
    """
    Stanley 조향 계산.

    반환: (steer_rad, 업데이트된 경로 인덱스)

    원리:
      δ = ψ_e + arctan(k * e_fa / (k_soft + v))

      ψ_e  : 헤딩 오차 (경로 방향 - 차량 yaw)
      e_fa : 전륜 위치에서 경로까지의 횡방향 오차 (부호 포함)
      k    : 횡방향 게인
      v    : 현재 속도 (절댓값)
    """
    n  = len(path)
    speed = max(abs(ev), 0.1)

    # 전륜 위치 계산
    sign = -1.0 if reverse else 1.0
    fx   = ex + sign * wheelbase * math.cos(eyaw)
    fy   = ey + sign * wheelbase * math.sin(eyaw)

    # 가장 가까운 경로점 탐색 (현재 idx 전후 탐색)
    search_start = max(0, path_idx - 5)
    search_end   = min(n, path_idx + 30)
    best_idx = path_idx
    best_d   = math.inf
    for i in range(search_start, search_end):
        d = dist2(fx, fy, path[i][0], path[i][1])
        if d < best_d:
            best_d = d; best_idx = i

    # 다음 인덱스 추적 (일정 거리 이내에 있으면 전진)
    while best_idx < n - 1 and dist2(ex, ey, path[best_idx][0], path[best_idx][1]) < 0.8:
        best_idx += 1

    ref_x, ref_y, ref_yaw = path[best_idx]

    # 헤딩 오차
    if reverse:
        heading_err = wrap_angle(wrap_angle(ref_yaw + math.pi) - eyaw)
    else:
        heading_err = wrap_angle(ref_yaw - eyaw)

    # 횡방향 오차 (전륜 기준, 부호 포함)
    dx = fx - ref_x; dy = fy - ref_y
    cross_err = -math.sin(ref_yaw) * dx + math.cos(ref_yaw) * dy

    # Stanley 공식
    raw_steer = K_HEAD * heading_err + \
                math.atan2(K_STANLEY * cross_err, K_SOFT + speed)

    if reverse:
        raw_steer = -raw_steer

    return clamp(raw_steer, -max_steer, max_steer), best_idx


# ─────────────────────────────────────────────────────────────────────────────
# 슬롯 분석 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _slot_center(slot) -> Tuple[float, float]:
    return (slot[0]+slot[1])/2, (slot[2]+slot[3])/2

def _slot_yaw(slot, reverse_in: bool) -> float:
    """슬롯 장축 방향 yaw. reverse_in이면 180° 반전."""
    w = slot[1] - slot[0]
    h = slot[3] - slot[2]
    base = math.pi/2 if h >= w else 0.0
    return wrap_angle(base + (math.pi if reverse_in else 0.0))

def _slot_entry_point(slot, gyaw, offset=5.0) -> Tuple[float, float]:
    """슬롯 중심에서 진입 방향 반대로 offset만큼 떨어진 정렬 포인트."""
    cx, cy = _slot_center(slot)
    # 진입 방향 = gyaw, 정렬 포인트는 그 반대쪽
    return cx - offset * math.cos(gyaw), cy - offset * math.sin(gyaw)


# ─────────────────────────────────────────────────────────────────────────────
# PlannerSkeleton — 핵심 학생 구현 클래스
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PlannerSkeleton:
    """RRT* 경로 계획 + Stanley 경로 추종 자율 주차 플래너."""

    map_data:  Optional[Dict[str, Any]] = None
    waypoints: List[Tuple[float, float]] = None   # 뼈대 호환용 (미사용)

    # ── 내부 상태 ──────────────────────────────────────────────────────────
    _checker:   CollisionChecker = field(default_factory=CollisionChecker)
    _extent:    Tuple = field(default=(0,75,0,50))
    _slots:     List  = field(default_factory=list)
    _occ_idx:   List  = field(default_factory=list)

    # 현재 목표
    _target_slot:   Optional[Tuple] = field(default=None)
    _target_yaw:    float = field(default=math.pi/2)
    _target_center: Optional[Tuple] = field(default=None)

    # 경로 (1단계: 정렬 포인트까지 / 2단계: 슬롯 진입)
    _phase:         int  = field(default=0)   # 0=계획 전, 1=정렬, 2=진입, 3=도착
    _path:          List = field(default_factory=list)
    _speeds:        List = field(default_factory=list)
    _path_idx:      int  = field(default=0)
    _reverse:       bool = field(default=False)

    def __post_init__(self) -> None:
        if self.waypoints is None:
            self.waypoints = []
        # dataclass field default_factory가 __init__에서 처리되지만
        # 명시적으로 초기화해 둔다.
        self._checker  = CollisionChecker()
        self._extent   = (0, 75, 0, 50)
        self._slots    = []
        self._occ_idx  = []
        self._target_slot   = None
        self._target_center = None
        self._path     = []
        self._speeds   = []
        self._path_idx = 0
        self._phase    = 0
        self._reverse  = False

    # ── 맵 수신 ────────────────────────────────────────────────────────────
    def set_map(self, map_payload: Dict[str, Any]) -> None:
        """시뮬레이터에서 한 번 보내주는 정적 맵 정보를 저장."""
        self.map_data = map_payload
        self.waypoints.clear()

        ext = map_payload.get("extent", (0, 75, 0, 50))
        self._extent  = tuple(map(float, ext))
        self._slots   = [tuple(map(float, s)) for s in map_payload.get("slots", [])]
        self._occ_idx = [bool(v) for v in map_payload.get("occupied_idx", [])]

        free_cnt = sum(1 for v in self._occ_idx if not v)
        print(f"[planner] map loaded  extent={self._extent}  "
              f"slots={len(self._slots)}  free={free_cnt}")

        # 목표 슬롯 초기화 → 다음 obs에서 갱신
        self._target_slot   = None
        self._target_center = None
        self._phase = 0

    # ── 경로 계획 ──────────────────────────────────────────────────────────
    def compute_path(self, obs: Dict[str, Any]) -> None:
        """관측값과 맵을 기반으로 목표 슬롯까지의 경로를 생성합니다."""
        if self.map_data is None or self._target_slot is None:
            return

        state = obs.get("state", {})
        sx    = float(state.get("x",   0.0))
        sy    = float(state.get("y",   0.0))
        syaw  = float(state.get("yaw", math.pi/2))

        slot     = self._target_slot
        gyaw     = self._target_yaw
        cx, cy   = self._target_center

        # 슬롯 진입 방향 결정
        # · 슬롯 장축이 y 방향(세로 슬롯)이면 위 또는 아래에서 전진/후진
        # · 자유 슬롯이 1개뿐이면 후진 주차 가정
        free_cnt = sum(1 for v in self._occ_idx if not v)
        self._reverse = (free_cnt == 1)

        # 장애물 맵 구축
        self._checker.build(self.map_data, target_slot=slot)

        # ── Phase 1: 현재 위치 → 정렬 포인트 (RRT*) ──────────────────────
        align_x, align_y = _slot_entry_point(slot, gyaw, offset=6.0)
        align_yaw = wrap_angle(gyaw + (0 if not self._reverse else math.pi))

        print(f"[planner] RRT* Phase1  "
              f"({sx:.1f},{sy:.1f}) → align({align_x:.1f},{align_y:.1f})")
        path1 = rrt_star(sx, sy, syaw,
                         align_x, align_y, align_yaw,
                         self._checker, self._extent)

        if path1 is None:
            print("[planner] RRT* Phase1 failed — 직진 폴백")
            path1 = [(sx, sy, syaw), (align_x, align_y, align_yaw)]

        # ── Phase 2: 정렬 포인트 → 슬롯 중심 (직선, 후진 가능) ──────────
        # 슬롯 근방 직선 진입은 단순 직선 경로로 처리
        # (RRT* 대신 직선을 쓰는 이유: 좁은 슬롯에서 샘플 수렴이 어려움)
        entry_gear_yaw = gyaw if not self._reverse else wrap_angle(gyaw + math.pi)
        path2 = [(align_x, align_y, entry_gear_yaw),
                 (cx, cy, gyaw)]

        # ── 두 단계 경로 연결 ─────────────────────────────────────────────
        # path1 마지막점과 path2 시작점이 거의 같으므로 중복 제거
        full_path = path1 + path2[1:]
        self._path     = full_path
        self._speeds   = _path_speed_profile(full_path,
                                              cruise=CRUISE_SPEED,
                                              goal=GOAL_SPEED,
                                              decel=DECEL_DIST)
        self._path_idx = 0
        self._phase    = 1
        print(f"[planner] path ready  nodes={len(full_path)}  "
              f"reverse={self._reverse}")

    # ── 제어 ───────────────────────────────────────────────────────────────
    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """경로를 따라가기 위한 조향/가감속 명령을 산출합니다."""
        state  = obs.get("state",  {})
        limits = obs.get("limits", {})

        ex   = float(state.get("x",   0.0))
        ey   = float(state.get("y",   0.0))
        eyaw = float(state.get("yaw", 0.0))
        ev   = float(state.get("v",   0.0))

        L_wb  = float(limits.get("L",        WHEELBASE))
        ms    = float(limits.get("maxSteer",  MAX_STEER_RAD))

        # ── 목표 슬롯 갱신 ────────────────────────────────────────────────
        self._update_target(obs)

        if self._target_slot is None:
            return {"steer": 0.0, "accel": 0.0, "brake": 0.5, "gear": "D"}

        # ── 경로가 없으면 계획 ────────────────────────────────────────────
        if self._phase == 0:
            self.compute_path(obs)
            if self._phase == 0:
                return {"steer": 0.0, "accel": 0.0, "brake": 0.5, "gear": "D"}

        # ── 도착 ──────────────────────────────────────────────────────────
        if self._phase == 3:
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        cx, cy = self._target_center
        d_goal  = dist2(ex, ey, cx, cy)
        yaw_err = abs(wrap_angle(eyaw - self._target_yaw))

        if d_goal < ARRIVE_DIST and yaw_err < ARRIVE_YAW:
            print("[planner] arrived!")
            self._phase = 3
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        # ── Stanley 조향 ──────────────────────────────────────────────────
        # 슬롯 진입 구간(경로 후반 ~30%)은 후진 여부 반영
        path = self._path
        n    = len(path)
        near_end   = self._path_idx >= int(n * 0.7)
        use_reverse = self._reverse and near_end

        steer, new_idx = stanley_steer(
            ex, ey, eyaw, ev,
            path, self._path_idx,
            L_wb, ms,
            reverse=use_reverse,
        )
        self._path_idx = new_idx

        # ── 속도 제어 (비례 제어) ─────────────────────────────────────────
        idx_clamped = min(self._path_idx, len(self._speeds) - 1)
        # 목표 전 감속 — 거리 기반으로 속도 상한 추가
        speed_limit = ENTRY_SPEED if d_goal < DECEL_DIST else CRUISE_SPEED
        desired_v   = min(self._speeds[idx_clamped], speed_limit)

        # 기어 결정
        gear = "R" if use_reverse else "D"

        # 기어 방향과 실제 속도 부호가 다르면 브레이크
        if gear == "D" and ev < -0.1:
            return {"steer": steer, "accel": 0.0, "brake": 0.8, "gear": gear}
        if gear == "R" and ev >  0.1:
            return {"steer": steer, "accel": 0.0, "brake": 0.8, "gear": gear}

        speed = abs(ev)
        err   = desired_v - speed
        if err > 0.15:
            accel = clamp(0.15 + 0.25 * err, 0.0, 0.65)
            brake = 0.0
        elif err < -0.2:
            accel = 0.0
            brake = clamp(0.1 + 0.3 * (-err), 0.0, 0.8)
        else:
            accel = 0.04
            brake = 0.0

        return {"steer": steer, "accel": accel, "brake": brake, "gear": gear}

    # ── 목표 슬롯 내부 갱신 ────────────────────────────────────────────────
    def _update_target(self, obs: Dict[str, Any]) -> None:
        raw = obs.get("target_slot")
        if raw and len(raw) == 4:
            slot = tuple(map(float, raw))
        elif self._slots:
            free = [i for i, o in enumerate(self._occ_idx) if not o]
            slot = self._slots[free[0]] if free else self._slots[0]
        else:
            return

        # 슬롯이 바뀐 경우 재계획
        if self._target_slot is None or \
           any(abs(a-b) > 1e-4 for a, b in zip(self._target_slot, slot)):
            free_cnt = sum(1 for v in self._occ_idx if not v)
            reverse  = (free_cnt == 1)
            self._target_slot   = slot
            self._target_center = _slot_center(slot)
            self._target_yaw    = _slot_yaw(slot, reverse_in=reverse)
            self._phase    = 0
            self._path     = []
            self._path_idx = 0
            print(f"[planner] new target  slot={slot}  "
                  f"yaw={math.degrees(self._target_yaw):.0f}°")


# ─────────────────────────────────────────────────────────────────────────────
# 전역 planner 인스턴스 (run_session에서 사용)
# ─────────────────────────────────────────────────────────────────────────────
planner = PlannerSkeleton()

STUDENT_REPLAY_DIR = "student_replays"


def _slugify(text: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    slug = slug.strip("_")
    return slug or "session"


def save_student_replay(
    frames: List[Dict[str, Any]], meta: Dict[str, Any]
) -> Optional[str]:
    if not frames:
        return None
    try:
        os.makedirs(STUDENT_REPLAY_DIR, exist_ok=True)
    except Exception as exc:
        print(f"[algo] replay dir error: {exc}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_key   = meta.get("map_key") or meta.get("map_name") or "session"
    filename  = f"{timestamp}_{_slugify(map_key)}.json"
    path      = os.path.join(STUDENT_REPLAY_DIR, filename)
    payload   = {"meta": meta, "frames": frames}
    try:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        print(f"[algo] replay saved: {path}")
        return path
    except Exception as exc:
        print(f"[algo] replay save failed: {exc}")
        return None


def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return planner.compute_control(obs)
    except Exception as exc:
        print(f"[algo] planner_step error: {exc}")
        return {"steer": 0.0, "accel": 0.0, "brake": 0.5, "gear": "D"}


# ─────────────────────────────────────────────────────────────────────────────
# TCP 통신 (뼈대 코드 그대로 유지)
# ─────────────────────────────────────────────────────────────────────────────
def run_session(sock: socket.socket, peer: Tuple[str, int]) -> None:
    """시뮬레이터와의 단일 TCP 세션을 처리."""
    print(f"[algo] connected to simulator at {peer}")
    buffer = b""
    frames: List[Dict[str, Any]] = []
    session_meta: Dict[str, Any] = {
        "peer":       {"host": peer[0], "port": peer[1]},
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "map_key":    None,
        "map_name":   None,
    }

    try:
        while True:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                print("[algo] simulator closed the connection")
                break

            buffer += chunk

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue

                try:
                    packet = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    print(f"[algo] bad JSON from simulator: {exc}")
                    continue

                if isinstance(packet, dict) and "map" in packet:
                    planner.set_map(packet["map"])
                    print("[algo] received static map payload")
                    mp = packet["map"]
                    session_meta["map_key"]     = mp.get("key")
                    session_meta["map_name"]    = mp.get("name")
                    session_meta["map_extent"]  = mp.get("extent")
                    session_meta["slots_total"] = len(mp.get("slots", []))
                    continue

                try:
                    cmd     = planner_step(packet)
                    payload = json.dumps(cmd, ensure_ascii=False) + "\n"
                    sock.sendall(payload.encode("utf-8"))
                    frames.append({"t": packet.get("t"), "obs": packet, "cmd": cmd})
                except BrokenPipeError:
                    print("[algo] send failed: broken pipe")
                    return
                except Exception as exc:
                    print(f"[algo] planner/send error: {exc}")

    except (ConnectionResetError, ConnectionAbortedError) as exc:
        print(f"[algo] connection error: {exc}")
    except Exception as exc:
        print(f"[algo] unexpected error while talking to simulator: {exc}")
    finally:
        session_meta["end_time"]    = datetime.now().isoformat(timespec="seconds")
        session_meta["frame_count"] = len(frames)
        save_student_replay(frames, session_meta)


def run_client(host: str, port: int) -> None:
    """시뮬레이터가 열어둔 포트에 접속해 세션을 유지한다."""
    backoff = 1.0
    while True:
        try:
            print(f"[algo] connecting to simulator at {host}:{port} ...")
            with socket.create_connection((host, port), timeout=2.0) as sock:
                sock.settimeout(0.2)
                run_session(sock, sock.getpeername())
                backoff = 1.0
        except KeyboardInterrupt:
            print("\n[algo] stopping by keyboard interrupt")
            break
        except (ConnectionRefusedError, TimeoutError, OSError) as exc:
            print(f"[algo] connect failed ({exc}); retrying in {backoff:.1f}s")
            time.sleep(backoff)
            backoff = min(backoff + 0.5, 5.0)
            continue

        print("[algo] lost connection - waiting 1.0s before retry")
        time.sleep(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55556)
    options = parser.parse_args()

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    run_client(options.host, options.port)


if __name__ == "__main__":
    main()
