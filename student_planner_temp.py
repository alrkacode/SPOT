"""자율 주차 알고리즘 - Hybrid A* + Pure Pursuit

Hybrid A* 알고리즘:
  - 차량 운동학 모델(자전거 모델)로 상태 공간 탐색
  - 연속 (x, y, yaw) 상태를 이산 격자로 해싱
  - 휴리스틱: 유클리드 거리 + yaw 정렬 비용
  - 전진/후진 양방향 탐색, 후진에 추가 비용 부여
  - 장애물: stationary 그리드 + line_rects + walls_rects

제어기:
  - Pure Pursuit: 경로 추종 조향
  - 속도 프로파일: 구간별 목표속도 + 비례 제어
"""

from dataclasses import dataclass, field
import heapq
import math
from typing import Any, Dict, List, Optional, Tuple

Point = Tuple[float, float]
Rect  = Tuple[float, float, float, float]


# ──────────────────────────────────────────────────────────────────────────────
# 차량 / 알고리즘 상수
# ──────────────────────────────────────────────────────────────────────────────
L          = 2.6          # wheelbase (m)
CAR_LF     = 1.6          # 앞 오버행
CAR_LR     = 1.4          # 뒤 오버행
CAR_W      = 1.6          # 차폭
MAX_STEER  = math.radians(35)
REVERSE_COST  = 3.0       # 후진 1m당 추가 비용
GEAR_SWITCH_COST = 5.0    # 기어 전환 비용
STEER_COST = 0.2          # 조향 1rad당 추가 비용

# Hybrid A* 격자 해상도
XY_RES  = 1.0    # m  (격자 해상도)
YAW_RES = math.radians(15)  # rad (yaw 해상도)

# 시뮬레이션 스텝
# SIM_DT * v >= XY_RES 이어야 이웃 노드가 다른 격자 셀로 이동
SIM_DT    = 1.2   # 한 번에 이동하는 시간(s)  → 1.2m 이동
N_STEERS  = 7     # 조향 샘플 수

# 차량 충돌 체크용 마진
VEHICLE_MARGIN = 0.25


# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────────────────────
def clamp(v, lo, hi): return max(lo, min(hi, v))
def wrap(a):
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def rect_center(r: Rect) -> Point:
    return ((r[0]+r[1])/2, (r[2]+r[3])/2)

def clamp_cmd(steer, accel, brake, gear, limits) -> Dict:
    ms = float(limits.get("maxSteer", MAX_STEER))
    return {
        "steer": clamp(float(steer), -ms, ms),
        "accel": clamp(float(accel), 0.0, 1.0),
        "brake": clamp(float(brake), 0.0, 1.0),
        "gear":  "R" if str(gear).upper().startswith("R") else "D",
    }

def pretty_print_map_summary(mp):
    slots = mp.get("slots") or []
    occ   = mp.get("occupied_idx") or []
    free  = len(slots) - sum(1 for v in occ if v)
    print(f"[algo] extent={mp.get('extent')}  slots={len(slots)} free={free}")
    sg = mp.get("grid",{}).get("stationary")
    if sg: print(f"[algo] grid={len(sg)}x{len(sg[0]) if sg else 0}")


# ──────────────────────────────────────────────────────────────────────────────
# 차량 폴리곤 & 충돌 판정
# ──────────────────────────────────────────────────────────────────────────────
def car_corners(x, y, yaw, margin=0.0):
    """차량 폴리곤 4개 꼭짓점 (후축 기준)."""
    lf = CAR_LF + margin; lr = CAR_LR + margin; hw = CAR_W/2 + margin
    pts = [(lf, hw),(lf,-hw),(-lr,-hw),(-lr, hw)]
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    return [(x+p[0]*cos_y-p[1]*sin_y, y+p[0]*sin_y+p[1]*cos_y) for p in pts]

def poly_aabb(poly):
    xs=[p[0] for p in poly]; ys=[p[1] for p in poly]
    return min(xs),max(xs),min(ys),max(ys)

def aabb_overlap(a, b):
    return not(a[1]<b[0] or b[1]<a[0] or a[3]<b[2] or b[3]<a[2])

def cross2d(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def segs_intersect(a, b, c, d):
    d1=cross2d(c,d,a); d2=cross2d(c,d,b)
    d3=cross2d(a,b,c); d4=cross2d(a,b,d)
    if ((d1>0 and d2<0) or (d1<0 and d2>0)) and \
       ((d3>0 and d4<0) or (d3<0 and d4>0)):
        return True
    return False

def poly_intersects_rect(poly, rect):
    """차량 폴리곤과 AABB 사각형 교차 검사 (SAT 근사)."""
    rx0,rx1,ry0,ry1 = rect[0],rect[1],rect[2],rect[3]
    paabb = poly_aabb(poly)
    raabb = (rx0,rx1,ry0,ry1)
    if not aabb_overlap(paabb, raabb): return False
    # 꼭짓점이 사각형 안에 있으면 충돌
    for px,py in poly:
        if rx0<=px<=rx1 and ry0<=py<=ry1: return True
    # 사각형 꼭짓점이 폴리곤 안에 있으면 충돌
    rect_pts = [(rx0,ry0),(rx1,ry0),(rx1,ry1),(rx0,ry1)]
    for rp in rect_pts:
        inside = True
        n = len(poly)
        for i in range(n):
            if cross2d(poly[i], poly[(i+1)%n], rp) < 0:
                inside = False; break
        if inside: return True
    # 변 교차 검사
    n = len(poly)
    rect_segs = [(rect_pts[i], rect_pts[(i+1)%4]) for i in range(4)]
    for i in range(n):
        for rs,re in rect_segs:
            if segs_intersect(poly[i], poly[(i+1)%n], rs, re):
                return True
    return False

def rect_contains_poly(rect, poly):
    return all(rect[0]<=p[0]<=rect[1] and rect[2]<=p[1]<=rect[3] for p in poly)


# ──────────────────────────────────────────────────────────────────────────────
# 장애물 맵
# ──────────────────────────────────────────────────────────────────────────────
class ObstacleMap:
    """line_rects + walls_rects + occupied 슬롯을 통합한 장애물 맵."""

    def __init__(self):
        self.rects: List[Rect] = []          # 모든 장애물 AABB
        self.target_slot: Optional[Rect] = None

    def build(self, mp: Dict, target_slot: Optional[Rect] = None):
        self.rects.clear()
        self.target_slot = target_slot

        # 외벽
        for r in mp.get("walls_rects", []):
            self.rects.append(tuple(map(float, r)))

        # 주차선 → 사각형 변환
        # 목표 슬롯에 인접한 선(진입 경계선)은 제외 (슬롯 진입 허용)
        LINE_HW = 0.25
        ts = target_slot
        for x1,y1,x2,y2 in mp.get("lines", []):
            x1,y1,x2,y2 = float(x1),float(y1),float(x2),float(y2)
            # 목표 슬롯 경계선은 제외
            if ts is not None:
                # 슬롯 ymin 근처 수평선 (진입 경계선)
                if abs(y1-y2)<1e-4 and abs(y1-ts[2])<0.5:
                    if min(x1,x2)<=ts[1]+1 and max(x1,x2)>=ts[0]-1:
                        continue
                # 슬롯 ymax 근처 수평선
                if abs(y1-y2)<1e-4 and abs(y1-ts[3])<0.5:
                    if min(x1,x2)<=ts[1]+1 and max(x1,x2)>=ts[0]-1:
                        continue
                # 슬롯 x범위 내 수직선
                if abs(x1-x2)<1e-4:
                    if ts[0]-0.5<=x1<=ts[1]+0.5:
                        continue
            if abs(x1-x2) < 1e-4:
                self.rects.append((min(x1,x2)-LINE_HW, max(x1,x2)+LINE_HW,
                                   min(y1,y2), max(y1,y2)))
            elif abs(y1-y2) < 1e-4:
                self.rects.append((min(x1,x2), max(x1,x2),
                                   min(y1,y2)-LINE_HW, max(y1,y2)+LINE_HW))

        # 점유된 슬롯
        slots = [tuple(map(float,s)) for s in mp.get("slots",[])]
        occ   = [bool(v) for v in mp.get("occupied_idx",[])]
        for slot, is_occ in zip(slots, occ):
            if is_occ:
                self.rects.append(slot)

    def is_collision(self, x, y, yaw, margin=VEHICLE_MARGIN) -> bool:
        poly = car_corners(x, y, yaw, margin)
        # 목표 슬롯 안은 충돌 면제
        if self.target_slot and rect_contains_poly(self.target_slot, poly):
            return False
        paabb = poly_aabb(poly)
        for rect in self.rects:
            if not aabb_overlap(paabb, (rect[0],rect[1],rect[2],rect[3])):
                continue
            if poly_intersects_rect(poly, rect):
                return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid A* 노드
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(order=True)
class HNode:
    f:     float
    x:     float = field(compare=False)
    y:     float = field(compare=False)
    yaw:   float = field(compare=False)
    gear:  str   = field(compare=False)  # "D" or "R"
    g:     float = field(compare=False)
    steer: float = field(compare=False)
    parent_key: Any = field(compare=False, default=None)

def state_key(x, y, yaw):
    """연속 상태 → 이산 격자 키."""
    ix = round(x / XY_RES)
    iy = round(y / XY_RES)
    iyaw = round(wrap(yaw) / YAW_RES)
    return (ix, iy, iyaw)


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid A* 탐색
# ──────────────────────────────────────────────────────────────────────────────
def heuristic(x, y, yaw, gx, gy, gyaw) -> float:
    """유클리드 거리 + yaw 정렬 비용."""
    d = math.hypot(x-gx, y-gy)
    dyaw = abs(wrap(yaw-gyaw))
    return d + dyaw * L  # yaw 오차를 거리로 환산

def hybrid_astar(
    sx, sy, syaw,           # 시작 상태
    gx, gy, gyaw,           # 목표 상태
    obs_map: ObstacleMap,
    extent: Tuple,          # (xmin,xmax,ymin,ymax)
    allow_reverse: bool = True,
    max_iter: int = 8000,
) -> Optional[List[Tuple[float,float,float,str]]]:
    """
    Hybrid A* 탐색.
    반환: [(x,y,yaw,gear), ...] 경로, 실패 시 None
    """
    xmin,xmax,ymin,ymax = extent

    steers = []
    for i in range(N_STEERS):
        steers.append(MAX_STEER * (2*i/(N_STEERS-1) - 1))

    start_key = state_key(sx, sy, syaw)
    start_node = HNode(
        f=heuristic(sx,sy,syaw,gx,gy,gyaw),
        x=sx, y=sy, yaw=syaw, gear="D", g=0.0, steer=0.0,
        parent_key=None
    )

    open_heap = [start_node]
    node_map: Dict[Any, HNode] = {start_key: start_node}
    closed: set = set()

    for _ in range(max_iter):
        if not open_heap: break
        cur = heapq.heappop(open_heap)
        cur_key = state_key(cur.x, cur.y, cur.yaw)

        if cur_key in closed: continue
        closed.add(cur_key)

        # 목표 도달 판정
        if (math.hypot(cur.x-gx, cur.y-gy) < XY_RES * 1.5 and
                abs(wrap(cur.yaw-gyaw)) < YAW_RES * 3):
            # 경로 역추적
            path = []
            node = cur
            while node is not None:
                path.append((node.x, node.y, node.yaw, node.gear))
                pk = node.parent_key
                node = node_map.get(pk) if pk is not None else None
            path.reverse()
            return path

        # 이웃 확장
        gears = ["D", "R"] if allow_reverse else ["D"]
        for gear in gears:
            for steer in steers:
                # 한 스텝 시뮬레이션
                x, y, yaw = cur.x, cur.y, cur.yaw
                v = 1.0 if gear=="D" else -1.0
                nx = x + v * math.cos(yaw) * SIM_DT
                ny = y + v * math.sin(yaw) * SIM_DT
                nyaw = wrap(yaw + (v/L) * math.tan(steer) * SIM_DT)

                # 맵 범위 체크
                if not (xmin+0.5 <= nx <= xmax-0.5 and ymin+0.5 <= ny <= ymax-0.5):
                    continue
                # 충돌 체크
                if obs_map.is_collision(nx, ny, nyaw):
                    continue

                # 비용 계산
                move_cost = abs(v) * SIM_DT
                reverse_cost = REVERSE_COST * SIM_DT if gear=="R" else 0.0
                steer_cost = STEER_COST * abs(steer)
                switch_cost = GEAR_SWITCH_COST if gear != cur.gear else 0.0
                ng = cur.g + move_cost + reverse_cost + steer_cost + switch_cost

                nb_key = state_key(nx, ny, nyaw)
                if nb_key in closed: continue

                existing = node_map.get(nb_key)
                if existing and existing.g <= ng: continue

                nh = heuristic(nx, ny, nyaw, gx, gy, gyaw)
                nb_node = HNode(
                    f=ng+nh, x=nx, y=ny, yaw=nyaw,
                    gear=gear, g=ng, steer=steer,
                    parent_key=cur_key
                )
                node_map[nb_key] = nb_node
                heapq.heappush(open_heap, nb_node)

    return None  # 탐색 실패


# ──────────────────────────────────────────────────────────────────────────────
# 경로 후처리
# ──────────────────────────────────────────────────────────────────────────────
def smooth_path_poses(path, window=3):
    """경로 위치를 이동평균으로 스무딩 (yaw/gear는 유지)."""
    n = len(path)
    if n <= window*2: return path
    result = []
    hw = window
    for i in range(n):
        lo, hi = max(0,i-hw), min(n,i+hw+1)
        xs = [path[j][0] for j in range(lo,hi)]
        ys = [path[j][1] for j in range(lo,hi)]
        result.append((sum(xs)/len(xs), sum(ys)/len(ys),
                       path[i][2], path[i][3]))
    return result

def assign_speeds(path, entry_speed=0.4, cruise_speed=3.0,
                  decel_dist=4.0, goal_speed=0.25):
    """각 경로 포인트에 목표 속도 할당."""
    n = len(path)
    speeds = [cruise_speed] * n
    # 마지막 구간 감속
    total = 0.0
    for i in range(n-1, 0, -1):
        d = math.hypot(path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
        total += d
        if total >= decel_dist: break
        t = max(0, 1.0 - total/decel_dist)
        speeds[i] = goal_speed + (cruise_speed-goal_speed) * (1-t)
    speeds[-1] = goal_speed
    # 기어 전환 구간 감속
    for i in range(1, n):
        if path[i][3] != path[i-1][3]:
            for j in range(max(0,i-5), min(n,i+5)):
                speeds[j] = min(speeds[j], entry_speed)
    # 후진 구간 속도 제한
    for i in range(n):
        if path[i][3] == "R":
            speeds[i] = min(speeds[i], 1.5)
    return speeds


# ──────────────────────────────────────────────────────────────────────────────
# Waypoint 기반 폴백 (Hybrid A* 실패 시)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Waypoint:
    x: float; y: float
    gear: str = "D"
    radius: float = 1.0
    speed: float = 1.6
    stop_here: bool = False

    @property
    def point(self): return (self.x, self.y)


# ──────────────────────────────────────────────────────────────────────────────
# PlannerSkeleton
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PlannerSkeleton:
    # 맵 데이터
    map_data:     Optional[Dict] = None
    map_extent:   Optional[Tuple] = None
    slots:        List[Rect] = field(default_factory=list)
    occupied_idx: List[bool] = field(default_factory=list)
    lines:        List[Tuple] = field(default_factory=list)
    walls_rects:  List[Rect] = field(default_factory=list)
    obs_map:      ObstacleMap = field(default_factory=ObstacleMap)

    # 경로 (Hybrid A* 결과)
    ha_path:   List[Tuple] = field(default_factory=list)  # (x,y,yaw,gear)
    ha_speeds: List[float] = field(default_factory=list)
    ha_idx:    int = 0

    # 폴백 Waypoint 경로
    waypoints:       List[Waypoint] = field(default_factory=list)
    active_waypoint: int = 0

    # 상태
    target_slot:   Optional[Rect]  = None
    target_center: Optional[Point] = None
    final_yaw:     float = math.pi/2
    path_computed: bool  = False
    use_fallback:  bool  = False
    arrived:       bool  = False

    left_aisle_x:         float = 4.0
    expected_orientation: str   = "front_in"

    def __post_init__(self):
        pass

    # ── 맵 수신 ────────────────────────────────────────────────────────────
    def set_map(self, mp: Dict) -> None:
        self.map_data    = mp
        self.map_extent  = tuple(map(float, mp.get("extent",(0,75,0,50))))
        self.slots       = [tuple(map(float,s)) for s in mp.get("slots",[])]
        self.occupied_idx= [bool(v) for v in mp.get("occupied_idx",[])]
        self.lines       = [tuple(map(float,l)) for l in mp.get("lines",[])]
        self.walls_rects = [tuple(map(float,r)) for r in mp.get("walls_rects",[])]
        free_cnt = sum(1 for v in self.occupied_idx if not v)
        self.expected_orientation = "rear_in" if free_cnt==1 else "front_in"
        self.left_aisle_x = self._estimate_aisle_x()
        self._reset()
        pretty_print_map_summary(mp)
        print(f"[algo] orientation={self.expected_orientation}")

    def _estimate_aisle_x(self) -> float:
        if not self.map_extent: return 4.0
        xmin, xmax, _, _ = self.map_extent
        if not self.slots: return clamp(xmin+4.0, xmin+2.0, xmax-2.0)
        first_x = min(s[0] for s in self.slots)
        return clamp(min(xmin+4.0, first_x-7.0), xmin+2.5, xmax-2.5)

    def _reset(self):
        self.ha_path.clear(); self.ha_speeds.clear(); self.ha_idx=0
        self.waypoints.clear(); self.active_waypoint=0
        self.target_slot=None; self.target_center=None
        self.path_computed=False; self.use_fallback=False; self.arrived=False

    # ── 목표 슬롯 감지 ────────────────────────────────────────────────────
    def _same_slot(self, slot) -> bool:
        if self.target_slot is None: return False
        return all(abs(a-b)<=1e-4 for a,b in zip(self.target_slot,slot))

    def _target_yaw(self, slot) -> float:
        w=slot[1]-slot[0]; h=slot[3]-slot[2]
        base = math.pi/2 if h>=w else 0.0
        if self.expected_orientation=="rear_in": base=wrap(base+math.pi)
        return base

    def _slot_open_side(self, slot) -> str:
        sx_min,sx_max,sy_min,sy_max = slot
        # 높은 Row는 아래서 진입
        if self.map_extent:
            _,_,ymin,ymax = self.map_extent
            if rect_center(slot)[1] > (ymin+ymax)*0.65: return "below"
        low=False; high=False
        for x1,y1,x2,y2 in self.lines:
            if abs(y1-y2)>1e-4: continue
            if not (min(x1,x2)<=sx_max+0.5 and max(x1,x2)>=sx_min-0.5): continue
            if abs(y1-sy_min)<=1.0: low=True
            if abs(y1-sy_max)<=1.0: high=True
        if low and not high: return "above"
        if high and not low: return "below"
        return "above"

    def _update_target(self, obs: Dict) -> None:
        raw = obs.get("target_slot")
        if raw and len(raw)==4:
            slot = tuple(map(float,raw))
        else:
            free=[i for i,o in enumerate(self.occupied_idx) if not o]
            slot=self.slots[free[0]] if free else (self.slots[0] if self.slots else None)
        if slot is None: return
        if self._same_slot(slot) and self.path_computed: return

        new_center = rect_center(slot)
        new_yaw    = self._target_yaw(slot)
        self._reset()
        self.target_slot   = slot
        self.target_center = new_center
        self.final_yaw     = new_yaw

    # ── Hybrid A* 경로 계획 ───────────────────────────────────────────────
    def compute_path(self, obs: Dict) -> None:
        if self.map_data is None or self.target_slot is None:
            self.use_fallback = True
            self.path_computed = True
            return

        state  = obs.get("state", {})
        sx     = float(state.get("x",   0.0))
        sy     = float(state.get("y",   0.0))
        syaw   = float(state.get("yaw", math.pi/2))

        slot   = self.target_slot
        cx, cy = self.target_center
        gyaw   = self.final_yaw

        print(f"[algo] Hybrid A* start=({sx:.1f},{sy:.1f},{math.degrees(syaw):.0f}°)"
              f" goal=({cx:.1f},{cy:.1f},{math.degrees(gyaw):.0f}°)")

        # 장애물 맵 구축 (목표 슬롯 충돌 면제)
        self.obs_map.build(self.map_data, target_slot=slot)

        # Hybrid A* 탐색
        path = hybrid_astar(
            sx, sy, syaw,
            cx, cy, gyaw,
            self.obs_map,
            self.map_extent,
            allow_reverse=True,
            max_iter=20000,
        )

        if path and len(path) >= 2:
            path = smooth_path_poses(path, window=2)
            speeds = assign_speeds(path, cruise_speed=3.0, goal_speed=0.3)
            self.ha_path   = path
            self.ha_speeds = speeds
            self.ha_idx    = 0
            self.use_fallback = False
            print(f"[algo] Hybrid A* success: {len(path)} nodes")
        else:
            print("[algo] Hybrid A* failed → fallback waypoints")
            self.use_fallback = True
            self.waypoints = self._build_fallback_waypoints(slot, sx, sy)
            self.active_waypoint = 0

        self.path_computed = True

    # ── 폴백 Waypoint 경로 ────────────────────────────────────────────────
    def _build_fallback_waypoints(self, slot, sx, sy) -> List[Waypoint]:
        if self.map_extent is None: return []
        xmin,xmax,ymin,ymax = self.map_extent
        cx,cy = rect_center(slot)
        open_side = self._slot_open_side(slot)
        aisle_x   = clamp(self.left_aisle_x, xmin+2.5, xmax-2.5)
        turn_x    = clamp(cx-4.5, aisle_x+2.0, xmax-2.5)

        if open_side == "above":
            align_y = clamp(slot[3]+2.0, ymin+2.5, ymax-2.5)
            entry_y = clamp(slot[3]+7.0, ymin+2.5, ymax-2.5)
            return [
                Waypoint(aisle_x, align_y, "D", 1.2, 1.6),
                Waypoint(turn_x,  align_y, "D", 1.1, 1.6),
                Waypoint(cx,      entry_y, "D", 1.5, 0.85),
                Waypoint(cx,      cy,      "R", 0.35, 0.55, stop_here=True),
            ]
        else:
            entry_y = clamp(slot[2]-2.0, ymin+2.5, ymax-2.5)
            align_y = clamp(entry_y-6.0, ymin+2.5, ymax-2.5)
            return [
                Waypoint(aisle_x, align_y, "D", 1.2, 1.6),
                Waypoint(turn_x,  align_y, "D", 1.1, 1.6),
                Waypoint(cx,      entry_y, "D", 1.5, 0.85),
                Waypoint(cx,      cy,      "D", 0.35, 0.55, stop_here=True),
            ]

    # ── Hybrid A* 경로 추종 제어 ──────────────────────────────────────────
    def _control_hybrid(self, obs: Dict) -> Dict:
        state  = obs.get("state", {}); limits = obs.get("limits", {})
        ex     = float(state.get("x",   0.0))
        ey     = float(state.get("y",   0.0))
        eyaw   = float(state.get("yaw", 0.0))
        ev_sgn = float(state.get("v",   0.0))
        ev     = abs(ev_sgn)
        pos    = (ex, ey)
        L_wb   = float(limits.get("L", L))
        ms     = float(limits.get("maxSteer", MAX_STEER))

        path   = self.ha_path
        speeds = self.ha_speeds
        n      = len(path)
        cx, cy = self.target_center

        # 도착 판정
        d_goal = dist(pos, (cx,cy))
        yaw_err= abs(wrap(eyaw-self.final_yaw))
        if d_goal < 0.8 and yaw_err < math.radians(20):
            self.arrived = True
            return clamp_cmd(0.0, 0.0, 1.0, "D", limits)

        # 현재 인덱스 전진 (지나친 포인트 스킵)
        while self.ha_idx < n-1:
            d = dist(pos, (path[self.ha_idx][0], path[self.ha_idx][1]))
            if d > 0.5: break
            self.ha_idx += 1

        idx = min(self.ha_idx, n-1)

        # 룩어헤드 포인트 탐색
        lookahead = clamp(1.5 + 0.5*ev, 1.5, 5.0)
        target_idx = idx
        walked = 0.0
        for i in range(idx, n-1):
            if path[i][3] != path[idx][3]: break  # 기어 전환 전까지만
            walked += dist((path[i][0],path[i][1]), (path[i+1][0],path[i+1][1]))
            if walked >= lookahead:
                target_idx = i+1; break
        else:
            target_idx = n-1

        tp = path[target_idx]
        gear = path[idx][3]

        # Pure Pursuit 조향
        dx = tp[0]-ex; dy = tp[1]-ey
        ld = max(dist(pos,(tp[0],tp[1])), 0.1)
        bearing = math.atan2(dy, dx)
        if gear == "R":
            alpha = wrap(bearing - wrap(eyaw+math.pi))
            steer = -math.atan2(2*L_wb*math.sin(alpha), ld)
        else:
            alpha = wrap(bearing - eyaw)
            steer = math.atan2(2*L_wb*math.sin(alpha), ld)

        # yaw 피드백 보정 (슬롯 근처에서 강화)
        if d_goal < 5.0:
            k_yaw = 0.4 * (1.0 - d_goal/5.0)
            yaw_fb = wrap(self.final_yaw - eyaw)
            steer += k_yaw * yaw_fb
        steer = clamp(steer, -ms, ms)

        # 속도 제어
        desired_v = speeds[idx]
        # heading error에 따라 감속
        he = abs(alpha)
        if he > math.radians(60): desired_v = min(desired_v, 0.5)
        elif he > math.radians(30): desired_v = min(desired_v, 1.0)

        # 기어 불일치 → 브레이크
        if gear=="D" and ev_sgn < -0.1: return clamp_cmd(steer,0.0,0.8,gear,limits)
        if gear=="R" and ev_sgn >  0.1: return clamp_cmd(steer,0.0,0.8,gear,limits)

        err = desired_v - ev
        if err > 0.1:   accel=clamp(0.2+0.3*err,0.0,0.7); brake=0.0
        elif err < -0.2: accel=0.0; brake=clamp(0.1+0.3*(-err),0.0,0.8)
        else:            accel=0.05; brake=0.0

        return clamp_cmd(steer, accel, brake, gear, limits)

    # ── 폴백 Waypoint 추종 제어 ───────────────────────────────────────────
    def _control_fallback(self, obs: Dict) -> Dict:
        state=obs.get("state",{}); limits=obs.get("limits",{})
        ex=float(state.get("x",0)); ey=float(state.get("y",0))
        eyaw=float(state.get("yaw",0)); sv=float(state.get("v",0))
        speed=abs(sv); pos=(ex,ey)
        L_wb=float(limits.get("L",L))

        # 도착 체크
        if self.target_center:
            if dist(pos,self.target_center)<0.8 and abs(wrap(eyaw-self.final_yaw))<math.radians(25):
                self.arrived=True
                return clamp_cmd(0.0,0.0,1.0,"D",limits)

        # Waypoint 전진
        wp=None
        wps=self.waypoints
        while self.active_waypoint<len(wps):
            wp=wps[self.active_waypoint]
            if wp.stop_here: break
            reach=wp.radius+min(0.6,speed*0.25)
            if dist(pos,wp.point)>reach: break
            self.active_waypoint+=1
        if wp is None: return clamp_cmd(0.0,0.0,0.5,"D",limits)

        # Pure Pursuit
        dx=wp.x-ex; dy=wp.y-ey
        ld=clamp(math.hypot(dx,dy),2.0,6.0)
        bearing=math.atan2(dy,dx)
        if wp.gear=="R":
            alpha=wrap(bearing-wrap(eyaw+math.pi))
            steer=-math.atan2(2*L_wb*math.sin(alpha),ld)
        else:
            alpha=wrap(bearing-eyaw)
            steer=math.atan2(2*L_wb*math.sin(alpha),ld)
        ms=float(limits.get("maxSteer",MAX_STEER))
        steer=clamp(steer,-ms,ms)

        # 속도
        desired=wp.speed
        he=abs(alpha)
        if he>math.radians(70): desired=min(desired,0.55)
        elif he>math.radians(35): desired=min(desired,0.85)
        if wp.stop_here:
            td=dist(pos,wp.point)
            desired=min(desired,max(0.15,td*0.45))
            ye=abs(wrap(eyaw-self.final_yaw))
            if td<0.7 and ye<math.radians(35) and speed<0.24: return clamp_cmd(steer,0.0,1.0,wp.gear,limits)
            if td<0.45 and ye<math.radians(35): return clamp_cmd(steer,0.0,0.7,wp.gear,limits)

        if wp.gear=="D" and sv<-0.12: return clamp_cmd(steer,0.0,0.7,wp.gear,limits)
        if wp.gear=="R" and sv> 0.12: return clamp_cmd(steer,0.0,0.7,wp.gear,limits)

        err=desired-speed
        if err>0.1:   accel=clamp(0.18+0.2*err,0.0,0.6); brake=0.0
        elif err<-0.15: accel=0.0; brake=clamp(0.12+0.3*(-err),0.0,0.75)
        else:          accel=0.05; brake=0.0
        return clamp_cmd(steer,accel,brake,wp.gear,limits)

    # ── 메인 제어 ─────────────────────────────────────────────────────────
    def compute_control(self, obs: Dict) -> Dict:
        self._update_target(obs)
        if self.target_slot is None:
            return clamp_cmd(0.0,0.0,0.5,"D",obs.get("limits",{}))

        if not self.path_computed:
            self.compute_path(obs)

        if self.arrived:
            return clamp_cmd(0.0,0.0,1.0,"D",obs.get("limits",{}))

        if self.use_fallback:
            return self._control_fallback(obs)
        else:
            return self._control_hybrid(obs)


# ──────────────────────────────────────────────────────────────────────────────
# 전역 인스턴스
# ──────────────────────────────────────────────────────────────────────────────
planner = PlannerSkeleton()

def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    planner.set_map(map_payload)

def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return planner.compute_control(obs)
    except Exception as exc:
        import traceback; traceback.print_exc()
        return {"steer":0.0,"accel":0.0,"brake":0.7,"gear":"D"}
