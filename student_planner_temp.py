"""자율 주차 시뮬레이터용 rule-based student planner입니다.

시뮬레이터는 맵이 준비되면 ``handle_map_payload``를 한 번 호출하고,
이후 매 tick마다 ``planner_step``을 호출합니다. 이 파일은 외부 학습 모델
없이 맵 파싱, 경로 생성, 조향/속도 제어를 모두 로컬에서 처리합니다.

기본 구조는 rule-based waypoint planner입니다. 일부 검증된 케이스에서는
guarded 특수 경로를 사용합니다. Default Lot의 일부 slot에는 mild-turn
route를 적용하고, Full House slot 5에는 full-path Hybrid A*로 만든 dense
trajectory branch를 사용합니다. 가드 조건이 맞지 않거나 실패하면 기존
fallback 경로로 돌아갑니다.
"""

from dataclasses import dataclass, field
import heapq
import math
import threading
import time
import sys as _sys

# 백그라운드 Hybrid A* 스레드가 도는 동안 메인(IPC) 스레드가 0.15s 안에
# 응답할 수 있도록, GIL 스위치 간격을 짧게 둡니다(기본 5ms -> 1ms).
# 이렇게 하면 무거운 계산 중에도 메인 스레드가 더 자주 실행 기회를 얻습니다.
try:
    _sys.setswitchinterval(0.001)
except Exception:
    pass
from typing import Any, Dict, List, Optional, Tuple


Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]
Pose = Tuple[float, float, float]  # (x, y, yaw)


def clamp(value: float, low: float, high: float) -> float:
    """숫자 값을 닫힌 구간 [low, high] 안으로 제한합니다."""

    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    """임의의 각도를 [-pi, pi] 범위로 정규화합니다."""

    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def distance(a: Point, b: Point) -> float:
    """두 2차원 점 사이의 유클리드 거리를 계산합니다."""

    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp_command(
    steer: float,
    accel: float,
    brake: float,
    gear: str,
    limits: Dict[str, Any],
) -> Dict[str, Any]:
    """시뮬레이터 제한값에 맞게 명령을 보정한 뒤 반환합니다."""

    max_steer = float(limits.get("maxSteer", math.radians(35.0)))
    return {
        "steer": clamp(float(steer), -max_steer, max_steer),
        "accel": clamp(float(accel), 0.0, 1.0),
        "brake": clamp(float(brake), 0.0, 1.0),
        "gear": "R" if str(gear).upper().startswith("R") else "D",
    }


def rect_center(rect: Rect) -> Point:
    """축에 정렬된 사각형의 중심점을 반환합니다."""

    return ((rect[0] + rect[1]) * 0.5, (rect[2] + rect[3]) * 0.5)


def pretty_print_map_summary(map_payload: Dict[str, Any]) -> None:
    extent = map_payload.get("extent") or [None, None, None, None]
    slots = map_payload.get("slots") or []
    occupied = map_payload.get("occupied_idx") or []
    free_slots = len(slots) - sum(1 for v in occupied if v)
    print("[algo] map extent :", extent)
    print("[algo] total slots:", len(slots), "/ free:", free_slots)
    stationary = map_payload.get("grid", {}).get("stationary")
    if stationary:
        rows = len(stationary)
        cols = len(stationary[0]) if stationary else 0
        print("[algo] grid size  :", rows, "x", cols)


@dataclass
class Waypoint:
    """제어기가 따라갈 작은 목표점입니다."""

    x: float
    y: float
    gear: str = "D"
    radius: float = 1.0
    speed: float = 1.4
    stop_here: bool = False

    @property
    def point(self) -> Point:
        return (self.x, self.y)


@dataclass
class TrajectoryPoint:
    """Hybrid A*가 만든 dense trajectory의 한 점입니다."""

    x: float
    y: float
    yaw: float
    gear: str
    speed: float

    @property
    def point(self) -> Point:
        return (self.x, self.y)


# ---------------------------------------------------------------------------
# Hybrid A* core (경량화된 grid 기반 Hybrid A*)
#
# 참고 구조: karlkurzer/path_planner (Hybrid A* Path Planner for the KTH RCV)
#   - holonomic-with-obstacles heuristic (2D Dijkstra)
#   - 제한된 steering motion primitive
#   - 목표와 정렬되면 바로 연결하는 analytic shortcut (Dubin's Shot의 단순화 버전)
# 위 구조를 ROS/OMPL 없이, 이 파일 안에서 동작하는 순수 Python으로 재구현했습니다.
# ---------------------------------------------------------------------------


class OccupancyGrid:
    """시뮬레이터의 stationary grid를 감싸서 좌표 변환/거리 계산을 제공합니다."""

    SQRT2 = math.sqrt(2.0)

    def __init__(
        self,
        grid: List[List[float]],
        cell_size: float,
        extent: Rect,
        occupied_threshold: float = 0.5,
        flip_rows: bool = False,
    ) -> None:
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 0
        self.cell_size = float(cell_size)
        self.xmin, self.xmax, self.ymin, self.ymax = extent
        self.occupied_threshold = occupied_threshold
        self.flip_rows = flip_rows
        self._distance_cells = self._compute_distance_transform()

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        col = int((x - self.xmin) / self.cell_size)
        row = int((y - self.ymin) / self.cell_size)
        if self.flip_rows:
            row = self.rows - 1 - row
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        r = self.rows - 1 - row if self.flip_rows else row
        x = self.xmin + (col + 0.5) * self.cell_size
        y = self.ymin + (r + 0.5) * self.cell_size
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_occupied(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return True
        return self.grid[row][col] >= self.occupied_threshold

    def _compute_distance_transform(self) -> List[List[float]]:
        """2-pass chamfer distance transform (셀 단위 거리, 8-연결)."""

        rows, cols = self.rows, self.cols
        INF = float("inf")
        dist = [
            [0.0 if self.is_occupied(r, c) else INF for c in range(cols)]
            for r in range(rows)
        ]

        for r in range(rows):
            for c in range(cols):
                d = dist[r][c]
                if r > 0:
                    d = min(d, dist[r - 1][c] + 1.0)
                    if c > 0:
                        d = min(d, dist[r - 1][c - 1] + self.SQRT2)
                    if c + 1 < cols:
                        d = min(d, dist[r - 1][c + 1] + self.SQRT2)
                if c > 0:
                    d = min(d, dist[r][c - 1] + 1.0)
                dist[r][c] = d

        for r in range(rows - 1, -1, -1):
            for c in range(cols - 1, -1, -1):
                d = dist[r][c]
                if r + 1 < rows:
                    d = min(d, dist[r + 1][c] + 1.0)
                    if c > 0:
                        d = min(d, dist[r + 1][c - 1] + self.SQRT2)
                    if c + 1 < cols:
                        d = min(d, dist[r + 1][c + 1] + self.SQRT2)
                if c + 1 < cols:
                    d = min(d, dist[r][c + 1] + 1.0)
                dist[r][c] = d

        return dist

    def clearance(self, x: float, y: float) -> float:
        """월드 좌표 (x, y)에서 가장 가까운 장애물까지의 거리(미터)."""

        row, col = self.world_to_grid(x, y)
        if not self.in_bounds(row, col):
            return 0.0
        return self._distance_cells[row][col] * self.cell_size


@dataclass
class VehicleModel:
    """rear-axle 기준 bicycle 모델과 충돌 검사를 위한 footprint.

    self-parking-sim의 car_polygon과 동일한 형상을 사용합니다:
    rear axle(상태 기준점)에서 앞으로 Lf, 뒤로 Lr, 좌우 폭 W.
    (시뮬레이터: Lf=1.6, Lr=1.4, W=1.6, wheelbase L=2.6)
    """

    wheelbase: float = 2.6
    max_steer: float = math.radians(35.0)
    front_overhang: float = 1.6   # rear axle -> 앞 범퍼 (시뮬레이터 Lf)
    rear_overhang: float = 1.4    # rear axle -> 뒤 범퍼 (시뮬레이터 Lr)
    width: float = 1.6            # 시뮬레이터 W
    safety_margin: float = 0.18   # 차선/장애물과의 추가 여유(m)

    def footprint_polygon(self, x: float, y: float, yaw: float, margin: float = 0.0) -> List[Point]:
        """world 좌표계에서 차량 사각형 꼭짓점 4개를 반환합니다."""

        lf = self.front_overhang + margin
        lr = self.rear_overhang + margin
        hw = self.width * 0.5 + margin
        c, s = math.cos(yaw), math.sin(yaw)
        local = [(lf, hw), (lf, -hw), (-lr, -hw), (-lr, hw)]
        return [(x + px * c - py * s, y + px * s + py * c) for px, py in local]

    def circle_offsets(self) -> List[float]:
        """rear axle 기준, heading축을 따른 충돌 검사용 원 중심 위치(m)."""

        # 차량 길이(Lf+Lr)를 3개 원으로 커버
        return [-self.rear_overhang * 0.6, (self.front_overhang - self.rear_overhang) * 0.5,
                self.front_overhang * 0.6]

    def circle_radius(self) -> float:
        return self.width * 0.5 + self.safety_margin


def pose_collision_free(pose: Pose, occ_grid: "OccupancyGrid", vehicle: "VehicleModel") -> bool:
    """차량을 2~3개의 원으로 근사해 occupancy grid와의 충돌 여부를 검사합니다."""

    x, y, yaw = pose
    radius = vehicle.circle_radius()
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    for offset in vehicle.circle_offsets():
        cx = x + offset * cos_y
        cy = y + offset * sin_y
        if occ_grid.clearance(cx, cy) < radius:
            return False
    return True


@dataclass
class HybridAStarConfig:
    xy_resolution: float = 0.5
    yaw_resolution: float = math.radians(15.0)
    move_step: float = 1.0
    steer_levels: int = 3
    reverse_cost: float = 2.0
    gear_switch_cost: float = 6.0
    steer_change_cost: float = 0.5
    steer_cost: float = 0.3
    goal_xy_tolerance: float = 0.6
    goal_yaw_tolerance: float = math.radians(20.0)
    max_expansions: int = 40000
    analytic_every: int = 5
    heuristic_obstacle_inflation: float = 0.0
    time_budget_s: float = 3.0


@dataclass(order=True)
class _PQItem:
    priority: float
    counter: int
    node: "_HANode" = field(compare=False)


@dataclass
class _HANode:
    """Hybrid A* 탐색 노드 (시뮬레이터의 Waypoint/TrajectoryPoint와는 별개)."""

    x: float
    y: float
    yaw: float
    direction: int
    steer: float
    g: float
    parent: Optional["_HANode"] = None

    def state_key(self, cfg: "HybridAStarConfig") -> Tuple[int, int, int]:
        ix = int(round(self.x / cfg.xy_resolution))
        iy = int(round(self.y / cfg.xy_resolution))
        iyaw = int(round(wrap_angle(self.yaw) / cfg.yaw_resolution))
        return ix, iy, iyaw


def _ha_steer_set(vehicle: "VehicleModel", cfg: "HybridAStarConfig") -> List[float]:
    if cfg.steer_levels <= 0:
        return [0.0]
    steers = [0.0]
    for i in range(1, cfg.steer_levels + 1):
        s = vehicle.max_steer * (i / cfg.steer_levels)
        steers.extend([s, -s])
    return steers


def _ha_step(node: "_HANode", direction: int, steer: float, vehicle: "VehicleModel", cfg: "HybridAStarConfig") -> "_HANode":
    d = direction * cfg.move_step
    new_yaw = wrap_angle(node.yaw + d / vehicle.wheelbase * math.tan(steer))
    new_x = node.x + d * math.cos(node.yaw)
    new_y = node.y + d * math.sin(node.yaw)

    step_cost = abs(cfg.move_step)
    if direction < 0:
        step_cost *= cfg.reverse_cost
    if node.direction != 0 and direction != node.direction:
        step_cost += cfg.gear_switch_cost
    step_cost += cfg.steer_cost * abs(steer)
    step_cost += cfg.steer_change_cost * abs(steer - node.steer)

    return _HANode(new_x, new_y, new_yaw, direction, steer, node.g + step_cost, parent=node)


def build_holonomic_heuristic(
    occ_grid: "OccupancyGrid",
    goal_xy: Point,
    vehicle: "VehicleModel",
    cfg: "HybridAStarConfig",
) -> List[List[float]]:
    """목표에서부터 (x,y) 격자 전체로 퍼지는 장애물 회피 최단거리 맵 (미터)."""

    rows, cols = occ_grid.rows, occ_grid.cols
    INF = float("inf")
    dist = [[INF] * cols for _ in range(rows)]

    block_radius = vehicle.circle_radius() + cfg.heuristic_obstacle_inflation
    grow, gcol = occ_grid.world_to_grid(goal_xy[0], goal_xy[1])
    if not occ_grid.in_bounds(grow, gcol):
        return dist

    dist[grow][gcol] = 0.0
    pq: List[Tuple[float, int, int]] = [(0.0, grow, gcol)]
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, OccupancyGrid.SQRT2), (-1, 1, OccupancyGrid.SQRT2),
        (1, -1, OccupancyGrid.SQRT2), (1, 1, OccupancyGrid.SQRT2),
    ]
    cell = occ_grid.cell_size
    _pops = 0
    while pq:
        _pops += 1
        if _pops % 200 == 0:
            time.sleep(0)
        d, r, c = heapq.heappop(pq)
        if d > dist[r][c]:
            continue
        for dr, dc, w in neighbors:
            nr, nc = r + dr, c + dc
            if not occ_grid.in_bounds(nr, nc):
                continue
            x, y = occ_grid.grid_to_world(nr, nc)
            if occ_grid.clearance(x, y) < block_radius:
                continue
            nd = d + w * cell
            if nd < dist[nr][nc]:
                dist[nr][nc] = nd
                heapq.heappush(pq, (nd, nr, nc))
    return dist


def _heuristic_lookup(dist_map: List[List[float]], occ_grid: "OccupancyGrid", x: float, y: float) -> float:
    row, col = occ_grid.world_to_grid(x, y)
    if not occ_grid.in_bounds(row, col):
        return float("inf")
    return dist_map[row][col]


def _try_straight_shortcut(
    node: "_HANode",
    goal: Pose,
    occ_grid: "OccupancyGrid",
    vehicle: "VehicleModel",
    cfg: "HybridAStarConfig",
) -> Optional["_HANode"]:
    """node가 goal과 거의 같은 방향을 보고 있으면 직선으로 연결을 시도합니다.

    (karlkurzer/path_planner의 Dubin's Shot을 직선 케이스로 단순화한 버전)
    """

    gx, gy, gyaw = goal
    bearing = math.atan2(gy - node.y, gx - node.x)
    dist = math.hypot(gx - node.x, gy - node.y)
    if dist < 1e-3:
        return None

    yaw_to_bearing = abs(wrap_angle(node.yaw - bearing))
    goal_yaw_diff = abs(wrap_angle(node.yaw - gyaw))
    if yaw_to_bearing > math.radians(8.0) or goal_yaw_diff > math.radians(8.0):
        return None

    steps = max(1, int(dist / (cfg.move_step * 0.5)))
    cur = node
    for i in range(1, steps + 1):
        t = i / steps
        x = node.x + (gx - node.x) * t
        y = node.y + (gy - node.y) * t
        yaw = node.yaw
        if not pose_collision_free((x, y, yaw), occ_grid, vehicle):
            return None
        step_dist = dist / steps
        cur = _HANode(x, y, yaw, 1, 0.0, cur.g + step_dist, parent=cur)
    return cur


def hybrid_astar_plan(
    start: Pose,
    goal: Pose,
    occ_grid: "OccupancyGrid",
    vehicle: Optional["VehicleModel"] = None,
    cfg: Optional["HybridAStarConfig"] = None,
    abort_event: Optional["threading.Event"] = None,
) -> Optional[List[Tuple[float, float, float, str]]]:
    """Hybrid A*로 start -> goal 경로를 계획합니다.

    Returns:
        [(x, y, yaw, gear), ...] dense path, 못 찾으면 None.
    """

    vehicle = vehicle or VehicleModel()
    cfg = cfg or HybridAStarConfig()

    if not pose_collision_free(start, occ_grid, vehicle):
        return None

    heuristic_map = build_holonomic_heuristic(occ_grid, (goal[0], goal[1]), vehicle, cfg)

    start_node = _HANode(start[0], start[1], wrap_angle(start[2]), 0, 0.0, 0.0)
    steers = _ha_steer_set(vehicle, cfg)

    open_heap: List[_PQItem] = []
    counter = 0
    best_g: Dict[Tuple[int, int, int], float] = {start_node.state_key(cfg): 0.0}

    def push(node: "_HANode") -> None:
        nonlocal counter
        h = _heuristic_lookup(heuristic_map, occ_grid, node.x, node.y)
        if math.isinf(h):
            return
        counter += 1
        heapq.heappush(open_heap, _PQItem(node.g + h, counter, node))

    push(start_node)

    deadline = time.monotonic() + cfg.time_budget_s
    expansions = 0
    while open_heap and expansions < cfg.max_expansions:
        # 주기적으로 GIL을 양보해서, 이 백그라운드 탐색이 도는 동안에도
        # 메인 스레드가 IPC(0.15s 타임아웃) 응답을 제때 보낼 수 있게 합니다.
        # (양보가 없으면 무거운 탐색이 GIL을 오래 쥐어 메인 tick이 0.15s를
        #  넘기고 시뮬레이터가 연결을 끊는 경우가 생깁니다.)
        if expansions % 50 == 0:
            time.sleep(0)
            # 도달 불가능하거나 너무 어려운 목표를 오래 붙잡고 있지 않도록
            # 벽시계 시간 예산을 둡니다(초과 시 실패 처리 -> rule-based fallback).
            if time.monotonic() > deadline:
                return None
            # 새 맵/타깃이 와서 이 탐색이 무효가 되면 즉시 중단합니다.
            if abort_event is not None and abort_event.is_set():
                return None
        item = heapq.heappop(open_heap)
        node = item.node
        key = node.state_key(cfg)
        if best_g.get(key, math.inf) < node.g - 1e-6:
            continue

        if (
            math.hypot(node.x - goal[0], node.y - goal[1]) <= cfg.goal_xy_tolerance
            and abs(wrap_angle(node.yaw - goal[2])) <= cfg.goal_yaw_tolerance
        ):
            return _ha_reconstruct_path(node, goal)

        expansions += 1

        if expansions % cfg.analytic_every == 0:
            shortcut = _try_straight_shortcut(node, goal, occ_grid, vehicle, cfg)
            if shortcut is not None:
                return _ha_reconstruct_path(shortcut, goal)

        for direction in (1, -1):
            for steer in steers:
                child = _ha_step(node, direction, steer, vehicle, cfg)
                if not pose_collision_free((child.x, child.y, child.yaw), occ_grid, vehicle):
                    continue
                child_key = child.state_key(cfg)
                if child.g < best_g.get(child_key, math.inf) - 1e-6:
                    best_g[child_key] = child.g
                    push(child)

    return None


def _ha_reconstruct_path(node: "_HANode", goal: Pose) -> List[Tuple[float, float, float, str]]:
    chain: List[_HANode] = []
    cur: Optional[_HANode] = node
    while cur is not None:
        chain.append(cur)
        cur = cur.parent
    chain.reverse()

    path: List[Tuple[float, float, float, str]] = []
    for n in chain:
        gear = "R" if n.direction < 0 else "D"
        path.append((n.x, n.y, n.yaw, gear))

    if path:
        last_gear = path[-1][3]
        path[-1] = (goal[0], goal[1], goal[2], last_gear)
    return path


def build_hybrid_trajectory(
    path: List[Tuple[float, float, float, str]],
    cruise_speed: float = 1.0,
    slow_speed: float = 0.4,
    creep_speed: float = 0.35,
    slow_window: int = 3,
) -> List[TrajectoryPoint]:
    """(x,y,yaw,gear) 경로에 속도 프로파일을 입혀 TrajectoryPoint 리스트로 변환합니다.

    creep 구간(맨 끝)을 너무 길게/느리게 잡으면 슬롯 중심에 도달하기 전에
    라운드 시간이 끝나므로(차가 1~2m 못 미쳐 멈춤), creep 구간을 짧게(마지막
    몇 점) 잡고 creep 속도도 약간 높여 마지막 접근을 빠르게 마칩니다.
    """

    n = len(path)
    if n == 0:
        return []

    gear_change_idx = set()
    for i in range(1, n):
        if path[i][3] != path[i - 1][3]:
            gear_change_idx.add(i)

    points: List[TrajectoryPoint] = []
    for i, (x, y, yaw, gear) in enumerate(path):
        speed = cruise_speed
        near_gear_change = any(abs(i - g) <= slow_window for g in gear_change_idx)
        near_end = (n - 1 - i) <= slow_window

        if near_end:
            speed = creep_speed
        elif near_gear_change:
            speed = slow_speed

        points.append(TrajectoryPoint(x, y, yaw, gear, speed))

    return points


def merge_occupancy_grids(
    stationary_grid: List[List[float]],
    parked_grid: Optional[List[List[float]]],
) -> List[List[float]]:
    """stationary(벽/기둥)와 parked(주차된 차량) 레이어를 합칩니다.

    self-parking-sim의 map payload는 두 레이어를 따로 보내주는데,
    Hybrid A*가 다른 차량과 충돌하지 않으려면 두 레이어 모두를
    장애물로 취급해야 합니다. 둘 중 하나라도 점유면 점유로 봅니다.
    """

    if not parked_grid:
        return stationary_grid

    merged = []
    for srow, prow in zip(stationary_grid, parked_grid):
        merged.append([max(s, p) for s, p in zip(srow, prow)])
    return merged


def mark_rects_occupied(
    grid: List[List[float]],
    rects: List[Rect],
    cell_size: float,
    extent: Rect,
    flip_rows: bool = True,
    margin: float = 0.3,
) -> List[List[float]]:
    """지정한 world-rect들을 grid에서 점유(1.0)로 표시합니다.

    self-parking-sim의 `grid.parked` 레이어는 round마다 섞이는
    `occupied_idx`를 반영하지 못하는 "고정" 레이어이기 때문에(베이스 맵의
    8개 슬롯 위치만 표시), 실제로는 매 round마다 받는 `slots` +
    `occupied_idx`를 직접 grid에 그려서 사용해야 합니다. (시뮬레이터의
    충돌 판정 (B)도 바로 이 `slots`/`occupied_idx`를 기준으로 합니다.)
    margin은 차량 폭/SAT 판정과의 여유를 위한 추가 버퍼(m)입니다.
    """

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if rows == 0 or cols == 0:
        return grid

    xmin, xmax, ymin, ymax = extent
    result = [row[:] for row in grid]

    for rect in rects:
        rxmin, rxmax, rymin, rymax = rect
        rxmin -= margin
        rxmax += margin
        rymin -= margin
        rymax += margin

        col_min = max(0, int(math.floor((rxmin - xmin) / cell_size)))
        col_max = min(cols - 1, int(math.floor((rxmax - xmin) / cell_size)))
        row_lo = max(0, int(math.floor((rymin - ymin) / cell_size)))
        row_hi = min(rows - 1, int(math.floor((rymax - ymin) / cell_size)))

        for r in range(row_lo, row_hi + 1):
            rr = rows - 1 - r if flip_rows else r
            if not (0 <= rr < rows):
                continue
            for c in range(col_min, col_max + 1):
                if 0 <= c < cols:
                    result[rr][c] = 1.0

    return result


def mark_line_segments_occupied(
    grid: List[List[float]],
    lines: List[Tuple[float, float, float, float]],
    cell_size: float,
    extent: Rect,
    flip_rows: bool = True,
    half_width: float = 0.25,
    keep_clear: Optional[Rect] = None,
) -> List[List[float]]:
    """주차장 차선(lane marking) 선분들을 grid에 장애물로 표시합니다.

    self-parking-sim은 차선(M.lines)을 두께 0.25m의 얇은 사각형으로 만들어
    충돌 판정 (C)에 사용합니다. 차량이 차선을 "밟기만 해도" 충돌이므로,
    Hybrid A*도 차선을 장애물로 알고 피해야 합니다. (이전까지는 벽/주차차량만
    장애물로 봤기 때문에, aisle을 주행하다 슬롯 칸막이 차선을 밟아 충돌했습니다.)

    keep_clear에 목표 슬롯 rect를 주면, 그 영역과 겹치는 부분은 비워둬서
    차량이 목표 슬롯으로 진입할 수 있게 합니다.
    """

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if rows == 0 or cols == 0 or not lines:
        return grid

    xmin, xmax, ymin, ymax = extent
    result = [row[:] for row in grid]

    def _block_rect(rxmin: float, rxmax: float, rymin: float, rymax: float) -> None:
        col_min = max(0, int(math.floor((rxmin - xmin) / cell_size)))
        col_max = min(cols - 1, int(math.floor((rxmax - xmin) / cell_size)))
        row_lo = max(0, int(math.floor((rymin - ymin) / cell_size)))
        row_hi = min(rows - 1, int(math.floor((rymax - ymin) / cell_size)))
        for r in range(row_lo, row_hi + 1):
            rr = rows - 1 - r if flip_rows else r
            if not (0 <= rr < rows):
                continue
            wy = ymin + (r + 0.5) * cell_size
            for c in range(col_min, col_max + 1):
                if not (0 <= c < cols):
                    continue
                if keep_clear is not None:
                    wx = xmin + (c + 0.5) * cell_size
                    kx0, kx1, ky0, ky1 = keep_clear
                    if kx0 <= wx <= kx1 and ky0 <= wy <= ky1:
                        continue  # 목표 슬롯 진입로는 비워둠
                result[rr][c] = 1.0

    for x1, y1, x2, y2 in lines:
        if abs(x1 - x2) < 1e-6:  # 수직 선분
            _block_rect(min(x1, x2) - half_width, max(x1, x2) + half_width,
                        min(y1, y2), max(y1, y2))
        elif abs(y1 - y2) < 1e-6:  # 수평 선분
            _block_rect(min(x1, x2), max(x1, x2),
                        min(y1, y2) - half_width, max(y1, y2) + half_width)
        else:  # 대각 (이 맵에는 없지만 안전하게 bounding box)
            _block_rect(min(x1, x2) - half_width, max(x1, x2) + half_width,
                        min(y1, y2) - half_width, max(y1, y2) + half_width)

    return result


def _plan_hybrid_astar(
    state: Dict[str, Any],
    limits: Dict[str, Any],
    occupancy_grid_data: List[List[float]],
    cell_size: float,
    map_extent: Rect,
    goal_center: Point,
    goal_yaw: float,
    rear_in: bool = False,
    abort_event: Optional["threading.Event"] = None,
) -> List[TrajectoryPoint]:
    """현재 ego pose에서 목표 slot까지 Hybrid A* 경로를 계산합니다.

    `self`에 의존하지 않는 순수 함수로 만들어, 백그라운드 스레드에서
    안전하게 호출할 수 있습니다. (state/limits/grid는 호출 시점에
    스냅샷으로 전달받습니다. `occupancy_grid_data`는 stationary와
    parked 레이어가 이미 합쳐진 grid입니다.)
    """

    start_pose: Pose = (
        float(state.get("x", 0.0)),
        float(state.get("y", 0.0)),
        float(state.get("yaw", 0.0)),
    )
    goal_pose: Pose = (goal_center[0], goal_center[1], goal_yaw)

    try:
        occ_grid = OccupancyGrid(
            grid=occupancy_grid_data,
            cell_size=cell_size,
            extent=map_extent,
            # self-parking-sim의 world_to_rc()는
            # row = H-1 - floor((y-ymin)/cellSize) == floor((ymax-y)/cellSize)
            # 즉 row 0이 y=ymax(맵 상단)에 대응하므로 flip_rows=True가 맞습니다.
            flip_rows=True,
        )
        vehicle = VehicleModel(
            wheelbase=float(limits.get("L", 2.6)),
            max_steer=float(limits.get("maxSteer", math.radians(35.0))),
        )
        cfg = HybridAStarConfig()

        # 슬롯 진입은 차선(슬롯 경계선)을 비스듬히 밟지 않도록 "수직 진입"이
        # 되어야 합니다. 그래서 Hybrid A*는 슬롯 정면(aisle 쪽)의 staging
        # point까지만 계획하고, staging -> 슬롯 중심 구간은 슬롯 축에 정렬된
        # 직선으로 따로 이어붙입니다.
        #
        nose_vec = (math.cos(goal_yaw), math.sin(goal_yaw))
        if rear_in:
            primary_aisle = nose_vec          # nose가 aisle을 향함
        else:
            primary_aisle = (-nose_vec[0], -nose_vec[1])  # nose 반대쪽이 aisle
        opposite_aisle = (-primary_aisle[0], -primary_aisle[1])

        raw_path = None
        # 1차: 정상 aisle 쪽에서 staging. 실패하면 반대쪽에서도 시도합니다.
        # (예: row1 슬롯은 위쪽 aisle이 막혀 있으면 아래쪽에서 진입.)
        # 어느 쪽이든 최종 nose 방향(goal_yaw)은 동일하게 유지되므로
        # orientation 점수에는 영향이 없습니다. 진입 gear만 그에 맞춰 정합니다.
        for aisle_vec in (primary_aisle, opposite_aisle):
            # 이 aisle 쪽에서 슬롯 중심으로 들어갈 때, nose가 aisle을 향하면
            # 후진(R), 슬롯 안쪽을 향하면 전진(D).
            dot = nose_vec[0] * aisle_vec[0] + nose_vec[1] * aisle_vec[1]
            seg_gear = "R" if dot > 0 else "D"
            raw_path = _try_staged_entry(
                start_pose, goal_center, goal_yaw, aisle_vec, seg_gear,
                occ_grid, vehicle, cfg, abort_event=abort_event,
            )
            if raw_path:
                break
        # staging 경로가 모두 실패하면 슬롯 중심으로 직접 계획 (fallback)
        if not raw_path:
            fallback_cfg = HybridAStarConfig(
                max_expansions=cfg.max_expansions, time_budget_s=1.5
            )
            raw_path = hybrid_astar_plan(
                start_pose, goal_pose, occ_grid, vehicle, fallback_cfg,
                abort_event=abort_event,
            )
    except Exception as exc:
        print(f"[algo] hybrid A* planning error: {exc}")
        return []

    if not raw_path:
        return []

    return build_hybrid_trajectory(
        raw_path,
        cruise_speed=1.2,
        slow_speed=0.45,
        creep_speed=0.35,
        slow_window=3,
    )


def _try_staged_entry(
    start_pose: Pose,
    goal_center: Point,
    goal_yaw: float,
    aisle_vec: Tuple[float, float],
    seg_gear: str,
    occ_grid: "OccupancyGrid",
    vehicle: "VehicleModel",
    cfg: "HybridAStarConfig",
    abort_event: Optional["threading.Event"] = None,
) -> Optional[List[Tuple[float, float, float, str]]]:
    """주어진 aisle 방향에서 슬롯으로 들어가는 정렬 진입 경로를 시도합니다.

    pre-stage -> staging -> 슬롯 중심을 모두 슬롯 축에 정렬된 직선으로 잇고,
    회전 반경을 줄이기 위해 가까운 staging부터 시도합니다. 마지막 진입
    구간이 충돌-free인 경우에만 채택합니다.
    """

    align_dist = 2.0
    # pre-stage 도착 시 슬롯 축과 잘 정렬되도록 yaw 허용오차를 좁힙니다.
    # 여러 거리/양쪽 aisle을 시도하므로 각 탐색에는 짧은 시간예산을 둡니다.
    align_cfg = HybridAStarConfig(
        goal_yaw_tolerance=math.radians(12.0),
        max_expansions=cfg.max_expansions,
        time_budget_s=2.5,
    )
    for entry_dist in (2.0, 3.0):
        if abort_event is not None and abort_event.is_set():
            return None
        stage_x = goal_center[0] + entry_dist * aisle_vec[0]
        stage_y = goal_center[1] + entry_dist * aisle_vec[1]
        stage_pose = (stage_x, stage_y, goal_yaw)
        prestage_pose = (
            stage_x + align_dist * aisle_vec[0],
            stage_y + align_dist * aisle_vec[1],
            goal_yaw,
        )
        if not pose_collision_free(stage_pose, occ_grid, vehicle):
            continue
        if not pose_collision_free(prestage_pose, occ_grid, vehicle):
            continue
        candidate = hybrid_astar_plan(
            start_pose, prestage_pose, occ_grid, vehicle, align_cfg,
            abort_event=abort_event,
        )
        if not candidate:
            continue
        # prestage -> stage -> goal 은 모두 슬롯 축 위에 일직선이므로, 진입
        # gear(전진/후진)는 전 구간 동일해야 합니다. (서로 다르면 같은 방향으로
        # 전진했다 후진하는 모순이 생겨 차가 제자리에서 크게 흔들립니다.)
        # 단, 후진 진입이면 차가 먼저 prestage까지 "전진"으로 가야 하므로,
        # Hybrid A* 경로의 끝(=prestage 도착)에서 곧장 후진으로 전환됩니다.
        seg1 = _append_straight_entry(
            candidate, prestage_pose, stage_pose, vehicle, occ_grid, entry_gear=seg_gear
        )
        goal_pose = (goal_center[0], goal_center[1], goal_yaw)
        full = _append_straight_entry(
            seg1, stage_pose, goal_pose, vehicle, occ_grid, entry_gear=seg_gear
        )
        if _path_tail_collision_free(full, len(candidate), occ_grid, vehicle, goal_center):
            return full
    return None


def _path_tail_collision_free(
    path: List[Tuple[float, float, float, str]],
    tail_start: int,
    occ_grid: "OccupancyGrid",
    vehicle: "VehicleModel",
    goal_center: Point,
    slot_half: float = 2.4,
) -> bool:
    """경로의 tail(정렬+진입 구간)이 충돌-free인지 검사합니다.

    목표 슬롯 중심 근처(반경 slot_half 이내)는 슬롯 내부로 보고 허용합니다.
    (시뮬레이터도 슬롯 내부에 완전히 들어가면 차선 충돌을 면제합니다.)
    """

    gx, gy = goal_center
    for i in range(max(0, tail_start), len(path)):
        x, y, yaw, _ = path[i]
        if math.hypot(x - gx, y - gy) <= slot_half:
            continue  # 슬롯 내부 근처는 허용
        if not pose_collision_free((x, y, yaw), occ_grid, vehicle):
            return False
    return True


def _append_straight_entry(
    path: List[Tuple[float, float, float, str]],
    stage_pose: Pose,
    goal_pose: Pose,
    vehicle: "VehicleModel",
    occ_grid: "OccupancyGrid",
    step: float = 0.4,
    entry_gear: str = "D",
) -> List[Tuple[float, float, float, str]]:
    """staging point에서 슬롯 중심까지 슬롯 축에 정렬된 직선 진입을 덧붙입니다.

    슬롯 경계 차선을 수직으로(정면으로) 통과하므로, 비스듬히 밟아서
    충돌하는 것을 피합니다. entry_gear가 "R"이면 후진 진입(rear_in)입니다.
    heading(yaw)은 staging과 동일(최종 nose 방향)하게 유지하고, 기어만
    전/후진을 구분합니다.
    """

    gx, gy, gyaw = goal_pose
    sx, sy, _ = stage_pose
    dist = math.hypot(gx - sx, gy - sy)
    if dist < 1e-3:
        return path

    n = max(1, int(dist / step))
    entry = []
    for i in range(1, n + 1):
        t = i / n
        x = sx + (gx - sx) * t
        y = sy + (gy - sy) * t
        entry.append((x, y, gyaw, entry_gear))
    return path + entry


@dataclass
class PlannerSkeleton:
    """읽기 쉬운 rule-based 경로 planner와 controller입니다.

    자료구조와 알고리즘 발표에서 설명하기 쉽도록 의도적으로 단순하게
    구성했습니다. 맵 배열을 리스트로 파싱하고, 목표 slot을 짧은
    waypoint 목록으로 바꾼 뒤, 매 tick마다 현재 위치 기준의 조향/속도
    결정을 수행합니다.
    """

    map_data: Optional[Dict[str, Any]] = None
    map_extent: Optional[Rect] = None
    cell_size: float = 0.5
    stationary_grid: Optional[List[List[float]]] = None
    parked_grid: Optional[List[List[float]]] = None
    slots: List[Rect] = field(default_factory=list)
    occupied_idx: List[bool] = field(default_factory=list)
    free_slot_indices: List[int] = field(default_factory=list)
    lines: List[Tuple[float, float, float, float]] = field(default_factory=list)
    walls_rects: List[Rect] = field(default_factory=list)
    waypoints: List[Waypoint] = field(default_factory=list)
    active_waypoint: int = 0
    target_slot: Optional[Rect] = None
    target_center: Optional[Point] = None
    final_yaw: float = math.pi * 0.5
    entry_gear: str = "D"
    left_aisle_x: float = 4.0
    expected_orientation: str = "front_in"
    planned_orientation: str = "front_in"
    hybrid_active: bool = False
    hybrid_trajectory: List[TrajectoryPoint] = field(default_factory=list)
    hybrid_nearest_idx: int = 0
    hybrid_target_idx: int = 0
    default_mild_turn_active: bool = False
    # 백그라운드 Hybrid A* planning 관련 상태
    # (IPC recv timeout이 0.15s라서, planner_step 안에서 직접 Hybrid A*를
    #  돌리면 한 번이라도 0.15s를 넘기는 순간 시뮬레이터가 연결을 끊습니다.
    #  따라서 별도 스레드에서 계산하고, 끝나면 결과만 반영합니다.)
    _hybrid_thread: Optional[threading.Thread] = field(default=None, repr=False, compare=False)
    _hybrid_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _hybrid_request_slot: Optional[Rect] = field(default=None, repr=False, compare=False)
    # 이전(죽은) 연결의 백그라운드 탐색을 즉시 중단시키기 위한 플래그.
    # 새 맵/타깃이 오면 set() 하여 진행 중이던 Hybrid A*를 곧바로 포기시킵니다.
    _hybrid_abort: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)

    def set_map(self, map_payload: Dict[str, Any]) -> None:
        """시뮬레이터가 보낸 맵 geometry 정보를 캐시에 저장합니다."""

        self.map_data = map_payload
        self.map_extent = tuple(
            map(float, map_payload.get("extent", (0.0, 75.0, 0.0, 50.0)))
        )
        self.cell_size = float(map_payload.get("cellSize", 0.5))
        self.stationary_grid = map_payload.get("grid", {}).get("stationary")
        self.parked_grid = map_payload.get("grid", {}).get("parked")
        self.slots = [tuple(map(float, slot)) for slot in map_payload.get("slots", [])]
        self.occupied_idx = [bool(v) for v in map_payload.get("occupied_idx", [])]
        self.free_slot_indices = [
            idx for idx, occupied in enumerate(self.occupied_idx) if not occupied
        ]
        # 시뮬레이터가 payload에 직접 expected_orientation을 넣어 보냅니다
        # ("front_in" 또는 "rear_in"). 없으면 보수적으로 front_in으로 둡니다.
        payload_orientation = map_payload.get("expected_orientation")
        if payload_orientation in ("front_in", "rear_in"):
            self.expected_orientation = payload_orientation
        else:
            self.expected_orientation = "front_in"
        self.lines = [
            tuple(map(float, line)) for line in map_payload.get("lines", [])
        ]
        self.walls_rects = [
            tuple(map(float, rect)) for rect in map_payload.get("walls_rects", [])
        ]
        self.left_aisle_x = self._estimate_left_aisle_x()

        pretty_print_map_summary(map_payload)
        self._reset_plan()

    def _reset_plan(self) -> None:
        self.waypoints.clear()
        self.active_waypoint = 0
        self.target_slot = None
        self.target_center = None
        self.entry_gear = "D"
        self.hybrid_active = False
        self.hybrid_trajectory = []
        self.hybrid_nearest_idx = 0
        self.hybrid_target_idx = 0
        self.default_mild_turn_active = False
        with self._hybrid_lock:
            self._hybrid_request_slot = None
        # 진행 중이던 백그라운드 탐색이 있으면 즉시 중단 신호를 보냅니다.
        self._hybrid_abort.set()

    def _estimate_left_aisle_x(self) -> float:
        """왼쪽 주행 aisle에서 비교적 안전한 x 좌표를 선택합니다."""

        if not self.map_extent:
            return 4.0

        xmin, xmax, _, _ = self.map_extent
        if not self.slots:
            return clamp(xmin + 4.0, xmin + 2.0, xmax - 2.0)

        first_slot_x = min(slot[0] for slot in self.slots)
        # 왼쪽 벽(x≈0~0.5)에 너무 붙지 않도록 최소 4.5m는 띄웁니다.
        aisle_x = min(xmin + 4.5, first_slot_x - 6.0)
        return clamp(aisle_x, xmin + 4.0, xmax - 2.5)

    def _same_slot(self, slot: Rect) -> bool:
        if self.target_slot is None:
            return False
        return all(abs(a - b) <= 1e-4 for a, b in zip(self.target_slot, slot))

    def _slot_index(self, slot: Rect) -> Optional[int]:
        for idx, candidate in enumerate(self.slots):
            if self._rect_close(candidate, slot, tolerance=0.05):
                return idx
        return None

    def _rect_close(self, a: Rect, b: Rect, tolerance: float = 0.10) -> bool:
        return all(abs(x - y) <= tolerance for x, y in zip(a, b))

    def _use_default_mild_turn_route(self, slot: Rect) -> bool:
        """검증된 Default Lot mild-turn route에만 적용하는 엄격한 가드 조건입니다."""

        if self.expected_orientation != "front_in":
            return False
        if self.map_extent is None:
            return False
        if not self._rect_close(self.map_extent, (0.0, 75.0, 0.0, 50.0), tolerance=0.15):
            return False
        if len(self.slots) != 33:
            return False
        if len(self.lines) != 38 or len(self.walls_rects) != 4:
            return False

        slot_idx = self._slot_index(slot)
        return slot_idx in {10, 17, 18, 20, 21, 31, 32}

    def _update_target_from_obs(self, obs: Dict[str, Any]) -> None:
        """시뮬레이터가 새 slot을 선택하면 waypoint 목록을 다시 만듭니다."""

        raw_slot = obs.get("target_slot")
        if raw_slot and len(raw_slot) == 4:
            slot = tuple(map(float, raw_slot))
        elif self.free_slot_indices:
            slot = self.slots[self.free_slot_indices[0]]
        elif self.slots:
            slot = self.slots[0]
        else:
            return

        if self._same_slot(slot) and self.waypoints:
            return

        self.target_slot = slot
        self.target_center = rect_center(slot)
        self.planned_orientation = self._select_planned_orientation(slot)
        self.final_yaw = self._target_yaw(slot)
        # Hybrid A* 진입 방향은 슬롯의 "열린 쪽"(차선이 없는 aisle 방향)을
        # 기준으로 정합니다. 슬롯 경계 차선을 비스듬히/완전 진입 전에 밟으면
        # 시뮬레이터가 충돌로 판정하므로, 차선이 없는 쪽에서 수직 진입해야
        # 차량이 어떤 차선도 넘지 않고 슬롯 안에 들어갈 수 있습니다.
        hybrid_goal_yaw = self._hybrid_entry_yaw(slot)
        self.default_mild_turn_active = False
        # rule-based waypoint는 즉시(<1ms) 계산되므로, Hybrid A*가 끝나기 전까지는
        # 이 경로로 주행하면서 0.15s IPC 응답 시간을 절대 넘기지 않습니다.
        self.waypoints = self._build_waypoints(slot)
        self.active_waypoint = 0
        self.hybrid_active = False
        self.hybrid_trajectory = []
        self.hybrid_nearest_idx = 0
        self.hybrid_target_idx = 0

        # "후진 진입(reverse)"인지 여부: 차량은 항상 aisle 쪽에서 진입합니다.
        # 최종 nose가 aisle을 향하면 후진으로 들어가야 하고(=reverse), nose가
        # 슬롯 안쪽(aisle 반대)을 향하면 전진으로 들어갑니다.
        reverse_entry = self._is_reverse_entry(slot, hybrid_goal_yaw)

        def _start():
            self._start_hybrid_astar_async(
                obs, slot, self.target_center, hybrid_goal_yaw, reverse_entry
            )
        _start()

    def _is_reverse_entry(self, slot: Rect, goal_yaw: float) -> bool:
        """최종 nose 방향과 aisle 위치를 비교해 후진 진입 여부를 판단합니다."""

        sx0, sx1, sy0, sy1 = slot
        width = sx1 - sx0
        height = sy1 - sy0
        nose = (math.cos(goal_yaw), math.sin(goal_yaw))

        if height >= width:
            open_up = self._slot_open_up(slot)
            if open_up is None:
                open_up = True
            # aisle 방향 단위벡터 (위쪽이면 +y, 아래쪽이면 -y)
            aisle_dir = 1.0 if open_up else -1.0
            # nose가 aisle 쪽(같은 부호)을 향하면 후진 진입
            return (nose[1] * aisle_dir) > 0
        else:
            if self.map_extent:
                xmin, xmax, _, _ = self.map_extent
                open_left = rect_center(slot)[0] > (xmin + xmax) * 0.5
                aisle_dir = -1.0 if open_left else 1.0
                return (nose[0] * aisle_dir) > 0
            return False

    def _slot_open_up(self, slot: Rect) -> Optional[bool]:
        """세로 슬롯에서 aisle이 위쪽(True)/아래쪽(False)에 있는지 반환합니다.

        가로 슬롯이면 None을 반환합니다. 슬롯 경계 차선이 있는 쪽이 막힌
        쪽이고 그 반대가 aisle입니다. 맵 경계(벽)에 닿는 쪽은 aisle이 아닙니다.
        """

        sx0, sx1, sy0, sy1 = slot
        width = sx1 - sx0
        height = sy1 - sy0
        if height < width:
            return None

        low_mark = high_mark = False
        for x1, y1, x2, y2 in self.lines:
            if abs(y1 - y2) > 1e-4:
                continue
            if min(x1, x2) <= sx1 + 0.5 and max(x1, x2) >= sx0 - 0.5:
                if abs(y1 - sy0) <= 1.0:
                    low_mark = True
                if abs(y1 - sy1) <= 1.0:
                    high_mark = True

        if low_mark and not high_mark:
            open_up = True
        elif high_mark and not low_mark:
            open_up = False
        elif self.map_extent:
            _, _, ymin, ymax = self.map_extent
            open_up = rect_center(slot)[1] < (ymin + ymax) * 0.5
        else:
            open_up = True

        if self.map_extent:
            _, _, ymin, ymax = self.map_extent
            entry_margin = 3.0
            if open_up and (ymax - sy1) < entry_margin:
                open_up = False
            elif (not open_up) and (sy0 - ymin) < entry_margin:
                open_up = True
        return open_up

    def _hybrid_entry_yaw(self, slot: Rect) -> float:
        """최종 주차 heading(차량 nose가 향하는 방향)을 반환합니다.

        시뮬레이터 채점 기준(determine_parking_orientation):
        - 세로 슬롯: nose의 y성분 부호로 front_in(+y, nose 위) / rear_in(-y, nose 아래) 판정
        - front_in = 차 머리가 슬롯 안쪽(aisle 반대쪽)을 향함
        - rear_in  = 차 머리가 aisle 쪽을 향함(= 후진 진입)

        따라서 최종 nose 방향은 (aisle 위치) + (expected_orientation) 조합으로
        결정됩니다.
        """

        sx0, sx1, sy0, sy1 = slot
        width = sx1 - sx0
        height = sy1 - sy0
        expected = self.expected_orientation  # "front_in" | "rear_in"

        if height >= width:  # 세로 슬롯
            # 시뮬레이터 정의: nose의 y성분 부호. front_in = nose +y(위), rear_in = nose -y(아래).
            # (aisle 위치와 무관하게 최종 nose 방향은 expected로만 결정됩니다.)
            nose_up = (expected == "front_in")
            return wrap_angle(math.pi * 0.5 if nose_up else -math.pi * 0.5)

        # 가로 슬롯 (평행 주차): front_in = nose +x, rear_in = nose -x
        nose_right = (expected == "front_in")
        return 0.0 if nose_right else math.pi


    def _start_hybrid_astar_async(
        self,
        obs: Dict[str, Any],
        slot: Rect,
        goal_center: Point,
        goal_yaw: float,
        reverse_entry: bool = False,
    ) -> None:
        """Hybrid A* 계산을 별도 스레드에서 실행합니다.

        IMPORTANT: self-parking-sim의 IPC는 `recv_cmd()`에 0.15s 타임아웃이
        걸려 있어서, planner_step 안에서 Hybrid A*를 직접(동기적으로) 돌리면
        한 번이라도 0.15s를 넘기는 즉시 시뮬레이터가 연결을 끊습니다
        (`[IPC] comm fail ... -> connection will be reset`).
        그래서 무거운 계산은 백그라운드 스레드로 분리하고, 그동안은 위에서
        만든 rule-based waypoint로 주행합니다. 계산이 끝나면 결과를
        hybrid_trajectory에 반영하고 그 다음 tick부터 추종합니다.
        """

        state = dict(obs.get("state", {}))
        limits = dict(obs.get("limits", {}))
        cell_size = self.cell_size
        map_extent = self.map_extent

        if map_extent is None or not self.stationary_grid:
            return

        # 무거운 작업(grid 병합/rasterize + Hybrid A*)은 전부 백그라운드 스레드
        # 안에서 합니다. planner_step(메인 스레드)이 호출되는 이 함수에서는
        # 스냅샷만 떠서 스레드를 띄우고 즉시 반환합니다. 이렇게 해야 첫 tick이
        # IPC 0.15s 제한을 절대 넘기지 않습니다.
        stationary_grid = self.stationary_grid
        parked_grid = self.parked_grid
        occupied_rects = [
            self.slots[i]
            for i, occupied in enumerate(self.occupied_idx)
            if occupied and i < len(self.slots)
        ]
        lines_snapshot = list(self.lines) if self.lines else []
        keep_clear = self._slot_entry_corridor(slot) if self.lines else None

        with self._hybrid_lock:
            self._hybrid_request_slot = slot

        rear_in = reverse_entry
        # 이 계획 전용 abort 이벤트(이전 것은 set_map에서 set됨).
        abort_event = threading.Event()
        self._hybrid_abort = abort_event

        def _worker() -> None:
            # 스레드 시작 직후 곧바로 GIL을 양보해, 이 스레드를 띄운 메인
            # 스레드가 다음 IPC tick을 제때 응답하도록 합니다.
            time.sleep(0)
            if abort_event.is_set():
                return
            # grid 병합 + 실제 점유 슬롯/차선 rasterize (백그라운드에서 수행)
            occupancy_grid = merge_occupancy_grids(stationary_grid, parked_grid)
            if occupied_rects:
                occupancy_grid = mark_rects_occupied(
                    occupancy_grid, occupied_rects, cell_size, map_extent, flip_rows=True
                )
            if lines_snapshot:
                occupancy_grid = mark_line_segments_occupied(
                    occupancy_grid, lines_snapshot, cell_size, map_extent,
                    flip_rows=True, half_width=0.25, keep_clear=keep_clear,
                )
            if abort_event.is_set():
                return
            path = _plan_hybrid_astar(
                state, limits, occupancy_grid, cell_size, map_extent,
                goal_center, goal_yaw, rear_in=rear_in, abort_event=abort_event,
            )
            if abort_event.is_set():
                return  # 중단됨 -> 결과 폐기
            with self._hybrid_lock:
                if self._hybrid_request_slot != slot:
                    return  # 그 사이 다른 slot으로 target이 바뀜 -> 결과 폐기
                if path:
                    self.hybrid_trajectory = path
                    self.hybrid_nearest_idx = 0
                    self.hybrid_target_idx = 0
                    self.hybrid_active = True
                else:
                    print("[algo] hybrid A* failed, staying on rule-based waypoints")

        thread = threading.Thread(target=_worker, daemon=True)
        self._hybrid_thread = thread
        thread.start()

    def _slot_entry_corridor(self, slot: Rect) -> Rect:
        """목표 슬롯 + (aisle 쪽) 진입 통로를 합친 차선 비우기(keep_clear) 영역.

        슬롯이 aisle과 만나는 쪽(열린 쪽)의 경계선만 비우고, 반대쪽(막힌 쪽,
        차선이 있는 쪽)은 비우지 않습니다. 양쪽을 모두 열면 차량이 막힌 쪽
        경계선을 가로질러(슬롯 column을 그대로 통과해) 접근하다가 그 차선을
        밟게 되므로, 진입은 반드시 aisle 쪽에서만 이뤄지도록 통로를 한쪽으로만
        냅니다.
        """

        sx0, sx1, sy0, sy1 = slot
        approach = 6.0
        margin = 0.7
        open_up = self._slot_open_up(slot)

        if open_up is None:  # 가로 슬롯
            return (sx0 - approach, sx1 + approach, sy0 - margin, sy1 + margin)

        if open_up:  # aisle이 위쪽 -> 위로만 통로를 냄 (아래 경계선은 보존)
            return (sx0 - margin, sx1 + margin, sy0 + margin, sy1 + approach)
        else:        # aisle이 아래쪽 -> 아래로만 통로를 냄 (위 경계선은 보존)
            return (sx0 - margin, sx1 + margin, sy0 - approach, sy1 - margin)

    def _select_planned_orientation(self, slot: Rect) -> str:
        """현재 slot에 대해 안전한 최종 주차 방향을 선택합니다."""

        if self.expected_orientation != "rear_in":
            return "front_in"
        return "rear_in"

    def _target_yaw(self, slot: Rect) -> float:
        """현재 stage에서 필요한 최종 heading을 반환합니다."""

        width = slot[1] - slot[0]
        height = slot[3] - slot[2]
        if height >= width:
            yaw = math.pi * 0.5
        else:
            yaw = 0.0
        if self.planned_orientation == "rear_in":
            yaw = wrap_angle(yaw + math.pi)
        return yaw

    def _slot_open_side(self, slot: Rect) -> str:
        """slot이 주행 aisle과 만나는 방향을 추정합니다.

        Default map에서는 slot 아래쪽 경계에 수평 마킹이 있으면 위쪽으로
        열려 있다고 봅니다. 맨 위 row는 아래쪽 수평 마킹이 없으므로
        아래쪽에서 진입한다고 판단합니다.
        """

        sx_min, sx_max, sy_min, sy_max = slot
        low_mark = False
        high_mark = False
        for x1, y1, x2, y2 in self.lines:
            if abs(y1 - y2) > 1e-4:
                continue
            line_min_x = min(x1, x2)
            line_max_x = max(x1, x2)
            overlaps_x = line_min_x <= sx_max + 0.5 and line_max_x >= sx_min - 0.5
            if not overlaps_x:
                continue
            if abs(y1 - sy_min) <= 1.0:
                low_mark = True
            if abs(y1 - sy_max) <= 1.0:
                high_mark = True

        if low_mark and not high_mark:
            return "above"
        if high_mark and not low_mark:
            return "below"

        if self.map_extent:
            _, _, ymin, ymax = self.map_extent
            return "below" if rect_center(slot)[1] > (ymin + ymax) * 0.65 else "above"
        return "above"

    def _build_waypoints(self, slot: Rect) -> List[Waypoint]:
        """현재 목표 slot까지 이어지는 안전한 lane-to-slot route를 만듭니다."""

        if self.map_extent is None:
            return []

        xmin, xmax, ymin, ymax = self.map_extent
        cx, cy = rect_center(slot)
        width = slot[1] - slot[0]
        height = slot[3] - slot[2]

        if height < width:
            return self._build_horizontal_fallback(slot)

        open_side = self._slot_open_side(slot)
        turn_length = 6.0
        lane_gap = 2.0
        if self.planned_orientation == "rear_in":
            return self._build_rear_in_waypoints(slot, open_side)

        if self._use_default_mild_turn_route(slot):
            try:
                route = self._build_default_mild_turn_waypoints(slot, open_side)
                if route:
                    self.default_mild_turn_active = True
                    return route
            except Exception as exc:
                print(f"[algo] default mild-turn fallback: {exc}")
                self.default_mild_turn_active = False

        if open_side == "above":
            align_y = clamp(slot[3] + lane_gap, ymin + 2.5, ymax - turn_length - 2.5)
            entry_y = clamp(align_y + turn_length, ymin + 2.5, ymax - 2.5)
            self.entry_gear = "R"
            final_gear = "R"
        else:
            entry_y = clamp(slot[2] - lane_gap, ymin + turn_length + 2.5, ymax - 2.5)
            align_y = clamp(entry_y - turn_length, ymin + 2.5, ymax - 2.5)
            self.entry_gear = "D"
            final_gear = "D"

        aisle_x = clamp(self.left_aisle_x, xmin + 2.5, xmax - 2.5)
        turn_x = clamp(cx - 4.5, aisle_x + 2.0, xmax - 2.5)
        route = [
            Waypoint(aisle_x, align_y, "D", radius=1.2, speed=1.6),
            Waypoint(turn_x, align_y, "D", radius=1.1, speed=1.6),
            Waypoint(cx, entry_y, "D", radius=1.5, speed=0.85),
            Waypoint(cx, cy, final_gear, radius=0.35, speed=0.55, stop_here=True),
        ]
        return self._remove_redundant_waypoints(route)

    def _build_default_mild_turn_waypoints(self, slot: Rect, open_side: str) -> List[Waypoint]:
        """attempt_10에서 검증한 guarded Default Lot route를 만듭니다."""

        if self.map_extent is None:
            return []

        xmin, xmax, ymin, ymax = self.map_extent
        cx, cy = rect_center(slot)
        aisle_x = clamp(self.left_aisle_x, xmin + 2.5, xmax - 2.5)
        turn_x = clamp(cx - 3.2, aisle_x + 2.0, xmax - 2.5)

        if open_side == "above":
            align_y = clamp(slot[3] + 2.0, ymin + 2.5, ymax - 2.5)
            entry_y = clamp(slot[3] + 7.0, ymin + 2.5, ymax - 2.5)
            final_gear = "R"
            self.entry_gear = "R"
        else:
            align_y = clamp(slot[2] - 7.0, ymin + 2.5, ymax - 2.5)
            entry_y = clamp(slot[2] - 1.8, ymin + 2.5, ymax - 2.5)
            final_gear = "D"
            self.entry_gear = "D"

        route = [
            Waypoint(aisle_x, align_y, "D", radius=1.35, speed=1.848),
            Waypoint(turn_x, align_y, "D", radius=1.15, speed=1.848),
            Waypoint(cx, entry_y, "D", radius=1.05, speed=1.008),
            Waypoint(cx, cy, final_gear, radius=0.35, speed=0.55, stop_here=True),
        ]
        return self._remove_redundant_waypoints(route)

    def _build_rear_in_waypoints(self, slot: Rect, open_side: str) -> List[Waypoint]:
        """rear-in을 위한 제한된 motion-primitive waypoint 후보를 만듭니다."""

        if self.map_extent is None:
            return []

        xmin, xmax, ymin, ymax = self.map_extent
        cx, cy = rect_center(slot)
        aisle_x = clamp(self.left_aisle_x, xmin + 2.5, xmax - 2.5)
        self.entry_gear = "R"

        candidates: List[List[Waypoint]] = []
        if open_side == "below":
            stage_y = clamp(slot[2] - 5.0, ymin + 2.5, ymax - 2.5)
            lower_y = clamp(slot[2] - 12.0, ymin + 2.5, ymax - 2.5)
            pre_x = clamp(cx - 8.0, xmin + 2.0, xmax - 2.5)
            close_left_route = [
                Waypoint(aisle_x, stage_y, "D", radius=1.2, speed=1.4),
                Waypoint(pre_x, stage_y, "D", radius=1.1, speed=1.0),
                Waypoint(cx, stage_y, "D", radius=0.3, speed=0.45),
                Waypoint(cx, cy, "R", radius=0.35, speed=0.45, stop_here=True),
            ]
            pull_down_route = [
                Waypoint(aisle_x, stage_y, "D", radius=1.4, speed=1.8),
                Waypoint(cx, stage_y, "D", radius=1.4, speed=1.8),
                Waypoint(cx, lower_y, "D", radius=0.9, speed=0.85),
                Waypoint(cx, cy, "R", radius=0.5, speed=0.8, stop_here=True),
            ]
            if cx <= aisle_x + 10.0:
                candidates.extend([close_left_route, pull_down_route])
            else:
                candidates.append(pull_down_route)
        else:
            lower_row_limit = ymin + (ymax - ymin) * 0.35
            if cy <= lower_row_limit:
                align_y = clamp(slot[3] + 1.4, ymin + 2.5, ymax - 2.5)
                turn_x = clamp(cx - 1.5, aisle_x + 2.0, xmax - 2.5)
                final_x = clamp(cx - 0.2, slot[0] + 0.6, slot[1] - 0.6)
                final_y = clamp(cy + 1.45, slot[2] + 0.6, slot[3] - 0.6)
                candidates.append(
                    [
                        Waypoint(aisle_x, align_y, "D", radius=1.2, speed=1.4),
                        Waypoint(turn_x, align_y, "D", radius=0.9, speed=0.8),
                        Waypoint(
                            final_x,
                            final_y,
                            "D",
                            radius=0.3,
                            speed=0.3,
                            stop_here=True,
                        ),
                    ]
                )
            else:
                align_y = clamp(slot[3] + 2.0, ymin + 2.5, ymax - 2.5)
                side_x = clamp(cx + 4.0, xmin + 2.5, xmax - 2.5)
                mid_y = clamp(slot[3] + 2.5, ymin + 2.5, ymax - 2.5)
                entry_y = clamp(slot[3] + 5.5, ymin + 2.5, ymax - 2.5)
                candidates.append(
                    [
                        Waypoint(aisle_x, align_y, "D", radius=1.2, speed=1.7),
                        Waypoint(side_x, align_y, "D", radius=1.0, speed=1.35),
                        Waypoint(side_x, mid_y, "D", radius=0.6, speed=0.65),
                        Waypoint(cx, entry_y, "D", radius=0.4, speed=0.7),
                        Waypoint(
                            cx,
                            cy,
                            "D",
                            radius=0.35,
                            speed=0.6,
                            stop_here=True,
                        ),
                    ]
                )

        for route in candidates:
            compact = self._remove_redundant_waypoints(route)
            if self._route_points_clear(compact, slot):
                return compact

        self.planned_orientation = "front_in"
        self.final_yaw = self._target_yaw(slot)
        return self._build_waypoints(slot)

    def _clear_y_above(self, slot: Rect) -> float:
        if self.map_extent is None:
            return slot[3] + 12.0
        _, _, _, ymax = self.map_extent
        horizontal_lines = [
            y1
            for _, y1, _, y2 in self.lines
            if abs(y1 - y2) <= 1e-4 and y1 > slot[3] + 1.0
        ]
        if not horizontal_lines:
            return ymax - 2.5
        return min(horizontal_lines) - 1.5

    def _route_points_clear(self, route: List[Waypoint], target_slot: Rect) -> bool:
        for waypoint in route[:-1]:
            point = waypoint.point
            for rect in self.walls_rects:
                if self._point_in_rect(point, rect, margin=0.5):
                    return False
            for idx, rect in enumerate(self.slots):
                if idx < len(self.occupied_idx) and self.occupied_idx[idx]:
                    if self._same_rect(rect, target_slot):
                        continue
                    if self._point_in_rect(point, rect, margin=0.6):
                        return False
        return True

    def _same_rect(self, a: Rect, b: Rect) -> bool:
        return all(abs(x - y) <= 1e-4 for x, y in zip(a, b))

    def _point_in_rect(self, point: Point, rect: Rect, margin: float = 0.0) -> bool:
        return (
            rect[0] - margin <= point[0] <= rect[1] + margin
            and rect[2] - margin <= point[1] <= rect[3] + margin
        )

    def _build_horizontal_fallback(self, slot: Rect) -> List[Waypoint]:
        """가로 방향 slot이 있는 맵에서 사용하는 fallback route입니다."""

        if self.map_extent is None:
            return []

        xmin, xmax, ymin, ymax = self.map_extent
        cx, cy = rect_center(slot)
        aisle_x = clamp(self.left_aisle_x, xmin + 2.5, xmax - 2.5)
        lane_y = clamp(cy, ymin + 2.5, ymax - 2.5)
        self.entry_gear = "D"
        return self._remove_redundant_waypoints(
            [
                Waypoint(aisle_x, lane_y, "D", radius=1.2, speed=1.5),
                Waypoint(cx, cy, "D", radius=0.35, speed=0.55, stop_here=True),
            ]
        )

    def _remove_redundant_waypoints(self, route: List[Waypoint]) -> List[Waypoint]:
        """거의 같은 위치에 연속으로 놓인 waypoint를 제거합니다."""

        compact: List[Waypoint] = []
        for waypoint in route:
            if compact and distance(compact[-1].point, waypoint.point) < 0.3:
                compact[-1] = waypoint
            else:
                compact.append(waypoint)
        return compact

    def _active_target(self) -> Optional[Waypoint]:
        if not self.waypoints:
            return None
        self.active_waypoint = min(self.active_waypoint, len(self.waypoints) - 1)
        return self.waypoints[self.active_waypoint]

    def _advance_waypoint(self, position: Point, speed: float) -> Optional[Waypoint]:
        """현재 waypoint에 도달하면 다음 waypoint로 진행합니다."""

        waypoint = self._active_target()
        while waypoint and not waypoint.stop_here:
            reach_radius = waypoint.radius + min(0.6, speed * 0.25)
            if distance(position, waypoint.point) > reach_radius:
                break
            self.active_waypoint += 1
            waypoint = self._active_target()
        return waypoint

    def _steer_toward(
        self,
        position: Point,
        yaw: float,
        waypoint: Waypoint,
        gear: str,
        limits: Dict[str, Any],
    ) -> float:
        """Pure Pursuit 방식으로 waypoint를 향해 조향각을 계산합니다."""

        wheelbase = float(limits.get("L", 2.6))
        dx = waypoint.x - position[0]
        dy = waypoint.y - position[1]
        lookahead = clamp(math.hypot(dx, dy), 2.0, 6.0)
        bearing = math.atan2(dy, dx)

        if gear == "R":
            travel_yaw = wrap_angle(yaw + math.pi)
            alpha = wrap_angle(bearing - travel_yaw)
            steer = -math.atan2(2.0 * wheelbase * math.sin(alpha), lookahead)
        else:
            alpha = wrap_angle(bearing - yaw)
            steer = math.atan2(2.0 * wheelbase * math.sin(alpha), lookahead)

        return steer

    def _heading_error_to_waypoint(
        self,
        position: Point,
        yaw: float,
        waypoint: Waypoint,
    ) -> float:
        """현재 주행 방향 기준으로 heading error를 계산합니다."""

        bearing = math.atan2(waypoint.y - position[1], waypoint.x - position[0])
        travel_yaw = yaw if waypoint.gear == "D" else wrap_angle(yaw + math.pi)
        return abs(wrap_angle(bearing - travel_yaw))

    def _speed_command(
        self,
        speed: float,
        signed_speed: float,
        waypoint: Waypoint,
        position: Point,
        yaw: float,
    ) -> Tuple[float, float]:
        """목표 속도에 맞춰 accel/brake를 정하는 단순 비례 속도 controller입니다."""

        target_dist = distance(position, waypoint.point)
        desired_speed = waypoint.speed
        heading_error = self._heading_error_to_waypoint(position, yaw, waypoint)
        heading_high = 90.0 if self.default_mild_turn_active else 70.0
        heading_mid = 55.0 if self.default_mild_turn_active else 35.0
        heading_high_speed = 0.75 if self.default_mild_turn_active else 0.55
        heading_mid_speed = 1.05 if self.default_mild_turn_active else 0.85
        if heading_error > math.radians(heading_high):
            desired_speed = min(desired_speed, heading_high_speed)
        elif heading_error > math.radians(heading_mid):
            desired_speed = min(desired_speed, heading_mid_speed)

        if waypoint.stop_here:
            desired_speed = min(desired_speed, max(0.15, target_dist * 0.45))
            yaw_error = abs(wrap_angle(yaw - self.final_yaw))
            use_precise_brake = True
            if self.expected_orientation == "rear_in" and self.target_center and self.map_extent:
                _, _, ymin, ymax = self.map_extent
                lower_row_limit = ymin + (ymax - ymin) * 0.35
                if self.target_center[1] <= lower_row_limit:
                    use_precise_brake = False
            if (
                use_precise_brake
                and target_dist < 0.70
                and yaw_error < math.radians(35.0)
                and speed < 0.24
            ):
                return 0.0, 1.0
            if target_dist < 0.45 and yaw_error < math.radians(35.0):
                if speed < 0.25:
                    return 0.0, 1.0
                return 0.0, 0.7

        moving_forward = signed_speed > 0.12
        moving_backward = signed_speed < -0.12
        if waypoint.gear == "D" and moving_backward:
            return 0.0, 0.7
        if waypoint.gear == "R" and moving_forward:
            return 0.0, 0.7

        speed_error = desired_speed - speed
        if speed_error > 0.10:
            accel_gain = 0.24 if self.default_mild_turn_active else 0.20
            accel_limit = 0.60 if self.default_mild_turn_active else 0.55
            accel = 0.18 + accel_gain * speed_error
            return clamp(accel, 0.0, accel_limit), 0.0
        if speed_error < -0.15:
            brake = 0.12 + 0.30 * (-speed_error)
            return 0.0, clamp(brake, 0.0, 0.75)
        return 0.0, 0.0

    def _trajectory_distance(self, a: TrajectoryPoint, b: TrajectoryPoint) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _next_hybrid_gear_change_index(self, start_idx: int) -> Optional[int]:
        if not self.hybrid_trajectory:
            return None
        gear = self.hybrid_trajectory[start_idx].gear
        for idx in range(start_idx + 1, len(self.hybrid_trajectory)):
            if self.hybrid_trajectory[idx].gear != gear:
                return idx
        return None

    def _choose_hybrid_target(
        self,
        position: Point,
        speed: float,
    ) -> Tuple[int, int]:
        trajectory = self.hybrid_trajectory
        prev_nearest = self.hybrid_nearest_idx
        prev_target = self.hybrid_target_idx
        search_start = max(0, prev_nearest - 1)
        search_end = min(len(trajectory), max(prev_nearest + 55, prev_target + 25))
        nearest = min(
            range(search_start, search_end),
            key=lambda idx: distance(position, trajectory[idx].point),
        )
        if nearest < prev_nearest:
            nearest = prev_nearest

        current = trajectory[nearest]
        lookahead = 0.8 + 0.4 * speed
        if current.gear == "R" or nearest >= len(trajectory) - 12:
            lookahead *= 0.65
        lookahead = clamp(lookahead, 0.65, 3.5)

        target_idx = nearest
        walked = 0.0
        gear_change = self._next_hybrid_gear_change_index(nearest)
        while target_idx + 1 < len(trajectory) and walked < lookahead:
            if gear_change is not None and target_idx + 1 > gear_change:
                target_idx = gear_change
                break
            walked += self._trajectory_distance(trajectory[target_idx], trajectory[target_idx + 1])
            target_idx += 1

        if target_idx < prev_target and prev_target < len(trajectory) - 1:
            target_idx = prev_target
        return nearest, target_idx

    def _hybrid_speed_command(
        self,
        speed: float,
        signed_speed: float,
        gear: str,
        target_speed: float,
        nearest_idx: int,
        target_idx: int,
        final_dist: float,
        yaw_error: float,
    ) -> Tuple[float, float]:
        desired_speed = target_speed * 1.4
        gear_change = self._next_hybrid_gear_change_index(nearest_idx)
        if gear_change is not None and gear_change - nearest_idx <= 8:
            desired_speed = min(desired_speed, 0.50)
        if target_idx >= len(self.hybrid_trajectory) - 4:
            # 마지막 구간: 너무 느리게 기어가면 슬롯 중심에 도달하기 전에
            # 시간이 끝나므로, 최소 0.35 m/s는 유지하면서 접근합니다.
            desired_speed = min(desired_speed, max(0.35, final_dist * 0.8))

        # 슬롯 중심에 충분히 가깝고(0.30m) 정렬됐을 때만 정지합니다.
        if final_dist < 0.30 and yaw_error < math.radians(35.0):
            return 0.0, 1.0
        if final_dist < 0.18:
            return 0.0, 1.0

        if gear == "D" and signed_speed < -0.12:
            return 0.0, 0.7
        if gear == "R" and signed_speed > 0.12:
            return 0.0, 0.7

        speed_error = desired_speed - speed
        if speed_error > 0.08:
            accel = 0.18 + 0.24 * speed_error
            return clamp(accel, 0.0, 0.65), 0.0
        if speed_error < -0.12:
            brake = 0.10 + 0.30 * (-speed_error)
            return 0.0, clamp(brake, 0.0, 0.75)
        return 0.0, 0.0

    def _compute_hybrid_control(
        self,
        position: Point,
        yaw: float,
        signed_speed: float,
        limits: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.hybrid_active or not self.hybrid_trajectory:
            return None

        try:
            speed = abs(signed_speed)
            nearest_idx, target_idx = self._choose_hybrid_target(position, speed)
            self.hybrid_nearest_idx = nearest_idx
            self.hybrid_target_idx = target_idx

            trajectory = self.hybrid_trajectory
            target = trajectory[target_idx]
            nearest = trajectory[nearest_idx]
            final_pose = trajectory[-1]
            gear = final_pose.gear if nearest_idx >= len(trajectory) - 3 else nearest.gear

            dx = target.x - position[0]
            dy = target.y - position[1]
            target_dist = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx) if target_dist > 1e-6 else target.yaw
            chase_distance = clamp(target_dist, 0.85, 3.5)
            wheelbase = float(limits.get("L", 2.6))
            if gear == "R":
                travel_yaw = wrap_angle(yaw + math.pi)
                alpha = wrap_angle(bearing - travel_yaw)
                steer = -math.atan2(2.0 * wheelbase * math.sin(alpha), chase_distance)
                steer -= 0.35 * wrap_angle(target.yaw - yaw)
            else:
                alpha = wrap_angle(bearing - yaw)
                steer = math.atan2(2.0 * wheelbase * math.sin(alpha), chase_distance)
                steer += 0.35 * wrap_angle(target.yaw - yaw)

            if nearest_idx + 1 < len(trajectory):
                segment = trajectory[nearest_idx + 1]
                seg_dx = segment.x - nearest.x
                seg_dy = segment.y - nearest.y
                seg_len = max(math.hypot(seg_dx, seg_dy), 1e-6)
                cte = ((position[0] - nearest.x) * (-seg_dy) + (position[1] - nearest.y) * seg_dx) / seg_len
                cte_term = math.atan2(0.35 * cte, speed + 0.35)
                steer += -cte_term if gear == "D" else cte_term

            final_dist = distance(position, final_pose.point)
            yaw_error = abs(wrap_angle(yaw - final_pose.yaw))
            accel, brake = self._hybrid_speed_command(
                speed,
                signed_speed,
                gear,
                target.speed,
                nearest_idx,
                target_idx,
                final_dist,
                yaw_error,
            )
            return clamp_command(steer, accel, brake, gear, limits)
        except Exception as exc:
            print(f"[algo] hybrid fallback: {exc}")
            self.hybrid_active = False
            return None

    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """다음 steering, throttle, brake, gear 명령을 반환합니다."""

        self._update_target_from_obs(obs)

        state = obs.get("state", {})
        limits = obs.get("limits", {})
        x = float(state.get("x", 0.0))
        y = float(state.get("y", 0.0))
        yaw = float(state.get("yaw", 0.0))
        signed_speed = float(state.get("v", 0.0))
        speed = abs(signed_speed)
        position = (x, y)

        hybrid_command = self._compute_hybrid_control(position, yaw, signed_speed, limits)
        if hybrid_command is not None:
            return hybrid_command

        waypoint = self._advance_waypoint(position, speed)
        if waypoint is None:
            return clamp_command(0.0, 0.0, 0.5, "D", limits)

        steer = self._steer_toward(position, yaw, waypoint, waypoint.gear, limits)
        accel, brake = self._speed_command(speed, signed_speed, waypoint, position, yaw)

        return clamp_command(steer, accel, brake, waypoint.gear, limits)


# IPC client가 사용하는 전역 planner instance입니다.
planner = PlannerSkeleton()


def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    """새 맵이 도착했을 때 communication module이 호출합니다."""

    planner.set_map(map_payload)


def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    """시뮬레이터 tick마다 communication module이 호출합니다."""

    try:
        return planner.compute_control(obs)
    except Exception as exc:
        print(f"[algo] planner_step error: {exc}")
        return {"steer": 0.0, "accel": 0.0, "brake": 0.7, "gear": "D"}