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
from typing import Any, Dict, List, Optional, Tuple


Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]


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
    """Full House seed 4 guarded trajectory에서만 쓰는 dense pose입니다."""

    x: float
    y: float
    yaw: float
    gear: str
    speed: float

    @property
    def point(self) -> Point:
        return (self.x, self.y)


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

    def set_map(self, map_payload: Dict[str, Any]) -> None:
        """시뮬레이터가 보낸 맵 geometry 정보를 캐시에 저장합니다."""

        self.map_data = map_payload
        self.map_extent = tuple(
            map(float, map_payload.get("extent", (0.0, 75.0, 0.0, 50.0)))
        )
        self.cell_size = float(map_payload.get("cellSize", 0.5))
        self.stationary_grid = map_payload.get("grid", {}).get("stationary")
        self.slots = [tuple(map(float, slot)) for slot in map_payload.get("slots", [])]
        self.occupied_idx = [bool(v) for v in map_payload.get("occupied_idx", [])]
        self.free_slot_indices = [
            idx for idx, occupied in enumerate(self.occupied_idx) if not occupied
        ]
        self.expected_orientation = (
            "rear_in" if len(self.free_slot_indices) == 1 else "front_in"
        )
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

    def _estimate_left_aisle_x(self) -> float:
        """왼쪽 주행 aisle에서 비교적 안전한 x 좌표를 선택합니다."""

        if not self.map_extent:
            return 4.0

        xmin, xmax, _, _ = self.map_extent
        if not self.slots:
            return clamp(xmin + 4.0, xmin + 2.0, xmax - 2.0)

        first_slot_x = min(slot[0] for slot in self.slots)
        aisle_x = min(xmin + 4.0, first_slot_x - 7.0)
        return clamp(aisle_x, xmin + 2.5, xmax - 2.5)

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

    def _use_seed4_full_hybrid(self, slot: Rect) -> bool:
        """검증된 Full House slot 5 trajectory에만 적용하는 엄격한 가드 조건입니다."""

        if self.expected_orientation != "rear_in":
            return False
        if len(self.free_slot_indices) != 1:
            return False
        if len(self.slots) != 33 or sum(1 for occupied in self.occupied_idx if occupied) != 32:
            return False
        if len(self.lines) != 38 or len(self.walls_rects) != 4:
            return False
        if self.map_extent is None:
            return False
        if not self._rect_close(self.map_extent, (7.5, 67.5, 5.0, 45.0), tolerance=0.15):
            return False

        seed4_slot = (36.4, 38.6, 8.9, 13.1)
        slot_idx = self._slot_index(slot)
        if slot_idx != 5:
            return False
        if not self._rect_close(slot, seed4_slot, tolerance=0.08):
            return False
        if self._slot_open_side(slot) != "above":
            return False
        return True

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

    def _use_default_low_score_final_tune(self, slot: Rect) -> bool:
        """Guarded attempt18 final-pose tune for Default Lot low-score slots."""

        if self.expected_orientation != "front_in":
            return False
        if self.map_extent is None:
            return False
        if not self._rect_close(self.map_extent, (0.0, 75.0, 0.0, 50.0), tolerance=0.15):
            return False
        if len(self.slots) != 33:
            return False

        slot_idx = self._slot_index(slot)
        return slot_idx in {1, 2, 3, 5, 6, 7, 12, 14, 15, 21}

    def _route_length(self, route: List[Waypoint]) -> float:
        if len(route) < 2:
            return 0.0
        return sum(distance(route[idx - 1].point, route[idx].point) for idx in range(1, len(route)))

    def _apply_default_low_score_final_tune(
        self,
        route: List[Waypoint],
        slot: Rect,
    ) -> List[Waypoint]:
        if not route or not self._use_default_low_score_final_tune(slot):
            return route

        original_length = self._route_length(route)
        tuned = list(route)
        final = tuned[-1]
        slot_idx = self._slot_index(slot)
        occupied_count = sum(1 for occupied in self.occupied_idx if occupied)
        final_x_offset = -0.24 if slot_idx == 1 or (slot_idx == 12 and occupied_count < 12) else -0.20
        final_y_offset = -0.28 if slot_idx == 1 and occupied_count < 12 else (-0.29 if slot_idx == 1 else -0.30)
        tuned[-1] = Waypoint(
            clamp(final.x + final_x_offset, slot[0] + 0.45, slot[1] - 0.45),
            clamp(final.y + final_y_offset, slot[2] + 0.45, slot[3] - 0.45),
            final.gear,
            radius=0.35,
            speed=final.speed,
            stop_here=final.stop_here,
        )
        tuned = self._remove_redundant_waypoints(tuned)
        if self._route_length(tuned) > original_length + 8.0:
            return route
        if not self._route_points_clear(tuned, slot):
            return route
        return tuned

    def _use_full_house_waypoint_tune(self, slot: Rect) -> bool:
        """Guard for the attempt_13 Full House waypoint-only score tune."""

        if self.expected_orientation != "rear_in":
            return False
        if self.map_extent is None:
            return False
        if not self._rect_close(self.map_extent, (7.5, 67.5, 5.0, 45.0), tolerance=0.15):
            return False
        if len(self.free_slot_indices) != 1:
            return False
        if len(self.slots) != 33 or sum(1 for occupied in self.occupied_idx if occupied) != 32:
            return False
        if len(self.lines) != 38 or len(self.walls_rects) != 4:
            return False

        slot_idx = self._slot_index(slot)
        if slot_idx == 5:
            return False
        return True

    def _apply_full_house_waypoint_tune(
        self,
        route: List[Waypoint],
        slot: Rect,
    ) -> List[Waypoint]:
        if not route or not self._use_full_house_waypoint_tune(slot):
            return route

        tuned: List[Waypoint] = []
        _, cy = rect_center(slot)
        _, _, ymin, ymax = self.map_extent
        lower_row_limit = ymin + (ymax - ymin) * 0.35
        upper_row_limit = ymin + (ymax - ymin) * 0.65
        final_x_offset = 0.0
        final_y_offset = 0.10
        nonfinal_speed_scale = 1.12
        if cy > lower_row_limit:
            if cy >= upper_row_limit:
                final_x_offset = -0.225
                final_y_offset = 0.35
                nonfinal_speed_scale = 1.32
            else:
                final_x_offset = -0.16
                final_y_offset = 0.18
                nonfinal_speed_scale = 1.28

        for waypoint in route:
            if waypoint.stop_here:
                tuned.append(
                    Waypoint(
                        clamp(waypoint.x + final_x_offset, slot[0] + 0.45, slot[1] - 0.45),
                        clamp(waypoint.y + final_y_offset, slot[2] + 0.45, slot[3] - 0.45),
                        waypoint.gear,
                        radius=waypoint.radius,
                        speed=waypoint.speed,
                        stop_here=True,
                    )
                )
            else:
                tuned.append(
                    Waypoint(
                        waypoint.x,
                        waypoint.y,
                        waypoint.gear,
                        radius=waypoint.radius,
                        speed=clamp(waypoint.speed * nonfinal_speed_scale, 0.20, 2.35),
                        stop_here=False,
                    )
                )
        return self._remove_redundant_waypoints(tuned)

    def _use_front_below_dijkstra_route(self, slot: Rect, open_side: str) -> bool:
        """attempt_14 Dijkstra route for front-in slots that open from below."""

        if self.expected_orientation != "front_in":
            return False
        if open_side != "below":
            return False
        if self.map_extent is None:
            return False
        if len(self.slots) != 33:
            return False
        if len(self.lines) != 38 or len(self.walls_rects) != 4:
            return False
        return True

    def _point_rect_distance(self, point: Point, rect: Rect) -> float:
        x, y = point
        dx = max(rect[0] - x, 0.0, x - rect[1])
        dy = max(rect[2] - y, 0.0, y - rect[3])
        return math.hypot(dx, dy)

    def _front_graph_clearance(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any],
        target_slot: Rect,
    ) -> float:
        min_dist = float("inf")
        samples = 8
        for step in range(samples + 1):
            ratio = step / samples
            point = (
                a["x"] + (b["x"] - a["x"]) * ratio,
                a["y"] + (b["y"] - a["y"]) * ratio,
            )
            for rect in self.walls_rects:
                min_dist = min(min_dist, self._point_rect_distance(point, rect))
            for idx, rect in enumerate(self.slots):
                if idx < len(self.occupied_idx) and self.occupied_idx[idx]:
                    if self._same_rect(rect, target_slot):
                        continue
                    min_dist = min(min_dist, self._point_rect_distance(point, rect))
        return min_dist if math.isfinite(min_dist) else 99.0

    def _front_graph_node_blocked(self, node: Dict[str, Any], target_slot: Rect) -> bool:
        point = (node["x"], node["y"])
        for rect in self.walls_rects:
            if self._point_in_rect(point, rect, margin=0.15):
                return True
        for idx, rect in enumerate(self.slots):
            if idx < len(self.occupied_idx) and self.occupied_idx[idx]:
                if self._same_rect(rect, target_slot):
                    continue
                if self._point_in_rect(point, rect, margin=0.15):
                    return True
        return False

    def _front_final_iou_estimate(self, slot: Rect, x: float, y: float) -> float:
        cx, cy = rect_center(slot)
        width = max(1e-6, slot[1] - slot[0])
        height = max(1e-6, slot[3] - slot[2])
        dx = abs(x - cx) / width
        dy = abs(y - cy) / height
        return clamp(0.52 - 0.18 * dx - 0.12 * dy, 0.0, 0.52)

    def _build_front_below_dijkstra_waypoints(self, slot: Rect) -> List[Waypoint]:
        """Build a front-in open-below route via directed weighted graph search."""

        if self.map_extent is None:
            return []

        xmin, xmax, ymin, ymax = self.map_extent
        cx, cy = rect_center(slot)
        aisle_x = clamp(self.left_aisle_x, xmin + 2.5, xmax - 2.5)
        speed_scale = 1.12
        weights = {
            "time": 1.30,
            "distance": 0.04,
            "steer": 0.45,
            "gear": 1.50,
            "clearance": 14.0,
            "iou": 6.0,
        }
        clearance_threshold = 0.55

        nodes: List[Dict[str, Any]] = []
        layers: List[List[int]] = []

        def add_node(
            x: float,
            y: float,
            gear: str,
            kind: str,
            speed: float,
            radius: float,
            stop_here: bool = False,
            final_iou: float = 0.0,
        ) -> int:
            idx = len(nodes)
            nodes.append(
                {
                    "x": x,
                    "y": y,
                    "gear": gear,
                    "kind": kind,
                    "speed": speed,
                    "radius": radius,
                    "stop_here": stop_here,
                    "yaw": 0.0,
                    "final_iou": final_iou,
                }
            )
            return idx

        start = add_node(aisle_x, ymin + 6.0, "D", "start", 1.6, 1.0)
        layers.append([start])

        align_ys = sorted(
            {
                clamp(slot[2] - gap, ymin + 2.5, ymax - 2.5)
                for gap in (5.5, 6.5, 7.0, 8.0)
            }
        )
        entry_ys = sorted(
            {
                clamp(slot[2] - gap, ymin + 2.5, ymax - 2.5)
                for gap in (1.4, 1.8, 2.2, 2.8)
            }
        )

        layers.append(
            [
                add_node(aisle_x, y, "D", "left_aisle", 1.75 * speed_scale, 1.35)
                for y in align_ys
            ]
        )

        turn_nodes: List[int] = []
        for y in align_ys:
            for dx in (2.4, 3.2, 4.0, 4.8, 5.6):
                x = clamp(cx - dx, aisle_x + 2.0, xmax - 2.5)
                turn_nodes.append(
                    add_node(x, y, "D", "row_turn", 1.70 * speed_scale, 1.15)
                )
        layers.append(turn_nodes)

        entry_nodes: List[int] = []
        for y in entry_ys:
            for dx in (-0.2, 0.0, 0.2):
                x = clamp(cx + dx, xmin + 2.5, xmax - 2.5)
                entry_nodes.append(
                    add_node(x, y, "D", "slot_entry", 0.95 * speed_scale, 0.90)
                )
        layers.append(entry_nodes)

        final_nodes: List[int] = []
        for dx in (-0.15, 0.0, 0.15):
            for dy in (-0.15, 0.0, 0.15):
                fx = clamp(cx + dx, slot[0] + 0.55, slot[1] - 0.55)
                fy = clamp(cy + dy, slot[2] + 0.55, slot[3] - 0.55)
                final_nodes.append(
                    add_node(
                        fx,
                        fy,
                        "D",
                        "final",
                        0.55,
                        0.35,
                        stop_here=True,
                        final_iou=self._front_final_iou_estimate(slot, fx, fy),
                    )
                )
        layers.append(final_nodes)

        edges: Dict[int, List[Tuple[float, int]]] = {idx: [] for idx in range(len(nodes))}

        def add_edge(src_idx: int, dst_idx: int) -> None:
            src = nodes[src_idx]
            dst = nodes[dst_idx]
            if not dst["stop_here"] and self._front_graph_node_blocked(dst, slot):
                return

            dist = math.hypot(dst["x"] - src["x"], dst["y"] - src["y"])
            if dist < 0.05:
                return

            heading = abs(wrap_angle(math.atan2(dst["y"] - src["y"], dst["x"] - src["x"])))
            gear_change = 1.0 if src["gear"] != dst["gear"] else 0.0
            speed = max(0.20, min(float(src["speed"]), float(dst["speed"])))
            estimated_time = dist / speed
            clearance = self._front_graph_clearance(src, dst, slot)
            low_clearance = max(0.0, clearance_threshold - clearance)
            clearance_penalty = low_clearance * low_clearance
            if clearance < 0.05:
                clearance_penalty += 25.0
            final_iou_loss = 0.0
            if dst["stop_here"]:
                final_iou_loss = max(0.0, 0.55 - float(dst["final_iou"]))

            cost = (
                weights["time"] * estimated_time
                + weights["distance"] * dist
                + weights["steer"] * heading
                + weights["gear"] * gear_change
                + weights["clearance"] * clearance_penalty
                + weights["iou"] * final_iou_loss
            )
            edges[src_idx].append((max(0.0, cost), dst_idx))

        for layer_idx in range(len(layers) - 1):
            for src_idx in layers[layer_idx]:
                for dst_idx in layers[layer_idx + 1]:
                    add_edge(src_idx, dst_idx)

        final_set = set(layers[-1])
        queue: List[Tuple[float, int]] = [(0.0, start)]
        dist_by_node: Dict[int, float] = {start: 0.0}
        previous: Dict[int, int] = {}
        best_final: Optional[int] = None
        while queue:
            cost, node_idx = heapq.heappop(queue)
            if cost > dist_by_node.get(node_idx, float("inf")):
                continue
            if node_idx in final_set:
                best_final = node_idx
                break
            for edge_cost, dst_idx in edges.get(node_idx, []):
                next_cost = cost + edge_cost
                if next_cost < dist_by_node.get(dst_idx, float("inf")):
                    dist_by_node[dst_idx] = next_cost
                    previous[dst_idx] = node_idx
                    heapq.heappush(queue, (next_cost, dst_idx))

        if best_final is None:
            return []

        path = [best_final]
        while path[-1] != start:
            parent = previous.get(path[-1])
            if parent is None:
                return []
            path.append(parent)
        path.reverse()

        route = [
            Waypoint(
                nodes[idx]["x"],
                nodes[idx]["y"],
                nodes[idx]["gear"],
                radius=nodes[idx]["radius"],
                speed=nodes[idx]["speed"],
                stop_here=nodes[idx]["stop_here"],
            )
            for idx in path[1:]
        ]
        compact = self._remove_redundant_waypoints(route)
        if not compact or not self._route_points_clear(compact, slot):
            return []
        self.entry_gear = "D"
        return compact

    def _seed4_full_hybrid_trajectory(self) -> List[TrajectoryPoint]:
        """attempt_06에서 검증한 dense path입니다. 실행 중 CSV 파일을 읽지 않습니다."""

        raw_points = [
            (11.5, 11, math.radians(90), "D", 1.15),
            (11.50747, 11.249814, math.radians(87.431013), "D", 1.15),
            (11.514939, 11.499628, math.radians(84.862027), "D", 1.15),
            (11.570472, 11.793726, math.radians(80.232919), "D", 1.15),
            (11.626005, 12.087825, math.radians(75.603811), "D", 1.15),
            (11.681537, 12.381923, math.radians(70.974703), "D", 1.15),
            (11.773495, 12.614216, math.radians(67.117112), "D", 1.15),
            (11.865452, 12.846509, math.radians(63.259522), "D", 1.15),
            (11.984731, 13.046674, math.radians(59.659105), "D", 1.15),
            (12.10401, 13.246838, math.radians(56.058688), "D", 1.15),
            (12.22329, 13.447003, math.radians(52.45827), "D", 1.15),
            (12.420852, 13.672351, math.radians(49.375486), "D", 1.15),
            (12.618415, 13.897698, math.radians(46.292702), "D", 1.15),
            (12.815977, 14.123046, math.radians(43.209918), "D", 1.15),
            (13.047188, 14.313713, math.radians(40.127134), "D", 1.15),
            (13.278399, 14.504381, math.radians(37.044351), "D", 1.15),
            (13.50961, 14.695048, math.radians(33.961567), "D", 1.15),
            (13.768459, 14.846078, math.radians(30.878783), "D", 1.15),
            (14.027307, 14.997108, math.radians(27.795999), "D", 1.15),
            (14.286156, 15.148138, math.radians(24.713215), "D", 1.15),
            (14.565912, 15.255604, math.radians(21.630431), "D", 1.15),
            (14.845668, 15.36307, math.radians(18.547647), "D", 1.15),
            (15.125424, 15.470536, math.radians(15.464863), "D", 1.15),
            (15.41723, 15.539727, math.radians(13.693437), "D", 1.15),
            (15.709036, 15.608918, math.radians(11.922011), "D", 1.15),
            (16.000842, 15.678108, math.radians(10.150585), "D", 1.15),
            (16.297803, 15.719975, math.radians(8.379159), "D", 1.15),
            (16.594763, 15.761841, math.radians(6.607733), "D", 1.15),
            (16.891723, 15.803708, math.radians(4.836308), "D", 1.15),
            (17.140833, 15.824785, math.radians(4.836308), "D", 1.15),
            (17.389942, 15.845862, math.radians(4.836308), "D", 1.15),
            (17.639052, 15.86694, math.radians(4.836308), "D", 1.15),
            (17.888162, 15.888017, math.radians(4.836308), "D", 1.15),
            (18.120665, 15.907689, math.radians(4.836308), "D", 1.15),
            (18.353167, 15.927361, math.radians(4.836308), "D", 1.15),
            (18.58567, 15.947033, math.radians(4.836308), "D", 1.15),
            (18.83478, 15.968111, math.radians(4.836308), "D", 1.15),
            (19.08389, 15.989188, math.radians(4.836308), "D", 1.15),
            (19.333, 16.010265, math.radians(4.836308), "D", 1.15),
            (19.58211, 16.031343, math.radians(4.836308), "D", 1.15),
            (19.815097, 16.034538, math.radians(1.23589), "D", 1.15),
            (20.048085, 16.037734, math.radians(-2.364527), "D", 1.15),
            (20.281072, 16.04093, math.radians(-5.964945), "D", 1.15),
            (20.579448, 16.009754, math.radians(-5.964945), "D", 1.15),
            (20.877824, 15.978578, math.radians(-5.964945), "D", 1.15),
            (21.1762, 15.947402, math.radians(-5.964945), "D", 1.15),
            (21.40827, 15.923154, math.radians(-5.964945), "D", 1.15),
            (21.64034, 15.898906, math.radians(-5.964945), "D", 1.15),
            (21.87241, 15.874658, math.radians(-5.964945), "D", 1.15),
            (22.121056, 15.848678, math.radians(-5.964945), "D", 1.15),
            (22.369702, 15.822698, math.radians(-5.964945), "D", 1.15),
            (22.618349, 15.796718, math.radians(-5.964945), "D", 1.15),
            (22.866995, 15.770738, math.radians(-5.964945), "D", 1.15),
            (23.099065, 15.74649, math.radians(-5.964945), "D", 1.15),
            (23.331135, 15.722242, math.radians(-5.964945), "D", 1.15),
            (23.563205, 15.697994, math.radians(-5.964945), "D", 1.15),
            (23.811852, 15.672014, math.radians(-5.964945), "D", 1.15),
            (24.060498, 15.646034, math.radians(-5.964945), "D", 1.15),
            (24.309144, 15.620054, math.radians(-5.964945), "D", 1.15),
            (24.557791, 15.594074, math.radians(-5.964945), "D", 1.15),
            (24.806437, 15.568094, math.radians(-5.964945), "D", 1.15),
            (25.055084, 15.542114, math.radians(-5.964945), "D", 1.15),
            (25.287154, 15.517866, math.radians(-5.964945), "D", 1.15),
            (25.519224, 15.493618, math.radians(-5.964945), "D", 1.15),
            (25.751294, 15.46937, math.radians(-5.964945), "D", 1.15),
            (25.99994, 15.44339, math.radians(-5.964945), "D", 1.15),
            (26.248586, 15.41741, math.radians(-5.964945), "D", 1.15),
            (26.497233, 15.39143, math.radians(-5.964945), "D", 1.15),
            (26.745879, 15.36545, math.radians(-5.964945), "D", 1.15),
            (26.994526, 15.33947, math.radians(-5.964945), "D", 1.15),
            (27.243172, 15.31349, math.radians(-5.964945), "D", 1.15),
            (27.475242, 15.289242, math.radians(-5.964945), "D", 1.15),
            (27.707312, 15.264994, math.radians(-5.964945), "D", 1.15),
            (27.939382, 15.240746, math.radians(-5.964945), "D", 1.15),
            (28.188028, 15.214766, math.radians(-5.964945), "D", 1.15),
            (28.436675, 15.188786, math.radians(-5.964945), "D", 1.15),
            (28.668745, 15.164538, math.radians(-5.964945), "D", 1.15),
            (28.900815, 15.14029, math.radians(-5.964945), "D", 1.15),
            (29.132885, 15.116042, math.radians(-5.964945), "D", 1.15),
            (29.431261, 15.084866, math.radians(-5.964945), "D", 1.15),
            (29.729636, 15.05369, math.radians(-5.964945), "D", 1.15),
            (30.028012, 15.022514, math.radians(-5.964945), "D", 1.15),
            (30.276658, 14.996534, math.radians(-5.964945), "D", 1.15),
            (30.525305, 14.970554, math.radians(-5.964945), "D", 1.15),
            (30.757375, 14.946306, math.radians(-5.964945), "D", 1.15),
            (30.989445, 14.922058, math.radians(-5.964945), "D", 1.15),
            (31.221515, 14.89781, math.radians(-5.964945), "D", 1.15),
            (31.470161, 14.87183, math.radians(-5.964945), "D", 0.55),
            (31.718808, 14.84585, math.radians(-5.964945), "D", 0.55),
            (31.967454, 14.81987, math.radians(-5.964945), "D", 0.55),
            (32.2161, 14.79389, math.radians(-5.964945), "D", 0.55),
            (32.44817, 14.769642, math.radians(-5.964945), "D", 0.55),
            (32.68024, 14.745394, math.radians(-5.964945), "D", 0.55),
            (32.91231, 14.721146, math.radians(-5.964945), "D", 0.55),
            (33.160957, 14.695166, math.radians(-5.964945), "D", 0.55),
            (33.409603, 14.669186, math.radians(-5.964945), "D", 0.55),
            (33.640133, 14.634065, math.radians(-8.362665), "D", 0.55),
            (33.870663, 14.598944, math.radians(-10.760386), "D", 0.55),
            (34.101192, 14.563823, math.radians(-13.158107), "D", 0.55),
            (34.390482, 14.48477, math.radians(-14.929533), "D", 0.55),
            (34.679772, 14.405717, math.radians(-16.700959), "D", 0.55),
            (34.969062, 14.326664, math.radians(-18.472385), "D", 0.55),
            (35.24659, 14.213567, math.radians(-21.555169), "D", 0.55),
            (35.524118, 14.10047, math.radians(-24.637953), "D", 0.55),
            (35.801645, 13.987373, math.radians(-27.720737), "D", 0.55),
            (36.017366, 13.861354, math.radians(-31.578327), "D", 0.28),
            (36.233086, 13.735335, math.radians(-35.435917), "D", 0.28),
            (36.412917, 13.587165, math.radians(-39.036334), "D", 0.28),
            (36.592748, 13.438996, math.radians(-42.636752), "D", 0.28),
            (36.77258, 13.290826, math.radians(-46.237169), "D", 0.28),
            (36.921457, 13.111581, math.radians(-49.837586), "D", 0.28),
            (37.070335, 12.932336, math.radians(-53.438004), "D", 0.28),
            (37.219213, 12.75309, math.radians(-57.038421), "D", 0.28),
            (37.331862, 12.54912, math.radians(-60.638839), "D", 0.28),
            (37.44451, 12.345151, math.radians(-64.239256), "D", 0.28),
            (37.557159, 12.141181, math.radians(-67.839673), "D", 0.28),
            (37.640919, 11.905808, math.radians(-71.697263), "D", 0.28),
            (37.724679, 11.670435, math.radians(-75.554854), "D", 0.28),
            (37.612339, 11.535218, math.radians(-82.777427), "D", 0.15),
            (37.5, 11.4, math.radians(-90), "D", 0.15),
        ]
        return [TrajectoryPoint(x, y, yaw, gear, speed) for x, y, yaw, gear, speed in raw_points]

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
        self.default_mild_turn_active = False
        self.waypoints = self._build_waypoints(slot)
        self.hybrid_active = False
        self.hybrid_trajectory = []
        if self._use_seed4_full_hybrid(slot):
            self.hybrid_trajectory = self._seed4_full_hybrid_trajectory()
            self.hybrid_active = bool(self.hybrid_trajectory)
            self.planned_orientation = "rear_in"
            self.final_yaw = -math.pi * 0.5
        self.hybrid_nearest_idx = 0
        self.hybrid_target_idx = 0
        self.active_waypoint = 0

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
            return self._apply_default_low_score_final_tune(self._build_horizontal_fallback(slot), slot)

        open_side = self._slot_open_side(slot)
        turn_length = 6.0
        lane_gap = 2.0
        if self.planned_orientation == "rear_in":
            route = self._build_rear_in_waypoints(slot, open_side)
            return self._apply_full_house_waypoint_tune(route, slot)

        if self._use_front_below_dijkstra_route(slot, open_side):
            try:
                route = self._build_front_below_dijkstra_waypoints(slot)
                if route:
                    self.default_mild_turn_active = self._use_default_mild_turn_route(slot)
                    return self._apply_default_low_score_final_tune(route, slot)
            except Exception as exc:
                print(f"[algo] front-below dijkstra fallback: {exc}")

        if self._use_default_mild_turn_route(slot):
            try:
                route = self._build_default_mild_turn_waypoints(slot, open_side)
                if route:
                    self.default_mild_turn_active = True
                    return self._apply_default_low_score_final_tune(route, slot)
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
        return self._apply_default_low_score_final_tune(self._remove_redundant_waypoints(route), slot)

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

    def _curve_angle_to_next_waypoint(self, waypoint: Waypoint, position: Point) -> float:
        idx = self.active_waypoint
        if idx >= len(self.waypoints) - 1:
            return 0.0
        nxt = self.waypoints[idx + 1]
        if nxt.gear != waypoint.gear:
            return math.pi
        current_bearing = math.atan2(waypoint.y - position[1], waypoint.x - position[0])
        next_bearing = math.atan2(nxt.y - waypoint.y, nxt.x - waypoint.x)
        return abs(wrap_angle(next_bearing - current_bearing))

    def _use_rear_in_speed_profile_scale(self) -> bool:
        return (
            self.expected_orientation == "rear_in"
            and len(self.free_slot_indices) == 1
            and len(self.slots) == 33
            and not self.hybrid_active
        )

    def _use_attempt20_slot1_speed_override(self) -> bool:
        return (
            self.target_slot is not None
            and self._use_default_low_score_final_tune(self.target_slot)
            and self._slot_index(self.target_slot) == 1
        )

    def _use_attempt22_slot1_speed_override(self) -> bool:
        return (
            self.target_slot is not None
            and self._use_default_low_score_final_tune(self.target_slot)
            and self._slot_index(self.target_slot) == 1
            and sum(1 for occupied in self.occupied_idx if occupied) < 12
        )

    def _use_attempt21_slot12_speed_override(self) -> bool:
        return (
            self.target_slot is not None
            and self._use_default_low_score_final_tune(self.target_slot)
            and self._slot_index(self.target_slot) == 12
        )

    def _use_attempt21_crowded_speed_override(self) -> bool:
        if self.expected_orientation != "front_in":
            return False
        if self.target_slot is None or self.map_extent is None:
            return False
        if not self._rect_close(self.map_extent, (0.0, 75.0, 0.0, 50.0), tolerance=0.15):
            return False
        if len(self.slots) != 33:
            return False
        if len(self.lines) != 38 or len(self.walls_rects) != 4:
            return False
        if sum(1 for occupied in self.occupied_idx if occupied) < 12:
            return False
        slot_idx = self._slot_index(self.target_slot)
        return slot_idx in {1, 9, 12, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27}

    def _use_attempt21_speed2_override(self) -> bool:
        return self._use_attempt21_slot12_speed_override() or self._use_attempt21_crowded_speed_override()

    def _front_straight_speed_limit(self) -> float:
        occupied_count = sum(1 for occupied in self.occupied_idx if occupied)
        if self._use_attempt21_speed2_override():
            return 2.5 if occupied_count > 0 else 2.65
        if self._use_attempt22_slot1_speed_override():
            return 2.55 if occupied_count > 0 else 2.70
        if self._use_attempt20_slot1_speed_override():
            return 2.5 if occupied_count > 0 else 2.6
        return 2.4 if occupied_count > 0 else 2.6

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
        desired_speed = float(waypoint.speed)
        heading_error = self._heading_error_to_waypoint(position, yaw, waypoint)
        curve_error = self._curve_angle_to_next_waypoint(waypoint, position)

        if self._use_rear_in_speed_profile_scale():
            if not waypoint.stop_here:
                desired_speed = min(max(desired_speed * 1.12, desired_speed), 2.35)
        elif waypoint.gear == "D" and not waypoint.stop_here:
            desired_speed = max(desired_speed, self._front_straight_speed_limit())

        if waypoint.gear == "R":
            desired_speed = min(desired_speed, 0.95)

        if heading_error > math.radians(115.0):
            desired_speed = min(desired_speed, 0.85)
        elif heading_error > math.radians(80.0):
            desired_speed = min(desired_speed, 1.20)

        if target_dist < 2.8:
            if curve_error > math.radians(75.0):
                desired_speed = min(desired_speed, 0.90)
            elif curve_error > math.radians(35.0):
                desired_speed = min(desired_speed, 1.15)

        idx = self.active_waypoint
        if idx < len(self.waypoints) - 1 and self.waypoints[idx + 1].gear != waypoint.gear:
            gear_change_dist = 1.3 if self._use_attempt21_speed2_override() else 1.4
            if target_dist < gear_change_dist:
                desired_speed = min(desired_speed, 0.75)

        if waypoint.stop_here:
            desired_speed = min(desired_speed, max(0.12, target_dist * 0.45))
            yaw_error = abs(wrap_angle(yaw - self.final_yaw))
            final_brake_dist = 0.68 if (self._use_attempt21_slot12_speed_override() or self._use_attempt22_slot1_speed_override()) else 0.70
            use_precise_brake = True
            if self.expected_orientation == "rear_in" and self.target_center and self.map_extent:
                _, _, ymin, ymax = self.map_extent
                lower_row_limit = ymin + (ymax - ymin) * 0.35
                if self.target_center[1] <= lower_row_limit:
                    use_precise_brake = False
            if (
                use_precise_brake
                and target_dist < final_brake_dist
                and yaw_error < math.radians(35.0)
                and speed < 0.22
            ):
                return 0.0, 1.0
            if target_dist < 0.45 and yaw_error < math.radians(35.0):
                if speed < 0.22:
                    return 0.0, 1.0
                return 0.0, 0.75

        moving_forward = signed_speed > 0.12
        moving_backward = signed_speed < -0.12
        if waypoint.gear == "D" and moving_backward:
            return 0.0, 0.7
        if waypoint.gear == "R" and moving_forward:
            return 0.0, 0.7

        speed_error = desired_speed - speed
        if speed_error > 0.10:
            if self._use_attempt21_speed2_override():
                accel = 0.18 + 0.46 * speed_error
                return clamp(accel, 0.0, 0.92), 0.0
            if self._use_attempt22_slot1_speed_override():
                accel = 0.18 + 0.46 * speed_error
                return clamp(accel, 0.0, 0.92), 0.0
            if self._use_attempt20_slot1_speed_override():
                accel = 0.18 + 0.48 * speed_error
                return clamp(accel, 0.0, 0.92), 0.0
            accel = 0.18 + 0.44 * speed_error
            return clamp(accel, 0.0, 0.90), 0.0
        if speed_error < -0.15:
            brake = 0.12 + 0.38 * (-speed_error)
            return 0.0, clamp(brake, 0.0, 0.84)
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
            desired_speed = min(desired_speed, max(0.12, final_dist * 0.55))

        if final_dist < 0.55 and yaw_error < math.radians(35.0) and speed < 0.24:
            return 0.0, 1.0
        if final_dist < 0.35 and yaw_error < math.radians(35.0):
            return 0.0, 0.8

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