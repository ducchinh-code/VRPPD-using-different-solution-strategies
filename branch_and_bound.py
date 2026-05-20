from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional
from models import Node, Request, Vehicle, Route, Solution


# ---------------------------------------------------------------------------
# Cấu trúc trạng thái một lộ trình đang xây dựng (một xe)
# ---------------------------------------------------------------------------
@dataclass
class BBState:
    route: list[int]          # Thứ tự node id đã chọn (không gồm depot đầu/cuối)
    load: float               # Tải trọng hiện tại
    cost: float               # Chi phí đi được đến node cuối cùng
    served: frozenset[int]    # Tập node id đã phục vụ


def _dist(a: Node, b: Node) -> float:
    return a.distance_to(b)


def _lower_bound(state: BBState, unserved: list[Node], depot: Node,
                 node_map: dict[int, Node]) -> float:
    """
    Cận dưới đơn giản: chi phí hiện tại + tổng khoảng cách nhỏ nhất
    từ mỗi node chưa phục vụ tới node gần nhất (relaxation).
    Giữ nhẹ để không làm chậm vòng lặp.
    """
    if not unserved:
        # Chỉ cần quay về depot
        last = node_map[state.route[-1]] if state.route else depot
        return state.cost + _dist(last, depot)

    last = node_map[state.route[-1]] if state.route else depot
    # Ước lượng: từ node hiện tại → node chưa phục vụ gần nhất → depot
    min_next = min(_dist(last, u) for u in unserved)
    min_to_depot = min(_dist(u, depot) for u in unserved)
    return state.cost + min_next + min_to_depot


# ---------------------------------------------------------------------------
# Giải một cụm nhỏ bằng B&B thực sự (stack-based DFS + pruning)
# ---------------------------------------------------------------------------
def _solve_chunk(
    depot: Node,
    requests: list[Request],
    vehicle: Vehicle,
    time_limit: float,
    start_time: float,
) -> list[Node]:
    """
    Trả về danh sách Node (không gồm depot) theo thứ tự tối ưu cho 1 xe.
    Dùng DFS + lower bound pruning + ràng buộc PD.
    """
    capacity = vehicle.capacity

    # Tập pickup và delivery
    pickup_ids  = {req.pickup.id  for req in requests}
    delivery_ids = {req.delivery.id for req in requests}
    # Mapping: delivery_id -> pickup_id (ràng buộc thứ tự)
    pickup_of: dict[int, int] = {req.delivery.id: req.pickup.id for req in requests}
    # Mapping: node_id -> demand
    demand_of: dict[int, float] = {}
    node_map:  dict[int, Node]  = {}
    for req in requests:
        demand_of[req.pickup.id]   = req.pickup.demand
        demand_of[req.delivery.id] = req.delivery.demand
        node_map[req.pickup.id]    = req.pickup
        node_map[req.delivery.id]  = req.delivery

    all_ids = list(node_map.keys())

    best_cost: list[float] = [float("inf")]
    best_route: list[list[int]] = [[]]

    # Stack: mỗi phần tử là BBState
    stack: list[BBState] = [BBState(route=[], load=0.0, cost=0.0, served=frozenset())]

    while stack:
        if time.perf_counter() - start_time > time_limit:
            break

        state = stack.pop()
        unserved_ids = [nid for nid in all_ids if nid not in state.served]

        # --- Cận dưới: cắt nhánh sớm ---
        unserved_nodes = [node_map[nid] for nid in unserved_ids]
        lb = _lower_bound(state, unserved_nodes, depot, node_map)
        if lb >= best_cost[0]:
            continue

        # --- Lộ trình hoàn chỉnh ---
        if not unserved_ids:
            last_node = node_map[state.route[-1]] if state.route else depot
            total = state.cost + _dist(last_node, depot)
            if total < best_cost[0]:
                best_cost[0] = total
                best_route[0] = list(state.route)
            continue

        # --- Phân nhánh: thử thêm từng node hợp lệ ---
        last_node = node_map[state.route[-1]] if state.route else depot

        for nid in unserved_ids:
            # Ràng buộc 1: delivery phải sau pickup tương ứng
            if nid in delivery_ids:
                pid = pickup_of[nid]
                if pid not in state.served:
                    continue  # Pickup chưa được phục vụ → bỏ qua

            # Ràng buộc 2: sức chứa
            new_load = state.load + demand_of[nid]
            if new_load > capacity:
                continue

            # Tính chi phí bước đi
            next_node = node_map[nid]
            step_cost = _dist(last_node, next_node)

            new_state = BBState(
                route   = state.route + [nid],
                load    = new_load,
                cost    = state.cost + step_cost,
                served  = state.served | {nid},
            )
            stack.append(new_state)

    return [node_map[nid] for nid in best_route[0]]


# ---------------------------------------------------------------------------
# Hàm công khai — được gọi từ main.py
# ---------------------------------------------------------------------------
def run_branch_and_bound_solver(
    nodes: list[Node],
    requests: list[Request],
    vehicles: list[Vehicle],
    capacity: int,
    time_limit_seconds: int = 30,
    **kwargs,          # bỏ qua tham số cũ như max_pairs
) -> Solution:
    """
    Branch & Bound thực sự (DFS + pruning).
    Mỗi xe được phân công một tập requests theo chiến lược greedy nearest-pickup,
    sau đó từng xe được tối ưu bằng B&B độc lập.
    """
    depot = nodes[0]
    start_time = time.perf_counter()

    # -----------------------------------------------------------------------
    # Phân công requests cho từng xe (nearest-depot heuristic)
    # Mỗi xe nhận yêu cầu cho đến khi đầy hoặc hết requests
    # -----------------------------------------------------------------------
    unassigned = list(requests)
    vehicle_requests: dict[int, list[Request]] = {v.id: [] for v in vehicles}
    vehicle_load: dict[int, float] = {v.id: 0.0 for v in vehicles}

    for req in unassigned:
        pickup_demand = req.pickup.demand
        # Tìm xe còn chỗ và chưa quá tải
        assigned = False
        for v in vehicles:
            if vehicle_load[v.id] + pickup_demand <= capacity:
                vehicle_requests[v.id].append(req)
                vehicle_load[v.id] += pickup_demand
                assigned = True
                break
        if not assigned:
            # Không có xe nào trống: đưa vào xe đầu tiên (vi phạm sẽ được báo)
            vehicle_requests[vehicles[0].id].append(req)

    # -----------------------------------------------------------------------
    # Giải từng xe bằng B&B
    # -----------------------------------------------------------------------
    solution_routes: list[Route] = []
    n_vehicles = len(vehicles)

    for idx, v in enumerate(vehicles):
        reqs = vehicle_requests[v.id]
        if not reqs:
            continue

        elapsed = time.perf_counter() - start_time
        remaining_time = time_limit_seconds - elapsed
        if remaining_time <= 0:
            # Hết thời gian: dùng thứ tự hiện tại (pickup→delivery xen kẽ)
            route_nodes = []
            for r in reqs:
                route_nodes.extend([r.pickup, r.delivery])
        else:
            # Chia đều thời gian còn lại cho các xe còn lại
            per_vehicle_time = remaining_time / max(1, n_vehicles - idx)
            route_nodes = _solve_chunk(
                depot=depot,
                requests=reqs,
                vehicle=v,
                time_limit=per_vehicle_time,
                start_time=time.perf_counter(),
            )

        if route_nodes:
            route = Route(vehicle=v, depot=depot)
            for n in route_nodes:
                route.append(n)
            solution_routes.append(route)

    return Solution(routes=solution_routes)
