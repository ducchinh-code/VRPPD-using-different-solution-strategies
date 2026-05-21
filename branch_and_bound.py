from __future__ import annotations
import time
from dataclasses import dataclass
from models import Node, Request, Vehicle, Route, Solution, cdist

EPS = 1e-9

@dataclass(frozen=True)
class BBState:
    routes: tuple[tuple[int, ...], ...]
    loads: tuple[float, ...]
    costs: tuple[float, ...]
    served: frozenset[int]
    active_pickups: tuple[frozenset[int], ...]

def _dist(a: Node, b: Node) -> float:
    return cdist(a, b)

def _route_finish_cost(route_ids: tuple[int, ...], depot: Node,
                       node_map: dict[int, Node]) -> float:
    if not route_ids:
        return 0.0
    return _dist(node_map[route_ids[-1]], depot)

def _state_complete_cost(state: BBState, depot: Node,
                         node_map: dict[int, Node]) -> float:
    return sum(state.costs) + sum(
        _route_finish_cost(route, depot, node_map)
        for route in state.routes
    )

def _lower_bound(state: BBState, unserved_ids: list[int], depot: Node,
                 node_map: dict[int, Node]) -> float:
    current = sum(state.costs)
    if not unserved_ids:
        return _state_complete_cost(state, depot, node_map)
    anchors = [
        node_map[route[-1]] if route else depot
        for route in state.routes
    ]
    unserved_nodes = [node_map[nid] for nid in unserved_ids]

    min_entry = min(
        _dist(anchor, node)
        for anchor in anchors
        for node in unserved_nodes
    )
    min_return = min(_dist(node, depot) for node in unserved_nodes)
    return current + min_entry + min_return

def _build_node_maps(requests: list[Request]):
    node_map: dict[int, Node] = {}
    demand_of: dict[int, float] = {}
    pickup_of: dict[int, int] = {}
    delivery_ids: set[int] = set()
    pickup_ids: set[int] = set()
    all_ids: list[int] = []

    for req in requests:
        pickup = req.pickup
        delivery = req.delivery

        node_map[pickup.id] = pickup
        node_map[delivery.id] = delivery
        demand_of[pickup.id] = pickup.demand
        demand_of[delivery.id] = delivery.demand
        pickup_of[delivery.id] = pickup.id
        pickup_ids.add(pickup.id)
        delivery_ids.add(delivery.id)
        all_ids.extend([pickup.id, delivery.id])

    return node_map, demand_of, pickup_of, pickup_ids, delivery_ids, tuple(all_ids)

def _initial_incumbent(depot: Node, requests: list[Request],
                       vehicles: list[Vehicle]) -> Solution:

    routes = [Route(vehicle, depot) for vehicle in vehicles]

    for req in requests:
        feasible_routes = []
        for idx, route in enumerate(routes):
            if req.pickup.demand > route.vehicle.capacity + EPS:
                continue

            last = route.nodes[-1] if route.nodes else depot
            old_tail = _dist(last, depot) if route.nodes else 0.0
            delta = (
                -old_tail
                + _dist(last, req.pickup)
                + _dist(req.pickup, req.delivery)
                + _dist(req.delivery, depot)
            )
            feasible_routes.append((delta, idx))

        if not feasible_routes:
            return Solution()

        _, best_idx = min(feasible_routes, key=lambda item: item[0])
        routes[best_idx].append(req.pickup)
        routes[best_idx].append(req.delivery)

    solution = Solution(routes=[route for route in routes if route.nodes])
    if not solution.is_feasible():
        return Solution()
    if any(abs(sum(node.demand for node in route.nodes)) > EPS
           for route in solution.routes):
        return Solution()
    return solution

def _routes_to_state_routes(solution: Solution,
                            vehicles: list[Vehicle],
                            vehicle_count: int) -> tuple[tuple[int, ...], ...]:
    by_pos: list[tuple[int, ...]] = [tuple() for _ in range(vehicle_count)]
    position_by_vehicle = {
        id(vehicle): idx
        for idx, vehicle in enumerate(vehicles)
    }

    for route in solution.routes:
        idx = position_by_vehicle.get(id(route.vehicle))
        if idx is not None:
            by_pos[idx] = tuple(node.id for node in route.nodes)
    return tuple(by_pos)

def _make_solution(routes_ids: tuple[tuple[int, ...], ...],
                   vehicles: list[Vehicle],
                   depot: Node,
                   node_map: dict[int, Node]) -> Solution:
    routes: list[Route] = []
    for idx, route_ids in enumerate(routes_ids):
        if not route_ids:
            continue
        route = Route(vehicles[idx], depot)
        for nid in route_ids:
            route.append(node_map[nid])
        routes.append(route)
    return Solution(routes)

def _can_start_empty_vehicle(state: BBState, vehicle_idx: int,
                             vehicles: list[Vehicle]) -> bool:
    if state.routes[vehicle_idx]:
        return True

    capacity = vehicles[vehicle_idx].capacity
    for prev_idx in range(vehicle_idx):
        if not state.routes[prev_idx] and vehicles[prev_idx].capacity == capacity:
            return False
    return True


def _branch_next_states(state: BBState,
                        unserved_ids: list[int],
                        depot: Node,
                        vehicles: list[Vehicle],
                        node_map: dict[int, Node],
                        demand_of: dict[int, float],
                        pickup_of: dict[int, int],
                        delivery_ids: set[int]) -> list[tuple[float, BBState]]:
    children: list[tuple[float, BBState]] = []

    for vehicle_idx, vehicle in enumerate(vehicles):
        if not _can_start_empty_vehicle(state, vehicle_idx, vehicles):
            continue

        route = state.routes[vehicle_idx]
        last_node = node_map[route[-1]] if route else depot

        for nid in unserved_ids:
            node = node_map[nid]
            new_load = state.loads[vehicle_idx] + demand_of[nid]
            if new_load < -EPS or new_load > vehicle.capacity + EPS:
                continue

            active_for_vehicle = state.active_pickups[vehicle_idx]
            if nid in delivery_ids:
                pickup_id = pickup_of[nid]
                if pickup_id not in active_for_vehicle:
                    continue
                next_active = frozenset(
                    pid for pid in active_for_vehicle
                    if pid != pickup_id
                )
            else:
                next_active = active_for_vehicle | {nid}

            routes = list(state.routes)
            loads = list(state.loads)
            costs = list(state.costs)
            active_pickups = list(state.active_pickups)

            step_cost = _dist(last_node, node)
            routes[vehicle_idx] = route + (nid,)
            loads[vehicle_idx] = 0.0 if abs(new_load) <= EPS else new_load
            costs[vehicle_idx] += step_cost
            active_pickups[vehicle_idx] = next_active

            child = BBState(
                routes=tuple(routes),
                loads=tuple(loads),
                costs=tuple(costs),
                served=state.served | {nid},
                active_pickups=tuple(active_pickups),
            )
            children.append((sum(child.costs), child))

    children.sort(key=lambda item: item[0], reverse=True)
    return children

def _solve_multi_vehicle_chunk(
    depot: Node,
    requests: list[Request],
    vehicles: list[Vehicle],
    time_limit_seconds: float,
) -> Solution:
    node_map, demand_of, pickup_of, _, delivery_ids, all_ids = _build_node_maps(requests)
    vehicle_count = len(vehicles)

    incumbent = _initial_incumbent(depot, requests, vehicles)
    best_cost = incumbent.total_cost if incumbent.routes else float("inf")
    best_routes = _routes_to_state_routes(incumbent, vehicles, vehicle_count)

    empty_routes = tuple(tuple() for _ in vehicles)
    initial = BBState(
        routes=empty_routes,
        loads=tuple(0.0 for _ in vehicles),
        costs=tuple(0.0 for _ in vehicles),
        served=frozenset(),
        active_pickups=tuple(frozenset() for _ in vehicles),
    )

    start_time = time.perf_counter()
    stack: list[BBState] = [initial]

    while stack:
        if time.perf_counter() - start_time > time_limit_seconds:
            break

        state = stack.pop()
        unserved_ids = [nid for nid in all_ids if nid not in state.served]

        lb = _lower_bound(state, unserved_ids, depot, node_map)
        if lb >= best_cost - EPS:
            continue

        if not unserved_ids:
            if any(state.active_pickups):
                continue
            if any(abs(load) > EPS for load in state.loads):
                continue

            total = _state_complete_cost(state, depot, node_map)
            if total < best_cost - EPS:
                best_cost = total
                best_routes = state.routes
            continue

        for _, child in _branch_next_states(
            state=state,
            unserved_ids=unserved_ids,
            depot=depot,
            vehicles=vehicles,
            node_map=node_map,
            demand_of=demand_of,
            pickup_of=pickup_of,
            delivery_ids=delivery_ids,
        ):
            stack.append(child)

    if best_cost == float("inf"):
        return Solution()
    return _make_solution(best_routes, vehicles, depot, node_map)

def run_branch_and_bound_solver(
    nodes: list[Node],
    requests: list[Request],
    vehicles: list[Vehicle],
    capacity: int,
    time_limit_seconds: int = 30,
    **kwargs,
) -> Solution:
    if not nodes or not requests or not vehicles:
        return Solution()

    depot = nodes[0]
    return _solve_multi_vehicle_chunk(
        depot=depot,
        requests=requests,
        vehicles=vehicles,
        time_limit_seconds=time_limit_seconds,
    )
