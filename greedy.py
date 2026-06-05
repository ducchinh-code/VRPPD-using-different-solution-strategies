from models import Route, Solution

INF = 10**18
EPS = 1e-9


def build_cost_matrix(nodes):
    return {
        a.id: {
            b.id: 0.0 if a.id == b.id else a.distance_to(b)
            for b in nodes
        }
        for a in nodes
    }


def mark_visited(cost_matrix, node_id):
    rows = cost_matrix.values() if hasattr(cost_matrix, "values") else cost_matrix
    for row in rows:
        if node_id in row:
            row[node_id] = -1


def can_visit(node, picked_requests, current_load, vehicle_capacity):
    active_pickups = picked_requests

    if node.is_depot:
        return False

    next_load = current_load + node.demand

    if node.is_pickup:
        return next_load <= vehicle_capacity + EPS

    if node.is_delivery:
        pickup_id = node.pickup_index
        if pickup_id not in active_pickups:
            return False
        return -EPS <= next_load <= vehicle_capacity + EPS

    return False


def get_best_next_node(
    current_node,
    nodes,
    cost_matrix,
    picked_requests,
    current_load,
    vehicle_capacity,
    visited,
):
    best_node = None
    best_cost = INF

    for node in nodes:
        if node.id in visited:
            continue

        if not can_visit(
            node,
            picked_requests,
            current_load,
            vehicle_capacity,
        ):
            continue

        move_cost = cost_matrix.get(current_node.id, {}).get(node.id, INF)
        if move_cost == -1:
            continue

        if move_cost < best_cost:
            best_cost = move_cost
            best_node = node

    return best_node


def validate_solution(nodes, routes):
    required = {node.id for node in nodes if not node.is_depot}
    served = [node.id for route in routes for node in route.nodes]
    served_set = set(served)

    missing = sorted(required - served_set)
    duplicate_count = len(served) - len(served_set)
    infeasible_routes = [
        route.vehicle.id
        for route in routes
        if not route.is_feasible()
    ]
    nonempty_load_routes = []

    for route in routes:
        final_load = sum(node.demand for node in route.nodes)
        if abs(final_load) > EPS:
            nonempty_load_routes.append((route.vehicle.id, final_load))

    if missing or duplicate_count or infeasible_routes or nonempty_load_routes:
        raise ValueError(
            "Greedy produced an invalid solution: "
            f"missing={missing}, "
            f"duplicate_count={duplicate_count}, "
            f"infeasible_routes={infeasible_routes}, "
            f"nonempty_load_routes={nonempty_load_routes}"
        )


def validate_solution_coverage(nodes, routes):
    validate_solution(nodes, routes)


def solve_greedy(nodes, vehicles):
    if not nodes:
        return Solution()

    depot = nodes[0]
    cost_matrix = build_cost_matrix(nodes)
    routes = []
    global_visited = {depot.id}

    for vehicle in vehicles:
        route = Route(vehicle, depot)
        current_node = depot
        current_load = 0.0
        active_pickups = set()

        while True:
            next_node = get_best_next_node(
                current_node=current_node,
                nodes=nodes,
                cost_matrix=cost_matrix,
                picked_requests=active_pickups,
                current_load=current_load,
                vehicle_capacity=vehicle.capacity,
                visited=global_visited,
            )

            if next_node is None:
                break

            route.append(next_node)
            global_visited.add(next_node.id)
            current_load += next_node.demand

            if next_node.is_pickup:
                active_pickups.add(next_node.id)
            elif next_node.is_delivery:
                active_pickups.remove(next_node.pickup_index)

            current_node = next_node

        if active_pickups or abs(current_load) > EPS:
            raise ValueError(
                "Greedy route ended before all picked requests were delivered: "
                f"vehicle={vehicle.id}, "
                f"active_pickups={sorted(active_pickups)}, "
                f"load={current_load}"
            )

        if route.nodes:
            routes.append(route)

    validate_solution(nodes, routes)
    return Solution(routes)
