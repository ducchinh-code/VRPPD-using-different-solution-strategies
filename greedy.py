# greedy.py

from models import Route, Solution


INF = 10**18


def build_cost_matrix(nodes):
    """
    Tạo ma trận cost:
        cost[i][j] = khoảng cách từ node i -> j

    Nếu cost = -1:
        nghĩa là node đã visit / không hợp lệ
    """

    n = len(nodes)

    cost = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):

            if i == j:
                cost[i][j] = 0
            else:
                cost[i][j] = nodes[i].distance_to(nodes[j])

    return cost


def mark_visited(cost_matrix, node_id):
    """
    Đánh dấu node đã visit:
        toàn bộ cột node_id = -1
    => không xe nào được chọn lại node này
    """

    n = len(cost_matrix)

    for i in range(n):
        cost_matrix[i][node_id] = -1


def can_visit(node, picked_requests, current_load, vehicle_capacity):
    """
    Kiểm tra node có thể visit hay không

    RULES:

    1. Pickup:
        chỉ được pickup nếu:
            current_load + demand <= capacity

    2. Delivery:
        chỉ được delivery nếu:
            pickup tương ứng đã được pickup trước đó
    """

    # DEPOT
    if node.is_depot:
        return False

    # PICKUP
    if node.is_pickup:

        if current_load + node.demand > vehicle_capacity:
            return False

        return True

    # DELIVERY
    if node.is_delivery:

        pickup_id = node.pickup_index

        if pickup_id not in picked_requests:
            return False

        return True

    return False


def get_best_next_node(
    current_node,
    nodes,
    cost_matrix,
    picked_requests,
    current_load,
    vehicle_capacity,
    visited
):
    """
    Chọn node hợp lệ gần nhất
    """

    best_node = None
    best_cost = INF

    for node in nodes:

        if node.id in visited:
            continue

        if not can_visit(
            node,
            picked_requests,
            current_load,
            vehicle_capacity
        ):
            continue

        move_cost = cost_matrix[current_node.id][node.id]

        if move_cost == -1:
            continue

        if move_cost < best_cost:
            best_cost = move_cost
            best_node = node

    return best_node


def solve_greedy(nodes, vehicles):
    """
    Greedy VRP-PD

    Ý tưởng:
    =========

    Với mỗi xe:
        - luôn chọn node hợp lệ gần nhất
        - sau khi visit:
            + pickup  -> thêm vào picked_requests
            + delivery -> giao hàng

    Điều kiện hợp lệ:
        - delivery chỉ được visit nếu pickup đã visit
        - pickup không vượt capacity

    Độ phức tạp:
        O(k * n^2)
    """

    depot = nodes[0]

    cost_matrix = build_cost_matrix(nodes)

    routes = []

    global_visited = set()
    global_visited.add(depot.id)

    for vehicle in vehicles:

        route = Route(vehicle, depot)

        current_node = depot

        current_load = 0

        picked_requests = set()

        while True:

            next_node = get_best_next_node(
                current_node=current_node,
                nodes=nodes,
                cost_matrix=cost_matrix,
                picked_requests=picked_requests,
                current_load=current_load,
                vehicle_capacity=vehicle.capacity,
                visited=global_visited
            )

            # không còn node hợp lệ
            if next_node is None:
                break

            # append vào route
            route.append(next_node)

            # đánh dấu visited
            global_visited.add(next_node.id)

            mark_visited(cost_matrix, next_node.id)

            # cập nhật tải
            current_load += next_node.demand

            # pickup
            if next_node.is_pickup:
                picked_requests.add(next_node.id)

            current_node = next_node

        # chỉ thêm route nếu có node
        if len(route.nodes) > 0:
            routes.append(route)

    return Solution(routes)
