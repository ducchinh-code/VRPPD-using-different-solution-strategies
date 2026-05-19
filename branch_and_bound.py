from __future__ import annotations
import pulp
import warnings
from models import Node, Request, Vehicle, Route, Solution

warnings.filterwarnings("ignore")

def run_branch_and_bound_solver(
    nodes: list[Node], 
    requests: list[Request], 
    vehicles: list[Vehicle], 
    capacity: int, 
    time_limit_seconds: int = 60,
    max_pairs: int | None = None  # Cờ điều khiển số lượng dữ liệu động
) -> Solution:
    """
    Thuật toán Branch & Bound có khả năng co giãn theo tham số max_pairs.
    Nếu max_pairs = None, nó sẽ cố gắng giải TOÀN BỘ file dữ liệu (Cẩn thận treo máy!).
    """
    # 1. ĐỘNG HÓA DỮ LIỆU ĐẦU VÀO
    if max_pairs is not None:
        requests_core = requests[:max_pairs]
        # Tính toán linh hoạt số xe cần thiết dựa trên tổng lượng hàng
        total_demand = sum(r.pickup.demand for r in requests_core)
        needed_vehicles = max(1, min(len(vehicles), int(total_demand // capacity) + 2))
        vehicles_core = vehicles[:needed_vehicles]
    else:
        # Nhận toàn bộ dữ liệu nếu user muốn
        requests_core = requests
        vehicles_core = vehicles
        
    core_nodes = [nodes[0]] # Depot
    for req in requests_core:
        core_nodes.append(req.pickup)
        core_nodes.append(req.delivery)
        
    depot = nodes[0]
    P = [req.pickup for req in requests_core]
    D = [req.delivery for req in requests_core]

    # Kiểm tra an toàn bộ nhớ
    num_vars = len(core_nodes) * len(core_nodes) * len(vehicles_core)
    if num_vars > 1000000:
        print(f"  [Cảnh báo] Số lượng biến dự kiến quá lớn ({num_vars}). Khả năng cao sẽ tràn RAM!")

    prob = pulp.LpProblem("VRPPD_Branch_and_Bound", pulp.LpMinimize)

    # Khởi tạo biến
    x = pulp.LpVariable.dicts("x", 
        ((i.id, j.id, k.id) for i in core_nodes for j in core_nodes for k in vehicles_core if i.id != j.id),
        cat='Binary'
    )
    T = pulp.LpVariable.dicts("T", (i.id for i in core_nodes), lowBound=0, cat='Continuous')
    L = pulp.LpVariable.dicts("L", (i.id for i in core_nodes), lowBound=0, cat='Continuous')

    # Hàm mục tiêu
    prob += pulp.lpSum(
        i.distance_to(j) * x[i.id, j.id, k.id] 
        for i in core_nodes for j in core_nodes for k in vehicles_core if i.id != j.id
    ), "Total_Cost"

    # Ràng buộc cơ bản
    for i in P + D:
        prob += pulp.lpSum(x[i.id, j.id, k.id] for j in core_nodes for k in vehicles_core if i.id != j.id) == 1
                           
    for req in requests_core:
        for k in vehicles_core:
            prob += pulp.lpSum(x[req.pickup.id, j.id, k.id] for j in core_nodes if j.id != req.pickup.id) == \
                    pulp.lpSum(x[req.delivery.id, j.id, k.id] for j in core_nodes if j.id != req.delivery.id)

    for h in core_nodes:
        for k in vehicles_core:
            prob += pulp.lpSum(x[i.id, h.id, k.id] for i in core_nodes if i.id != h.id) == \
                    pulp.lpSum(x[h.id, j.id, k.id] for j in core_nodes if j.id != h.id)

    for k in vehicles_core:
        prob += pulp.lpSum(x[depot.id, j.id, k.id] for j in core_nodes if j.id != depot.id) <= 1
        prob += pulp.lpSum(x[i.id, depot.id, k.id] for i in core_nodes if i.id != depot.id) <= 1

    BigM_T = len(core_nodes) * 10
    prob += L[depot.id] == 0
    
    for k in vehicles_core:
        for i in core_nodes:
            for j in core_nodes:
                if i.id != j.id and j.id != depot.id:
                    prob += T[j.id] >= T[i.id] + 1 - BigM_T * (1 - x[i.id, j.id, k.id])
                    prob += L[j.id] >= L[i.id] + j.demand - (capacity * 2) * (1 - x[i.id, j.id, k.id])
        for i in core_nodes:
            prob += L[i.id] <= capacity

    for req in requests_core:
        prob += T[req.pickup.id] + 1 <= T[req.delivery.id]

    # Gọi bộ giải
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit_seconds, msg=False)
    prob.solve(solver)

    # Đóng gói kết quả
    solution_routes = []
    node_dict = {n.id: n for n in core_nodes}

    if pulp.LpStatus[prob.status] in ['Optimal', 'Feasible'] and pulp.value(prob.objective) is not None:
        for k in vehicles_core:
            current_node_id = depot.id
            route_nodes = []
            while True:
                next_node_id = None
                for j in core_nodes:
                    if current_node_id != j.id:
                        var = x.get((current_node_id, j.id, k.id))
                        if var and var.varValue and var.varValue > 0.5:
                            next_node_id = j.id
                            break
                if next_node_id is None or next_node_id == depot.id:
                    break
                route_nodes.append(node_dict[next_node_id])
                current_node_id = next_node_id
            
            if route_nodes:
                r = Route(vehicle=k, depot=depot)
                for n in route_nodes:
                    r.append(n)
                solution_routes.append(r)

    return Solution(routes=solution_routes)
