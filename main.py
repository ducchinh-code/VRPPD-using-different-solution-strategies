import time
import os
from utils  import parse_input
from models import Node, Request, Vehicle, Route, Solution

def run_greedy(nodes, requests, vehicles, capacity) -> Solution:
    # TODO: implement greedy.py
    return Solution()

def run_divide_and_conquer(nodes, requests, vehicles, capacity) -> Solution:
    from divide_and_conquer import divide_kmeans, build_routes_greedy, two_opt

    K = 5
    USE_2OPT = True
    depot = nodes[0]

    clusters, _, _ = divide_kmeans(requests, K)

    all_routes = []
    v_offset = 0
    for cid in sorted(clusters):
        reqs = clusters[cid]
        routes = build_routes_greedy(reqs, vehicles, depot, vehicle_idx_start=v_offset)

        if USE_2OPT:
            routes = [two_opt(r) for r in routes]
            
        v_offset += len(routes)
        all_routes.extend(routes)
        
    return Solution(routes=all_routes)

def run_branch_and_bound(nodes, requests, vehicles, capacity) -> Solution:
    # TODO: implement branch_and_bound.py
    return Solution()

def run_genetic_algorithm(nodes, requests, vehicles, capacity) -> Solution:
    from geneticAlgorithm import genetic_algorithm
    return genetic_algorithm(nodes, requests, vehicles, capacity)

SEP  = "═" * 70
SEP2 = "─" * 70
SEP3 = "·" * 70

def print_solution(name: str, solution: Solution, elapsed: float) -> None:
    print(f"\n  ┌─ {name}")

    if not solution.routes:
        print(f"  │  (chưa có nghiệm — stub)")
        print(f"  └{SEP2[2:]}")
        return

    for route in solution.routes:
        U    = route.compute_U()
        rpt  = route.feasibility_report()
        ok   = "✓ KHẢ THI" if rpt["feasible"] else \
               f"✗ CAP={len(rpt['cap_violations'])} PREC={len(rpt['prec_violations'])}"

        def node_str(n: Node) -> str:
            tag = "p" if n.is_pickup else "d"
            return f"{tag}{n.id}(U={U[n.id]})"

        seq = " → ".join(node_str(n) for n in route.nodes)
        print(f"  │  Xe {route.vehicle.id:>2} [Q={route.vehicle.capacity}]: "
              f"0(U=0) → {seq} → 0")

        load = 0.0
        load_seq = ["L=0"]
        for n in route.nodes:
            load += n.demand
            load_seq.append(f"L={load:+.0f}")
        print(f"  │       Tải  : {' → '.join(load_seq)}")
        print(f"  │       Cost : {rpt['total_cost']:.2f}   {ok}")

    print(f"  │")
    print(f"  │  Tổng cost  : {solution.total_cost:.2f}")
    print(f"  │  Số tuyến   : {len(solution.routes)}")
    print(f"  │  Khả thi    : {'✓ Tất cả' if solution.is_feasible() else '✗ Có vi phạm'}")
    print(f"  │  Thời gian  : {elapsed*1000:.1f} ms")
    print(f"  └{SEP2[2:]}")

def print_comparison(results: list[tuple[str, Solution, float]]) -> None:
    print(f"\n{SEP}")
    print(f"  {'BẢNG SO SÁNH':^66}")
    print(SEP)
    hdr = (f"  {'Chiến thuật':<26} {'Cost':>10} {'Tuyến':>6} "
           f"{'Khả thi':>8} {'T(ms)':>8}")
    print(hdr)
    print(f"  {SEP2}")

    best_cost = min(
        (s.total_cost for _, s, _ in results if s.routes),
        default=float("inf")
    )

    for name, sol, elapsed in results:
        if not sol.routes:
            cost_str = "—"
            n_routes = "—"
            feasible = "—"
        else:
            cost_str = f"{sol.total_cost:.2f}"
            n_routes = str(len(sol.routes))
            feasible = "✓" if sol.is_feasible() else "✗"

        marker = " ★" if sol.routes and sol.total_cost == best_cost else "  "
        print(f"  {name:<26} {cost_str:>10} {n_routes:>6} "
              f"{feasible:>8} {elapsed*1000:>7.1f}{marker}")

    print(f"  {SEP2}")
    print(f"  ★ = nghiệm tốt nhất (min cost)")

STRATEGIES = [
    ("Greedy",               run_greedy),
    ("Divide & Conquer",     run_divide_and_conquer),
    ("Branch & Bound",       run_branch_and_bound),
    ("Genetic Algorithm",    run_genetic_algorithm),
]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "lc101.txt"),
        os.path.join(script_dir, "data", "pdp_100", "lc101.txt"),
    ]
    file_path = next((p for p in candidates if os.path.isfile(p)), None)
    if file_path is None:
        print("[LỖI] Không tìm thấy file dữ liệu (lc101.txt).")
        return

    nodes, requests, vehicles, capacity = parse_input(file_path)
    depot      = nodes[0]
    node_by_id = {n.id: n for n in nodes}

    print(SEP)
    print("  VRPPD SOLVER — So sánh chiến thuật")
    print(SEP)
    print(f"  File       : {os.path.basename(file_path)}")
    print(f"  |V|        : {len(nodes)}  (depot + {len(requests)} pickup + {len(requests)} delivery)")
    print(f"  |K|        : {len(vehicles)} xe  (Q_k = {capacity})")
    print(f"  |R|        : {len(requests)} cặp yêu cầu (pᵢ, dᵢ)")

    print(f"\n{SEP}")
    print(f"  KẾT QUẢ TỪNG CHIẾN THUẬT")
    print(SEP)

    results = []
    for name, func in STRATEGIES:
        t0 = time.perf_counter()
        try:
            solution = func(nodes, requests, vehicles, capacity)
        except NotImplementedError:
            solution = Solution()
        elapsed = time.perf_counter() - t0

        print_solution(name, solution, elapsed)
        results.append((name, solution, elapsed))

    print_comparison(results)

    print(f"\n{SEP}\n  Xong!\n{SEP}\n")

if __name__ == "__main__":
    main()