import os
import time

from models import Node, Solution
from utils import parse_input


B_AND_B_TIME_LIMIT_SECONDS = 30


def run_greedy(nodes, requests, vehicles, capacity) -> Solution:
    from greedy import solve_greedy

    return solve_greedy(nodes, vehicles)


def run_divide_and_conquer(nodes, requests, vehicles, capacity) -> Solution:
    from divide_and_conquer import divide_kmeans, build_routes_greedy, two_opt

    k = 5
    use_2opt = True
    depot = nodes[0]

    clusters, _, _ = divide_kmeans(requests, k)

    all_routes = []
    vehicle_idx_start = 0
    for cid in sorted(clusters):
        cluster_requests = clusters[cid]
        routes = build_routes_greedy(
            cluster_requests,
            vehicles,
            depot,
            vehicle_idx_start=vehicle_idx_start,
        )

        if use_2opt:
            routes = [two_opt(route) for route in routes]

        vehicle_idx_start += len(routes)
        all_routes.extend(routes)

    return Solution(routes=all_routes)


def run_branch_and_bound(nodes, requests, vehicles, capacity) -> Solution:
    from branch_and_bound import run_branch_and_bound_solver
    return run_branch_and_bound_solver(
        nodes=nodes,
        requests=requests,
        vehicles=vehicles,
        capacity=capacity,
        time_limit_seconds=B_AND_B_TIME_LIMIT_SECONDS,
    )


def run_genetic_algorithm(nodes, requests, vehicles, capacity) -> Solution:
    from geneticAlgorithm import genetic_algorithm

    return genetic_algorithm(nodes, requests, vehicles, capacity)


SEP = "=" * 70
SEP2 = "-" * 70


def solution_coverage_report(nodes: list[Node], solution: Solution) -> dict:
    required = {node.id for node in nodes if not node.is_depot}
    served = [node.id for route in solution.routes for node in route.nodes]
    served_set = set(served)

    return {
        "complete": required == served_set and len(served) == len(served_set),
        "required_count": len(required),
        "served_count": len(served),
        "unique_served_count": len(served_set),
        "missing_count": len(required - served_set),
        "duplicate_count": len(served) - len(served_set),
    }


def solution_is_valid(nodes: list[Node], solution: Solution) -> bool:
    coverage = solution_coverage_report(nodes, solution)
    return solution.is_feasible() and coverage["complete"]


def print_solution(name: str, solution: Solution, elapsed: float,
                   nodes: list[Node], error: Exception | None = None) -> None:
    print(f"\n{name}")
    print(SEP2)

    if error is not None:
        print(f"  Skipped/error: {type(error).__name__}: {error}")
        print(f"  Time: {elapsed * 1000:.1f} ms")
        return

    if not solution.routes:
        print("  No solution returned.")
        print(f"  Time: {elapsed * 1000:.1f} ms")
        return

    for route in solution.routes:
        report = route.feasibility_report()
        route_ok = "OK" if report["feasible"] else (
            f"BAD cap={len(report['cap_violations'])} "
            f"prec={len(report['prec_violations'])}"
        )

        def node_str(node: Node) -> str:
            tag = "p" if node.is_pickup else "d"
            return f"{tag}{node.id}"

        sequence = " -> ".join(node_str(node) for node in route.nodes)
        print(
            f"  Vehicle {route.vehicle.id:>2} [Q={route.vehicle.capacity}]: "
            f"0 -> {sequence} -> 0"
        )

        load = 0.0
        load_sequence = ["0"]
        for node in route.nodes:
            load += node.demand
            load_sequence.append(f"{load:+.0f}")

        print(f"    Load: {' -> '.join(load_sequence)}")
        print(f"    Cost: {report['total_cost']:.2f}  {route_ok}")

    coverage = solution_coverage_report(nodes, solution)
    valid = solution_is_valid(nodes, solution)

    print()
    print(f"  Total cost : {solution.total_cost:.2f}")
    print(f"  Routes     : {len(solution.routes)}")
    print(f"  Feasible   : {'yes' if solution.is_feasible() else 'no'}")
    print(f"  Complete   : {'yes' if coverage['complete'] else 'no'}")
    print(
        "  Coverage   : "
        f"{coverage['unique_served_count']}/{coverage['required_count']} unique, "
        f"missing={coverage['missing_count']}, "
        f"duplicates={coverage['duplicate_count']}"
    )
    print(f"  Valid      : {'yes' if valid else 'no'}")
    print(f"  Time       : {elapsed * 1000:.1f} ms")


def print_comparison(
    results: list[tuple[str, Solution, float, Exception | None]],
    nodes: list[Node],
) -> None:
    print(f"\n{SEP}")
    print(f"{'COMPARISON':^70}")
    print(SEP)
    print(f"{'Strategy':<24} {'Cost':>10} {'Routes':>8} {'Valid':>8} {'T(ms)':>10}")
    print(SEP2)

    best_cost = min(
        (
            solution.total_cost
            for _, solution, _, error in results
            if error is None and solution.routes and solution_is_valid(nodes, solution)
        ),
        default=float("inf"),
    )

    for name, solution, elapsed, error in results:
        if error is not None or not solution.routes:
            cost = "-"
            routes = "-"
            valid = "-"
            marker = ""
        else:
            cost = f"{solution.total_cost:.2f}"
            routes = str(len(solution.routes))
            valid = "yes" if solution_is_valid(nodes, solution) else "no"
            marker = " *" if solution.total_cost == best_cost else ""

        print(
            f"{name:<24} {cost:>10} {routes:>8} "
            f"{valid:>8} {elapsed * 1000:>10.1f}{marker}"
        )

    print(SEP2)
    print("* = lowest cost among valid complete solutions")


STRATEGIES = [
    ("Greedy", run_greedy),
    ("Divide & Conquer", run_divide_and_conquer),
    ("Branch & Bound", run_branch_and_bound),
    ("Genetic Algorithm", run_genetic_algorithm),
]


def find_default_data_file() -> str | None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "lc103.txt"),
        os.path.join(script_dir, "data", "pdp_100", "lc103.txt"),
    ]
    return next((path for path in candidates if os.path.isfile(path)), None)


def main() -> None:
    file_path = find_default_data_file()
    if file_path is None:
        print("[ERROR] Could not find lc101.txt.")
        return

    nodes, requests, vehicles, capacity = parse_input(file_path)

    print(SEP)
    print("VRPPD SOLVER - strategy comparison")
    print(SEP)
    print(f"File     : {os.path.basename(file_path)}")
    print(
        f"Nodes    : {len(nodes)} "
        f"(depot + {len(requests)} pickup + {len(requests)} delivery)"
    )
    print(f"Vehicles : {len(vehicles)} (capacity = {capacity})")
    print(f"Requests : {len(requests)} pickup-delivery pairs")

    print(f"\n{SEP}")
    print("STRATEGY RESULTS")
    print(SEP)

    results = []
    for name, func in STRATEGIES:
        start = time.perf_counter()
        error = None
        try:
            solution = func(nodes, requests, vehicles, capacity)
        except Exception as exc:
            solution = Solution()
            error = exc

        elapsed = time.perf_counter() - start
        print_solution(name, solution, elapsed, nodes, error)
        results.append((name, solution, elapsed, error))

    print_comparison(results, nodes)
    print(f"\n{SEP}\nDone.\n{SEP}")


if __name__ == "__main__":
    main()
