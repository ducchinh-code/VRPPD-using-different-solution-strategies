from __future__ import annotations

import os
import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import csv
import gc
from dataclasses import dataclass

from models import Node, Solution, _DIST
from utils import parse_input


FILES_PER_GROUP = 20
B_AND_B_TIME_LIMIT = 60
GA_TIME_LIMIT = 60

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")

SELECTED_FILES = {
    "pdp_100": [
        "lc101.txt", "lc103.txt", "lc105.txt",
        "lr101.txt", "lr103.txt", "lr105.txt",
        "lrc101.txt", "lrc103.txt",
        "lc201.txt", "lr201.txt",
        "lc102.txt", "lc104.txt",
        "lr102.txt", "lr104.txt",
        "lrc102.txt", "lrc104.txt",
        "lc202.txt", "lc204.txt",
        "lr202.txt", "lrc201.txt",
    ],
    "pdp_200": [
        "LC1_2_1.txt", "LC1_2_3.txt", "LC1_2_5.txt",
        "LR1_2_1.txt", "LR1_2_3.txt", "LR1_2_5.txt",
        "LRC1_2_1.txt", "LRC1_2_3.txt",
        "LC2_2_1.txt", "LR2_2_1.txt",
        "LC1_2_2.txt", "LC1_2_4.txt",
        "LR1_2_2.txt", "LR1_2_4.txt",
        "LRC1_2_2.txt", "LRC1_2_4.txt",
        "LC2_2_2.txt", "LC2_2_4.txt",
        "LR2_2_2.txt", "LRC2_2_1.txt",
    ],
}

def run_greedy(nodes, requests, vehicles, capacity) -> Solution:
    from greedy import solve_greedy
    return solve_greedy(nodes, vehicles)


def run_divide_and_conquer(nodes, requests, vehicles, capacity) -> Solution:
    from divide_and_conquer import divide_kmeans, build_routes_greedy, two_opt

    k = min(5, len(requests))
    depot = nodes[0]
    clusters, _, _ = divide_kmeans(requests, k)

    all_routes = []
    vehicle_idx_start = 0
    for cid in sorted(clusters):
        cluster_requests = clusters[cid]
        routes = build_routes_greedy(
            cluster_requests, vehicles, depot,
            vehicle_idx_start=vehicle_idx_start,
        )
        routes = [two_opt(route) for route in routes]
        vehicle_idx_start += len(routes)
        all_routes.extend(routes)

    return Solution(routes=all_routes)


def run_branch_and_bound(nodes, requests, vehicles, capacity) -> Solution:
    from branch_and_bound import run_branch_and_bound_solver
    return run_branch_and_bound_solver(
        nodes=nodes, requests=requests,
        vehicles=vehicles, capacity=capacity,
        time_limit_seconds=B_AND_B_TIME_LIMIT,
    )


def run_genetic_algorithm(nodes, requests, vehicles, capacity) -> Solution:
    from geneticAlgorithm import genetic_algorithm
    return genetic_algorithm(
        nodes, requests, vehicles, capacity,
        time_limit_seconds=GA_TIME_LIMIT,
    )


STRATEGIES = [
    ("Greedy",              run_greedy),
    ("Divide & Conquer",    run_divide_and_conquer),
    ("Branch & Bound",      run_branch_and_bound),
    ("Genetic Algorithm",   run_genetic_algorithm),
]


@dataclass
class RunResult:
    strategy:    str
    dataset:     str
    file:        str
    num_nodes:   int
    num_requests: int
    cost:        float  = float("inf")
    routes:      int    = 0
    feasible:    bool   = False
    complete:    bool   = False
    valid:       bool   = False
    time_ms:     float  = 0.0
    error:       str    = ""

def solution_coverage(nodes: list[Node], sol: Solution) -> dict:
    required  = {n.id for n in nodes if not n.is_depot}
    served    = [n.id for r in sol.routes for n in r.nodes]
    served_s  = set(served)
    return {
        "complete":      required == served_s and len(served) == len(served_s),
        "required":      len(required),
        "served_unique": len(served_s),
        "missing":       len(required - served_s),
        "duplicates":    len(served) - len(served_s),
    }

def run_single(strategy_name, strategy_fn, nodes, requests, vehicles,
               capacity, dataset, filename) -> RunResult:
    result = RunResult(
        strategy=strategy_name,
        dataset=dataset,
        file=filename,
        num_nodes=len(nodes),
        num_requests=len(requests),
    )

    start = time.perf_counter()
    try:
        sol = strategy_fn(nodes, requests, vehicles, capacity)
    except Exception as exc:
        result.time_ms = (time.perf_counter() - start) * 1000
        result.error   = f"{type(exc).__name__}: {exc}"
        return result

    elapsed = time.perf_counter() - start
    result.time_ms = elapsed * 1000

    if sol is None or not sol.routes:
        result.error = "No solution"
        return result

    cov = solution_coverage(nodes, sol)
    result.cost     = sol.total_cost
    result.routes   = len(sol.routes)
    result.feasible = sol.is_feasible()
    result.complete = cov["complete"]
    result.valid    = result.feasible and result.complete
    return result

SEP  = "=" * 110
SEP2 = "-" * 110

def print_group_table(group: str, results: list[RunResult]) -> None:
    print(f"\n{SEP}")
    print(f"  BỘ DỮ LIỆU: {group.upper()}  ({len(set(r.file for r in results))} files × "
          f"{len(STRATEGIES)} thuật toán = {len(results)} runs)")
    print(SEP)

    header = (f"{'File':<16} {'Thuật toán':<22} {'Cost':>12} {'Routes':>7} "
              f"{'Valid':>6} {'Feasible':>9} {'Time(ms)':>10} {'Ghi chú'}")
    print(header)
    print(SEP2)

    files_in_order = list(dict.fromkeys(r.file for r in results))
    for fname in files_in_order:
        file_results = [r for r in results if r.file == fname]
        valid_costs   = [r.cost for r in file_results if r.valid]
        best_cost     = min(valid_costs) if valid_costs else None

        for r in file_results:
            marker = ""
            if r.error:
                cost_s   = "ERROR"
                routes_s = "-"
                valid_s  = "-"
                feas_s   = "-"
                marker   = r.error[:30]
            else:
                cost_s   = f"{r.cost:.2f}"
                routes_s = str(r.routes)
                valid_s  = "✓" if r.valid else "✗"
                feas_s   = "✓" if r.feasible else "✗"
                if best_cost is not None and r.valid and abs(r.cost - best_cost) < 1e-6:
                    marker = "★ best"

            print(f"{r.file:<16} {r.strategy:<22} {cost_s:>12} {routes_s:>7} "
                  f"{valid_s:>6} {feas_s:>9} {r.time_ms:>10.1f} {marker}")
        print(SEP2)


def compute_group_summary(group: str, results: list[RunResult]) -> list[dict]:
    files = list(dict.fromkeys(r.file for r in results))
    strategies = [name for name, _ in STRATEGIES]
    summary = []

    for strat in strategies:
        strat_results = [r for r in results if r.strategy == strat]
        valid_results = [r for r in strat_results if r.valid]

        total_files     = len(strat_results)
        valid_count     = len(valid_results)
        avg_cost        = sum(r.cost for r in valid_results) / max(valid_count, 1)
        total_cost      = sum(r.cost for r in valid_results)
        avg_time        = sum(r.time_ms for r in strat_results) / max(total_files, 1)
        avg_routes      = sum(r.routes for r in valid_results) / max(valid_count, 1)

        wins = 0
        for fname in files:
            file_results = [r for r in results if r.file == fname]
            valid_costs  = [r.cost for r in file_results if r.valid]
            if not valid_costs:
                continue
            best = min(valid_costs)
            strat_r = next((r for r in file_results if r.strategy == strat), None)
            if strat_r and strat_r.valid and abs(strat_r.cost - best) < 1e-6:
                wins += 1

        summary.append({
            "strategy":      strat,
            "group":         group,
            "valid_count":   valid_count,
            "total_files":   total_files,
            "avg_cost":      avg_cost,
            "total_cost":    total_cost,
            "avg_routes":    avg_routes,
            "avg_time_ms":   avg_time,
            "wins":          wins,
        })

    return summary


def print_group_summary(group: str, summary: list[dict]) -> None:
    print(f"\n{'':>4}📊 TỔNG HỢP — {group.upper()}")
    print(f"{'':>4}{'-' * 100}")
    print(f"{'':>4}{'Thuật toán':<22} {'Valid':>8} {'Avg Cost':>12} "
          f"{'Total Cost':>14} {'Avg Routes':>11} {'Avg Time':>12} {'Wins':>6}")
    print(f"{'':>4}{'-' * 100}")

    for s in summary:
        valid_s = f"{s['valid_count']}/{s['total_files']}"
        print(f"{'':>4}{s['strategy']:<22} {valid_s:>8} {s['avg_cost']:>12.2f} "
              f"{s['total_cost']:>14.2f} {s['avg_routes']:>11.1f} "
              f"{s['avg_time_ms']:>10.1f}ms {s['wins']:>6}")
    print(f"{'':>4}{'-' * 100}")


def print_overall_ranking(all_summaries: list[dict]) -> None:
    strategies = [name for name, _ in STRATEGIES]
    groups     = list(dict.fromkeys(s["group"] for s in all_summaries))

    print(f"\n{SEP}")
    print(f"{'BẢNG XẾP HẠNG TỔNG THỂ':^110}")
    print(SEP)

    print(f"\n{'Thuật toán':<22}", end="")
    for g in groups:
        print(f" │ {'Valid':>5} {'AvgCost':>10} {'Wins':>5} {'AvgTime':>10}", end="")
    print(f" │ {'TỔNG WINS':>10} {'TỔNG ĐIỂM':>11}")
    print("-" * (22 + len(groups) * 36 + 26))

    scores = {s: 0 for s in strategies}
    total_wins = {s: 0 for s in strategies}

    for strat in strategies:
        print(f"{strat:<22}", end="")
        for g in groups:
            s = next(x for x in all_summaries if x["strategy"] == strat and x["group"] == g)
            valid_s = f"{s['valid_count']}/{s['total_files']}"
            print(f" │ {valid_s:>5} {s['avg_cost']:>10.1f} {s['wins']:>5} "
                  f"{s['avg_time_ms']:>8.0f}ms", end="")
            total_wins[strat] += s["wins"]

        for g in groups:
            group_data = [x for x in all_summaries if x["group"] == g]
            ranked = sorted(
                [x for x in group_data if x["valid_count"] > 0],
                key=lambda x: x["avg_cost"]
            )
            for rank, x in enumerate(ranked):
                if x["strategy"] == strat:
                    scores[strat] += max(len(STRATEGIES) - rank, 0)

        print(f" │ {total_wins[strat]:>10} {scores[strat]:>11}")

    print("-" * (22 + len(groups) * 36 + 26))

def save_csv(results: list[RunResult], filepath: str) -> None:
    fields = [
        "dataset", "file", "num_nodes", "num_requests",
        "strategy", "cost", "routes", "feasible", "complete",
        "valid", "time_ms", "error"
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "dataset":      r.dataset,
                "file":         r.file,
                "num_nodes":    r.num_nodes,
                "num_requests": r.num_requests,
                "strategy":     r.strategy,
                "cost":         f"{r.cost:.2f}" if r.cost < float("inf") else "INF",
                "routes":       r.routes,
                "feasible":     r.feasible,
                "complete":     r.complete,
                "valid":        r.valid,
                "time_ms":      f"{r.time_ms:.1f}",
                "error":        r.error,
            })
    print(f"\n  💾 Kết quả đã lưu ra: {filepath}")


def main() -> None:
    print(SEP)
    print(f"{'THỰC NGHIỆM VRPPD — SO SÁNH 4 THUẬT TOÁN':^110}")
    print(SEP)
    print(f"  Files per group : {FILES_PER_GROUP}")
    print(f"  B&B time limit  : {B_AND_B_TIME_LIMIT}s")
    print(f"  GA time limit   : {GA_TIME_LIMIT}s")
    print(f"  Strategies      : {', '.join(name for name, _ in STRATEGIES)}")
    print(f"  Data groups     : {', '.join(SELECTED_FILES.keys())}")
    print(SEP)

    all_results:   list[RunResult] = []
    all_summaries: list[dict]      = []

    for group, filenames in SELECTED_FILES.items():
        group_dir = os.path.join(DATA_DIR, group)
        if not os.path.isdir(group_dir):
            print(f"\n  ⚠ Thư mục {group_dir} không tồn tại — bỏ qua.")
            continue

        existing = [
            f for f in filenames
            if os.path.isfile(os.path.join(group_dir, f))
        ][:FILES_PER_GROUP]

        if not existing:
            print(f"\n  ⚠ Không tìm thấy file nào trong {group_dir} — bỏ qua.")
            continue

        print(f"\n{SEP}")
        print(f"  ▶ Đang chạy bộ {group.upper()} ({len(existing)} files)")
        print(SEP)

        group_results: list[RunResult] = []

        for fi, fname in enumerate(existing, 1):
            filepath = os.path.join(group_dir, fname)
            print(f"\n  [{fi}/{len(existing)}] File: {fname}")

            _DIST.clear()
            gc.collect()

            try:
                nodes, requests, vehicles, capacity = parse_input(filepath)
            except Exception as exc:
                print(f"    ✗ Lỗi đọc file: {exc}")
                continue

            print(f"    Nodes={len(nodes)}, Requests={len(requests)}, "
                  f"Vehicles={len(vehicles)}, Capacity={capacity}")

            for strat_name, strat_fn in STRATEGIES:
                _DIST.clear()

                sys.stdout.write(f"    → {strat_name:<22} ... ")
                sys.stdout.flush()

                result = run_single(
                    strat_name, strat_fn,
                    nodes, requests, vehicles, capacity,
                    group, fname,
                )
                group_results.append(result)

                if result.error:
                    print(f"ERROR ({result.time_ms:.0f}ms) — {result.error[:50]}")
                else:
                    valid_tag = "✓" if result.valid else "✗"
                    print(f"cost={result.cost:>10.2f}  routes={result.routes:>3}  "
                          f"valid={valid_tag}  ({result.time_ms:.0f}ms)")

        print_group_table(group, group_results)
        summary = compute_group_summary(group, group_results)
        print_group_summary(group, summary)

        all_results.extend(group_results)
        all_summaries.extend(summary)

    if all_summaries:
        print_overall_ranking(all_summaries)

    csv_path = os.path.join(SCRIPT_DIR, "experiment_results.csv")
    save_csv(all_results, csv_path)

    print(f"\n{SEP}")
    print(f"{'HOÀN THÀNH':^110}")
    print(SEP)

if __name__ == "__main__":
    main()
