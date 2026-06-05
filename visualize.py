"""
VRPPD Route Visualizer
======================
Giao diện chọn file dữ liệu + thuật toán, chạy giải, và trực quan hóa lộ trình.

Cách dùng:
    python visualize.py
"""

from __future__ import annotations

import os
import sys
import io
import time
import tkinter as tk
from tkinter import ttk

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np

from models import Node, Request, Vehicle, Route, Solution, _DIST
from utils import parse_input

# ── Đường dẫn dữ liệu ───────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

B_AND_B_TIME_LIMIT = 30
GA_TIME_LIMIT = 60

# ── Các thuật toán ────────────────────────────────────────────────────────────

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


STRATEGIES = {
    "Greedy":              run_greedy,
    "Divide & Conquer":    run_divide_and_conquer,
    "Branch & Bound":      run_branch_and_bound,
    "Genetic Algorithm":   run_genetic_algorithm,
}

# ── Hàm phụ trợ ──────────────────────────────────────────────────────────────

def scan_data_files() -> dict[str, list[str]]:
    """Quét thư mục data/ và trả về {group_name: [file_name, ...]}."""
    groups: dict[str, list[str]] = {}
    if not os.path.isdir(DATA_DIR):
        return groups
    for group in sorted(os.listdir(DATA_DIR)):
        group_path = os.path.join(DATA_DIR, group)
        if not os.path.isdir(group_path):
            continue
        files = sorted(
            f for f in os.listdir(group_path)
            if f.endswith(".txt") and os.path.isfile(os.path.join(group_path, f))
        )
        if files:
            groups[group] = files
    return groups


def solution_coverage(nodes: list[Node], sol: Solution) -> dict:
    required = {n.id for n in nodes if not n.is_depot}
    served = [n.id for r in sol.routes for n in r.nodes]
    served_s = set(served)
    return {
        "complete":      required == served_s and len(served) == len(served_s),
        "required":      len(required),
        "served_unique": len(served_s),
        "missing":       len(required - served_s),
        "duplicates":    len(served) - len(served_s),
    }

# ── Bảng màu ─────────────────────────────────────────────────────────────────

ROUTE_COLORS = [
    "#4fc1ff", "#ff79c6", "#50fa7b", "#ffb86c", "#bd93f9",
    "#f1fa8c", "#8be9fd", "#ff5555", "#6272a4", "#f8f8f2",
    "#4ec9b0", "#f48c42", "#c586c0", "#9cdcfe", "#ce9178",
    "#dcdcaa", "#6a9955", "#e06c75", "#98c379", "#61afef",
]

# ── Giao diện chọn file + thuật toán ─────────────────────────────────────────

class SelectorDialog:
    """Cửa sổ tkinter để chọn bộ dữ liệu, file và thuật toán."""

    def __init__(self):
        self.result: dict | None = None
        self.groups = scan_data_files()

        if not self.groups:
            raise FileNotFoundError(
                f"Không tìm thấy dữ liệu trong {DATA_DIR}. "
                "Hãy đảm bảo thư mục data/ chứa các thư mục con pdp_100, pdp_200, ..."
            )

        self.root = tk.Tk()
        self.root.title("VRPPD — Chọn dữ liệu & thuật toán")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(False, False)

        self._setup_styles()
        self._build_ui()
        self._center_window()

    # ── Styles ──

    def _setup_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("Dark.TFrame", background="#1e1e2e")
        style.configure("Dark.TLabel", background="#1e1e2e",
                         foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#1e1e2e",
                         foreground="#89b4fa", font=("Segoe UI", 16, "bold"))
        style.configure("Subtitle.TLabel", background="#1e1e2e",
                         foreground="#a6adc8", font=("Segoe UI", 9))
        style.configure("Dark.TLabelframe", background="#1e1e2e",
                         foreground="#cdd6f4", font=("Segoe UI", 10, "bold"))
        style.configure("Dark.TLabelframe.Label", background="#1e1e2e",
                         foreground="#89b4fa", font=("Segoe UI", 10, "bold"))
        style.configure("Dark.TCombobox", fieldbackground="#313244",
                         background="#313244", foreground="#cdd6f4",
                         selectbackground="#45475a", selectforeground="#cdd6f4",
                         font=("Consolas", 10))
        style.map("Dark.TCombobox",
                  fieldbackground=[("readonly", "#313244")],
                  foreground=[("readonly", "#cdd6f4")])
        style.configure("Run.TButton", background="#89b4fa",
                         foreground="#1e1e2e", font=("Segoe UI", 11, "bold"),
                         padding=(20, 10))
        style.map("Run.TButton",
                  background=[("active", "#74c7ec"), ("pressed", "#b4befe")])

    # ── Build UI ──

    def _build_ui(self):
        main = ttk.Frame(self.root, style="Dark.TFrame", padding=30)
        main.pack(fill="both", expand=True)

        # Tiêu đề
        ttk.Label(main, text="🚛  VRPPD Route Visualizer",
                  style="Title.TLabel").pack(pady=(0, 2))
        ttk.Label(main, text="Chọn file dữ liệu và thuật toán để chạy & xem kết quả",
                  style="Subtitle.TLabel").pack(pady=(0, 20))

        # ── Khung chọn ──
        sel_frame = ttk.LabelFrame(main, text="  Cấu hình  ",
                                    style="Dark.TLabelframe", padding=15)
        sel_frame.pack(fill="x", pady=(0, 15))

        # Bộ dữ liệu
        row1 = ttk.Frame(sel_frame, style="Dark.TFrame")
        row1.pack(fill="x", pady=5)
        ttk.Label(row1, text="Bộ dữ liệu:", style="Dark.TLabel",
                  width=14).pack(side="left")
        self.group_var = tk.StringVar()
        self.group_cb = ttk.Combobox(
            row1, textvariable=self.group_var,
            values=list(self.groups.keys()),
            state="readonly", style="Dark.TCombobox", width=30,
        )
        self.group_cb.pack(side="left", padx=(5, 0))
        self.group_cb.bind("<<ComboboxSelected>>", self._on_group_change)

        # File
        row2 = ttk.Frame(sel_frame, style="Dark.TFrame")
        row2.pack(fill="x", pady=5)
        ttk.Label(row2, text="File:", style="Dark.TLabel",
                  width=14).pack(side="left")
        self.file_var = tk.StringVar()
        self.file_cb = ttk.Combobox(
            row2, textvariable=self.file_var,
            state="readonly", style="Dark.TCombobox", width=30,
        )
        self.file_cb.pack(side="left", padx=(5, 0))

        # Thuật toán
        row3 = ttk.Frame(sel_frame, style="Dark.TFrame")
        row3.pack(fill="x", pady=5)
        ttk.Label(row3, text="Thuật toán:", style="Dark.TLabel",
                  width=14).pack(side="left")
        self.strat_var = tk.StringVar()
        self.strat_cb = ttk.Combobox(
            row3, textvariable=self.strat_var,
            values=list(STRATEGIES.keys()),
            state="readonly", style="Dark.TCombobox", width=30,
        )
        self.strat_cb.pack(side="left", padx=(5, 0))
        self.strat_cb.current(0)

        # Nút chạy
        btn_frame = ttk.Frame(main, style="Dark.TFrame")
        btn_frame.pack(pady=(10, 0))
        self.run_btn = ttk.Button(
            btn_frame, text="▶  Chạy & Trực quan hóa",
            style="Run.TButton", command=self._on_run,
        )
        self.run_btn.pack()

        # Thiết lập mặc định
        self.group_cb.current(0)
        self._on_group_change()

    # ── Callbacks ──

    def _on_group_change(self, _event=None):
        group = self.group_var.get()
        files = self.groups.get(group, [])
        self.file_cb["values"] = files
        if files:
            self.file_cb.current(0)

    def _on_run(self):
        group = self.group_var.get()
        fname = self.file_var.get()
        strat = self.strat_var.get()

        if not group or not fname or not strat:
            return

        self.result = {
            "group": group,
            "file":  fname,
            "path":  os.path.join(DATA_DIR, group, fname),
            "strategy": strat,
        }
        self.root.destroy()

    def _center_window(self):
        self.root.update_idletasks()
        w = self.root.winfo_reqwidth()
        h = self.root.winfo_reqheight()
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"+{x}+{y}")

    def run(self) -> dict | None:
        self.root.mainloop()
        return self.result

# ── Trực quan hóa kết quả ────────────────────────────────────────────────────

def visualize_solution(
    solution: Solution,
    nodes: list[Node],
    strategy_name: str,
    file_name: str,
    elapsed_ms: float,
    capacity: float,
) -> None:
    """Vẽ biểu đồ lộ trình VRPPD."""

    depot = nodes[0]
    cov = solution_coverage(nodes, solution)
    feasible = solution.is_feasible()
    valid = feasible and cov["complete"]

    num_routes = len(solution.routes)
    cmap = plt.colormaps["tab20"].resampled(max(num_routes, 1))

    # ── Tạo figure ──

    fig = plt.figure(figsize=(18, 10), facecolor="#0d1117")
    fig.canvas.manager.set_window_title(
        f"VRPPD — {strategy_name} — {file_name}"
    )

    # Chia layout: biểu đồ chính (trái) + bảng thông tin (phải)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

    # ── Biểu đồ bản đồ (trái) ──

    ax_map.set_facecolor("#161b22")
    ax_map.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax_map.spines.values():
        spine.set_color("#30363d")
    ax_map.xaxis.label.set_color("#8b949e")
    ax_map.yaxis.label.set_color("#8b949e")
    ax_map.grid(True, color="#21262d", linewidth=0.5, zorder=0)
    ax_map.set_xlabel("X", fontsize=10)
    ax_map.set_ylabel("Y", fontsize=10)

    # Vẽ từng tuyến xe
    legend_handles = []
    for i, route in enumerate(solution.routes):
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]

        # Đường nối: depot → nodes → depot
        xs = [depot.x] + [n.x for n in route.nodes] + [depot.x]
        ys = [depot.y] + [n.y for n in route.nodes] + [depot.y]

        ax_map.plot(xs, ys, "-", color=color, linewidth=1.8, alpha=0.7, zorder=2)

        # Mũi tên chỉ hướng di chuyển
        for j in range(len(xs) - 1):
            mx = (xs[j] + xs[j + 1]) / 2
            my = (ys[j] + ys[j + 1]) / 2
            dx = xs[j + 1] - xs[j]
            dy = ys[j + 1] - ys[j]
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                ax_map.annotate(
                    "", xy=(mx + dx * 0.01, my + dy * 0.01),
                    xytext=(mx - dx * 0.01, my - dy * 0.01),
                    arrowprops=dict(arrowstyle="->", color=color,
                                    lw=1.5, mutation_scale=12),
                    zorder=3,
                )

        # Pickup markers (tam giác lên)
        pickups = [n for n in route.nodes if n.is_pickup]
        if pickups:
            ax_map.scatter(
                [n.x for n in pickups], [n.y for n in pickups],
                c=color, marker="^", s=90, zorder=5,
                edgecolors="white", linewidths=0.5,
            )

        # Delivery markers (hình vuông)
        deliveries = [n for n in route.nodes if n.is_delivery]
        if deliveries:
            ax_map.scatter(
                [n.x for n in deliveries], [n.y for n in deliveries],
                c=color, marker="s", s=70, zorder=5,
                edgecolors="white", linewidths=0.5,
            )

        # Nhãn node ID
        for node in route.nodes:
            ax_map.annotate(
                str(node.id), (node.x, node.y),
                textcoords="offset points", xytext=(0, 6),
                color=color, fontsize=5.5, ha="center", fontweight="bold",
                zorder=7,
            )

        cost = route.total_cost()
        legend_handles.append(
            mpatches.Patch(
                color=color,
                label=f"Xe {route.vehicle.id}  ({len(route.nodes)} nodes, "
                      f"cost={cost:.1f})",
            )
        )

    # Depot
    ax_map.scatter(depot.x, depot.y, c="#ff5555", marker="*", s=600,
                   zorder=10, edgecolors="white", linewidths=0.8)
    ax_map.annotate("DEPOT", (depot.x, depot.y),
                    textcoords="offset points", xytext=(0, 12),
                    color="#ff5555", fontsize=9, ha="center",
                    fontweight="bold", zorder=11)

    # Legend (biểu đồ)
    legend_handles += [
        plt.scatter([], [], c="#aaa", marker="^", s=70,
                    label="△  Pickup  (demand > 0)"),
        plt.scatter([], [], c="#aaa", marker="s", s=60,
                    label="□  Delivery (demand < 0)"),
        plt.scatter([], [], c="#ff5555", marker="*", s=120,
                    label="★  Depot"),
    ]
    ax_map.legend(
        handles=legend_handles, loc="lower left", fontsize=7,
        facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3",
        framealpha=0.95,
    )

    # Tiêu đề biểu đồ
    ax_map.set_title(
        f"Lộ trình VRPPD — {strategy_name}\n"
        f"File: {file_name}  |  {num_routes} tuyến xe",
        color="#e6edf3", fontsize=13, fontweight="bold", pad=14,
    )

    # ── Bảng thông tin (phải) ──

    ax_info.set_facecolor("#161b22")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")

    info_items = [
        ("THÔNG TIN KẾT QUẢ", None, "#89b4fa", 14, "bold"),
        ("", None, None, 6, "normal"),
        ("Thuật toán", strategy_name, "#cdd6f4", 11, "normal"),
        ("File", file_name, "#cdd6f4", 11, "normal"),
        ("", None, None, 6, "normal"),
        ("─" * 28, None, "#30363d", 9, "normal"),
        ("", None, None, 4, "normal"),
        ("Tổng chi phí", f"{solution.total_cost:.2f}", "#50fa7b", 12, "bold"),
        ("Số tuyến xe", str(num_routes), "#cdd6f4", 11, "normal"),
        ("Tổng số node", str(len(nodes)), "#cdd6f4", 11, "normal"),
        ("Số yêu cầu", str(cov["required"] // 2), "#cdd6f4", 11, "normal"),
        ("Sức chứa xe", str(int(capacity)), "#cdd6f4", 11, "normal"),
        ("", None, None, 6, "normal"),
        ("─" * 28, None, "#30363d", 9, "normal"),
        ("", None, None, 4, "normal"),
        ("Feasible", "✓  Có" if feasible else "✗  Không",
         "#50fa7b" if feasible else "#ff5555", 11, "bold"),
        ("Complete", "✓  Có" if cov["complete"] else "✗  Không",
         "#50fa7b" if cov["complete"] else "#ff5555", 11, "bold"),
        ("Valid", "✓  Có" if valid else "✗  Không",
         "#50fa7b" if valid else "#ff5555", 12, "bold"),
        ("", None, None, 6, "normal"),
        ("Thiếu", str(cov["missing"]), "#cdd6f4", 10, "normal"),
        ("Trùng", str(cov["duplicates"]), "#cdd6f4", 10, "normal"),
        ("", None, None, 6, "normal"),
        ("─" * 28, None, "#30363d", 9, "normal"),
        ("", None, None, 4, "normal"),
        ("Thời gian", f"{elapsed_ms:.1f} ms", "#ffb86c", 11, "bold"),
    ]

    y_pos = 0.95
    for item in info_items:
        label, value, color, size, weight = item
        if value is None:
            # Tiêu đề / phân cách
            ax_info.text(0.5, y_pos, label, transform=ax_info.transAxes,
                         fontsize=size, fontweight=weight, color=color or "#cdd6f4",
                         ha="center", va="top", fontfamily="Segoe UI")
            y_pos -= 0.025 * (size / 10)
        else:
            ax_info.text(0.08, y_pos, label, transform=ax_info.transAxes,
                         fontsize=size - 1, color="#8b949e", ha="left", va="top",
                         fontfamily="Segoe UI")
            ax_info.text(0.92, y_pos, value, transform=ax_info.transAxes,
                         fontsize=size, fontweight=weight, color=color,
                         ha="right", va="top", fontfamily="Segoe UI")
            y_pos -= 0.033

    # ── Chi tiết từng tuyến (phần dưới bảng) ──

    if num_routes <= 15:
        y_pos -= 0.02
        ax_info.text(0.5, y_pos, "─" * 28, transform=ax_info.transAxes,
                     fontsize=9, color="#30363d", ha="center", va="top")
        y_pos -= 0.025
        ax_info.text(0.5, y_pos, "CHI TIẾT TUYẾN XE",
                     transform=ax_info.transAxes, fontsize=10,
                     fontweight="bold", color="#89b4fa", ha="center", va="top")
        y_pos -= 0.035

        for i, route in enumerate(solution.routes):
            if y_pos < 0.02:
                break
            color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
            report = route.feasibility_report()
            ok_tag = "✓" if report["feasible"] else "✗"
            text = (f"Xe {route.vehicle.id:>2}: {len(route.nodes)} nodes  "
                    f"cost={report['total_cost']:.1f}  {ok_tag}")
            ax_info.text(0.08, y_pos, text, transform=ax_info.transAxes,
                         fontsize=7.5, color=color, ha="left", va="top",
                         fontfamily="Consolas")
            y_pos -= 0.028

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    fig.suptitle(
        "VRPPD — Vehicle Routing Problem with Pickup and Delivery",
        color="#58a6ff", fontsize=11, fontweight="bold", y=0.98,
        fontfamily="Segoe UI",
    )

    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  VRPPD Route Visualizer")
    print("=" * 60)

    # 1. Mở dialog chọn file + thuật toán
    try:
        dialog = SelectorDialog()
    except FileNotFoundError as e:
        print(f"\n  ✗ {e}")
        return

    selection = dialog.run()

    if selection is None:
        print("\n  Đã hủy.")
        return

    filepath = selection["path"]
    strategy_name = selection["strategy"]
    filename = selection["file"]
    group = selection["group"]

    print(f"\n  File      : {group}/{filename}")
    print(f"  Thuật toán: {strategy_name}")

    # 2. Đọc dữ liệu
    print(f"\n  Đang đọc dữ liệu...")
    try:
        nodes, requests, vehicles, capacity = parse_input(filepath)
    except Exception as exc:
        print(f"  ✗ Lỗi đọc file: {exc}")
        return

    print(f"  Nodes    = {len(nodes)}")
    print(f"  Requests = {len(requests)} cặp pickup-delivery")
    print(f"  Vehicles = {len(vehicles)} (capacity = {capacity})")

    # 3. Chạy thuật toán
    _DIST.clear()
    strategy_fn = STRATEGIES[strategy_name]

    print(f"\n  Đang chạy {strategy_name}...")
    start = time.perf_counter()
    try:
        solution = strategy_fn(nodes, requests, vehicles, capacity)
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  ✗ Lỗi: {type(exc).__name__}: {exc}")
        print(f"  Thời gian: {elapsed:.1f} ms")
        return

    elapsed_ms = (time.perf_counter() - start) * 1000

    if solution is None or not solution.routes:
        print(f"  ✗ Không tìm được lời giải.")
        print(f"  Thời gian: {elapsed_ms:.1f} ms")
        return

    # 4. In kết quả tóm tắt
    cov = solution_coverage(nodes, solution)
    valid = solution.is_feasible() and cov["complete"]

    print(f"\n  ── Kết quả ──")
    print(f"  Tổng chi phí : {solution.total_cost:.2f}")
    print(f"  Số tuyến     : {len(solution.routes)}")
    print(f"  Valid         : {'✓' if valid else '✗'}")
    print(f"  Thời gian    : {elapsed_ms:.1f} ms")

    # 5. Trực quan hóa
    print(f"\n  Đang vẽ biểu đồ...")
    visualize_solution(
        solution=solution,
        nodes=nodes,
        strategy_name=strategy_name,
        file_name=f"{group}/{filename}",
        elapsed_ms=elapsed_ms,
        capacity=capacity,
    )

    print(f"\n  Hoàn thành.")


if __name__ == "__main__":
    main()
