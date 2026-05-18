"""
VRPDP Solver — Divide & Conquer with K-means Clustering
Dataset : lc101 (Li & Lim PDPTW Benchmark)

Cách chạy:
    python vrpdp_solver.py              # file lc101.txt cùng thư mục
    python vrpdp_solver.py path/to/lc101.txt

Yêu cầu:
    pip install numpy scikit-learn matplotlib
"""

import sys
import os
import math
import warnings
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')          # không cần màn hình, xuất file PNG
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# PHẦN 1: ĐỌC DỮ LIỆU
# ══════════════════════════════════════════════════════════════

def parse_lc101(filepath):
    """
    Đọc file Li & Lim PDPTW.
    Trả về: (num_vehicles, capacity, nodes_dict, pairs_list)
      - nodes_dict : {id: {...}}
      - pairs_list : [(pickup_id, delivery_id), ...]
    """
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]

    h            = lines[0].split()
    num_vehicles = int(h[0])
    capacity     = int(h[1])

    nodes = {}
    for line in lines[1:]:
        p   = line.split()
        nid = int(p[0])
        nodes[nid] = {
            'id'     : nid,
            'x'      : float(p[1]),
            'y'      : float(p[2]),
            'demand' : int(p[3]),
            'early'  : int(p[4]),
            'late'   : int(p[5]),
            'service': int(p[6]),
            'col7'   : int(p[7]),   # delivery→pickup partner
            'col8'   : int(p[8]),   # pickup→delivery partner
        }

    # demand>0 = pickup, col8 = delivery partner ID
    pairs = [
        (nid, n['col8'])
        for nid, n in nodes.items()
        if nid != 0 and n['demand'] > 0 and n['col8'] > 0
    ]
    return num_vehicles, capacity, nodes, pairs


# ══════════════════════════════════════════════════════════════
# PHẦN 2: TIỆN ÍCH
# ══════════════════════════════════════════════════════════════

def euclidean(a, b):
    return math.hypot(a['x'] - b['x'], a['y'] - b['y'])


def evaluate_route(route, nodes, capacity, depot_id=0):
    depot = nodes[depot_id]
    prev  = depot
    time  = 0.0
    load  = 0
    dist  = 0.0
    tw_v  = 0
    cap_v = 0

    for nid in route:
        n     = nodes[nid]
        d     = euclidean(prev, n)
        dist += d
        time += d
        if time < n['early']:
            time = n['early']
        if time > n['late']:
            tw_v += 1
        time += n['service']
        load += n['demand']
        if load < 0 or load > capacity:
            cap_v += 1
        prev = n

    dist += euclidean(prev, depot)
    return {
        'distance'      : dist,
        'tw_violations' : tw_v,
        'cap_violations': cap_v,
        'feasible'      : (tw_v == 0 and cap_v == 0),
    }


# ══════════════════════════════════════════════════════════════
# PHẦN 3: [DIVIDE] K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════

def divide_kmeans(pairs, nodes, k):
    """
    Feature: trung điểm (midpoint) của mỗi cặp pickup–delivery.
    Các cặp gần nhau về địa lý được gom vào cùng cụm.
    """
    X = np.array([
        [(nodes[pid]['x'] + nodes[did]['x']) / 2,
         (nodes[pid]['y'] + nodes[did]['y']) / 2]
        for pid, did in pairs
    ])

    k      = min(k, len(pairs))
    km     = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X)

    clusters = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[int(lbl)].append(pairs[i])

    return dict(clusters), km.cluster_centers_, labels


# ══════════════════════════════════════════════════════════════
# PHẦN 4: [CONQUER] GREEDY NEAREST-NEIGHBOR
# ══════════════════════════════════════════════════════════════

def build_routes_greedy(cluster_pairs, nodes, capacity, depot_id=0):
    """
    Greedy Nearest-Neighbor cho PDPTW:
      - Ưu tiên nút (pickup hoặc delivery) gần nhất thỏa mãn ràng buộc.
      - Pickup phải đến trước delivery của cùng cặp.
      - Không vượt quá capacity tại bất kỳ thời điểm nào.
      - Đến trong cửa sổ thời gian [early, late].
    """
    routes    = []
    remaining = list(cluster_pairs)

    while remaining:
        route      = []
        load       = 0
        time       = 0.0
        cur        = nodes[depot_id]
        in_transit = []          # đã pickup, chưa delivery

        while True:
            best_nid = None
            best_d   = float('inf')
            best_act = None      # ('pickup', pair) | ('delivery', pair)

            # Thử thêm pickup mới
            for pair in remaining:
                pid, _ = pair
                pn     = nodes[pid]
                d      = euclidean(cur, pn)
                arr    = max(time + d, pn['early'])
                if arr <= pn['late'] and load + pn['demand'] <= capacity:
                    if d < best_d:
                        best_d, best_nid, best_act = d, pid, ('pickup', pair)

            # Thử giao hàng đang trên xe
            for pair in in_transit:
                _, did = pair
                dn     = nodes[did]
                d      = euclidean(cur, dn)
                arr    = max(time + d, dn['early'])
                if arr <= dn['late']:
                    if d < best_d:
                        best_d, best_nid, best_act = d, did, ('delivery', pair)

            if best_act is None:
                break

            act_type, pair = best_act
            node = nodes[best_nid]
            d    = euclidean(cur, node)
            arr  = max(time + d, node['early'])
            time = arr + node['service']
            load += node['demand']
            route.append(best_nid)
            cur = node

            if act_type == 'pickup':
                in_transit.append(pair)
                remaining.remove(pair)
            else:
                in_transit.remove(pair)

        # Ép giao hàng còn lại (có thể vi phạm TW)
        for _, did in in_transit:
            route.append(did)

        if route:
            routes.append(route)
        elif remaining:
            pid, did = remaining.pop(0)
            routes.append([pid, did])

    return routes


# ══════════════════════════════════════════════════════════════
# PHẦN 5: [IMPROVE] 2-OPT LOCAL SEARCH
# ══════════════════════════════════════════════════════════════

def check_precedence(route, nodes):
    """Pickup phải xuất hiện trước delivery tương ứng."""
    pos = {nid: i for i, nid in enumerate(route)}
    for nid in route:
        n = nodes[nid]
        if n['demand'] > 0 and n['col8'] > 0:
            if n['col8'] in pos and pos[nid] > pos[n['col8']]:
                return False
    return True


def two_opt(route, nodes, capacity):
    """2-opt cải thiện tuyến, giữ ràng buộc thứ tự P→D."""
    best    = route[:]
    best_ev = evaluate_route(best, nodes, capacity)
    changed = True

    while changed:
        changed = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                candidate = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if not check_precedence(candidate, nodes):
                    continue
                ev = evaluate_route(candidate, nodes, capacity)
                if ev['feasible'] and ev['distance'] < best_ev['distance']:
                    best, best_ev = candidate, ev
                    changed = True

    return best


# ══════════════════════════════════════════════════════════════
# PHẦN 6: VISUALIZE
# ══════════════════════════════════════════════════════════════

CLUSTER_COLORS = [
    '#4ec9b0','#f48c42','#c586c0','#9cdcfe',
    '#ce9178','#dcdcaa','#6a9955','#4fc1ff',
]


def visualize(nodes, clusters, all_routes, k, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.patch.set_facecolor('#0f1117')

    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.grid(True, color='#21262d', linewidth=0.8, zorder=0)

    # Panel trái: Phân cụm
    ax = axes[0]
    ax.set_title(f'BƯỚC 1 — Phân cụm K-means  (K = {k})',
                 color='#e6edf3', fontsize=13, fontweight='bold', pad=14)

    for cid, cpairs in clusters.items():
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        for pid, did in cpairs:
            px, py = nodes[pid]['x'], nodes[pid]['y']
            dx, dy = nodes[did]['x'], nodes[did]['y']
            ax.plot([px, dx], [py, dy], '-', color=color, alpha=0.3, lw=1.2, zorder=1)
            ax.scatter(px, py, c=color, marker='^', s=110, zorder=5,
                       edgecolors='#161b22', linewidths=0.6)
            ax.scatter(dx, dy, c=color, marker='s', s=85, zorder=5,
                       edgecolors='#161b22', linewidths=0.6)
            ax.annotate(str(pid), (px, py), color=color,
                        fontsize=5, ha='center', va='bottom', zorder=6)
            ax.annotate(str(did), (dx, dy), color=color,
                        fontsize=5, ha='center', va='top',    zorder=6)

    ax.scatter(nodes[0]['x'], nodes[0]['y'],
               c='#ff5555', marker='*', s=500, zorder=10)
    ax.annotate('DEPOT', (nodes[0]['x'], nodes[0]['y'] + 1.5),
                color='#ff5555', fontsize=8, ha='center', fontweight='bold')

    handles = [mpatches.Patch(color=CLUSTER_COLORS[i],
               label=f'Cụm {i+1}  ({len(clusters[i])} cặp)')
               for i in sorted(clusters)] + [
        plt.scatter([], [], c='#aaa', marker='^', s=80, label='Pickup'),
        plt.scatter([], [], c='#aaa', marker='s', s=70, label='Delivery'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=7.5,
              facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')
    ax.set_xlabel('X'); ax.set_ylabel('Y')

    # Panel phải: Tuyến đường
    ax = axes[1]
    ax.set_title(f'BƯỚC 2 — Tuyến đường VRPDP  ({len(all_routes)} tuyến)',
                 color='#e6edf3', fontsize=13, fontweight='bold', pad=14)

    cmap = plt.colormaps['tab20'].resampled(max(len(all_routes), 1))
    dx0, dy0 = nodes[0]['x'], nodes[0]['y']

    for i, route in enumerate(all_routes):
        color = cmap(i)
        xs = [dx0] + [nodes[n]['x'] for n in route] + [dx0]
        ys = [dy0] + [nodes[n]['y'] for n in route] + [dy0]
        ax.plot(xs, ys, '-o', color=color,
                markersize=4, linewidth=1.6, alpha=0.85, zorder=3)
        for nid in route:
            ax.annotate(str(nid), (nodes[nid]['x'], nodes[nid]['y'] + 0.8),
                        color='#e6edf3', fontsize=5, ha='center', zorder=7)

    ax.scatter(dx0, dy0, c='#ff5555', marker='*', s=500, zorder=10)
    ax.annotate('DEPOT', (dx0, dy0 + 1.5),
                color='#ff5555', fontsize=8, ha='center', fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y')

    fig.suptitle(
        'VRPDP — Divide & Conquer with K-means\n'
        'Benchmark LC101  |  Li & Lim PDPTW',
        color='#e6edf3', fontsize=15, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"  → Biểu đồ: {save_path}")


# ══════════════════════════════════════════════════════════════
# PHẦN 7: MAIN
# ══════════════════════════════════════════════════════════════

def main():
    # ── Lấy đường dẫn file ──────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) >= 2:
        target = sys.argv[1]
    else:
        target = script_dir   # mặc định: thư mục hiện tại

    # Nếu là folder → liệt kê file cho chọn
    if os.path.isdir(target):
        txt_files = sorted([
            f for f in os.listdir(target)
            if f.endswith('.txt')
        ])
        if not txt_files:
            print(f"[LỖI] Không có file .txt nào trong: {target}")
            sys.exit(1)

        print(f"\nDanh sách file trong '{target}':")
        for i, fname in enumerate(txt_files):
            print(f"  [{i+1}] {fname}")

        choice = input("\nChọn số thứ tự file muốn chạy: ").strip()
        try:
            filepath = os.path.join(target, txt_files[int(choice) - 1])
        except (ValueError, IndexError):
            print("[LỖI] Lựa chọn không hợp lệ.")
            sys.exit(1)

    elif os.path.isfile(target):
        filepath = target
    else:
        print(f"[LỖI] Không tìm thấy file hoặc folder: {target}")
        sys.exit(1)

    K        = 5      # Số cụm K-means (thay đổi tuỳ ý)
    USE_2OPT = True   # Bật/tắt cải thiện 2-opt

    SEP = "═" * 65
    print(SEP)
    print("  VRPDP — Divide & Conquer with K-means  |  LC101")
    print(SEP)

    # 1. Đọc dữ liệu
    num_vehicles, capacity, nodes, pairs = parse_lc101(filepath)
    print(f"\n📦  DỮ LIỆU")
    print(f"    File             : {os.path.basename(filepath)}")
    print(f"    Số xe tối đa     : {num_vehicles}")
    print(f"    Sức chứa xe      : {capacity}")
    print(f"    Số nút KH        : {len(nodes)-1}  (+ 1 depot)")
    print(f"    Số cặp yêu cầu   : {len(pairs)}")

    # 2. DIVIDE
    print(f"\n🔀  [DIVIDE] K-MEANS  (K={K})")
    clusters, centers, labels = divide_kmeans(pairs, nodes, K)
    for cid in sorted(clusters):
        print(f"    Cụm {cid+1}: {len(clusters[cid]):2d} cặp  | "
              f"tâm=({centers[cid][0]:.1f}, {centers[cid][1]:.1f})")

    # 3. CONQUER + 2-OPT
    print(f"\n🚛  [CONQUER] XÂY DỰNG TUYẾN")
    all_routes = []

    for cid in sorted(clusters):
        cpairs = clusters[cid]
        routes = build_routes_greedy(cpairs, nodes, capacity)
        if USE_2OPT:
            routes = [two_opt(r, nodes, capacity) for r in routes]

        print(f"\n    ┌─ Cụm {cid+1}  ({len(cpairs)} cặp → {len(routes)} tuyến)")
        for r in routes:
            ev  = evaluate_route(r, nodes, capacity)
            idx = len(all_routes) + 1
            ok  = "✓ KHẢ THI" if ev['feasible'] else \
                  f"⚠ TW={ev['tw_violations']} CAP={ev['cap_violations']}"
            seq = '→'.join(map(str, r))
            print(f"    │  Tuyến {idx:2d}: 0→{seq}→0")
            print(f"    │           d={ev['distance']:.2f}  {ok}")
            all_routes.append(r)
        print(f"    └{'─'*50}")

    # 4. Tổng kết
    evals      = [evaluate_route(r, nodes, capacity) for r in all_routes]
    total_dist = sum(e['distance']       for e in evals)
    n_feasible = sum(1 for e in evals if e['feasible'])
    total_tw   = sum(e['tw_violations']  for e in evals)
    total_cap  = sum(e['cap_violations'] for e in evals)
    dists      = [e['distance'] for e in evals]

    print(f"\n{SEP}")
    print(f"  📊  KẾT QUẢ")
    print(SEP)
    print(f"    Tổng số tuyến        : {len(all_routes)}")
    print(f"    Tuyến khả thi        : {n_feasible} / {len(all_routes)}")
    print(f"    Tổng khoảng cách     : {total_dist:.2f}")
    print(f"    Vi phạm cửa sổ TG    : {total_tw}")
    print(f"    Vi phạm tải trọng    : {total_cap}")
    print(f"    Tuyến ngắn / dài nhất: {min(dists):.2f}  /  {max(dists):.2f}")
    print(f"    Trung bình/tuyến     : {total_dist/len(all_routes):.2f}")

    # Bảng chi tiết
    print(f"\n  {'Tuyến':>6} | {'Khoảng cách':>12} | {'TW vi phạm':>10} | "
          f"{'CAP vi phạm':>11} | {'# Nút':>6} | Khả thi?")
    print(f"  {'─'*6}-+-{'─'*12}-+-{'─'*10}-+-{'─'*11}-+-{'─'*6}-+-{'─'*8}")
    for i, (r, ev) in enumerate(zip(all_routes, evals)):
        ok = "✓" if ev['feasible'] else "✗"
        print(f"  {i+1:>6} | {ev['distance']:>12.2f} | {ev['tw_violations']:>10} | "
              f"{ev['cap_violations']:>11} | {len(r):>6} | {ok}")

    # 5. Vẽ biểu đồ
    out_img = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'vrpdp_solution.png')
    print(f"\n🎨  VẼ BIỂU ĐỒ")
    visualize(nodes, clusters, all_routes, K, out_img)
    print(f"\n{SEP}")
    print("  Xong!")
    print(SEP)


if __name__ == '__main__':
    main()