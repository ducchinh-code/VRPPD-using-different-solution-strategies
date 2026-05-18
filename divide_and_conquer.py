import warnings
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from models import Node, Request, Vehicle, Route, Solution
from utils  import parse_input

warnings.filterwarnings("ignore")

def travel_cost(a: Node, b: Node) -> float:
    return a.distance_to(b)

def evaluate_route(route: Route) -> dict:
    return route.feasibility_report()

def divide_kmeans(
    requests: list[Request],
    k: int
) -> tuple[dict[int, list[Request]], np.ndarray, np.ndarray]:
    X = np.array([
        [
            (r.pickup.x + r.delivery.x) / 2,
            (r.pickup.y + r.delivery.y) / 2,
        ]
        for r in requests
    ])

    k  = min(k, len(requests))
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X)

    clusters: dict[int, list[Request]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[int(lbl)].append(requests[i])

    return dict(clusters), km.cluster_centers_, labels

def build_routes_greedy(
    cluster_requests: list[Request],
    vehicles: list[Vehicle],
    depot: Node,
    vehicle_idx_start: int = 0
) -> list[Route]:

    routes: list[Route] = []
    remaining = list(cluster_requests)
    v_idx = vehicle_idx_start

    while remaining:
        if v_idx >= len(vehicles):
            v_idx = len(vehicles) - 1

        vehicle = vehicles[v_idx]
        route   = Route(vehicle=vehicle, depot=depot)
        v_idx  += 1

        load        = 0.0
        cur         = depot
        in_transit: list[Request] = []

        while True:
            best_node   = None
            best_dist   = float('inf')
            best_action = None

            for req in remaining:
                p    = req.pickup
                d    = travel_cost(cur, p)
                if load + p.demand <= vehicle.capacity:
                    if d < best_dist:
                        best_dist, best_node, best_action = d, p, ('pickup', req)

            for req in in_transit:
                d_node = req.delivery
                d      = travel_cost(cur, d_node)
                if d < best_dist:
                    best_dist, best_node, best_action = d, d_node, ('delivery', req)

            if best_action is None:
                break

            act_type, req = best_action
            load += best_node.demand
            route.append(best_node)
            cur = best_node

            if act_type == 'pickup':
                in_transit.append(req)
                remaining.remove(req)
            else:
                in_transit.remove(req)

        for req in in_transit:
            route.append(req.delivery)

        if route.nodes:
            routes.append(route)
        elif remaining:
            req = remaining.pop(0)
            r   = Route(vehicle=vehicle, depot=depot)
            r.append(req.pickup)
            r.append(req.delivery)
            routes.append(r)
    return routes

def check_precedence_list(node_list: list[Node]) -> bool:
    pos = {n.id: i for i, n in enumerate(node_list)}
    for node in node_list:
        if node.is_delivery:
            p_id = node.pickup_index
            if p_id not in pos or pos[p_id] >= pos[node.id]:
                return False
    return True

def check_capacity_list(node_list: list[Node], capacity: float) -> bool:
    load = 0.0
    for node in node_list:
        load += node.demand
        if load < 0 or load > capacity:
            return False
    return True

def two_opt(route: Route) -> Route:
    best_nodes = route.nodes[:]
    best_cost  = route.total_cost()
    changed    = True
    while changed:
        changed = False
        n = len(best_nodes)
        for i in range(n - 1):
            for j in range(i + 2, n):
                candidate = best_nodes[:i] + best_nodes[i:j+1][::-1] + best_nodes[j+1:]
                if not check_precedence_list(candidate):
                    continue
                if not check_capacity_list(candidate, route.vehicle.capacity):
                    continue
                depot = route.depot
                c_cost = depot.distance_to(candidate[0])
                for k in range(len(candidate) - 1):
                    c_cost += candidate[k].distance_to(candidate[k+1])
                c_cost += candidate[-1].distance_to(depot)
                if c_cost < best_cost - 1e-9:
                    best_nodes, best_cost = candidate, c_cost
                    changed = True
    new_route = Route(vehicle=route.vehicle, depot=route.depot)
    new_route.nodes = best_nodes
    return new_route

CLUSTER_COLORS = [
    '#4ec9b0', '#f48c42', '#c586c0', '#9cdcfe',
    '#ce9178', '#dcdcaa', '#6a9955', '#4fc1ff',
    '#ff79c6', '#50fa7b', '#ffb86c', '#bd93f9',
]

def visualize(
    nodes_dict: dict[int, Node],
    clusters: dict[int, list[Request]],
    solution: Solution,
    k: int,
    save_path: str
) -> None:
    depot = nodes_dict[0]
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
    ax = axes[0]
    ax.set_title(f'BƯỚC 1 — Phân cụm K-means  (K = {k})',
                 color='#e6edf3', fontsize=13, fontweight='bold', pad=14)
    for cid, reqs in clusters.items():
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        for req in reqs:
            p, d = req.pickup, req.delivery
            ax.plot([p.x, d.x], [p.y, d.y], '-', color=color,
                    alpha=0.3, lw=1.2, zorder=1)
            ax.scatter(p.x, p.y, c=color, marker='^', s=110, zorder=5,
                       edgecolors='#161b22', linewidths=0.6)
            ax.scatter(d.x, d.y, c=color, marker='s', s=85, zorder=5,
                       edgecolors='#161b22', linewidths=0.6)
            ax.annotate(str(p.id), (p.x, p.y), color=color,
                        fontsize=5, ha='center', va='bottom', zorder=6)
            ax.annotate(str(d.id), (d.x, d.y), color=color,
                        fontsize=5, ha='center', va='top', zorder=6)
    ax.scatter(depot.x, depot.y, c='#ff5555', marker='*', s=500, zorder=10)
    ax.annotate('DEPOT', (depot.x, depot.y + 1.5),
                color='#ff5555', fontsize=8, ha='center', fontweight='bold')
    handles = [
        mpatches.Patch(color=CLUSTER_COLORS[i],
                       label=f'Cụm {i+1}  ({len(clusters[i])} cặp)')
        for i in sorted(clusters)
    ] + [
        plt.scatter([], [], c='#aaa', marker='^', s=80, label='Pickup  (qᵢ > 0)'),
        plt.scatter([], [], c='#aaa', marker='s', s=70, label='Delivery (qᵢ < 0)'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=7.5,
              facecolor='#21262d', edgecolor='#30363d', labelcolor='#e6edf3')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax = axes[1]
    ax.set_title(
        f'BƯỚC 2 — Tuyến đường VRPPD  ({len(solution.routes)} tuyến)',
        color='#e6edf3', fontsize=13, fontweight='bold', pad=14
    )
    cmap = plt.colormaps['tab20'].resampled(max(len(solution.routes), 1))

    for i, route in enumerate(solution.routes):
        color = cmap(i)
        xs = [depot.x] + [n.x for n in route.nodes] + [depot.x]
        ys = [depot.y] + [n.y for n in route.nodes] + [depot.y]
        ax.plot(xs, ys, '-o', color=color,
                markersize=4, linewidth=1.6, alpha=0.85, zorder=3)
        for node in route.nodes:
            marker = '^' if node.is_pickup else 's'
            ax.scatter(node.x, node.y, c=[color], marker=marker,
                       s=60, zorder=5, edgecolors='white', linewidths=0.4)
            ax.annotate(str(node.id), (node.x, node.y + 0.8),
                        color='#e6edf3', fontsize=5, ha='center', zorder=7)

    ax.scatter(depot.x, depot.y, c='#ff5555', marker='*', s=500, zorder=10)
    ax.annotate('DEPOT', (depot.x, depot.y + 1.5),
                color='#ff5555', fontsize=8, ha='center', fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y')

    fig.suptitle(
        'VRPPD — Divide & Conquer with K-means\n'
        'Benchmark LC101  |  Li & Lim (không có Time Window)',
        color='#e6edf3', fontsize=15, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"  → Biểu đồ lưu tại: {save_path}")
