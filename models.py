import math

# Shared Euclidean distance cache — avoids redundant math.hypot calls
_DIST: dict[tuple[int, int], float] = {}

def cdist(a: "Node", b: "Node") -> float:
    k = (a.id, b.id)
    d = _DIST.get(k)
    if d is None:
        d = math.hypot(a.x - b.x, a.y - b.y)
        _DIST[k] = d
    return d

class Node:
    def __init__(self, node_id: int, x: float, y: float,
                 demand: float,
                 pickup_index: int, delivery_index: int):
        self.id             = node_id
        self.x              = x
        self.y              = y
        self.demand         = demand
        self.pickup_index   = pickup_index
        self.delivery_index = delivery_index

    @property
    def is_depot(self) -> bool:
        return self.id == 0

    @property
    def is_pickup(self) -> bool:
        return self.demand > 0

    @property
    def is_delivery(self) -> bool:
        return self.demand < 0

    def distance_to(self, other: "Node") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __repr__(self):
        kind = "depot" if self.is_depot else ("pickup" if self.is_pickup else "delivery")
        return f"Node(id={self.id}, {kind}, q={self.demand:+g})"

class Request:
    def __init__(self, pickup: Node, delivery: Node):
        self.pickup   = pickup
        self.delivery = delivery

    def __repr__(self):
        return (f"Request(p={self.pickup.id} [q={self.pickup.demand:+g}], "
                f"d={self.delivery.id} [q={self.delivery.demand:+g}])")

class Vehicle:
    def __init__(self, vehicle_id: int, capacity: float):
        self.id       = vehicle_id
        self.capacity = capacity

    def __repr__(self):
        return f"Vehicle(id={self.id}, Q={self.capacity})"

class Route:
    def __init__(self, vehicle: Vehicle, depot: Node):
        self.vehicle = vehicle
        self.depot   = depot
        self.nodes: list[Node] = []

    def append(self, node: Node) -> None:
        self.nodes.append(node)

    def compute_U(self) -> dict[int, int]:
        U = {self.depot.id: 0}
        for order, node in enumerate(self.nodes, start=1):
            U[node.id] = order
        return U

    def total_cost(self) -> float:
        if not self.nodes:
            return 0.0
        cost = cdist(self.depot, self.nodes[0])
        for i in range(len(self.nodes) - 1):
            cost += cdist(self.nodes[i], self.nodes[i + 1])
        cost += cdist(self.nodes[-1], self.depot)
        return cost

    def is_feasible(self) -> bool:
        return self._check_precedence_U() and self._check_capacity()

    def _check_capacity(self) -> bool:
        load = 0.0
        for node in self.nodes:
            load += node.demand
            if load < 0 or load > self.vehicle.capacity:
                return False
        return True

    def _check_precedence_U(self) -> bool:
        U = self.compute_U()
        for node in self.nodes:
            if node.is_delivery:
                p_id = node.pickup_index
                if p_id not in U:
                    return False
                if not (U[p_id] < U[node.id]):
                    return False
        return True

    def feasibility_report(self) -> dict:
        U   = self.compute_U()
        cap_violations  = []
        prec_violations = []
        load = 0.0

        for node in self.nodes:
            load += node.demand
            if load < 0 or load > self.vehicle.capacity:
                cap_violations.append((node.id, round(load, 4)))
            if node.is_delivery:
                p_id = node.pickup_index
                if p_id not in U or not (U[p_id] < U[node.id]):
                    prec_violations.append((p_id, node.id))

        return {
            "feasible"        : not cap_violations and not prec_violations,
            "cap_violations"  : cap_violations,
            "prec_violations" : prec_violations,
            "U"               : U,
            "total_cost"      : self.total_cost(),
            "num_nodes"       : len(self.nodes),
        }

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        ids = " → ".join(str(n.id) for n in self.nodes)
        return (f"Route(vehicle={self.vehicle.id}, "
                f"cost={self.total_cost():.2f}, "
                f"nodes=[{ids}])")

class Solution:
    def __init__(self, routes: list[Route] | None = None):
        self.routes: list[Route] = routes if routes is not None else []

    @property
    def total_cost(self) -> float:
        return sum(r.total_cost() for r in self.routes)

    def is_feasible(self) -> bool:
        return all(r.is_feasible() for r in self.routes)

    def __str__(self):
        lines = []
        for route in self.routes:
            U   = route.compute_U()
            ids = " → ".join(
                f"{n.id}(U={U[n.id]})" for n in route.nodes
            )
            tag = "✓" if route.is_feasible() else "✗"
            lines.append(
                f"  Xe {route.vehicle.id:>2} [Q={route.vehicle.capacity}]: "
                f"0(U=0) → {ids} → 0  "
                f"cost={route.total_cost():.2f}  {tag}"
            )
        header = (f"Solution | total_cost={self.total_cost:.2f} | "
                  f"vehicles={len(self.routes)}")
        return header + "\n" + ("\n".join(lines) if lines else "  (no routes)")

    def __repr__(self):
        return (f"Solution(vehicles={len(self.routes)}, "
                f"total_cost={self.total_cost:.2f}, "
                f"feasible={self.is_feasible()})")
