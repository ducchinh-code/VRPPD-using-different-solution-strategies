import math


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class Node:
    """Đại diện cho một điểm trong bài toán VRPPD.

    Demand dương  → pickup  (lấy hàng)
    Demand âm     → delivery (giao hàng)
    Demand = 0    → depot
    """

    def __init__(self, node_id, x, y, demand, pickup_index, delivery_index):
        self.id             = node_id
        self.x              = x
        self.y              = y
        self.demand         = demand          # >0: pickup | <0: delivery | 0: depot
        self.pickup_index   = pickup_index    # ID node pickup tương ứng (0 nếu không có)
        self.delivery_index = delivery_index  # ID node delivery tương ứng (0 nếu không có)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def is_depot(self) -> bool:
        return self.demand == 0 and self.pickup_index == 0 and self.delivery_index == 0

    @property
    def is_pickup(self) -> bool:
        return self.demand > 0

    @property
    def is_delivery(self) -> bool:
        return self.demand < 0

    def distance_to(self, other: "Node") -> float:
        """Khoảng cách Euclidean đến node khác."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def __repr__(self):
        kind = "depot" if self.is_depot else ("pickup" if self.is_pickup else "delivery")
        return f"Node({self.id}, {kind}, demand={self.demand})"


# ---------------------------------------------------------------------------
# Route  —  lịch trình của một xe / tài xế
# ---------------------------------------------------------------------------
class Route:
    """Lịch trình phục vụ của một xe.

    nodes: danh sách Node theo thứ tự thăm (không bao gồm depot).
    depot: node xuất phát / kết thúc.
    capacity: tải trọng tối đa của xe.
    """

    def __init__(self, depot: Node, capacity: float):
        self.depot    = depot
        self.capacity = capacity
        self.nodes: list[Node] = []   # danh sách node theo thứ tự phục vụ

    # ------------------------------------------------------------------
    # Kiểm tra ràng buộc
    # ------------------------------------------------------------------
    def is_feasible(self) -> bool:
        """Kiểm tra tất cả ràng buộc: capacity và precedence."""
        return self._check_capacity() and self._check_precedence()

    def _check_capacity(self) -> bool:
        """Tải trọng xe không được âm hay vượt quá giới hạn tại bất kỳ điểm nào."""
        load = 0.0
        for node in self.nodes:
            load += node.demand
            if load < 0 or load > self.capacity:
                return False
        return True

    def _check_precedence(self) -> bool:
        """Mỗi pickup phải được thăm trước delivery tương ứng."""
        visited_ids = set()
        for node in self.nodes:
            if node.is_delivery:
                # pickup_index trỏ tới ID node pickup tương ứng
                if node.pickup_index not in visited_ids:
                    return False
            visited_ids.add(node.id)
        return True

    # ------------------------------------------------------------------
    # Chi phí
    # ------------------------------------------------------------------
    def total_distance(self) -> float:
        """Tổng quãng đường: depot → nodes → depot."""
        if not self.nodes:
            return 0.0
        dist = self.depot.distance_to(self.nodes[0])
        for i in range(len(self.nodes) - 1):
            dist += self.nodes[i].distance_to(self.nodes[i + 1])
        dist += self.nodes[-1].distance_to(self.depot)
        return dist

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        ids = " → ".join(str(n.id) for n in self.nodes)
        return f"Route(dist={self.total_distance():.2f}, nodes=[{ids}])"


# ---------------------------------------------------------------------------
# Solution  —  tập hợp các Route
# ---------------------------------------------------------------------------
class Solution:
    """Lời giải hoàn chỉnh cho bài toán VRPPD.

    drivers: danh sách các Route (mỗi Route = một xe/tài xế).
    total_cost: tổng chi phí = tổng quãng đường tất cả xe.
    """

    def __init__(self, drivers: list[Route] | None = None):
        self.drivers: list[Route] = drivers if drivers is not None else []

    # ------------------------------------------------------------------
    # Chi phí
    # ------------------------------------------------------------------
    @property
    def total_cost(self) -> float:
        return sum(r.total_distance() for r in self.drivers)

    # ------------------------------------------------------------------
    # Kiểm tra
    # ------------------------------------------------------------------
    def is_feasible(self) -> bool:
        return all(r.is_feasible() for r in self.drivers)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------
    def __str__(self):
        lines = []
        for i, route in enumerate(self.drivers):
            ids = " → ".join(str(n.id) for n in route.nodes)
            lines.append(
                f"  Driver {i+1:>2}: depot → {ids} → depot  "
                f"(dist={route.total_distance():.2f})"
            )
        return "\n".join(lines) if lines else "  (no routes)"

    def __repr__(self):
        return (f"Solution(drivers={len(self.drivers)}, "
                f"total_cost={self.total_cost:.2f})")
