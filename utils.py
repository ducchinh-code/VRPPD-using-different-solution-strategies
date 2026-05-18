from models import Node, Request, Vehicle

def parse_input(file_path: str) -> tuple[list[Node], list[Request], list[Vehicle], int]:
    with open(file_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    header       = lines[0].split()
    num_vehicles = int(header[0])
    capacity     = int(header[1])

    nodes: list[Node] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 9:
            continue
        node = Node(
            node_id        = int(parts[0]),
            x              = float(parts[1]),
            y              = float(parts[2]),
            demand         = float(parts[3]),
            pickup_index   = int(parts[7]),
            delivery_index = int(parts[8]),
        )
        nodes.append(node)
    assert nodes[0].id == 0 and nodes[0].demand == 0, \
        "Node đầu tiên phải là depot với demand = 0"
    node_by_id: dict[int, Node] = {n.id: n for n in nodes}
    requests: list[Request] = []
    for node in nodes:
        if node.is_pickup and node.delivery_index > 0:
            pickup   = node
            delivery = node_by_id.get(node.delivery_index)
            if delivery is not None and delivery.is_delivery:
                requests.append(Request(pickup, delivery))
    vehicles: list[Vehicle] = [
        Vehicle(vehicle_id=k + 1, capacity=capacity)
        for k in range(num_vehicles)
    ]

    return nodes, requests, vehicles, capacity