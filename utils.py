from models import Node

DEPOT       = None
def parse_input(file_path: str):
    global DEPOT
    nodes = []

    with open(file_path) as f:
        lines = f.readlines()

    first        = lines[0].split()
    num_vehicles = int(first[0])

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
            delivery_index = int(parts[8])
        )
        nodes.append(node)

    DEPOT = nodes[0]

    return nodes, num_vehicles