from utils import parse_input


def main():
    file_path = "data/pdp_100/lc101.txt"
    nodes, num_vehicles = parse_input(file_path)

    print(f"File        : {file_path}")
    print(f"Num vehicles: {num_vehicles}")
    print(f"Total nodes : {len(nodes)}")
    print()

    depot = nodes[0]
    print(f"Depot → {depot}")
    print()

    pickups   = [n for n in nodes if n.is_pickup]
    deliveries = [n for n in nodes if n.is_delivery]
    print(f"Pickups    : {len(pickups)}")
    print(f"Deliveries : {len(deliveries)}")
    print()

    print(f"{'ID':>4} {'X':>6} {'Y':>6} {'Demand':>8} {'Pickup':>8} {'Delivery':>10}  Type")
    print("-" * 58)
    for n in nodes:
        kind = "depot" if n.is_depot else ("pickup" if n.is_pickup else "delivery")
        print(f"{n.id:>4} {n.x:>6.1f} {n.y:>6.1f} {n.demand:>8.1f} "
              f"{n.pickup_index:>8} {n.delivery_index:>10}  {kind}")


if __name__ == "__main__":
    main()