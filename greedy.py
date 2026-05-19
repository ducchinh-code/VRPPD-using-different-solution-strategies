from models import Node, Request, Vehicle, Route, Solution

def _route_load(route: Route) -> float:
    return sum(n.demand for n in route.nodes)

def _can_append_pair(route: Route, req: Request) -> bool:
    cap  = route.vehicle.capacity
    load = _route_load(route)

    load_after_pickup = load + req.pickup.demand
    if load_after_pickup < 0 or load_after_pickup > cap:
        return False

    load_after_delivery = load_after_pickup + req.delivery.demand
    if load_after_delivery < 0 or load_after_delivery > cap:
        return False

    return True


def _delta_cost_append(route: Route, req: Request) -> float:
    depot = route.depot
    p     = req.pickup
    d     = req.delivery

    last = route.nodes[-1] if route.nodes else depot

    cost_new = (last.distance_to(p)
                + p.distance_to(d)
                + d.distance_to(depot))

    cost_old = last.distance_to(depot)

    return cost_new - cost_old

def solve_greedy(nodes: list[Node],
                 requests: list[Request],
                 vehicles: list[Vehicle]) -> Solution:

    depot            = nodes[0]
    pending          = list(requests)
    routes: list[Route] = []

    for vehicle in vehicles:
        if not pending:
            break

        route = Route(vehicle=vehicle, depot=depot)

        while True:
            best_req   = None
            best_delta = float("inf")

            for req in pending:
                if _can_append_pair(route, req):
                    delta = _delta_cost_append(route, req)
                    if delta < best_delta:
                        best_delta = delta
                        best_req   = req

            if best_req is None:
                break  
            route.append(best_req.pickup)
            route.append(best_req.delivery)
            pending.remove(best_req)

        if route.nodes:
            routes.append(route)

    return Solution(routes=routes)
