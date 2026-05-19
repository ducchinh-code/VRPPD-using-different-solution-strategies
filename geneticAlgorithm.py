from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from models import Node, Request, Vehicle, Route, Solution, cdist

POPULATION_SIZE   = 50
MAX_GENERATIONS   = 200
NO_IMPROVE_LIMIT  = 40
ELITE_RATIO       = 0.10
TOURNAMENT_K      = 3
CROSSOVER_PROB    = 0.80
MUTATION_PROB     = 0.30
LOCAL_SEARCH_PROB = 0.20

class PenaltyManager:
    def __init__(self, cap: float = 10.0, prec: float = 10.0,
                 target_rate: float = 0.20):
        self.cap   = cap
        self.prec  = prec
        self.target = target_rate

    def update(self, population: list[Individual]) -> None:
        infeasible = sum(
            1 for ind in population
            if ind.solution is not None and sum(_penalty_raw(ind.solution)) > 0
        )
        rate   = infeasible / max(len(population), 1)
        factor = 1.2 if rate > self.target else 0.85
        self.cap  = max(1.0, min(self.cap  * factor, 1000.0))
        self.prec = max(1.0, min(self.prec * factor, 1000.0))


def _penalty_raw(sol: Solution) -> tuple[float, int]:
    cap_viol  = 0.0
    prec_viol = 0
    for route in sol.routes:
        load = 0.0
        pos: dict[int, int] = {}
        for i, node in enumerate(route.nodes):
            load += node.demand
            pos[node.id] = i
            if load < 0 or load > route.vehicle.capacity:
                cap_viol += max(load - route.vehicle.capacity, -load)
            if node.is_delivery and node.pickup_index not in pos:
                prec_viol += 1
    return cap_viol, prec_viol


def route_cost(nodes: list[Node], depot: Node) -> float:
    if not nodes:
        return 0.0
    cost = cdist(depot, nodes[0])
    for i in range(len(nodes) - 1):
        cost += cdist(nodes[i], nodes[i + 1])
    cost += cdist(nodes[-1], depot)
    return cost

def is_precedence_ok(nodes: list[Node]) -> bool:
    pos = {n.id: i for i, n in enumerate(nodes)}
    for n in nodes:
        if n.is_delivery:
            if n.pickup_index not in pos or pos[n.pickup_index] >= pos[n.id]:
                return False
    return True

def is_capacity_ok(nodes: list[Node], capacity: float) -> bool:
    load = 0.0
    for n in nodes:
        load += n.demand
        if load < 0 or load > capacity:
            return False
    return True

@dataclass
class Individual:
    chromosome: list[int]
    fitness:    float = field(default=float("inf"))
    solution:   Optional[Solution] = field(default=None, repr=False)

    def copy(self) -> "Individual":
        c = Individual(chromosome=self.chromosome[:], fitness=self.fitness)
        c.solution = self.solution
        return c

def repair(chromosome: list[int], requests: list[Request]) -> list[int]:
    required: set[int] = set()
    for r in requests:
        required.add(r.pickup.id)
        required.add(r.delivery.id)

    seen:  set[int]  = set()
    chrom: list[int] = []
    for nid in chromosome:
        if nid in required and nid not in seen:
            seen.add(nid)
            chrom.append(nid)
    for nid in required - seen:
        chrom.append(nid)

    changed = True
    while changed:
        changed = False
        pos = {nid: i for i, nid in enumerate(chrom)}
        for r in requests:
            pid, did = r.pickup.id, r.delivery.id
            if pos[pid] > pos[did]:
                chrom.remove(pid)
                idx = chrom.index(did)
                chrom.insert(idx, pid)
                pos = {nid: i for i, nid in enumerate(chrom)}
                changed = True
    return chrom

def decode(chromosome: list[int], node_by_id: dict[int, Node],
           vehicles: list[Vehicle], depot: Node, capacity: float,
           max_nodes_per_route: int = 0) -> Solution:

    routes:    list[Route] = []
    v_iter     = iter(vehicles)
    vehicle    = next(v_iter)
    cur_nodes: list[Node] = []
    load       = 0.0
    placed:    set[int] = set()
    _extra_v_count = 0

    def _next_vehicle() -> Vehicle:
        nonlocal _extra_v_count
        v = next(v_iter, None)
        if v is None:
            _extra_v_count += 1
            v = Vehicle(
                vehicle_id=-(1000 + _extra_v_count),
                capacity=capacity,
            )
        return v

    def flush():
        nonlocal vehicle, cur_nodes, load
        if cur_nodes:
            r = Route(vehicle=vehicle, depot=depot)
            r.nodes = cur_nodes[:]
            routes.append(r)
        vehicle   = _next_vehicle()
        cur_nodes = []
        load      = 0.0

    for nid in chromosome:
        if nid in placed:
            continue
        node     = node_by_id[nid]
        new_load = load + node.demand

        route_full = max_nodes_per_route > 0 and len(cur_nodes) >= max_nodes_per_route
        if new_load < 0 or new_load > capacity or route_full:
            flush()
            new_load = node.demand

        if node.is_delivery:
            pid = node.pickup_index
            if pid not in placed:
                pnode = node_by_id.get(pid)
                if pnode is not None:
                    p_load = load + pnode.demand
                    if p_load < 0 or p_load > capacity:
                        flush()
                        p_load = pnode.demand
                    cur_nodes.append(pnode)
                    placed.add(pid)
                    load     = p_load
                    new_load = load + node.demand

        cur_nodes.append(node)
        placed.add(nid)
        load = new_load

    flush()
    return Solution(routes=routes)

def compute_penalty(sol: Solution, penalty_mgr: PenaltyManager) -> float:
    cap_viol, prec_viol = _penalty_raw(sol)
    return penalty_mgr.cap * cap_viol + penalty_mgr.prec * prec_viol

def evaluate(ind: Individual, node_by_id: dict[int, Node],
             vehicles: list[Vehicle], depot: Node, capacity: float,
             requests: list[Request], penalty_mgr: PenaltyManager,
             max_nodes_per_route: int = 0) -> None:
    sol          = decode(ind.chromosome, node_by_id, vehicles, depot,
                          capacity, max_nodes_per_route)
    ind.solution = sol
    ind.fitness  = sol.total_cost + compute_penalty(sol, penalty_mgr)

def greedy_individual(requests: list[Request], node_by_id: dict[int, Node],
                      vehicles: list[Vehicle], depot: Node, capacity: float,
                      penalty_mgr: PenaltyManager,
                      max_nodes_per_route: int = 0) -> Individual:
    sorted_reqs = sorted(requests, key=lambda r: depot.distance_to(r.pickup))
    chrom: list[int] = []
    for r in sorted_reqs:
        chrom.append(r.pickup.id)
        chrom.append(r.delivery.id)
    chrom = repair(chrom, requests)
    ind   = Individual(chromosome=chrom)
    evaluate(ind, node_by_id, vehicles, depot, capacity, requests,
             penalty_mgr, max_nodes_per_route)
    return ind

def random_individual(requests: list[Request], node_by_id: dict[int, Node],
                      vehicles: list[Vehicle], depot: Node, capacity: float,
                      penalty_mgr: PenaltyManager,
                      max_nodes_per_route: int = 0) -> Individual:
    reqs = requests[:]
    random.shuffle(reqs)
    chrom: list[int] = []
    for r in reqs:
        chrom.append(r.pickup.id)
        chrom.append(r.delivery.id)
    chrom = repair(chrom, requests)
    ind   = Individual(chromosome=chrom)
    evaluate(ind, node_by_id, vehicles, depot, capacity, requests,
             penalty_mgr, max_nodes_per_route)
    return ind

def init_population(size: int, requests: list[Request],
                    node_by_id: dict[int, Node],
                    vehicles: list[Vehicle], depot: Node, capacity: float,
                    penalty_mgr: PenaltyManager,
                    max_nodes_per_route: int = 0) -> list[Individual]:
    greedy_seeds = min(2, size)
    pop: list[Individual] = [
        greedy_individual(requests, node_by_id, vehicles, depot, capacity,
                          penalty_mgr, max_nodes_per_route)
        for _ in range(greedy_seeds)
    ]
    pop += [
        random_individual(requests, node_by_id, vehicles, depot, capacity,
                          penalty_mgr, max_nodes_per_route)
        for _ in range(size - greedy_seeds)
    ]
    return pop

def tournament_select(population: list[Individual],
                      k: int = TOURNAMENT_K) -> Individual:
    candidates = random.sample(population, min(k, len(population)))
    return min(candidates, key=lambda x: x.fitness)

def order_crossover(pa: Individual, pb: Individual,
                    requests: list[Request]) -> Individual:
    size = len(pa.chromosome)
    if size == 0:
        return pa.copy()
    i, j    = sorted(random.sample(range(size), 2))
    segment = pa.chromosome[i:j + 1]
    seg_set = set(segment)
    remainder   = [nid for nid in pb.chromosome if nid not in seg_set]
    child_chrom = remainder[:i] + segment + remainder[i:]
    child_chrom = repair(child_chrom, requests)
    return Individual(chromosome=child_chrom)

def relocate_mutate(ind: Individual, requests: list[Request]) -> Individual:
    if not requests:
        return ind.copy()
    chrom    = ind.chromosome[:]
    req      = random.choice(requests)
    pid, did = req.pickup.id, req.delivery.id
    chrom    = [nid for nid in chrom if nid not in (pid, did)]
    p_pos    = random.randint(0, len(chrom))
    chrom.insert(p_pos, pid)
    d_pos = random.randint(p_pos + 1, len(chrom))
    chrom.insert(d_pos, did)
    return Individual(chromosome=chrom)

def swap_mutate(ind: Individual, requests: list[Request]) -> Individual:
    chrom = ind.chromosome[:]
    if len(chrom) < 2:
        return ind.copy()
    i, j = random.sample(range(len(chrom)), 2)
    chrom[i], chrom[j] = chrom[j], chrom[i]
    chrom = repair(chrom, requests)
    return Individual(chromosome=chrom)


def get_mutation_prob(no_improve: int, no_improve_lim: int) -> float:
    boost = (no_improve / max(no_improve_lim, 1)) * 0.40
    return min(MUTATION_PROB + boost, 0.70)

def or_opt_route(route: Route, depot: Node, capacity: float,
                 seg_len: int = 1) -> Route:
    nodes = route.nodes[:]
    m     = len(nodes)
    if m < 2:
        return route

    improved = True
    while improved:
        improved = False
        for i in range(m):
            node   = nodes[i]
            prev_i = nodes[i - 1] if i > 0     else depot
            next_i = nodes[i + 1] if i < m - 1 else depot

            delta_remove = (cdist(prev_i, next_i)
                            - cdist(prev_i, node) - cdist(node, next_i))

            rest = nodes[:i] + nodes[i + 1:]

            for j in range(m):
                if j == i:
                    continue
                prev_j = rest[j - 1] if j > 0     else depot
                next_j = rest[j]     if j < m - 1 else depot

                delta_insert = (cdist(prev_j, node) + cdist(node, next_j)
                                - cdist(prev_j, next_j))

                if delta_remove + delta_insert < -1e-9:
                    candidate = rest[:j] + [node] + rest[j:]
                    if (is_precedence_ok(candidate)
                            and is_capacity_ok(candidate, capacity)):
                        nodes    = candidate
                        improved = True
                        break
            if improved:
                break

    new_r       = Route(vehicle=route.vehicle, depot=depot)
    new_r.nodes = nodes
    return new_r


def two_opt_route(route: Route, depot: Node, capacity: float) -> Route:
    nodes     = route.nodes[:]
    best_cost = route_cost(nodes, depot)
    n         = len(nodes)
    improved  = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                candidate = nodes[:i] + nodes[i:j + 1][::-1] + nodes[j + 1:]
                if not is_precedence_ok(candidate):
                    continue
                if not is_capacity_ok(candidate, capacity):
                    continue
                c = route_cost(candidate, depot)
                if c < best_cost - 1e-9:
                    nodes, best_cost = candidate, c
                    improved = True
                    break
            if improved:
                break
    new_r       = Route(vehicle=route.vehicle, depot=depot)
    new_r.nodes = nodes
    return new_r

def relocate_request(sol: Solution, requests: list[Request],
                     depot: Node, capacity: float,
                     max_passes: int = 1) -> Solution:
    routes: list[Route] = []
    for r in sol.routes:
        nr       = Route(vehicle=r.vehicle, depot=depot)
        nr.nodes = r.nodes[:]
        routes.append(nr)

    for _ in range(max_passes):
        any_improved = False
        for req in requests:
            pid, did = req.pickup.id, req.delivery.id

            src_idx = next(
                (i for i, r in enumerate(routes) if any(n.id == pid for n in r.nodes)),
                None,
            )
            if src_idx is None:
                continue

            src_nodes    = routes[src_idx].nodes
            stripped     = [n for n in src_nodes if n.id not in (pid, did)]
            old_src_cost = route_cost(src_nodes, depot)
            new_src_cost = route_cost(stripped, depot)

            best_move = None

            for dst_idx, dst_route in enumerate(routes):
                base         = stripped if dst_idx == src_idx else dst_route.nodes[:]
                old_dst_cost = new_src_cost if dst_idx == src_idx \
                               else route_cost(base, depot)

                base_delta = new_src_cost - old_src_cost
                bm = len(base)

                found_improvement = False
                for p_pos in range(bm + 1):
                    prev_p = base[p_pos - 1] if p_pos > 0  else depot
                    next_p = base[p_pos]     if p_pos < bm else depot
                    delta_p = (cdist(prev_p, req.pickup) + cdist(req.pickup, next_p)
                               - cdist(prev_p, next_p))

                    for d_pos in range(p_pos + 1, bm + 2):
                        if d_pos == p_pos + 1:
                            prev_d = req.pickup
                            next_d = base[p_pos] if p_pos < bm else depot
                        else:
                            ip = d_pos - 2
                            ix = d_pos - 1
                            prev_d = base[ip] if 0 <= ip < bm else depot
                            next_d = base[ix] if 0 <= ix < bm else depot

                        delta_d = (cdist(prev_d, req.delivery) + cdist(req.delivery, next_d)
                                   - cdist(prev_d, next_d))

                        if base_delta + delta_p + delta_d < -1e-9:
                            # Only build candidate when we have a promising move
                            tmp2 = base[:p_pos] + [req.pickup] + base[p_pos:]
                            tmp2.insert(d_pos, req.delivery)
                            if (is_capacity_ok(tmp2, capacity)
                                    and is_precedence_ok(tmp2)):
                                best_move = (dst_idx, p_pos, d_pos)
                                found_improvement = True
                                break
                    if found_improvement:
                        break
                if found_improvement:
                    break

            if best_move is not None:
                dst_idx, p_pos, d_pos = best_move
                routes[src_idx].nodes = stripped
                base = (stripped if dst_idx == src_idx else routes[dst_idx].nodes)[:]
                base.insert(p_pos, req.pickup)
                base.insert(d_pos, req.delivery)
                routes[dst_idx].nodes = base
                any_improved = True

        if not any_improved:
            break

    return Solution(routes=[r for r in routes if r.nodes])

def local_search(ind: Individual, node_by_id: dict[int, Node],
                 vehicles: list[Vehicle], depot: Node, capacity: float,
                 requests: list[Request], penalty_mgr: PenaltyManager,
                 max_nodes_per_route: int = 0) -> Individual:

    if ind.solution is None:
        evaluate(ind, node_by_id, vehicles, depot, capacity, requests,
                 penalty_mgr, max_nodes_per_route)

    sol = ind.solution

    new_routes = []
    for r in sol.routes:
        if not r.nodes:
            continue
        r = or_opt_route(r, depot, capacity, seg_len=1)
        new_routes.append(r)

    sol = Solution(routes=[r for r in new_routes if r.nodes])
    sol = relocate_request(sol, requests, depot, capacity)

    new_chrom = [n.id for r in sol.routes for n in r.nodes]
    new_chrom = repair(new_chrom, requests)

    new_ind         = Individual(chromosome=new_chrom, solution=sol)
    new_ind.fitness = sol.total_cost + compute_penalty(sol, penalty_mgr)
    return new_ind

def run_ga(requests: list[Request], vehicles: list[Vehicle],
           depot: Node, capacity: float, node_by_id: dict[int, Node],
           pop_size: int = POPULATION_SIZE, max_gen: int = MAX_GENERATIONS,
           no_improve_lim: int = NO_IMPROVE_LIMIT,
           verbose: bool = False) -> Solution:

    max_nodes_per_route = 0
    if verbose:
        print(f"  [GA] constraint: capacity-only (no node-count cap)")

    penalty_mgr = PenaltyManager()

    population = init_population(pop_size, requests, node_by_id, vehicles, depot,
                                 capacity, penalty_mgr, max_nodes_per_route)
    population.sort(key=lambda x: x.fitness)
    best           = population[0].copy()
    best_feasible  = None
    no_improve     = 0

    if verbose:
        print(f"  [GA] Gen   0 | best fitness = {best.fitness:.2f}")

    for gen in range(1, max_gen + 1):
        penalty_mgr.update(population)

        elite_count = max(1, int(pop_size * ELITE_RATIO))
        elites      = [ind.copy() for ind in population[:elite_count]]
        mut_prob = get_mutation_prob(no_improve, no_improve_lim)

        offspring: list[Individual] = []
        while len(offspring) < pop_size - elite_count:
            pa = tournament_select(population)
            pb = tournament_select(population)

            if random.random() < CROSSOVER_PROB:
                child = order_crossover(pa, pb, requests)
            else:
                child = pa.copy()

            if random.random() < mut_prob:
                if random.random() < 0.5:
                    child = relocate_mutate(child, requests)
                else:
                    child = swap_mutate(child, requests)

            if random.random() < LOCAL_SEARCH_PROB:
                child = local_search(child, node_by_id, vehicles, depot, capacity,
                                     requests, penalty_mgr, max_nodes_per_route)
            else:
                evaluate(child, node_by_id, vehicles, depot, capacity, requests,
                         penalty_mgr, max_nodes_per_route)

            offspring.append(child)

        population = elites + offspring
        population.sort(key=lambda x: x.fitness)
        population = population[:pop_size]

        for ind in population:
            if (ind.solution is not None
                    and compute_penalty(ind.solution, penalty_mgr) == 0):
                if (best_feasible is None
                        or ind.solution.total_cost < best_feasible.solution.total_cost):
                    best_feasible = ind.copy()

        current_best = population[0]
        if current_best.fitness < best.fitness - 1e-9:
            best       = current_best.copy()
            no_improve = 0
            if verbose:
                feas_cost = (f"{best_feasible.solution.total_cost:.2f}"
                             if best_feasible else "none yet")
                print(f"  [GA] Gen {gen:>4} | fitness={best.fitness:.2f}  "
                      f"best_feasible={feas_cost}  *"
                      f"  [cap={penalty_mgr.cap:.1f} prec={penalty_mgr.prec:.1f}"
                      f"  mut={mut_prob:.2f}]")
        else:
            no_improve += 1
            if verbose and gen % 20 == 0:
                feas_cost = (f"{best_feasible.solution.total_cost:.2f}"
                             if best_feasible else "none yet")
                print(f"  [GA] Gen {gen:>4} | fitness={best.fitness:.2f}  "
                      f"best_feasible={feas_cost}"
                      f"  (no_improve={no_improve}  mut={mut_prob:.2f})")

        if no_improve >= no_improve_lim:
            if verbose:
                print(f"  [GA] Early stop at gen {gen} "
                      f"(no improvement for {no_improve_lim} gens)")
            break

    if verbose:
        feas_cost = (f"{best_feasible.solution.total_cost:.2f}"
                     if best_feasible else "none")
        print(f"  [GA] Done  | best fitness={best.fitness:.2f}  "
              f"best_feasible_cost={feas_cost}")

    final = best_feasible if best_feasible is not None else best
    if final.solution is None:
        evaluate(final, node_by_id, vehicles, depot, capacity, requests,
                 penalty_mgr, max_nodes_per_route)
    return final.solution if final.solution is not None else Solution()

def genetic_algorithm(nodes: list[Node], requests: list[Request],
                      vehicles: list[Vehicle], capacity: float) -> Solution:
    depot      = nodes[0]
    node_by_id = {n.id: n for n in nodes}
    return run_ga(
        requests=requests,
        vehicles=vehicles,
        depot=depot,
        capacity=capacity,
        node_by_id=node_by_id,
        pop_size=POPULATION_SIZE,
        max_gen=MAX_GENERATIONS,
        no_improve_lim=NO_IMPROVE_LIMIT,
        verbose=True,
    )