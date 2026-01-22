"""NSGA-II for 30x4 irrigation wheel-group scheduling + quantitative "near-optimal" report.

What you get after ONE execution:
  (1) Convergence logs (per generation): feasible ratio, |Pareto front|, best objectives.
  (2) Multi-run robustness: repeat NSGA-II with multiple random seeds; summarize best/median/worst/IQR.
  (3) Baseline comparison: random schedules + simple paired heuristics.
  (4) A-posteriori neighborhood test: local perturbations around the chosen solution.

Assumptions
-----------
- 120 laterals are named as '<NodeID>_L' and '<NodeID>_R', e.g. 'J11_L', 'J11_R'.
- Each node has exactly two laterals (_L/_R), so there are 60 field nodes.
- Group size is 4 laterals (two node-pairs per group).

Hard constraint
---------------
- For every opened node, pressure_head >= Hmin (no deficit allowed).

Objectives (minimize)
---------------------
  f1 = Var(surplus) across all 120 laterals
  f2 = Mean(surplus) across all 120 laterals
  f3 = pair_split_penalty: count of nodes whose L/R are split into different groups

Selection (your preference)
---------------------------
- NSGA-II optimizes (f1,f2,f3). Final "recommended" solution is picked from Pareto set
  via weighted sum on min-max normalized objectives with priority weights:
      variance > mean > pair completion

Files
-----
- Requires your `tree_evaluator.py` in the same folder (or on PYTHONPATH).
- Requires Nodes.xlsx and Pipes.xlsx.

Outputs
-------
- convergence_best_run.csv
- pareto_best_run.csv
- runs_summary.csv
- baselines.csv
- neighborhood_test.csv

"""

from __future__ import annotations
import time
import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

# Your fast evaluator
from tree_evaluator import (
    TreeHydraulicEvaluator,
    load_nodes_xlsx,
    load_pipes_xlsx,
    is_field_node_id,
    build_lateral_ids_for_field_nodes,
)


# =========================
# Configuration
# =========================

@dataclass(frozen=True)
class NSGA2Config:
    pop_size: int = 200
    generations: int = 2000
    p_crossover: float = 0.9
    p_mutation: float = 0.3
    tournament_k: int = 2

    group_size: int = 4
    n_groups: int = 30

    q_lateral: float = 0.012

    # Used only when selecting ONE solution from the final Pareto set
    w_var: float = 3.0
    w_mean: float = 2.0
    w_pair: float = 1.0


@dataclass
class Individual:
    perm: List[str]                      # permutation of laterals (len=120)
    f: Tuple[float, float, float]        # (var, mean, pair_penalty)
    violation: float                     # total constraint violation (0 if feasible)
    feasible: bool
    rank: int = 10**9
    crowding: float = 0.0


@dataclass
class GenLog:
    gen: int
    feasible: int
    pop: int
    front0_feasible: int
    best_var: float
    best_mean: float
    best_pair: float


@dataclass
class RunResult:
    seed: int
    chosen: Individual
    pareto: List[Individual]
    logs: List[GenLog]


# =========================
# Helpers
# =========================

def perm_to_groups(perm: List[str], group_size: int, n_groups: int) -> List[List[str]]:
    assert len(perm) == group_size * n_groups
    return [perm[i * group_size:(i + 1) * group_size] for i in range(n_groups)]


def pair_split_penalty(perm: List[str], group_size: int) -> float:
    """Penalty encouraging a node's two laterals (L/R) to be in the SAME irrigation group."""
    pos = {lat: i for i, lat in enumerate(perm)}
    bases = {lat.rsplit('_', 1)[0] for lat in perm}

    split = 0
    for b in bases:
        l = f"{b}_L"
        r = f"{b}_R"
        if l not in pos or r not in pos:
            split += 1
            continue
        if (pos[l] // group_size) != (pos[r] // group_size):
            split += 1
    return float(split)


def dominates(a: Individual, b: Individual) -> bool:
    """Deb's constraint-domination."""
    if a.feasible and not b.feasible:
        return True
    if not a.feasible and b.feasible:
        return False
    if not a.feasible and not b.feasible:
        return a.violation < b.violation
    return all(x <= y for x, y in zip(a.f, b.f)) and any(x < y for x, y in zip(a.f, b.f))


def fast_non_dominated_sort(pop: List[Individual]) -> List[List[int]]:
    S = [set() for _ in pop]
    n = [0 for _ in pop]
    fronts: List[List[int]] = [[]]

    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if dominates(pop[p], pop[q]):
                S[p].add(q)
            elif dominates(pop[q], pop[p]):
                n[p] += 1
        if n[p] == 0:
            pop[p].rank = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    pop[q].rank = i + 1
                    nxt.append(q)
        i += 1
        fronts.append(nxt)

    fronts.pop()
    return fronts


def crowding_distance(pop: List[Individual], front: List[int]) -> None:
    if not front:
        return
    m = len(pop[0].f)
    for idx in front:
        pop[idx].crowding = 0.0

    for k in range(m):
        front_sorted = sorted(front, key=lambda i: pop[i].f[k])
        pop[front_sorted[0]].crowding = float("inf")
        pop[front_sorted[-1]].crowding = float("inf")

        fmin = pop[front_sorted[0]].f[k]
        fmax = pop[front_sorted[-1]].f[k]
        denom = (fmax - fmin) if (fmax - fmin) != 0.0 else 1.0

        for j in range(1, len(front_sorted) - 1):
            prev_f = pop[front_sorted[j - 1]].f[k]
            next_f = pop[front_sorted[j + 1]].f[k]
            pop[front_sorted[j]].crowding += (next_f - prev_f) / denom


def tournament_select(pop: List[Individual], k: int) -> Individual:
    cand = random.sample(pop, k)
    cand.sort(key=lambda ind: (ind.rank, -ind.crowding, ind.violation))
    return cand[0]


# =========================
# Permutation operators
# =========================

def order_crossover(p1: List[str], p2: List[str]) -> Tuple[List[str], List[str]]:
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))

    def make_child(x: List[str], y: List[str]) -> List[str]:
        child = [None] * n  # type: ignore
        child[a:b] = x[a:b]
        fill = [g for g in y if g not in child[a:b]]
        ptr = 0
        for i in list(range(0, a)) + list(range(b, n)):
            child[i] = fill[ptr]
            ptr += 1
        return child  # type: ignore

    return make_child(p1, p2), make_child(p2, p1)


def swap_mutation(perm: List[str], p: float) -> None:
    if random.random() > p:
        return
    i, j = random.sample(range(len(perm)), 2)
    perm[i], perm[j] = perm[j], perm[i]


def group_swap_mutation(perm: List[str], group_size: int, p: float) -> None:
    if random.random() > p:
        return
    n_groups = len(perm) // group_size
    g1, g2 = random.sample(range(n_groups), 2)
    s1, e1 = g1 * group_size, (g1 + 1) * group_size
    s2, e2 = g2 * group_size, (g2 + 1) * group_size
    perm[s1:e1], perm[s2:e2] = perm[s2:e2], perm[s1:e1]


# =========================
# Evaluation with caching
# =========================

@dataclass
class EvalContext:
    evaluator: TreeHydraulicEvaluator
    lateral_to_node: Dict[str, str]
    cfg: NSGA2Config

    # group cache: key=tuple(sorted(group_laterals)), value=dict[node_id]=pressure_head
    group_cache: Dict[Tuple[str, ...], Dict[str, float]]


def evaluate_group_cached(ctx: EvalContext, group: List[str]) -> Dict[str, float]:
    key = tuple(sorted(group))
    if key in ctx.group_cache:
        return ctx.group_cache[key]

    res = ctx.evaluator.evaluate_group(group, lateral_to_node=ctx.lateral_to_node, q_lateral=ctx.cfg.q_lateral)
    ctx.group_cache[key] = res.pressures
    return res.pressures


def evaluate_perm(ctx: EvalContext, perm: List[str]) -> Individual:
    groups = perm_to_groups(perm, ctx.cfg.group_size, ctx.cfg.n_groups)

    surpluses: List[float] = []
    violation = 0.0

    for g in groups:
        pressures = evaluate_group_cached(ctx, g)
        for lat in g:
            nid = ctx.lateral_to_node[lat]
            p = pressures[nid]
            s = p - ctx.evaluator.Hmin
            if s < 0.0:
                violation += -s
            surpluses.append(s)

    feasible = (violation <= 0.0)

    mean_s = sum(surpluses) / len(surpluses)
    var_s = sum((x - mean_s) ** 2 for x in surpluses) / len(surpluses)
    pen_pair = pair_split_penalty(perm, ctx.cfg.group_size)

    return Individual(perm=perm, f=(var_s, mean_s, pen_pair), violation=violation, feasible=feasible)


# =========================
# Pareto selection and preference pick
# =========================

def pick_one_by_weighted_sum(front: List[Individual], cfg: NSGA2Config) -> Individual:
    if len(front) == 1:
        return front[0]

    m = len(front[0].f)
    mins = [min(ind.f[k] for ind in front) for k in range(m)]
    maxs = [max(ind.f[k] for ind in front) for k in range(m)]

    def norm(v: float, k: int) -> float:
        den = (maxs[k] - mins[k])
        return 0.0 if den == 0.0 else (v - mins[k]) / den

    best = front[0]
    best_score = float("inf")
    for ind in front:
        score = (
            cfg.w_var * norm(ind.f[0], 0)
            + cfg.w_mean * norm(ind.f[1], 1)
            + cfg.w_pair * norm(ind.f[2], 2)
        )
        if score < best_score:
            best_score = score
            best = ind
    return best


def preference_score_relative(ind: Individual, ref: Individual, cfg: NSGA2Config) -> float:
    """A stable preference score for cross-run/baseline comparison.

    Uses ratios to a reference solution to avoid min-max sensitivity.
    Lower is better.
    """
    # Avoid division by zero (though var/mean should be >0 in practice)
    v0 = ref.f[0] if ref.f[0] > 1e-12 else 1e-12
    m0 = ref.f[1] if ref.f[1] > 1e-12 else 1e-12
    p0 = ref.f[2] if ref.f[2] > 1e-12 else 1.0

    return cfg.w_var * (ind.f[0] / v0) + cfg.w_mean * (ind.f[1] / m0) + cfg.w_pair * (ind.f[2] / p0)


# =========================
# NSGA-II main (single run)
# =========================

def run_nsga2_once(ctx: EvalContext, lateral_ids: List[str], seed: int) -> RunResult:
    random.seed(seed)

    # Build initial population (mixture: some paired, some random)
    pop: List[Individual] = []

    # Paired init: build 30 groups each containing 2 node-pairs (penalty=0)
    bases = sorted({x.rsplit('_', 1)[0] for x in lateral_ids})
    for _ in range(max(1, ctx.cfg.pop_size // 5)):
        bb = bases[:]
        random.shuffle(bb)
        perm: List[str] = []
        for i in range(0, len(bb), 2):
            b1, b2 = bb[i], bb[i + 1]
            perm += [f"{b1}_L", f"{b1}_R", f"{b2}_L", f"{b2}_R"]
        pop.append(evaluate_perm(ctx, perm))

    # Random remainder
    while len(pop) < ctx.cfg.pop_size:
        perm = lateral_ids[:]
        random.shuffle(perm)
        pop.append(evaluate_perm(ctx, perm))

    logs: List[GenLog] = []

    for gen in range(ctx.cfg.generations):
        fronts = fast_non_dominated_sort(pop)
        for fr in fronts:
            crowding_distance(pop, fr)

        # Log best feasible in current population
        feas = [ind for ind in pop if ind.feasible]
        front0 = [pop[i] for i in fronts[0] if pop[i].feasible]
        if feas:
            best = min(feas, key=lambda ind: (ind.f[0], ind.f[1], ind.f[2]))
            logs.append(
                GenLog(
                    gen=gen + 1,
                    feasible=len(feas),
                    pop=len(pop),
                    front0_feasible=len(front0),
                    best_var=best.f[0],
                    best_mean=best.f[1],
                    best_pair=best.f[2],
                )
            )
        else:
            # no feasible yet
            logs.append(
                GenLog(
                    gen=gen + 1,
                    feasible=0,
                    pop=len(pop),
                    front0_feasible=0,
                    best_var=float("inf"),
                    best_mean=float("inf"),
                    best_pair=float("inf"),
                )
            )

        # Create offspring
        offspring: List[Individual] = []
        while len(offspring) < ctx.cfg.pop_size:
            p1 = tournament_select(pop, ctx.cfg.tournament_k)
            p2 = tournament_select(pop, ctx.cfg.tournament_k)

            c1p, c2p = p1.perm[:], p2.perm[:]
            if random.random() < ctx.cfg.p_crossover:
                c1p, c2p = order_crossover(p1.perm, p2.perm)

            swap_mutation(c1p, ctx.cfg.p_mutation)
            swap_mutation(c2p, ctx.cfg.p_mutation)
            group_swap_mutation(c1p, ctx.cfg.group_size, ctx.cfg.p_mutation)
            group_swap_mutation(c2p, ctx.cfg.group_size, ctx.cfg.p_mutation)

            offspring.append(evaluate_perm(ctx, c1p))
            if len(offspring) < ctx.cfg.pop_size:
                offspring.append(evaluate_perm(ctx, c2p))

        # Combine and select next generation
        combined = pop + offspring
        fronts = fast_non_dominated_sort(combined)
        for fr in fronts:
            crowding_distance(combined, fr)

        new_pop: List[Individual] = []
        for fr in fronts:
            if len(new_pop) + len(fr) <= ctx.cfg.pop_size:
                new_pop.extend([combined[i] for i in fr])
            else:
                fr_sorted = sorted(fr, key=lambda i: combined[i].crowding, reverse=True)
                needed = ctx.cfg.pop_size - len(new_pop)
                new_pop.extend([combined[i] for i in fr_sorted[:needed]])
                break
        pop = new_pop

    fronts = fast_non_dominated_sort(pop)
    pareto = [pop[i] for i in fronts[0] if pop[i].feasible]
    if not pareto:
        chosen = min(pop, key=lambda ind: ind.violation)
        return RunResult(seed=seed, chosen=chosen, pareto=[], logs=logs)

    chosen = pick_one_by_weighted_sum(pareto, ctx.cfg)
    return RunResult(seed=seed, chosen=chosen, pareto=pareto, logs=logs)


# =========================
# Baselines
# =========================

def baseline_random(ctx: EvalContext, lateral_ids: List[str], n: int, seed: int) -> Individual:
    random.seed(seed)
    best: Optional[Individual] = None
    for _ in range(n):
        perm = lateral_ids[:]
        random.shuffle(perm)
        ind = evaluate_perm(ctx, perm)
        if best is None:
            best = ind
            continue
        # Deb compare on (feasible, violation, preference to best)
        if dominates(ind, best):
            best = ind
        elif ind.feasible and best.feasible:
            # tie-break by (var, mean, pair)
            if ind.f < best.f:
                best = ind
        elif (not ind.feasible) and (not best.feasible) and ind.violation < best.violation:
            best = ind
    assert best is not None
    return best


def baseline_paired_sequential(ctx: EvalContext, lateral_ids: List[str]) -> Individual:
    bases = sorted({x.rsplit('_', 1)[0] for x in lateral_ids})
    perm: List[str] = []
    for i in range(0, len(bases), 2):
        b1, b2 = bases[i], bases[i + 1]
        perm += [f"{b1}_L", f"{b1}_R", f"{b2}_L", f"{b2}_R"]
    return evaluate_perm(ctx, perm)


def baseline_paired_random(ctx: EvalContext, lateral_ids: List[str], seed: int) -> Individual:
    random.seed(seed)
    bases = sorted({x.rsplit('_', 1)[0] for x in lateral_ids})
    random.shuffle(bases)
    perm: List[str] = []
    for i in range(0, len(bases), 2):
        b1, b2 = bases[i], bases[i + 1]
        # Randomize within-pair order slightly
        if random.random() < 0.5:
            perm += [f"{b1}_L", f"{b1}_R", f"{b2}_L", f"{b2}_R"]
        else:
            perm += [f"{b2}_L", f"{b2}_R", f"{b1}_L", f"{b1}_R"]
    return evaluate_perm(ctx, perm)


# =========================
# Neighborhood test
# =========================

def neighborhood_test(ctx: EvalContext, ref: Individual, trials: int, seed: int) -> Dict[str, float]:
    random.seed(seed)
    feasible_n = 0
    better_pref = 0
    better_pareto = 0

    best_neighbor: Optional[Individual] = None

    for _ in range(trials):
        perm = ref.perm[:]

        # Small perturbation: swap two laterals OR swap two groups
        if random.random() < 0.7:
            i, j = random.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
        else:
            gs = ctx.cfg.group_size
            ng = len(perm) // gs
            g1, g2 = random.sample(range(ng), 2)
            s1, e1 = g1 * gs, (g1 + 1) * gs
            s2, e2 = g2 * gs, (g2 + 1) * gs
            perm[s1:e1], perm[s2:e2] = perm[s2:e2], perm[s1:e1]

        ind = evaluate_perm(ctx, perm)
        if not ind.feasible:
            continue

        feasible_n += 1

        # Preference improvement (relative score)
        if preference_score_relative(ind, ref, ctx.cfg) < preference_score_relative(ref, ref, ctx.cfg) - 1e-12:
            better_pref += 1

        # Pareto improvement
        if dominates(ind, ref):
            better_pareto += 1

        if best_neighbor is None:
            best_neighbor = ind
        else:
            # Pick best neighbor by the same selection rule as final preference (relative score)
            if preference_score_relative(ind, ref, ctx.cfg) < preference_score_relative(best_neighbor, ref, ctx.cfg):
                best_neighbor = ind

    out: Dict[str, float] = {
        "trials": float(trials),
        "feasible_neighbors": float(feasible_n),
        "better_pref": float(better_pref),
        "better_pareto": float(better_pareto),
        "p_feasible": float(feasible_n) / float(trials) if trials else 0.0,
        "p_better_pref_given_feasible": float(better_pref) / float(feasible_n) if feasible_n else 0.0,
        "p_better_pareto_given_feasible": float(better_pareto) / float(feasible_n) if feasible_n else 0.0,
    }

    if best_neighbor is not None:
        out.update(
            {
                "best_neighbor_var": best_neighbor.f[0],
                "best_neighbor_mean": best_neighbor.f[1],
                "best_neighbor_pair": best_neighbor.f[2],
                "best_neighbor_pref_rel": preference_score_relative(best_neighbor, ref, ctx.cfg),
            }
        )
    return out


# =========================
# Reporting I/O
# =========================

def write_convergence_csv(path: Path, logs: List[GenLog]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen", "feasible", "pop", "feasible_ratio", "front0_feasible", "best_var", "best_mean", "best_pair"])
        for r in logs:
            feas_ratio = (r.feasible / r.pop) if r.pop else 0.0
            w.writerow([r.gen, r.feasible, r.pop, feas_ratio, r.front0_feasible, r.best_var, r.best_mean, r.best_pair])


def write_pareto_csv(path: Path, pareto: List[Individual]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["var", "mean", "pair_penalty"])
        for ind in sorted(pareto, key=lambda x: (x.f[0], x.f[1], x.f[2])):
            w.writerow([ind.f[0], ind.f[1], ind.f[2]])


def summary_stats(values: List[float]) -> Dict[str, float]:
    values = sorted(values)
    if not values:
        return {"best": math.nan, "median": math.nan, "worst": math.nan, "q1": math.nan, "q3": math.nan, "iqr": math.nan}
    n = len(values)
    best = values[0]
    worst = values[-1]
    median = statistics.median(values)
    q1 = statistics.median(values[: n // 2])
    q3 = statistics.median(values[(n + 1) // 2 :])
    return {"best": best, "median": median, "worst": worst, "q1": q1, "q3": q3, "iqr": q3 - q1}


# =========================
# Top-level driver
# =========================

def run_all(
    nodes_xlsx: str,
    pipes_xlsx: str,
    root: str,
    H0: float,
    Hmin: float,
    cfg: NSGA2Config,
    outdir: str,
    n_runs: int = 10,
    base_seed: int = 20260120,
    baseline_random_n: int = 2000,
    neighborhood_trials: int = 5000,
) -> None:
    t0_total = time.perf_counter()
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes_xlsx(nodes_xlsx)
    edges = load_pipes_xlsx(pipes_xlsx)
    evaluator = TreeHydraulicEvaluator(nodes=nodes, edges=edges, root=root, H0=H0, Hmin=Hmin)

    field_nodes = [nid for nid in nodes.keys() if is_field_node_id(nid)]
    lateral_ids, lateral_to_node = build_lateral_ids_for_field_nodes(field_nodes)

    if len(lateral_ids) != cfg.group_size * cfg.n_groups:
        raise ValueError(f"Expect {cfg.group_size*cfg.n_groups} laterals but got {len(lateral_ids)}")

    # Shared cache across all runs to speed up (group evaluations repeat a lot)
    ctx = EvalContext(evaluator=evaluator, lateral_to_node=lateral_to_node, cfg=cfg, group_cache={})

    # (2) Multi-run NSGA-II
    results: List[RunResult] = []
    for i in range(n_runs):
        seed = base_seed + i
        # IMPORTANT: keep cache; if you want strict independence, clear cache each run.
        rr = run_nsga2_once(ctx, lateral_ids, seed)
        results.append(rr)

    # Choose the best run by your preference on its Pareto set (already chosen inside each run)
    best_run = min(results, key=lambda rr: (rr.chosen.violation, rr.chosen.f[0], rr.chosen.f[1], rr.chosen.f[2]))

    # (1) Convergence logs for best run
    write_convergence_csv(out_path / "convergence_best_run.csv", best_run.logs)
    write_pareto_csv(out_path / "pareto_best_run.csv", best_run.pareto)

    # Summaries across runs
    vars_ = [rr.chosen.f[0] for rr in results if rr.chosen.feasible]
    means_ = [rr.chosen.f[1] for rr in results if rr.chosen.feasible]
    pairs_ = [rr.chosen.f[2] for rr in results if rr.chosen.feasible]

    var_stat = summary_stats(vars_)
    mean_stat = summary_stats(means_)
    pair_stat = summary_stats(pairs_)

    # (3) Baselines
    b_rand = baseline_random(ctx, lateral_ids, n=baseline_random_n, seed=base_seed + 999)
    b_pair_seq = baseline_paired_sequential(ctx, lateral_ids)
    b_pair_rand = baseline_paired_random(ctx, lateral_ids, seed=base_seed + 1001)

    baselines = {
        "random_best_of_N": b_rand,
        "paired_sequential": b_pair_seq,
        "paired_random": b_pair_rand,
    }

    # Dominance vs baselines (best_run.chosen compared to each baseline)
    with (out_path / "baselines.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "name",
            "feasible",
            "violation",
            "var",
            "mean",
            "pair_penalty",
            "chosen_dominates_baseline",
            "baseline_dominates_chosen",
            "pref_rel_to_chosen",
        ])
        for name, ind in baselines.items():
            w.writerow([
                name,
                ind.feasible,
                ind.violation,
                ind.f[0],
                ind.f[1],
                ind.f[2],
                dominates(best_run.chosen, ind),
                dominates(ind, best_run.chosen),
                preference_score_relative(ind, best_run.chosen, cfg),
            ])

    # (4) Neighborhood test
    neigh = neighborhood_test(ctx, best_run.chosen, trials=neighborhood_trials, seed=base_seed + 2002)
    with (out_path / "neighborhood_test.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(neigh.keys()))
        w.writerow([neigh[k] for k in neigh.keys()])

    # Runs summary CSV
    with (out_path / "runs_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "feasible", "violation", "var", "mean", "pair_penalty"])
        for rr in results:
            w.writerow([rr.seed, rr.chosen.feasible, rr.chosen.violation, rr.chosen.f[0], rr.chosen.f[1], rr.chosen.f[2]])

    # Console report (human-readable)
    print("\n=== NSGA-II Near-Optimality Quantitative Report ===")
    print(f"Best run seed: {best_run.seed}")
    print(f"Chosen feasible={best_run.chosen.feasible} violation={best_run.chosen.violation:.6g}")
    print(f"Chosen objectives: var={best_run.chosen.f[0]:.6g}, mean={best_run.chosen.f[1]:.6g}, pair_penalty={best_run.chosen.f[2]:.0f}")

    print("\n[1] Convergence]")
    last = best_run.logs[-1]
    print(f"Final gen={last.gen} feasible: {last.feasible}/{last.pop} (ratio={last.feasible/last.pop:.3f}) front0_feasible={last.front0_feasible}")
    print("Saved: convergence_best_run.csv, pareto_best_run.csv")

    print("\n[2] Multi-run robustness (feasible runs only)]")
    print(f"n_runs={n_runs}, feasible_runs={len(vars_)}")
    print(f"Var   : best={var_stat['best']:.6g}, median={var_stat['median']:.6g}, worst={var_stat['worst']:.6g}, IQR={var_stat['iqr']:.6g}")
    print(f"Mean  : best={mean_stat['best']:.6g}, median={mean_stat['median']:.6g}, worst={mean_stat['worst']:.6g}, IQR={mean_stat['iqr']:.6g}")
    print(f"Pair  : best={pair_stat['best']:.6g}, median={pair_stat['median']:.6g}, worst={pair_stat['worst']:.6g}, IQR={pair_stat['iqr']:.6g}")
    print("Saved: runs_summary.csv")

    print("\n[3] Baseline comparison]")
    for name, ind in baselines.items():
        rel = preference_score_relative(ind, best_run.chosen, cfg)
        print(
            f"{name}: feasible={ind.feasible} vio={ind.violation:.6g} "
            f"(var={ind.f[0]:.6g}, mean={ind.f[1]:.6g}, pair={ind.f[2]:.0f}) "
            f"chosen_dominates={dominates(best_run.chosen, ind)} baseline_dominates={dominates(ind, best_run.chosen)} pref_rel={rel:.6g}"
        )
    print("Saved: baselines.csv")

    print("\n[4] A-posteriori neighborhood test]")
    print(f"Trials={int(neigh['trials'])}, feasible_neighbors={int(neigh['feasible_neighbors'])} (p={neigh['p_feasible']:.3f})")
    print(
        f"P(better_pref | feasible)={neigh['p_better_pref_given_feasible']:.4f}, "
        f"P(dominates | feasible)={neigh['p_better_pareto_given_feasible']:.4f}"
    )
    if 'best_neighbor_pref_rel' in neigh:
        print(
            f"Best neighbor: var={neigh['best_neighbor_var']:.6g}, mean={neigh['best_neighbor_mean']:.6g}, "
            f"pair={neigh['best_neighbor_pair']:.0f}, pref_rel={neigh['best_neighbor_pref_rel']:.6g}"
        )
    print("Saved: neighborhood_test.csv")

    # Also print the chosen groups (optional)
    print("\nChosen schedule groups (1..30):")
    groups = perm_to_groups(best_run.chosen.perm, cfg.group_size, cfg.n_groups)
    for i, g in enumerate(groups, 1):
        print(i, g)
    # Timing report
    elapsed_s = time.perf_counter() - t0_total
    print(f"\n[Timing] Total optimization wall time: {elapsed_s:.2f} s ({elapsed_s/60.0:.2f} min)")
    with (out_path / 'timing.txt').open('w', encoding='utf-8') as f:
        f.write(f'total_wall_time_seconds={elapsed_s:.6f}\n')
        f.write(f'total_wall_time_minutes={elapsed_s/60.0:.6f}\n')

if __name__ == "__main__":
    cfg = NSGA2Config(
        pop_size=200,
        generations=2000,
        p_crossover=0.9,
        p_mutation=0.3,
        tournament_k=2,
        group_size=4,
        n_groups=30,
        q_lateral=0.012,
        w_var=3.0,
        w_mean=2.0,
        w_pair=1.0,
    )

    # Adjust paths
    nodes_xlsx = "Nodes.xlsx"
    pipes_xlsx = "Pipes.xlsx"

    root = "J0"      # source/root node id
    H0 = 25.0        # source head (m)
    Hmin = 11.59     # minimum required head (m)

    run_all(
        nodes_xlsx=nodes_xlsx,
        pipes_xlsx=pipes_xlsx,
        root=root,
        H0=H0,
        Hmin=Hmin,
        cfg=cfg,
        outdir="nsga2_report_out",
        n_runs=10,
        base_seed=20260120,
        baseline_random_n=2000,
        neighborhood_trials=5000,
    )
