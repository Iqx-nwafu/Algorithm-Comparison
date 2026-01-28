"""Dynamic multi-objective optimization for wheel-group irrigation scheduling.

This script implements FOUR widely-used multi-objective optimizers:
  1) NSGA-II (Nondominated Sorting Genetic Algorithm II)
  2) SPEA2   (Strength Pareto Evolutionary Algorithm 2)
  3) MOEA/D  (Decomposition-based Multiobjective EA)
  4) MOPSO   (Multi-Objective Particle Swarm Optimization; random-key permutation)

It is designed to be *drop-in compatible* with the same objective / constraint design you used in NSGA-II:

Hard constraint (must satisfy):
  - For every opened node, pressure_head >= Hmin
    (implemented via total violation = sum(max(0, Hmin - pressure_head)))

Objectives (minimize):
  f1 = Var(surplus)  across all 120 laterals
  f2 = Mean(surplus) across all 120 laterals
  f3 = pair_split_penalty: number of nodes whose L/R are split into different groups

Decision encoding:
  - A schedule is a permutation of 120 laterals.
  - Every consecutive `group_size` laterals form one irrigation group.

Dynamic self-adaptive optimization:
  - Optional `dynamic` mode supports a list of scenarios (e.g., different H0 or demand states).
  - For each scenario, the optimizer warm-starts from the previous scenario's archive/best solutions.
  - Self-adaptation adjusts mutation rate (EA) or inertia/restart (PSO) based on feasibility ratio and stagnation.

Dependencies:
  - Python stdlib only.
  - Requires your `tree_evaluator.py` on PYTHONPATH or in the same directory.
    It must provide:
      TreeHydraulicEvaluator
      load_nodes_xlsx, load_pipes_xlsx
      is_field_node_id
      build_lateral_ids_for_field_nodes

Example usage:
  python mo_irrigation_dynamic.py nsga2  --nodes Nodes.xlsx --pipes Pipes.xlsx --root J0 --H0 25 --Hmin 11.59 --out runs/nsga2
  python mo_irrigation_dynamic.py spea2  --nodes Nodes.xlsx --pipes Pipes.xlsx --root J0 --H0 25 --Hmin 11.59 --out runs/spea2
  python mo_irrigation_dynamic.py moead  --nodes Nodes.xlsx --pipes Pipes.xlsx --root J0 --H0 25 --Hmin 11.59 --out runs/moead
  python mo_irrigation_dynamic.py mopso  --nodes Nodes.xlsx --pipes Pipes.xlsx --root J0 --H0 25 --Hmin 11.59 --out runs/mopso

  # Dynamic (scenario list), warm-start across scenarios:
  python mo_irrigation_dynamic.py dynamic --algo nsga2 --nodes Nodes.xlsx --pipes Pipes.xlsx --root J0 \
        --Hmin 11.59 --H0_list 23.75,25,26.25 --out runs/dynamic_nsga2

Outputs (per run):
  - convergence.csv : per-iteration logs (feasible ratio, archive size, best objectives)
  - pareto.csv      : final feasible Pareto archive (objectives)
  - chosen_groups.txt : recommended schedule (30 groups)
  - timing.txt

Notes:
  - This is a research-grade reference implementation focusing on correctness and traceability.
  - You can increase performance by:
      (i) increasing cache reuse (already enabled),
      (ii) reducing population size / iterations,
      (iii) parallelizing evaluations (not included here to keep stdlib-only and deterministic).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --- Your fast evaluator (must exist) ---
try:
    from tree_evaluator import (
        TreeHydraulicEvaluator,
        load_nodes_xlsx,
        load_pipes_xlsx,
        is_field_node_id,
        build_lateral_ids_for_field_nodes,
    )
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import tree_evaluator.py. Put it in the same folder or on PYTHONPATH.\n"
        "Required symbols: TreeHydraulicEvaluator, load_nodes_xlsx, load_pipes_xlsx, is_field_node_id, build_lateral_ids_for_field_nodes.\n"
        f"Original import error: {e}"
    )


# =========================
# Common configuration
# =========================

@dataclass(frozen=True)
class ProblemConfig:
    group_size: int = 4
    n_groups: int = 30
    q_lateral: float = 0.012

    # Preference weights (only for picking ONE solution from final Pareto archive)
    w_var: float = 3.0
    w_mean: float = 2.0
    w_pair: float = 1.0


@dataclass
class Individual:
    perm: List[str]
    f: Tuple[float, float, float]
    violation: float
    feasible: bool


# =========================
# Helpers
# =========================

def perm_to_groups(perm: Sequence[str], group_size: int, n_groups: int) -> List[List[str]]:
    if len(perm) != group_size * n_groups:
        raise ValueError(f"Expected {group_size*n_groups} laterals but got {len(perm)}")
    return [list(perm[i * group_size : (i + 1) * group_size]) for i in range(n_groups)]


def pair_split_penalty(perm: Sequence[str], group_size: int) -> float:
    """Penalty encouraging a node's two laterals (L/R) to be in the SAME irrigation group."""
    pos = {lat: i for i, lat in enumerate(perm)}
    bases = {lat.rsplit("_", 1)[0] for lat in perm}

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


def nondominated_filter(pop: Sequence[Individual]) -> List[Individual]:
    """Return feasible nondominated set (Pareto front) from a population."""
    feas = [p for p in pop if p.feasible]
    out: List[Individual] = []
    for i, a in enumerate(feas):
        dominated_flag = False
        for j, b in enumerate(feas):
            if i == j:
                continue
            if dominates(b, a):
                dominated_flag = True
                break
        if not dominated_flag:
            out.append(a)
    return out


def pick_one_by_weighted_sum(front: Sequence[Individual], cfg: ProblemConfig) -> Individual:
    """Pick one preferred solution from a Pareto set by min-max normalized weighted sum."""
    front = list(front)
    if len(front) == 1:
        return front[0]

    mins = [min(ind.f[k] for ind in front) for k in range(3)]
    maxs = [max(ind.f[k] for ind in front) for k in range(3)]

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


def _clamp_nonneg(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return v if v > 0.0 else 0.0


def normalize_weights_3(w_var: float, w_mean: float, w_pair: float) -> Tuple[float, float, float]:
    """Normalize (w_var, w_mean, w_pair) to sum to 1 with non-negative entries."""
    a = _clamp_nonneg(w_var)
    b = _clamp_nonneg(w_mean)
    c = _clamp_nonneg(w_pair)
    s = a + b + c
    if s <= 0.0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (a / s, b / s, c / s)


def pick_one_by_weighted_sum_w(front: Sequence[Individual], w: Tuple[float, float, float]) -> Individual:
    """Like pick_one_by_weighted_sum, but accept an explicit weight tuple."""
    wv, wm, wp = normalize_weights_3(w[0], w[1], w[2])

    front = list(front)
    if len(front) == 1:
        return front[0]

    mins = [min(ind.f[k] for ind in front) for k in range(3)]
    maxs = [max(ind.f[k] for ind in front) for k in range(3)]

    def norm(v: float, k: int) -> float:
        den = (maxs[k] - mins[k])
        return 0.0 if den == 0.0 else (v - mins[k]) / den

    best = front[0]
    best_score = float("inf")
    for ind in front:
        score = wv * norm(ind.f[0], 0) + wm * norm(ind.f[1], 1) + wp * norm(ind.f[2], 2)
        if score < best_score:
            best_score = score
            best = ind
    return best



def pick_one_by_weighted_sum_2d(front: Sequence[Individual], w_var: float, w_mean: float) -> Individual:
    """Pick ONE solution from a Pareto archive using ONLY (var, mean) weights.

    - Uses feasible solutions if present; otherwise returns the lowest-violation individual.
    - Uses min-max normalization on var and mean within the candidate set.
    """
    if not front:
        raise ValueError("Empty front")

    feas = [ind for ind in front if bool(ind.feasible)]
    cand = feas if feas else list(front)

    if not feas:
        return min(cand, key=lambda x: float(x.violation))

    vmin = min(float(ind.f[0]) for ind in cand)
    vmax = max(float(ind.f[0]) for ind in cand)
    mmin = min(float(ind.f[1]) for ind in cand)
    mmax = max(float(ind.f[1]) for ind in cand)

    dv = (vmax - vmin) if (vmax > vmin) else 1.0
    dm = (mmax - mmin) if (mmax > mmin) else 1.0

    best = cand[0]
    best_score = float("inf")
    for ind in cand:
        v = (float(ind.f[0]) - vmin) / dv
        m = (float(ind.f[1]) - mmin) / dm
        s = float(w_var) * v + float(w_mean) * m
        if s < best_score:
            best_score = s
            best = ind
    return best



def generate_weight_scan_line(grid: int, w_pair_fixed: float = 0.0) -> List[Tuple[float, float, float]]:
    """2D sweep over (w_var, w_mean) with fixed w_pair, then normalized to sum=1."""
    g = int(grid)
    if g < 2:
        raise ValueError("--grid must be >= 2")
    out: List[Tuple[float, float, float]] = []
    for i in range(g):
        t = i / (g - 1)
        wv = t
        wm = 1.0 - t
        wp = float(w_pair_fixed)
        out.append(normalize_weights_3(wv, wm, wp))
    # de-dup (can happen when wp dominates and (wv,wm) become tiny after normalization)
    uniq: List[Tuple[float, float, float]] = []
    seen = set()
    for w in out:
        key = (round(w[0], 12), round(w[1], 12), round(w[2], 12))
        if key not in seen:
            seen.add(key)
            uniq.append(w)
    return uniq


def generate_weight_scan_simplex(H: int) -> List[Tuple[float, float, float]]:
    """Full simplex-lattice scan for 3 weights: (i/H, j/H, k/H) with i+j+k=H."""
    HH = int(H)
    if HH < 1:
        raise ValueError("--simplex_H must be >= 1")
    out: List[Tuple[float, float, float]] = []
    for i in range(HH + 1):
        for j in range(HH + 1 - i):
            k = HH - i - j
            out.append((i / HH, j / HH, k / HH))
    return out

# =========================
# Permutation operators
# =========================

def order_crossover(p1: Sequence[str], p2: Sequence[str], rng: random.Random) -> Tuple[List[str], List[str]]:
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))

    def make_child(x: Sequence[str], y: Sequence[str]) -> List[str]:
        child: List[Optional[str]] = [None] * n
        child[a:b] = list(x[a:b])
        fill = [g for g in y if g not in child[a:b]]
        ptr = 0
        for i in list(range(0, a)) + list(range(b, n)):
            child[i] = fill[ptr]
            ptr += 1
        return [c for c in child if c is not None]

    return make_child(p1, p2), make_child(p2, p1)


def swap_mutation(perm: List[str], p: float, rng: random.Random) -> None:
    if rng.random() > p:
        return
    i, j = rng.sample(range(len(perm)), 2)
    perm[i], perm[j] = perm[j], perm[i]


def group_swap_mutation(perm: List[str], group_size: int, p: float, rng: random.Random) -> None:
    if rng.random() > p:
        return
    n_groups = len(perm) // group_size
    g1, g2 = rng.sample(range(n_groups), 2)
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
    pcfg: ProblemConfig
    group_cache: Dict[Tuple[str, ...], Dict[str, float]]


def evaluate_group_cached(ctx: EvalContext, group: Sequence[str]) -> Dict[str, float]:
    key = tuple(sorted(group))
    if key in ctx.group_cache:
        return ctx.group_cache[key]

    res = ctx.evaluator.evaluate_group(list(group), lateral_to_node=ctx.lateral_to_node, q_lateral=ctx.pcfg.q_lateral)
    ctx.group_cache[key] = res.pressures
    return res.pressures


def evaluate_perm(ctx: EvalContext, perm: Sequence[str]) -> Individual:
    groups = perm_to_groups(perm, ctx.pcfg.group_size, ctx.pcfg.n_groups)

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
    pen_pair = pair_split_penalty(perm, ctx.pcfg.group_size)

    return Individual(perm=list(perm), f=(var_s, mean_s, pen_pair), violation=violation, feasible=feasible)


# =========================
# I/O helpers
# =========================

@dataclass
class IterLog:
    it: int
    feasible: int
    pop: int
    feasible_ratio: float
    archive_size: int
    best_var: float
    best_mean: float
    best_pair: float


def write_convergence_csv(path: Path, logs: Sequence[IterLog]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "iter",
            "feasible",
            "pop",
            "feasible_ratio",
            "archive_size",
            "best_var",
            "best_mean",
            "best_pair",
        ])
        for r in logs:
            w.writerow([
                r.it,
                r.feasible,
                r.pop,
                r.feasible_ratio,
                r.archive_size,
                r.best_var,
                r.best_mean,
                r.best_pair,
            ])


def write_pareto_csv(path: Path, pareto: Sequence[Individual]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["var", "mean", "pair_penalty", "violation", "feasible"])
        for ind in sorted(pareto, key=lambda x: (x.violation, x.f[0], x.f[1], x.f[2])):
            w.writerow([ind.f[0], ind.f[1], ind.f[2], ind.violation, int(ind.feasible)])


def write_chosen_groups(path: Path, chosen: Individual, pcfg: ProblemConfig) -> None:
    groups = perm_to_groups(chosen.perm, pcfg.group_size, pcfg.n_groups)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"feasible={chosen.feasible} violation={chosen.violation}\n")
        f.write(f"objectives: var={chosen.f[0]}, mean={chosen.f[1]}, pair_penalty={chosen.f[2]}\n\n")
        for i, g in enumerate(groups, 1):
            f.write(f"Group {i:02d}: {g}\n")


# =========================
# NSGA-II
# =========================


@dataclass(frozen=True)
class NSGA2Config:
    pop_size: int = 200
    generations: int = 2000
    p_crossover: float = 0.9
    p_mutation: float = 0.30
    tournament_k: int = 2


@dataclass
class _NSGAInd:
    perm: List[str]
    f: Tuple[float, float, float]
    violation: float
    feasible: bool
    rank: int = 10**9
    crowding: float = 0.0


def _nsga_dominates(a: _NSGAInd, b: _NSGAInd) -> bool:
    """Deb's constraint-domination (same rule as other algorithms)."""
    if a.feasible and not b.feasible:
        return True
    if not a.feasible and b.feasible:
        return False
    if not a.feasible and not b.feasible:
        return a.violation < b.violation
    return all(x <= y for x, y in zip(a.f, b.f)) and any(x < y for x, y in zip(a.f, b.f))


def _nsga_fast_non_dominated_sort(pop: List[_NSGAInd]) -> List[List[int]]:
    """Fast non-dominated sorting (O(N^2)), sets ind.rank."""
    S = [set() for _ in pop]
    n = [0 for _ in pop]
    fronts: List[List[int]] = [[]]

    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if _nsga_dominates(pop[p], pop[q]):
                S[p].add(q)
            elif _nsga_dominates(pop[q], pop[p]):
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


def _nsga_crowding_distance(pop: List[_NSGAInd], front: List[int]) -> None:
    if not front:
        return
    m = 3
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


def _nsga_tournament_select(pop: List[_NSGAInd], k: int, rng: random.Random) -> _NSGAInd:
    cand = rng.sample(pop, k)
    cand.sort(key=lambda ind: (ind.rank, -ind.crowding, ind.violation))
    return cand[0]


def _nsga_wrap(ind: Individual) -> _NSGAInd:
    return _NSGAInd(perm=list(ind.perm), f=ind.f, violation=ind.violation, feasible=bool(ind.feasible))


def run_nsga2(
    ctx: EvalContext,
    lateral_ids: List[str],
    pcfg: ProblemConfig,
    cfg: NSGA2Config,
    seed: int,
    init_perms: Optional[Sequence[Sequence[str]]] = None,
) -> Tuple[Individual, List[Individual], List[IterLog]]:
    """NSGA-II runner.

    Returns the same data types as SPEA2/MOEA-D/MOPSO:
      - chosen: Individual
      - pareto: List[Individual] (feasible nondominated set)
      - logs  : List[IterLog]

    init_perms is used in dynamic mode to warm-start the initial population.
    """
    rng = random.Random(int(seed))

    # ---- build initial population (warm-start -> paired -> random) ----
    pop: List[_NSGAInd] = []
    seen = set()

    def _try_add(perm: Sequence[str]) -> None:
        if len(pop) >= int(cfg.pop_size):
            return
        if len(perm) != len(lateral_ids):
            return
        t = tuple(perm)
        if t in seen:
            return
        seen.add(t)
        pop.append(_nsga_wrap(evaluate_perm(ctx, perm)))

    if init_perms:
        for p in init_perms:
            _try_add(p)

    bases = sorted({x.rsplit("_", 1)[0] for x in lateral_ids})
    n_paired = max(1, int(cfg.pop_size) // 5)
    for _ in range(n_paired):
        if len(pop) >= int(cfg.pop_size):
            break
        bb = bases[:]
        rng.shuffle(bb)
        perm: List[str] = []
        for i in range(0, len(bb), 2):
            b1, b2 = bb[i], bb[i + 1]
            perm += [f"{b1}_L", f"{b1}_R", f"{b2}_L", f"{b2}_R"]
        _try_add(perm)

    while len(pop) < int(cfg.pop_size):
        perm = lateral_ids[:]
        rng.shuffle(perm)
        _try_add(perm)

    logs: List[IterLog] = []

    # ---- main loop ----
    for gen in range(1, int(cfg.generations) + 1):
        fronts = _nsga_fast_non_dominated_sort(pop)
        for fr in fronts:
            _nsga_crowding_distance(pop, fr)

        feas = [ind for ind in pop if ind.feasible]
        feas_ratio = (len(feas) / len(pop)) if pop else 0.0

        front0_feas = [pop[i] for i in fronts[0] if pop[i].feasible] if fronts else []

        if front0_feas:
            best = min(front0_feas, key=lambda ind: (ind.f[0], ind.f[1], ind.f[2]))
            best_var, best_mean, best_pair = best.f
        elif feas:
            best = min(feas, key=lambda ind: (ind.f[0], ind.f[1], ind.f[2]))
            best_var, best_mean, best_pair = best.f
        else:
            best_var, best_mean, best_pair = float("inf"), float("inf"), float("inf")

        logs.append(
            IterLog(
                it=gen,
                feasible=len(feas),
                pop=len(pop),
                feasible_ratio=feas_ratio,
                archive_size=len(front0_feas),
                best_var=best_var,
                best_mean=best_mean,
                best_pair=best_pair,
            )
        )

        # offspring
        offspring: List[_NSGAInd] = []
        while len(offspring) < int(cfg.pop_size):
            p1 = _nsga_tournament_select(pop, int(cfg.tournament_k), rng)
            p2 = _nsga_tournament_select(pop, int(cfg.tournament_k), rng)

            c1p, c2p = list(p1.perm), list(p2.perm)
            if rng.random() < float(cfg.p_crossover):
                c1p, c2p = order_crossover(p1.perm, p2.perm, rng)

            swap_mutation(c1p, float(cfg.p_mutation), rng)
            swap_mutation(c2p, float(cfg.p_mutation), rng)
            group_swap_mutation(c1p, pcfg.group_size, float(cfg.p_mutation), rng)
            group_swap_mutation(c2p, pcfg.group_size, float(cfg.p_mutation), rng)

            offspring.append(_nsga_wrap(evaluate_perm(ctx, c1p)))
            if len(offspring) < int(cfg.pop_size):
                offspring.append(_nsga_wrap(evaluate_perm(ctx, c2p)))

        # elitist selection
        combined = pop + offspring
        fronts = _nsga_fast_non_dominated_sort(combined)
        for fr in fronts:
            _nsga_crowding_distance(combined, fr)

        new_pop: List[_NSGAInd] = []
        for fr in fronts:
            if len(new_pop) + len(fr) <= int(cfg.pop_size):
                new_pop.extend([combined[i] for i in fr])
            else:
                fr_sorted = sorted(fr, key=lambda i: combined[i].crowding, reverse=True)
                needed = int(cfg.pop_size) - len(new_pop)
                new_pop.extend([combined[i] for i in fr_sorted[:needed]])
                break
        pop = new_pop

    # final pareto
    fronts = _nsga_fast_non_dominated_sort(pop)
    pareto_nsga = [pop[i] for i in fronts[0] if pop[i].feasible] if fronts else []

    if pareto_nsga:
        pareto = [Individual(perm=list(ind.perm), f=ind.f, violation=ind.violation, feasible=True) for ind in pareto_nsga]
        chosen = pick_one_by_weighted_sum(pareto, pcfg)
        return chosen, pareto, logs

    # no feasible: return lowest-violation
    best = min(pop, key=lambda ind: float(ind.violation))
    chosen = Individual(perm=list(best.perm), f=best.f, violation=best.violation, feasible=False)
    return chosen, [], logs


# =========================
# SPEA2
# =========================

@dataclass(frozen=True)
class SPEA2Config:
    pop_size: int = 200
    iterations: int = 2000
    archive_size: int = 200

    p_crossover: float = 0.9
    p_mutation: float = 0.25  # base, will be self-adapted

    # self-adaptation
    target_feasible_ratio: float = 0.70
    p_mut_min: float = 0.10
    p_mut_max: float = 0.60
    stagnation_window: int = 80


def _pseudo_obj_for_density(ind: Individual, vio_scale: float = 1e3) -> Tuple[float, float, float]:
    if ind.feasible:
        return ind.f
    # Push infeasible away in objective space to avoid density artifacts
    v = ind.violation
    return (ind.f[0] + vio_scale * v, ind.f[1] + vio_scale * v, ind.f[2] + vio_scale * v)


def _euclid(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _spea2_fitness(pool: Sequence[Individual]) -> List[float]:
    """Compute SPEA2 fitness for a pool (pop + archive).

    Fitness = raw_fitness + density
      raw_fitness = sum_{j dominates i} strength(j)
      strength(j) = number of solutions dominated by j
      density = 1 / (sigma_k + 2)
    Dominance uses Deb constraint-domination.
    """
    n = len(pool)
    strength = [0] * n
    dominates_set: List[List[int]] = [[] for _ in range(n)]
    dominated_by: List[List[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(pool[i], pool[j]):
                dominates_set[i].append(j)
            elif dominates(pool[j], pool[i]):
                dominated_by[i].append(j)

    for i in range(n):
        strength[i] = len(dominates_set[i])

    raw = [0.0] * n
    for i in range(n):
        s = 0
        for j in dominated_by[i]:
            s += strength[j]
        raw[i] = float(s)

    # Density: distance to k-th nearest neighbor in objective space
    objs = [_pseudo_obj_for_density(ind) for ind in pool]
    k = max(1, int(math.sqrt(n)))
    density = [0.0] * n
    for i in range(n):
        dists = sorted(_euclid(objs[i], objs[j]) for j in range(n) if j != i)
        sigma_k = dists[min(k - 1, len(dists) - 1)] if dists else 0.0
        density[i] = 1.0 / (sigma_k + 2.0)

    return [raw[i] + density[i] for i in range(n)]


def _spea2_environmental_selection(pool: Sequence[Individual], fitness: Sequence[float], archive_size: int) -> List[Individual]:
    """Select archive from pool.

    Standard SPEA2:
      - Include all with fitness < 1
      - If too many: truncate by density (remove most crowded)
      - If too few: fill with best fitness
    """
    pool = list(pool)
    fitness = list(fitness)
    archive = [pool[i] for i in range(len(pool)) if fitness[i] < 1.0]

    if len(archive) > archive_size:
        # Truncation based on nearest-neighbor distances (objective space)
        objs = [_pseudo_obj_for_density(ind) for ind in archive]
        # Iteratively remove the individual with minimal distance to others
        while len(archive) > archive_size:
            m = len(archive)
            dist_matrix = [[0.0] * m for _ in range(m)]
            for i in range(m):
                for j in range(i + 1, m):
                    d = _euclid(objs[i], objs[j])
                    dist_matrix[i][j] = d
                    dist_matrix[j][i] = d
            # For each i, sort distances to others
            nn_sorted = [sorted(dist_matrix[i][j] for j in range(m) if j != i) for i in range(m)]
            # Find most crowded: lexicographically smallest nn distance vector
            remove_idx = min(range(m), key=lambda i: nn_sorted[i])
            archive.pop(remove_idx)
            objs.pop(remove_idx)

    elif len(archive) < archive_size:
        # Fill with best fitness (ascending)
        idx_sorted = sorted(range(len(pool)), key=lambda i: fitness[i])
        for i in idx_sorted:
            if pool[i] in archive:
                continue
            archive.append(pool[i])
            if len(archive) >= archive_size:
                break

    return archive


def _spea2_tournament(archive: Sequence[Individual], fitness_map: Dict[int, float], rng: random.Random) -> Individual:
    i, j = rng.sample(range(len(archive)), 2)
    a = archive[i]
    b = archive[j]
    # Lower fitness is better
    fa = fitness_map[id(a)]
    fb = fitness_map[id(b)]
    if fa < fb:
        return a
    if fb < fa:
        return b
    # Tie-break by smaller violation then objectives
    if a.violation != b.violation:
        return a if a.violation < b.violation else b
    return a if a.f < b.f else b


def run_spea2(
    ctx: EvalContext,
    lateral_ids: List[str],
    pcfg: ProblemConfig,
    cfg: SPEA2Config,
    seed: int,
) -> Tuple[Individual, List[Individual], List[IterLog]]:
    rng = random.Random(seed)

    # Initial population: mixture of paired (pen_pair=0) and random
    bases = sorted({x.rsplit("_", 1)[0] for x in lateral_ids})
    pop: List[Individual] = []

    n_paired = max(1, cfg.pop_size // 5)
    for _ in range(n_paired):
        bb = bases[:]
        rng.shuffle(bb)
        perm: List[str] = []
        for i in range(0, len(bb), 2):
            b1, b2 = bb[i], bb[i + 1]
            perm += [f"{b1}_L", f"{b1}_R", f"{b2}_L", f"{b2}_R"]
        pop.append(evaluate_perm(ctx, perm))

    while len(pop) < cfg.pop_size:
        perm = lateral_ids[:]
        rng.shuffle(perm)
        pop.append(evaluate_perm(ctx, perm))

    archive: List[Individual] = []
    logs: List[IterLog] = []

    # Self-adaptive mutation
    p_mut = cfg.p_mutation
    best_var_hist: List[float] = []

    for it in range(1, cfg.iterations + 1):
        pool = pop + archive
        fit = _spea2_fitness(pool)
        archive = _spea2_environmental_selection(pool, fit, cfg.archive_size)

        # Build fitness map for tournament selection (archive only)
        # Recompute fitness for the *archive* in the current pool context
        # For stability, map by object id.
        fit_map: Dict[int, float] = {}
        for ind, fval in zip(pool, fit):
            fit_map[id(ind)] = fval

        feas = [ind for ind in pop if ind.feasible]
        feas_ratio = (len(feas) / len(pop)) if pop else 0.0

        # Best feasible in archive (preferred)
        arch_feas = [ind for ind in archive if ind.feasible]
        if arch_feas:
            best = min(arch_feas, key=lambda x: (x.f[0], x.f[1], x.f[2]))
        elif feas:
            best = min(feas, key=lambda x: (x.f[0], x.f[1], x.f[2]))
        else:
            best = min(pop, key=lambda x: x.violation)

        logs.append(
            IterLog(
                it=it,
                feasible=len(feas),
                pop=len(pop),
                feasible_ratio=feas_ratio,
                archive_size=len(archive),
                best_var=best.f[0] if best.feasible else float("inf"),
                best_mean=best.f[1] if best.feasible else float("inf"),
                best_pair=best.f[2] if best.feasible else float("inf"),
            )
        )

        if best.feasible:
            best_var_hist.append(best.f[0])

        # --- self-adaptation of mutation rate ---
        if feas_ratio < cfg.target_feasible_ratio:
            p_mut = min(cfg.p_mut_max, p_mut * 1.05 + 0.01)
        else:
            p_mut = max(cfg.p_mut_min, p_mut * 0.98)

        if len(best_var_hist) > cfg.stagnation_window:
            recent = best_var_hist[-cfg.stagnation_window :]
            if min(recent) >= min(best_var_hist[:-cfg.stagnation_window]) - 1e-12:
                p_mut = min(cfg.p_mut_max, p_mut + 0.05)

        # Reproduction: build next population
        next_pop: List[Individual] = []
        if not archive:
            archive = pop[:]  # fallback

        while len(next_pop) < cfg.pop_size:
            p1 = _spea2_tournament(archive, fit_map, rng)
            p2 = _spea2_tournament(archive, fit_map, rng)

            c1p, c2p = list(p1.perm), list(p2.perm)
            if rng.random() < cfg.p_crossover:
                c1p, c2p = order_crossover(p1.perm, p2.perm, rng)

            swap_mutation(c1p, p_mut, rng)
            swap_mutation(c2p, p_mut, rng)
            group_swap_mutation(c1p, pcfg.group_size, p_mut, rng)
            group_swap_mutation(c2p, pcfg.group_size, p_mut, rng)

            next_pop.append(evaluate_perm(ctx, c1p))
            if len(next_pop) < cfg.pop_size:
                next_pop.append(evaluate_perm(ctx, c2p))

        pop = next_pop

    pareto = nondominated_filter(archive) if archive else nondominated_filter(pop)
    if not pareto:
        chosen = min(archive if archive else pop, key=lambda x: x.violation)
        return chosen, [], logs

    chosen = pick_one_by_weighted_sum(pareto, pcfg)
    return chosen, pareto, logs


# =========================
# MOEA/D
# =========================

@dataclass(frozen=True)
class MOEADConfig:
    pop_size: int = 200
    iterations: int = 2000

    neighborhood_size: int = 200

    p_crossover: float = 0.9
    p_mutation: float = 0.25  # base, will be self-adapted

    # self-adaptation
    target_feasible_ratio: float = 0.70
    p_mut_min: float = 0.10
    p_mut_max: float = 0.60
    stagnation_window: int = 100


def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num = 1
    den = 1
    for i in range(1, k + 1):
        num *= n - (k - i)
        den *= i
    return num // den


def generate_weights_3obj(n_weights: int) -> List[Tuple[float, float, float]]:
    """Simplex-lattice weight generation for 3 objectives."""
    # Find smallest H such that C(H+2,2) >= n
    H = 1
    while _comb(H + 2, 2) < n_weights:
        H += 1

    weights: List[Tuple[float, float, float]] = []
    for i in range(H + 1):
        for j in range(H + 1 - i):
            k = H - i - j
            w1 = i / H
            w2 = j / H
            w3 = k / H
            weights.append((w1, w2, w3))
            if len(weights) >= n_weights:
                return weights
    return weights[:n_weights]


def _weight_dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _tchebycheff(ind: Individual, w: Tuple[float, float, float], z: Tuple[float, float, float]) -> float:
    eps = 1e-12
    return max(
        (w[0] if w[0] > eps else eps) * abs(ind.f[0] - z[0]),
        (w[1] if w[1] > eps else eps) * abs(ind.f[1] - z[1]),
        (w[2] if w[2] > eps else eps) * abs(ind.f[2] - z[2]),
    )


def _better_for_subproblem(a: Individual, b: Individual, w: Tuple[float, float, float], z: Tuple[float, float, float]) -> bool:
    """Comparator for MOEA/D replacement.

    Priority:
      feasible > infeasible
      lower violation (if both infeasible)
      lower scalar value (if both feasible)
    """
    if a.feasible and not b.feasible:
        return True
    if not a.feasible and b.feasible:
        return False
    if not a.feasible and not b.feasible:
        return a.violation < b.violation

    return _tchebycheff(a, w, z) < _tchebycheff(b, w, z)


def run_moead(
    ctx: EvalContext,
    lateral_ids: List[str],
    pcfg: ProblemConfig,
    cfg: MOEADConfig,
    seed: int,
) -> Tuple[Individual, List[Individual], List[IterLog]]:
    rng = random.Random(seed)

    weights = generate_weights_3obj(cfg.pop_size)

    # Neighborhood in weight space
    neigh: List[List[int]] = []
    for i, wi in enumerate(weights):
        d = [(j, _weight_dist(wi, wj)) for j, wj in enumerate(weights) if j != i]
        d.sort(key=lambda x: x[1])
        neigh.append([j for j, _ in d[: cfg.neighborhood_size]])

    # Init population (paired + random)
    bases = sorted({x.rsplit("_", 1)[0] for x in lateral_ids})
    pop: List[Individual] = []

    n_paired = max(1, cfg.pop_size // 5)
    for _ in range(n_paired):
        bb = bases[:]
        rng.shuffle(bb)
        perm: List[str] = []
        for i in range(0, len(bb), 2):
            b1, b2 = bb[i], bb[i + 1]
            perm += [f"{b1}_L", f"{b1}_R", f"{b2}_L", f"{b2}_R"]
        pop.append(evaluate_perm(ctx, perm))

    while len(pop) < cfg.pop_size:
        perm = lateral_ids[:]
        rng.shuffle(perm)
        pop.append(evaluate_perm(ctx, perm))

    # Ideal point z (objective minima among feasible; if none feasible, use current objectives anyway)
    def update_ideal(z: Tuple[float, float, float], ind: Individual) -> Tuple[float, float, float]:
        return (min(z[0], ind.f[0]), min(z[1], ind.f[1]), min(z[2], ind.f[2]))

    z = (float("inf"), float("inf"), float("inf"))
    for ind in pop:
        if ind.feasible:
            z = update_ideal(z, ind)
    if any(math.isinf(x) for x in z):
        # No feasible yet
        for ind in pop:
            z = update_ideal(z, ind)

    logs: List[IterLog] = []

    p_mut = cfg.p_mutation
    best_var_hist: List[float] = []

    for it in range(1, cfg.iterations + 1):
        feas = [ind for ind in pop if ind.feasible]
        feas_ratio = (len(feas) / len(pop)) if pop else 0.0

        # log best feasible in population
        if feas:
            best = min(feas, key=lambda x: (x.f[0], x.f[1], x.f[2]))
            best_var_hist.append(best.f[0])
            best_var, best_mean, best_pair = best.f
        else:
            best = min(pop, key=lambda x: x.violation)
            best_var, best_mean, best_pair = float("inf"), float("inf"), float("inf")

        # archive is approximated by nondominated set of current population
        archive = nondominated_filter(pop)

        logs.append(
            IterLog(
                it=it,
                feasible=len(feas),
                pop=len(pop),
                feasible_ratio=feas_ratio,
                archive_size=len(archive),
                best_var=best_var,
                best_mean=best_mean,
                best_pair=best_pair,
            )
        )

        # --- self-adaptation of mutation rate ---
        if feas_ratio < cfg.target_feasible_ratio:
            p_mut = min(cfg.p_mut_max, p_mut * 1.05 + 0.01)
        else:
            p_mut = max(cfg.p_mut_min, p_mut * 0.98)

        if len(best_var_hist) > cfg.stagnation_window:
            recent = best_var_hist[-cfg.stagnation_window :]
            if min(recent) >= min(best_var_hist[:-cfg.stagnation_window]) - 1e-12:
                p_mut = min(cfg.p_mut_max, p_mut + 0.05)

        # Main MOEA/D loop: update each subproblem
        for i in range(cfg.pop_size):
            P = neigh[i]
            a, b = rng.sample(P, 2) if len(P) >= 2 else (i, rng.randrange(cfg.pop_size))
            p1, p2 = pop[a], pop[b]

            child_perm = list(p1.perm)
            if rng.random() < cfg.p_crossover:
                c1, _ = order_crossover(p1.perm, p2.perm, rng)
                child_perm = c1

            swap_mutation(child_perm, p_mut, rng)
            group_swap_mutation(child_perm, pcfg.group_size, p_mut, rng)

            child = evaluate_perm(ctx, child_perm)

            # update ideal point
            if child.feasible or all(math.isfinite(x) for x in z):
                z = (min(z[0], child.f[0]), min(z[1], child.f[1]), min(z[2], child.f[2]))

            # replacement in neighborhood
            for j in P + [i]:
                if _better_for_subproblem(child, pop[j], weights[j], z):
                    pop[j] = child

    pareto = nondominated_filter(pop)
    if not pareto:
        chosen = min(pop, key=lambda x: x.violation)
        return chosen, [], logs

    chosen = pick_one_by_weighted_sum(pareto, pcfg)
    return chosen, pareto, logs


# =========================
# MOPSO (random-key permutation)
# =========================

@dataclass(frozen=True)
class MOPSOConfig:
    swarm_size: int = 200
    iterations: int = 2000
    archive_size: int = 200

    # PSO params
    w_start: float = 0.90
    w_end: float = 0.40
    c1: float = 1.50
    c2: float = 1.50
    v_max: float = 0.50  # velocity clamp

    # mutation / restart
    p_mutation: float = 0.15
    restart_fraction: float = 0.10
    stagnation_window: int = 120


@dataclass
class Particle:
    x: List[float]
    v: List[float]
    pbest_x: List[float]
    pbest_ind: Individual
    ind: Individual


def _keys_to_perm(keys: Sequence[float], items: Sequence[str]) -> List[str]:
    idx = sorted(range(len(keys)), key=lambda i: keys[i])
    return [items[i] for i in idx]


def _crowding_distance(front: Sequence[Individual]) -> List[float]:
    """Crowding distance for archive leader selection / pruning (feasible front assumed)."""
    n = len(front)
    if n == 0:
        return []
    if n == 1:
        return [float("inf")]

    m = 3
    cd = [0.0] * n
    for k in range(m):
        order = sorted(range(n), key=lambda i: front[i].f[k])
        cd[order[0]] = float("inf")
        cd[order[-1]] = float("inf")
        fmin = front[order[0]].f[k]
        fmax = front[order[-1]].f[k]
        denom = (fmax - fmin) if (fmax - fmin) != 0.0 else 1.0
        for t in range(1, n - 1):
            prev_f = front[order[t - 1]].f[k]
            next_f = front[order[t + 1]].f[k]
            cd[order[t]] += (next_f - prev_f) / denom
    return cd


def _update_archive(archive: List[Individual], cand: Iterable[Individual], max_size: int, rng: random.Random) -> List[Individual]:
    pool = list(archive) + [c for c in cand]
    front = nondominated_filter(pool)
    if len(front) <= max_size:
        return front

    # prune by crowding distance (remove smallest)
    while len(front) > max_size:
        cd = _crowding_distance(front)
        # remove the most crowded (smallest cd), but avoid inf
        idx = min(range(len(front)), key=lambda i: cd[i] if math.isfinite(cd[i]) else float("inf"))
        front.pop(idx)
    return front


def _select_leader(archive: Sequence[Individual], rng: random.Random) -> Individual:
    """Select a leader from archive, biased to less-crowded solutions."""
    if len(archive) == 1:
        return archive[0]
    cd = _crowding_distance(archive)
    # roulette on cd (replace inf by max finite)
    finite = [c for c in cd if math.isfinite(c)]
    base = max(finite) if finite else 1.0
    w = [(c if math.isfinite(c) else base) for c in cd]
    s = sum(w)
    if s <= 0:
        return rng.choice(list(archive))
    r = rng.random() * s
    acc = 0.0
    for ind, wi in zip(archive, w):
        acc += wi
        if acc >= r:
            return ind
    return archive[-1]


def _mutate_keys(keys: List[float], p: float, rng: random.Random) -> None:
    if rng.random() > p:
        return
    # two simple discrete-friendly mutations
    if rng.random() < 0.5:
        i, j = rng.sample(range(len(keys)), 2)
        keys[i], keys[j] = keys[j], keys[i]
    else:
        i = rng.randrange(len(keys))
        keys[i] += rng.gauss(0.0, 0.1)


def run_mopso(
    ctx: EvalContext,
    lateral_ids: List[str],
    pcfg: ProblemConfig,
    cfg: MOPSOConfig,
    seed: int,
) -> Tuple[Individual, List[Individual], List[IterLog]]:
    rng = random.Random(seed)

    dim = len(lateral_ids)

    # init archive, swarm
    swarm: List[Particle] = []

    # heuristic init: some paired-friendly keys
    bases = sorted({x.rsplit("_", 1)[0] for x in lateral_ids})

    def init_keys_paired() -> List[float]:
        bb = bases[:]
        rng.shuffle(bb)
        perm: List[str] = []
        for i in range(0, len(bb), 2):
            b1, b2 = bb[i], bb[i + 1]
            perm += [f"{b1}_L", f"{b1}_R", f"{b2}_L", f"{b2}_R"]
        # build keys so that sorting yields this perm
        pos = {lat: i for i, lat in enumerate(perm)}
        keys = [0.0] * dim
        for i, lat in enumerate(lateral_ids):
            # small noise to break ties
            keys[i] = float(pos[lat]) + rng.random() * 1e-3
        return keys

    n_paired = max(1, cfg.swarm_size // 5)
    for i in range(cfg.swarm_size):
        if i < n_paired:
            x = init_keys_paired()
        else:
            x = [rng.random() for _ in range(dim)]
        v = [rng.uniform(-cfg.v_max, cfg.v_max) for _ in range(dim)]
        perm = _keys_to_perm(x, lateral_ids)
        ind = evaluate_perm(ctx, perm)
        swarm.append(Particle(x=x, v=v, pbest_x=list(x), pbest_ind=ind, ind=ind))

    archive: List[Individual] = _update_archive([], (p.ind for p in swarm), cfg.archive_size, rng)
    logs: List[IterLog] = []

    best_var_hist: List[float] = []

    for it in range(1, cfg.iterations + 1):
        # inertia schedule
        w = cfg.w_start + (cfg.w_end - cfg.w_start) * (it / cfg.iterations)

        feas = [p.ind for p in swarm if p.ind.feasible]
        feas_ratio = (len(feas) / len(swarm)) if swarm else 0.0

        arch_feas = [a for a in archive if a.feasible]
        if arch_feas:
            best = min(arch_feas, key=lambda x: (x.f[0], x.f[1], x.f[2]))
            best_var_hist.append(best.f[0])
            best_var, best_mean, best_pair = best.f
        elif feas:
            best = min(feas, key=lambda x: (x.f[0], x.f[1], x.f[2]))
            best_var_hist.append(best.f[0])
            best_var, best_mean, best_pair = best.f
        else:
            best = min((p.ind for p in swarm), key=lambda x: x.violation)
            best_var, best_mean, best_pair = float("inf"), float("inf"), float("inf")

        logs.append(
            IterLog(
                it=it,
                feasible=len(feas),
                pop=len(swarm),
                feasible_ratio=feas_ratio,
                archive_size=len(archive),
                best_var=best_var,
                best_mean=best_mean,
                best_pair=best_pair,
            )
        )

        # stagnation -> restart a small fraction of particles
        do_restart = False
        if len(best_var_hist) > cfg.stagnation_window:
            recent = best_var_hist[-cfg.stagnation_window :]
            if min(recent) >= min(best_var_hist[:-cfg.stagnation_window]) - 1e-12:
                do_restart = True

        leader = _select_leader(arch_feas if arch_feas else archive, rng) if archive else best

        # Update particles
        for p in swarm:
            # If restart: re-sample some particles (retain their pbest)
            if do_restart and rng.random() < cfg.restart_fraction:
                p.x = [rng.random() for _ in range(dim)]
                p.v = [rng.uniform(-cfg.v_max, cfg.v_max) for _ in range(dim)]

            # Leader position in key space is unknown (we only have permutation).
            # Use pbest as attractor + random perturbation, and rely on archive leader via mutation.
            # This is a common compromise for combinatorial MOPSO.

            for d in range(dim):
                r1 = rng.random()
                r2 = rng.random()
                cognitive = cfg.c1 * r1 * (p.pbest_x[d] - p.x[d])
                social = cfg.c2 * r2 * (p.pbest_x[d] - p.x[d])  # fallback to pbest (stable)
                p.v[d] = w * p.v[d] + cognitive + social
                # clamp
                if p.v[d] > cfg.v_max:
                    p.v[d] = cfg.v_max
                elif p.v[d] < -cfg.v_max:
                    p.v[d] = -cfg.v_max
                p.x[d] += p.v[d]

            _mutate_keys(p.x, cfg.p_mutation, rng)

            perm = _keys_to_perm(p.x, lateral_ids)
            p.ind = evaluate_perm(ctx, perm)

            # update personal best by constraint-domination
            if dominates(p.ind, p.pbest_ind):
                p.pbest_ind = p.ind
                p.pbest_x = list(p.x)
            elif (p.ind.feasible and p.pbest_ind.feasible) and (p.ind.f < p.pbest_ind.f):
                p.pbest_ind = p.ind
                p.pbest_x = list(p.x)
            elif (not p.ind.feasible) and (not p.pbest_ind.feasible) and (p.ind.violation < p.pbest_ind.violation):
                p.pbest_ind = p.ind
                p.pbest_x = list(p.x)

        archive = _update_archive(archive, (p.ind for p in swarm), cfg.archive_size, rng)

    pareto = [a for a in archive if a.feasible]
    if not pareto:
        chosen = min((p.ind for p in swarm), key=lambda x: x.violation)
        return chosen, [], logs

    chosen = pick_one_by_weighted_sum(pareto, pcfg)
    return chosen, pareto, logs


# =========================
# Dynamic runner (scenario warm-start)
# =========================

@dataclass(frozen=True)
class DynamicConfig:
    per_scenario_iterations: int = 800


def _build_context(
    nodes_xlsx: str,
    pipes_xlsx: str,
    root: str,
    H0: float,
    Hmin: float,
    pcfg: ProblemConfig,
) -> Tuple[EvalContext, List[str]]:
    nodes = load_nodes_xlsx(nodes_xlsx)
    edges = load_pipes_xlsx(pipes_xlsx)

    evaluator = TreeHydraulicEvaluator(nodes=nodes, edges=edges, root=root, H0=H0, Hmin=Hmin)

    field_nodes = [nid for nid in nodes.keys() if is_field_node_id(nid)]
    lateral_ids, lateral_to_node = build_lateral_ids_for_field_nodes(field_nodes)

    if len(lateral_ids) != pcfg.group_size * pcfg.n_groups:
        raise ValueError(f"Expect {pcfg.group_size*pcfg.n_groups} laterals but got {len(lateral_ids)}")

    ctx = EvalContext(evaluator=evaluator, lateral_to_node=lateral_to_node, pcfg=pcfg, group_cache={})
    return ctx, lateral_ids


def _inject_warmstart_population(
    lateral_ids: List[str],
    warm_perms: Sequence[Sequence[str]],
    needed: int,
    rng: random.Random,
) -> List[List[str]]:
    out: List[List[str]] = []
    for p in warm_perms:
        if len(out) >= needed:
            break
        out.append(list(p))
    while len(out) < needed:
        perm = lateral_ids[:]
        rng.shuffle(perm)
        out.append(perm)
    return out


def run_dynamic(
    algo: str,
    nodes_xlsx: str,
    pipes_xlsx: str,
    root: str,
    Hmin: float,
    H0_list: List[float],
    pcfg: ProblemConfig,
    outdir: str,
    seed: int,
) -> None:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    warm_archive_perms: List[List[str]] = []

    summary_rows: List[List[object]] = []

    for sidx, H0 in enumerate(H0_list, 1):
        scen_dir = out_path / f"scenario_{sidx:02d}_H0_{H0:g}"
        scen_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()

        ctx, lateral_ids = _build_context(nodes_xlsx, pipes_xlsx, root, H0, Hmin, pcfg)

        # Warm-start by priming the cache with warm solutions' groups (optional) and by seeding RNG.
        rng = random.Random(seed + sidx)
        if warm_archive_perms:
            # Evaluate a few warm perms to seed cache and to ensure feasibility/objective update
            for p in warm_archive_perms[: min(50, len(warm_archive_perms))]:
                _ = evaluate_perm(ctx, p)

        # Run algorithm (iterations shortened per scenario)
        if algo == "nsga2":
            cfg = NSGA2Config(pop_size=200, generations=2000, p_crossover=0.9, p_mutation=0.30, tournament_k=2)
            chosen, pareto, logs = run_nsga2(ctx, lateral_ids, pcfg, cfg, seed + sidx, init_perms=warm_archive_perms)
        elif algo == "spea2":
            cfg = SPEA2Config(pop_size=200, iterations=2000, archive_size=200)
            chosen, pareto, logs = run_spea2(ctx, lateral_ids, pcfg, cfg, seed + sidx)
        elif algo == "moead":
            cfg = MOEADConfig(pop_size=200, iterations=2000, neighborhood_size=200)
            chosen, pareto, logs = run_moead(ctx, lateral_ids, pcfg, cfg, seed + sidx)
        elif algo == "mopso":
            cfg = MOPSOConfig(swarm_size=200, iterations=2000, archive_size=200)
            chosen, pareto, logs = run_mopso(ctx, lateral_ids, pcfg, cfg, seed + sidx)
        else:
            raise ValueError("algo must be one of: nsga2, spea2, moead, mopso")

        # Persist per-scenario results
        write_convergence_csv(scen_dir / "convergence.csv", logs)
        write_pareto_csv(scen_dir / "pareto.csv", pareto)
        write_chosen_groups(scen_dir / "chosen_groups.txt", chosen, pcfg)

        elapsed = time.perf_counter() - t0
        with (scen_dir / "timing.txt").open("w", encoding="utf-8") as f:
            f.write(f"seconds={elapsed:.6f}\n")

        # Update warm archive perms (carry over feasible Pareto set; else carry best)
        if pareto:
            warm_archive_perms = [ind.perm for ind in pareto]
        else:
            warm_archive_perms = [chosen.perm]

        summary_rows.append([
            sidx,
            H0,
            int(chosen.feasible),
            float(chosen.violation),
            float(chosen.f[0]),
            float(chosen.f[1]),
            float(chosen.f[2]),
            float(elapsed),
        ])

    # Summary CSV
    with (out_path / "dynamic_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "H0", "feasible", "violation", "var", "mean", "pair_penalty", "seconds"])
        w.writerows(summary_rows)


def _run_static_algo(
    algo: str,
    ctx: EvalContext,
    lateral_ids: List[str],
    pcfg: ProblemConfig,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[Individual, List[Individual], List[IterLog]]:
    """Dispatch helper shared by normal CLI run and weight_scan."""
    a = str(algo).lower()
    if a == "nsga2":
        cfg = NSGA2Config(pop_size=int(args.pop), generations=int(args.iters), p_crossover=0.9, p_mutation=0.30, tournament_k=2)
        return run_nsga2(ctx, lateral_ids, pcfg, cfg, int(seed))
    if a == "spea2":
        cfg = SPEA2Config(pop_size=int(args.pop), iterations=int(args.iters), archive_size=int(args.archive))
        return run_spea2(ctx, lateral_ids, pcfg, cfg, int(seed))
    if a == "moead":
        cfg = MOEADConfig(pop_size=int(args.pop), iterations=int(args.iters), neighborhood_size=int(args.neigh))
        return run_moead(ctx, lateral_ids, pcfg, cfg, int(seed))
    if a == "mopso":
        cfg = MOPSOConfig(swarm_size=int(args.swarm), iterations=int(args.iters), archive_size=int(args.archive))
        return run_mopso(ctx, lateral_ids, pcfg, cfg, int(seed))
    raise ValueError("algo must be one of: nsga2, spea2, moead, mopso")


def run_weight_scan_cli(args: argparse.Namespace) -> None:
    """Weight scan over *preference weights* used to pick one solution from the Pareto archive.

    For each replicate seed:
      1) run the chosen multi-objective algorithm once to get a Pareto archive
      2) for each weight vector w, pick a recommended schedule from that archive (min-max normalized weighted sum)
    Then aggregate per-weight average performance points across replicates.
    """
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    pcfg = ProblemConfig(q_lateral=float(args.q_lateral))

    ctx, lateral_ids = _build_context(args.nodes, args.pipes, args.root, float(args.H0), float(args.Hmin), pcfg)

    # Weight set
    if args.mode == "line":
        weights = generate_weight_scan_line(int(args.grid), float(args.w_pair_fixed))
    elif args.mode == "simplex":
        if int(args.n_weights) > 0:
            weights = generate_weights_3obj(int(args.n_weights))
        else:
            weights = generate_weight_scan_simplex(int(args.simplex_H))
    else:
        raise ValueError("--mode must be 'line' or 'simplex'")

    # Per-weight stats accumulator
    acc: Dict[int, List[Tuple[int, float, Tuple[float, float, float], float]]] = {i: [] for i in range(len(weights))}
    # tuple: (feasible_int, violation, (var,mean,pair), seconds_for_run)

    raw_csv = out_path / "weight_scan_raw.csv"
    pts_csv = out_path / "weight_scan_points.csv"

    t_all = time.perf_counter()

    with raw_csv.open("w", newline="") as fraw:
        wraw = csv.writer(fraw)
        wraw.writerow([
            "algo",
            "w_idx",
            "w_var",
            "w_mean",
            "w_pair",
            "rep",
            "seed",
            "feasible",
            "violation",
            "var",
            "mean",
            "pair_penalty",
            "run_seconds",
        ])

        for rep in range(int(args.repeats)):
            run_seed = int(args.seed) + rep
            # Reset cache to keep memory bounded across replicates
            ctx.group_cache.clear()
            t0 = time.perf_counter()
            chosen0, pareto, logs = _run_static_algo(args.algo, ctx, lateral_ids, pcfg, args, run_seed)
            run_seconds = time.perf_counter() - t0

            # Keep a reference front (so you can still inspect "the front")
            if rep == 0 and int(args.keep_reference_front) == 1:
                write_pareto_csv(out_path / "pareto_reference.csv", pareto)

            if int(args.save_runs) == 1:
                rdir = out_path / f"rep_{rep:02d}_seed_{run_seed}"
                rdir.mkdir(parents=True, exist_ok=True)
                write_convergence_csv(rdir / "convergence.csv", logs)
                write_pareto_csv(rdir / "pareto.csv", pareto)
                write_chosen_groups(rdir / "chosen_groups.txt", chosen0, pcfg)
                with (rdir / "timing.txt").open("w", encoding="utf-8") as ft:
                    ft.write(f"seconds={run_seconds:.6f}\n")

            for w_idx, wvec in enumerate(weights):
                if pareto:
                    ind = pick_one_by_weighted_sum_w(pareto, wvec)
                else:
                    ind = chosen0  # fallback when no feasible archive exists
                acc[w_idx].append((int(ind.feasible), float(ind.violation), (float(ind.f[0]), float(ind.f[1]), float(ind.f[2])), float(run_seconds)))

                wraw.writerow([
                    str(args.algo),
                    w_idx,
                    wvec[0],
                    wvec[1],
                    wvec[2],
                    rep,
                    run_seed,
                    int(ind.feasible),
                    float(ind.violation),
                    float(ind.f[0]),
                    float(ind.f[1]),
                    float(ind.f[2]),
                    float(run_seconds),
                ])

    # Aggregate points (per weight)
    with pts_csv.open("w", newline="") as fpts:
        wpts = csv.writer(fpts)
        wpts.writerow([
            "algo",
            "w_idx",
            "w_var",
            "w_mean",
            "w_pair",
            "repeats",
            "feasible_rate",
            "var_mean",
            "var_std",
            "mean_mean",
            "mean_std",
            "pair_mean",
            "pair_std",
            "violation_mean",
            "violation_std",
        ])

        for w_idx, wvec in enumerate(weights):
            rows = acc[w_idx]
            feas_flags = [r[0] for r in rows]
            feas_rate = sum(feas_flags) / max(1, len(feas_flags))

            vios = [r[1] for r in rows]
            fvals = [r[2] for r in rows]

            # If some runs are infeasible, report objective averages over feasible runs (if any)
            feas_f = [fv for (flag, _, fv, _) in rows if flag == 1]
            use = feas_f if feas_f else fvals

            var_list = [u[0] for u in use]
            mean_list = [u[1] for u in use]
            pair_list = [u[2] for u in use]

            def _m(xs: List[float]) -> float:
                return float(sum(xs) / len(xs)) if xs else float("nan")

            def _s(xs: List[float]) -> float:
                if len(xs) <= 1:
                    return 0.0
                try:
                    return float(statistics.pstdev(xs))
                except Exception:
                    return 0.0

            wpts.writerow([
                str(args.algo),
                w_idx,
                wvec[0],
                wvec[1],
                wvec[2],
                int(args.repeats),
                feas_rate,
                _m(var_list),
                _s(var_list),
                _m(mean_list),
                _s(mean_list),
                _m(pair_list),
                _s(pair_list),
                _m(vios),
                _s(vios),
            ])

    elapsed_all = time.perf_counter() - t_all
    with (out_path / "weight_scan_timing.txt").open("w", encoding="utf-8") as ft:
        ft.write(f"seconds_total={elapsed_all:.6f}\n")
        ft.write(f"n_weights={len(weights)}\n")
        ft.write(f"repeats={int(args.repeats)}\n")

    print(f"Weight scan finished. Outputs: {out_path.resolve()}")
    print(f"  - {raw_csv.name} (per-replicate per-weight picks)")
    print(f"  - {pts_csv.name} (average performance point per weight)")
    if int(args.keep_reference_front) == 1:
        print("  - pareto_reference.csv (front of rep=0)")



def run_pareto_sweep_cli(args: argparse.Namespace) -> None:
    """MORL-style sweep over 2D preference weights (w_var, w_mean).

    Output is intentionally aligned with the MORL script's `pareto_sweep.csv`:
      columns = [w_var, w_mean, fail_rate, var_mean, mean_mean]

    Interpretation here:
      - We run the chosen multi-objective optimizer `repeats` times (different seeds).
      - For each replicate, we obtain a Pareto archive.
      - For each weight pair, we pick ONE schedule from that archive using only (var, mean) weights.
      - 'fail' means the picked schedule is infeasible (constraint violation > 0).
      - var_mean / mean_mean are averaged over feasible picks only (if none feasible -> NaN).
    """
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    pcfg = ProblemConfig(q_lateral=float(getattr(args, "q_lateral", 0.012)))
    ctx, lateral_ids = _build_context(args.nodes, args.pipes, args.root, float(args.H0), float(args.Hmin), pcfg)

    grid = int(args.grid)
    if grid < 2:
        raise ValueError("grid must be >= 2")

    weights_2d: List[Tuple[float, float]] = []
    for k in range(grid):
        w_var = k / float(grid - 1)
        w_mean = 1.0 - w_var
        weights_2d.append((float(w_var), float(w_mean)))

    # per-weight accumulators across repeats
    fail_cnt = [0 for _ in range(len(weights_2d))]
    var_vals: List[List[float]] = [[] for _ in range(len(weights_2d))]
    mean_vals: List[List[float]] = [[] for _ in range(len(weights_2d))]

    t_all = time.perf_counter()

    for rep in range(int(args.repeats)):
        run_seed = int(args.seed) + rep
        # keep cache bounded
        ctx.group_cache.clear()

        chosen0, pareto, logs = _run_static_algo(args.algo, ctx, lateral_ids, pcfg, args, run_seed)

        # optional: keep one reference front for inspection
        if rep == 0 and int(getattr(args, "keep_reference_front", 0)) == 1:
            write_pareto_csv(out_path / "pareto_reference.csv", pareto)

        if int(getattr(args, "save_runs", 0)) == 1:
            rdir = out_path / f"rep_{rep:02d}_seed_{run_seed}"
            rdir.mkdir(parents=True, exist_ok=True)
            write_convergence_csv(rdir / "convergence.csv", logs)
            write_pareto_csv(rdir / "pareto.csv", pareto)
            write_chosen_groups(rdir / "chosen_groups.txt", chosen0, pcfg)

        for i, (w_var, w_mean) in enumerate(weights_2d):
            if pareto:
                ind = pick_one_by_weighted_sum_2d(pareto, w_var=w_var, w_mean=w_mean)
            else:
                ind = chosen0  # fallback

            if not bool(ind.feasible):
                fail_cnt[i] += 1
            else:
                var_vals[i].append(float(ind.f[0]))
                mean_vals[i].append(float(ind.f[1]))

    def _avg(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    rows: List[Tuple[float, float, float, float, float]] = []
    for i, (w_var, w_mean) in enumerate(weights_2d):
        repeats = max(1, int(args.repeats))
        fail_rate = float(fail_cnt[i]) / float(repeats)
        rows.append((w_var, w_mean, fail_rate, _avg(var_vals[i]), _avg(mean_vals[i])))

    csv_path = out_path / "pareto_sweep.csv"
    with csv_path.open("w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["w_var", "w_mean", "fail_rate", "var_mean", "mean_mean"])
        for r in rows:
            wcsv.writerow(list(r))

    elapsed = time.perf_counter() - t_all
    with (out_path / "pareto_sweep_timing.txt").open("w", encoding="utf-8") as ft:
        ft.write(f"seconds_total={elapsed:.6f}\n")
        ft.write(f"grid={grid}\n")
        ft.write(f"repeats={int(args.repeats)}\n")
        ft.write(f"algo={str(args.algo)}\n")

    for (w_var, w_mean, fr, v, m) in rows:
        print(f"[Sweep] w_var={w_var:.2f} w_mean={w_mean:.2f} fail_rate={fr:.3f} var_mean={v:.6g} mean_mean={m:.6g}")

    print(f"[Sweep] saved: {csv_path.resolve()}")



# =========================
# CLI driver
# =========================


def _parse_float_list(s: str) -> List[float]:
    items = [x.strip() for x in s.split(",") if x.strip()]
    return [float(x) for x in items]


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic multi-objective irrigation scheduling (NSGA2/SPEA2/MOEA-D/MOPSO)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--nodes", required=True, help="Nodes.xlsx")
        p.add_argument("--pipes", required=True, help="Pipes.xlsx")
        p.add_argument("--root", required=True, help="Root node id (e.g., J0)")
        p.add_argument("--H0", type=float, required=True, help="Source head (m)")
        p.add_argument("--Hmin", type=float, required=True, help="Minimum required head (m)")
        p.add_argument("--q_lateral", type=float, default=0.012, help="Flow per lateral (m3/s)")
        p.add_argument("--out", required=True, help="Output directory")
        p.add_argument("--seed", type=int, default=20260124)

    # NSGA-II
    p0 = sub.add_parser("nsga2", help="Run NSGA-II")
    add_common(p0)
    p0.add_argument("--pop", type=int, default=200)
    p0.add_argument("--iters", type=int, default=2000, help="Generations")
    p0.add_argument("--p_mut", type=float, default=0.30, help="Mutation probability (swap/group-swap)")
    p0.add_argument("--p_cross", type=float, default=0.90, help="Crossover probability")
    p0.add_argument("--tourn", type=int, default=2, help="Tournament size")

    # SPEA2
    p1 = sub.add_parser("spea2", help="Run SPEA2")
    add_common(p1)
    p1.add_argument("--pop", type=int, default=200)
    p1.add_argument("--iters", type=int, default=1500)
    p1.add_argument("--archive", type=int, default=200)

    # MOEA/D
    p2 = sub.add_parser("moead", help="Run MOEA/D")
    add_common(p2)
    p2.add_argument("--pop", type=int, default=200)
    p2.add_argument("--iters", type=int, default=2000)
    p2.add_argument("--neigh", type=int, default=20)

    # MOPSO
    p3 = sub.add_parser("mopso", help="Run MOPSO")
    add_common(p3)
    p3.add_argument("--swarm", type=int, default=120)
    p3.add_argument("--iters", type=int, default=2000)
    p3.add_argument("--archive", type=int, default=200)

    # Weight scan (preference sweep over Pareto archive picks)
    pw = sub.add_parser("weight_scan", help="Preference weight sweep: pick one solution from Pareto for many weight vectors")
    pw.add_argument("--algo", choices=["nsga2", "spea2", "moead", "mopso"], required=True, help="Base multi-objective optimizer")
    add_common(pw)

    # weight sweep definition
    pw.add_argument("--mode", choices=["line", "simplex"], default="line",
                    help="line: sweep w_var vs w_mean with fixed w_pair; simplex: 3D simplex lattice")
    pw.add_argument("--grid", type=int, default=11, help="(mode=line) number of points along w_var in [0,1]")
    pw.add_argument("--w_pair_fixed", type=float, default=0.0, help="(mode=line) fixed w_pair before normalization")
    pw.add_argument("--simplex_H", type=int, default=10, help="(mode=simplex) lattice resolution H (yields C(H+2,2) weights)")
    pw.add_argument("--n_weights", type=int, default=0, help="(mode=simplex) if >0, generate approx this many weights (auto-H)")

    # repetition/outputs
    pw.add_argument("--repeats", type=int, default=3, help="Number of replicate runs (different seeds) to average")
    pw.add_argument("--keep_reference_front", type=int, default=1, help="Write pareto_reference.csv from rep=0")
    pw.add_argument("--save_runs", type=int, default=0, help="Also persist per-replicate runs in subfolders (larger outputs)")

    # algorithm hyperparams (shared names for convenience)
    pw.add_argument("--pop", type=int, default=200, help="SPEA2/MOEA-D population size")
    pw.add_argument("--iters", type=int, default=1500, help="Iterations")
    pw.add_argument("--archive", type=int, default=200, help="SPEA2/MOPSO archive size")
    pw.add_argument("--neigh", type=int, default=20, help="MOEA-D neighborhood size")
    pw.add_argument("--swarm", type=int, default=120, help="MOPSO swarm size")


    # MORL-style Pareto sweep (2D weights only; output matches morl_irrigation_dmoqn.py)
    ps = sub.add_parser("pareto_sweep", help="MORL-style sweep over w_var:w_mean; writes pareto_sweep.csv")
    ps.add_argument("--algo", choices=["nsga2", "spea2", "moead", "mopso"], required=True, help="Base multi-objective optimizer")
    add_common(ps)
    ps.add_argument("--grid", type=int, default=11, help="Number of points along w_var in [0,1]")
    ps.add_argument("--repeats", type=int, default=3, help="Number of replicate runs (different seeds) to average")
    ps.add_argument("--keep_reference_front", type=int, default=0, help="Write pareto_reference.csv from rep=0 (optional)")
    ps.add_argument("--save_runs", type=int, default=0, help="Also persist per-replicate runs in subfolders (larger outputs)")

    # algorithm hyperparams (same names as weight_scan for convenience)
    ps.add_argument("--pop", type=int, default=200, help="SPEA2/MOEA-D population size")
    ps.add_argument("--iters", type=int, default=1500, help="Iterations")
    ps.add_argument("--archive", type=int, default=200, help="SPEA2/MOPSO archive size")
    ps.add_argument("--neigh", type=int, default=20, help="MOEA-D neighborhood size")
    ps.add_argument("--swarm", type=int, default=120, help="MOPSO swarm size")


    # Dynamic
    pd = sub.add_parser("dynamic", help="Dynamic scenario warm-start (self-adaptive)")
    pd.add_argument("--algo", choices=["nsga2", "spea2", "moead", "mopso"], required=True)
    pd.add_argument("--nodes", required=True)
    pd.add_argument("--pipes", required=True)
    pd.add_argument("--root", required=True)
    pd.add_argument("--Hmin", type=float, required=True)
    pd.add_argument("--H0_list", required=True, help="Comma-separated H0 list, e.g. 23.75,25,26.25")
    pd.add_argument("--q_lateral", type=float, default=0.012)
    pd.add_argument("--out", required=True)
    pd.add_argument("--seed", type=int, default=20260124)

    args = parser.parse_args()

    pcfg = ProblemConfig(q_lateral=float(getattr(args, "q_lateral", 0.012)))
    if args.cmd == "weight_scan":
        run_weight_scan_cli(args)
        return

    if args.cmd == "pareto_sweep":
        run_pareto_sweep_cli(args)
        return


    if args.cmd == "dynamic":
        H0_list = _parse_float_list(args.H0_list)
        run_dynamic(
            algo=args.algo,
            nodes_xlsx=args.nodes,
            pipes_xlsx=args.pipes,
            root=args.root,
            Hmin=float(args.Hmin),
            H0_list=H0_list,
            pcfg=pcfg,
            outdir=args.out,
            seed=int(args.seed),
        )
        print(f"Dynamic run finished. Outputs: {args.out}")
        return

    # Static run
    ctx, lateral_ids = _build_context(args.nodes, args.pipes, args.root, float(args.H0), float(args.Hmin), pcfg)

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    if args.cmd == "nsga2":
        cfg = NSGA2Config(
            pop_size=int(args.pop),
            generations=int(args.iters),
            p_crossover=float(args.p_cross),
            p_mutation=float(args.p_mut),
            tournament_k=int(args.tourn),
        )
        chosen, pareto, logs = run_nsga2(ctx, lateral_ids, pcfg, cfg, int(args.seed))
    elif args.cmd == "spea2":
        cfg = SPEA2Config(pop_size=int(args.pop), iterations=int(args.iters), archive_size=int(args.archive))
        chosen, pareto, logs = run_spea2(ctx, lateral_ids, pcfg, cfg, int(args.seed))
    elif args.cmd == "moead":
        cfg = MOEADConfig(pop_size=int(args.pop), iterations=int(args.iters), neighborhood_size=int(args.neigh))
        chosen, pareto, logs = run_moead(ctx, lateral_ids, pcfg, cfg, int(args.seed))
    elif args.cmd == "mopso":
        cfg = MOPSOConfig(swarm_size=int(args.swarm), iterations=int(args.iters), archive_size=int(args.archive))
        chosen, pareto, logs = run_mopso(ctx, lateral_ids, pcfg, cfg, int(args.seed))
    else:
        raise RuntimeError("Unknown cmd")

    write_convergence_csv(out_path / "convergence.csv", logs)
    write_pareto_csv(out_path / "pareto.csv", pareto)
    write_chosen_groups(out_path / "chosen_groups.txt", chosen, pcfg)

    elapsed = time.perf_counter() - t0
    with (out_path / "timing.txt").open("w", encoding="utf-8") as f:
        f.write(f"seconds={elapsed:.6f}\n")

    print("\n=== Result ===")
    print(f"Chosen feasible={chosen.feasible} violation={chosen.violation:.6g}")
    print(f"Objectives: var={chosen.f[0]:.6g}, mean={chosen.f[1]:.6g}, pair_penalty={chosen.f[2]:.0f}")
    print(f"Pareto archive size: {len(pareto)}")
    print(f"Outputs: {out_path.resolve()}")
    print(f"Wall time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
