# morl_irrigation_tree_batchfix_with_swap_sa.py
# Integrated: (1) batch REINFORCE (optional) + (2) swap hill-climb / simulated annealing (SA)
# Design goals:
# - Reuse existing hydraulic group cache (_group_cache) and schedule_stats().
# - Local search operates by swapping two laterals between two groups (capacity preserved).
# - Log `best_cost` as a non-increasing (monotone) sequence (best-so-far).
#
# Typical usage (local search only):
#   python morl_irrigation_tree_batchfix_with_swap_sa.py --nodes_xlsx Nodes.xlsx --pipes_xlsx Pipes.xlsx \
#     --mode local --init_random 3000 --local_method sa --local_steps 50000
#
# Typical usage (train then local):
#   python morl_irrigation_tree_batchfix_with_swap_sa.py --nodes_xlsx Nodes.xlsx --pipes_xlsx Pipes.xlsx \
#     --mode train+local --episodes 500 --batch_size 32 --lr 0.03 --init_random 1000 --local_steps 30000

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

import tree_evaluator


@dataclass
class TrainingConfig:
    # Policy-gradient training (optional)
    episodes: int = 500
    batch_size: int = 32
    lr: float = 0.03
    w_mean: float = 0.5
    w_var: float = 0.5
    w_pair: float = 1.0  # Weight on pair_split_penalty (align with NSGA)
    baseline_momentum: float = 0.9
    eps_explore: float = 0.0  # IMPORTANT: keep 0 by default to avoid mismatch between sampling and gradient

    # Problem size
    group_size: int = 4
    n_groups: Optional[int] = None

    # Hydraulic evaluation
    q_lateral: float = 0.012

    # Seeds
    seed: int = 0


@dataclass(frozen=True)
class GroupStats:
    """Cached hydraulic summary for a group of laterals.

    - sum_surplus / sumsq_surplus are over all laterals in the group (typically 4).
    - violation is the sum of deficits below Hmin (0 means feasible under NSGA's definition).
    - feasible indicates (evaluate_group ok) AND (violation == 0).
    """

    sum_surplus: float
    sumsq_surplus: float
    violation: float
    feasible: bool


def _base_id(lateral_id: str) -> str:
    """Convert 'J11_L' or 'J11_R' -> 'J11'."""
    return lateral_id.rsplit("_", 1)[0] if "_" in lateral_id else lateral_id


class MORLHydraulicSolver:
    def __init__(
            self,
            evaluator: tree_evaluator.TreeHydraulicEvaluator,
            lateral_ids: List[str],
            lateral_to_node: Dict[str, str],
            cfg: TrainingConfig,
    ) -> None:
        self.evaluator = evaluator
        self.lateral_ids = lateral_ids
        self.lateral_to_node = lateral_to_node
        self.cfg = cfg

        n_lats = len(lateral_ids)
        if cfg.n_groups is None:
            if n_lats % cfg.group_size != 0:
                raise ValueError(
                    f"Number of laterals ({n_lats}) is not divisible by group_size ({cfg.group_size})."
                )
            self.cfg.n_groups = n_lats // cfg.group_size

        if self.cfg.n_groups * self.cfg.group_size != n_lats:
            raise ValueError(
                f"Expected {self.cfg.n_groups * self.cfg.group_size} laterals, got {n_lats}."
            )

        self.theta = np.zeros((n_lats, self.cfg.n_groups), dtype=np.float64)
        self.rng = random.Random(cfg.seed)
        self.baseline = 0.0

        # Cache: key = tuple(sorted(lateral_ids_in_group)) => GroupStats (sum/sumsq/violation)
        self._group_cache: Dict[Tuple[str, ...], GroupStats] = {}

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def _sample_schedule(self) -> Tuple[List[int], List[List[int]], List[int]]:
        """Sample one schedule under capacity constraints.

        Returns
        - assignment: list[int], length n_lats, assignment[i] = group index
        - schedule_nested: list[list[int]], group -> lateral indices
        - order: list[int], the shuffled decision order used (MUST be replayed in updates)
        """
        n_lats = len(self.lateral_ids)
        n_groups = self.cfg.n_groups
        group_size = self.cfg.group_size

        order = list(range(n_lats))
        self.rng.shuffle(order)

        counts = [0] * n_groups
        assignment: List[int] = [-1] * n_lats

        for i in order:
            admissible = [g for g in range(n_groups) if counts[g] < group_size]
            logits = self.theta[i, admissible]
            probs = self._softmax(logits)

            eps = float(self.cfg.eps_explore)
            if eps > 0:
                probs = (1.0 - eps) * probs + eps / len(probs)

            # categorical sample
            r = self.rng.random()
            cum = 0.0
            pick_idx = 0
            for idx, p in enumerate(probs):
                cum += float(p)
                if r <= cum:
                    pick_idx = idx
                    break

            g = admissible[pick_idx]
            assignment[i] = g
            counts[g] += 1

        schedule_nested: List[List[int]] = [[] for _ in range(n_groups)]
        for lat_idx, g in enumerate(assignment):
            schedule_nested[g].append(lat_idx)

        return assignment, schedule_nested, order

    def _eval_group_stats(self, group: List[int]) -> GroupStats:
        """Return cached hydraulic summary for one group.

        Aligns with NSGA evaluation:
        - surplus = pressure_head(node) - Hmin
        - violation = sum of deficits below Hmin (0 means feasible)
        """
        if not group:
            return GroupStats(0.0, 0.0, 0.0, True)

        lat_ids = [self.lateral_ids[i] for i in group]
        key = tuple(sorted(lat_ids))
        if key in self._group_cache:
            return self._group_cache[key]

        try:
            res = self.evaluator.evaluate_group(
                lat_ids,
                self.lateral_to_node,
                q_lateral=float(self.cfg.q_lateral),
            )
        except Exception:
            stats = GroupStats(0.0, 0.0, 1e6, False)
            self._group_cache[key] = stats
            return stats

        pressures = getattr(res, "pressures", None)
        ok_flag = bool(getattr(res, "ok", False))

        if not isinstance(pressures, dict):
            stats = GroupStats(0.0, 0.0, 1e6, False)
            self._group_cache[key] = stats
            return stats

        sum_sur = 0.0
        sumsq_sur = 0.0
        violation = 0.0

        for lid in lat_ids:
            node_id = self.lateral_to_node[lid]
            p = float(pressures[node_id])
            s = p - float(self.evaluator.Hmin)
            if s < 0.0:
                violation += -s
            sum_sur += s
            sumsq_sur += s * s

        # If the evaluator flags failure, force infeasible even if violation is numerically 0.
        if not ok_flag:
            violation = max(violation, 1e6)

        stats = GroupStats(float(sum_sur), float(sumsq_sur), float(violation), bool(violation <= 0.0 and ok_flag))
        self._group_cache[key] = stats
        return stats

    def _pair_split_penalty(self, schedule_nested: List[List[int]]) -> int:
        """NSGA-style pair split penalty: count node-pairs whose L/R fall into different groups."""
        n_lats = len(self.lateral_ids)
        assignment = [-1] * n_lats
        for g, group in enumerate(schedule_nested):
            for idx in group:
                assignment[idx] = g

        base_to_groups: Dict[str, set] = {}
        for idx, lid in enumerate(self.lateral_ids):
            base = _base_id(lid)
            base_to_groups.setdefault(base, set()).add(assignment[idx])

        return sum(1 for gs in base_to_groups.values() if len(gs) > 1)

    def schedule_stats(
            self, schedule_nested: List[List[int]]
    ) -> Tuple[float, float, float, int, float, bool, float]:
        """Return NSGA-aligned metrics for a schedule.

        Returns:
            (mean_surplus, var_surplus, std_surplus, pair_penalty, violation, feasible, cost)

        Notes:
        - mean/var are computed across ALL laterals (e.g., 120), not across group-means.
        - cost uses std (sqrt(var)) for better unit consistency with mean.
        - Infeasible schedules are heavily penalized.
        """
        total_sum = 0.0
        total_sumsq = 0.0
        total_violation = 0.0
        total_n = 0

        for g in schedule_nested:
            st = self._eval_group_stats(g)
            total_sum += st.sum_surplus
            total_sumsq += st.sumsq_surplus
            total_violation += st.violation
            total_n += len(g)

        if total_n <= 0:
            mean = 0.0
            var = 0.0
        else:
            mean = float(total_sum / total_n)
            var = float(total_sumsq / total_n - mean * mean)
            if var < 0.0 and var > -1e-12:
                var = 0.0

        std = float(math.sqrt(max(0.0, var)))
        pair_pen = int(self._pair_split_penalty(schedule_nested))
        feasible = bool(total_violation <= 0.0)

        cost = float(self.cfg.w_var * std + self.cfg.w_mean * mean + self.cfg.w_pair * float(pair_pen))

        # Strong constraint handling: infeasible schedules are never competitive.
        if not feasible:
            cost = float(cost + 1e9 + 1e6 * float(total_violation))

        return mean, var, std, pair_pen, float(total_violation), feasible, cost

    def _update_parameters(self, assignments: List[int], advantage: float, order: List[int]) -> None:
        """Policy gradient update; MUST replay the same `order` used in sampling."""
        n_groups = self.cfg.n_groups
        group_size = self.cfg.group_size
        counts = [0] * n_groups

        for i in order:
            g_selected = assignments[i]
            admissible = [g for g in range(n_groups) if counts[g] < group_size]
            if g_selected not in admissible:
                admissible = admissible + [g_selected]

            logits = self.theta[i, admissible]
            probs = self._softmax(logits)

            for j, g in enumerate(admissible):
                ind = 1.0 if g == g_selected else 0.0
                grad = advantage * (ind - float(probs[j]))
                self.theta[i, g] += float(self.cfg.lr) * grad

            counts[g_selected] += 1

    def train(self) -> None:
        """Mini-batch REINFORCE with standardized advantages."""
        bs = int(self.cfg.batch_size)
        if bs <= 0:
            raise ValueError("batch_size must be positive")

        m = float(self.cfg.baseline_momentum)
        if not (0.0 <= m < 1.0):
            raise ValueError("baseline_momentum must be in [0,1)")

        for ep in range(1, int(self.cfg.episodes) + 1):
            batch_samples = [self._sample_schedule() for _ in range(bs)]

            costs = np.empty((bs,), dtype=float)
            means = np.empty((bs,), dtype=float)
            vars_ = np.empty((bs,), dtype=float)

            for k, (_, schedule_nested, _) in enumerate(batch_samples):
                gm, gv, gs, gp, vio, feas, cost = self.schedule_stats(schedule_nested)
                means[k] = gm
                vars_[k] = gv
                costs[k] = cost

            rewards = -costs

            # advantage BEFORE baseline update
            advantages = rewards - float(self.baseline)

            # baseline update using batch mean reward
            batch_mean_reward = float(np.mean(rewards))
            self.baseline = m * float(self.baseline) + (1.0 - m) * batch_mean_reward

            # standardize advantages
            adv_mean = float(np.mean(advantages))
            adv_std = float(np.std(advantages))
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

            # apply updates
            for k, (assignments, _, order) in enumerate(batch_samples):
                self._update_parameters(assignments, float(advantages[k]), order)

            if ep == 1 or ep % max(1, int(self.cfg.episodes) // 10) == 0:
                k_best = int(np.argmin(costs))
                print(
                    f"train_ep={ep:6d} | cost_mean={float(np.mean(costs)):.4f} cost_min={float(np.min(costs)):.4f} "
                    f"| best_mean={float(means[k_best]):.4f} best_std={math.sqrt(max(0.0, float(vars_[k_best]))):.4f} best_var={float(vars_[k_best]):.4f} "
                    f"| baseline={float(self.baseline):.4f}"
                )

    def best_of_sampling(self, n_samples: int, *, label: str = "sample") -> Tuple[
        List[List[int]], float, float, float, int, float]:
        """Sample n schedules and return the best (lowest cost)."""
        best_schedule: Optional[List[List[int]]] = None
        best_cost = float("inf")
        best_mean = float("inf")
        best_var = float("inf")
        best_std = float("inf")
        best_pair = int(1e9)

        for k in range(1, int(n_samples) + 1):
            _, sched, _ = self._sample_schedule()
            gm, gv, gs, gp, vio, feas, cost = self.schedule_stats(sched)
            if cost < best_cost:
                best_cost = cost
                best_mean = gm
                best_var = gv
                best_std = gs
                best_pair = int(gp)
                best_schedule = [list(g) for g in sched]

            if k % max(1, int(n_samples) // 5) == 0:
                print(f"{label}: {k}/{n_samples} best_cost={best_cost:.4f}")

        assert best_schedule is not None
        return best_schedule, best_mean, best_var, best_std, best_pair, best_cost

    def local_search_swap(
            self,
            start_schedule: List[List[int]],
            *,
            steps: int = 50000,
            method: str = "sa",  # 'hc' or 'sa'
            T0: Optional[float] = None,
            cool_alpha: float = 0.85,
            cool_interval: int = 2000,
            patience: int = 8000,
            log_every: int = 1000,
            seed: Optional[int] = None,
    ) -> Tuple[List[List[int]], float, float, float, int, float]:
        """Swap-based local search (hill-climb or simulated annealing), NSGA-aligned.

        Objectives aligned to NSGA (but scalarized for local search):
        - mean surplus across ALL laterals
        - std surplus across ALL laterals (sqrt(var))
        - pair_split_penalty (L/R split across different groups)

        Move:
        - pick two groups, swap one lateral index from each group (capacity preserved).

        Performance notes:
        - Only two groups are re-evaluated hydraulically per proposal.
        - Global mean/std are updated incrementally from cached (sum, sumsq, violation).
        - Pair penalty is updated incrementally from the two swapped laterals.
        """
        n_groups = self.cfg.n_groups
        group_size = self.cfg.group_size
        n_lats = len(self.lateral_ids)

        rng = random.Random(self.cfg.seed + 1337 if seed is None else seed)

        # Deep copy schedule
        sched = [list(g) for g in start_schedule]

        # Group hydraulic stats
        group_stats = [self._eval_group_stats(g) for g in sched]
        group_sum = [st.sum_surplus for st in group_stats]
        group_sumsq = [st.sumsq_surplus for st in group_stats]
        group_vio = [st.violation for st in group_stats]

        total_sum = float(sum(group_sum))
        total_sumsq = float(sum(group_sumsq))
        total_vio = float(sum(group_vio))

        def mean_var_std(tsum: float, tsumsq: float) -> Tuple[float, float, float]:
            mean = float(tsum / n_lats)
            var = float(tsumsq / n_lats - mean * mean)
            if var < 0.0 and var > -1e-12:
                var = 0.0
            std = float(math.sqrt(max(0.0, var)))
            return mean, var, std

        # Pair penalty bookkeeping
        assignment = [-1] * n_lats
        for g, grp in enumerate(sched):
            for idx in grp:
                assignment[idx] = g

        base_of = [_base_id(lid) for lid in self.lateral_ids]
        base_to_indices: Dict[str, List[int]] = {}
        for idx, b in enumerate(base_of):
            base_to_indices.setdefault(b, []).append(idx)

        def pair_contrib(base: str) -> int:
            idxs = base_to_indices.get(base, [])
            if len(idxs) < 2:
                return 0
            return 1 if assignment[idxs[0]] != assignment[idxs[1]] else 0

        current_pair = sum(pair_contrib(b) for b in base_to_indices.keys())

        current_mean, current_var, current_std = mean_var_std(total_sum, total_sumsq)
        current_cost = float(
            self.cfg.w_var * current_std + self.cfg.w_mean * current_mean + self.cfg.w_pair * float(current_pair))
        if total_vio > 0.0:
            current_cost = float(current_cost + 1e9 + 1e6 * total_vio)

        best_sched = [list(g) for g in sched]
        best_cost = float(current_cost)
        best_mean = float(current_mean)
        best_var = float(current_var)
        best_std = float(current_std)
        best_pair = int(current_pair)

        # SA temperature
        if method not in {"hc", "sa"}:
            raise ValueError("method must be 'hc' or 'sa'")
        T = 0.0
        if method == "sa":
            if T0 is None:
                T = 0.05 * max(1.0, float(current_cost))
            else:
                T = float(T0)
            T = max(T, 1e-8)

        accept = 0
        propose = 0
        no_improve = 0

        for step in range(1, int(steps) + 1):
            propose += 1
            g1, g2 = rng.sample(range(n_groups), 2)
            p1 = rng.randrange(group_size)
            p2 = rng.randrange(group_size)

            a = sched[g1][p1]
            b = sched[g2][p2]

            # Swap in-place
            sched[g1][p1], sched[g2][p2] = b, a

            # Incremental pair penalty update (affected bases only)
            affected = {base_of[a], base_of[b]}
            old_pair_aff = sum(pair_contrib(x) for x in affected)

            # Update assignments consistent with the swap
            assignment[a], assignment[b] = g2, g1
            new_pair_aff = sum(pair_contrib(x) for x in affected)
            new_pair = int(current_pair - old_pair_aff + new_pair_aff)

            # Incremental hydraulic update (only two groups changed)
            old_s1, old_ss1, old_v1 = group_sum[g1], group_sumsq[g1], group_vio[g1]
            old_s2, old_ss2, old_v2 = group_sum[g2], group_sumsq[g2], group_vio[g2]

            st1 = self._eval_group_stats(sched[g1])
            st2 = self._eval_group_stats(sched[g2])

            group_sum[g1], group_sumsq[g1], group_vio[g1] = st1.sum_surplus, st1.sumsq_surplus, st1.violation
            group_sum[g2], group_sumsq[g2], group_vio[g2] = st2.sum_surplus, st2.sumsq_surplus, st2.violation

            new_total_sum = total_sum - old_s1 - old_s2 + st1.sum_surplus + st2.sum_surplus
            new_total_sumsq = total_sumsq - old_ss1 - old_ss2 + st1.sumsq_surplus + st2.sumsq_surplus
            new_total_vio = total_vio - old_v1 - old_v2 + st1.violation + st2.violation

            new_mean, new_var, new_std = mean_var_std(new_total_sum, new_total_sumsq)

            new_cost = float(self.cfg.w_var * new_std + self.cfg.w_mean * new_mean + self.cfg.w_pair * float(new_pair))
            if new_total_vio > 0.0:
                new_cost = float(new_cost + 1e9 + 1e6 * float(new_total_vio))

            delta = float(new_cost - current_cost)

            do_accept = False
            if delta <= 0.0:
                do_accept = True
            elif method == "sa":
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    do_accept = True

            if do_accept:
                accept += 1
                total_sum = float(new_total_sum)
                total_sumsq = float(new_total_sumsq)
                total_vio = float(new_total_vio)
                current_pair = int(new_pair)

                current_mean = float(new_mean)
                current_var = float(new_var)
                current_std = float(new_std)
                current_cost = float(new_cost)

                if new_cost < best_cost:
                    best_cost = float(new_cost)
                    best_mean = float(new_mean)
                    best_var = float(new_var)
                    best_std = float(new_std)
                    best_pair = int(new_pair)
                    best_sched = [list(g) for g in sched]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert swap
                sched[g1][p1], sched[g2][p2] = a, b

                # Revert assignments and pair penalty
                assignment[a], assignment[b] = g1, g2

                # Revert group stats
                group_sum[g1], group_sumsq[g1], group_vio[g1] = old_s1, old_ss1, old_v1
                group_sum[g2], group_sumsq[g2], group_vio[g2] = old_s2, old_ss2, old_v2

                no_improve += 1

            # cooling
            if method == "sa" and (step % max(1, int(cool_interval)) == 0):
                T = max(1e-8, T * float(cool_alpha))

            if step == 1 or step % max(1, int(log_every)) == 0:
                acc_rate = accept / max(1, propose)
                if method == "sa":
                    print(
                        f"ls_step={step:6d} | current_cost={current_cost:.4f} | best_cost={best_cost:.4f} | "
                        f"best_mean={best_mean:.4f} best_std={best_std:.4f} best_var={best_var:.4f} best_pair={best_pair:d} | "
                        f"T={T:.6f} | acc={acc_rate:.3f}"
                    )
                else:
                    print(
                        f"ls_step={step:6d} | current_cost={current_cost:.4f} | best_cost={best_cost:.4f} | "
                        f"best_mean={best_mean:.4f} best_std={best_std:.4f} best_var={best_var:.4f} best_pair={best_pair:d} | "
                        f"acc={acc_rate:.3f}"
                    )

            if no_improve >= int(patience):
                print(f"Early stop: no best improvement for {patience} proposals. best_cost={best_cost:.4f}")
                break

        return best_sched, best_mean, best_var, best_std, best_pair, best_cost


def _print_schedule(lateral_ids: List[str], schedule_nested: List[List[int]]) -> None:
    print("\nSchedule (group -> lateral IDs)")
    for g, group in enumerate(schedule_nested, start=1):
        lids = [lateral_ids[i] for i in group]
        print(f"Group {g:02d}: {lids}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch PG + Swap/SA local search for irrigation grouping")

    ap.add_argument("--nodes_xlsx", type=str, required=True, help="Path to Nodes.xlsx")
    ap.add_argument("--pipes_xlsx", type=str, required=True, help="Path to Pipes.xlsx")

    ap.add_argument("--root", type=str, default="J0", help="Root/source node id")
    ap.add_argument("--H0", type=float, default=25.0, help="Source head (m)")
    ap.add_argument("--Hmin", type=float, default=11.59, help="Minimum required pressure head (m)")

    ap.add_argument(
        "--mode",
        type=str,
        default="local",
        choices=["train", "local", "train+local"],
        help="Run mode: train only, local search only, or train then local search",
    )

    # Training args
    ap.add_argument("--episodes", type=int, default=500, help="Training episodes (outer iterations)")
    ap.add_argument("--batch_size", type=int, default=32, help="Schedules sampled per training episode")
    ap.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    ap.add_argument("--w_var", type=float, default=0.5, help="Weight on variance term")
    ap.add_argument("--w_mean", type=float, default=0.5, help="Weight on mean term")
    ap.add_argument("--w_pair", type=float, default=1.0, help="Weight on pair-split penalty term")
    ap.add_argument("--baseline_momentum", type=float, default=0.9, help="Baseline EMA momentum")
    ap.add_argument(
        "--eps_explore",
        type=float,
        default=0.0,
        help="Exploration smoothing epsilon (keep 0 for consistent gradients)",
    )

    # Problem / hydraulic
    ap.add_argument("--group_size", type=int, default=4, help="Laterals per group")
    ap.add_argument("--q_lateral", type=float, default=0.012, help="Flow per lateral")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")

    # Initialization for local search
    ap.add_argument(
        "--init_random",
        type=int,
        default=3000,
        help="How many random samples to find a good starting schedule for local search",
    )

    # Local search args
    ap.add_argument("--local_steps", type=int, default=50000, help="Swap proposals")
    ap.add_argument(
        "--local_method",
        type=str,
        default="sa",
        choices=["hc", "sa"],
        help="Local search method: hill-climb (hc) or simulated annealing (sa)",
    )
    ap.add_argument(
        "--sa_T0",
        type=float,
        default=-1.0,
        help="Initial SA temperature (<=0 means auto = 0.05*initial_cost)",
    )
    ap.add_argument("--sa_alpha", type=float, default=0.85, help="Cooling multiplier")
    ap.add_argument("--sa_interval", type=int, default=2000, help="Cooling interval (steps)")

    ap.add_argument(
        "--patience",
        type=int,
        default=8000,
        help="Early stop if no best improvement for this many proposals",
    )
    ap.add_argument("--log_every", type=int, default=1000, help="Local-search logging interval")

    args = ap.parse_args()

    # Load network data
    nodes = tree_evaluator.load_nodes_xlsx(args.nodes_xlsx)
    edges = tree_evaluator.load_pipes_xlsx(args.pipes_xlsx)

    field_nodes = [nid for nid in nodes.keys() if tree_evaluator.is_field_node_id(nid)]
    lateral_ids, lateral_to_node = tree_evaluator.build_lateral_ids_for_field_nodes(field_nodes)

    cfg = TrainingConfig(
        episodes=int(args.episodes),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        w_mean=float(args.w_mean),
        w_var=float(args.w_var),
        w_pair=float(args.w_pair),
        baseline_momentum=float(args.baseline_momentum),
        eps_explore=float(args.eps_explore),
        group_size=int(args.group_size),
        n_groups=None,
        q_lateral=float(args.q_lateral),
        seed=int(args.seed),
    )

    evaluator = tree_evaluator.TreeHydraulicEvaluator(
        nodes=nodes,
        edges=edges,
        root=str(args.root),
        H0=float(args.H0),
        Hmin=float(args.Hmin),
    )

    solver = MORLHydraulicSolver(evaluator, lateral_ids, lateral_to_node, cfg)

    print(
        f"Config: mode={args.mode}, w_var={cfg.w_var}, w_mean={cfg.w_mean}, w_pair={cfg.w_pair}, group_size={cfg.group_size}, "
        f"q_lateral={cfg.q_lateral}, seed={cfg.seed}"
    )

    # Optional training
    if args.mode in {"train", "train+local"}:
        print(
            "Training with "
            f"episodes={cfg.episodes}, batch_size={cfg.batch_size}, lr={cfg.lr}, "
            f"w_var={cfg.w_var}, w_mean={cfg.w_mean}, w_pair={cfg.w_pair}, eps_explore={cfg.eps_explore}, q_lateral={cfg.q_lateral}"
        )
        solver.train()

    # Local search
    if args.mode in {"local", "train+local"}:
        print(f"\nFinding start schedule via random sampling: init_random={args.init_random}")

        # Candidate A: best among random samples
        start_sched, start_mean, start_var, start_std, start_pair, start_cost = solver.best_of_sampling(
            int(args.init_random), label="init_random"
        )
        print(
            f"Start schedule: mean={start_mean:.6f} std={start_std:.6f} var={start_var:.6f} pair={start_pair:d} cost={start_cost:.6f}"
        )

        # Candidate B (if trained): take best-of sampling from current policy distribution (extra exploitation)
        if args.mode == "train+local":
            pol_sched, pol_mean, pol_var, pol_std, pol_pair, pol_cost = solver.best_of_sampling(
                max(200, int(args.init_random) // 5), label="post_train_sample"
            )
            if pol_cost < start_cost:
                start_sched, start_mean, start_var, start_std, start_pair, start_cost = pol_sched, pol_mean, pol_var, pol_std, pol_pair, pol_cost
                print(
                    f"Using post-train start schedule: mean={start_mean:.6f} std={start_std:.6f} var={start_var:.6f} pair={start_pair:d} cost={start_cost:.6f}"
                )

        # Run local search
        T0 = None if float(args.sa_T0) <= 0 else float(args.sa_T0)
        best_sched, best_mean, best_var, best_std, best_pair, best_cost = solver.local_search_swap(
            start_sched,
            steps=int(args.local_steps),
            method=str(args.local_method),
            T0=T0,
            cool_alpha=float(args.sa_alpha),
            cool_interval=int(args.sa_interval),
            patience=int(args.patience),
            log_every=int(args.log_every),
        )

        print("\n--- Local-search best summary ---")
        print(f"best_mean_surplus = {best_mean:.6f}")
        print(f"best_std_surplus  = {best_std:.6f}")
        print(f"best_var_surplus  = {best_var:.6f}")
        print(f"best_pair_penalty = {best_pair:d}")
        print(f"best_cost         = {best_cost:.6f}")
        _print_schedule(lateral_ids, best_sched)


if __name__ == "__main__":
    main()
