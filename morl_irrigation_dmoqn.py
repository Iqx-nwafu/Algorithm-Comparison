"""morl_irrigation_dmoqn.py

Dynamic wheel-irrigation scheduling via Multi-Objective Reinforcement Learning (MORL).

This script is designed to be a drop-in companion to your existing NSGA-II workflow.
It reuses the same hydraulic evaluator interface and the same hard constraint:
    pressure_head(node) >= Hmin   for every opened lateral

Key design choices aligned to your NSGA objectives/rules:
- Decision horizon: 30 steps (30 irrigation groups).
- Each step opens a group of 4 laterals = 2 field nodes × (L,R) laterals.
- No repetition: every field node is used exactly once (therefore every lateral is used exactly once).
- Pair completion (NSGA pair_split_penalty) is satisfied by construction (always open L/R together).

MORL objective / reward:
- Two objectives: minimize surplus-head variance and surplus-head mean over ALL 120 laterals.
- We define a *vector reward* r_t in R^2 as the negative incremental change of
  (Var(surplus_all_so_far), Mean(surplus_all_so_far)).
  Summing rewards across the episode telescopes to (-Var_final, -Mean_final).

Algorithms implemented:
1) DMOQN (Deep Multi-Objective Q-Network):
   - Q(s,a,w) outputs a 2D Q-vector; action selection uses preference weights w (scalarization).
   - Universal MORL: w is part of the network input; during training, w is sampled.

2) Multi-objective policy gradient (A2C-style, preference-conditioned):
   - A policy scores feasible actions; learning maximizes scalarized return w·G.

Dynamic scenario (optional):
- To model time-varying operating conditions, each step draws a scenario index that changes source head H0.
  The scenario index is part of the observation, so the policy can adapt online.

Dependencies:
- numpy
- torch
- Your existing tree_evaluator.py (same interface as used in NSGA.py)
- Nodes.xlsx, Pipes.xlsx

Quick start (example):
  python morl_irrigation_dmoqn.py train_dmoqn \
    --nodes Nodes.xlsx --pipes Pipes.xlsx --root J0 --H0 25.0 --Hmin 11.59 \
    --q_lateral 0.012 --out runs/dmoqn

  python morl_irrigation_dmoqn.py evaluate \
    --nodes Nodes.xlsx --pipes Pipes.xlsx --root J0 --H0 25.0 --Hmin 11.59 \
    --q_lateral 0.012 --model runs/dmoqn/best_model.pt --out runs/dmoqn_eval

Notes
- If your evaluator is expensive, you may prefer --precompute 1 to precompute all (node-pair) groups
  per scenario, because the action space is limited to C(60,2)=1770 candidate groups.

Authoring note:
- This file is self-contained except for tree_evaluator.py and the input spreadsheets.
"""

from __future__ import annotations
import json
import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    raise ImportError(
        "PyTorch is required. Install with: pip install torch\n"
        f"Original error: {e}"
    )

# === Reuse your existing fast hydraulic evaluator interface (same as NSGA.py) ===
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
        "tree_evaluator.py with the expected interface is required (same as your NSGA.py).\n"
        "Make sure tree_evaluator.py is in the same folder or on PYTHONPATH.\n"
        f"Original error: {e}"
    )


# =========================
# Utilities
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_dominated(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if b dominates a for minimization objectives."""
    return np.all(b <= a) and np.any(b < a)




def pareto_front_indices_2d(values: np.ndarray) -> np.ndarray:
    """Return indices of Pareto-nondominated points (minimization) for 2D objectives.

    values: shape (N,2) where columns are [var, mean] and both are minimized.
    """
    if values.size == 0:
        return np.asarray([], dtype=np.int64)
    n = values.shape[0]
    keep: list[int] = []
    for i in range(n):
        a = values[i]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            b = values[j]
            if np.all(b <= a) and np.any(b < a):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)


def hypervolume_2d_min(values: np.ndarray, ref: tuple[float, float]) -> float:
    """2D hypervolume for minimization objectives.

    Computes dominated area between the Pareto front points and the reference point (ref),
    assuming ref is *worse* (larger) than all points in both coordinates.
    """
    if values.size == 0:
        return 0.0

    # Keep only finite points
    m = np.isfinite(values).all(axis=1)
    values = values[m]
    if values.size == 0:
        return 0.0

    idx = pareto_front_indices_2d(values)
    front = values[idx]
    if front.size == 0:
        return 0.0

    # Sort by x (var) ascending; for a proper Pareto front in minimization, y should be non-increasing.
    order = np.argsort(front[:, 0], kind="mergesort")
    front = front[order]

    ref_x, ref_y = float(ref[0]), float(ref[1])
    hv = 0.0
    cur_y = ref_y

    for x, y in front:
        x = float(x)
        y = float(y)
        if y < cur_y:
            width = max(0.0, ref_x - x)
            height = cur_y - y
            hv += width * height
            cur_y = y

    return float(hv)


def make_weight_grid_line(n: int) -> np.ndarray:
    """Line grid on simplex for 2 objectives: w_var in [0,1], w_mean=1-w_var."""
    n = int(n)
    if n < 2:
        raise ValueError("eval_grid must be >= 2")
    w_var = np.linspace(0.0, 1.0, n, dtype=np.float32)
    w_mean = 1.0 - w_var
    ws = np.stack([w_var, w_mean], axis=1)
    # Normalize defensively
    ws = ws / np.maximum(1e-12, ws.sum(axis=1, keepdims=True))
    return ws
# =========================
# Configs
# =========================

@dataclass(frozen=True)
class EnvConfig:
    group_size: int = 4
    n_groups: int = 30

    q_lateral: float = 0.012

    # Dynamic scenario: source head levels (relative multipliers)
    # If scenario_k=1, only the base H0 is used.
    scenario_k: int = 3
    scenario_rel: Tuple[float, ...] = (0.95, 1.00, 1.05)

    # Constraint handling
    infeasible_penalty: float = 50.0  # applied to both objectives (negative reward)


@dataclass(frozen=True)
class DMOQNConfig:
    # RL core
    gamma: float = 0.99
    buffer_size: int = 200_000
    batch_size: int = 128
    learning_rate: float = 2e-4
    target_update_interval: int = 1000
    train_after_steps: int = 5_000
    train_every: int = 4
    grad_clip: float = 5.0

    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 150_000

    # Preference weights w sampling: Dirichlet(alpha_var, alpha_mean)
    # To reflect NSGA priority (variance > mean), default alpha=(3,2).
    pref_alpha_var: float = 3.0
    pref_alpha_mean: float = 2.0

    # Network
    node_emb_dim: int = 32
    hidden_dim: int = 256

    # Reward normalization (stabilizes multi-objective scales)
    reward_norm: bool = True
    reward_norm_eps: float = 1e-6

    # Training length
    total_env_steps: int = 400_000
    eval_every_steps: int = 20_000
    eval_episodes: int = 200


@dataclass(frozen=True)
class PGConfig:
    gamma: float = 0.99
    learning_rate: float = 2e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 5.0

    pref_alpha_var: float = 3.0
    pref_alpha_mean: float = 2.0

    node_emb_dim: int = 32
    hidden_dim: int = 256

    episodes: int = 50_000
    eval_every: int = 2000
    eval_episodes: int = 200


# =========================
# Action space construction
# =========================


def build_base_nodes_and_pairs(nodes: Dict[str, object]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Return base field nodes (length=60) and all unordered pairs arrays (i_idx, j_idx) length=1770."""
    field_nodes = [nid for nid in nodes.keys() if is_field_node_id(nid)]
    field_nodes = sorted(field_nodes)
    if len(field_nodes) * 2 != 120:
        # Your data convention is 60 field nodes x 2 laterals = 120.
        # If your dataset differs, adjust the action construction accordingly.
        raise ValueError(
            f"Expect 60 field nodes (120 laterals) but got {len(field_nodes)} field nodes. "
            "If your dataset is different, update build_base_nodes_and_pairs()."
        )

    pairs_i: List[int] = []
    pairs_j: List[int] = []
    n = len(field_nodes)
    for i in range(n):
        for j in range(i + 1, n):
            pairs_i.append(i)
            pairs_j.append(j)

    return field_nodes, np.asarray(pairs_i, dtype=np.int64), np.asarray(pairs_j, dtype=np.int64)


def pair_to_group_laterals(base_nodes: List[str], i: int, j: int) -> List[str]:
    """Action (i,j) -> 4 laterals (i_L,i_R,j_L,j_R)."""
    b1 = base_nodes[i]
    b2 = base_nodes[j]
    return [f"{b1}_L", f"{b1}_R", f"{b2}_L", f"{b2}_R"]


# =========================
# Precompute / cache hydraulic outcomes for all actions
# =========================


class GroupTable:
    """Caches group outcomes per scenario and pair action.

    Stores:
      - surpluses: shape (K, n_pairs, 4)
      - feasible:  shape (K, n_pairs) (True if all surpluses >= 0)

    This makes action masking and reward computation fast.
    """

    def __init__(
        self,
        evaluator: TreeHydraulicEvaluator,
        base_nodes: List[str],
        lateral_to_node: Dict[str, str],
        pairs_i: np.ndarray,
        pairs_j: np.ndarray,
        H0_base: float,
        Hmin: float,
        q_lateral: float,
        scenario_rel: List[float],
    ) -> None:
        self.evaluator = evaluator
        self.base_nodes = base_nodes
        self.lateral_to_node = lateral_to_node
        self.pairs_i = pairs_i
        self.pairs_j = pairs_j
        self.H0_base = H0_base
        self.Hmin = Hmin
        self.q_lateral = q_lateral
        self.scenario_rel = scenario_rel

        self.K = len(scenario_rel)
        self.n_pairs = len(pairs_i)

        self.surpluses = np.zeros((self.K, self.n_pairs, 4), dtype=np.float32)
        self.feasible = np.zeros((self.K, self.n_pairs), dtype=np.bool_)

    def _set_evaluator_H0(self, H0: float) -> None:
        # Best-effort: try to set attribute.
        if hasattr(self.evaluator, "H0"):
            setattr(self.evaluator, "H0", float(H0))
        # else: evaluator might not expose H0 as attribute; in that case,
        # you should adapt tree_evaluator.py or create separate evaluator per scenario.

    def precompute(self, verbose: bool = True) -> None:
        t0 = time.perf_counter()
        old_H0 = getattr(self.evaluator, "H0", None)

        for k, rel in enumerate(self.scenario_rel):
            H0k = self.H0_base * rel
            self._set_evaluator_H0(H0k)

            if verbose:
                print(f"[Precompute] scenario {k}/{self.K-1}: H0={H0k:.6g} (rel={rel:.3f})")

            for a in range(self.n_pairs):
                i = int(self.pairs_i[a])
                j = int(self.pairs_j[a])
                group = pair_to_group_laterals(self.base_nodes, i, j)

                res = self.evaluator.evaluate_group(
                    group, lateral_to_node=self.lateral_to_node, q_lateral=self.q_lateral
                )

                sur: List[float] = []
                ok = True
                for lat in group:
                    nid = self.lateral_to_node[lat]
                    p = res.pressures[nid]
                    s = float(p - self.Hmin)
                    sur.append(s)
                    if s < 0.0:
                        ok = False

                self.surpluses[k, a, :] = np.asarray(sur, dtype=np.float32)
                self.feasible[k, a] = ok

        # restore
        if old_H0 is not None:
            self._set_evaluator_H0(float(old_H0))

        if verbose:
            print(f"[Precompute] done. elapsed={time.perf_counter()-t0:.2f}s")


# =========================
# Environment
# =========================


class IrrigationSchedulingEnv:
    """Dynamic scheduling: at each step choose 2 remaining field nodes (action pair) to irrigate."""

    def __init__(
        self,
        base_nodes: List[str],
        pairs_i: np.ndarray,
        pairs_j: np.ndarray,
        group_table: GroupTable,
        cfg: EnvConfig,
        rng: np.random.Generator,
    ) -> None:
        self.base_nodes = base_nodes
        self.pairs_i = pairs_i
        self.pairs_j = pairs_j
        self.group_table = group_table
        self.cfg = cfg
        self.rng = rng

        self.n_nodes = len(base_nodes)
        self.n_pairs = len(pairs_i)
        self.K = group_table.K

        # episode state
        self.t = 0
        self.remaining = np.ones(self.n_nodes, dtype=np.bool_)
        self.surpluses_all: List[float] = []
        self.prev_mean = 0.0
        self.prev_var = 0.0
        self.scenario = 0

    def reset(self) -> Dict[str, np.ndarray | int | float]:
        self.t = 0
        self.remaining[:] = True
        self.surpluses_all = []
        self.prev_mean = 0.0
        self.prev_var = 0.0
        self.scenario = int(self.rng.integers(0, self.K))
        return self._obs()

    def _obs(self) -> Dict[str, np.ndarray | int | float]:
        return {
            "mask": self.remaining.astype(np.float32),
            "t": int(self.t),
            "scenario": int(self.scenario),
        }

    def _objective_stats(self) -> Tuple[float, float]:
        if not self.surpluses_all:
            return 0.0, 0.0
        x = np.asarray(self.surpluses_all, dtype=np.float64)
        mean = float(np.mean(x))
        var = float(np.mean((x - mean) ** 2))
        return var, mean

    def feasible_actions(self, scenario: Optional[int] = None) -> np.ndarray:
        """Return indices of feasible actions under current remaining set and a given scenario."""
        k = self.scenario if scenario is None else int(scenario)
        rem = self.remaining
        ii = self.pairs_i
        jj = self.pairs_j

        # both nodes remaining AND hydraulically feasible
        m = self.group_table.feasible[k] & rem[ii] & rem[jj]
        return np.nonzero(m)[0]

    def step(self, action_idx: int) -> Tuple[Dict[str, np.ndarray | int | float], np.ndarray, bool, Dict]:
        """Take an action: open group (pair of nodes)."""
        done = False
        info: Dict = {}

        # action validity
        a = int(action_idx)
        i = int(self.pairs_i[a])
        j = int(self.pairs_j[a])

        if not (self.remaining[i] and self.remaining[j]):
            # invalid: repeated node
            r = np.asarray([-self.cfg.infeasible_penalty, -self.cfg.infeasible_penalty], dtype=np.float32)
            return self._obs(), r, True, {"reason": "repeated_node"}

        if not self.group_table.feasible[self.scenario, a]:
            # invalid: violates Hmin
            r = np.asarray([-self.cfg.infeasible_penalty, -self.cfg.infeasible_penalty], dtype=np.float32)
            return self._obs(), r, True, {"reason": "hydraulic_infeasible"}

        # apply action: remove two nodes
        self.remaining[i] = False
        self.remaining[j] = False

        # add 4 surpluses for this group
        sur4 = self.group_table.surpluses[self.scenario, a, :].tolist()
        self.surpluses_all.extend(sur4)

        # compute reward as negative incremental change of (var, mean)
        var_new, mean_new = self._objective_stats()
        d_var = var_new - self.prev_var
        d_mean = mean_new - self.prev_mean
        self.prev_var, self.prev_mean = var_new, mean_new

        r_vec = np.asarray([-d_var, -d_mean], dtype=np.float32)

        # advance time
        self.t += 1
        if self.t >= self.cfg.n_groups:
            done = True
        else:
            # scenario evolves (dynamic)
            self.scenario = int(self.rng.integers(0, self.K))

        info.update(
            {
                "var": var_new,
                "mean": mean_new,
                "d_var": d_var,
                "d_mean": d_mean,
                "action_nodes": (i, j),
            }
        )
        return self._obs(), r_vec, done, info


# =========================
# Replay Buffer
# =========================


class ReplayBuffer:
    def __init__(self, capacity: int, n_nodes: int) -> None:
        self.capacity = int(capacity)
        self.n_nodes = int(n_nodes)
        self.ptr = 0
        self.size = 0

        self.mask = np.zeros((capacity, n_nodes), dtype=np.float32)
        self.t = np.zeros((capacity,), dtype=np.int16)
        self.scenario = np.zeros((capacity,), dtype=np.int16)
        self.ai = np.zeros((capacity,), dtype=np.int16)
        self.aj = np.zeros((capacity,), dtype=np.int16)
        self.r = np.zeros((capacity, 2), dtype=np.float32)
        self.nmask = np.zeros((capacity, n_nodes), dtype=np.float32)
        self.nt = np.zeros((capacity,), dtype=np.int16)
        self.nscenario = np.zeros((capacity,), dtype=np.int16)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.w = np.zeros((capacity, 2), dtype=np.float32)

    def add(
        self,
        mask: np.ndarray,
        t: int,
        scenario: int,
        action_nodes: Tuple[int, int],
        r_vec: np.ndarray,
        nmask: np.ndarray,
        nt: int,
        nscenario: int,
        done: bool,
        w: np.ndarray,
    ) -> None:
        i = self.ptr
        self.mask[i] = mask
        self.t[i] = t
        self.scenario[i] = scenario
        self.ai[i] = int(action_nodes[0])
        self.aj[i] = int(action_nodes[1])
        self.r[i] = r_vec
        self.nmask[i] = nmask
        self.nt[i] = nt
        self.nscenario[i] = nscenario
        self.done[i] = 1.0 if done else 0.0
        self.w[i] = w

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "mask": self.mask[idx],
            "t": self.t[idx],
            "scenario": self.scenario[idx],
            "ai": self.ai[idx],
            "aj": self.aj[idx],
            "r": self.r[idx],
            "nmask": self.nmask[idx],
            "nt": self.nt[idx],
            "nscenario": self.nscenario[idx],
            "done": self.done[idx],
            "w": self.w[idx],
        }


# =========================
# Normalizer
# =========================


class RunningNorm2:
    """Running normalization for 2D rewards."""

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.n = 0
        self.mean = np.zeros((2,), dtype=np.float64)
        self.m2 = np.zeros((2,), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones((2,), dtype=np.float64)
        var = self.m2 / (self.n - 1)
        return np.sqrt(np.maximum(var, self.eps))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std()


# =========================
# Networks
# =========================


class MOQNetwork(nn.Module):
    """Preference-conditioned multi-objective Q network.

    Inputs:
      - mask: (B, n_nodes) in {0,1} float
      - t: (B,) step index
      - scenario: (B,) scenario index
      - action nodes (ai, aj): (B,)
      - w: (B,2) preference weights

    Output:
      - Qvec: (B,2)

    Action selection uses scalarization q = w·Qvec.
    """

    def __init__(self, n_nodes: int, K: int, node_emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.K = K
        self.emb = nn.Embedding(n_nodes, node_emb_dim)

        # state features: raw mask (n_nodes) + aggregated embedding (node_emb_dim) + t_frac + scenario onehot(K)
        self.state_mlp = nn.Sequential(
            nn.Linear(n_nodes + node_emb_dim + 1 + K, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # action features: (emb_i+emb_j, |emb_i-emb_j|) => 2*node_emb_dim
        self.action_mlp = nn.Sequential(
            nn.Linear(2 * node_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # joint + preference -> Qvec
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def encode_state(self, mask: torch.Tensor, t: torch.Tensor, scenario: torch.Tensor, T: int) -> torch.Tensor:
        # mask: (B,n)
        B, n = mask.shape
        assert n == self.n_nodes

        # aggregate node embeddings over remaining nodes
        idx = torch.arange(self.n_nodes, device=mask.device).unsqueeze(0).expand(B, -1)
        node_e = self.emb(idx)  # (B,n,emb)
        wmask = mask.unsqueeze(-1)  # (B,n,1)
        agg = (node_e * wmask).sum(dim=1)  # (B,emb)
        denom = wmask.sum(dim=1).clamp_min(1.0)
        agg = agg / denom

        t_frac = (t.float() / float(max(1, T - 1))).unsqueeze(-1)  # (B,1)
        scen_oh = F.one_hot(scenario.long(), num_classes=self.K).float()  # (B,K)

        x = torch.cat([mask, agg, t_frac, scen_oh], dim=-1)
        return self.state_mlp(x)

    def encode_action(self, ai: torch.Tensor, aj: torch.Tensor) -> torch.Tensor:
        ei = self.emb(ai.long())
        ej = self.emb(aj.long())
        x = torch.cat([ei + ej, torch.abs(ei - ej)], dim=-1)
        return self.action_mlp(x)

    def forward(
        self,
        mask: torch.Tensor,
        t: torch.Tensor,
        scenario: torch.Tensor,
        ai: torch.Tensor,
        aj: torch.Tensor,
        w: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        s = self.encode_state(mask, t, scenario, T)
        a = self.encode_action(ai, aj)
        x = torch.cat([s, a, w.float()], dim=-1)
        return self.head(x)


class PolicyNet(nn.Module):
    """Preference-conditioned policy scoring network for variable action sets."""

    def __init__(self, n_nodes: int, K: int, node_emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.K = K
        self.emb = nn.Embedding(n_nodes, node_emb_dim)

        self.state_mlp = nn.Sequential(
            nn.Linear(n_nodes + node_emb_dim + 1 + K + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.action_mlp = nn.Sequential(
            nn.Linear(2 * node_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.logit_head = nn.Linear(hidden_dim + hidden_dim, 1)

    def encode_state(self, mask: torch.Tensor, t: torch.Tensor, scenario: torch.Tensor, w: torch.Tensor, T: int) -> torch.Tensor:
        B, n = mask.shape
        idx = torch.arange(self.n_nodes, device=mask.device).unsqueeze(0).expand(B, -1)
        node_e = self.emb(idx)
        wmask = mask.unsqueeze(-1)
        agg = (node_e * wmask).sum(dim=1)
        denom = wmask.sum(dim=1).clamp_min(1.0)
        agg = agg / denom

        t_frac = (t.float() / float(max(1, T - 1))).unsqueeze(-1)
        scen_oh = F.one_hot(scenario.long(), num_classes=self.K).float()

        x = torch.cat([mask, agg, t_frac, scen_oh, w.float()], dim=-1)
        return self.state_mlp(x)

    def encode_action(self, ai: torch.Tensor, aj: torch.Tensor) -> torch.Tensor:
        ei = self.emb(ai.long())
        ej = self.emb(aj.long())
        x = torch.cat([ei + ej, torch.abs(ei - ej)], dim=-1)
        return self.action_mlp(x)

    def logits_for_actions(
        self,
        mask: torch.Tensor,
        t: torch.Tensor,
        scenario: torch.Tensor,
        w: torch.Tensor,
        ai: torch.Tensor,
        aj: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """Compute logits for a batch of candidate actions.

        Shapes:
          mask: (1,n) or (B,n)
          ai,aj: (A,) action candidates for each state (assume B=1 for simplicity in training loop)
        """
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if scenario.dim() == 0:
            scenario = scenario.unsqueeze(0)
        if w.dim() == 1:
            w = w.unsqueeze(0)

        s = self.encode_state(mask, t, scenario, w, T)  # (1,h)
        s = s.expand(ai.shape[0], -1)  # (A,h)
        a = self.encode_action(ai, aj)  # (A,h)
        x = torch.cat([s, a], dim=-1)
        return self.logit_head(x).squeeze(-1)  # (A,)


class ValueNet(nn.Module):
    def __init__(self, n_nodes: int, K: int, node_emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.K = K
        self.emb = nn.Embedding(n_nodes, node_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(n_nodes + node_emb_dim + 1 + K + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, mask: torch.Tensor, t: torch.Tensor, scenario: torch.Tensor, w: torch.Tensor, T: int) -> torch.Tensor:
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        B, n = mask.shape
        idx = torch.arange(self.n_nodes, device=mask.device).unsqueeze(0).expand(B, -1)
        node_e = self.emb(idx)
        wmask = mask.unsqueeze(-1)
        agg = (node_e * wmask).sum(dim=1)
        denom = wmask.sum(dim=1).clamp_min(1.0)
        agg = agg / denom

        t_frac = (t.float() / float(max(1, T - 1))).unsqueeze(-1)
        scen_oh = F.one_hot(scenario.long(), num_classes=self.K).float()
        x = torch.cat([mask, agg, t_frac, scen_oh, w.float()], dim=-1)
        return self.mlp(x).squeeze(-1)


# =========================
# Preference sampling
# =========================


def sample_preference(alpha_var: float, alpha_mean: float, rng: np.random.Generator) -> np.ndarray:
    w = rng.dirichlet(np.asarray([alpha_var, alpha_mean], dtype=np.float64))
    return w.astype(np.float32)


# =========================
# Action selection helpers
# =========================


def eps_by_step(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return eps_end
    frac = step / float(max(1, decay_steps))
    return eps_start + frac * (eps_end - eps_start)


@torch.no_grad()
def select_action_dmoqn(
    net: MOQNetwork,
    env: IrrigationSchedulingEnv,
    obs: Dict,
    w: np.ndarray,
    cfg: EnvConfig,
    dcfg: DMOQNConfig,
    global_step: int,
    device: torch.device,
) -> Tuple[int, Optional[Tuple[int, int]]]:
    feas = env.feasible_actions()
    if feas.size == 0:
        return -1, None

    eps = eps_by_step(global_step, dcfg.eps_start, dcfg.eps_end, dcfg.eps_decay_steps)
    if random.random() < eps:
        a = int(np.random.choice(feas))
        i = int(env.pairs_i[a])
        j = int(env.pairs_j[a])
        return a, (i, j)

    # greedy
    mask = torch.tensor(obs["mask"], dtype=torch.float32, device=device).unsqueeze(0)
    t = torch.tensor([obs["t"]], dtype=torch.int64, device=device)
    scen = torch.tensor([obs["scenario"]], dtype=torch.int64, device=device)
    w_t = torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(0)

    ai = torch.tensor(env.pairs_i[feas], dtype=torch.int64, device=device)
    aj = torch.tensor(env.pairs_j[feas], dtype=torch.int64, device=device)

    qvec = net(mask.expand(ai.shape[0], -1), t.expand(ai.shape[0]), scen.expand(ai.shape[0]), ai, aj, w_t.expand(ai.shape[0], -1), cfg.n_groups)
    q = (qvec * w_t.expand(ai.shape[0], -1)).sum(dim=-1)
    best = int(torch.argmax(q).item())
    a = int(feas[best])
    i = int(env.pairs_i[a])
    j = int(env.pairs_j[a])
    return a, (i, j)


@torch.no_grad()
def greedy_action_dmoqn(
    net: MOQNetwork,
    env: IrrigationSchedulingEnv,
    obs: Dict,
    w: np.ndarray,
    cfg: EnvConfig,
    device: torch.device,
) -> Tuple[int, Optional[Tuple[int, int]]]:
    feas = env.feasible_actions()
    if feas.size == 0:
        return -1, None

    mask = torch.tensor(obs["mask"], dtype=torch.float32, device=device).unsqueeze(0)
    t = torch.tensor([obs["t"]], dtype=torch.int64, device=device)
    scen = torch.tensor([obs["scenario"]], dtype=torch.int64, device=device)
    w_t = torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(0)

    ai = torch.tensor(env.pairs_i[feas], dtype=torch.int64, device=device)
    aj = torch.tensor(env.pairs_j[feas], dtype=torch.int64, device=device)

    qvec = net(mask.expand(ai.shape[0], -1), t.expand(ai.shape[0]), scen.expand(ai.shape[0]), ai, aj, w_t.expand(ai.shape[0], -1), cfg.n_groups)
    q = (qvec * w_t.expand(ai.shape[0], -1)).sum(dim=-1)
    best = int(torch.argmax(q).item())
    a = int(feas[best])
    i = int(env.pairs_i[a])
    j = int(env.pairs_j[a])
    return a, (i, j)


@torch.no_grad()
def select_action_pg(
    policy: PolicyNet,
    env: IrrigationSchedulingEnv,
    obs: Dict,
    w: np.ndarray,
    cfg: EnvConfig,
    device: torch.device,
    sample: bool = True,
) -> Tuple[int, Optional[Tuple[int, int]], Optional[torch.Tensor]]:
    feas = env.feasible_actions()
    if feas.size == 0:
        return -1, None, None

    mask = torch.tensor(obs["mask"], dtype=torch.float32, device=device)
    t = torch.tensor(obs["t"], dtype=torch.int64, device=device)
    scen = torch.tensor(obs["scenario"], dtype=torch.int64, device=device)
    w_t = torch.tensor(w, dtype=torch.float32, device=device)

    ai = torch.tensor(env.pairs_i[feas], dtype=torch.int64, device=device)
    aj = torch.tensor(env.pairs_j[feas], dtype=torch.int64, device=device)

    logits = policy.logits_for_actions(mask, t, scen, w_t, ai, aj, cfg.n_groups)
    probs = torch.softmax(logits, dim=0)

    if sample:
        dist = torch.distributions.Categorical(probs=probs)
        k = int(dist.sample().item())
        logp = dist.log_prob(torch.tensor(k, device=device))
    else:
        k = int(torch.argmax(probs).item())
        logp = torch.log(probs[k].clamp_min(1e-12))

    a = int(feas[k])
    i = int(env.pairs_i[a])
    j = int(env.pairs_j[a])
    return a, (i, j), logp


# =========================
# Training: DMOQN
# =========================


def dmoqn_update(
    net: MOQNetwork,
    target: MOQNetwork,
    buf: ReplayBuffer,
    env_for_masking: IrrigationSchedulingEnv,
    pairs_i: np.ndarray,
    pairs_j: np.ndarray,
    group_table: GroupTable,
    cfg: EnvConfig,
    dcfg: DMOQNConfig,
    optim: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    batch = buf.sample(dcfg.batch_size)

    mask = torch.tensor(batch["mask"], dtype=torch.float32, device=device)
    t = torch.tensor(batch["t"], dtype=torch.int64, device=device)
    scen = torch.tensor(batch["scenario"], dtype=torch.int64, device=device)
    ai = torch.tensor(batch["ai"], dtype=torch.int64, device=device)
    aj = torch.tensor(batch["aj"], dtype=torch.int64, device=device)
    r = torch.tensor(batch["r"], dtype=torch.float32, device=device)
    nmask = torch.tensor(batch["nmask"], dtype=torch.float32, device=device)
    nt = torch.tensor(batch["nt"], dtype=torch.int64, device=device)
    nscen = torch.tensor(batch["nscenario"], dtype=torch.int64, device=device)
    done = torch.tensor(batch["done"], dtype=torch.float32, device=device)
    w = torch.tensor(batch["w"], dtype=torch.float32, device=device)

    # current Q vector
    qvec = net(mask, t, scen, ai, aj, w, cfg.n_groups)  # (B,2)

    # compute next action by greedy w·Q_online, but value from target network
    with torch.no_grad():
        q_next_vec = torch.zeros_like(qvec)

        # loop per sample (action sets vary)
        for b in range(qvec.shape[0]):
            if done[b].item() >= 0.5:
                q_next_vec[b] = 0.0
                continue

            # candidates: both nodes remaining AND hydraulically feasible for next scenario
            rem = nmask[b].cpu().numpy().astype(bool)
            k = int(nscen[b].item())
            cand = group_table.feasible[k] & rem[pairs_i] & rem[pairs_j]
            idx = np.nonzero(cand)[0]
            if idx.size == 0:
                q_next_vec[b] = 0.0
                continue

            # evaluate online for argmax
            ai_c = torch.tensor(pairs_i[idx], dtype=torch.int64, device=device)
            aj_c = torch.tensor(pairs_j[idx], dtype=torch.int64, device=device)

            mask_b = nmask[b].unsqueeze(0).expand(ai_c.shape[0], -1)
            t_b = nt[b].expand(ai_c.shape[0])
            scen_b = nscen[b].expand(ai_c.shape[0])
            w_b = w[b].unsqueeze(0).expand(ai_c.shape[0], -1)

            qv_online = net(mask_b, t_b, scen_b, ai_c, aj_c, w_b, cfg.n_groups)
            q_online = (qv_online * w_b).sum(dim=-1)
            best_idx = int(torch.argmax(q_online).item())

            # value from target network
            qv_t = target(mask_b[best_idx:best_idx+1], t_b[best_idx:best_idx+1], scen_b[best_idx:best_idx+1], ai_c[best_idx:best_idx+1], aj_c[best_idx:best_idx+1], w_b[best_idx:best_idx+1], cfg.n_groups)
            q_next_vec[b] = qv_t.squeeze(0)

        target_vec = r + (1.0 - done).unsqueeze(-1) * dcfg.gamma * q_next_vec

    loss = F.smooth_l1_loss(qvec, target_vec)

    optim.zero_grad(set_to_none=True)
    loss.backward()
    if dcfg.grad_clip is not None and dcfg.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(net.parameters(), dcfg.grad_clip)
    optim.step()

    return float(loss.item())


@torch.no_grad()
def evaluate_policy_dmoqn(
    net: MOQNetwork,
    env: IrrigationSchedulingEnv,
    cfg: EnvConfig,
    dcfg: DMOQNConfig,
    device: torch.device,
    n_episodes: int,
    w_eval: np.ndarray,
) -> Dict[str, float]:
    vars_: List[float] = []
    means_: List[float] = []
    fail = 0

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            a, nodes = greedy_action_dmoqn(net, env, obs, w_eval, cfg, device)
            if a < 0:
                fail += 1
                break
            obs, r, done, info = env.step(a)
            if done:
                vars_.append(float(info.get("var", 0.0)))
                means_.append(float(info.get("mean", 0.0)))

    out = {
        "episodes": float(n_episodes),
        "fail": float(fail),
        "fail_rate": float(fail) / float(max(1, n_episodes)),
        "var_mean": float(np.mean(vars_)) if vars_ else float("nan"),
        "var_median": float(np.median(vars_)) if vars_ else float("nan"),
        "mean_mean": float(np.mean(means_)) if means_ else float("nan"),
        "mean_median": float(np.median(means_)) if means_ else float("nan"),
    }
    return out


def train_dmoqn(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    outdir = ensure_dir(args.out)

    # Load network and evaluator
    nodes = load_nodes_xlsx(args.nodes)
    edges = load_pipes_xlsx(args.pipes)
    evaluator = TreeHydraulicEvaluator(nodes=nodes, edges=edges, root=args.root, H0=args.H0, Hmin=args.Hmin)

    base_nodes, pairs_i, pairs_j = build_base_nodes_and_pairs(nodes)
    lateral_ids, lateral_to_node = build_lateral_ids_for_field_nodes(base_nodes)

    env_cfg = EnvConfig(
        q_lateral=args.q_lateral,
        scenario_k=args.scenario_k,
        scenario_rel=tuple(args.scenario_rel),
        infeasible_penalty=args.infeasible_penalty,
    )

    dcfg = DMOQNConfig(
        total_env_steps=args.total_steps,
        eval_every_steps=args.eval_every,
        eval_episodes=args.eval_episodes,
        eps_decay_steps=args.eps_decay,
        pref_alpha_var=args.pref_alpha_var,
        pref_alpha_mean=args.pref_alpha_mean,
        reward_norm=bool(args.reward_norm),
    )

    scenario_rel = list(env_cfg.scenario_rel)
    scenario_rel = scenario_rel[: args.scenario_k]
    if len(scenario_rel) != args.scenario_k:
        raise ValueError("scenario_k and scenario_rel length mismatch")

    group_table = GroupTable(
        evaluator=evaluator,
        base_nodes=base_nodes,
        lateral_to_node=lateral_to_node,
        pairs_i=pairs_i,
        pairs_j=pairs_j,
        H0_base=args.H0,
        Hmin=args.Hmin,
        q_lateral=args.q_lateral,
        scenario_rel=scenario_rel,
    )

    if args.precompute:
        group_table.precompute(verbose=True)

    # RNG
    rng = np.random.default_rng(args.seed)

    env = IrrigationSchedulingEnv(base_nodes, pairs_i, pairs_j, group_table, env_cfg, rng)

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    net = MOQNetwork(n_nodes=len(base_nodes), K=group_table.K, node_emb_dim=dcfg.node_emb_dim, hidden_dim=dcfg.hidden_dim).to(device)
    target = MOQNetwork(n_nodes=len(base_nodes), K=group_table.K, node_emb_dim=dcfg.node_emb_dim, hidden_dim=dcfg.hidden_dim).to(device)
    target.load_state_dict(net.state_dict())
    target.eval()

    optim = torch.optim.Adam(net.parameters(), lr=dcfg.learning_rate)

    buf = ReplayBuffer(dcfg.buffer_size, n_nodes=len(base_nodes))

    # reward normalization
    rnorm = RunningNorm2(eps=dcfg.reward_norm_eps)

    # logging
    log_path = outdir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(
            [
                "global_step",
                "episode",
                "ep_len",
                "w_var",
                "w_mean",
                "final_var",
                "final_mean",
                "fail",
                "buffer_size",
                "loss_last",
                "eps",
            ]
        )

    global_step = 0
    episode = 0
    loss_last = float("nan")

    t0 = time.perf_counter()

    # periodic evaluation preference grid (small simplex line grid)
    eval_ws = make_weight_grid_line(args.eval_grid)
    # Keep a monotone (non-decreasing) reference point for comparable HV across evaluations
    hv_ref: tuple[float, float] | None = None

    # Use a separate RNG/env for evaluation to avoid perturbing training randomness
    rng_eval = np.random.default_rng(int(args.seed) + 999)
    eval_env = IrrigationSchedulingEnv(base_nodes, pairs_i, pairs_j, group_table, env_cfg, rng_eval)

    best_hv = -float("inf")
    best_score = float("inf")  # multi-weight average score (lower is better)
    best_path = outdir / "best_model.pt"

    # evaluation logs
    eval_sweep_path = outdir / "eval_sweep.csv"
    eval_summary_path = outdir / "eval_summary.csv"
    if not eval_sweep_path.exists():
        with eval_sweep_path.open("w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["global_step", "w_var", "w_mean", "episodes", "fail_rate", "var_mean", "mean_mean", "score"])
    if not eval_summary_path.exists():
        with eval_summary_path.open("w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["global_step", "hv", "avg_score", "feasible_w", "total_w", "mean_fail_rate", "ref_var", "ref_mean", "saved_best"])

    while global_step < dcfg.total_env_steps:
        episode += 1
        obs = env.reset()
        done = False
        ep_len = 0

        w_pref = sample_preference(dcfg.pref_alpha_var, dcfg.pref_alpha_mean, rng)

        final_var = float("nan")
        final_mean = float("nan")
        fail = 0

        while not done:
            a, nodes_ij = select_action_dmoqn(net, env, obs, w_pref, env_cfg, dcfg, global_step, device)

            if a < 0 or nodes_ij is None:
                # no feasible action
                r_vec = np.asarray([-env_cfg.infeasible_penalty, -env_cfg.infeasible_penalty], dtype=np.float32)
                if dcfg.reward_norm:
                    rnorm.update(r_vec)
                    r_vec_n = rnorm.normalize(r_vec).astype(np.float32)
                else:
                    r_vec_n = r_vec

                # terminal transition (self-loop)
                buf.add(
                    mask=np.asarray(obs["mask"], dtype=np.float32),
                    t=int(obs["t"]),
                    scenario=int(obs["scenario"]),
                    action_nodes=(0, 0),
                    r_vec=r_vec_n,
                    nmask=np.asarray(obs["mask"], dtype=np.float32),
                    nt=int(obs["t"]),
                    nscenario=int(obs["scenario"]),
                    done=True,
                    w=w_pref,
                )
                fail = 1
                break

            nobs, r_vec, done, info = env.step(a)

            final_var = float(info.get("var", 0.0))
            final_mean = float(info.get("mean", 0.0))

            if dcfg.reward_norm:
                rnorm.update(r_vec)
                r_vec_n = rnorm.normalize(r_vec).astype(np.float32)
            else:
                r_vec_n = r_vec

            buf.add(
                mask=np.asarray(obs["mask"], dtype=np.float32),
                t=int(obs["t"]),
                scenario=int(obs["scenario"]),
                action_nodes=nodes_ij,
                r_vec=r_vec_n,
                nmask=np.asarray(nobs["mask"], dtype=np.float32),
                nt=int(nobs["t"]),
                nscenario=int(nobs["scenario"]),
                done=bool(done),
                w=w_pref,
            )

            obs = nobs
            ep_len += 1
            global_step += 1

            # update
            if buf.size >= dcfg.train_after_steps and (global_step % dcfg.train_every == 0):
                net.train()
                loss_last = dmoqn_update(
                    net=net,
                    target=target,
                    buf=buf,
                    env_for_masking=env,
                    pairs_i=pairs_i,
                    pairs_j=pairs_j,
                    group_table=group_table,
                    cfg=env_cfg,
                    dcfg=dcfg,
                    optim=optim,
                    device=device,
                )
                net.eval()

            # target update
            if global_step > 0 and (global_step % dcfg.target_update_interval == 0):
                target.load_state_dict(net.state_dict())

            if global_step >= dcfg.total_env_steps:
                break

        eps_now = eps_by_step(global_step, dcfg.eps_start, dcfg.eps_end, dcfg.eps_decay_steps)

        # log episode
        with log_path.open("a", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(
                [
                    global_step,
                    episode,
                    ep_len,
                    float(w_pref[0]),
                    float(w_pref[1]),
                    final_var,
                    final_mean,
                    fail,
                    buf.size,
                    loss_last,
                    eps_now,
                ]
            )

        # periodic evaluation ===
        if global_step > 0 and (global_step % dcfg.eval_every_steps == 0):
            net.eval()

            rows = []
            pts = []
            scores = []
            fail_rates = []
            var_list = []
            mean_list = []

            for w in eval_ws:
                w_eval = np.asarray(w, dtype=np.float32)
                stats = evaluate_policy_dmoqn(net, eval_env, env_cfg, dcfg, device, dcfg.eval_episodes, w_eval)

                var_m = float(stats["var_mean"])
                mean_m = float(stats["mean_mean"])
                fail_r = float(stats["fail_rate"])

                # scalar score consistent with NSGA priority (var > mean)
                score = 3.0 * var_m + 2.0 * mean_m

                # Penalize failures / NaNs heavily in the multi-weight composite score
                if (not np.isfinite(score)) or (fail_r > 0.0):
                    score_eff = 1e9
                else:
                    score_eff = score

                rows.append([global_step, float(w_eval[0]), float(w_eval[1]), float(stats["episodes"]), fail_r, var_m, mean_m, score_eff])
                scores.append(score_eff)
                fail_rates.append(fail_r)

                if np.isfinite(var_m) and np.isfinite(mean_m):
                    var_list.append(var_m)
                    mean_list.append(mean_m)

                # For HV, keep only feasible/valid points (no failures)
                if (fail_r == 0.0) and np.isfinite(var_m) and np.isfinite(mean_m):
                    pts.append([var_m, mean_m])

            # Append sweep CSV
            with eval_sweep_path.open("a", newline="") as f:
                wcsv = csv.writer(f)
                wcsv.writerows(rows)

            avg_score = float(np.mean(scores)) if scores else float("inf")
            mean_fail = float(np.mean(fail_rates)) if fail_rates else 1.0

            # Update monotone HV reference point (use worst observed among the grid + margin)
            saved_best = 0
            hv = 0.0
            if var_list and mean_list:
                v_worst = float(np.max(var_list))
                m_worst = float(np.max(mean_list))
                cand_ref = (
                    v_worst * (1.0 + float(args.hv_ref_margin)) + 1e-6,
                    m_worst * (1.0 + float(args.hv_ref_margin)) + 1e-6,
                )
                if hv_ref is None:
                    hv_ref = cand_ref
                else:
                    hv_ref = (max(hv_ref[0], cand_ref[0]), max(hv_ref[1], cand_ref[1]))

                if pts:
                    hv = hypervolume_2d_min(np.asarray(pts, dtype=np.float64), hv_ref)

            feasible_w = int(len(pts))
            total_w = int(len(eval_ws))

            print(
                f"[EvalSweep] step={global_step} grid={total_w} feasible={feasible_w} "
                f"mean_fail_rate={mean_fail:.3f} hv={hv:.6g} avg_score={avg_score:.6g} "
                f"ref=({0.0 if hv_ref is None else hv_ref[0]:.6g},{0.0 if hv_ref is None else hv_ref[1]:.6g})"
            )

            # Best checkpoint selection
            improved = False
            if args.ckpt_metric == "hv":
                improved = hv > best_hv + 1e-12
            elif args.ckpt_metric == "score":
                improved = avg_score < best_score - 1e-12
            else:
                # hybrid: maximize HV; tie-break by avg_score
                if hv > best_hv + 1e-12:
                    improved = True
                elif abs(hv - best_hv) <= 1e-12 and avg_score < best_score - 1e-12:
                    improved = True

            if improved:
                best_hv = hv
                best_score = avg_score
                torch.save(
                    {
                        "algo": "dmoqn",
                        "model": net.state_dict(),
                        "n_nodes": len(base_nodes),
                        "K": group_table.K,
                        "node_emb_dim": dcfg.node_emb_dim,
                        "hidden_dim": dcfg.hidden_dim,
                        "env_cfg": env_cfg.__dict__,
                        "dmoqn_cfg": dcfg.__dict__,
                        "scenario_rel": scenario_rel,
                        "seed": args.seed,
                        "root": args.root,
                        "H0": float(args.H0),
                        "Hmin": float(args.Hmin),
                        "q_lateral": float(args.q_lateral),
                        "eval_grid": int(args.eval_grid),
                        "ckpt_metric": str(args.ckpt_metric),
                        "hv_ref": None if hv_ref is None else (float(hv_ref[0]), float(hv_ref[1])),
                        "eval_hv": float(hv),
                        "eval_avg_score": float(avg_score),
                        "eval_feasible_w": feasible_w,
                        "eval_total_w": total_w,
                    },
                    best_path,
                )
                saved_best = 1
                print(
                    f"[Checkpoint] saved best_model.pt (best_hv={best_hv:.6g}, "
                    f"best_avg_score={best_score:.6g}, metric={args.ckpt_metric})"
                )

            # Append summary CSV
            with eval_summary_path.open("a", newline="") as f:
                wcsv = csv.writer(f)
                wcsv.writerow(
                    [
                        global_step,
                        hv,
                        avg_score,
                        feasible_w,
                        total_w,
                        mean_fail,
                        0.0 if hv_ref is None else hv_ref[0],
                        0.0 if hv_ref is None else hv_ref[1],
                        saved_best,
                    ]
                )

    elapsed = time.perf_counter() - t0
    print(f"[Done] training finished. steps={global_step} episodes={episode} elapsed={elapsed:.2f}s")
    print(f"Outputs in: {outdir}")


# =========================
# Evaluation + Pareto sweep helpers
# =========================


def decode_schedule(
    base_nodes: List[str],
    schedule_ij: List[Tuple[int, int]],
) -> List[Dict[str, object]]:
    """Convert (i,j) index schedule to human-readable fields."""
    out: List[Dict[str, object]] = []
    for step, (i, j) in enumerate(schedule_ij, start=1):
        ni = base_nodes[int(i)]
        nj = base_nodes[int(j)]
        out.append(
            {
                "step": step,
                "node_i": ni,
                "node_j": nj,
                "laterals": [f"{ni}_L", f"{ni}_R", f"{nj}_L", f"{nj}_R"],
            }
        )
    return out


def save_schedule_csv(path: Path, decoded: List[Dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "node_i", "node_j", "lateral_1", "lateral_2", "lateral_3", "lateral_4"])
        for row in decoded:
            lats = list(row["laterals"])  # type: ignore[index]
            w.writerow([row["step"], row["node_i"], row["node_j"], lats[0], lats[1], lats[2], lats[3]])


def load_dmoqn_checkpoint(ckpt_path: str | Path, device: torch.device) -> Tuple[MOQNetwork, Dict]:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if ckpt.get("algo") not in (None, "dmoqn"):
        raise ValueError(f"Checkpoint algo mismatch: {ckpt.get('algo')}")

    n_nodes = int(ckpt["n_nodes"])
    K = int(ckpt["K"])
    node_emb_dim = int(ckpt["node_emb_dim"])
    hidden_dim = int(ckpt["hidden_dim"])

    net = MOQNetwork(n_nodes=n_nodes, K=K, node_emb_dim=node_emb_dim, hidden_dim=hidden_dim).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net, ckpt
@torch.no_grad()
def run_greedy_episode_dmoqn(
    net: MOQNetwork,
    env: IrrigationSchedulingEnv,
    cfg: EnvConfig,
    device: torch.device,
    w_eval: np.ndarray,
) -> Dict:
    obs = env.reset()
    schedule: List[Tuple[int, int]] = []
    scen_hist: List[int] = [int(obs["scenario"])]

    done = False
    fail = 0
    final_var = float("nan")
    final_mean = float("nan")

    while not done:
        a, ij = greedy_action_dmoqn(net, env, obs, w_eval, cfg, device)
        if a < 0 or ij is None:
            fail = 1
            break
        schedule.append(ij)

        obs, r, done, info = env.step(a)
        scen_hist.append(int(obs["scenario"]))

        if done:
            final_var = float(info.get("var", 0.0))
            final_mean = float(info.get("mean", 0.0))

    return {
        "schedule": schedule,
        "scenarios": scen_hist,
        "final_var": final_var,
        "final_mean": final_mean,
        "fail": fail,
    }


def evaluate_cmd(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    outdir = ensure_dir(args.out)

    # Build evaluator + environment
    nodes = load_nodes_xlsx(args.nodes)
    edges = load_pipes_xlsx(args.pipes)
    evaluator = TreeHydraulicEvaluator(nodes=nodes, edges=edges, root=args.root, H0=args.H0, Hmin=args.Hmin)

    base_nodes, pairs_i, pairs_j = build_base_nodes_and_pairs(nodes)
    _, lateral_to_node = build_lateral_ids_for_field_nodes(base_nodes)

    scenario_rel = list(args.scenario_rel[: args.scenario_k])

    group_table = GroupTable(
        evaluator=evaluator,
        base_nodes=base_nodes,
        lateral_to_node=lateral_to_node,
        pairs_i=pairs_i,
        pairs_j=pairs_j,
        H0_base=args.H0,
        Hmin=args.Hmin,
        q_lateral=args.q_lateral,
        scenario_rel=scenario_rel,
    )
    if args.precompute:
        group_table.precompute(verbose=True)

    env_cfg = EnvConfig(
        q_lateral=args.q_lateral,
        scenario_k=args.scenario_k,
        scenario_rel=tuple(scenario_rel),
        infeasible_penalty=args.infeasible_penalty,
    )

    rng = np.random.default_rng(args.seed)
    env = IrrigationSchedulingEnv(base_nodes, pairs_i, pairs_j, group_table, env_cfg, rng)

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

    w_eval = np.asarray([args.w_var, args.w_mean], dtype=np.float32)
    w_eval = w_eval / float(max(1e-12, float(np.sum(w_eval))))

    if args.algo.lower() == "dmoqn":
        net, ckpt = load_dmoqn_checkpoint(args.model, device)
        stats = evaluate_policy_dmoqn(net, env, env_cfg, DMOQNConfig(), device, args.episodes, w_eval)

        # also save one representative greedy schedule
        rep = run_greedy_episode_dmoqn(net, env, env_cfg, device, w_eval)
        decoded = decode_schedule(base_nodes, rep["schedule"])
        save_schedule_csv(outdir / "schedule.csv", decoded)

    else:
        raise ValueError("Only dmoqn evaluation is included in this continuation. If you trained PG, use the PG loader.")

    # dump stats
    with (outdir / "eval_stats.json").open("w", encoding="utf-8") as f:
        json.dump({"w": w_eval.tolist(), "stats": stats}, f, indent=2, ensure_ascii=False)

    print(f"[Eval] saved stats to {outdir / 'eval_stats.json'}")
    print(f"[Eval] saved schedule to {outdir / 'schedule.csv'}")


def pareto_sweep_cmd(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    outdir = ensure_dir(args.out)

    # Build evaluator + environment
    nodes = load_nodes_xlsx(args.nodes)
    edges = load_pipes_xlsx(args.pipes)
    evaluator = TreeHydraulicEvaluator(nodes=nodes, edges=edges, root=args.root, H0=args.H0, Hmin=args.Hmin)

    base_nodes, pairs_i, pairs_j = build_base_nodes_and_pairs(nodes)
    _, lateral_to_node = build_lateral_ids_for_field_nodes(base_nodes)

    scenario_rel = list(args.scenario_rel[: args.scenario_k])

    group_table = GroupTable(
        evaluator=evaluator,
        base_nodes=base_nodes,
        lateral_to_node=lateral_to_node,
        pairs_i=pairs_i,
        pairs_j=pairs_j,
        H0_base=args.H0,
        Hmin=args.Hmin,
        q_lateral=args.q_lateral,
        scenario_rel=scenario_rel,
    )
    if args.precompute:
        group_table.precompute(verbose=True)

    env_cfg = EnvConfig(
        q_lateral=args.q_lateral,
        scenario_k=args.scenario_k,
        scenario_rel=tuple(scenario_rel),
        infeasible_penalty=args.infeasible_penalty,
    )

    rng = np.random.default_rng(args.seed)
    env = IrrigationSchedulingEnv(base_nodes, pairs_i, pairs_j, group_table, env_cfg, rng)

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    net, ckpt = load_dmoqn_checkpoint(args.model, device)

    grid = int(args.grid)
    if grid < 2:
        raise ValueError("grid must be >=2")

    rows: List[Dict[str, float]] = []
    for k in range(grid):
        w_var = k / float(grid - 1)
        w_mean = 1.0 - w_var
        w = np.asarray([w_var, w_mean], dtype=np.float32)
        stats = evaluate_policy_dmoqn(net, env, env_cfg, DMOQNConfig(), device, args.episodes, w)
        rows.append(
            {
                "w_var": float(w_var),
                "w_mean": float(w_mean),
                "fail_rate": float(stats["fail_rate"]),
                "var_mean": float(stats["var_mean"]),
                "mean_mean": float(stats["mean_mean"]),
            }
        )
        print(
            f"[Sweep] w_var={w_var:.2f} w_mean={w_mean:.2f} "
            f"fail_rate={stats['fail_rate']:.3f} var={stats['var_mean']:.6g} mean={stats['mean_mean']:.6g}"
        )

    # write csv
    csv_path = outdir / "pareto_sweep.csv"
    with csv_path.open("w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["w_var", "w_mean", "fail_rate", "var_mean", "mean_mean"])
        for r in rows:
            wcsv.writerow([r["w_var"], r["w_mean"], r["fail_rate"], r["var_mean"], r["mean_mean"]])

    print(f"[Sweep] saved: {csv_path}")


# =========================
# CLI
# =========================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared
    def add_shared(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--nodes", type=str, required=True)
        sp.add_argument("--pipes", type=str, required=True)
        sp.add_argument("--root", type=str, required=True)
        sp.add_argument("--H0", type=float, required=True)
        sp.add_argument("--Hmin", type=float, required=True)
        sp.add_argument("--q_lateral", type=float, default=0.012)
        sp.add_argument("--scenario_k", type=int, default=3)
        sp.add_argument("--scenario_rel", type=float, nargs="+", default=[0.95, 1.0, 1.05])
        sp.add_argument("--precompute", type=int, default=1)
        sp.add_argument("--infeasible_penalty", type=float, default=50.0)
        sp.add_argument("--cuda", type=int, default=1)
        sp.add_argument("--seed", type=int, default=123)

    # train_dmoqn
    sp = sub.add_parser("train_dmoqn")
    add_shared(sp)
    sp.add_argument("--out", type=str, required=True)
    sp.add_argument("--total_steps", type=int, default=400_000)
    sp.add_argument("--eval_every", type=int, default=20_000)
    sp.add_argument("--eval_episodes", type=int, default=200)
    sp.add_argument("--eval_grid", type=int, default=7, help="Small weight grid size for periodic multi-weight evaluation")
    sp.add_argument(
    "--ckpt_metric",
    type=str,
    default="hybrid",
    choices=["hybrid", "hv", "score"],
    help="Best-checkpoint selection: hv=maximize hypervolume; score=minimize multi-weight average score; hybrid=hv then score",
    )
    sp.add_argument("--hv_ref_margin", type=float, default=0.10, help="Relative margin for HV reference point (monotone non-decreasing)")
    sp.add_argument("--eps_decay", type=int, default=150_000)
    sp.add_argument("--pref_alpha_var", type=float, default=3.0)
    sp.add_argument("--pref_alpha_mean", type=float, default=2.0)
    sp.add_argument("--reward_norm", type=int, default=1)

    # evaluate
    sp = sub.add_parser("evaluate")
    add_shared(sp)
    sp.add_argument("--model", type=str, required=True)
    sp.add_argument("--algo", type=str, default="dmoqn")
    sp.add_argument("--w_var", type=float, default=0.6)
    sp.add_argument("--w_mean", type=float, default=0.4)
    sp.add_argument("--episodes", type=int, default=200)
    sp.add_argument("--out", type=str, required=True)

    # pareto_sweep
    sp = sub.add_parser("pareto_sweep")
    add_shared(sp)
    sp.add_argument("--model", type=str, required=True)
    sp.add_argument("--algo", type=str, default="dmoqn")
    sp.add_argument("--grid", type=int, default=11)
    sp.add_argument("--episodes", type=int, default=200)
    sp.add_argument("--out", type=str, required=True)

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "train_dmoqn":
        train_dmoqn(args)
    elif args.cmd == "evaluate":
        evaluate_cmd(args)
    elif args.cmd == "pareto_sweep":
        pareto_sweep_cmd(args)
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
