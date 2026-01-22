# -*- coding: utf-8 -*-
"""DNN（策略网络）求解灌溉管网轮灌组优化：120 个斗管（Jxx_L/Jxx_R）分成 30 个轮灌组（每组 4 个）

已落实你的要求：
1) 目标：富余水头均值最小 + 富余水头方差最小 + “同一支管内完成”偏好（非强制）。权重比：方差:均值:同支管 = 3:2:1。
2) 约束：
   - 强约束：运行节点实际压力水头 >= 工作水头 H_req。
   - 覆盖约束：120 个斗管在 30 个轮灌组全部参与灌溉且不允许重复（每个斗管恰好出现一次）。
3) 输入：Nodes.xlsx, Pipes.xlsx；自动识别 60 个田间节点 J11..J16, J21..J26, ..., J101..J106，构造 120 个斗管。
4) 输出：
   - best_schedule.json：最佳轮灌组方案（30x4）与指标
   - near_opt_report.json：近最优量化报告
   - outdir/*.csv：仿 NSGA 的“可分析/可绘图”数据（收敛曲线、样本云图、基线、邻域检验等）

运行示例：
  python DNN_merged_full_with_NSGA_like_exports.py \
    --nodes Nodes.xlsx --pipes Pipes.xlsx \
    --episodes 20000 --samples 10000 --random_pool 2000 --local_iters 10000 \
    --log_every 50 --neighborhood_trials 5000 --outdir dnn_report_out

依赖：pandas, numpy, torch, openpyxl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# -----------------------------
# 0) CSV / Report helpers (NSGA-like)
# -----------------------------

def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def neighborhood_test_dnn(
    ev: "TreeHydraulicEvaluator",
    obj: "ScheduleObjective",
    ref_schedule: List[List[str]],
    trials: int,
    seed: int,
) -> Dict:
    """仿 NSGA 的邻域后验检验：围绕 ref_schedule 做随机扰动，统计邻域内可行率与更优比例。"""

    rng = random.Random(seed)

    perm = [x for g in ref_schedule for x in g]
    group_size = len(ref_schedule[0])
    n_groups = len(ref_schedule)

    ref_m = obj.evaluate(ref_schedule)
    ref_J = float(ref_m["J_norm"])

    feasible_n = 0
    better_pref = 0

    best_neighbor_m: Optional[Dict] = None
    best_neighbor_J = float("inf")

    for _ in range(trials):
        p = perm[:]

        # 70%：交换两条斗管；30%：交换两组
        if rng.random() < 0.7:
            i, j = rng.sample(range(len(p)), 2)
            p[i], p[j] = p[j], p[i]
        else:
            g1, g2 = rng.sample(range(n_groups), 2)
            s1, e1 = g1 * group_size, (g1 + 1) * group_size
            s2, e2 = g2 * group_size, (g2 + 1) * group_size
            p[s1:e1], p[s2:e2] = p[s2:e2], p[s1:e1]

        sched = [p[i * group_size:(i + 1) * group_size] for i in range(n_groups)]
        m = obj.evaluate(sched)
        if m["feasible"] < 0.5:
            continue

        feasible_n += 1
        J = float(m["J_norm"])
        if J + 1e-12 < ref_J:
            better_pref += 1

        if J < best_neighbor_J:
            best_neighbor_J = J
            best_neighbor_m = m

    out = {
        "trials": int(trials),
        "feasible_neighbors": int(feasible_n),
        "p_feasible": float(feasible_n) / float(trials) if trials else 0.0,
        "better_pref": int(better_pref),
        "p_better_pref_given_feasible": float(better_pref) / float(feasible_n) if feasible_n else 0.0,
        "ref_J_norm": float(ref_J),
        "ref_var": float(ref_m["var_surplus"]),
        "ref_mean": float(ref_m["mean_surplus"]),
        "ref_cohesion": float(ref_m["cohesion_pen"]),
        "ref_min_pressure": float(ref_m["min_pressure"]),
    }

    if best_neighbor_m is not None:
        out.update(
            {
                "best_neighbor_J_norm": float(best_neighbor_m["J_norm"]),
                "best_neighbor_var": float(best_neighbor_m["var_surplus"]),
                "best_neighbor_mean": float(best_neighbor_m["mean_surplus"]),
                "best_neighbor_cohesion": float(best_neighbor_m["cohesion_pen"]),
                "best_neighbor_min_pressure": float(best_neighbor_m["min_pressure"]),
            }
        )

    return out


def write_dnn_report_files(
    outdir: str,
    args,
    pack: Dict,
    bf_m: Dict,
    rnd1_m: Dict,
    policy_best_m: Dict,
    local_best_m: Dict,
    policy_samples_rows: List[Dict],
    random_pool_rows: List[Dict],
    neighborhood: Dict,
) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # (1) convergence_best_run.csv
    conv_rows = pack.get("train_rows", [])
    _write_csv(
        out / "convergence_best_run.csv",
        fieldnames=[
            "ep", "feasible", "reward", "J_norm", "J_raw", "mean_surplus", "var_surplus",
            "cohesion_pen", "min_pressure", "baseline", "entropy", "best_J_norm_so_far", "elapsed_s",
        ],
        rows=conv_rows,
    )

    # (2) policy_samples.csv (approx Pareto cloud)
    _write_csv(
        out / "policy_samples.csv",
        fieldnames=["k", "feasible", "J_norm", "J_raw", "mean_surplus", "var_surplus", "cohesion_pen", "min_pressure"],
        rows=policy_samples_rows,
    )

    # (3) random_pool.csv
    _write_csv(
        out / "random_pool.csv",
        fieldnames=["i", "feasible", "J_norm", "J_raw", "mean_surplus", "var_surplus", "cohesion_pen", "min_pressure"],
        rows=random_pool_rows,
    )

    # (4) baselines.csv
    base_rows: List[Dict] = []
    for name, m in [
        ("branch_first", bf_m),
        ("random_one", rnd1_m),
        ("policy_best", policy_best_m),
        ("local_best", local_best_m),
    ]:
        base_rows.append(
            {
                "name": name,
                "feasible": int(m.get("feasible", 0.0) > 0.5),
                "J_norm": float(m.get("J_norm", float("nan"))),
                "J_raw": float(m.get("J_raw", float("nan"))),
                "var_surplus": float(m.get("var_surplus", float("nan"))),
                "mean_surplus": float(m.get("mean_surplus", float("nan"))),
                "cohesion_pen": float(m.get("cohesion_pen", float("nan"))),
                "min_pressure": float(m.get("min_pressure", float("nan"))),
            }
        )
    _write_csv(
        out / "baselines.csv",
        fieldnames=["name", "feasible", "J_norm", "J_raw", "var_surplus", "mean_surplus", "cohesion_pen", "min_pressure"],
        rows=base_rows,
    )

    # (5) neighborhood_test.csv
    _write_csv(out / "neighborhood_test.csv", fieldnames=list(neighborhood.keys()), rows=[neighborhood])

    # (6) runs_summary.csv (single-run)
    run_row = {
        "seed": int(args.seed),
        "feasible": int(local_best_m.get("feasible", 0.0) > 0.5),
        "J_norm": float(local_best_m.get("J_norm", float("nan"))),
        "var_surplus": float(local_best_m.get("var_surplus", float("nan"))),
        "mean_surplus": float(local_best_m.get("mean_surplus", float("nan"))),
        "cohesion_pen": float(local_best_m.get("cohesion_pen", float("nan"))),
        "min_pressure": float(local_best_m.get("min_pressure", float("nan"))),
        "episodes": int(args.episodes),
        "samples": int(args.samples),
        "random_pool": int(args.random_pool),
        "local_iters": int(args.local_iters),
    }
    _write_csv(
        out / "runs_summary.csv",
        fieldnames=[
            "seed", "feasible", "J_norm", "var_surplus", "mean_surplus", "cohesion_pen", "min_pressure",
            "episodes", "samples", "random_pool", "local_iters",
        ],
        rows=[run_row],
    )


# -----------------------------
# 1) 水力评价器：树状稳态（Hazen-Williams）
# -----------------------------

@dataclass(frozen=True)
class HydraulicConfig:
    H_source: float = 25.0         # 水源可提供总水头（m）
    H_req: float = 11.59           # 斗管进口最低所需压力水头（m）
    q_lateral: float = 0.012       # 单个斗管流量（m3/s）
    C_UPVC: float = 150.0          # Hazen-Williams C
    C_PE: float = 140.0            # Hazen-Williams C


class TreeHydraulicEvaluator:
    """基于 Nodes/Pipes 表，构建以水库为根的有向树（FromNode->ToNode）。"""

    def __init__(self, nodes_xlsx: str, pipes_xlsx: str, cfg: HydraulicConfig):
        self.cfg = cfg
        self.nodes_df = pd.read_excel(nodes_xlsx)
        self.pipes_df = pd.read_excel(pipes_xlsx)

        for col in ["Nodel ID", "Z"]:
            if col not in self.nodes_df.columns:
                raise ValueError(f"Nodes.xlsx 缺少列: {col}")
        for col in ["FromNode", "ToNode", "Length_m", "Diameter_m", "Material"]:
            if col not in self.pipes_df.columns:
                raise ValueError(f"Pipes.xlsx 缺少列: {col}")

        self.Z: Dict[str, float] = self.nodes_df.set_index("Nodel ID")["Z"].astype(float).to_dict()

        self.root = self._infer_root_node(self.nodes_df)

        self.parent: Dict[str, str] = {}
        self.children: Dict[str, List[str]] = {}
        self.pipe_by_to: Dict[str, Dict] = {}

        for _, r in self.pipes_df.iterrows():
            u = str(r["FromNode"]).strip()
            v = str(r["ToNode"]).strip()
            self.parent[v] = u
            self.children.setdefault(u, []).append(v)
            self.pipe_by_to[v] = {
                "Pipe ID": str(r.get("Pipe ID", "")),
                "FromNode": u,
                "ToNode": v,
                "Length_m": float(r["Length_m"]),
                "Diameter_m": float(r["Diameter_m"]),
                "Material": str(r["Material"]).strip(),
            }

        self.topo = self._preorder(self.root)
        self.post = list(reversed(self.topo))

        self.field_nodes = self._infer_field_nodes(list(self.Z.keys()))
        self.laterals = self._build_laterals(self.field_nodes)
        self.branch_id = {lat: self._branch_of_lateral(lat) for lat in self.laterals}

        self.node_feat = self._compute_node_features()
        self.lateral_feat = self._compute_lateral_features()

    @staticmethod
    def _infer_root_node(nodes_df: pd.DataFrame) -> str:
        if "Node Type" in nodes_df.columns:
            roots = nodes_df.loc[nodes_df["Node Type"].astype(str).str.contains("水库"), "Nodel ID"].tolist()
            if roots:
                return str(roots[0]).strip()
        return "J0"

    @staticmethod
    def _node_num(node_id: str) -> Optional[int]:
        node_id = str(node_id).strip()
        if not node_id.startswith("J"):
            return None
        try:
            return int(node_id[1:])
        except Exception:
            return None

    def _infer_field_nodes(self, all_nodes: List[str]) -> List[str]:
        out: List[str] = []
        for nid in all_nodes:
            n = self._node_num(nid)
            if n is None:
                continue
            if n >= 11 and (n % 10) in (1, 2, 3, 4, 5, 6):
                out.append(str(nid).strip())
        out = sorted(out, key=lambda x: self._node_num(x) or 0)
        if len(out) != 60:
            print(f"[WARN] 识别到田间节点 {len(out)} 个（期望 60）。若命名规则不同，请修改 _infer_field_nodes()。")
        return out

    @staticmethod
    def _build_laterals(field_nodes: List[str]) -> List[str]:
        laterals: List[str] = []
        for n in field_nodes:
            laterals.append(f"{n}_L")
            laterals.append(f"{n}_R")
        return laterals

    def _branch_of_lateral(self, lat: str) -> int:
        nid = lat.split("_")[0]
        n = self._node_num(nid)
        if n is None:
            return -1
        return n // 10

    def _preorder(self, root: str) -> List[str]:
        order: List[str] = []
        stack = [root]
        seen = set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            order.append(u)
            for v in self.children.get(u, []):
                stack.append(v)
        return order

    def _hazen_williams_hf(self, pipe: Dict, Q: float) -> float:
        L = float(pipe["Length_m"])
        D = float(pipe["Diameter_m"])
        mat = str(pipe["Material"]).upper()
        C = self.cfg.C_UPVC if mat == "UPVC" else self.cfg.C_PE
        if Q <= 0.0:
            return 0.0
        return 10.67 * L * (Q ** 1.852) / ((C ** 1.852) * (D ** 4.87))

    def _compute_node_features(self) -> Dict[str, Dict[str, float]]:
        feat: Dict[str, Dict[str, float]] = {self.root: {"depth": 0.0, "path_len": 0.0, "res": 0.0}}
        for u in self.topo:
            for v in self.children.get(u, []):
                pipe = self.pipe_by_to[v]
                depth = feat[u]["depth"] + 1.0
                path_len = feat[u]["path_len"] + float(pipe["Length_m"])
                res = feat[u]["res"] + float(pipe["Length_m"]) / (float(pipe["Diameter_m"]) ** 4.87)
                feat[v] = {"depth": depth, "path_len": path_len, "res": res}
        return feat

    def _compute_lateral_features(self) -> Dict[str, np.ndarray]:
        feats: Dict[str, np.ndarray] = {}
        branches = [self.branch_id[lat] for lat in self.laterals]
        bmin, bmax = (min(branches), max(branches)) if branches else (0, 1)
        denom = max(1, bmax - bmin)

        for lat in self.laterals:
            nid, side = lat.split("_")
            z = float(self.Z.get(nid, 0.0))
            nf = self.node_feat.get(nid, {"depth": 0.0, "path_len": 0.0, "res": 0.0})
            depth = float(nf["depth"])
            path_len = float(nf["path_len"])
            res = float(nf["res"])
            bid = float(self.branch_id[lat])
            bid_norm = (bid - bmin) / denom
            side_f = 0.0 if side.upper() == "L" else 1.0
            feats[lat] = np.array([z, depth, path_len, res, bid_norm, side_f], dtype=np.float32)
        return feats

    def evaluate_group(self, group_laterals: List[str]) -> Dict:
        demand: Dict[str, float] = {n: 0.0 for n in self.topo}
        for lat in group_laterals:
            nid = lat.split("_")[0]
            if nid not in demand:
                continue
            demand[nid] += self.cfg.q_lateral

        flow: Dict[str, float] = {n: 0.0 for n in self.topo}
        for u in self.post:
            q = demand.get(u, 0.0)
            for c in self.children.get(u, []):
                q += flow[c]
            flow[u] = q

        H_total: Dict[str, float] = {self.root: float(self.Z.get(self.root, 0.0)) + self.cfg.H_source}
        for u in self.topo:
            Hu = H_total.get(u)
            if Hu is None:
                continue
            for v in self.children.get(u, []):
                pipe = self.pipe_by_to[v]
                Q = flow[v]
                hf = self._hazen_williams_hf(pipe, Q)
                H_total[v] = Hu - hf

        pressure_by_lat: Dict[str, float] = {}
        feasible = True
        min_pressure = float("inf")
        for lat in group_laterals:
            nid = lat.split("_")[0]
            if nid not in H_total:
                pressure_by_lat[lat] = float("nan")
                feasible = False
                continue
            Hp = float(H_total[nid]) - float(self.Z.get(nid, 0.0))
            pressure_by_lat[lat] = Hp
            min_pressure = min(min_pressure, Hp)
            if Hp + 1e-9 < self.cfg.H_req:
                feasible = False

        surplus_by_lat = {lat: pressure_by_lat[lat] - self.cfg.H_req for lat in group_laterals}
        return {
            "feasible": feasible,
            "min_pressure": min_pressure,
            "pressure_by_lat": pressure_by_lat,
            "surplus_by_lat": surplus_by_lat,
        }


# -----------------------------
# 2) 目标函数（含规范化）
# -----------------------------

class ScheduleObjective:
    """J_norm = 3*(var/var_ref) + 2*(mean/mean_ref) + cohesion_pen"""

    def __init__(self, evaluator: TreeHydraulicEvaluator, ref_schedule: Optional[List[List[str]]] = None):
        self.ev = evaluator
        self.branch_id = evaluator.branch_id

        if ref_schedule is None:
            ref_schedule = make_schedule_branch_first(evaluator.laterals, evaluator.branch_id, seed=0)

        ref_stats = self._stats(ref_schedule)

        # 若基线不可行，自动找一个随机可行解做 reference
        if ref_stats["feasible"] < 0.5:
            for i in range(2000):
                trial = make_schedule_random(self.ev.laterals, seed=10000 + i)
                s = self._stats(trial)
                if s["feasible"] > 0.5:
                    ref_stats = s
                    break

        self.mean_ref = max(1e-9, float(ref_stats["mean_surplus"]))
        self.var_ref = max(1e-9, float(ref_stats["var_surplus"]))

    def _stats(self, schedule: List[List[str]]) -> Dict[str, float]:
        surpluses: List[float] = []
        cohesion_pens: List[float] = []
        feasible = True
        min_pressure = float("inf")

        for g in schedule:
            gr = self.ev.evaluate_group(g)
            feasible = feasible and gr["feasible"]
            min_pressure = min(min_pressure, gr["min_pressure"])

            for lat in g:
                surpluses.append(float(gr["surplus_by_lat"][lat]))

            uniq = len(set(self.branch_id[lat] for lat in g))
            cohesion_pens.append((uniq - 1) / 3.0)

        s = np.asarray(surpluses, dtype=float)
        mean_s = float(np.mean(s))
        var_s = float(np.var(s))
        cohesion_pen = float(np.mean(cohesion_pens))

        J_raw = 3.0 * var_s + 2.0 * mean_s + 1.0 * cohesion_pen

        return {
            "feasible": float(feasible),
            "min_pressure": float(min_pressure),
            "mean_surplus": mean_s,
            "var_surplus": var_s,
            "cohesion_pen": cohesion_pen,
            "J_raw": float(J_raw),
        }

    def evaluate(self, schedule: List[List[str]]) -> Dict[str, float]:
        stats = self._stats(schedule)
        mean_s = float(stats["mean_surplus"])
        var_s = float(stats["var_surplus"])
        cohesion_pen = float(stats["cohesion_pen"])

        J_norm = 3.0 * (var_s / self.var_ref) + 2.0 * (mean_s / self.mean_ref) + 1.0 * cohesion_pen
        stats["J_norm"] = float(J_norm)
        return stats


# -----------------------------
# 3) DNN 策略网络（REINFORCE + baseline）
# -----------------------------

class PolicyNet(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 128):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.ctx = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, emb_rem: torch.Tensor, ctx_group: torch.Tensor, ctx_global: torch.Tensor) -> torch.Tensor:
        N, H = emb_rem.shape
        ctx = self.ctx(torch.cat([ctx_group, ctx_global], dim=-1)).unsqueeze(0).expand(N, H)
        logits = self.score(torch.cat([emb_rem, ctx], dim=-1)).squeeze(-1)
        return logits


@torch.no_grad()
def greedy_rollout(net: PolicyNet, ev: TreeHydraulicEvaluator, objective: ScheduleObjective,
                   group_size: int = 4, n_groups: int = 30, device: str = "cpu") -> Tuple[List[List[str]], Dict]:
    schedule = rollout(net, ev, objective, group_size, n_groups, device=device, greedy=True)[0]
    metrics = objective.evaluate(schedule)
    return schedule, metrics


def rollout(net: PolicyNet, ev: TreeHydraulicEvaluator, objective: ScheduleObjective,
            group_size: int = 4, n_groups: int = 30, device: str = "cpu",
            greedy: bool = False,
            infeasible_penalty: float = 50.0,
            repair_trials: int = 40) -> Tuple[List[List[str]], torch.Tensor, torch.Tensor, bool]:

    lats = ev.laterals
    feats = torch.tensor(np.stack([ev.lateral_feat[lat] for lat in lats], axis=0), device=device)
    emb_all = net.embed(feats)

    remaining = list(range(len(lats)))
    chosen_global: List[int] = []

    schedule: List[List[str]] = []
    sum_logprob = torch.tensor(0.0, device=device)
    sum_entropy = torch.tensor(0.0, device=device)

    feasible_episode = True

    for _ in range(n_groups):
        chosen_group: List[int] = []

        for _k in range(group_size):
            if chosen_group:
                ctx_group = emb_all[torch.tensor(chosen_group, device=device)].mean(dim=0)
            else:
                ctx_group = torch.zeros(emb_all.shape[1], device=device)

            if chosen_global:
                ctx_global = emb_all[torch.tensor(chosen_global, device=device)].mean(dim=0)
            else:
                ctx_global = torch.zeros(emb_all.shape[1], device=device)

            rem_idx = torch.tensor(remaining, device=device, dtype=torch.long)
            emb_rem = emb_all[rem_idx]
            logits = net(emb_rem, ctx_group, ctx_global)

            if greedy:
                pick_pos = int(torch.argmax(logits).item())
                pick = remaining[pick_pos]
            else:
                dist = torch.distributions.Categorical(logits=logits)
                pick_pos = int(dist.sample().item())
                pick = remaining[pick_pos]
                sum_logprob = sum_logprob + dist.log_prob(torch.tensor(pick_pos, device=device))
                sum_entropy = sum_entropy + dist.entropy()

            chosen_group.append(pick)
            chosen_global.append(pick)
            remaining.pop(pick_pos)

        group_lats = [lats[i] for i in chosen_group]
        gr = ev.evaluate_group(group_lats)

        if (not gr["feasible"]) and (not greedy):
            repaired = False
            for _ in range(repair_trials):
                if not remaining:
                    break
                out_pos = random.randrange(group_size)
                in_pick_pos = random.randrange(len(remaining))
                out_idx = chosen_group[out_pos]
                in_idx = remaining[in_pick_pos]

                trial = chosen_group[:]
                trial[out_pos] = in_idx
                trial_lats = [lats[i] for i in trial]
                if ev.evaluate_group(trial_lats)["feasible"]:
                    chosen_group = trial
                    remaining[in_pick_pos] = out_idx
                    repaired = True
                    break

            if not repaired:
                feasible_episode = False

        schedule.append([lats[i] for i in chosen_group])

    return schedule, sum_logprob, sum_entropy, feasible_episode


def train_dnn(
    ev: TreeHydraulicEvaluator,
    episodes: int = 10000,
    lr: float = 1e-3,
    seed: int = 0,
    hidden: int = 128,
    entropy_coef: float = 0.01,
    infeasible_penalty: float = 50.0,
    device: str = "cpu",
    log_every: int = 50,
) -> Tuple[PolicyNet, ScheduleObjective, Dict]:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    objective = ScheduleObjective(ev)
    net = PolicyNet(feat_dim=6, hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    baseline = 0.0
    best = {"J_norm": float("inf"), "schedule": None, "metrics": None}

    train_rows: List[Dict] = []

    for ep in range(1, episodes + 1):
        net.train()
        t0 = time.perf_counter()

        sched, sum_logprob, sum_entropy, feasible_rollout = rollout(
            net, ev, objective,
            group_size=4, n_groups=30, device=device,
            greedy=False,
            infeasible_penalty=infeasible_penalty,
        )

        metrics = objective.evaluate(sched)
        J = float(metrics["J_norm"])

        reward = -J
        feasible = bool(feasible_rollout and metrics["feasible"] > 0.5)
        if not feasible:
            reward -= infeasible_penalty

        baseline = 0.90 * baseline + 0.10 * reward
        adv = reward - baseline
        loss = -(adv * sum_logprob) - entropy_coef * sum_entropy

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if feasible and J < float(best["J_norm"]):
            best["J_norm"] = J
            best["schedule"] = sched
            best["metrics"] = metrics

        if (log_every <= 1) or (ep % log_every == 0) or (ep in (1, 10, 50, 100)):
            row = {
                "ep": int(ep),
                "feasible": int(feasible),
                "reward": float(reward),
                "J_norm": float(J),
                "J_raw": float(metrics.get("J_raw", float("nan"))),
                "mean_surplus": float(metrics["mean_surplus"]),
                "var_surplus": float(metrics["var_surplus"]),
                "cohesion_pen": float(metrics["cohesion_pen"]),
                "min_pressure": float(metrics["min_pressure"]),
                "baseline": float(baseline),
                "entropy": float(sum_entropy.detach().cpu().item()) if torch.is_tensor(sum_entropy) else float(sum_entropy),
                "best_J_norm_so_far": float(best["J_norm"]),
                "elapsed_s": float(time.perf_counter() - t0),
            }
            train_rows.append(row)
            print(f"ep={ep:5d}  J_norm={J:8.4f}  reward={reward:8.4f}  feasible={feasible}")

    log = {
        "episodes": episodes,
        "best_J_norm": float(best["J_norm"]),
        "best_metrics": best["metrics"],
        "baseline_ref": {"mean_ref": objective.mean_ref, "var_ref": objective.var_ref},
    }

    return net, objective, {"train_log": log, "best": best, "train_rows": train_rows}


# -----------------------------
# 4) 基线、局部搜索、近最优量化报告
# -----------------------------


def make_schedule_branch_first(laterals: List[str], branch_id: Dict[str, int], seed: int = 0) -> List[List[str]]:
    rng = random.Random(seed)
    by: Dict[int, List[str]] = {}
    for lat in laterals:
        by.setdefault(branch_id[lat], []).append(lat)
    schedule: List[List[str]] = []
    for b in sorted(by.keys()):
        xs = by[b][:]
        rng.shuffle(xs)
        for i in range(0, len(xs), 4):
            schedule.append(xs[i:i + 4])
    return schedule


def make_schedule_random(laterals: List[str], seed: int = 0) -> List[List[str]]:
    rng = random.Random(seed)
    xs = laterals[:]
    rng.shuffle(xs)
    return [xs[i:i + 4] for i in range(0, len(xs), 4)]


def is_schedule_valid(schedule: List[List[str]]) -> bool:
    flat = [x for g in schedule for x in g]
    return (
        (len(flat) == 120)
        and (len(set(flat)) == 120)
        and all(len(g) == 4 for g in schedule)
        and (len(schedule) == 30)
    )


def local_search_improve(
    schedule: List[List[str]],
    objective: ScheduleObjective,
    iters: int = 8000,
    seed: int = 0,
) -> Tuple[List[List[str]], Dict]:

    rng = random.Random(seed)
    best = [g[:] for g in schedule]
    best_m = objective.evaluate(best)
    best_J = float(best_m["J_norm"])

    positions: List[Tuple[int, int]] = []
    for gi, g in enumerate(best):
        for li in range(len(g)):
            positions.append((gi, li))

    for t in range(1, iters + 1):
        (g1, i1), (g2, i2) = rng.sample(positions, 2)
        if g1 == g2:
            continue

        cand = [g[:] for g in best]
        cand[g1][i1], cand[g2][i2] = cand[g2][i2], cand[g1][i1]

        if not is_schedule_valid(cand):
            continue

        m = objective.evaluate(cand)
        if m["feasible"] < 0.5:
            continue

        J = float(m["J_norm"])
        if J + 1e-12 < best_J:
            best, best_J, best_m = cand, J, m

        if t in (1000, 3000, 5000) or (t % 4000 == 0):
            print(f"local t={t:6d}  best_J_norm={best_J:8.4f}")

    return best, best_m


def near_opt_report(best: Dict, baselines: Dict[str, Dict], random_pool: List[Dict], local_best: Dict) -> Dict:
    J_best = float(best["J_norm"])
    J_bf = float(baselines["branch_first"]["J_norm"])

    better = sum(1 for r in random_pool if float(r["J_norm"]) + 1e-12 < J_best)
    N = max(1, len(random_pool))

    p_upper_95 = 1.0 - (0.05 ** (1.0 / N)) if better == 0 else None
    percentile = float((N - better) / N)

    J_local = float(local_best["J_norm"])
    local_improve = (J_best - J_local) / max(1e-12, abs(J_best))

    gap_vs_bf = (J_bf - J_best) / max(1e-12, abs(J_bf))

    return {
        "best": best,
        "baseline_branch_first": baselines["branch_first"],
        "baseline_random_one": baselines["random_one"],
        "gap_vs_branch_first_fraction": gap_vs_bf,
        "random_pool": {
            "N": N,
            "count_better_than_best": better,
            "percentile_ge_best": percentile,
            "p_upper_95_random_beats_best": p_upper_95,
        },
        "local_search": {
            "J_norm_after_local_search": J_local,
            "improvement_fraction": local_improve,
        },
        "interpretation": {
            "rule_of_thumb": "若 best 的 percentile_ge_best >= 0.99 且局部搜索改进 <1%，可视为工程近最优（相对随机/邻域）。",
            "note": "该报告提供经验近最优证据，不等价于严格全局最优证明。",
        },
    }


# -----------------------------
# 5) 主程序
# -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=str, default="Nodes.xlsx")
    ap.add_argument("--pipes", type=str, default="Pipes.xlsx")

    ap.add_argument("--H_source", type=float, default=25.0)
    ap.add_argument("--H_req", type=float, default=11.59)
    ap.add_argument("--q_lateral", type=float, default=0.012)
    ap.add_argument("--C_UPVC", type=float, default=150.0)
    ap.add_argument("--C_PE", type=float, default=140.0)

    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--entropy", type=float, default=0.01)
shi
    ap.add_argument("--samples", type=int, default=2000, help="训练后，从策略网络采样的候选解数量")
    ap.add_argument("--random_pool", type=int, default=2000, help="用于‘近最优报告’的随机可行解数量")
    ap.add_argument("--local_iters", type=int, default=10000, help="局部搜索迭代次数")

    ap.add_argument("--out_json", type=str, default="best_schedule.json")
    ap.add_argument("--report_json", type=str, default="near_opt_report.json")

    # NSGA-like exports
    ap.add_argument("--outdir", type=str, default="dnn_report_out", help="导出CSV报表目录")
    ap.add_argument("--log_every", type=int, default=50, help="每多少个episode记录一次收敛日志")
    ap.add_argument("--neighborhood_trials", type=int, default=5000, help="邻域后验检验次数")

    args = ap.parse_args()

    cfg = HydraulicConfig(
        H_source=args.H_source,
        H_req=args.H_req,
        q_lateral=args.q_lateral,
        C_UPVC=args.C_UPVC,
        C_PE=args.C_PE,
    )

    ev = TreeHydraulicEvaluator(args.nodes, args.pipes, cfg)

    # 基线
    obj0 = ScheduleObjective(ev)
    bf = make_schedule_branch_first(ev.laterals, ev.branch_id, seed=args.seed)
    rnd1 = make_schedule_random(ev.laterals, seed=args.seed)
    bf_m = obj0.evaluate(bf)
    rnd1_m = obj0.evaluate(rnd1)

    print("[baseline] branch-first:", bf_m)
    print("[baseline] random-one :", rnd1_m)

    # 训练
    net, obj, pack = train_dnn(
        ev,
        episodes=args.episodes,
        lr=args.lr,
        seed=args.seed,
        hidden=args.hidden,
        entropy_coef=args.entropy,
        device="cpu",
        log_every=args.log_every,
    )

    # 采样候选解并取最优（policy-best）
    net.eval()
    best_sched: Optional[List[List[str]]] = None
    best_m: Optional[Dict] = None
    best_J = float("inf")

    # greedy 候选
    g_sched, g_m = greedy_rollout(net, ev, obj)
    if g_m["feasible"] > 0.5 and float(g_m["J_norm"]) < best_J:
        best_sched, best_m, best_J = g_sched, g_m, float(g_m["J_norm"])

    policy_samples_rows: List[Dict] = []

    for k in range(args.samples):
        sched, _, _, feasible_rollout = rollout(net, ev, obj, greedy=False)
        m = obj.evaluate(sched)

        policy_samples_rows.append(
            {
                "k": int(k),
                "feasible": int((m["feasible"] > 0.5) and feasible_rollout),
                "J_norm": float(m["J_norm"]),
                "J_raw": float(m.get("J_raw", float("nan"))),
                "mean_surplus": float(m["mean_surplus"]),
                "var_surplus": float(m["var_surplus"]),
                "cohesion_pen": float(m["cohesion_pen"]),
                "min_pressure": float(m["min_pressure"]),
            }
        )

        if feasible_rollout and m["feasible"] > 0.5:
            J = float(m["J_norm"])
            if J < best_J:
                best_sched, best_m, best_J = sched, m, J

    if best_sched is None or best_m is None:
        raise RuntimeError("未找到可行解。请检查 q_lateral / H_source / H_req，或增大 episodes/samples。")

    print("[policy-best] metrics:", best_m)

    # 局部搜索进一步收敛（local-best）
    ls_sched, ls_m = local_search_improve(best_sched, obj, iters=args.local_iters, seed=args.seed)
    print("[local-best]  metrics:", ls_m)

    # 随机可行池（用于近最优量化 + 绘图）
    random_pool: List[Dict] = []
    random_pool_rows: List[Dict] = []

    for i in range(args.random_pool):
        sched = make_schedule_random(ev.laterals, seed=args.seed + 10000 + i)
        m = obj.evaluate(sched)

        random_pool_rows.append(
            {
                "i": int(i),
                "feasible": int(m["feasible"] > 0.5),
                "J_norm": float(m["J_norm"]),
                "J_raw": float(m.get("J_raw", float("nan"))),
                "mean_surplus": float(m["mean_surplus"]),
                "var_surplus": float(m["var_surplus"]),
                "cohesion_pen": float(m["cohesion_pen"]),
                "min_pressure": float(m["min_pressure"]),
            }
        )

        if m["feasible"] > 0.5:
            random_pool.append(m)

    if len(random_pool) < max(10, args.random_pool // 5):
        print("[WARN] 随机可行解数量偏少：约束可能过紧或参数不合理；近最优统计置信度会下降。")

    report = near_opt_report(
        best=ls_m,
        baselines={"branch_first": bf_m, "random_one": rnd1_m},
        random_pool=random_pool,
        local_best=ls_m,
    )

    # 保存 JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"schedule": ls_sched, "metrics": ls_m}, f, ensure_ascii=False, indent=2)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[saved] schedule -> {args.out_json}")
    print(f"[saved] report   -> {args.report_json}")

    # 邻域检验 CSV
    neighborhood = neighborhood_test_dnn(
        ev,
        obj,
        ref_schedule=ls_sched,
        trials=int(args.neighborhood_trials),
        seed=int(args.seed) + 2002,
    )

    # 写出 NSGA-like CSV
    write_dnn_report_files(
        outdir=args.outdir,
        args=args,
        pack=pack,
        bf_m=bf_m,
        rnd1_m=rnd1_m,
        policy_best_m=best_m,
        local_best_m=ls_m,
        policy_samples_rows=policy_samples_rows,
        random_pool_rows=random_pool_rows,
        neighborhood=neighborhood,
    )

    print(f"[saved] csv reports -> {args.outdir}/")


if __name__ == "__main__":
    main()
