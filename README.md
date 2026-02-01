# Multi-objective Rotational Irrigation Grouping and Deep Reinforcement Learning Algorithm Comparison

This repository implements the algorithmic prototype described in *Multi-objective allocation of rotational irrigation groups in large-scale tree-structured irrigation pipe networks using a deep multi-objective Q-network*, and provides several baseline algorithms for controlled comparison. The study targets **rotational irrigation grouping** in gravity-fed (self-pressure) drip irrigation pipe networks: terminal outlets (laterals) are partitioned into multiple irrigation groups and operated sequentially within an irrigation cycle. Each group must satisfy **pressure safety constraints** (minimum operating head at every active outlet) while jointly optimizing two objectives: the **mean** and **variance** of redundant (surplus) head, which reflect energy waste and hydraulic non-uniformity, respectively:contentReference[oaicite:2]{index=2}.

## Paper contributions (high level)

- **Problem formulation**: The rotational irrigation grouping task is modeled as a multi-objective combinatorial optimization problem with objectives \(F_1\) (mean redundant head) and \(F_2\) (variance of redundant head). Hard constraints include coverage, group-capacity, and minimum operating head constraints:contentReference[oaicite:3]{index=3}. A fast hydraulic evaluator is used to assess candidate schedules efficiently.
- **Method**: The problem is reformulated as a finite-horizon Markov decision process (MDP). A preference-conditioned multi-objective Q-network is designed with state encoding, action encoding, and preference-weight input. Action masking is used so that every chosen action satisfies the minimum head constraint, and a differential vector reward makes the cumulative reward equivalent to the negative objective values (dense feedback):contentReference[oaicite:4]{index=4}.
- **Evidence**: In the case study, DMOQN achieves broader Pareto-front coverage than NSGA-II, MOEA/D, and SPEA2, and fills Pareto regions not covered by the baselines:contentReference[oaicite:5]{index=5}. After training, the end-to-end inference time for a full irrigation cycle is **0.561 s**, approximately four orders of magnitude faster than online search (3,800–7,500 s) using conventional multi-objective optimizers:contentReference[oaicite:6]{index=6}.

## Repository layout and file roles

This repository includes scripts for the proposed multi-objective reinforcement learning method and comparison baselines, plus a lightweight hydraulic solver.

| File | Role | What it does (core functions) |
| --- | --- | --- |
| `tree_evaluator.py` | **Tree-structured hydraulic evaluator** | Implements a fast, graph-traversal-based hydraulic solver for tree-structured irrigation networks. It builds a directed tree from the node and pipe Excel files, computes discharges and head losses (GB/T 50485–2020 power-law friction model), and provides rapid feasibility checks (e.g., minimum working head) and objective evaluation for any candidate irrigation group sequence. The file typically contains Excel-loading helpers (such as `load_nodes_xlsx` and `load_pipes_xlsx`) and core evaluator methods (such as `TreeHydraulicEvaluator.evaluate_group`). |
| `mo_irrigation_dynamic.py` | **Multi-objective metaheuristics (baselines)** | Implements common multi-objective optimization algorithms (e.g., NSGA-II, SPEA2, MOEA/D, MOPSO) for solving rotational irrigation grouping. It parses command-line arguments, evaluates candidates using `tree_evaluator.py`, and applies evolutionary operators (selection, crossover, mutation, etc.) to approximate a Pareto set. It can be used for offline optimization and as a baseline for online re-optimization under time-varying hydraulic scenarios. |
| `morl_irrigation_dmoqn.py` | **Dynamic-Weight Deep Multi-Objective Q-Network (DMOQN)** | PyTorch implementation of the preference-conditioned, dynamically weighted Deep Multi-Objective Q-Network described in the paper. It typically contains the environment wrapper (state/action/reward definition), the DMOQN network, replay/optimization logic, and evaluation utilities. The implementation uses **action masking** to exclude hydraulically infeasible actions and a **differential vector reward** so the cumulative reward matches the negative objective values. Entry points commonly include `train_dmoqn` (training), `evaluate` (single preference/scenario evaluation), and `pareto_sweep` (weight sweep to generate a Pareto front). |
| `run_batch.py` | **Batch runner** | Batch execution script for evaluation. It usually reads the input Excel files, checks that a trained model exists, iterates over a list of source-head \(H_0\) settings, and repeatedly calls `morl_irrigation_dmoqn.py evaluate`, saving results under `runs/`. |
| `Nodes.xlsx`, `Pipes.xlsx` | **Network topology inputs** | Excel files describing the irrigation network. `Nodes.xlsx` typically stores node IDs and elevations, while `Pipes.xlsx` stores pipe connectivity and attributes (length, diameter, material, etc.). `tree_evaluator.py` reads them to build the network model. |

### Optional: suggested file naming (for clarity)

If you want a more explicit, paper-aligned naming convention (without changing code logic), you can rename files (or add soft links/aliases) as follows:

- `tree_evaluator.py` → `tree_hydraulic_evaluator.py`
- `morl_irrigation_dmoqn.py` → `dmoqn_train_eval.py`
- `mo_irrigation_dynamic.py` → `baselines_moea.py`
- `run_batch.py` → `run_experiments.py`

## Installation & dependencies

1. **Environment**: Python ≥ 3.9 recommended. The RL component depends on PyTorch (recommended 2.0+). Baselines typically depend on standard scientific libraries such as `numpy`, `scipy`, and `pandas`.
2. **Install dependencies** (from repo root):

```bash
pip install -r requirements.txt  # or manually install pandas, numpy, torch, openpyxl, etc.

[README.md](https://github.com/user-attachments/files/24989957/README.md)
