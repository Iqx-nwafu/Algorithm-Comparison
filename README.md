<div align="center">

# DMOQN: Deep Multi-Objective Q-Network for Irrigation Scheduling
# 基于深度多目标Q网络的大型树状管网轮灌调度

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)

[English](#english) | [中文](#chinese)

</div>

---

<a name="english"></a>
## 🇬🇧 English Version

### Overview

This repository contains the official implementation of the **Dynamic-Weight Deep Multi-Objective Q-Network (DMOQN)** for rotational irrigation scheduling in large-scale tree-structured pipe networks.

The code supports the research paper:
> **Multi-objective allocation of rotational irrigation groups in large-scale tree-structured irrigation pipe networks using a deep multi-objective Q-network**
> *Qianxi Li, Wene Wang, Chenchen Lou*

This project addresses the **Rotational Irrigation Scheduling Problem**, optimizing valve grouping sequences in gravity-fed irrigation networks to minimize two conflicting objectives:
1.  **Variance of Redundant Head ($F_2$):** Ensuring hydraulic uniformity.
2.  **Mean Redundant Head ($F_1$):** Reducing excess energy and improving safety margins.

The proposed **DMOQN** method uses a preference-conditioned RL agent to generate Pareto-optimal solutions in real-time (approx. 0.561s per schedule), significantly outperforming traditional evolutionary algorithms in computational efficiency.

### Key Features

* **Preference-Conditioned RL:** The DMOQN agent takes preference weights $\vec{w}$ as input, allowing a single model to cover the entire Pareto front without retraining.
* **Fast Hydraulic Evaluator:** A custom, graph-traversal-based hydraulic solver (`TreeHydraulicEvaluator`) leverages the acyclic property of tree networks to achieve $O(N)$ complexity, enabling large-scale RL training.
* **Action Masking:** Hard constraints (Minimum Operating Head $H_{min}$) are enforced directly in the action space, ensuring all generated schedules are physically feasible.
* **Dynamic Robustness:** The model is trained to adapt to varying source pressure scenarios ($H_0$), ensuring stability under environmental uncertainty.
* **Comprehensive Baselines:** Includes optimized implementations of **NSGA-II**, **SPEA2**, **MOEA/D**, and **MOPSO** for rigorous benchmarking.

### Repository Structure

#### Core Implementation
| File | Description |
| :--- | :--- |
| **`morl_irrigation_dmoqn.py`** | **[Core Algorithm]** Implements the DMOQN agent, the RL environment (`IrrigationSchedulingEnv`), the preference sampling mechanism, and the training/evaluation loops. |
| **`tree_evaluator.py`** | **[Simulation Engine]** A lightweight, high-performance hydraulic simulator. It parses Excel topologies, builds a directed tree graph, and calculates head losses using the **GB/T 50485-2020** standard. |

#### Baselines & Utilities
| File | Description |
| :--- | :--- |
| **`mo_irrigation_dynamic.py`** | **[Baselines]** Implementation of meta-heuristic algorithms (NSGA-II, SPEA2, MOEA/D, MOPSO). Designed to share the same hydraulic evaluator and objective functions as DMOQN for fair comparison. |
| **`run_batch.py`** | **[Utility]** A batch execution script for running parameter sweeps and sensitivity analyses across different source head ($H_0$) scenarios. |

#### Data
| File | Description |
| :--- | :--- |
| **`Nodes.xlsx`** | Defines network topology, including node IDs and elevation data ($Z$). |
| **`Pipes.xlsx`** | Defines hydraulic properties, including pipe connectivity, length, diameter, and material coefficients. |

### Requirements

* Python 3.9+
* PyTorch 2.0+ (CUDA recommended for training)
* NumPy, Pandas
* OpenPyXL (for Excel I/O)

### Usage

#### 1. Train DMOQN (Proposed Method)
Train the agent to learn the optimal policy across dynamic hydraulic scenarios.

```bash
python morl_irrigation_dmoqn.py train_dmoqn \
    --nodes Nodes.xlsx \
    --pipes Pipes.xlsx \
    --root J0 \
    --H0 25.0 \
    --Hmin 11.59 \
    --q_lateral 0.012 \
    --out runs/dmoqn_experiment \
    --total_steps 400000 \
    --cuda 1
