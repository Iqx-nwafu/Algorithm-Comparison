"""Tree steady-state hydraulic evaluator for irrigation pipe networks.

Designed for your wheel-irrigation grouping problem:
- Fixed topology/diameters.
- Each group opens 4 laterals (斗管), each q_lateral = 0.012 m^3/s.
- Source provides hydraulic head H0 = 25 m at reservoir node (e.g., J0).
- Constraint (hard): pressure head at each opened lateral's node >= Hmin = 11.59 m.

Key idea (tree network):
- Pipe flow equals sum of demands in its downstream subtree.
- Hydraulic head drop along a pipe equals friction loss + minor loss (independent of elevation).
- Pressure head at node i = hydraulic_head(i) - elevation_z(i).

This file:
1) Reads Nodes.xlsx and Pipes.xlsx.
2) Builds a rooted tree (parent/children, postorder and preorder traversals).
3) Evaluates one group (a set of opened laterals) very fast.

IMPORTANT
- You said “按照规范计算水头损失”. GB/T 50485-2020 provides a power-law friction form
  with coefficients (f, m, b) depending on pipe material and flow regime.
  Those coefficients are NOT hardcoded here; fill them in `GBT_COEFS` from your standard table.
- Minor losses (valves/tees) are optional; set zeta per pipe if you have them.

Usage example is at the bottom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import pandas as pd


# =========================
# 1) GB/T 50485 friction model (power-law)
# =========================

@dataclass(frozen=True)
class GbtCoef:
    """Coefficients for GB/T 50485-2020 power-law friction loss.

    The standard expresses friction head loss as a power function of flow, diameter and length.
    You MUST fill in correct coefficients from the standard's table.

    Unit convention in the standard (commonly used):
      - Q in L/h
      - D in mm
      - L in m
    This implementation converts from SI (m^3/s, m) into those units.
    """

    f: float
    m: float
    b: float


# Fill these from GB/T 50485-2020 tables.
# Provide at least for the materials you use in Pipes.xlsx (UPVC, PE).
# The keys here are (material_upper, regime), where regime in {"turbulent", "laminar"}.
GBT_COEFS: Dict[Tuple[str, str], GbtCoef] = {
    # Values provided by you (GB/T 50485-2020).
    # In your operating range (q_lateral=0.012 m^3/s and D>=0.16 m), flows are effectively turbulent (Re >> 2320).
    # We still populate both regimes to avoid KeyError; if you later operate in true laminar range, replace the laminar entries
    # with the correct coefficients from the standard.
    ("UPVC", "turbulent"): GbtCoef(f=0.464, m=1.77, b=4.77),
    ("UPVC", "laminar"):   GbtCoef(f=0.464, m=1.77, b=4.77),
    ("PE", "turbulent"):   GbtCoef(f=0.505, m=1.75, b=4.75),
    ("PE", "laminar"):     GbtCoef(f=0.505, m=1.75, b=4.75),
}


def _si_to_gbt_units(Q_m3s: float, D_m: float) -> Tuple[float, float]:
    """Convert SI (m^3/s, m) to GB/T convention (L/h, mm)."""
    Q_Lh = Q_m3s * 3.6e6
    D_mm = D_m * 1000.0
    return Q_Lh, D_mm


def friction_loss_gbt(Q_m3s: float, L_m: float, D_m: float, material: str, nu_m2s: float = 1.004e-6) -> float:
    """Friction head loss (m) for one pipe segment using GB/T 50485 power-law.

    Parameters
    ----------
    Q_m3s : float
        Flow in the pipe (m^3/s). Must be >= 0.
    L_m : float
        Pipe length (m).
    D_m : float
        Pipe diameter (m).
    material : str
        Material label (e.g., 'UPVC', 'PE').
    nu_m2s : float
        Kinematic viscosity (m^2/s), used only to decide laminar/turbulent regime.
        Default ~ water at 20°C.

    Returns
    -------
    hf : float
        Friction head loss (m).
    """

    if Q_m3s <= 0.0 or L_m <= 0.0:
        return 0.0
    if D_m <= 0.0:
        raise ValueError("Diameter must be positive")

    # Decide flow regime by Reynolds number (Re = vD/nu)
    A = math.pi * (D_m ** 2) / 4.0
    v = Q_m3s / A
    Re = v * D_m / nu_m2s
    regime = "laminar" if Re <= 2320.0 else "turbulent"

    key = (material.strip().upper(), regime)
    if key not in GBT_COEFS:
        raise KeyError(
            f"Missing GB/T coefficients for {key}. Fill GBT_COEFS with (f,m,b) from the standard table."
        )

    coef = GBT_COEFS[key]
    Q_Lh, D_mm = _si_to_gbt_units(Q_m3s, D_m)

    # Standard power-law form: hf = f * (Q^m) * L / (D^b)
    # (This is a generic template consistent with the standard description; ensure exact form/units match your table.)
    hf = coef.f * (Q_Lh ** coef.m) * L_m / (D_mm ** coef.b)
    return hf


def minor_loss_zeta(Q_m3s: float, D_m: float, zeta: float, g: float = 9.80665) -> float:
    """Minor head loss h = zeta * v^2 / (2g)."""
    if Q_m3s <= 0.0 or zeta <= 0.0:
        return 0.0
    A = math.pi * (D_m ** 2) / 4.0
    v = Q_m3s / A
    return zeta * (v ** 2) / (2.0 * g)


# =========================
# 2) Tree network model
# =========================

@dataclass
class Node:
    node_id: str
    z: float


@dataclass
class Edge:
    edge_id: str
    u: str
    v: str
    L: float
    D: float
    material: str
    zeta: float = 0.0  # optional minor-loss coefficient


@dataclass
class GroupResult:
    ok: bool
    min_margin: float
    min_pressure_head: float
    pressures: Dict[str, float]  # pressure head at nodes with demand
    hydraulic_heads: Dict[str, float]  # hydraulic head for those nodes


class TreeHydraulicEvaluator:
    """Fast steady-state evaluator for a rooted tree network."""

    def __init__(
        self,
        nodes: Dict[str, Node],
        edges: List[Edge],
        root: str,
        H0: float,
        Hmin: float,
        nu_m2s: float = 1.004e-6,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.root = root
        self.H0 = float(H0)
        self.Hmin = float(Hmin)
        self.nu = float(nu_m2s)

        # Build undirected adjacency then root it.
        self._adj: Dict[str, List[Tuple[str, Edge]]] = {nid: [] for nid in nodes}
        for e in edges:
            if e.u not in nodes or e.v not in nodes:
                raise KeyError(f"Edge {e.edge_id} references unknown nodes {e.u}, {e.v}")
            self._adj[e.u].append((e.v, e))
            self._adj[e.v].append((e.u, e))

        self.parent: Dict[str, Optional[str]] = {root: None}
        self.parent_edge: Dict[str, Optional[Edge]] = {root: None}
        self.children: Dict[str, List[str]] = {nid: [] for nid in nodes}

        self._build_rooted_tree()
        self._postorder: List[str] = self._compute_postorder()
        self._preorder: List[str] = self._compute_preorder()

    def _build_rooted_tree(self) -> None:
        """Root the graph at self.root using DFS; verifies connectivity and no back-edge cycles."""
        stack = [self.root]
        visited = set([self.root])

        while stack:
            u = stack.pop()
            for v, e in self._adj[u]:
                if v not in visited:
                    visited.add(v)
                    self.parent[v] = u
                    self.parent_edge[v] = e
                    self.children[u].append(v)
                    stack.append(v)

        if len(visited) != len(self.nodes):
            missing = set(self.nodes.keys()) - visited
            raise ValueError(f"Network is not connected to root {self.root}. Missing: {sorted(missing)[:10]}...")

        # Simple tree check: E must be N-1 for a tree.
        if len(self.edges) != len(self.nodes) - 1:
            raise ValueError(
                f"Expected a tree (E=N-1). Got N={len(self.nodes)}, E={len(self.edges)}. "
                "If you have loops, this evaluator needs extension."
            )

    def _compute_postorder(self) -> List[str]:
        order: List[str] = []

        def dfs(u: str) -> None:
            for c in self.children[u]:
                dfs(c)
            order.append(u)

        dfs(self.root)
        return order

    def _compute_preorder(self) -> List[str]:
        order: List[str] = []
        stack = [self.root]
        while stack:
            u = stack.pop()
            order.append(u)
            # reverse for stable ordering
            for c in reversed(self.children[u]):
                stack.append(c)
        return order

    def evaluate_group(
        self,
        open_laterals: Iterable[str],
        lateral_to_node: Dict[str, str],
        q_lateral: float = 0.012,
    ) -> GroupResult:
        """Evaluate one irrigation group (typically 4 laterals open).

        Parameters
        ----------
        open_laterals:
            Iterable of lateral IDs (e.g., 'J11_L', 'J11_R', ...).
        lateral_to_node:
            Mapping from lateral ID to the network node ID where demand applies.
        q_lateral:
            Flow per lateral (m^3/s).

        Returns
        -------
        GroupResult
            ok=True if all opened nodes satisfy pressure head >= Hmin.
        """

        # Node demands (m^3/s)
        demand: Dict[str, float] = {nid: 0.0 for nid in self.nodes}
        opened_nodes: List[str] = []
        for lat in open_laterals:
            if lat not in lateral_to_node:
                raise KeyError(f"Unknown lateral id: {lat}")
            nid = lateral_to_node[lat]
            demand[nid] += q_lateral
            opened_nodes.append(nid)

        # 1) subtree flow accumulation (postorder)
        subflow: Dict[str, float] = {nid: 0.0 for nid in self.nodes}
        for u in self._postorder:
            q = demand[u]
            for c in self.children[u]:
                q += subflow[c]
            subflow[u] = q

        # 2) hydraulic head propagation (preorder)
        Hhyd: Dict[str, float] = {nid: float("nan") for nid in self.nodes}
        Hhyd[self.root] = self.H0

        for u in self._preorder:
            Hu = Hhyd[u]
            for c in self.children[u]:
                e = self.parent_edge[c]
                assert e is not None
                Q = subflow[c]  # flow through edge u->c

                hf = friction_loss_gbt(Q, e.L, e.D, e.material, nu_m2s=self.nu)
                hj = minor_loss_zeta(Q, e.D, e.zeta)

                Hhyd[c] = Hu - (hf + hj)

        # 3) compute pressure heads for opened nodes
        pressures: Dict[str, float] = {}
        hyd_heads: Dict[str, float] = {}
        min_pressure = float("inf")
        min_margin = float("inf")

        for nid in opened_nodes:
            Hi = Hhyd[nid]
            pi = Hi - self.nodes[nid].z  # pressure head
            pressures[nid] = pi
            hyd_heads[nid] = Hi
            min_pressure = min(min_pressure, pi)
            min_margin = min(min_margin, pi - self.Hmin)

        ok = min_margin >= 0.0
        return GroupResult(
            ok=ok,
            min_margin=min_margin,
            min_pressure_head=min_pressure,
            pressures=pressures,
            hydraulic_heads=hyd_heads,
        )


# =========================
# 3) Helpers for your Excel inputs and lateral IDs
# =========================


def load_nodes_xlsx(path: str) -> Dict[str, Node]:
    df = pd.read_excel(path)
    # Expected columns: 'Nodel ID', 'Z'
    nodes: Dict[str, Node] = {}
    for _, r in df.iterrows():
        nid = str(r["Nodel ID"]).strip()
        z = float(r["Z"])
        nodes[nid] = Node(node_id=nid, z=z)
    return nodes


def load_pipes_xlsx(path: str, default_zeta: float = 0.0) -> List[Edge]:
    df = pd.read_excel(path)
    # Expected columns: 'Pipe ID','FromNode','ToNode','Length_m','Diameter_m','Material'
    edges: List[Edge] = []
    for _, r in df.iterrows():
        edges.append(
            Edge(
                edge_id=str(r["Pipe ID"]).strip(),
                u=str(r["FromNode"]).strip(),
                v=str(r["ToNode"]).strip(),
                L=float(r["Length_m"]),
                D=float(r["Diameter_m"]),
                material=str(r["Material"]).strip().upper(),
                zeta=float(default_zeta),
            )
        )
    return edges


def build_lateral_ids_for_field_nodes(field_node_ids: Iterable[str]) -> Tuple[List[str], Dict[str, str]]:
    """Create 2 laterals per field node: <node>_L and <node>_R.

    Returns
    -------
    lateral_ids: list[str]
    lateral_to_node: dict[str,str]
    """
    lateral_ids: List[str] = []
    lateral_to_node: Dict[str, str] = {}
    for nid in field_node_ids:
        for side in ("L", "R"):
            lid = f"{nid}_{side}"
            lateral_ids.append(lid)
            lateral_to_node[lid] = nid
    return lateral_ids, lateral_to_node


def is_field_node_id(nid: str) -> bool:
    """Heuristic for your data: field nodes are J11..J106 (>=11), mainline includes J0..J10."""
    nid = nid.strip().upper()
    if not nid.startswith("J"):
        return False
    try:
        num = int(nid[1:])
    except ValueError:
        return False
    return num >= 11


# =========================
# 4) Minimal example
# =========================

if __name__ == "__main__":
    # Paths for your uploaded files (edit as needed)
    nodes_path = "Nodes.xlsx"
    pipes_path = "Pipes.xlsx"

    nodes = load_nodes_xlsx(nodes_path)
    edges = load_pipes_xlsx(pipes_path)

    root = "J0"
    H0 = 25.0
    Hmin = 11.59

    # Build lateral list for all field nodes
    field_nodes = [nid for nid in nodes.keys() if is_field_node_id(nid)]
    lateral_ids, lateral_to_node = build_lateral_ids_for_field_nodes(field_nodes)

    print(f"Field nodes: {len(field_nodes)} | Laterals: {len(lateral_ids)}")

    evaluator = TreeHydraulicEvaluator(nodes=nodes, edges=edges, root=root, H0=H0, Hmin=Hmin)

    # Example: pick any 4 laterals (replace with your RL actions)
    demo_group = lateral_ids[:4]

    # NOTE: You MUST fill GBT_COEFS before running this.
    res = evaluator.evaluate_group(demo_group, lateral_to_node=lateral_to_node, q_lateral=0.012)
    print("ok=", res.ok, "min_margin=", res.min_margin, "min_pressure_head=", res.min_pressure_head)
    for nid, p in res.pressures.items():
        print(nid, "pressure_head=", p)
