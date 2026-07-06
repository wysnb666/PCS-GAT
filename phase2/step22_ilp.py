from __future__ import annotations

import networkx as nx

from MPC4plus.phase2.step22_instance import Step22Instance


Edge = tuple[int, int]


def solve_step22_instance_ilp(instance: Step22Instance) -> set[Edge]:
    """
    Solve Step 2.2 as an ILP/MIP on G1.

    Decision variables
    ------------------
    x_e in {0,1} for each edge e in E(G1)
        whether e belongs to the selected path-cycle cover C

    y_i in {0,1} for each bad component K_i
        whether K_i is saturated by C

    Constraints
    -----------
    1. Degree constraints for path-cycle cover:
         sum_{e incident to v} x_e <= 2   for all v in V(G1)

    2. Saturation linkage:
         y_i <= sum_{e incident to K_i} x_e   for all bad components K_i

       Since the objective maximizes sum y_i, this is enough:
       if at least one incident edge is selected, the optimum can set y_i = 1;
       otherwise y_i must be 0.

    Objective
    ---------
    maximize sum_i y_i

    Returns
    -------
    C as a set of edges in G1.
    """
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "ILP backend requires scipy.optimize.milp and numpy to be available."
        ) from exc

    edges = instance.edges
    bad_component_nodes = instance.bad_component_nodes
    incident_idxs = instance.incident_edge_indices_by_bad_component

    n_x = len(edges)
    n_y = len(bad_component_nodes)
    n_var = n_x + n_y

    if n_var == 0:
        return set()

    # scipy.optimize.milp minimizes c^T x, so maximize sum(y_i) by minimizing -sum(y_i)
    c = np.zeros(n_var, dtype=float)
    c[n_x:] = -1.0

    integrality = np.ones(n_var, dtype=int)
    bounds = Bounds(lb=np.zeros(n_var), ub=np.ones(n_var))

    constraints: list[LinearConstraint] = []

    # Degree constraints: for each vertex, selected incident edges <= 2
    for v in instance.G1.nodes():
        row = np.zeros(n_var, dtype=float)
        for idx, (u, w) in enumerate(edges):
            if u == v or w == v:
                row[idx] = 1.0
        constraints.append(LinearConstraint(row, lb=-np.inf, ub=2.0))

    # Saturation linkage constraints: y_i <= sum incident x_e
    for i, edge_idxs in enumerate(incident_idxs):
        row = np.zeros(n_var, dtype=float)
        for idx in edge_idxs:
            row[idx] = 1.0
        row[n_x + i] = -1.0
        constraints.append(LinearConstraint(row, lb=0.0, ub=np.inf))

    result = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
    )

    if result.x is None or result.status not in {0, 1}:
        raise RuntimeError(f"ILP/MIP Step 2.2 solver failed with status {result.status}")

    x = np.rint(result.x[:n_x]).astype(int)

    C: set[Edge] = set()
    for idx, take in enumerate(x):
        if take == 1:
            C.add(edges[idx])

    return C