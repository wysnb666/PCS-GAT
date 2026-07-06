from __future__ import annotations

import networkx as nx

from MPC4plus.core.graph_utils import normalize_edge
from MPC4plus.phase2.bad_components import get_bad_components


def get_unsaturated_bad_components(H: nx.Graph, C: set[tuple[int, int]]) -> list[nx.Graph]:
    """
    Return bad components not saturated by C.
    """
    unsat = []
    for sg in get_bad_components(H):
        nodes = set(sg.nodes())
        saturated = any((u in nodes or v in nodes) for (u, v) in C)
        if not saturated:
            unsat.append(sg.copy())
    return unsat


def build_M_C(M: set[tuple[int, int]], H: nx.Graph, C: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """
    Notation 3:
    M_C is obtained from M by removing all edges that are both:
      - in M
      - in bad components not saturated by C
    """
    M_C = set(M)
    unsat_bad = get_unsaturated_bad_components(H, C)

    for sg in unsat_bad:
        sg_edges = {normalize_edge(u, v) for u, v in sg.edges()}
        for e in list(M_C):
            if e in sg_edges:
                M_C.remove(e)

    return M_C