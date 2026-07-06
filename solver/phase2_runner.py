from __future__ import annotations

import networkx as nx

from MPC4plus.phase1.modify_H_M import run_phase1
from MPC4plus.core.components import get_component_kind_map, validate_phase1_output
from MPC4plus.phase2.bad_components import get_bad_components
from MPC4plus.phase2.auxiliary_graph import build_auxiliary_graph_G1
from MPC4plus.phase2.path_cycle_cover import (
    compute_max_weight_path_cycle_cover,
    saturated_bad_component_count,
)
from MPC4plus.phase2.matching_projection import build_M_C


def build_demo_graph_phase2() -> nx.Graph:
    """
    A small graph designed so that after phase 1,
    H contains bad components and G1 has rescue edges.
    """
    G = nx.Graph()
    G.add_nodes_from(range(1, 9))
    G.add_edges_from([
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (2, 5),
        (4, 7),
        (6, 1),
        (3, 8),
    ])
    return G


def print_phase2_summary(G: nx.Graph) -> None:
    phase1 = run_phase1(G)

    print("=== After Phase 1 ===")
    print("M:", sorted(phase1.M))
    print("H edges:", sorted(tuple(sorted(e)) for e in phase1.H.edges()))
    print("H components:", get_component_kind_map(phase1.H))

    ok, msg = validate_phase1_output(phase1.H, phase1.M)
    print("Phase 1 validation:", ok, msg)

    bad = get_bad_components(phase1.H)
    print("\nBad components:")
    for i, sg in enumerate(bad):
        print(f"  B{i}: nodes={sorted(sg.nodes())}, edges={sorted(tuple(sorted(e)) for e in sg.edges())}")

    G1 = build_auxiliary_graph_G1(G, phase1.H)
    print("\nG1 edges:", sorted(tuple(sorted(e)) for e in G1.edges()))

    C_bf = compute_max_weight_path_cycle_cover(G1, phase1.H, backend="bruteforce")
    print("\n[Step 2.2 backend = bruteforce]")
    print("C edges:", sorted(tuple(sorted(e)) for e in C_bf))
    print("weight(C):", saturated_bad_component_count(phase1.H, C_bf))

    C_ilp = compute_max_weight_path_cycle_cover(G1, phase1.H, backend="ilp")
    print("\n[Step 2.2 backend = ilp]")
    print("C edges:", sorted(tuple(sorted(e)) for e in C_ilp))
    print("weight(C):", saturated_bad_component_count(phase1.H, C_ilp))

    M_C = build_M_C(phase1.M, phase1.H, C_ilp)
    print("M_C (from ILP backend):", sorted(M_C))


def main() -> None:
    G = build_demo_graph_phase2()
    print_phase2_summary(G)


if __name__ == "__main__":
    main()