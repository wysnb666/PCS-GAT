from __future__ import annotations

import networkx as nx

from MPC4plus.phase1.modify_H_M import run_phase1
from MPC4plus.core.components import get_component_kind_map, validate_phase1_output


def build_demo_graph_case_c1() -> nx.Graph:
    """
    A small graph where:
      M can be {(1,2), (3,4)}
      outside vertex 5 connects to 1 and 3
    Then Step 1.1 with C1 merges them into a 5-path.
    """
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([
        (1, 2),
        (3, 4),
        (1, 5),
        (3, 5),
    ])
    return G


def build_demo_graph_step12_triangle() -> nx.Graph:
    """
    A graph where after Step 1.1 no augmenting triple exists,
    and Step 1.2 turns an edge component into a triangle:
    outside vertex 3 is adjacent to both endpoints 1 and 2.
    """
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    G.add_edges_from([
        (1, 2),
        (1, 3),
        (2, 3),
    ])
    return G


def build_demo_graph_step12_star() -> nx.Graph:
    """
    A graph where after Step 1.1 no augmenting triple exists,
    and Step 1.2 turns an edge component into a star:
    only one endpoint is adjacent to one or more outside vertices.
    """
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4])
    G.add_edges_from([
        (1, 2),
        (1, 3),
        (1, 4),
    ])
    return G


def print_phase1_summary(state) -> None:
    print("=== Phase 1 Summary ===")
    print("Matching M:", sorted(state.M))
    print("H nodes:", sorted(state.H.nodes()))
    print("H edges:", sorted(tuple(sorted(e)) for e in state.H.edges()))
    print("Components:")
    for nodes, kind in get_component_kind_map(state.H):
        print(f"  nodes={nodes}, kind={kind}")

    ok, msg = validate_phase1_output(state.H, state.M)
    print("Validation:", ok, msg)


def main() -> None:
    print("\n--- Demo 1: C1 creates a 5-path ---")
    G1 = build_demo_graph_case_c1()
    state1 = run_phase1(G1)
    print_phase1_summary(state1)

    print("\n--- Demo 2: Step 1.2 creates a triangle ---")
    G2 = build_demo_graph_step12_triangle()
    state2 = run_phase1(G2)
    print_phase1_summary(state2)

    print("\n--- Demo 3: Step 1.2 creates a star ---")
    G3 = build_demo_graph_step12_star()
    state3 = run_phase1(G3)
    print_phase1_summary(state3)


if __name__ == "__main__":
    main()