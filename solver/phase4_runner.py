from __future__ import annotations

import networkx as nx

from MPC4plus.phase1.modify_H_M import run_phase1
from MPC4plus.phase2.auxiliary_graph import build_auxiliary_graph_G1
from MPC4plus.phase2.matching_projection import build_M_C
from MPC4plus.phase2.path_cycle_cover import compute_max_weight_path_cycle_cover
from MPC4plus.phase4.operation_runner import run_phase4_operations


def build_demo_graph_phase4() -> nx.Graph:
    """
    A small demo graph for Phase 4.

    It is not intended to realize every strict paper structure on its own,
    but to provide a runnable end-to-end pipeline through Phase 1-4.
    """
    G = nx.Graph()
    G.add_nodes_from(range(1, 13))
    G.add_edges_from([
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (9, 10),
        (11, 12),

        (2, 5),
        (4, 7),
        (6, 9),
        (8, 11),

        (3, 10),
        (5, 12),
        (1, 8),
    ])
    return G


def _sorted_edges(edges):
    return sorted(tuple(sorted(e)) for e in edges)


def print_phase4_summary(state) -> None:
    print("=== Phase 4 Summary ===")
    print("iterations:", state.iteration)
    print("final C:", sorted(state.C))
    print("history length:", len(state.history))

    for i, meta in enumerate(state.metas):
        print(f"\nComponent {i}")
        print("  nodes:", sorted(meta.nodes))
        print("  center_nodes:", sorted(meta.center_nodes) if meta.center_nodes else None)
        print("  center_kind:", meta.center_kind)
        print("  anchors:", meta.anchors)
        print("  j_anchor_map:", meta.j_anchor_map)
        print("  critical_2_anchors:", sorted(meta.critical_2_anchors))
        print("  responsible_1_anchors:", sorted(meta.responsible_1_anchors))
        print("  s_value:", meta.s_value)
        print("  opt_value:", meta.opt_value)
        print("  critical_ratio:", meta.critical_ratio)
        print("  is_critical:", meta.is_critical)
        print("  critical_case:", meta.critical_case)
        print("  critical_reason:", meta.critical_reason)
        print("  critical_signature:", meta.critical_signature)

        for j, sat in enumerate(meta.satellites):
            print(f"    satellite {j}:")
            print("      nodes:", sorted(sat.nodes))
            print("      kind:", sat.kind)
            print("      rescue_edges:", _sorted_edges(sat.rescue_edges))
            print("      supporting_anchors:", sorted(sat.supporting_anchors))
            print("      rescue_anchor:", sat.rescue_anchor)
            print("      is_critical_satellite:", sat.is_critical_satellite)

    if state.history:
        print("\nChosen operations:")
        for t, op in enumerate(state.history, start=1):
            print(
                f"  step {t}: {op.op_type}, "
                f"add={op.added_edge}, remove={list(op.removed_edges)}, note={op.note}"
            )


def main() -> None:
    G = build_demo_graph_phase4()

    phase1 = run_phase1(G)
    G1 = build_auxiliary_graph_G1(G, phase1.H)
    C = compute_max_weight_path_cycle_cover(G1, phase1.H)
    M_C = build_M_C(phase1.M, phase1.H, C)

    state = run_phase4_operations(
        G=G,
        H=phase1.H,
        M_C=M_C,
        C=C,
    )

    print_phase4_summary(state)


if __name__ == "__main__":
    main()