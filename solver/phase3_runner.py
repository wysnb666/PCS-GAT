from __future__ import annotations

import networkx as nx

from MPC4plus.phase1.modify_H_M import run_phase1
from MPC4plus.phase2.auxiliary_graph import build_auxiliary_graph_G1
from MPC4plus.phase2.path_cycle_cover import compute_max_weight_path_cycle_cover
from MPC4plus.phase2.matching_projection import build_M_C
from MPC4plus.phase3.h_plus_c import build_H_plus_C
from MPC4plus.phase3.composite_components import build_all_component_meta
from MPC4plus.phase3.anchors import enrich_meta_with_anchor_info
from MPC4plus.phase3.critical import mark_critical_components


def build_demo_graph_phase3() -> nx.Graph:
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


def main() -> None:
    G = build_demo_graph_phase3()

    phase1 = run_phase1(G)
    G1 = build_auxiliary_graph_G1(G, phase1.H)
    C = compute_max_weight_path_cycle_cover(G1, phase1.H)
    M_C = build_M_C(phase1.M, phase1.H, C)

    HC = build_H_plus_C(phase1.H, C)

    metas = build_all_component_meta(HC, phase1.H)
    metas = [enrich_meta_with_anchor_info(meta, phase1.H, HC) for meta in metas]
    metas = mark_critical_components(metas, HC, M_C)

    print("=== Phase 3 Summary ===")
    print("C:", sorted(tuple(sorted(e)) for e in C))
    print("M_C:", sorted(M_C))

    for i, meta in enumerate(metas):
        print(f"\nComponent {i}")
        print("  nodes:", sorted(meta.nodes))
        print("  is_composite:", meta.is_composite)
        print("  center_nodes:", sorted(meta.center_nodes) if meta.center_nodes else None)
        print("  center_kind:", meta.center_kind)
        print("  anchors:", meta.anchors)
        print("  j_anchor_map:", meta.j_anchor_map)
        print("  s_value:", meta.s_value)
        print("  is_critical:", meta.is_critical)

        for j, sat in enumerate(meta.satellites):
            print(f"    satellite {j}:")
            print("      nodes:", sorted(sat.nodes))
            print("      kind:", sat.kind)
            print("      rescue_edges:", sorted(tuple(sorted(e)) for e in sat.rescue_edges))
            print("      supporting_anchors:", sorted(sat.supporting_anchors))


if __name__ == "__main__":
    main()