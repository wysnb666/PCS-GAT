from __future__ import annotations

import networkx as nx

from MPC4plus.phase5.phase5_runner import solve_mpc4plus
from MPC4plus.phase5.exact_solver import graph_to_path_nodes, is_feasible_solution_graph
from MPC4plus.phase5.decomposition import component_bucket_name, summarize_phase5_buckets
from MPC4plus.solver.phase4_runner import build_demo_graph_phase4


def _sorted_edges(edges):
    return sorted(tuple(sorted(e)) for e in edges)


def print_phase5_symbol_explanation() -> None:
    print("=== Phase 5 Symbols ===")
    print("R        : all 2-anchors and responsible 1-anchors in H + C")
    print("Ki       : components K with |V(K) ∩ R| = i")
    print("K1c/K2c  : critical components inside K1 / K2")
    print("Rc       : all 2-anchors in critical components")
    print("Uc       : vertices in critical satellite-elements whose rescue-anchor is in Rc")
    print("Gc_nodes : V(G) \\ (Rc ∪ Uc)")
    print("lhs      : sum_{i=1}^5 i |Ki|")
    print("rhs      : (5/7) * r * (|K1c| + 2|K2c|)")
    print()


def print_phase5_summary(result) -> None:
    print("=== Phase 5 Summary ===")
    print("recursion_depth:", result.recursion_depth)
    print("decision:", result.phase5_decision.reason)
    print("lhs:", result.phase5_decision.lhs_value)
    print("rhs:", result.phase5_decision.rhs_value)
    print("has_critical:", result.phase5_decision.has_critical)

    b = result.phase5_buckets
    counts = summarize_phase5_buckets(b)
    print("\nBucket counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    print("\nKey sets:")
    print("  R:", sorted(b.R))
    print("  Rc:", sorted(b.Rc))
    print("  Uc:", sorted(b.Uc))
    print("  Gc_nodes:", sorted(b.Gc_nodes))

    print("\nResponsible components:")
    if not b.responsible_components:
        print("  (none)")
    else:
        for idx, meta in enumerate(b.responsible_components):
            print(f"  responsible_component {idx}:")
            print("    nodes:", sorted(meta.nodes))
            print("    responsible_case:", meta.responsible_case)
            print("    responsible_reason:", meta.responsible_reason)
            print("    responsible_signature:", meta.responsible_signature)

    print("\nComponent roles:")
    if result.phase4_state is not None:
        for idx, meta in enumerate(result.phase4_state.metas):
            role = component_bucket_name(meta, b)
            print(f"  Component {idx}:")
            print("    bucket:", role)
            print("    nodes:", sorted(meta.nodes))
            print("    center_kind:", meta.center_kind)
            print("    anchors:", meta.anchors)
            print("    j_anchor_map:", meta.j_anchor_map)
            print("    critical_2_anchors:", sorted(meta.critical_2_anchors))
            print("    responsible_1_anchors:", sorted(meta.responsible_1_anchors))
            print("    s_value:", meta.s_value)
            print("    opt_value:", meta.opt_value)
            print("    critical_ratio:", meta.critical_ratio)
            print("    is_critical:", meta.is_critical)
            print("    critical_case:", meta.critical_case)
            print("    critical_reason:", meta.critical_reason)
            print("    critical_signature:", meta.critical_signature)
            print("    is_responsible_component:", meta.is_responsible_component)
            print("    responsible_case:", meta.responsible_case)
            print("    responsible_reason:", meta.responsible_reason)
            print("    responsible_signature:", meta.responsible_signature)
            for j, sat in enumerate(meta.satellites):
                print(f"      satellite {j}:")
                print("        nodes:", sorted(sat.nodes))
                print("        kind:", sat.kind)
                print("        rescue_edges:", _sorted_edges(sat.rescue_edges))
                print("        supporting_anchors:", sorted(sat.supporting_anchors))
                print("        rescue_anchor:", sat.rescue_anchor)
                print("        is_critical_satellite:", sat.is_critical_satellite)

    sol = result.solution_graph
    print("\nFinal solution:")
    print("  covered vertices:", sol.number_of_nodes())
    print("  edges:", _sorted_edges(sol.edges()))
    print("  feasible:", is_feasible_solution_graph(sol))

    print("\nSolution paths:")
    if sol.number_of_nodes() == 0:
        print("  (empty)")
    else:
        for idx, nodes in enumerate(nx.connected_components(sol)):
            comp = sol.subgraph(nodes).copy()
            print(f"  path {idx}: {graph_to_path_nodes(comp)}")


def main() -> None:
    print_phase5_symbol_explanation()
    G = build_demo_graph_phase4()
    result = solve_mpc4plus(G)
    print_phase5_summary(result)


if __name__ == "__main__":
    main()