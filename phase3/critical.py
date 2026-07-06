from __future__ import annotations

import itertools
import networkx as nx

from MPC4plus.core.graph_utils import connected_components_with_subgraphs
from MPC4plus.phase3.component_meta import ComponentMeta


Edge = tuple[int, int]


def compute_s_value(K_nodes: set[int], M_C: set[Edge]) -> int:
    """
    Notation 5:
    s(K) = |V(K) ∩ V(M_C)|
    """
    mc_vertices = set()
    for u, v in M_C:
        mc_vertices.add(u)
        mc_vertices.add(v)
    return len(K_nodes & mc_vertices)


def is_valid_solution_subgraph(S: nx.Graph) -> bool:
    """
    Feasible solution interpretation for the current reproduction code:
    every connected component must be a path with at least 4 vertices.
    """
    for comp in connected_components_with_subgraphs(S):
        n = comp.number_of_nodes()
        m = comp.number_of_edges()

        if n < 4:
            return False
        if m != n - 1:
            return False

        degrees = sorted(dict(comp.degree()).values())
        if degrees != [1, 1] + [2] * (n - 2):
            return False
    return True


def brute_force_opt_of_component(K: nx.Graph) -> int:
    """
    Exact opt(K) by brute force on small components.

    This remains the exact-value backstop for the current research-reproduction code.
    """
    nodes = list(K.nodes())
    edges = list(K.edges())

    best = 0
    for r in range(len(edges) + 1):
        for subset in itertools.combinations(edges, r):
            S = nx.Graph()
            S.add_nodes_from(nodes)
            S.add_edges_from(subset)

            nonisolated = [v for v, d in S.degree() if d > 0]
            T = S.subgraph(nonisolated).copy()

            if T.number_of_nodes() == 0:
                continue

            if is_valid_solution_subgraph(T):
                best = max(best, T.number_of_nodes())
    return best


def _anchor_type_counts(meta: ComponentMeta) -> tuple[int, int, int]:
    n0 = 0
    n1 = 0
    n2 = 0
    for a in meta.anchors:
        j = meta.j_anchor_map.get(a, 0)
        if j == 0:
            n0 += 1
        elif j == 1:
            n1 += 1
        elif j == 2:
            n2 += 1
    return n0, n1, n2


def _ordered_five_path_anchors(meta: ComponentMeta, HC: nx.Graph) -> list[int]:
    if meta.center_nodes is None or meta.center_kind != "five_path":
        return []

    sg = HC.subgraph(meta.center_nodes).copy()
    if sg.number_of_nodes() != 5 or sg.number_of_edges() != 4:
        return []

    endpoints = [v for v, d in sg.degree() if d == 1]
    if len(endpoints) != 2:
        return []

    try:
        return nx.shortest_path(sg, endpoints[0], endpoints[1])
    except nx.NetworkXNoPath:
        return []


def _two_anchor_positions_on_five_path(meta: ComponentMeta, HC: nx.Graph) -> tuple[int, ...]:
    ordered = _ordered_five_path_anchors(meta, HC)
    if len(ordered) != 5:
        return tuple()

    pos = {v: i for i, v in enumerate(ordered)}
    out = []
    for a in meta.anchors:
        if meta.j_anchor_map.get(a, 0) == 2 and a in pos:
            out.append(pos[a])
    return tuple(sorted(out))


def _rescue_anchor_multiplicity(meta: ComponentMeta) -> tuple[int, ...]:
    """
    For each anchor, count how many satellites uniquely use it as rescue-anchor.
    This is a compact structure signature helpful for catalog matching.
    """
    counts = {a: 0 for a in meta.anchors}
    for sat in meta.satellites:
        distinct_supporting = sorted(set(sat.supporting_anchors))
        if len(distinct_supporting) == 1 and distinct_supporting[0] in counts:
            counts[distinct_supporting[0]] += 1
    return tuple(sorted(counts[a] for a in sorted(counts)))


def _build_critical_signature(meta: ComponentMeta, HC: nx.Graph) -> dict[str, object]:
    n0, n1, n2 = _anchor_type_counts(meta)
    signature: dict[str, object] = {
        "center_kind": meta.center_kind,
        "num_nodes": len(meta.nodes),
        "num_anchors": len(meta.anchors),
        "num_satellites": len(meta.satellites),
        "satellite_kinds": tuple(sorted(s.kind for s in meta.satellites)),
        "rescue_anchor_multiplicity": _rescue_anchor_multiplicity(meta),
        "n0": n0,
        "n1": n1,
        "n2": n2,
        "s_value": meta.s_value,
    }

    if meta.center_kind == "five_path":
        signature["two_anchor_positions_on_five_path"] = _two_anchor_positions_on_five_path(meta, HC)
    else:
        signature["two_anchor_positions_on_five_path"] = tuple()

    return signature


def _match_catalog_case(signature: dict[str, object]) -> tuple[bool, str, str]:
    """
    Explicit catalog matching for the surviving critical structures.

    This is still a compact structural catalog rather than a full Fig.1 graph-isomorphism
    recognizer, but it is more explicit than the previous pure rule filter.
    """
    center_kind = signature["center_kind"]
    n0 = int(signature["n0"])
    n1 = int(signature["n1"])
    n2 = int(signature["n2"])
    s_value = int(signature["s_value"])
    num_satellites = int(signature["num_satellites"])
    positions = tuple(signature["two_anchor_positions_on_five_path"])

    # Lemma 10 / 11
    if n2 == 0:
        return False, "", "Lemma 10: no 2-anchor implies non-critical"
    if n2 >= 3:
        return False, "", "Lemma 11: at least three 2-anchors implies non-critical"

    # Stars do not survive the critical catalog in our current reconstruction.
    if center_kind == "star":
        return False, "", "critical catalog excludes star centers"

    # One-2-anchor catalog: first row of Fig.1
    if n2 == 1:
        if center_kind == "edge":
            if s_value != 8:
                return False, "", "one-2-anchor edge catalog requires s(K)=8"
            if n1 != 1 or n0 != 0:
                return False, "", "one-2-anchor edge catalog requires anchor pattern (n0,n1,n2)=(0,1,1)"
            if num_satellites != 3:
                return False, "", "one-2-anchor edge catalog requires exactly three satellites"
            return True, "one-2-anchor-edge-s8", "matched explicit one-2-anchor edge catalog"

        if center_kind == "five_path":
            if s_value != 10:
                return False, "", "one-2-anchor 5-path catalog requires s(K)=10"
            if n1 != 2 or n0 != 2:
                return False, "", "one-2-anchor 5-path catalog requires anchor pattern (2,2,1)"
            if num_satellites != 3:
                return False, "", "one-2-anchor 5-path catalog requires exactly three satellites"
            return True, "one-2-anchor-five-path-s10", "matched explicit one-2-anchor 5-path catalog"

        return False, "", "one-2-anchor catalog allows only edge or 5-path centers"

    # Two-2-anchor catalog: remaining rows of Fig.1
    if n2 == 2:
        if center_kind == "edge":
            return False, "", "Lemma 12(1): two 2-anchors with edge center is non-critical"

        if center_kind != "five_path":
            return False, "", "two-2-anchor catalog allows only 5-path centers"

        if s_value not in {14, 16, 18}:
            return False, "", "two-2-anchor 5-path catalog requires s(K) in {14,16,18}"

        if len(positions) != 2:
            return False, "", "could not recover the two 2-anchor positions on the 5-path"

        if s_value in {16, 18}:
            if positions != (1, 3):
                return False, "", "Lemma 12: for s(K)=16 or 18 the two 2-anchors must be at positions (1,3)"
            return True, f"two-2-anchor-five-path-s{s_value}", "matched explicit internal-pair 5-path catalog"

        allowed_pairs_s14 = {
            (0, 2),
            (1, 2),
            (2, 3),
            (2, 4),
            (1, 3),
            (0, 3),
            (1, 4),
        }
        if positions not in allowed_pairs_s14:
            return False, "", "Lemma 12(5): the s(K)=14 pair is not in the allowed catalog"
        return True, "two-2-anchor-five-path-s14", "matched explicit s(K)=14 5-path catalog"

    return False, "", "anchor pattern is outside the critical catalog"


def _confirm_critical_case_with_exact_opt(
    meta: ComponentMeta,
    case: str,
) -> tuple[bool, str]:
    """
    Final exact confirmation after the structural catalog.

    The paper defines criticality by s(K)/opt(K) >= 14/11. We use exact opt(K) here,
    after catalog-level structural filtering.
    """
    if meta.opt_value <= 0:
        return False, "opt(K)=0"

    meta.critical_ratio = meta.s_value / meta.opt_value
    if meta.critical_ratio < (14 / 11):
        return False, f"ratio {meta.s_value}/{meta.opt_value} is below 14/11"

    if case == "one-2-anchor-edge-s8":
        return True, "catalog + exact ratio confirmed one-2-anchor edge case"

    if case == "one-2-anchor-five-path-s10":
        return True, "catalog + exact ratio confirmed one-2-anchor 5-path case"

    if case == "two-2-anchor-five-path-s18":
        if meta.opt_value in {13, 14}:
            return True, "catalog + exact ratio confirmed s(K)=18 case"
        return False, "s(K)=18 catalog case requires opt(K) in {13,14}"

    if case == "two-2-anchor-five-path-s16":
        if meta.opt_value == 12:
            return True, "catalog + exact ratio confirmed s(K)=16 case"
        return False, "s(K)=16 catalog case requires opt(K)=12"

    if case == "two-2-anchor-five-path-s14":
        if meta.opt_value == 11:
            return True, "catalog + exact ratio confirmed s(K)=14 case"
        return False, "s(K)=14 catalog case requires opt(K)=11"

    return False, "unknown catalog case after exact confirmation"


def mark_critical_components(
    metas: list[ComponentMeta],
    HC: nx.Graph,
    M_C: set[Edge],
) -> list[ComponentMeta]:
    """
    Structured reproduction of Definition 8 with an explicit critical catalog.

    Pipeline:
    1. compute s(K);
    2. build a structural signature;
    3. match the signature against the explicit surviving critical catalog;
    4. only then compute exact opt(K);
    5. confirm via exact ratio / case-specific equalities.
    """
    node_to_component_idx = {}
    HC_components = connected_components_with_subgraphs(HC)
    for idx, sg in enumerate(HC_components):
        for v in sg.nodes():
            node_to_component_idx[v] = idx

    comp_graphs = [sg.copy() for sg in HC_components]

    for meta in metas:
        meta.s_value = compute_s_value(meta.nodes, M_C)
        meta.opt_value = 0
        meta.critical_ratio = 0.0
        meta.is_critical = False
        meta.critical_case = None
        meta.critical_reason = ""
        meta.critical_2_anchors = set()

        meta.critical_signature = _build_critical_signature(meta, HC)

        possible, case, reason = _match_catalog_case(meta.critical_signature)
        if not possible:
            meta.critical_reason = reason
            continue

        any_node = next(iter(meta.nodes))
        idx = node_to_component_idx[any_node]
        K_graph = comp_graphs[idx]

        meta.opt_value = brute_force_opt_of_component(K_graph)
        is_critical, confirm_reason = _confirm_critical_case_with_exact_opt(meta, case)

        meta.is_critical = is_critical
        meta.critical_case = case if is_critical else None
        meta.critical_reason = confirm_reason

        if meta.is_critical:
            meta.critical_2_anchors = {
                a for a in meta.anchors if meta.j_anchor_map.get(a, 0) == 2
            }

    return metas