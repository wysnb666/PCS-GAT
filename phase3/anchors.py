from __future__ import annotations

import networkx as nx

from MPC4plus.core.components import classify_component
from MPC4plus.core.graph_utils import normalize_edge
from MPC4plus.phase3.component_meta import ComponentMeta, SatelliteInfo
from MPC4plus.phase3.composite_components import build_all_component_meta
from MPC4plus.phase3.critical import mark_critical_components
from MPC4plus.phase3.h_plus_c import build_H_plus_C


Edge = tuple[int, int]


def compute_anchors(meta: ComponentMeta, H: nx.Graph) -> list[int]:
    """
    Definition 7:
    - if center element is a 5-path or edge, all vertices are anchors
    - if center element is a star, only the center vertex is an anchor
    """
    if meta.center_nodes is None:
        return []

    center_sg = H.subgraph(meta.center_nodes).copy()
    center_kind = classify_component(center_sg)

    if center_kind in {"five_path", "edge"}:
        return sorted(center_sg.nodes())

    if center_kind == "star":
        for v, d in center_sg.degree():
            if d == center_sg.number_of_nodes() - 1:
                return [v]

    return []


def compute_rescue_edges_and_supporting_anchors(
    meta: ComponentMeta,
    H: nx.Graph,
    HC: nx.Graph,
) -> None:
    """
    For each satellite, find all edges in HC between that satellite and the center anchors.
    Those are rescue-edges; the anchor endpoints are supporting anchors.
    """
    anchors = set(meta.anchors)
    for sat in meta.satellites:
        sat.rescue_edges = []
        sat.supporting_anchors = []
        sat.rescue_anchor = None
        sat.is_critical_satellite = False

        for u, v in HC.edges():
            if u in sat.nodes and v in anchors:
                sat.rescue_edges.append((u, v))
                sat.supporting_anchors.append(v)
            elif v in sat.nodes and u in anchors:
                sat.rescue_edges.append((u, v))
                sat.supporting_anchors.append(u)


def compute_j_anchor_map(meta: ComponentMeta) -> dict[int, int]:
    """
    j-anchor: an anchor supporting exactly j satellites.
    """
    count = {a: 0 for a in meta.anchors}
    for sat in meta.satellites:
        for a in sorted(set(sat.supporting_anchors)):
            if a in count:
                count[a] += 1
    return count


def _infer_unique_rescue_anchor(sat: SatelliteInfo) -> int | None:
    distinct_supporting = sorted(set(sat.supporting_anchors))
    if len(distinct_supporting) != 1:
        return None
    return distinct_supporting[0]


def mark_critical_satellites(meta: ComponentMeta) -> ComponentMeta:
    """
    Definition 9:
    a critical satellite-element is a satellite-element whose rescue-anchor is critical.
    """
    critical_anchors = set(meta.critical_2_anchors)

    for sat in meta.satellites:
        sat.rescue_anchor = _infer_unique_rescue_anchor(sat)
        sat.is_critical_satellite = (
            sat.rescue_anchor is not None and sat.rescue_anchor in critical_anchors
        )

    return meta


def enrich_metas_with_critical_satellites(
    metas: list[ComponentMeta],
) -> list[ComponentMeta]:
    for meta in metas:
        mark_critical_satellites(meta)
    return metas


def get_critical_satellite_indices(meta: ComponentMeta) -> list[int]:
    """
    Return indices of satellite-elements marked critical by Definition 9.
    """
    out: list[int] = []
    for idx, sat in enumerate(meta.satellites):
        if sat.is_critical_satellite:
            out.append(idx)
    return out


def _simulate_component_metas_after_edit(
    H: nx.Graph,
    C: set[Edge],
    M_C: set[Edge],
) -> tuple[nx.Graph, list[ComponentMeta]]:
    HC = build_H_plus_C(H, C)
    metas = build_all_component_meta(HC, H)
    metas = [enrich_meta_with_anchor_info(meta, H, HC) for meta in metas]
    metas = mark_critical_components(metas, HC, M_C)
    metas = enrich_metas_with_critical_satellites(metas)
    return HC, metas


def _component_idx_containing_vertex(metas: list[ComponentMeta], v: int) -> int | None:
    for idx, meta in enumerate(metas):
        if v in meta.nodes:
            return idx
    return None


def _candidate_move_edges(
    G: nx.Graph,
    C: set[Edge],
    sat: SatelliteInfo,
    anchor: int,
) -> list[Edge]:
    out: list[Edge] = []
    for x in sorted(sat.nodes):
        if not G.has_edge(anchor, x):
            continue
        edge = normalize_edge(anchor, x)
        if edge in C:
            continue
        out.append(edge)
    return out


def is_responsible_1_anchor(
    G: nx.Graph,
    H: nx.Graph,
    C: set[Edge],
    M_C: set[Edge],
    metas: list[ComponentMeta],
    component_idx: int,
    anchor: int,
) -> bool:
    """
    Strict Definition 11 test.

    Let K be the current component indexed by `component_idx` and let `anchor` be a
    1-anchor of K. Then `anchor` is responsible iff there exists a critical satellite
    element S adjacent to `anchor` such that after moving S to `anchor`, the resulting
    component containing `anchor` is still critical.
    """
    meta = metas[component_idx]
    if anchor not in meta.anchors:
        return False
    if meta.j_anchor_map.get(anchor) != 1:
        return False
    if not meta.is_critical:
        return False

    critical_satellite_indices = get_critical_satellite_indices(meta)
    for sat_idx in critical_satellite_indices:
        sat = meta.satellites[sat_idx]

        rescue_edges = [normalize_edge(*e) for e in sat.rescue_edges]
        if len(rescue_edges) != 1:
            continue

        for move_edge in _candidate_move_edges(G, C, sat, anchor):
            new_C = set(C)
            new_C.discard(rescue_edges[0])
            new_C.add(move_edge)

            _, new_metas = _simulate_component_metas_after_edit(H, new_C, M_C)
            new_idx = _component_idx_containing_vertex(new_metas, anchor)
            if new_idx is None:
                continue
            if new_metas[new_idx].is_critical:
                return True

    return False


def compute_responsible_1_anchors(
    G: nx.Graph,
    H: nx.Graph,
    C: set[Edge],
    M_C: set[Edge],
    metas: list[ComponentMeta],
    component_idx: int,
) -> set[int]:
    meta = metas[component_idx]
    result: set[int] = set()

    for anchor in meta.anchors:
        if meta.j_anchor_map.get(anchor) == 1:
            if is_responsible_1_anchor(G, H, C, M_C, metas, component_idx, anchor):
                result.add(anchor)

    return result


def enrich_metas_with_responsible_1_anchors(
    G: nx.Graph,
    H: nx.Graph,
    C: set[Edge],
    M_C: set[Edge],
    metas: list[ComponentMeta],
) -> list[ComponentMeta]:
    """
    Fill `meta.responsible_1_anchors` using the strict Definition 11 simulation test.

    Assumes critical components and critical satellites have already been marked.
    """
    for idx, meta in enumerate(metas):
        meta.responsible_1_anchors = compute_responsible_1_anchors(
            G=G,
            H=H,
            C=C,
            M_C=M_C,
            metas=metas,
            component_idx=idx,
        )
    return metas


def enrich_meta_with_anchor_info(meta: ComponentMeta, H: nx.Graph, HC: nx.Graph) -> ComponentMeta:
    meta.anchors = compute_anchors(meta, H)
    compute_rescue_edges_and_supporting_anchors(meta, H, HC)
    meta.j_anchor_map = compute_j_anchor_map(meta)
    meta.critical_2_anchors = set()
    meta.responsible_1_anchors = set()

    for sat in meta.satellites:
        sat.rescue_anchor = None
        sat.is_critical_satellite = False

    return meta