from __future__ import annotations

import networkx as nx

from MPC4plus.core.graph_utils import normalize_edge
from MPC4plus.phase3.anchors import get_critical_satellite_indices
from MPC4plus.phase3.component_meta import ComponentMeta, SatelliteInfo
from MPC4plus.phase4.operations import OperationCandidate


Edge = tuple[int, int]


def _find_component_idx_containing_vertex(metas: list[ComponentMeta], v: int) -> int | None:
    for idx, meta in enumerate(metas):
        if v in meta.nodes:
            return idx
    return None


def _find_satellite_idx_containing_vertex(meta: ComponentMeta, v: int) -> int | None:
    for idx, sat in enumerate(meta.satellites):
        if v in sat.nodes:
            return idx
    return None


def _is_anchor(meta: ComponentMeta, v: int) -> bool:
    return v in set(meta.anchors)


def _anchor_j(meta: ComponentMeta, v: int) -> int | None:
    return meta.j_anchor_map.get(v, None)


def _is_non_responsible_1_anchor(meta: ComponentMeta, v: int) -> bool:
    return _anchor_j(meta, v) == 1 and v not in meta.responsible_1_anchors


def _rescue_edges_of_satellite(sat: SatelliteInfo) -> tuple[Edge, ...]:
    return tuple(normalize_edge(*e) for e in sat.rescue_edges)


def _feature_pack(
    op_type: str,
    source_meta: ComponentMeta,
    target_meta: ComponentMeta,
) -> dict[str, float]:
    return {
        "expected_g_drop": 1.0,
        "source_component_size": float(len(source_meta.nodes)),
        "target_component_size": float(len(target_meta.nodes)),
        "source_is_critical": float(source_meta.is_critical),
        "target_is_critical": float(target_meta.is_critical),
        "target_num_satellites": float(len(target_meta.satellites)),
        "target_num_responsible_1_anchors": float(len(target_meta.responsible_1_anchors)),
    }


def enumerate_operation_candidates(
    G: nx.Graph,
    H: nx.Graph,
    C: set[Edge],
    metas: list[ComponentMeta],
) -> list[OperationCandidate]:
    """
    Enumerate all currently legal Phase 4 candidates.

    Alignment with the paper:
    - Operation 1: v' is a 0-anchor or a non-responsible 1-anchor.
    - Operation 2: v' lies in the unique satellite of a component whose center is edge/star.
    - Operation 3: v' lies in a satellite of a component whose center is five_path, or
      in a component having at least two satellites.

    Source satellites are read from the explicit Definition-9 structure field
    `sat.is_critical_satellite`.
    """
    candidates: list[OperationCandidate] = []

    for src_idx, src_meta in enumerate(metas):
        if not src_meta.is_critical:
            continue

        for sat_idx in get_critical_satellite_indices(src_meta):
            sat = src_meta.satellites[sat_idx]
            sat_nodes = set(sat.nodes)
            sat_rescue_edges = _rescue_edges_of_satellite(sat)
            if len(sat_rescue_edges) != 1:
                continue

            for v in sorted(sat_nodes):
                for vp in sorted(G.neighbors(v)):
                    edge = normalize_edge(v, vp)
                    if edge in C:
                        continue
                    if vp in sat_nodes:
                        continue

                    tgt_idx = _find_component_idx_containing_vertex(metas, vp)
                    if tgt_idx is None:
                        continue
                    tgt_meta = metas[tgt_idx]

                    if _is_anchor(tgt_meta, vp):
                        j = _anchor_j(tgt_meta, vp)
                        if j == 0:
                            candidates.append(
                                OperationCandidate(
                                    op_type="op1",
                                    source_component_idx=src_idx,
                                    source_satellite_idx=sat_idx,
                                    source_vertex=v,
                                    target_component_idx=tgt_idx,
                                    target_vertex=vp,
                                    added_edge=edge,
                                    removed_edges=(sat_rescue_edges[0],),
                                    features=_feature_pack("op1", src_meta, tgt_meta),
                                    note="target is 0-anchor",
                                )
                            )
                            continue

                        if _is_non_responsible_1_anchor(tgt_meta, vp):
                            candidates.append(
                                OperationCandidate(
                                    op_type="op1",
                                    source_component_idx=src_idx,
                                    source_satellite_idx=sat_idx,
                                    source_vertex=v,
                                    target_component_idx=tgt_idx,
                                    target_vertex=vp,
                                    added_edge=edge,
                                    removed_edges=(sat_rescue_edges[0],),
                                    features=_feature_pack("op1", src_meta, tgt_meta),
                                    note="target is strict non-responsible 1-anchor",
                                )
                            )
                            continue

                    tgt_sat_idx = _find_satellite_idx_containing_vertex(tgt_meta, vp)
                    if tgt_sat_idx is None:
                        continue

                    tgt_sat = tgt_meta.satellites[tgt_sat_idx]
                    tgt_sat_rescue_edges = _rescue_edges_of_satellite(tgt_sat)
                    if len(tgt_sat_rescue_edges) != 1:
                        continue

                    if tgt_meta.center_kind in {"edge", "star"} and len(tgt_meta.satellites) == 1:
                        candidates.append(
                            OperationCandidate(
                                op_type="op2",
                                source_component_idx=src_idx,
                                source_satellite_idx=sat_idx,
                                source_vertex=v,
                                target_component_idx=tgt_idx,
                                target_vertex=vp,
                                added_edge=edge,
                                removed_edges=(sat_rescue_edges[0],),
                                features=_feature_pack("op2", src_meta, tgt_meta),
                                note="target lies in unique satellite of edge/star component",
                            )
                        )
                        continue

                    if tgt_meta.center_kind == "five_path" or len(tgt_meta.satellites) >= 2:
                        candidates.append(
                            OperationCandidate(
                                op_type="op3",
                                source_component_idx=src_idx,
                                source_satellite_idx=sat_idx,
                                source_vertex=v,
                                target_component_idx=tgt_idx,
                                target_vertex=vp,
                                added_edge=edge,
                                removed_edges=(sat_rescue_edges[0], tgt_sat_rescue_edges[0]),
                                features=_feature_pack("op3", src_meta, tgt_meta),
                                note="target lies in satellite of five_path / multi-satellite component",
                            )
                        )

    unique: dict[tuple, OperationCandidate] = {}
    for c in candidates:
        key = (
            c.op_type,
            c.source_component_idx,
            c.source_satellite_idx,
            c.source_vertex,
            c.target_component_idx,
            c.target_vertex,
            c.added_edge,
            c.removed_edges,
        )
        unique[key] = c

    return list(unique.values())