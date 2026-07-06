from __future__ import annotations

import networkx as nx

from MPC4plus.solver.phase3_runner import build_demo_graph_phase3
from MPC4plus.phase1.modify_H_M import run_phase1
from MPC4plus.phase2.auxiliary_graph import build_auxiliary_graph_G1
from MPC4plus.phase2.path_cycle_cover import compute_max_weight_path_cycle_cover
from MPC4plus.phase2.matching_projection import build_M_C
from MPC4plus.phase3.h_plus_c import build_H_plus_C
from MPC4plus.phase3.composite_components import build_all_component_meta
from MPC4plus.phase3.anchors import enrich_meta_with_anchor_info
from MPC4plus.phase3.critical import mark_critical_components


def test_phase3_pipeline_runs():
    G = build_demo_graph_phase3()
    phase1 = run_phase1(G)
    G1 = build_auxiliary_graph_G1(G, phase1.H)
    C = compute_max_weight_path_cycle_cover(G1, phase1.H)
    M_C = build_M_C(phase1.M, phase1.H, C)

    HC = build_H_plus_C(phase1.H, C)
    metas = build_all_component_meta(HC, phase1.H)
    metas = [enrich_meta_with_anchor_info(meta, phase1.H, HC) for meta in metas]
    metas = mark_critical_components(metas, HC, M_C)

    assert isinstance(metas, list)
    assert len(metas) >= 1


def test_structured_critical_component_one_two_anchor_edge():
    """
    A known critical configuration of the first-row type:
    center is an edge, anchors have pattern {2-anchor, 1-anchor}, and s(K)=8.
    """
    H = nx.Graph()
    H.add_edge(1, 2)
    H.add_edges_from([(3, 4), (5, 6), (7, 8)])

    C = {(1, 3), (1, 5), (2, 7)}
    M_C = {(1, 2), (3, 4), (5, 6), (7, 8)}

    HC = build_H_plus_C(H, C)
    metas = build_all_component_meta(HC, H)
    metas = [enrich_meta_with_anchor_info(meta, H, HC) for meta in metas]
    metas = mark_critical_components(metas, HC, M_C)

    assert len(metas) == 1
    meta = metas[0]
    assert meta.s_value == 8
    assert meta.is_critical is True
    assert meta.critical_case == "one-2-anchor-edge-s8"
    assert meta.critical_2_anchors == {1}
    assert meta.critical_signature["center_kind"] == "edge"
    assert meta.critical_signature["n0"] == 0
    assert meta.critical_signature["n1"] == 1
    assert meta.critical_signature["n2"] == 1
    assert meta.critical_signature["num_satellites"] == 3


def test_structured_noncritical_component_two_two_anchor_edge():
    """
    Lemma 12(1): if K has exactly two 2-anchors and the center is an edge,
    then K is not critical.
    """
    H = nx.Graph()
    H.add_edge(1, 2)
    H.add_edges_from([(3, 4), (5, 6), (7, 8), (9, 10)])

    C = {(1, 3), (1, 5), (2, 7), (2, 9)}
    M_C = {(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)}

    HC = build_H_plus_C(H, C)
    metas = build_all_component_meta(HC, H)
    metas = [enrich_meta_with_anchor_info(meta, H, HC) for meta in metas]
    metas = mark_critical_components(metas, HC, M_C)

    assert len(metas) == 1
    meta = metas[0]
    assert meta.s_value == 10
    assert meta.is_critical is False
    assert "Lemma 12(1)" in meta.critical_reason


def test_structured_signature_for_two_two_anchor_five_path():
    """
    Check that the new catalog signature explicitly records the two 2-anchor
    positions on a 5-path center.
    """
    H = nx.Graph()
    H.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])   # 5-path center
    H.add_edges_from([(6, 7), (8, 9), (10, 11), (12, 13)])

    # make anchors 2 and 4 into 2-anchors
    C = {(2, 6), (2, 8), (4, 10), (4, 12)}
    M_C = {
        (1, 2), (2, 3), (3, 4), (4, 5),
        (6, 7), (8, 9), (10, 11), (12, 13),
    }

    HC = build_H_plus_C(H, C)
    metas = build_all_component_meta(HC, H)
    metas = [enrich_meta_with_anchor_info(meta, H, HC) for meta in metas]
    metas = mark_critical_components(metas, HC, M_C)

    assert len(metas) == 1
    meta = metas[0]
    assert meta.critical_signature["center_kind"] == "five_path"
    assert meta.critical_signature["n2"] == 2
    assert meta.critical_signature["two_anchor_positions_on_five_path"] == (1, 3)