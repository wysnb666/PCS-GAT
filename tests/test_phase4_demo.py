from __future__ import annotations

import networkx as nx

from MPC4plus.phase3.anchors import (
    enrich_metas_with_critical_satellites,
    get_critical_satellite_indices,
)
from MPC4plus.phase3.component_meta import ComponentMeta, SatelliteInfo
from MPC4plus.phase4.candidate_enumerator import enumerate_operation_candidates
from MPC4plus.phase4.operation_runner import rebuild_phase4_state
from MPC4plus.phase4.operations import (
    GreedyOperationSelector,
    Phase4State,
    apply_operation_candidate,
)


def _manual_metas_for_op1() -> list[ComponentMeta]:
    src = ComponentMeta(
        nodes={1, 2, 3, 4},
        is_composite=True,
        H_component_node_sets=[{1, 2}, {3, 4}],
        center_nodes={1, 2},
        center_kind="edge",
        satellites=[
            SatelliteInfo(
                nodes={3, 4},
                kind="edge",
                rescue_edges=[(2, 3)],
                supporting_anchors=[2],
                rescue_anchor=2,
                is_critical_satellite=True,
            )
        ],
        anchors=[1, 2],
        j_anchor_map={1: 0, 2: 2},
        critical_2_anchors={2},
        responsible_1_anchors=set(),
        s_value=8,
        opt_value=6,
        critical_ratio=8 / 6,
        is_critical=True,
        critical_case="one-2-anchor-edge-s8",
    )
    tgt = ComponentMeta(
        nodes={10, 11},
        is_composite=False,
        H_component_node_sets=[{10, 11}],
        center_nodes={10, 11},
        center_kind="edge",
        satellites=[],
        anchors=[10, 11],
        j_anchor_map={10: 0, 11: 1},
        critical_2_anchors=set(),
        responsible_1_anchors={11},
        s_value=2,
        opt_value=0,
        critical_ratio=0.0,
        is_critical=False,
        critical_case=None,
    )
    return [src, tgt]


def test_get_critical_satellite_indices():
    metas = _manual_metas_for_op1()
    idxs = get_critical_satellite_indices(metas[0])
    assert idxs == [0]


def test_enrich_metas_with_critical_satellites_marks_definition9():
    meta = ComponentMeta(
        nodes={1, 2, 3, 4, 5, 6},
        is_composite=True,
        H_component_node_sets=[{1, 2}, {3, 4}, {5, 6}],
        center_nodes={1, 2},
        center_kind="edge",
        satellites=[
            SatelliteInfo(
                nodes={3, 4},
                kind="edge",
                rescue_edges=[(1, 3)],
                supporting_anchors=[1],
            ),
            SatelliteInfo(
                nodes={5, 6},
                kind="edge",
                rescue_edges=[(2, 5)],
                supporting_anchors=[2],
            ),
        ],
        anchors=[1, 2],
        j_anchor_map={1: 2, 2: 0},
        critical_2_anchors={1},
        responsible_1_anchors=set(),
        s_value=8,
        opt_value=6,
        critical_ratio=8 / 6,
        is_critical=True,
        critical_case="one-2-anchor-edge-s8",
    )

    metas = enrich_metas_with_critical_satellites([meta])
    out = metas[0]

    assert out.satellites[0].rescue_anchor == 1
    assert out.satellites[0].is_critical_satellite is True

    assert out.satellites[1].rescue_anchor == 2
    assert out.satellites[1].is_critical_satellite is False


def test_enumerate_phase4_op1_candidate_to_zero_anchor_only():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (3, 10), (3, 11), (10, 11)])
    H = nx.Graph()
    H.add_edges_from([(1, 2), (3, 4), (10, 11)])
    C = {(2, 3)}

    metas = _manual_metas_for_op1()
    candidates = enumerate_operation_candidates(G, H, C, metas)

    assert any(c.op_type == "op1" and c.added_edge == (3, 10) for c in candidates)
    assert all(c.added_edge != (3, 11) for c in candidates)


def test_apply_phase4_candidate_replaces_edge():
    G = nx.Graph()
    H = nx.Graph()
    metas = _manual_metas_for_op1()
    C = {(2, 3)}

    candidates = enumerate_operation_candidates(
        nx.Graph([(1, 2), (2, 3), (3, 4), (3, 10), (10, 11)]),
        H,
        C,
        metas,
    )
    chosen = GreedyOperationSelector().select(
        candidates,
        Phase4State(G=G, H=H, M_C=set(), C=C, HC=nx.Graph(), metas=metas),
    )
    assert chosen is not None

    new_C = apply_operation_candidate(C, chosen)
    assert (2, 3) not in new_C
    assert (3, 10) in new_C


def test_strict_responsible_1_anchor_detection():
    G = nx.Graph()
    H = nx.Graph()

    H.add_edge(1, 2)
    H.add_edges_from([(3, 4), (5, 6), (7, 8)])

    C = {(1, 3), (1, 5), (2, 7)}
    M_C = {(1, 2), (3, 4), (5, 6), (7, 8)}

    G.add_edges_from(H.edges())
    G.add_edges_from([(2, 3), (2, 5)])

    state = rebuild_phase4_state(G, H, M_C, C)
    assert len(state.metas) == 1
    meta = state.metas[0]

    assert meta.is_critical is True
    assert meta.critical_case == "one-2-anchor-edge-s8"
    assert meta.j_anchor_map == {1: 2, 2: 1}
    assert meta.critical_2_anchors == {1}
    assert meta.responsible_1_anchors == {2}

    critical_sats = get_critical_satellite_indices(meta)
    assert sorted(critical_sats) == [0, 1]

    candidates = enumerate_operation_candidates(G, H, C, state.metas)
    assert all(c.target_vertex != 2 for c in candidates)