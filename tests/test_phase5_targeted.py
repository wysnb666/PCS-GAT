from __future__ import annotations

from types import SimpleNamespace

import networkx as nx

from MPC4plus.phase3.component_meta import ComponentMeta, SatelliteInfo
from MPC4plus.phase3.responsible import mark_responsible_components
from MPC4plus.phase5.decomposition import classify_phase5_buckets
from MPC4plus.phase5.decision import decide_phase5_route
from MPC4plus.phase5.exact_solver import is_feasible_solution_graph
from MPC4plus.phase5.path_constructor import construct_Pv_for_anchor, construct_all_Pv
from MPC4plus.phase5.phase5_runner import solve_mpc4plus


def _make_critical_component_meta(offset: int = 0) -> ComponentMeta:
    def s(x: int) -> int:
        return x + offset

    return ComponentMeta(
        nodes={s(1), s(2), s(3), s(4), s(5), s(6)},
        is_composite=True,
        H_component_node_sets=[{s(1), s(2)}, {s(3), s(4)}, {s(5), s(6)}],
        center_nodes={s(1), s(2)},
        center_kind="edge",
        satellites=[
            SatelliteInfo(
                nodes={s(3), s(4)},
                kind="edge",
                rescue_edges=[(s(1), s(3))],
                supporting_anchors=[s(1)],
                rescue_anchor=s(1),
                is_critical_satellite=True,
            ),
            SatelliteInfo(
                nodes={s(5), s(6)},
                kind="edge",
                rescue_edges=[(s(1), s(5))],
                supporting_anchors=[s(1)],
                rescue_anchor=s(1),
                is_critical_satellite=True,
            ),
        ],
        anchors=[s(1), s(2)],
        j_anchor_map={s(1): 2, s(2): 0},
        critical_2_anchors={s(1)},
        responsible_1_anchors=set(),
        s_value=8,
        opt_value=6,
        critical_ratio=8 / 6,
        is_critical=True,
        critical_case="one-2-anchor-edge-s8",
    )


def _make_noncritical_component_with_responsible_1_anchor(offset: int = 0) -> ComponentMeta:
    def s(x: int) -> int:
        return x + offset

    return ComponentMeta(
        nodes={s(1), s(2), s(3), s(4)},
        is_composite=True,
        H_component_node_sets=[{s(1), s(2)}, {s(3), s(4)}],
        center_nodes={s(1), s(2)},
        center_kind="edge",
        satellites=[
            SatelliteInfo(
                nodes={s(3), s(4)},
                kind="edge",
                rescue_edges=[(s(1), s(3))],
                supporting_anchors=[s(1)],
                rescue_anchor=s(1),
                is_critical_satellite=False,
            )
        ],
        anchors=[s(1), s(2)],
        j_anchor_map={s(1): 1, s(2): 0},
        critical_2_anchors=set(),
        responsible_1_anchors={s(1)},
        s_value=6,
        opt_value=5,
        critical_ratio=6 / 5,
        is_critical=False,
        critical_case=None,
    )


def _make_isolated_five_path_meta(nodes: list[int]) -> ComponentMeta:
    node_set = set(nodes)
    return ComponentMeta(
        nodes=node_set,
        is_composite=False,
        H_component_node_sets=[node_set],
        center_nodes=node_set,
        center_kind="five_path",
        satellites=[],
        anchors=nodes[:],
        j_anchor_map={v: 0 for v in nodes},
        critical_2_anchors=set(),
        responsible_1_anchors=set(),
        s_value=0,
        opt_value=0,
        critical_ratio=0.0,
        is_critical=False,
        critical_case=None,
    )


def test_responsible_component_catalog_marks_explicit_case():
    meta = _make_noncritical_component_with_responsible_1_anchor(offset=0)
    metas = mark_responsible_components([meta])
    out = metas[0]

    assert out.is_responsible_component is True
    assert out.responsible_case is not None
    assert out.responsible_signature["num_responsible_1_anchors"] == 1


def test_phase5_targeted_buckets_rc_uc_gc_are_precise():
    G = nx.Graph()
    G.add_nodes_from(range(1, 12))

    critical_meta = _make_critical_component_meta(offset=0)
    buckets = classify_phase5_buckets(G, [critical_meta])

    assert buckets.R == {1}
    assert buckets.Rc == {1}
    assert buckets.Uc == {3, 4, 5, 6}
    assert buckets.Gc_nodes == {2, 7, 8, 9, 10, 11}


def test_phase5_targeted_decision_recurse_boundary_true():
    G = nx.Graph()
    G.add_nodes_from(range(1, 12))

    critical_meta = _make_critical_component_meta(offset=0)
    buckets = classify_phase5_buckets(G, [critical_meta])
    decision = decide_phase5_route(buckets)

    assert decision.has_critical is True
    assert decision.lhs_value == 1
    assert decision.rhs_value > 1
    assert decision.recurse_on_Gc is True


def test_phase5_targeted_decision_direct_when_no_critical():
    G = nx.Graph()
    G.add_nodes_from(range(1, 10))

    noncritical_meta = _make_noncritical_component_with_responsible_1_anchor(offset=0)
    noncritical_meta = mark_responsible_components([noncritical_meta])[0]

    buckets = classify_phase5_buckets(G, [noncritical_meta])
    decision = decide_phase5_route(buckets)

    assert decision.has_critical is False
    assert decision.recurse_on_Gc is False
    assert "no critical component exists" in decision.reason


def test_phase5_targeted_construct_Pv_structured_local_path():
    G = nx.Graph()
    G.add_edges_from([
        (4, 3),
        (3, 1),
        (1, 2),
        (2, 5),
        (5, 6),
    ])

    meta = _make_critical_component_meta(offset=0)
    Pv = construct_Pv_for_anchor(G, [meta], 1)

    assert Pv.number_of_nodes() >= 5
    assert 1 in Pv.nodes()
    assert nx.is_connected(Pv)


def test_phase5_targeted_construct_Pv_prefers_branch_style_two_satellites():
    G = nx.Graph()
    G.add_edges_from([
        (4, 3),
        (3, 1),
        (1, 2),
        (2, 5),
        (5, 6),
    ])

    meta = _make_critical_component_meta(offset=0)
    Pv = construct_Pv_for_anchor(G, [meta], 1)

    assert set([1, 2, 3, 4, 5, 6]).issuperset(Pv.nodes())
    assert Pv.number_of_nodes() == 6


def test_phase5_targeted_construct_all_Pv_two_disjoint_anchors():
    G = nx.Graph()
    G.add_edges_from([
        (4, 3), (3, 1), (1, 2), (2, 5), (5, 6),
        (14, 13), (13, 11), (11, 12), (12, 15), (15, 16),
    ])

    meta_a = _make_critical_component_meta(offset=0)
    meta_b = _make_critical_component_meta(offset=10)

    U = construct_all_Pv(G, [meta_a, meta_b], Rc={1, 11})

    assert U.number_of_nodes() >= 10
    assert len(list(nx.connected_components(U))) == 2
    assert is_feasible_solution_graph(U)


def test_phase5_targeted_recursive_branch_stitches_Pv_and_recursive_solution(monkeypatch):
    G = nx.Graph()
    G.add_edges_from([
        (4, 3), (3, 1), (1, 2), (2, 5), (5, 6),
        (7, 8), (8, 9), (9, 10), (10, 11),
    ])

    top_meta = _make_critical_component_meta(offset=0)
    recursive_meta = _make_isolated_five_path_meta([7, 8, 9, 10, 11])

    def fake_compute_phase1_to_phase4(graph: nx.Graph):
        nodes = set(graph.nodes())
        if {1, 2, 3, 4, 5, 6}.issubset(nodes):
            phase4_state = SimpleNamespace(metas=[top_meta])
            return None, set(), set(), phase4_state

        phase4_state = SimpleNamespace(metas=[recursive_meta])
        return None, set(), set(), phase4_state

    monkeypatch.setattr(
        "MPC4plus.phase5.phase5_runner._compute_phase1_to_phase4",
        fake_compute_phase1_to_phase4,
    )

    result = solve_mpc4plus(G)

    assert result.phase5_decision.recurse_on_Gc is True
    assert result.solution_graph.number_of_nodes() >= 10
    assert is_feasible_solution_graph(result.solution_graph)
    assert len(list(nx.connected_components(result.solution_graph))) == 2