from __future__ import annotations

import json
from types import SimpleNamespace

import networkx as nx

from MPC4plus.phase1.modify_H_M import run_phase1
from MPC4plus.phase2.auxiliary_graph import build_auxiliary_graph_G1
from MPC4plus.phase2.matching_projection import build_M_C
from MPC4plus.phase2.path_cycle_cover import compute_max_weight_path_cycle_cover
from MPC4plus.phase4.operation_runner import run_phase4_operations
from MPC4plus.phase4.operations import CanonicalRuleSelector, OperationCandidate
from MPC4plus.experiments.baselines.rule_runner import run_rule_phase4_pipeline
from MPC4plus.experiments.export.candidate_table_builder import (
    sample_to_candidate_rows,
    samples_to_candidate_rows,
)
from MPC4plus.experiments.export.phase4_trace_exporter import Phase4TraceCollector


def _fake_satellite(
    nodes,
    *,
    kind="satellite",
    rescue_edges=(),
    supporting_anchors=(),
    rescue_anchor=None,
    is_critical_satellite=False,
):
    return SimpleNamespace(
        nodes=set(nodes),
        kind=kind,
        rescue_edges=tuple(rescue_edges),
        supporting_anchors=set(supporting_anchors),
        rescue_anchor=rescue_anchor,
        is_critical_satellite=is_critical_satellite,
    )


def _fake_meta(
    nodes,
    *,
    is_composite=True,
    center_nodes=(),
    center_kind="edge",
    anchors=(),
    j_anchor_map=None,
    critical_2_anchors=(),
    responsible_1_anchors=(),
    s_value=None,
    opt_value=None,
    critical_ratio=None,
    is_critical=False,
    critical_case="",
    critical_reason="",
    critical_signature=None,
    is_responsible_component=False,
    responsible_case="",
    responsible_reason="",
    responsible_signature=None,
    satellites=(),
):
    return SimpleNamespace(
        nodes=set(nodes),
        is_composite=is_composite,
        center_nodes=set(center_nodes),
        center_kind=center_kind,
        anchors=set(anchors),
        j_anchor_map=dict(j_anchor_map or {}),
        critical_2_anchors=set(critical_2_anchors),
        responsible_1_anchors=set(responsible_1_anchors),
        s_value=s_value,
        opt_value=opt_value,
        critical_ratio=critical_ratio,
        is_critical=is_critical,
        critical_case=critical_case,
        critical_reason=critical_reason,
        critical_signature=dict(critical_signature or {}),
        is_responsible_component=is_responsible_component,
        responsible_case=responsible_case,
        responsible_reason=responsible_reason,
        responsible_signature=dict(responsible_signature or {}),
        satellites=list(satellites),
    )


def _fake_state(
    *,
    iteration,
    G,
    H,
    C,
    M_C,
    metas,
    history,
):
    return SimpleNamespace(
        iteration=iteration,
        G=G,
        H=H,
        C=set(C),
        M_C=set(M_C),
        metas=list(metas),
        history=list(history),
    )


def _build_manual_trace_case():
    """
    Build a stable synthetic Phase 4 decision case for exporter/table tests.

    We do NOT depend on the full Phase 1-4 pipeline here, because the pipeline
    may legitimately produce zero Phase 4 operations on a particular demo graph.
    """
    G = nx.Graph()
    G.add_edges_from(
        [
            (1, 2),
            (2, 3),
            (4, 5),
            (5, 6),
            (2, 5),
        ]
    )
    H = nx.Graph()
    H.add_edges_from([(1, 2), (4, 5)])

    sat0 = _fake_satellite(
        nodes=[3],
        rescue_edges=[(2, 3)],
        supporting_anchors=[2],
        rescue_anchor=2,
        is_critical_satellite=True,
    )
    sat1 = _fake_satellite(
        nodes=[6],
        rescue_edges=[(5, 6)],
        supporting_anchors=[5],
        rescue_anchor=5,
        is_critical_satellite=False,
    )

    meta0 = _fake_meta(
        nodes=[1, 2, 3],
        center_nodes=[1, 2],
        center_kind="edge",
        anchors=[2],
        j_anchor_map={2: 2},
        critical_2_anchors=[2],
        responsible_1_anchors=[],
        s_value=2,
        opt_value=3,
        critical_ratio=2 / 3,
        is_critical=True,
        critical_case="manual_case",
        critical_reason="manual",
        critical_signature={"kind": "manual_critical"},
        is_responsible_component=False,
        responsible_case="",
        responsible_reason="",
        responsible_signature={},
        satellites=[sat0],
    )
    meta1 = _fake_meta(
        nodes=[4, 5, 6],
        center_nodes=[4, 5],
        center_kind="edge",
        anchors=[5],
        j_anchor_map={5: 1},
        critical_2_anchors=[],
        responsible_1_anchors=[5],
        s_value=2,
        opt_value=3,
        critical_ratio=2 / 3,
        is_critical=False,
        critical_case="",
        critical_reason="",
        critical_signature={},
        is_responsible_component=True,
        responsible_case="manual_resp",
        responsible_reason="manual",
        responsible_signature={"kind": "manual_responsible"},
        satellites=[sat1],
    )

    candidate0 = OperationCandidate(
        op_type="op1",
        source_component_idx=0,
        source_satellite_idx=0,
        source_vertex=3,
        target_component_idx=1,
        target_vertex=5,
        added_edge=(3, 5),
        removed_edges=((2, 3),),
        features={"expected_g_drop": 1.0, "toy_feature": 7.0},
        note="manual selected candidate",
    )
    candidate1 = OperationCandidate(
        op_type="op3",
        source_component_idx=0,
        source_satellite_idx=0,
        source_vertex=3,
        target_component_idx=1,
        target_vertex=4,
        added_edge=(3, 4),
        removed_edges=((2, 3),),
        features={"expected_g_drop": 0.0, "toy_feature": 2.0},
        note="manual unselected candidate",
    )

    state_before = _fake_state(
        iteration=0,
        G=G,
        H=H,
        C={(2, 3)},
        M_C={(1, 2), (4, 5)},
        metas=[meta0, meta1],
        history=[],
    )
    state_after = _fake_state(
        iteration=1,
        G=G,
        H=H,
        C={(3, 5)},
        M_C={(1, 2), (4, 5)},
        metas=[meta1],  # toy after-state: one critical component disappeared
        history=[candidate0],
    )

    return state_before, [candidate0, candidate1], candidate0, state_after


def _build_phase4_demo_graph() -> nx.Graph:
    G = nx.Graph()
    matched_edges = [(1, 2), (3, 4), (5, 6), (7, 8)]
    G.add_edges_from(matched_edges)
    G.add_edges_from(
        [
            (1, 6),
            (2, 5),
            (3, 8),
            (4, 7),
            (2, 9),
            (9, 10),
            (4, 11),
            (11, 12),
            (6, 13),
            (8, 14),
        ]
    )
    return G


def test_trace_collector_manual_record_decision_creates_sample() -> None:
    state_before, candidates, selected_candidate, state_after = _build_manual_trace_case()

    collector = Phase4TraceCollector()
    collector.start_graph("manual_graph")
    collector.record_decision(
        state_before=state_before,
        candidates=candidates,
        selected_candidate=selected_candidate,
        state_after=state_after,
        selector_name="canonical_rule",
        terminated_after_apply=False,
    )

    samples = collector.samples()
    assert len(samples) == 1

    sample = samples[0]
    assert sample.graph_id == "manual_graph"
    assert sample.state.iteration == 0
    assert sample.state.num_candidates_total == 2
    assert sample.decision.selector_name == "canonical_rule"
    assert sample.decision.selected_candidate_idx == 0
    assert len(sample.components) == 2
    assert len(sample.candidates) == 2


def test_trace_collector_rule_labels_have_single_positive_manual_case() -> None:
    state_before, candidates, selected_candidate, state_after = _build_manual_trace_case()

    collector = Phase4TraceCollector()
    collector.start_graph("manual_rule_graph")
    collector.record_decision(
        state_before=state_before,
        candidates=candidates,
        selected_candidate=selected_candidate,
        state_after=state_after,
        selector_name="canonical_rule",
        terminated_after_apply=False,
    )

    sample = collector.samples()[0]

    selected_by_rule_sum = sum(
        int(c.labels.get("selected_by_rule", 0))
        for c in sample.candidates
    )
    selected_in_run_sum = sum(
        int(c.labels.get("selected_in_run", 0))
        for c in sample.candidates
    )

    assert selected_by_rule_sum == 1
    assert selected_in_run_sum == 1
    assert sample.decision.selected_candidate_idx == 0


def test_candidate_table_builder_flattens_one_manual_sample() -> None:
    state_before, candidates, selected_candidate, state_after = _build_manual_trace_case()

    collector = Phase4TraceCollector()
    collector.start_graph("flat_manual_graph")
    collector.record_decision(
        state_before=state_before,
        candidates=candidates,
        selected_candidate=selected_candidate,
        state_after=state_after,
        selector_name="canonical_rule",
        terminated_after_apply=False,
    )

    sample = collector.samples()[0]
    rows = sample_to_candidate_rows(sample)

    assert len(rows) == 2
    assert all(row["graph_id"] == "flat_manual_graph" for row in rows)
    assert all("candidate_idx" in row for row in rows)
    assert all("op_type" in row for row in rows)
    assert all("label__selected_in_run" in row for row in rows)

    positive_rule_rows = [row for row in rows if row.get("label__selected_by_rule", 0) == 1]
    assert len(positive_rule_rows) == 1
    assert positive_rule_rows[0]["candidate_idx"] == 0


def test_candidate_table_builder_flattens_multiple_manual_samples() -> None:
    state_before, candidates, selected_candidate, state_after = _build_manual_trace_case()

    collector = Phase4TraceCollector()
    collector.start_graph("multi_manual_graph")
    collector.record_decision(
        state_before=state_before,
        candidates=candidates,
        selected_candidate=selected_candidate,
        state_after=state_after,
        selector_name="canonical_rule",
        terminated_after_apply=False,
    )
    collector.record_decision(
        state_before=state_before,
        candidates=candidates,
        selected_candidate=selected_candidate,
        state_after=state_after,
        selector_name="canonical_rule",
        terminated_after_apply=True,
    )

    samples = collector.samples()
    rows = samples_to_candidate_rows(samples)

    assert len(samples) == 2
    assert len(rows) == 4

    sample_ids_in_rows = {row["sample_id"] for row in rows}
    sample_ids_in_samples = {sample.sample_id for sample in samples}
    assert sample_ids_in_rows == sample_ids_in_samples


def test_trace_jsonl_dump_roundtrip_manual_case(tmp_path) -> None:
    state_before, candidates, selected_candidate, state_after = _build_manual_trace_case()

    collector = Phase4TraceCollector()
    collector.start_graph("jsonl_manual_graph")
    collector.record_decision(
        state_before=state_before,
        candidates=candidates,
        selected_candidate=selected_candidate,
        state_after=state_after,
        selector_name="canonical_rule",
        terminated_after_apply=False,
    )

    out_path = tmp_path / "phase4_trace.jsonl"
    collector.dump_jsonl(str(out_path))

    assert out_path.exists()

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    first_obj = json.loads(lines[0])
    assert first_obj["graph_id"] == "jsonl_manual_graph"
    assert "state" in first_obj
    assert "components" in first_obj
    assert "candidates" in first_obj
    assert "decision" in first_obj
    assert "outcome" in first_obj


def test_rule_phase4_pipeline_trace_hook_is_safe_even_if_no_samples() -> None:
    """
    Integration-level safety test:
    the pipeline should run with trace enabled, but we do NOT require
    this particular graph to necessarily trigger a Phase 4 operation.
    """
    G = _build_phase4_demo_graph()

    result = run_rule_phase4_pipeline(
        G,
        graph_id="pipeline_graph",
        backend="ilp",
        enable_trace=True,
    )

    assert result.trace_collector is not None
    assert isinstance(result.trace_collector.samples(), list)
    assert isinstance(result.final_state.C, set)


def test_run_phase4_operations_trace_hook_is_safe_even_if_no_samples() -> None:
    """
    Another integration-level safety test:
    attaching trace_collector must not break the normal Phase 4 pipeline.
    """
    G = _build_phase4_demo_graph()

    phase1_state = run_phase1(G)
    M = set(phase1_state.M)
    H = phase1_state.H

    G1 = build_auxiliary_graph_G1(G, H)
    C = set(compute_max_weight_path_cycle_cover(G1, H, backend="ilp"))
    M_C = set(build_M_C(M, H, C))

    collector = Phase4TraceCollector()
    collector.start_graph("safe_graph")

    final_state = run_phase4_operations(
        G=G,
        H=H,
        M_C=M_C,
        C=C,
        selector=CanonicalRuleSelector(),
        trace_collector=collector,
    )

    assert isinstance(final_state.C, set)
    assert isinstance(collector.samples(), list)