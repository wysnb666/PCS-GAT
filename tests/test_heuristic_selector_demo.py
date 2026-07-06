from __future__ import annotations

from types import SimpleNamespace

from MPC4plus.phase4.operations import OperationCandidate
from MPC4plus.experiments.baselines.heuristic_selector import (
    OneStepHeuristicScore,
    OneStepHeuristicSelector,
)


def _fake_state():
    return SimpleNamespace(
        G=object(),
        H=object(),
        M_C=set(),
        C=set(),
        metas=[],
    )


def _fake_meta(*, is_critical: bool, is_responsible_component: bool):
    return SimpleNamespace(
        is_critical=is_critical,
        is_responsible_component=is_responsible_component,
    )


def test_one_step_heuristic_returns_none_on_empty_candidates() -> None:
    selector = OneStepHeuristicSelector()
    chosen = selector.select([], _fake_state())
    assert chosen is None


def test_one_step_heuristic_prefers_fewer_critical_components(monkeypatch) -> None:
    selector = OneStepHeuristicSelector()
    state = _fake_state()

    candidate_a = OperationCandidate(
        op_type="op1",
        source_component_idx=0,
        source_satellite_idx=0,
        source_vertex=1,
        target_component_idx=1,
        target_vertex=2,
        added_edge=(1, 2),
        removed_edges=(),
        features={"expected_g_drop": 0.0},
        note="candidate_a",
    )
    candidate_b = OperationCandidate(
        op_type="op1",
        source_component_idx=0,
        source_satellite_idx=0,
        source_vertex=1,
        target_component_idx=1,
        target_vertex=3,
        added_edge=(1, 3),
        removed_edges=(),
        features={"expected_g_drop": 0.0},
        note="candidate_b",
    )

    def fake_apply_operation_candidate(C, candidate):
        return {candidate.added_edge}

    def fake_rebuild_phase4_state(G, H, M_C, C):
        if (1, 2) in C:
            # worse: 2 critical components after
            metas = [
                _fake_meta(is_critical=True, is_responsible_component=False),
                _fake_meta(is_critical=True, is_responsible_component=False),
            ]
        else:
            # better: 1 critical component after
            metas = [
                _fake_meta(is_critical=True, is_responsible_component=False),
            ]
        return SimpleNamespace(metas=metas)

    monkeypatch.setattr(
        "MPC4plus.experiments.baselines.heuristic_selector.apply_operation_candidate",
        fake_apply_operation_candidate,
    )
    monkeypatch.setattr(
        "MPC4plus.experiments.baselines.heuristic_selector.rebuild_phase4_state",
        fake_rebuild_phase4_state,
    )

    chosen = selector.select([candidate_a, candidate_b], state)
    assert chosen == candidate_b


def test_one_step_heuristic_prefers_fewer_responsible_components_on_tie(monkeypatch) -> None:
    selector = OneStepHeuristicSelector()
    state = _fake_state()

    candidate_a = OperationCandidate(
        op_type="op1",
        source_component_idx=0,
        source_satellite_idx=0,
        source_vertex=2,
        target_component_idx=1,
        target_vertex=4,
        added_edge=(2, 4),
        removed_edges=(),
        features={"expected_g_drop": 0.0},
        note="candidate_a",
    )
    candidate_b = OperationCandidate(
        op_type="op1",
        source_component_idx=0,
        source_satellite_idx=0,
        source_vertex=2,
        target_component_idx=1,
        target_vertex=5,
        added_edge=(2, 5),
        removed_edges=(),
        features={"expected_g_drop": 0.0},
        note="candidate_b",
    )

    def fake_apply_operation_candidate(C, candidate):
        return {candidate.added_edge}

    def fake_rebuild_phase4_state(G, H, M_C, C):
        if (2, 4) in C:
            # same critical count, worse responsible count
            metas = [
                _fake_meta(is_critical=True, is_responsible_component=True),
                _fake_meta(is_critical=False, is_responsible_component=True),
            ]
        else:
            # same critical count, better responsible count
            metas = [
                _fake_meta(is_critical=True, is_responsible_component=False),
                _fake_meta(is_critical=False, is_responsible_component=False),
            ]
        return SimpleNamespace(metas=metas)

    monkeypatch.setattr(
        "MPC4plus.experiments.baselines.heuristic_selector.apply_operation_candidate",
        fake_apply_operation_candidate,
    )
    monkeypatch.setattr(
        "MPC4plus.experiments.baselines.heuristic_selector.rebuild_phase4_state",
        fake_rebuild_phase4_state,
    )

    chosen = selector.select([candidate_a, candidate_b], state)
    assert chosen == candidate_b


def test_one_step_heuristic_uses_expected_g_drop_as_third_priority(monkeypatch) -> None:
    selector = OneStepHeuristicSelector()
    state = _fake_state()

    candidate_a = OperationCandidate(
        op_type="op2",
        source_component_idx=0,
        source_satellite_idx=0,
        source_vertex=7,
        target_component_idx=1,
        target_vertex=8,
        added_edge=(7, 8),
        removed_edges=(),
        features={"expected_g_drop": 0.0},
        note="candidate_a",
    )
    candidate_b = OperationCandidate(
        op_type="op2",
        source_component_idx=0,
        source_satellite_idx=0,
        source_vertex=7,
        target_component_idx=1,
        target_vertex=9,
        added_edge=(7, 9),
        removed_edges=(),
        features={"expected_g_drop": 2.0},
        note="candidate_b",
    )

    def fake_apply_operation_candidate(C, candidate):
        return {candidate.added_edge}

    def fake_rebuild_phase4_state(G, H, M_C, C):
        # same critical/responsible result for both candidates
        return SimpleNamespace(
            metas=[
                _fake_meta(is_critical=True, is_responsible_component=False),
                _fake_meta(is_critical=False, is_responsible_component=False),
            ]
        )

    monkeypatch.setattr(
        "MPC4plus.experiments.baselines.heuristic_selector.apply_operation_candidate",
        fake_apply_operation_candidate,
    )
    monkeypatch.setattr(
        "MPC4plus.experiments.baselines.heuristic_selector.rebuild_phase4_state",
        fake_rebuild_phase4_state,
    )

    chosen = selector.select([candidate_a, candidate_b], state)
    assert chosen == candidate_b


def test_one_step_heuristic_score_is_lexicographic() -> None:
    s1 = OneStepHeuristicScore(
        num_critical_after=1,
        num_responsible_after=2,
        neg_expected_g_drop=-1.0,
        candidate_tiebreak=(0, 1),
    )
    s2 = OneStepHeuristicScore(
        num_critical_after=2,
        num_responsible_after=0,
        neg_expected_g_drop=-10.0,
        candidate_tiebreak=(0, 0),
    )
    assert s1 < s2