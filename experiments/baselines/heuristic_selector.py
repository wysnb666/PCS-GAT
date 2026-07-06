from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from MPC4plus.phase4.operation_runner import rebuild_phase4_state
from MPC4plus.phase4.operations import (
    OperationCandidate,
    OperationSelector,
    Phase4State,
    apply_operation_candidate,
)


@dataclass(frozen=True, order=True)
class OneStepHeuristicScore:
    """
    Lexicographic score for one-step lookahead heuristic.

    Smaller is better.
    """
    num_critical_after: int
    num_responsible_after: int
    neg_expected_g_drop: float
    candidate_tiebreak: tuple


class OneStepHeuristicSelector(OperationSelector):
    """
    One-step lookahead heuristic baseline.

    For each legal Phase 4 candidate:
      1) apply it once to C
      2) rebuild the next Phase 4 state
      3) compare the resulting local structure

    Priority:
      (1) fewer critical components after apply
      (2) fewer responsible components after apply
      (3) larger expected_g_drop from existing candidate feature
      (4) deterministic fallback tie-break

    This is intentionally simple, interpretable, and non-learning.
    """

    _priority = {
        "op1": 0,
        "op2": 1,
        "op3": 2,
    }

    def select(
        self,
        candidates: Sequence[OperationCandidate],
        state: Phase4State,
    ) -> OperationCandidate | None:
        if not candidates:
            return None

        best_candidate: OperationCandidate | None = None
        best_score: OneStepHeuristicScore | None = None

        for candidate in candidates:
            score = self._score_candidate(state, candidate)
            if best_score is None or score < best_score:
                best_candidate = candidate
                best_score = score

        return best_candidate

    def _score_candidate(
        self,
        state: Phase4State,
        candidate: OperationCandidate,
    ) -> OneStepHeuristicScore:
        new_C = apply_operation_candidate(state.C, candidate)
        next_state = rebuild_phase4_state(
            G=state.G,
            H=state.H,
            M_C=state.M_C,
            C=new_C,
        )

        num_critical_after = sum(1 for meta in next_state.metas if meta.is_critical)
        num_responsible_after = sum(
            1 for meta in next_state.metas if meta.is_responsible_component
        )
        neg_expected_g_drop = -float(candidate.features.get("expected_g_drop", 0.0))

        return OneStepHeuristicScore(
            num_critical_after=num_critical_after,
            num_responsible_after=num_responsible_after,
            neg_expected_g_drop=neg_expected_g_drop,
            candidate_tiebreak=self._candidate_tiebreak_key(candidate),
        )

    def _candidate_tiebreak_key(self, candidate: OperationCandidate) -> tuple:
        """
        Deterministic fallback order.

        We intentionally keep this aligned with the current canonical/rule
        selector style so that the heuristic differs mainly in the one-step
        structural lookahead, not in arbitrary randomness.
        """
        return (
            self._priority.get(candidate.op_type, 99),
            candidate.source_component_idx,
            candidate.source_satellite_idx,
            candidate.source_vertex,
            candidate.target_component_idx,
            candidate.target_vertex,
            candidate.added_edge,
            candidate.removed_edges,
        )