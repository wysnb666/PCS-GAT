from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from MPC4plus.experiments.baselines.xgb_ranker import XGBPhase4Ranker
from MPC4plus.experiments.datasets.candidate_table_dataset import CandidateTableDataset
from MPC4plus.phase4.operations import OperationCandidate, OperationSelector, Phase4State


@dataclass(frozen=True)
class XGBSelectorPrediction:
    candidate_idx: int
    score: float


class XGBPhase4Selector(OperationSelector):
    """
    Wrap a trained XGBPhase4Ranker as a real Phase 4 selector.

    Current design choice:
    - Build a temporary candidate-level table for the current decision point
    - Use the fitted XGBoost model to score all candidates
    - Select the top-1 candidate
    - Break score ties deterministically

    This keeps the interface compatible with the existing Phase 4 selector API:
        select(candidates, state) -> OperationCandidate | None
    """

    _priority = {
        "op1": 0,
        "op2": 1,
        "op3": 2,
    }

    def __init__(self, ranker: XGBPhase4Ranker) -> None:
        self.ranker = ranker

    def select(
        self,
        candidates: Sequence[OperationCandidate],
        state: Phase4State,
    ) -> OperationCandidate | None:
        if not candidates:
            return None

        dataset = self._build_live_dataset(candidates, state)
        scores = self.ranker.predict_scores(dataset)

        if len(scores) != len(candidates):
            raise RuntimeError("Predicted score count does not match candidate count.")

        best_idx = self._argmax_with_tiebreak(candidates, scores)
        return candidates[best_idx]

    def predict_top1(self, candidates: Sequence[OperationCandidate], state: Phase4State) -> XGBSelectorPrediction | None:
        if not candidates:
            return None

        dataset = self._build_live_dataset(candidates, state)
        scores = self.ranker.predict_scores(dataset)
        best_idx = self._argmax_with_tiebreak(candidates, scores)
        return XGBSelectorPrediction(candidate_idx=best_idx, score=float(scores[best_idx]))

    def _build_live_dataset(
        self,
        candidates: Sequence[OperationCandidate],
        state: Phase4State,
    ) -> CandidateTableDataset:
        """
        Convert the current live decision point into a minimal candidate table.

        Important current design constraint:
        - the first-stage XGB training currently uses only dataset.feature_names()
        - and feature_names are inferred from columns prefixed with "feature__"
        - so for the first usable selector version, we only need to expose
          feature__ columns derived from candidate.features

        We still add a few metadata fields for consistency/debugging.
        """
        sample_id = f"live_it_{state.iteration}"
        rows: list[dict] = []

        for idx, candidate in enumerate(candidates):
            row = {
                "graph_id": "live_graph",
                "sample_id": sample_id,
                "iteration": state.iteration,
                "selector_name": "xgb_live_selector",
                "candidate_idx": idx,
                "op_type": candidate.op_type,
                "source_component_idx": candidate.source_component_idx,
                "source_satellite_idx": candidate.source_satellite_idx,
                "source_vertex": candidate.source_vertex,
                "target_component_idx": candidate.target_component_idx,
                "target_vertex": candidate.target_vertex,
            }

            for key, value in candidate.features.items():
                row[f"feature__{key}"] = value

            rows.append(row)

        return CandidateTableDataset(rows)

    def _argmax_with_tiebreak(
        self,
        candidates: Sequence[OperationCandidate],
        scores: Sequence[float],
    ) -> int:
        best_idx = 0
        best_key = self._candidate_rank_key(candidates[0], float(scores[0]))

        for idx in range(1, len(candidates)):
            key = self._candidate_rank_key(candidates[idx], float(scores[idx]))
            if key > best_key:
                best_key = key
                best_idx = idx

        return best_idx

    def _candidate_rank_key(self, candidate: OperationCandidate, score: float) -> tuple:
        """
        Maximize score first, then break ties deterministically.

        We use a stable fallback ordering aligned with the existing selector style.
        Since Python compares tuples lexicographically in ascending order, and
        we want argmax, we place higher score first and then prefer "smaller"
        structural keys by negating the comparison direction via argmax logic:
        larger score wins; for exact equal score, the tuple parts after score are
        negated conceptually by using reversed preference through ordering below.

        Concretely:
        - score: larger is better
        - for equal score, smaller priority/index tuple is better
        So we encode fallback terms with negative-style ordering by wrapping them
        in a tuple and comparing (score, reversed-order surrogate).
        """
        fallback = (
            -self._priority.get(candidate.op_type, 99),
            -candidate.source_component_idx,
            -candidate.source_satellite_idx,
            -candidate.source_vertex,
            -candidate.target_component_idx,
            -candidate.target_vertex,
            self._edge_sort_key(candidate.added_edge),
            self._removed_edges_sort_key(candidate.removed_edges),
        )
        return (float(score), fallback)

    @staticmethod
    def _edge_sort_key(edge: tuple[int, int]) -> tuple[int, int]:
        # smaller edge should win under tie; encode as negative surrogate through argmax usage
        return (-edge[0], -edge[1])

    @staticmethod
    def _removed_edges_sort_key(edges: tuple[tuple[int, int], ...]) -> tuple:
        # prefer lexicographically smaller removed-edge tuples under tie
        return tuple((-u, -v) for (u, v) in edges)