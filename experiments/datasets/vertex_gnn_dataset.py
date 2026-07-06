from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from MPC4plus.experiments.models.vertex_gnn_selector import CandidateGraphSpec


@dataclass(frozen=True)
class VertexGNNTrainingExample:
    """
    One graph-aware Phase 4 decision example for vertex-level GNN training.

    This corresponds to one decision point:
      - one current graph/state
      - a list of candidate specs
      - one positive candidate label (or a general label vector)
    """
    graph_id: str
    sample_id: str
    iteration: int
    node_list: list[int]
    node_features: torch.Tensor          # [N, D]
    adj_norm: torch.Tensor               # [N, N]
    candidate_specs: list[CandidateGraphSpec]
    labels: torch.Tensor                 # [K]
    selected_candidate_idx: int | None


class VertexGNNTrainingDataset:
    """
    Lightweight in-memory dataset for vertex-level GNN training examples.
    """

    def __init__(self, examples: Iterable[VertexGNNTrainingExample]) -> None:
        self._examples = list(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> VertexGNNTrainingExample:
        return self._examples[idx]

    def examples(self) -> list[VertexGNNTrainingExample]:
        return list(self._examples)

    def num_examples(self) -> int:
        return len(self._examples)

    def num_positive_examples(self) -> int:
        total = 0
        for ex in self._examples:
            total += int((ex.labels > 0).sum().item())
        return total

    def summary(self) -> dict[str, Any]:
        if not self._examples:
            return {
                "num_examples": 0,
                "num_positive_labels": 0,
                "avg_num_candidates": 0.0,
                "node_feature_dim": 0,
            }

        avg_num_candidates = sum(len(ex.candidate_specs) for ex in self._examples) / len(self._examples)
        node_feature_dim = int(self._examples[0].node_features.shape[1])

        return {
            "num_examples": len(self._examples),
            "num_positive_labels": self.num_positive_examples(),
            "avg_num_candidates": avg_num_candidates,
            "node_feature_dim": node_feature_dim,
        }


def build_vertex_gnn_example_from_live_inputs(
    *,
    graph_id: str,
    sample_id: str,
    iteration: int,
    live_inputs: dict[str, Any],
    selected_candidate_idx: int | None,
) -> VertexGNNTrainingExample:
    """
    Convert live graph inputs (as produced by build_live_phase4_graph_inputs)
    into a training example.

    Labels are currently one-hot over candidates:
      - 1 at selected_candidate_idx
      - 0 elsewhere

    This aligns with first-stage supervision using rule labels.
    """
    node_list = list(live_inputs["node_list"])
    node_features = live_inputs["node_features"]
    adj_norm = live_inputs["adj_norm"]
    candidate_specs = list(live_inputs["candidate_specs"])

    num_candidates = len(candidate_specs)
    labels = torch.zeros(num_candidates, dtype=torch.float32)
    if selected_candidate_idx is not None:
        if not (0 <= selected_candidate_idx < num_candidates):
            raise ValueError(
                f"selected_candidate_idx={selected_candidate_idx} out of range for {num_candidates} candidates."
            )
        labels[selected_candidate_idx] = 1.0

    return VertexGNNTrainingExample(
        graph_id=graph_id,
        sample_id=sample_id,
        iteration=iteration,
        node_list=node_list,
        node_features=node_features,
        adj_norm=adj_norm,
        candidate_specs=candidate_specs,
        labels=labels,
        selected_candidate_idx=selected_candidate_idx,
    )


def batch_vertex_gnn_examples(
    examples: list[VertexGNNTrainingExample],
) -> list[VertexGNNTrainingExample]:
    """
    Placeholder batching helper.

    For this first GNN version, we keep batching simple:
    - one decision example at a time
    - return the list unchanged

    This keeps training logic straightforward and avoids premature complexity
    around variable-size graph batching.
    """
    return list(examples)