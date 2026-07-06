from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from MPC4plus.core.graph_utils import normalize_edge
from MPC4plus.phase3.component_meta import ComponentMeta, SatelliteInfo
from MPC4plus.phase4.operations import OperationCandidate
from MPC4plus.experiments.export.phase4_trace_schema import (
    Phase4CandidateRecord,
    Phase4ComponentRecord,
    Phase4DecisionRecord,
    Phase4DecisionSample,
    Phase4OutcomeRecord,
    Phase4SatelliteRecord,
    Phase4StateSummary,
)


class Phase4TraceCollector:
    def __init__(self) -> None:
        self._graph_id: str | None = None
        self._samples: list[Phase4DecisionSample] = []

    def start_graph(self, graph_id: str) -> None:
        self._graph_id = graph_id

    def clear(self) -> None:
        self._graph_id = None
        self._samples = []

    def samples(self) -> list[Phase4DecisionSample]:
        return list(self._samples)

    def to_dicts(self) -> list[dict]:
        return [asdict(sample) for sample in self._samples]

    def dump_jsonl(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for record in self.to_dicts():
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")

    def record_decision(
        self,
        *,
        state_before,
        candidates: Sequence[OperationCandidate],
        selected_candidate: OperationCandidate | None,
        state_after,
        selector_name: str,
        terminated_after_apply: bool | None = None,
    ) -> None:
        if self._graph_id is None:
            raise RuntimeError(
                "Phase4TraceCollector.start_graph(graph_id) must be called before record_decision(...)."
            )

        state_summary = self._build_state_summary(
            state_before=state_before,
            candidates=candidates,
        )
        components = self._build_component_records(state_before.metas)
        candidate_records, selected_candidate_idx = self._build_candidate_records(
            candidates=candidates,
            selected_candidate=selected_candidate,
            selector_name=selector_name,
        )
        decision = self._build_decision_record(
            selector_name=selector_name,
            selected_candidate_idx=selected_candidate_idx,
        )
        outcome = self._build_outcome_record(
            state_before=state_before,
            state_after=state_after,
            terminated_after_apply=terminated_after_apply,
        )
        sample_id = f"{self._graph_id}_it_{state_before.iteration:04d}"

        self._samples.append(
            Phase4DecisionSample(
                sample_id=sample_id,
                graph_id=self._graph_id,
                state=state_summary,
                components=components,
                candidates=candidate_records,
                decision=decision,
                outcome=outcome,
            )
        )

    def _build_state_summary(self, *, state_before, candidates: Sequence[OperationCandidate]) -> Phase4StateSummary:
        num_candidates_op1 = sum(1 for c in candidates if c.op_type == "op1")
        num_candidates_op2 = sum(1 for c in candidates if c.op_type == "op2")
        num_candidates_op3 = sum(1 for c in candidates if c.op_type == "op3")
        return Phase4StateSummary(
            iteration=state_before.iteration,
            num_nodes=state_before.G.number_of_nodes(),
            num_edges=state_before.G.number_of_edges(),
            num_H_edges=state_before.H.number_of_edges(),
            num_C_edges=len(state_before.C),
            num_M_C_edges=len(state_before.M_C),
            num_components=len(state_before.metas),
            num_critical_components=self._count_critical_components(state_before.metas),
            num_responsible_components=self._count_responsible_components(state_before.metas),
            num_candidates_total=len(candidates),
            num_candidates_op1=num_candidates_op1,
            num_candidates_op2=num_candidates_op2,
            num_candidates_op3=num_candidates_op3,
            history_length=len(state_before.history),
            H_edges=[normalize_edge(*e) for e in sorted(state_before.H.edges())],
            C_edges=[normalize_edge(*e) for e in sorted(state_before.C)],
            M_C_edges=[normalize_edge(*e) for e in sorted(state_before.M_C)],
        )

    def _build_component_records(self, metas: Sequence[ComponentMeta]) -> list[Phase4ComponentRecord]:
        records: list[Phase4ComponentRecord] = []
        for idx, meta in enumerate(metas):
            records.append(
                Phase4ComponentRecord(
                    component_idx=idx,
                    nodes=sorted(meta.nodes),
                    is_composite=meta.is_composite,
                    center_nodes=sorted(meta.center_nodes) if meta.center_nodes is not None else [],
                    center_kind=meta.center_kind,
                    anchors=sorted(meta.anchors),
                    j_anchor_map=dict(sorted(meta.j_anchor_map.items())),
                    critical_2_anchors=sorted(meta.critical_2_anchors),
                    responsible_1_anchors=sorted(meta.responsible_1_anchors),
                    s_value=meta.s_value,
                    opt_value=meta.opt_value,
                    critical_ratio=meta.critical_ratio,
                    is_critical=meta.is_critical,
                    critical_case=meta.critical_case,
                    critical_reason=meta.critical_reason,
                    critical_signature=dict(meta.critical_signature),
                    is_responsible_component=meta.is_responsible_component,
                    responsible_case=meta.responsible_case,
                    responsible_reason=meta.responsible_reason,
                    responsible_signature=dict(meta.responsible_signature),
                    satellites=self._build_satellite_records(meta.satellites),
                    H_component_node_sets=[
                        sorted(nodes) for nodes in getattr(meta, "H_component_node_sets", [])
                    ],
                )
            )
        return records

    def _build_satellite_records(self, satellites: Sequence[SatelliteInfo]) -> list[Phase4SatelliteRecord]:
        records: list[Phase4SatelliteRecord] = []
        for idx, sat in enumerate(satellites):
            records.append(
                Phase4SatelliteRecord(
                    satellite_idx=idx,
                    nodes=sorted(sat.nodes),
                    kind=sat.kind,
                    rescue_edges=[normalize_edge(*e) for e in sat.rescue_edges],
                    supporting_anchors=sorted(sat.supporting_anchors),
                    rescue_anchor=sat.rescue_anchor,
                    is_critical_satellite=sat.is_critical_satellite,
                )
            )
        return records

    def _build_candidate_records(
        self,
        *,
        candidates: Sequence[OperationCandidate],
        selected_candidate: OperationCandidate | None,
        selector_name: str,
    ) -> tuple[list[Phase4CandidateRecord], int | None]:
        records: list[Phase4CandidateRecord] = []
        selected_candidate_idx: int | None = None

        for idx, candidate in enumerate(candidates):
            is_selected = candidate == selected_candidate
            if is_selected:
                selected_candidate_idx = idx

            labels: dict[str, float | int | bool] = {
                "selected_in_run": int(is_selected),
            }
            if selector_name == "canonical_rule":
                labels["selected_by_rule"] = int(is_selected)

            records.append(
                Phase4CandidateRecord(
                    candidate_idx=idx,
                    op_type=candidate.op_type,
                    source_component_idx=candidate.source_component_idx,
                    source_satellite_idx=candidate.source_satellite_idx,
                    source_vertex=candidate.source_vertex,
                    target_component_idx=candidate.target_component_idx,
                    target_vertex=candidate.target_vertex,
                    added_edge=normalize_edge(*candidate.added_edge),
                    removed_edges=[normalize_edge(*e) for e in candidate.removed_edges],
                    features=dict(candidate.features),
                    note=candidate.note,
                    labels=labels,
                )
            )
        return records, selected_candidate_idx

    def _build_decision_record(
        self,
        *,
        selector_name: str,
        selected_candidate_idx: int | None,
    ) -> Phase4DecisionRecord:
        return Phase4DecisionRecord(
            selector_name=selector_name,
            selected_candidate_idx=selected_candidate_idx,
        )

    def _build_outcome_record(
        self,
        *,
        state_before,
        state_after,
        terminated_after_apply: bool | None,
    ) -> Phase4OutcomeRecord:
        return Phase4OutcomeRecord(
            num_critical_before=self._count_critical_components(state_before.metas),
            num_critical_after=self._count_critical_components(state_after.metas),
            num_responsible_before=self._count_responsible_components(state_before.metas),
            num_responsible_after=self._count_responsible_components(state_after.metas),
            num_C_edges_before=len(state_before.C),
            num_C_edges_after=len(state_after.C),
            terminated_after_apply=terminated_after_apply,
        )

    @staticmethod
    def _count_critical_components(metas: Sequence[ComponentMeta]) -> int:
        return sum(1 for meta in metas if meta.is_critical)

    @staticmethod
    def _count_responsible_components(metas: Sequence[ComponentMeta]) -> int:
        return sum(1 for meta in metas if meta.is_responsible_component)