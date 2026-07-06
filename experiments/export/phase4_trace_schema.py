from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


Edge = tuple[int, int]


@dataclass
class Phase4SatelliteRecord:
    satellite_idx: int
    nodes: list[int]
    kind: str | None
    rescue_edges: list[Edge]
    supporting_anchors: list[int]
    rescue_anchor: int | None
    is_critical_satellite: bool


@dataclass
class Phase4ComponentRecord:
    component_idx: int
    nodes: list[int]
    is_composite: bool
    center_nodes: list[int]
    center_kind: str | None
    anchors: list[int]
    j_anchor_map: dict[int, int]
    critical_2_anchors: list[int]
    responsible_1_anchors: list[int]
    s_value: int | None
    opt_value: int | None
    critical_ratio: float | None
    is_critical: bool
    critical_case: str | None
    critical_reason: str | None
    critical_signature: dict[str, Any]
    is_responsible_component: bool
    responsible_case: str | None
    responsible_reason: str | None
    responsible_signature: dict[str, Any]
    satellites: list[Phase4SatelliteRecord]
    H_component_node_sets: list[list[int]] = field(default_factory=list)


@dataclass
class Phase4CandidateRecord:
    candidate_idx: int
    op_type: str
    source_component_idx: int
    source_satellite_idx: int
    source_vertex: int
    target_component_idx: int
    target_vertex: int
    added_edge: Edge
    removed_edges: list[Edge]
    features: dict[str, float | int | bool]
    note: str
    labels: dict[str, float | int | bool]


@dataclass
class Phase4StateSummary:
    iteration: int
    num_nodes: int
    num_edges: int
    num_H_edges: int
    num_C_edges: int
    num_M_C_edges: int
    num_components: int
    num_critical_components: int
    num_responsible_components: int
    num_candidates_total: int
    num_candidates_op1: int
    num_candidates_op2: int
    num_candidates_op3: int
    history_length: int
    H_edges: list[Edge] = field(default_factory=list)
    C_edges: list[Edge] = field(default_factory=list)
    M_C_edges: list[Edge] = field(default_factory=list)


@dataclass
class Phase4DecisionRecord:
    selector_name: str
    selected_candidate_idx: int | None


@dataclass
class Phase4OutcomeRecord:
    num_critical_before: int
    num_critical_after: int
    num_responsible_before: int
    num_responsible_after: int
    num_C_edges_before: int
    num_C_edges_after: int
    terminated_after_apply: bool | None


@dataclass
class Phase4DecisionSample:
    sample_id: str
    graph_id: str
    state: Phase4StateSummary
    components: list[Phase4ComponentRecord]
    candidates: list[Phase4CandidateRecord]
    decision: Phase4DecisionRecord
    outcome: Phase4OutcomeRecord


def sample_to_dict(sample: Phase4DecisionSample) -> dict[str, Any]:
    return asdict(sample)