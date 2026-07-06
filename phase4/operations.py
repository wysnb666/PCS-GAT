from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import networkx as nx

from MPC4plus.core.graph_utils import normalize_edge
from MPC4plus.phase3.component_meta import ComponentMeta


Edge = tuple[int, int]


@dataclass(frozen=True)
class OperationCandidate:
    """
    One legal candidate operation in Phase 4.

    Fields:
    - op_type: "op1" / "op2" / "op3"
    - source_component_idx: index of current critical source component
    - source_satellite_idx: index of the source critical satellite in that component
    - source_vertex: endpoint chosen inside source satellite
    - target_component_idx: index of target component
    - target_vertex: endpoint chosen outside source satellite
    - added_edge: the new edge added into C
    - removed_edges: edge(s) removed from C
    - features: reserved for future ML ranking
    - note: human-readable explanation for debugging / tests
    """
    op_type: str
    source_component_idx: int
    source_satellite_idx: int
    source_vertex: int
    target_component_idx: int
    target_vertex: int
    added_edge: Edge
    removed_edges: tuple[Edge, ...]
    features: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "added_edge", normalize_edge(*self.added_edge))
        object.__setattr__(
            self,
            "removed_edges",
            tuple(normalize_edge(*e) for e in self.removed_edges),
        )


@dataclass
class Phase4State:
    """
    Runtime state during repeated Phase 4 operations.
    """
    G: nx.Graph
    H: nx.Graph
    M_C: set[Edge]
    C: set[Edge]
    HC: nx.Graph
    metas: list[ComponentMeta]
    iteration: int = 0
    history: list[OperationCandidate] = field(default_factory=list)

    # runtime diagnostics / termination info
    cycle_detected: bool = False
    termination_reason: str | None = None
    repeated_state_first_seen_iteration: int | None = None


class OperationSelector(Protocol):
    def select(
        self,
        candidates: Sequence[OperationCandidate],
        state: Phase4State,
    ) -> OperationCandidate | None:
        ...


class CanonicalRuleSelector:
    """
    Deterministic rule baseline selector.

    The paper requires repeated application of legal operations until none is
    applicable, but does not enforce a unique tie-breaking rule among multiple
    legal candidates. So here we use a stable deterministic baseline, which can
    later be replaced by an ML scorer/ranker.

    Note: this class intentionally preserves the current project behavior. The
    tie-break still uses the existing priority/feature ordering; this change is
    primarily a naming/interface upgrade for Phase 4 trace export.
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

        def key(c: OperationCandidate):
            return (
                self._priority.get(c.op_type, 99),
                -float(c.features.get("expected_g_drop", 0.0)),
                c.source_component_idx,
                c.source_satellite_idx,
                c.source_vertex,
                c.target_component_idx,
                c.target_vertex,
                c.added_edge,
                c.removed_edges,
            )

        return sorted(candidates, key=key)[0]


# Backward-compatible alias. Existing code/tests may still import this name.
GreedyOperationSelector = CanonicalRuleSelector


def apply_operation_candidate(
    C: set[Edge],
    candidate: OperationCandidate,
) -> set[Edge]:
    """
    Apply one candidate to C only.

    Phase 4 edits only C; H and M_C remain fixed.
    """
    new_C = set(C)
    for e in candidate.removed_edges:
        new_C.discard(normalize_edge(*e))
    new_C.add(normalize_edge(*candidate.added_edge))
    return new_C