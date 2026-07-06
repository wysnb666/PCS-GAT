from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SatelliteInfo:
    nodes: set[int]
    kind: str

    # Edges in C connecting this satellite to the center anchors.
    rescue_edges: list[tuple[int, int]] = field(default_factory=list)

    # Anchor endpoints of the rescue-edges.
    supporting_anchors: list[int] = field(default_factory=list)

    # Structured fields used by Phase 3 / Phase 4 / Phase 5.
    rescue_anchor: Optional[int] = None
    is_critical_satellite: bool = False


@dataclass
class ComponentMeta:
    """
    Metadata for one connected component K of H + C.
    """
    nodes: set[int]
    is_composite: bool
    H_component_node_sets: list[set[int]]

    center_nodes: Optional[set[int]] = None
    center_kind: Optional[str] = None

    satellites: list[SatelliteInfo] = field(default_factory=list)

    anchors: list[int] = field(default_factory=list)
    j_anchor_map: dict[int, int] = field(default_factory=dict)

    # Critical-side metadata
    critical_2_anchors: set[int] = field(default_factory=set)
    responsible_1_anchors: set[int] = field(default_factory=set)

    s_value: int = 0
    opt_value: int = 0
    critical_ratio: float = 0.0

    is_critical: bool = False
    critical_case: Optional[str] = None
    critical_reason: str = ""
    critical_signature: dict[str, object] = field(default_factory=dict)

    # Responsible-side metadata
    is_responsible_component: bool = False
    responsible_case: Optional[str] = None
    responsible_reason: str = ""
    responsible_signature: dict[str, object] = field(default_factory=dict)