from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from MPC4plus.phase3.component_meta import ComponentMeta


@dataclass
class Phase5Buckets:
    """
    Notation 7 / 8 style buckets.
    """
    R: set[int] = field(default_factory=set)
    K: list[ComponentMeta] = field(default_factory=list)

    K0: list[ComponentMeta] = field(default_factory=list)
    K1: list[ComponentMeta] = field(default_factory=list)
    K2: list[ComponentMeta] = field(default_factory=list)
    K3: list[ComponentMeta] = field(default_factory=list)
    K4: list[ComponentMeta] = field(default_factory=list)
    K5: list[ComponentMeta] = field(default_factory=list)

    K1c: list[ComponentMeta] = field(default_factory=list)
    K2c: list[ComponentMeta] = field(default_factory=list)

    Rc: set[int] = field(default_factory=set)
    Uc: set[int] = field(default_factory=set)
    Gc_nodes: set[int] = field(default_factory=set)

    responsible_components: list[ComponentMeta] = field(default_factory=list)


def is_isolated_bad_component_of_H(meta: ComponentMeta) -> bool:
    return (not meta.is_composite) and meta.center_kind in {"edge", "triangle", "star"}


def is_isolated_five_path(meta: ComponentMeta) -> bool:
    return (not meta.is_composite) and meta.center_kind == "five_path"


def is_responsible_component(meta: ComponentMeta) -> bool:
    return meta.is_responsible_component


def build_R(metas: list[ComponentMeta]) -> set[int]:
    R: set[int] = set()
    for meta in metas:
        for a in meta.anchors:
            if meta.j_anchor_map.get(a, 0) == 2:
                R.add(a)
        R.update(meta.responsible_1_anchors)
    return R


def _count_vertices_in_R(meta: ComponentMeta, R: set[int]) -> int:
    return len(meta.nodes & R)


def component_bucket_name(meta: ComponentMeta, buckets: Phase5Buckets) -> str:
    """
    Human-readable primary role for one component.

    Priority rule:
    1. critical buckets first (K1c / K2c)
    2. responsible-component label next
    3. generic Ki buckets afterwards
    4. isolated bad H-component if excluded from K
    """
    if meta in buckets.K1c:
        return "K1c"
    if meta in buckets.K2c:
        return "K2c"

    # Important:
    # a responsible component may also lie in K1 / K2 / ...
    # For display/debugging we prefer the more structural label.
    if meta in buckets.responsible_components:
        if meta.responsible_case:
            return f"responsible_component:{meta.responsible_case}"
        return "responsible_component"

    if meta in buckets.K0:
        return "K0"
    if meta in buckets.K1:
        return "K1"
    if meta in buckets.K2:
        return "K2"
    if meta in buckets.K3:
        return "K3"
    if meta in buckets.K4:
        return "K4"
    if meta in buckets.K5:
        return "K5"

    if is_isolated_bad_component_of_H(meta):
        return "isolated_bad_H"
    return "unclassified"


def summarize_phase5_buckets(buckets: Phase5Buckets) -> dict[str, int]:
    return {
        "K": len(buckets.K),
        "K0": len(buckets.K0),
        "K1": len(buckets.K1),
        "K2": len(buckets.K2),
        "K3": len(buckets.K3),
        "K4": len(buckets.K4),
        "K5": len(buckets.K5),
        "K1c": len(buckets.K1c),
        "K2c": len(buckets.K2c),
        "R": len(buckets.R),
        "Rc": len(buckets.Rc),
        "Uc": len(buckets.Uc),
        "Gc_nodes": len(buckets.Gc_nodes),
        "responsible_components": len(buckets.responsible_components),
    }


def classify_phase5_buckets(
    G: nx.Graph,
    metas: list[ComponentMeta],
) -> Phase5Buckets:
    buckets = Phase5Buckets()
    buckets.R = build_R(metas)

    for meta in metas:
        if is_responsible_component(meta):
            buckets.responsible_components.append(meta)

        if is_isolated_bad_component_of_H(meta):
            continue

        buckets.K.append(meta)
        count = _count_vertices_in_R(meta, buckets.R)

        if count == 0:
            buckets.K0.append(meta)
        elif count == 1:
            buckets.K1.append(meta)
            if meta.is_critical:
                buckets.K1c.append(meta)
        elif count == 2:
            buckets.K2.append(meta)
            if meta.is_critical:
                buckets.K2c.append(meta)
        elif count == 3:
            buckets.K3.append(meta)
        elif count == 4:
            buckets.K4.append(meta)
        elif count >= 5:
            buckets.K5.append(meta)

    for meta in metas:
        if not meta.is_critical:
            continue
        for a in meta.anchors:
            if meta.j_anchor_map.get(a, 0) == 2:
                buckets.Rc.add(a)

    for meta in metas:
        if not meta.is_critical:
            continue
        for sat in meta.satellites:
            if sat.is_critical_satellite and sat.rescue_anchor in buckets.Rc:
                buckets.Uc.update(sat.nodes)

    buckets.Gc_nodes = set(G.nodes()) - buckets.Rc - buckets.Uc
    return buckets