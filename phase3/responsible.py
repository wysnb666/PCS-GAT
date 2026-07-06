from __future__ import annotations

from MPC4plus.phase3.component_meta import ComponentMeta


def _anchor_type_counts(meta: ComponentMeta) -> tuple[int, int, int]:
    n0 = 0
    n1 = 0
    n2 = 0
    for a in meta.anchors:
        j = meta.j_anchor_map.get(a, 0)
        if j == 0:
            n0 += 1
        elif j == 1:
            n1 += 1
        elif j == 2:
            n2 += 1
    return n0, n1, n2


def _build_responsible_signature(meta: ComponentMeta) -> dict[str, object]:
    n0, n1, n2 = _anchor_type_counts(meta)
    signature: dict[str, object] = {
        "center_kind": meta.center_kind,
        "num_nodes": len(meta.nodes),
        "num_anchors": len(meta.anchors),
        "num_satellites": len(meta.satellites),
        "n0": n0,
        "n1": n1,
        "n2": n2,
        "num_responsible_1_anchors": len(meta.responsible_1_anchors),
        "num_critical_2_anchors": len(meta.critical_2_anchors),
        "s_value": meta.s_value,
        "satellite_kinds": tuple(sorted(s.kind for s in meta.satellites)),
    }
    return signature


def _match_responsible_catalog(signature: dict[str, object]) -> tuple[bool, str | None, str]:
    """
    Project-side explicit responsible-component catalog.

    This is a structured-signature recognizer aligned with the paper's discussion:
    responsible but not critical components arise from critical structures after
    removing a critical satellite-element (Figure 2 / Lemma 14 discussion).

    The catalog below is explicit at the signature level rather than a full graph-
    isomorphism recognizer of every Figure-2 drawing.
    """
    center_kind = signature["center_kind"]
    n2 = int(signature["n2"])
    num_resp1 = int(signature["num_responsible_1_anchors"])
    s_value = int(signature["s_value"])

    if num_resp1 == 0:
        return False, None, "no responsible 1-anchor"

    # First-row responsible descendants: one responsible 1-anchor, non-critical
    if num_resp1 == 1 and n2 == 0:
        if center_kind == "edge" and s_value in {6, 8}:
            return True, "fig2-line1-edge", "matched explicit responsible edge catalog"
        if center_kind == "five_path" and s_value in {8, 10}:
            return True, "fig2-line1-five-path", "matched explicit responsible 5-path catalog"
        return True, "responsible-one-anchor-generic", "matched generic one-responsible-anchor catalog"

    # Mixed case: one responsible 1-anchor together with one 2-anchor
    if num_resp1 == 1 and n2 == 1:
        if center_kind in {"edge", "five_path"} and s_value in {10, 12, 14, 16}:
            return True, "fig2-mixed-1resp-1two", "matched mixed responsible/2-anchor catalog"
        return True, "responsible-mixed-generic", "matched generic mixed responsible/2-anchor catalog"

    # Two responsible 1-anchors, no 2-anchor
    if num_resp1 == 2 and n2 == 0:
        return True, "fig2-two-responsible", "matched two-responsible-anchor catalog"

    # One 2-anchor and two responsible 1-anchors
    if num_resp1 == 2 and n2 == 1:
        return True, "fig2-one-2anchor-two-responsible", "matched high-order responsible catalog"

    return True, "responsible-generic", "matched fallback responsible catalog"


def mark_responsible_components(metas: list[ComponentMeta]) -> list[ComponentMeta]:
    """
    Mark responsible components explicitly.

    By the paper's convention after Lemma 14, a component is treated as responsible
    only when it is responsible and not critical.
    """
    for meta in metas:
        meta.responsible_signature = _build_responsible_signature(meta)
        meta.is_responsible_component = False
        meta.responsible_case = None
        meta.responsible_reason = ""

        if meta.is_critical:
            meta.responsible_reason = "critical components are not marked responsible at this stage"
            continue

        possible, case, reason = _match_responsible_catalog(meta.responsible_signature)
        if possible:
            meta.is_responsible_component = True
            meta.responsible_case = case
            meta.responsible_reason = reason
        else:
            meta.responsible_reason = reason

    return metas