from __future__ import annotations

import networkx as nx

from MPC4plus.phase3.anchors import (
    enrich_meta_with_anchor_info,
    enrich_metas_with_critical_satellites,
    enrich_metas_with_responsible_1_anchors,
)
from MPC4plus.phase3.composite_components import build_all_component_meta
from MPC4plus.phase3.critical import mark_critical_components
from MPC4plus.phase3.h_plus_c import build_H_plus_C
from MPC4plus.phase3.responsible import mark_responsible_components
from MPC4plus.phase4.candidate_enumerator import enumerate_operation_candidates
from MPC4plus.phase4.operations import (
    CanonicalRuleSelector,
    GreedyOperationSelector,
    OperationSelector,
    Phase4State,
    apply_operation_candidate,
)


Edge = tuple[int, int]


def rebuild_phase4_state(
    G: nx.Graph,
    H: nx.Graph,
    M_C: set[Edge],
    C: set[Edge],
) -> Phase4State:
    HC = build_H_plus_C(H, C)
    metas = build_all_component_meta(HC, H)
    metas = [enrich_meta_with_anchor_info(meta, H, HC) for meta in metas]
    metas = mark_critical_components(metas, HC, M_C)
    metas = enrich_metas_with_critical_satellites(metas)
    metas = enrich_metas_with_responsible_1_anchors(G, H, C, M_C, metas)
    metas = mark_responsible_components(metas)

    return Phase4State(
        G=G,
        H=H,
        M_C=set(M_C),
        C=set(C),
        HC=HC,
        metas=metas,
    )


def _phase4_state_signature(state: Phase4State) -> tuple:
    """
    Phase 4 only edits C while G/H/M_C remain fixed for one run.
    Therefore an exact signature of the current C-set is enough to detect
    repeated states inside one Phase 4 trajectory.
    """
    return tuple(sorted(state.C))


def run_phase4_operations(
    G: nx.Graph,
    H: nx.Graph,
    M_C: set[Edge],
    C: set[Edge],
    selector: OperationSelector | None = None,
    max_rounds: int | None = None,
    trace_collector=None,
    detect_repeated_states: bool = True,
) -> Phase4State:
    """
    Repeatedly enumerate -> select -> apply until no candidate exists.

    H and M_C stay fixed; only C is edited in Phase 4.

    Safety guard:
    - when detect_repeated_states=True, terminate early if the next C-state
      has already appeared before in this Phase 4 run.
    """
    if selector is None:
        selector = CanonicalRuleSelector()
    if max_rounds is None:
        max_rounds = max(1, 5 * G.number_of_nodes())

    state = rebuild_phase4_state(G, H, M_C, C)
    state.cycle_detected = False
    state.termination_reason = None
    state.repeated_state_first_seen_iteration = None

    visited_signatures: dict[tuple, int] = {
        _phase4_state_signature(state): state.iteration
    }

    for _ in range(max_rounds):
        candidates = enumerate_operation_candidates(state.G, state.H, state.C, state.metas)
        if not candidates:
            state.termination_reason = "no_candidate"
            break

        chosen = selector.select(candidates, state)
        if chosen is None:
            state.termination_reason = "selector_returned_none"
            break

        new_C = apply_operation_candidate(state.C, chosen)
        if new_C == state.C:
            state.termination_reason = "no_change_after_apply"
            break

        new_state = rebuild_phase4_state(state.G, state.H, state.M_C, new_C)
        new_state.iteration = state.iteration + 1
        new_state.history = list(state.history) + [chosen]

        new_sig = _phase4_state_signature(new_state)
        if detect_repeated_states and new_sig in visited_signatures:
            state.cycle_detected = True
            state.repeated_state_first_seen_iteration = visited_signatures[new_sig]
            state.termination_reason = (
                "repeat_state_detected("
                f"next_iteration={new_state.iteration}, "
                f"first_seen_iteration={visited_signatures[new_sig]}"
                ")"
            )
            break

        if trace_collector is not None:
            selector_name = (
                "canonical_rule"
                if isinstance(selector, CanonicalRuleSelector)
                else selector.__class__.__name__
            )
            trace_collector.record_decision(
                state_before=state,
                candidates=candidates,
                selected_candidate=chosen,
                state_after=new_state,
                selector_name=selector_name,
                terminated_after_apply=None,
            )

        state = new_state
        visited_signatures[new_sig] = state.iteration

    else:
        state.termination_reason = f"max_rounds_reached({max_rounds})"

    return state