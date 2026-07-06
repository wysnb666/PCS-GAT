from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import networkx as nx

from MPC4plus.experiments.export.phase4_trace_exporter import Phase4TraceCollector
from MPC4plus.phase1.modify_H_M import run_phase1
from MPC4plus.phase2.auxiliary_graph import build_auxiliary_graph_G1
from MPC4plus.phase2.matching_projection import build_M_C
from MPC4plus.phase2.path_cycle_cover import compute_max_weight_path_cycle_cover
from MPC4plus.phase4.operation_runner import run_phase4_operations
from MPC4plus.phase4.operations import OperationSelector


def _to_dict(obj: Any) -> dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return vars(obj)
    raise TypeError(f"Unsupported object type for dict conversion: {type(obj)}")


def candidate_key_from_dict(cand: dict[str, Any]) -> tuple:
    return (
        str(cand["op_type"]),
        int(cand["source_component_idx"]),
        int(cand["source_satellite_idx"]),
        int(cand["source_vertex"]),
        int(cand["target_component_idx"]),
        int(cand["target_vertex"]),
        tuple(int(x) for x in cand["added_edge"]),
        tuple(tuple(int(y) for y in e) for e in cand["removed_edges"]),
    )


def _canonical_op_priority(op_type: str) -> int:
    if op_type == "op1":
        return 0
    if op_type == "op2":
        return 1
    if op_type == "op3":
        return 2
    return 99


def _canonical_structural_tiebreak_key_from_dict(cand: dict[str, Any]) -> tuple:
    return (
        _canonical_op_priority(str(cand["op_type"])),
        int(cand["source_component_idx"]),
        int(cand["source_satellite_idx"]),
        int(cand["source_vertex"]),
        int(cand["target_component_idx"]),
        int(cand["target_vertex"]),
        tuple(int(x) for x in cand["added_edge"]),
        tuple(tuple(int(y) for y in e) for e in cand["removed_edges"]),
    )


def _canonical_rank_key_from_dict(cand: dict[str, Any]) -> tuple:
    features = cand.get("features", {})
    return (
        _canonical_op_priority(str(cand["op_type"])),
        -float(features.get("expected_g_drop", 0.0)),
        int(cand["source_component_idx"]),
        int(cand["source_satellite_idx"]),
        int(cand["source_vertex"]),
        int(cand["target_component_idx"]),
        int(cand["target_vertex"]),
        tuple(int(x) for x in cand["added_edge"]),
        tuple(tuple(int(y) for y in e) for e in cand["removed_edges"]),
    )


class PrefixThenForcedThenCanonicalSelector(OperationSelector):
    def __init__(
        self,
        *,
        prefix_keys: list[tuple],
        forced_key: tuple,
    ) -> None:
        self.prefix_keys = list(prefix_keys)
        self.forced_key = forced_key
        self.step_idx = 0

    def select(self, candidates, state):
        if not candidates:
            return None

        if self.step_idx < len(self.prefix_keys):
            key = self.prefix_keys[self.step_idx]
            self.step_idx += 1
            return _find_candidate_by_key(candidates, key)

        if self.step_idx == len(self.prefix_keys):
            self.step_idx += 1
            return _find_candidate_by_key(candidates, self.forced_key)

        self.step_idx += 1
        return _canonical_pick(candidates)


def _find_candidate_by_key(candidates, key: tuple):
    for cand in candidates:
        cand_dict = _candidate_obj_to_dict(cand)
        if candidate_key_from_dict(cand_dict) == key:
            return cand
    raise ValueError(f"Unable to find candidate with key={key!r} during replay.")


def _canonical_pick(candidates):
    best = None
    best_key = None
    for cand in candidates:
        cand_dict = _candidate_obj_to_dict(cand)
        rank_key = _canonical_rank_key_from_dict(cand_dict)
        if best is None or rank_key < best_key:
            best = cand
            best_key = rank_key
    return best


def _candidate_obj_to_dict(cand: Any) -> dict[str, Any]:
    if is_dataclass(cand):
        return asdict(cand)
    if isinstance(cand, dict):
        return cand
    if hasattr(cand, "__dict__"):
        return vars(cand)
    raise TypeError(f"Unsupported candidate object type: {type(cand)}")


def _replay_graph_with_selector(
    G: nx.Graph,
    *,
    selector: OperationSelector,
    backend: str,
    graph_id: str,
) -> dict[str, Any]:
    phase1 = run_phase1(G)
    G1 = build_auxiliary_graph_G1(G, phase1.H)
    C = compute_max_weight_path_cycle_cover(G1, phase1.H, backend=backend)
    M_C = build_M_C(phase1.M, phase1.H, C)

    collector = Phase4TraceCollector()
    collector.start_graph(graph_id)

    final_state = run_phase4_operations(
        G=G,
        H=phase1.H,
        M_C=M_C,
        C=C,
        selector=selector,
        trace_collector=collector,
    )

    return {
        "samples": [_to_dict(s) for s in collector.samples()],
        "termination_reason": final_state.termination_reason,
        "cycle_detected": bool(final_state.cycle_detected),
    }


def _lookahead_primary_score(
    sample_after_apply: dict[str, Any],
    candidate_dict: dict[str, Any],
) -> tuple:
    outcome = sample_after_apply["outcome"]
    features = candidate_dict.get("features", {})

    num_critical_after = int(outcome.get("num_critical_after", 10**9))
    num_responsible_after = int(outcome.get("num_responsible_after", 10**9))
    terminated_penalty = 0 if bool(outcome.get("terminated_after_apply", False)) else 1
    num_c_edges_after = int(outcome.get("num_C_edges_after", 10**9))
    expected_g_drop = float(features.get("expected_g_drop", 0.0))

    return (
        num_critical_after,
        num_responsible_after,
        terminated_penalty,
        num_c_edges_after,
        -expected_g_drop,
    )


def _invalid_primary_score() -> tuple:
    """
    Worst possible score used when replay fails to even reach the target step.
    Such a candidate should never be preferred over a valid replay candidate.
    """
    return (
        10**9,
        10**9,
        1,
        10**9,
        0.0,
    )


def _write_candidate_primary_score_features(
    cand: dict[str, Any],
    primary_score: tuple,
    *,
    replay_reached_target_step: int,
    replay_failed_before_target_step: int,
    replay_cycle_detected: int,
) -> None:
    features = cand.setdefault("features", {})
    features["lookahead_primary_num_critical_after"] = int(primary_score[0])
    features["lookahead_primary_num_responsible_after"] = int(primary_score[1])
    features["lookahead_primary_terminated_penalty"] = int(primary_score[2])
    features["lookahead_primary_num_C_edges_after"] = int(primary_score[3])
    features["lookahead_primary_neg_expected_g_drop"] = float(primary_score[4])

    features["lookahead_replay_reached_target_step"] = int(replay_reached_target_step)
    features["lookahead_replay_failed_before_target_step"] = int(replay_failed_before_target_step)
    features["lookahead_replay_cycle_detected"] = int(replay_cycle_detected)


def add_one_step_lookahead_labels_for_graph_samples(
    G: nx.Graph,
    samples: list[Any],
    *,
    backend: str = "ilp",
    graph_id: str,
) -> list[dict[str, Any]]:
    """
    Add `selected_by_lookahead` labels to a graph's Phase 4 trace samples.

    Important:
    - `samples` must come from the canonical rule run on this same graph
    - we replay the graph from scratch for each candidate using:
        prefix(rule decisions before this step)
        + forced current candidate
        + canonical continuation

    Tie policy:
    - compare candidates by the primary one-step score
    - if multiple candidates tie on the primary score and the original rule-selected
      candidate is among the tied set, preserve the rule choice

    Robustness policy:
    - if a forced replay terminates before reaching the target step (for example
      because repeat-state detection stops the run early), treat that candidate
      as invalid and assign it the worst possible primary score instead of crashing.
    """
    sample_dicts = [_to_dict(s) for s in samples]
    sample_dicts = sorted(sample_dicts, key=lambda s: int(s["state"]["iteration"]))

    prefix_keys: list[tuple] = []

    for step_idx, sample in enumerate(sample_dicts):
        candidates = sample["candidates"]
        if not candidates:
            continue

        selected_rule_idx = int(sample["decision"]["selected_candidate_idx"])
        primary_scores: list[tuple] = []

        for cand_idx, cand in enumerate(candidates):
            forced_key = candidate_key_from_dict(cand)

            replay_selector = PrefixThenForcedThenCanonicalSelector(
                prefix_keys=prefix_keys,
                forced_key=forced_key,
            )
            replay_result = _replay_graph_with_selector(
                G,
                selector=replay_selector,
                backend=backend,
                graph_id=graph_id,
            )

            replay_samples = replay_result["samples"]
            replay_cycle_detected = int(bool(replay_result["cycle_detected"]))

            if step_idx >= len(replay_samples):
                primary_score = _invalid_primary_score()
                _write_candidate_primary_score_features(
                    cand,
                    primary_score,
                    replay_reached_target_step=0,
                    replay_failed_before_target_step=1,
                    replay_cycle_detected=replay_cycle_detected,
                )
            else:
                replay_step_sample = replay_samples[step_idx]
                primary_score = _lookahead_primary_score(replay_step_sample, cand)
                _write_candidate_primary_score_features(
                    cand,
                    primary_score,
                    replay_reached_target_step=1,
                    replay_failed_before_target_step=0,
                    replay_cycle_detected=replay_cycle_detected,
                )

            primary_scores.append(primary_score)

        invalid_score = _invalid_primary_score()
        all_invalid = all(score == invalid_score for score in primary_scores)

        if all_invalid:
            best_idx = selected_rule_idx
            best_primary = invalid_score
            best_indices = list(range(len(candidates)))
            decision_kind = "all_invalid_fallback_rule"
        else:
            best_primary = min(primary_scores)
            best_indices = [i for i, score in enumerate(primary_scores) if score == best_primary]

            if len(best_indices) == 1:
                best_idx = best_indices[0]
                decision_kind = "score_driven"
            elif selected_rule_idx in best_indices:
                best_idx = selected_rule_idx
                decision_kind = "tie_preserve_rule"
            else:
                best_idx = min(
                    best_indices,
                    key=lambda i: _canonical_structural_tiebreak_key_from_dict(candidates[i]),
                )
                decision_kind = "tie_break_structural"

        for cand_idx, cand in enumerate(candidates):
            labels = cand.setdefault("labels", {})
            labels["selected_by_lookahead"] = 1 if cand_idx == best_idx else 0
            labels["lookahead_primary_best"] = 1 if primary_scores[cand_idx] == best_primary else 0

        sample["decision"]["selected_candidate_idx_by_lookahead"] = int(best_idx)
        sample["decision"]["lookahead_decision_kind"] = decision_kind
        sample["decision"]["lookahead_num_primary_best_candidates"] = int(len(best_indices))
        sample["decision"]["lookahead_rule_in_primary_best"] = int(selected_rule_idx in best_indices)

        prefix_keys.append(candidate_key_from_dict(candidates[selected_rule_idx]))

    return sample_dicts