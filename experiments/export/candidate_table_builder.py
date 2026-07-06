from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from MPC4plus.experiments.export.phase4_trace_schema import Phase4DecisionSample


def _to_sample_dict(sample: Phase4DecisionSample | dict[str, Any]) -> dict[str, Any]:
    if is_dataclass(sample):
        return asdict(sample)
    return dict(sample)


def _component_index(sample_dict: dict[str, Any]) -> dict[int, dict[str, Any]]:
    components = sample_dict.get("components", [])
    return {int(comp["component_idx"]): comp for comp in components}


def _flatten_prefixed(prefix: str, obj: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k, v in obj.items():
        flat[f"{prefix}{k}"] = v
    return flat


def sample_to_candidate_rows(
    sample: Phase4DecisionSample | dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert one Phase4DecisionSample into candidate-level flat rows.

    One decision sample -> many candidate rows.
    """
    s = _to_sample_dict(sample)

    graph_id = s["graph_id"]
    sample_id = s["sample_id"]
    state = s["state"]
    decision = s["decision"]
    outcome = s["outcome"]
    candidates = s["candidates"]
    comp_map = _component_index(s)

    base_row = {
        "graph_id": graph_id,
        "sample_id": sample_id,
        "iteration": state["iteration"],
        "selector_name": decision["selector_name"],
        "selected_candidate_idx": decision["selected_candidate_idx"],
        "num_nodes": state["num_nodes"],
        "num_edges": state["num_edges"],
        "num_H_edges": state["num_H_edges"],
        "num_C_edges": state["num_C_edges"],
        "num_M_C_edges": state["num_M_C_edges"],
        "num_components": state["num_components"],
        "num_critical_components": state["num_critical_components"],
        "num_responsible_components": state["num_responsible_components"],
        "num_candidates_total": state["num_candidates_total"],
        "num_candidates_op1": state["num_candidates_op1"],
        "num_candidates_op2": state["num_candidates_op2"],
        "num_candidates_op3": state["num_candidates_op3"],
        "history_length": state["history_length"],
        "outcome_num_critical_before": outcome["num_critical_before"],
        "outcome_num_critical_after": outcome["num_critical_after"],
        "outcome_num_responsible_before": outcome["num_responsible_before"],
        "outcome_num_responsible_after": outcome["num_responsible_after"],
        "outcome_num_C_edges_before": outcome["num_C_edges_before"],
        "outcome_num_C_edges_after": outcome["num_C_edges_after"],
        "terminated_after_apply": outcome["terminated_after_apply"],
    }

    rows: list[dict[str, Any]] = []
    for cand in candidates:
        row = dict(base_row)

        source_comp = comp_map.get(int(cand["source_component_idx"]), {})
        target_comp = comp_map.get(int(cand["target_component_idx"]), {})

        row.update(
            {
                "candidate_idx": cand["candidate_idx"],
                "op_type": cand["op_type"],
                "source_component_idx": cand["source_component_idx"],
                "source_satellite_idx": cand["source_satellite_idx"],
                "source_vertex": cand["source_vertex"],
                "target_component_idx": cand["target_component_idx"],
                "target_vertex": cand["target_vertex"],
                "added_edge": tuple(cand["added_edge"]),
                "removed_edges": [tuple(e) for e in cand["removed_edges"]],
                "note": cand["note"],
                "source_component_size": len(source_comp.get("nodes", [])),
                "target_component_size": len(target_comp.get("nodes", [])),
                "source_center_kind": source_comp.get("center_kind"),
                "target_center_kind": target_comp.get("center_kind"),
                "source_is_critical": source_comp.get("is_critical"),
                "target_is_critical": target_comp.get("is_critical"),
                "source_is_responsible_component": source_comp.get("is_responsible_component"),
                "target_is_responsible_component": target_comp.get("is_responsible_component"),
                "source_num_satellites": len(source_comp.get("satellites", [])),
                "target_num_satellites": len(target_comp.get("satellites", [])),
                "source_s_value": source_comp.get("s_value"),
                "target_s_value": target_comp.get("s_value"),
                "source_opt_value": source_comp.get("opt_value"),
                "target_opt_value": target_comp.get("opt_value"),
                "source_critical_ratio": source_comp.get("critical_ratio"),
                "target_critical_ratio": target_comp.get("critical_ratio"),
            }
        )

        row.update(_flatten_prefixed("feature__", cand.get("features", {})))
        row.update(_flatten_prefixed("label__", cand.get("labels", {})))

        rows.append(row)

    return rows


def samples_to_candidate_rows(
    samples: Iterable[Phase4DecisionSample | dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        rows.extend(sample_to_candidate_rows(sample))
    return rows


def load_samples_from_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_candidate_rows_jsonl(rows: Iterable[dict[str, Any]], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def dump_candidate_rows_csv(rows: Iterable[dict[str, Any]], path: str) -> None:
    rows = list(rows)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([])
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)