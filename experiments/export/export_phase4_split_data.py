from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MPC4plus.experiments.baselines.rule_runner import run_rule_phase4_pipeline
from MPC4plus.experiments.datasets.graph_dataset_io import (
    load_graph_records_json,
    load_split_manifest_json,
    record_to_graph,
    select_records_by_ids,
)
from MPC4plus.experiments.export.candidate_table_builder import (
    samples_to_candidate_rows,
)
from MPC4plus.experiments.export.one_step_lookahead_labeler import (
    add_one_step_lookahead_labels_for_graph_samples,
)


def export_phase4_split_data(
    *,
    graphs_json: str,
    split_json: str,
    split_name: str,
    output_dir: str,
    backend: str = "ilp",
    max_decisions_per_graph: int | None = None,
    dedup_state_signatures: bool = False,
) -> dict[str, Any]:
    """
    Export Phase 4 training data for one split.

    Outputs:
      - phase4_trace.jsonl
      - phase4_candidates.jsonl
      - phase4_candidates.csv
      - export_info.json

    This exporter augments every candidate with:
      labels["selected_by_lookahead"]

    Optional controls:
      - max_decisions_per_graph: keep at most K decision samples per graph
      - dedup_state_signatures: skip repeated structural states inside the same graph
    """
    records = load_graph_records_json(graphs_json)
    manifest = load_split_manifest_json(split_json)

    if split_name == "train":
        selected_ids = manifest.train_ids
    elif split_name == "val":
        selected_ids = manifest.val_ids
    elif split_name == "test":
        selected_ids = manifest.test_ids
    else:
        raise ValueError(f"Unsupported split name: {split_name}")

    selected_records = select_records_by_ids(records, selected_ids)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trace_path = output_path / "phase4_trace.jsonl"
    candidates_jsonl_path = output_path / "phase4_candidates.jsonl"
    candidates_csv_path = output_path / "phase4_candidates.csv"
    export_info_path = output_path / "export_info.json"

    all_samples: list[dict[str, Any]] = []

    total_raw_trace_samples = 0
    total_kept_trace_samples = 0
    per_graph_filter_stats: list[dict[str, Any]] = []

    for record in selected_records:
        G = record_to_graph(record)

        result = run_rule_phase4_pipeline(
            G,
            graph_id=record.graph_id,
            backend=backend,
            enable_trace=True,
        )

        if result.trace_collector is None:
            continue

        raw_samples = result.trace_collector.samples()
        augmented_samples = add_one_step_lookahead_labels_for_graph_samples(
            G,
            raw_samples,
            backend=backend,
            graph_id=record.graph_id,
        )

        _validate_augmented_samples(
            graph_id=record.graph_id,
            samples=augmented_samples,
        )

        filtered_samples, filter_stats = _filter_graph_samples(
            augmented_samples,
            max_decisions_per_graph=max_decisions_per_graph,
            dedup_state_signatures=dedup_state_signatures,
        )
        filter_stats["graph_id"] = record.graph_id
        per_graph_filter_stats.append(filter_stats)

        total_raw_trace_samples += int(filter_stats["num_raw_samples"])
        total_kept_trace_samples += int(filter_stats["num_kept_samples"])

        all_samples.extend(filtered_samples)

    with trace_path.open("w", encoding="utf-8") as f:
        for sample in all_samples:
            sample_dict = _to_jsonable_obj(sample)
            f.write(json.dumps(sample_dict, ensure_ascii=False))
            f.write("\n")

    candidate_rows = samples_to_candidate_rows(all_samples)

    with candidates_jsonl_path.open("w", encoding="utf-8") as f:
        for row in candidate_rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    if candidate_rows:
        fieldnames = _ordered_union_keys(candidate_rows)
        with candidates_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in candidate_rows:
                writer.writerow(row)
    else:
        candidates_csv_path.write_text("", encoding="utf-8")

    export_info = {
        "split_name": split_name,
        "backend": backend,
        "max_decisions_per_graph": max_decisions_per_graph,
        "dedup_state_signatures": dedup_state_signatures,
        "num_graphs": len(selected_records),
        "num_trace_samples_raw": total_raw_trace_samples,
        "num_trace_samples": len(all_samples),
        "num_trace_samples_kept": total_kept_trace_samples,
        "num_candidate_rows": len(candidate_rows),
        "graphs_json": graphs_json,
        "split_json": split_json,
        "trace_path": str(trace_path),
        "candidates_jsonl_path": str(candidates_jsonl_path),
        "candidates_csv_path": str(candidates_csv_path),
        "lookahead_label_name_trace": "selected_by_lookahead",
        "lookahead_label_name_candidate_table": "label__selected_by_lookahead",
        "per_graph_filter_stats": per_graph_filter_stats,
    }
    export_info_path.write_text(
        json.dumps(export_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return export_info


def _to_jsonable_obj(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return vars(obj)
    raise TypeError(f"Unsupported sample type for JSON serialization: {type(obj)}")


def _ordered_union_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def _validate_augmented_samples(
    *,
    graph_id: str,
    samples: list[dict[str, Any]],
) -> None:
    for sample_idx, sample in enumerate(samples):
        candidates = sample.get("candidates", [])
        if not candidates:
            continue

        pos_count = 0
        for cand_idx, cand in enumerate(candidates):
            labels = cand.get("labels", {})
            if "selected_by_lookahead" not in labels:
                raise RuntimeError(
                    f"Missing selected_by_lookahead label: "
                    f"graph_id={graph_id!r}, sample_idx={sample_idx}, cand_idx={cand_idx}"
                )
            if int(labels["selected_by_lookahead"]) == 1:
                pos_count += 1

        if pos_count != 1:
            raise RuntimeError(
                f"Invalid lookahead label multiplicity: "
                f"graph_id={graph_id!r}, sample_idx={sample_idx}, pos_count={pos_count}"
            )

        decision = sample.get("decision", {})
        if "selected_candidate_idx_by_lookahead" not in decision:
            raise RuntimeError(
                f"Missing selected_candidate_idx_by_lookahead in decision: "
                f"graph_id={graph_id!r}, sample_idx={sample_idx}"
            )


def _candidate_signature(cand: dict[str, Any]) -> tuple:
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


def _satellite_signature(sat: dict[str, Any]) -> tuple:
    return (
        int(sat["satellite_idx"]),
        tuple(int(x) for x in sat.get("nodes", [])),
        sat.get("kind"),
        tuple(tuple(int(y) for y in e) for e in sat.get("rescue_edges", [])),
        tuple(int(x) for x in sat.get("supporting_anchors", [])),
        sat.get("rescue_anchor"),
        bool(sat.get("is_critical_satellite", False)),
    )


def _component_signature(comp: dict[str, Any]) -> tuple:
    return (
        int(comp["component_idx"]),
        tuple(int(x) for x in comp.get("nodes", [])),
        bool(comp.get("is_composite", False)),
        tuple(int(x) for x in comp.get("center_nodes", [])),
        comp.get("center_kind"),
        tuple(int(x) for x in comp.get("anchors", [])),
        tuple((int(k), int(v)) for k, v in sorted(comp.get("j_anchor_map", {}).items())),
        tuple(int(x) for x in comp.get("critical_2_anchors", [])),
        tuple(int(x) for x in comp.get("responsible_1_anchors", [])),
        int(comp.get("s_value", 0) or 0),
        int(comp.get("opt_value", 0) or 0),
        float(comp.get("critical_ratio", 0.0) or 0.0),
        bool(comp.get("is_critical", False)),
        bool(comp.get("is_responsible_component", False)),
        tuple(_satellite_signature(s) for s in comp.get("satellites", [])),
    )


def _state_signature(sample: dict[str, Any]) -> tuple:
    state = sample.get("state", {})
    components = sample.get("components", [])
    candidates = sample.get("candidates", [])

    return (
        int(state.get("num_nodes", 0)),
        int(state.get("num_edges", 0)),
        int(state.get("num_H_edges", 0)),
        int(state.get("num_C_edges", 0)),
        int(state.get("num_M_C_edges", 0)),
        int(state.get("num_components", 0)),
        int(state.get("num_critical_components", 0)),
        int(state.get("num_responsible_components", 0)),
        tuple(_component_signature(comp) for comp in components),
        tuple(_candidate_signature(c) for c in candidates),
    )


def _filter_graph_samples(
    samples: list[dict[str, Any]],
    *,
    max_decisions_per_graph: int | None,
    dedup_state_signatures: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    seen_signatures: set[tuple] = set()
    num_dropped_by_dedup = 0
    num_dropped_by_cap = 0

    for sample in samples:
        if dedup_state_signatures:
            sig = _state_signature(sample)
            if sig in seen_signatures:
                num_dropped_by_dedup += 1
                continue
            seen_signatures.add(sig)

        if max_decisions_per_graph is not None and len(kept) >= max_decisions_per_graph:
            num_dropped_by_cap += 1
            continue

        kept.append(sample)

    stats = {
        "num_raw_samples": len(samples),
        "num_kept_samples": len(kept),
        "num_dropped_by_dedup": num_dropped_by_dedup,
        "num_dropped_by_cap": num_dropped_by_cap,
    }
    return kept, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch export Phase 4 trace/candidate data for one split."
    )
    parser.add_argument(
        "--graphs-json",
        type=str,
        required=True,
        help="Path to graph records JSON.",
    )
    parser.add_argument(
        "--split-json",
        type=str,
        required=True,
        help="Path to train/val/test split manifest JSON.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Which split to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for phase4_trace.jsonl / phase4_candidates.* outputs.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ilp",
        choices=["ilp", "bruteforce"],
    )
    parser.add_argument(
        "--max-decisions-per-graph",
        type=int,
        default=None,
        help="Optional cap on kept decision samples per graph after augmentation.",
    )
    parser.add_argument(
        "--dedup-state-signatures",
        action="store_true",
        help="If set, repeated structural states inside one graph are dropped.",
    )

    args = parser.parse_args()

    info = export_phase4_split_data(
        graphs_json=args.graphs_json,
        split_json=args.split_json,
        split_name=args.split,
        output_dir=args.output_dir,
        backend=args.backend,
        max_decisions_per_graph=args.max_decisions_per_graph,
        dedup_state_signatures=args.dedup_state_signatures,
    )

    print(f"[OK] Exported split={args.split}")
    print(f"[OK] num_graphs={info['num_graphs']}")
    print(f"[OK] num_trace_samples_raw={info['num_trace_samples_raw']}")
    print(f"[OK] num_trace_samples={info['num_trace_samples']}")
    print(f"[OK] num_candidate_rows={info['num_candidate_rows']}")
    print(f"[OK] output_dir={args.output_dir}")


if __name__ == "__main__":
    main()