from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
)
from MPC4plus.experiments.export.one_step_lookahead_labeler import (
    add_one_step_lookahead_labels_for_graph_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Phase4-dense dataset by filtering only selected splits "
            "(typically train) while keeping the other splits unchanged."
        )
    )
    parser.add_argument(
        "--source-graphs-json",
        type=str,
        required=True,
        help="Path to source graphs.json",
    )
    parser.add_argument(
        "--source-split-json",
        type=str,
        required=True,
        help="Path to source split.json",
    )
    parser.add_argument(
        "--source-edge-map-json",
        type=str,
        required=True,
        help="Path to source edge_map.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output dataset directory",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ilp",
        choices=["ilp", "bruteforce"],
        help="Backend used in screening runs",
    )
    parser.add_argument(
        "--filter-splits",
        type=str,
        default="train",
        help="Comma-separated split names to filter, e.g. train or train,val",
    )
    parser.add_argument(
        "--min-phase4-decisions",
        type=int,
        default=1,
        help="Keep graph only if canonical rule trace has at least this many Phase 4 decisions",
    )
    parser.add_argument(
        "--require-score-driven-disagreement",
        action="store_true",
        help="Require at least --min-score-driven-disagreements score-driven disagreements",
    )
    parser.add_argument(
        "--min-score-driven-disagreements",
        type=int,
        default=1,
        help="Minimum number of score-driven disagreements required if --require-score-driven-disagreement is set",
    )
    parser.add_argument(
        "--lookahead-max-decisions-per-graph",
        type=int,
        default=3,
        help="When screening disagreements, only analyze the first K decision points per graph to bound runtime",
    )
    return parser.parse_args()


def load_raw_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def split_name_set(s: str) -> set[str]:
    xs = {x.strip() for x in s.split(",") if x.strip()}
    valid = {"train", "val", "test"}
    bad = xs - valid
    if bad:
        raise ValueError(f"Unsupported split names in --filter-splits: {sorted(bad)}")
    return xs


def analyze_graph_for_phase4_density(
    *,
    graph_id: str,
    G,
    backend: str,
    min_phase4_decisions: int,
    require_score_driven_disagreement: bool,
    min_score_driven_disagreements: int,
    lookahead_max_decisions_per_graph: int,
) -> dict[str, Any]:
    result = run_rule_phase4_pipeline(
        G,
        graph_id=graph_id,
        backend=backend,
        enable_trace=True,
    )

    raw_samples = []
    if result.trace_collector is not None:
        raw_samples = list(result.trace_collector.samples())

    num_phase4_decisions = len(raw_samples)
    stats: dict[str, Any] = {
        "graph_id": graph_id,
        "num_phase4_decisions": num_phase4_decisions,
        "num_screened_decisions_for_lookahead": 0,
        "num_score_driven_disagreements": 0,
        "keep": False,
        "reject_reason": None,
    }

    if num_phase4_decisions < min_phase4_decisions:
        stats["reject_reason"] = "too_few_phase4_decisions"
        return stats

    if require_score_driven_disagreement:
        capped_samples = raw_samples[:lookahead_max_decisions_per_graph]
        augmented = add_one_step_lookahead_labels_for_graph_samples(
            G,
            capped_samples,
            backend=backend,
            graph_id=graph_id,
        )

        score_driven = 0
        for sample in augmented:
            decision = sample.get("decision", {})
            selected_rule_idx = int(decision["selected_candidate_idx"])
            selected_lookahead_idx = int(decision["selected_candidate_idx_by_lookahead"])
            decision_kind = str(decision.get("lookahead_decision_kind", ""))

            if (
                decision_kind == "score_driven"
                and selected_rule_idx != selected_lookahead_idx
            ):
                score_driven += 1

        stats["num_screened_decisions_for_lookahead"] = len(capped_samples)
        stats["num_score_driven_disagreements"] = score_driven

        if score_driven < min_score_driven_disagreements:
            stats["reject_reason"] = "too_few_score_driven_disagreements"
            return stats

    stats["keep"] = True
    return stats


def main() -> None:
    args = parse_args()

    filter_splits = split_name_set(args.filter_splits)

    source_graphs_raw = load_raw_json(args.source_graphs_json)
    if not isinstance(source_graphs_raw, list):
        raise RuntimeError("Expected source graphs.json to contain a list of graph records")

    source_edge_map = load_raw_json(args.source_edge_map_json)
    source_records = load_graph_records_json(args.source_graphs_json)
    source_manifest = load_split_manifest_json(args.source_split_json)

    raw_by_id = {r["graph_id"]: r for r in source_graphs_raw}
    typed_by_id = {r.graph_id: r for r in source_records}

    split_to_ids = {
        "train": list(source_manifest.train_ids),
        "val": list(source_manifest.val_ids),
        "test": list(source_manifest.test_ids),
    }

    kept_split_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    screening_stats: list[dict[str, Any]] = []

    split_counter_before = {k: len(v) for k, v in split_to_ids.items()}
    split_counter_after = Counter()

    for split_name, ids in split_to_ids.items():
        if split_name not in filter_splits:
            kept_split_ids[split_name] = list(ids)
            split_counter_after[split_name] = len(ids)
            for gid in ids:
                screening_stats.append(
                    {
                        "graph_id": gid,
                        "split": split_name,
                        "num_phase4_decisions": None,
                        "num_screened_decisions_for_lookahead": None,
                        "num_score_driven_disagreements": None,
                        "keep": True,
                        "reject_reason": "unfiltered_split_passthrough",
                    }
                )
            continue

        for gid in ids:
            if gid not in typed_by_id:
                raise RuntimeError(f"Graph id missing from records: {gid}")
            record = typed_by_id[gid]
            G = record_to_graph(record)

            stats = analyze_graph_for_phase4_density(
                graph_id=gid,
                G=G,
                backend=args.backend,
                min_phase4_decisions=args.min_phase4_decisions,
                require_score_driven_disagreement=args.require_score_driven_disagreement,
                min_score_driven_disagreements=args.min_score_driven_disagreements,
                lookahead_max_decisions_per_graph=args.lookahead_max_decisions_per_graph,
            )
            stats["split"] = split_name
            screening_stats.append(stats)

            if stats["keep"]:
                kept_split_ids[split_name].append(gid)

        split_counter_after[split_name] = len(kept_split_ids[split_name])

    kept_ids_all = set(
        kept_split_ids["train"] + kept_split_ids["val"] + kept_split_ids["test"]
    )

    kept_graphs_raw = [raw_by_id[gid] for gid in source_graphs_raw_ids_in_order(source_graphs_raw) if gid in kept_ids_all]
    kept_edge_map = {gid: source_edge_map[gid] for gid in kept_ids_all if gid in source_edge_map}

    output_dir = Path(args.output_dir)
    graphs_out = output_dir / "graphs.json"
    split_out = output_dir / "split.json"
    edge_map_out = output_dir / "edge_map.json"
    info_out = output_dir / "dataset_info.json"
    screening_out = output_dir / "screening_summary.json"

    dump_json(graphs_out, kept_graphs_raw)
    dump_json(
        split_out,
        {
            "train_ids": kept_split_ids["train"],
            "val_ids": kept_split_ids["val"],
            "test_ids": kept_split_ids["test"],
        },
    )
    dump_json(edge_map_out, kept_edge_map)

    kept_train = kept_split_ids["train"]
    screening_train = [x for x in screening_stats if x["split"] == "train"]
    screening_train_filtered = [x for x in screening_train if x["reject_reason"] != "unfiltered_split_passthrough"]

    dataset_info = {
        "source_graphs_json": args.source_graphs_json,
        "source_split_json": args.source_split_json,
        "source_edge_map_json": args.source_edge_map_json,
        "backend": args.backend,
        "filter_splits": sorted(filter_splits),
        "min_phase4_decisions": args.min_phase4_decisions,
        "require_score_driven_disagreement": bool(args.require_score_driven_disagreement),
        "min_score_driven_disagreements": args.min_score_driven_disagreements,
        "lookahead_max_decisions_per_graph": args.lookahead_max_decisions_per_graph,
        "num_graphs_total_before": len(source_graphs_raw),
        "num_graphs_total_after": len(kept_graphs_raw),
        "split_counts_before": split_counter_before,
        "split_counts_after": dict(split_counter_after),
        "num_train_screened": len(screening_train_filtered),
        "num_train_kept": len(kept_train),
        "num_train_rejected": len(screening_train_filtered) - len(kept_train),
    }

    dump_json(info_out, dataset_info)
    dump_json(screening_out, screening_stats)

    print(f"[OK] Built Phase4-dense dataset at: {output_dir}")
    print(f"[OK] split_counts_before={split_counter_before}")
    print(f"[OK] split_counts_after={dict(split_counter_after)}")
    print(f"[OK] num_graphs_total_after={len(kept_graphs_raw)}")


def source_graphs_raw_ids_in_order(source_graphs_raw: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for r in source_graphs_raw:
        gid = r.get("graph_id")
        if gid is None:
            raise RuntimeError("A raw graph record is missing graph_id")
        ids.append(gid)
    return ids


if __name__ == "__main__":
    main()