from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze disagreement between rule labels and one-step lookahead labels."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to phase4_candidates.jsonl",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to save disagreement summary JSON",
    )
    return parser.parse_args()


def _primary_score_tuple_from_row(row: dict) -> tuple:
    required = [
        "feature__lookahead_primary_num_critical_after",
        "feature__lookahead_primary_num_responsible_after",
        "feature__lookahead_primary_terminated_penalty",
        "feature__lookahead_primary_num_C_edges_after",
        "feature__lookahead_primary_neg_expected_g_drop",
    ]
    missing = [k for k in required if k not in row]
    if missing:
        raise RuntimeError(
            "Missing candidate-specific lookahead primary score fields. "
            f"Did you re-export after updating one_step_lookahead_labeler.py? Missing: {missing}"
        )

    return (
        int(row["feature__lookahead_primary_num_critical_after"]),
        int(row["feature__lookahead_primary_num_responsible_after"]),
        int(row["feature__lookahead_primary_terminated_penalty"]),
        int(row["feature__lookahead_primary_num_C_edges_after"]),
        float(row["feature__lookahead_primary_neg_expected_g_drop"]),
    )


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output_json)

    rows = load_jsonl(input_path)
    if not rows:
        raise RuntimeError(f"No rows found in: {input_path}")

    groups = defaultdict(list)
    graph_ids_all = set()

    for row in rows:
        graph_id = row["graph_id"]
        graph_ids_all.add(graph_id)
        sample_id = row.get("sample_id")
        key = (graph_id, sample_id if sample_id is not None else row["iteration"])
        groups[key].append(row)

    num_graphs_total = len(graph_ids_all)
    num_decision_points = 0
    num_decision_points_with_disagreement = 0
    num_score_driven_disagreements = 0
    num_tie_only_disagreements = 0

    graph_decision_count = Counter()
    graph_disagreement_count = Counter()
    graph_score_driven_disagreement_count = Counter()
    graph_tie_only_disagreement_count = Counter()

    op_counter_rule = Counter()
    op_counter_lookahead = Counter()
    op_counter_score_driven_rule = Counter()
    op_counter_score_driven_lookahead = Counter()

    examples_score_driven = []
    examples_tie_only = []

    for key, cand_rows in groups.items():
        graph_id = key[0]
        num_decision_points += 1
        graph_decision_count[graph_id] += 1

        rule_rows = [r for r in cand_rows if int(r.get("label__selected_by_rule", 0)) == 1]
        look_rows = [r for r in cand_rows if int(r.get("label__selected_by_lookahead", 0)) == 1]

        if len(rule_rows) != 1:
            raise RuntimeError(
                f"Expected exactly one rule-positive row in decision {key}, got {len(rule_rows)}"
            )
        if len(look_rows) != 1:
            raise RuntimeError(
                f"Expected exactly one lookahead-positive row in decision {key}, got {len(look_rows)}"
            )

        rule_row = rule_rows[0]
        look_row = look_rows[0]

        rule_idx = int(rule_row["candidate_idx"])
        look_idx = int(look_row["candidate_idx"])

        rule_op = str(rule_row["op_type"])
        look_op = str(look_row["op_type"])
        op_counter_rule[rule_op] += 1
        op_counter_lookahead[look_op] += 1

        if rule_idx != look_idx:
            num_decision_points_with_disagreement += 1
            graph_disagreement_count[graph_id] += 1

            rule_score = _primary_score_tuple_from_row(rule_row)
            look_score = _primary_score_tuple_from_row(look_row)

            detail = {
                "graph_id": graph_id,
                "sample_id": rule_row.get("sample_id"),
                "iteration": int(rule_row["iteration"]),
                "num_candidates": len(cand_rows),
                "rule_candidate_idx": rule_idx,
                "lookahead_candidate_idx": look_idx,
                "rule_op_type": rule_op,
                "lookahead_op_type": look_op,
                "rule_score_tuple": rule_score,
                "lookahead_score_tuple": look_score,
                "rule_primary_best": int(rule_row.get("label__lookahead_primary_best", 0)),
                "lookahead_primary_best": int(look_row.get("label__lookahead_primary_best", 0)),
            }

            if look_score != rule_score:
                num_score_driven_disagreements += 1
                graph_score_driven_disagreement_count[graph_id] += 1
                op_counter_score_driven_rule[rule_op] += 1
                op_counter_score_driven_lookahead[look_op] += 1
                if len(examples_score_driven) < 50:
                    examples_score_driven.append(detail)
            else:
                num_tie_only_disagreements += 1
                graph_tie_only_disagreement_count[graph_id] += 1
                if len(examples_tie_only) < 50:
                    examples_tie_only.append(detail)

    summary = {
        "input": str(input_path),
        "num_candidate_rows": len(rows),
        "num_graphs_total": num_graphs_total,
        "num_decision_points": num_decision_points,
        "num_decision_points_with_disagreement": num_decision_points_with_disagreement,
        "ratio_decision_points_with_disagreement": (
            num_decision_points_with_disagreement / num_decision_points
            if num_decision_points else 0.0
        ),
        "num_score_driven_disagreements": num_score_driven_disagreements,
        "num_tie_only_disagreements": num_tie_only_disagreements,
        "ratio_score_driven_among_disagreements": (
            num_score_driven_disagreements / num_decision_points_with_disagreement
            if num_decision_points_with_disagreement else 0.0
        ),
        "ratio_tie_only_among_disagreements": (
            num_tie_only_disagreements / num_decision_points_with_disagreement
            if num_decision_points_with_disagreement else 0.0
        ),
        "op_type_distribution_rule_selected": dict(op_counter_rule),
        "op_type_distribution_lookahead_selected": dict(op_counter_lookahead),
        "op_type_distribution_rule_selected_on_score_driven_disagreement": dict(op_counter_score_driven_rule),
        "op_type_distribution_lookahead_selected_on_score_driven_disagreement": dict(op_counter_score_driven_lookahead),
        "top_graphs_by_num_phase4_decisions": [
            {"graph_id": gid, "num_phase4_decisions": cnt}
            for gid, cnt in graph_decision_count.most_common(20)
        ],
        "top_graphs_by_num_disagreements": [
            {"graph_id": gid, "num_disagreements": cnt}
            for gid, cnt in graph_disagreement_count.most_common(20)
        ],
        "top_graphs_by_num_score_driven_disagreements": [
            {"graph_id": gid, "num_score_driven_disagreements": cnt}
            for gid, cnt in graph_score_driven_disagreement_count.most_common(20)
        ],
        "top_graphs_by_num_tie_only_disagreements": [
            {"graph_id": gid, "num_tie_only_disagreements": cnt}
            for gid, cnt in graph_tie_only_disagreement_count.most_common(20)
        ],
        "score_driven_disagreement_examples_head": examples_score_driven,
        "tie_only_disagreement_examples_head": examples_tie_only,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Analyzed candidate rows: {len(rows)}")
    print(f"[OK] num_graphs_total={summary['num_graphs_total']}")
    print(f"[OK] num_decision_points={summary['num_decision_points']}")
    print(f"[OK] num_decision_points_with_disagreement={summary['num_decision_points_with_disagreement']}")
    print(f"[OK] num_score_driven_disagreements={summary['num_score_driven_disagreements']}")
    print(f"[OK] num_tie_only_disagreements={summary['num_tie_only_disagreements']}")
    print(
        "[OK] ratio_tie_only_among_disagreements="
        f"{summary['ratio_tie_only_among_disagreements']:.4f}"
    )
    print(f"[OK] Summary saved to: {output_path}")


if __name__ == "__main__":
    main()