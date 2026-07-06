from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_mean(xs):
    xs = list(xs)
    return mean(xs) if xs else 0.0


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


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output_json)

    rows = load_jsonl(input_path)
    if not rows:
        raise RuntimeError(f"No rows found in: {input_path}")

    # group candidate rows by decision point
    # sample_id is the most reliable grouping key if present; fall back to (graph_id, iteration)
    groups = defaultdict(list)
    graph_ids_all = set()

    for row in rows:
        graph_id = row["graph_id"]
        graph_ids_all.add(graph_id)

        sample_id = row.get("sample_id")
        if sample_id is not None:
            key = (graph_id, sample_id)
        else:
            key = (graph_id, row["iteration"])
        groups[key].append(row)

    num_decision_points = 0
    num_decision_points_with_disagreement = 0
    num_graphs_with_phase4 = 0
    graphs_with_disagreement = set()

    num_candidates_per_decision = []
    disagreement_candidate_counts = []

    op_counter_all = Counter()
    op_counter_rule = Counter()
    op_counter_lookahead = Counter()
    op_counter_disagreement_rule = Counter()
    op_counter_disagreement_lookahead = Counter()

    graph_decision_count = Counter()
    graph_disagreement_count = Counter()

    disagreement_details = []

    for key, cand_rows in groups.items():
        graph_id = key[0]
        graph_decision_count[graph_id] += 1
        num_decision_points += 1

        if graph_decision_count[graph_id] == 1:
            num_graphs_with_phase4 += 1

        num_candidates = len(cand_rows)
        num_candidates_per_decision.append(num_candidates)

        for r in cand_rows:
            op_counter_all[r["op_type"]] += 1

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
            graphs_with_disagreement.add(graph_id)
            graph_disagreement_count[graph_id] += 1
            disagreement_candidate_counts.append(num_candidates)

            op_counter_disagreement_rule[rule_op] += 1
            op_counter_disagreement_lookahead[look_op] += 1

            disagreement_details.append(
                {
                    "graph_id": graph_id,
                    "sample_id": rule_row.get("sample_id"),
                    "iteration": int(rule_row["iteration"]),
                    "num_candidates": num_candidates,
                    "rule_candidate_idx": rule_idx,
                    "lookahead_candidate_idx": look_idx,
                    "rule_op_type": rule_op,
                    "lookahead_op_type": look_op,
                    "rule_expected_g_drop": float(rule_row.get("feature__expected_g_drop", 0.0)),
                    "lookahead_expected_g_drop": float(look_row.get("feature__expected_g_drop", 0.0)),
                    "rule_outcome_num_critical_after": int(rule_row.get("outcome_num_critical_after", 0)),
                    "lookahead_outcome_num_critical_after": int(look_row.get("outcome_num_critical_after", 0)),
                    "rule_outcome_num_responsible_after": int(rule_row.get("outcome_num_responsible_after", 0)),
                    "lookahead_outcome_num_responsible_after": int(
                        look_row.get("outcome_num_responsible_after", 0)
                    ),
                    "rule_outcome_num_C_edges_after": int(rule_row.get("outcome_num_C_edges_after", 0)),
                    "lookahead_outcome_num_C_edges_after": int(
                        look_row.get("outcome_num_C_edges_after", 0)
                    ),
                    "rule_terminated_after_apply": int(bool(rule_row.get("terminated_after_apply", False))),
                    "lookahead_terminated_after_apply": int(bool(look_row.get("terminated_after_apply", False))),
                }
            )

    num_graphs_total = len(graph_ids_all)
    num_graphs_with_disagreement = len(graphs_with_disagreement)

    summary = {
        "input": str(input_path),
        "num_candidate_rows": len(rows),
        "num_graphs_total": num_graphs_total,
        "num_graphs_with_phase4": num_graphs_with_phase4,
        "num_graphs_with_disagreement": num_graphs_with_disagreement,
        "ratio_graphs_with_phase4": (
            num_graphs_with_phase4 / num_graphs_total if num_graphs_total else 0.0
        ),
        "ratio_graphs_with_disagreement": (
            num_graphs_with_disagreement / num_graphs_total if num_graphs_total else 0.0
        ),
        "num_decision_points": num_decision_points,
        "num_decision_points_with_disagreement": num_decision_points_with_disagreement,
        "ratio_decision_points_with_disagreement": (
            num_decision_points_with_disagreement / num_decision_points
            if num_decision_points
            else 0.0
        ),
        "avg_num_candidates_per_decision": safe_mean(num_candidates_per_decision),
        "avg_num_candidates_per_disagreement_decision": safe_mean(disagreement_candidate_counts),
        "op_type_distribution_all_candidates": dict(op_counter_all),
        "op_type_distribution_rule_selected": dict(op_counter_rule),
        "op_type_distribution_lookahead_selected": dict(op_counter_lookahead),
        "op_type_distribution_rule_selected_on_disagreement": dict(op_counter_disagreement_rule),
        "op_type_distribution_lookahead_selected_on_disagreement": dict(op_counter_disagreement_lookahead),
        "top_graphs_by_num_phase4_decisions": [
            {"graph_id": gid, "num_phase4_decisions": cnt}
            for gid, cnt in graph_decision_count.most_common(20)
        ],
        "top_graphs_by_num_disagreements": [
            {"graph_id": gid, "num_disagreements": cnt}
            for gid, cnt in graph_disagreement_count.most_common(20)
        ],
        "disagreement_examples_head": disagreement_details[:50],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Analyzed candidate rows: {len(rows)}")
    print(f"[OK] num_graphs_total={summary['num_graphs_total']}")
    print(f"[OK] num_graphs_with_phase4={summary['num_graphs_with_phase4']}")
    print(f"[OK] num_graphs_with_disagreement={summary['num_graphs_with_disagreement']}")
    print(f"[OK] num_decision_points={summary['num_decision_points']}")
    print(f"[OK] num_decision_points_with_disagreement={summary['num_decision_points_with_disagreement']}")
    print(
        "[OK] ratio_decision_points_with_disagreement="
        f"{summary['ratio_decision_points_with_disagreement']:.4f}"
    )
    print(f"[OK] Summary saved to: {output_path}")


if __name__ == "__main__":
    main()