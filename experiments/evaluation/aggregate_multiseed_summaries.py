from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple test_summary.json files into mean/std statistics."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of summary json paths",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to save aggregated result",
    )
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def m(xs):
    xs = list(xs)
    return mean(xs) if xs else 0.0


def s(xs):
    xs = list(xs)
    return pstdev(xs) if len(xs) > 1 else 0.0


def main() -> None:
    args = parse_args()
    summaries = [load_json(p) for p in args.inputs]

    if not summaries:
        raise RuntimeError("No input summaries provided.")

    selectors = set()
    for sm in summaries:
        selectors.update(sm.get("selector_summary", {}).keys())
    selectors = sorted(selectors)

    result = {
        "num_runs": len(summaries),
        "inputs": args.inputs,
        "global_metrics": {},
        "selector_metrics": {},
    }

    global_keys = [
        "num_graphs",
        "graphs_with_phase4_steps_gt_0",
        "graphs_with_selector_difference",
        "ratio_phase4_steps_gt_0",
        "ratio_selector_difference",
    ]

    for key in global_keys:
        vals = [float(sm.get(key, 0.0)) for sm in summaries]
        result["global_metrics"][key] = {
            "mean": m(vals),
            "std": s(vals),
            "values": vals,
        }

    selector_metric_keys = [
        "better_than_rule",
        "equal_to_rule",
        "worse_than_rule",
        "mean_covered_vertices",
        "mean_coverage_ratio",
        "mean_phase4_solve_seconds",
        "mean_phase5_solve_seconds",
        "mean_total_solve_seconds",
    ]

    for selector in selectors:
        result["selector_metrics"][selector] = {}
        for metric in selector_metric_keys:
            vals = [
                float(sm.get("selector_summary", {}).get(selector, {}).get(metric, 0.0))
                for sm in summaries
            ]
            result["selector_metrics"][selector][metric] = {
                "mean": m(vals),
                "std": s(vals),
                "values": vals,
            }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Aggregated {len(summaries)} summary files.")
    print(f"[OK] Output: {out}")


if __name__ == "__main__":
    main()