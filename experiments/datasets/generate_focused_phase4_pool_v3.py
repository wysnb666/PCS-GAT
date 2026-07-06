from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MPC4plus.experiments.datasets.generate_harder_graph_dataset import (
    generate_connected_er_graph,
    is_harder_graph_proxy,
)
from MPC4plus.experiments.datasets.graph_dataset_io import (
    GraphRecord,
    build_edge_map,
    graph_to_record,
    make_graph_split_manifest,
    save_edge_map_json,
    save_graph_records_json,
    save_split_manifest_json,
)


def build_focused_phase4_pool_records(
    *,
    seed: int = 42,
    graphs_per_combo: int = 160,
) -> list[GraphRecord]:
    """
    Focused v3 pool:
      - emphasize parameter regions that already produced useful selector-sensitive graphs
      - keep the same harder_graph_proxy filter as harder_round_v1/v2
      - do not over-spread to low-yield parameter ranges

    Current focused combos:
      (30, 0.10), (30, 0.14), (40, 0.08)
    """
    rng = random.Random(seed)
    combos = [
        (30, 0.10),
        (30, 0.14),
        (40, 0.08),
    ]

    records: list[GraphRecord] = []
    for n, p in combos:
        p_tag = str(p).replace(".", "p")
        collected = 0
        attempts = 0

        while collected < graphs_per_combo:
            attempts += 1
            if attempts > graphs_per_combo * 300:
                raise RuntimeError(
                    f"Too many attempts while building focused Phase4 pool for n={n}, p={p}"
                )

            G = generate_connected_er_graph(n=n, p=p, rng=rng)
            if not is_harder_graph_proxy(G):
                continue

            graph_id = f"focused_er_n{n}_p{p_tag}_{collected:03d}"
            records.append(graph_to_record(G, graph_id=graph_id))
            collected += 1

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a focused ER graph pool for Phase 4 dense-data mining."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--graphs-per-combo",
        type=int,
        default=160,
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_focused_phase4_pool_records(
        seed=args.seed,
        graphs_per_combo=args.graphs_per_combo,
    )
    edge_map = build_edge_map(records)
    manifest = make_graph_split_manifest(
        [r.graph_id for r in records],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    save_graph_records_json(str(output_dir / "graphs.json"), records)
    save_edge_map_json(str(output_dir / "edge_map.json"), edge_map)
    save_split_manifest_json(str(output_dir / "split.json"), manifest)

    dataset_info = {
        "dataset_type": "focused_phase4_pool_v3",
        "seed": args.seed,
        "graphs_per_combo": args.graphs_per_combo,
        "num_graphs": len(records),
        "train_count": len(manifest.train_ids),
        "val_count": len(manifest.val_ids),
        "test_count": len(manifest.test_ids),
        "combos": [
            {"n": 30, "p": 0.10},
            {"n": 30, "p": 0.14},
            {"n": 40, "p": 0.08},
        ],
        "hardness_filter": {
            "avg_deg_min": 2.4,
            "density_max": 0.28,
            "require_degree_range_gt_1": True,
            "max_leaves_fraction": 1 / 3,
            "max_degree_min": 5,
        },
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Generated focused Phase4 pool with {len(records)} graphs.")
    print(f"[OK] Output dir: {output_dir}")


if __name__ == "__main__":
    main()