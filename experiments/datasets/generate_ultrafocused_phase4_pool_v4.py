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


def build_ultrafocused_phase4_pool_records(
    *,
    seed: int = 42,
    num_p010: int = 280,
    num_p014: int = 140,
) -> list[GraphRecord]:
    """
    Ultra-focused v4 pool:
      - primary:  n=30, p=0.10
      - auxiliary: n=30, p=0.14
    """
    rng = random.Random(seed)
    combo_specs = [
        (30, 0.10, num_p010),
        (30, 0.14, num_p014),
    ]

    records: list[GraphRecord] = []
    for n, p, target_count in combo_specs:
        p_tag = str(p).replace(".", "p")
        collected = 0
        attempts = 0

        while collected < target_count:
            attempts += 1
            if attempts > target_count * 400:
                raise RuntimeError(
                    f"Too many attempts while building ultra-focused pool for n={n}, p={p}"
                )

            G = generate_connected_er_graph(n=n, p=p, rng=rng)
            if not is_harder_graph_proxy(G):
                continue

            graph_id = f"ultrafocused_er_n{n}_p{p_tag}_{collected:03d}"
            records.append(graph_to_record(G, graph_id=graph_id))
            collected += 1

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an ultra-focused Phase 4 pool concentrated on n=30."
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-p010", type=int, default=280)
    parser.add_argument("--num-p014", type=int, default=140)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_ultrafocused_phase4_pool_records(
        seed=args.seed,
        num_p010=args.num_p010,
        num_p014=args.num_p014,
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
        "dataset_type": "ultrafocused_phase4_pool_v4",
        "seed": args.seed,
        "num_graphs": len(records),
        "train_count": len(manifest.train_ids),
        "val_count": len(manifest.val_ids),
        "test_count": len(manifest.test_ids),
        "combos": [
            {"n": 30, "p": 0.10, "count": args.num_p010},
            {"n": 30, "p": 0.14, "count": args.num_p014},
        ],
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Generated ultra-focused Phase4 pool with {len(records)} graphs.")
    print(f"[OK] Output dir: {output_dir}")


if __name__ == "__main__":
    main()