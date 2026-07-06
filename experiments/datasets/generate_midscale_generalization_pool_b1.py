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


def build_midscale_generalization_pool_records(
    *,
    seed: int = 42,
    count_n40_p008: int = 160,
    count_n50_p006: int = 160,
    count_n60_p005: int = 160,
) -> list[GraphRecord]:
    """
    Mid-scale generalization pool:
      - n=40, p=0.08
      - n=50, p=0.06
      - n=60, p=0.05

    Goal:
      validate whether the GNN selector still behaves well on larger graphs,
      without jumping immediately to very sparse/very large regimes that often
      produce too few useful Phase 4 decisions.
    """
    rng = random.Random(seed)
    combo_specs = [
        (40, 0.08, count_n40_p008),
        (50, 0.06, count_n50_p006),
        (60, 0.05, count_n60_p005),
    ]

    records: list[GraphRecord] = []
    for n, p, target_count in combo_specs:
        p_tag = str(p).replace(".", "p")
        collected = 0
        attempts = 0

        while collected < target_count:
            attempts += 1
            if attempts > target_count * 500:
                raise RuntimeError(
                    f"Too many attempts while building mid-scale pool for n={n}, p={p}"
                )

            G = generate_connected_er_graph(n=n, p=p, rng=rng)
            if not is_harder_graph_proxy(G):
                continue

            graph_id = f"midscale_er_n{n}_p{p_tag}_{collected:03d}"
            records.append(graph_to_record(G, graph_id=graph_id))
            collected += 1

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a mid-scale generalization ER pool for line B."
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count-n40-p008", type=int, default=160)
    parser.add_argument("--count-n50-p006", type=int, default=160)
    parser.add_argument("--count-n60-p005", type=int, default=160)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_midscale_generalization_pool_records(
        seed=args.seed,
        count_n40_p008=args.count_n40_p008,
        count_n50_p006=args.count_n50_p006,
        count_n60_p005=args.count_n60_p005,
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
        "dataset_type": "midscale_generalization_pool_b1",
        "seed": args.seed,
        "num_graphs": len(records),
        "train_count": len(manifest.train_ids),
        "val_count": len(manifest.val_ids),
        "test_count": len(manifest.test_ids),
        "combos": [
            {"n": 40, "p": 0.08, "count": args.count_n40_p008},
            {"n": 50, "p": 0.06, "count": args.count_n50_p006},
            {"n": 60, "p": 0.05, "count": args.count_n60_p005},
        ],
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Generated mid-scale generalization pool with {len(records)} graphs.")
    print(f"[OK] Output dir: {output_dir}")


if __name__ == "__main__":
    main()