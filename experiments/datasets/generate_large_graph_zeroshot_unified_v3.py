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
    build_edge_map,
    graph_to_record,
    save_edge_map_json,
    save_graph_records_json,
)


def build_large_graph_zeroshot_unified_v3_records(
    *,
    seed: int = 314159,
    count_n200: int = 30,
    count_n500: int = 30,
    count_n800: int = 30,
    count_n1000: int = 30,
    p_n200: float = 0.025,
    p_n500: float = 0.012,
    p_n800: float = 0.008,
    p_n1000: float = 0.006,
) -> list:
    rng = random.Random(seed)
    specs = [
        (200, p_n200, count_n200),
        (500, p_n500, count_n500),
        (800, p_n800, count_n800),
        (1000, p_n1000, count_n1000),
    ]

    records = []
    for n, p, target_count in specs:
        if target_count <= 0:
            continue

        p_tag = str(p).replace(".", "p")
        collected = 0
        attempts = 0
        max_attempts = max(1000, target_count * 1000)

        while collected < target_count:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Too many attempts while building large zeroshot set for n={n}, p={p}. "
                    f"Collected {collected}/{target_count}."
                )

            G = generate_connected_er_graph(n=n, p=p, rng=rng)
            if not is_harder_graph_proxy(G):
                continue

            graph_id = f"largezero_er_n{n}_p{p_tag}_{collected:03d}"
            records.append(graph_to_record(G, graph_id=graph_id))
            collected += 1

        print(f"[OK] n={n}, p={p}, collected={collected}, attempts={attempts}")

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the final unified large-graph zero-shot test set for n=200/500/800/1000."
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=314159)

    parser.add_argument("--count-n200", type=int, default=30)
    parser.add_argument("--count-n500", type=int, default=30)
    parser.add_argument("--count-n800", type=int, default=30)
    parser.add_argument("--count-n1000", type=int, default=30)

    parser.add_argument("--p-n200", type=float, default=0.025)
    parser.add_argument("--p-n500", type=float, default=0.012)
    parser.add_argument("--p-n800", type=float, default=0.008)
    parser.add_argument("--p-n1000", type=float, default=0.006)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_large_graph_zeroshot_unified_v3_records(
        seed=args.seed,
        count_n200=args.count_n200,
        count_n500=args.count_n500,
        count_n800=args.count_n800,
        count_n1000=args.count_n1000,
        p_n200=args.p_n200,
        p_n500=args.p_n500,
        p_n800=args.p_n800,
        p_n1000=args.p_n1000,
    )

    edge_map = build_edge_map(records)

    save_graph_records_json(str(output_dir / "graphs.json"), records)
    save_edge_map_json(str(output_dir / "edge_map.json"), edge_map)

    split = {
        "train_ids": [],
        "val_ids": [],
        "test_ids": [r.graph_id for r in records],
    }
    (output_dir / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    combos = [
        {"n": 200, "p": args.p_n200, "count": args.count_n200},
        {"n": 500, "p": args.p_n500, "count": args.count_n500},
        {"n": 800, "p": args.p_n800, "count": args.count_n800},
        {"n": 1000, "p": args.p_n1000, "count": args.count_n1000},
    ]

    dataset_info = {
        "dataset_type": "large_graph_zeroshot_unified_v3",
        "seed": args.seed,
        "num_graphs": len(records),
        "test_only": True,
        "combos": combos,
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Generated final large-graph zero-shot test set with {len(records)} graphs.")
    print(f"[OK] Output dir: {output_dir}")


if __name__ == "__main__":
    main()
