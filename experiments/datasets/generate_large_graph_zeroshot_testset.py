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


def build_large_graph_zeroshot_records(
    *,
    seed: int = 314159,
    count_n100: int = 10,
    count_n150: int = 10,
    count_n200: int = 10,
    count_n500: int = 0,
    p_n100: float = 0.05,
    p_n150: float = 0.035,
    p_n200: float = 0.025,
    p_n500: float = 0.012,
) -> list:
    """
    Build a shared test-only zero-shot dataset for larger graphs.

    Defaults preserve the user's current 100/150/200 setup and add an optional
    n=500 bucket that can be enabled by setting count_n500 > 0.

    Notes:
      - We keep the same structural hardness proxy used in earlier larger-graph tests.
      - p_n500 is intentionally exposed as a CLI parameter because the best stable
        value may depend on the user's compute budget and desired Phase 4 density.
    """
    rng = random.Random(seed)
    specs = [
        (100, p_n100, count_n100),
        (150, p_n150, count_n150),
        (200, p_n200, count_n200),
        (500, p_n500, count_n500),
    ]

    records = []
    for n, p, target_count in specs:
        if target_count <= 0:
            continue

        p_tag = str(p).replace(".", "p")
        collected = 0
        attempts = 0

        while collected < target_count:
            attempts += 1
            if attempts > target_count * 600:
                raise RuntimeError(
                    f"Too many attempts while building large zeroshot set for n={n}, p={p}"
                )

            G = generate_connected_er_graph(n=n, p=p, rng=rng)
            if not is_harder_graph_proxy(G):
                continue

            graph_id = f"largezero_er_n{n}_p{p_tag}_{collected:03d}"
            records.append(graph_to_record(G, graph_id=graph_id))
            collected += 1

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a shared large-graph test-only dataset for zero-shot extrapolation."
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--count-n100", type=int, default=10)
    parser.add_argument("--count-n150", type=int, default=10)
    parser.add_argument("--count-n200", type=int, default=10)
    parser.add_argument(
        "--count-n500",
        type=int,
        default=0,
        help="Optional number of n=500 graphs to add to the shared zero-shot test set.",
    )
    parser.add_argument("--p-n100", type=float, default=0.05)
    parser.add_argument("--p-n150", type=float, default=0.035)
    parser.add_argument("--p-n200", type=float, default=0.025)
    parser.add_argument(
        "--p-n500",
        type=float,
        default=0.012,
        help="ER edge probability for optional n=500 graphs.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_large_graph_zeroshot_records(
        seed=args.seed,
        count_n100=args.count_n100,
        count_n150=args.count_n150,
        count_n200=args.count_n200,
        count_n500=args.count_n500,
        p_n100=args.p_n100,
        p_n150=args.p_n150,
        p_n200=args.p_n200,
        p_n500=args.p_n500,
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

    combos = []
    if args.count_n100 > 0:
        combos.append({"n": 100, "p": args.p_n100, "count": args.count_n100})
    if args.count_n150 > 0:
        combos.append({"n": 150, "p": args.p_n150, "count": args.count_n150})
    if args.count_n200 > 0:
        combos.append({"n": 200, "p": args.p_n200, "count": args.count_n200})
    if args.count_n500 > 0:
        combos.append({"n": 500, "p": args.p_n500, "count": args.count_n500})

    dataset_info = {
        "dataset_type": "large_graph_zeroshot_testset",
        "seed": args.seed,
        "num_graphs": len(records),
        "test_only": True,
        "combos": combos,
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Generated large-graph zero-shot test set with {len(records)} graphs.")
    print(f"[OK] Output dir: {output_dir}")


if __name__ == "__main__":
    main()