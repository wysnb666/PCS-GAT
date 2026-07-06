from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx

from MPC4plus.experiments.datasets.graph_dataset_io import (
    GraphRecord,
    build_edge_map,
    graph_to_record,
    make_graph_split_manifest,
    save_edge_map_json,
    save_graph_records_json,
    save_split_manifest_json,
)


def generate_connected_er_graph(
    *,
    n: int,
    p: float,
    rng: random.Random,
    max_tries: int = 200,
) -> nx.Graph:
    """
    Generate a connected ER graph G(n, p).

    We retry until the graph is connected and has at least one edge.
    """
    for _ in range(max_tries):
        seed = rng.randint(0, 10**9)
        G = nx.gnp_random_graph(n=n, p=p, seed=seed)
        if G.number_of_edges() == 0:
            continue
        if nx.is_connected(G):
            return G

    raise RuntimeError(
        f"Failed to generate connected ER graph after {max_tries} tries for n={n}, p={p}."
    )


def build_first_round_er_records(
    *,
    seed: int = 42,
    graphs_per_combo: int = 40,
) -> list[GraphRecord]:
    """
    First-round experiment graph set:

      n in {16, 20, 30, 40}
      p:
        16 -> {0.18, 0.24}
        20 -> {0.14, 0.20}
        30 -> {0.10, 0.14}
        40 -> {0.08, 0.12}
      each (n, p) -> 40 graphs by default

    Total default graph count:
      4 * 2 * 40 = 320
    """
    rng = random.Random(seed)

    combos = [
        (16, 0.18),
        (16, 0.24),
        (20, 0.14),
        (20, 0.20),
        (30, 0.10),
        (30, 0.14),
        (40, 0.08),
        (40, 0.12),
    ]

    records: list[GraphRecord] = []
    for n, p in combos:
        p_tag = str(p).replace(".", "p")
        for idx in range(graphs_per_combo):
            G = generate_connected_er_graph(n=n, p=p, rng=rng)
            graph_id = f"er_n{n}_p{p_tag}_{idx:03d}"
            records.append(graph_to_record(G, graph_id=graph_id))

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the first-round ER graph dataset and split manifest."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write graphs.json, edge_map.json, split.json, dataset_info.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--graphs-per-combo",
        type=int,
        default=40,
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

    records = build_first_round_er_records(
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
        "dataset_type": "first_round_er",
        "seed": args.seed,
        "graphs_per_combo": args.graphs_per_combo,
        "num_graphs": len(records),
        "train_count": len(manifest.train_ids),
        "val_count": len(manifest.val_ids),
        "test_count": len(manifest.test_ids),
        "combos": [
            {"n": 16, "p": 0.18},
            {"n": 16, "p": 0.24},
            {"n": 20, "p": 0.14},
            {"n": 20, "p": 0.20},
            {"n": 30, "p": 0.10},
            {"n": 30, "p": 0.14},
            {"n": 40, "p": 0.08},
            {"n": 40, "p": 0.12},
        ],
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Generated dataset with {len(records)} graphs.")
    print(f"[OK] Output dir: {output_dir}")


if __name__ == "__main__":
    main()