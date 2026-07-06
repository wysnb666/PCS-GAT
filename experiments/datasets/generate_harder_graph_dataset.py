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
    max_tries: int = 500,
) -> nx.Graph:
    for _ in range(max_tries):
        seed = rng.randint(0, 10**9)
        G = nx.gnp_random_graph(n=n, p=p, seed=seed)
        if G.number_of_edges() == 0:
            continue
        if nx.is_connected(G):
            return G
    raise RuntimeError(f"Failed to generate connected ER graph for n={n}, p={p}")


def is_harder_graph_proxy(G: nx.Graph) -> bool:
    """
    First-stage proxy filter for a 'harder' graph.

    We do NOT try to exactly predict Phase 4 conflict here.
    We use a lightweight structural filter to avoid graphs that are:
      - too tree-like / too sparse
      - too dense / too easy
      - too regular/simple

    This is just the first step before later adding
    Phase4-aware filtering if needed.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n < 10:
        return False

    avg_deg = (2.0 * m) / n
    degs = [d for _, d in G.degree()]
    max_deg = max(degs)
    min_deg = min(degs)
    num_leaves = sum(1 for d in degs if d == 1)

    density = nx.density(G)

    # avoid too sparse / too trivial
    if avg_deg < 2.4:
        return False

    # avoid too dense
    if density > 0.28:
        return False

    # avoid near-regular / too uniform simple shapes
    if max_deg - min_deg <= 1:
        return False

    # avoid too many leaves
    if num_leaves > n // 3:
        return False

    # prefer some hub/bridge flavor
    if max_deg < 5:
        return False

    return True


def build_harder_round_v1_records(
    *,
    seed: int = 42,
    graphs_per_combo: int = 30,
) -> list[GraphRecord]:
    """
    Harder round v1:
      sizes shifted upward
      still ER-based
      filtered by a structural hardness proxy

    Suggested combos:
      (30, 0.10), (30, 0.14)
      (40, 0.08), (40, 0.12)
      (60, 0.06), (60, 0.10)
      (80, 0.05), (80, 0.08)

    Default total target:
      8 combos * 30 = 240 graphs
    """
    rng = random.Random(seed)
    combos = [
        (30, 0.10),
        (30, 0.14),
        (40, 0.08),
        (40, 0.12),
        (60, 0.06),
        (60, 0.10),
        (80, 0.05),
        (80, 0.08),
    ]

    records: list[GraphRecord] = []
    for n, p in combos:
        p_tag = str(p).replace(".", "p")
        collected = 0
        attempts = 0

        while collected < graphs_per_combo:
            attempts += 1
            if attempts > graphs_per_combo * 200:
                raise RuntimeError(
                    f"Too many attempts while building harder graphs for n={n}, p={p}"
                )

            G = generate_connected_er_graph(n=n, p=p, rng=rng)
            if not is_harder_graph_proxy(G):
                continue

            graph_id = f"harder_er_n{n}_p{p_tag}_{collected:03d}"
            records.append(graph_to_record(G, graph_id=graph_id))
            collected += 1

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate harder-round graph dataset with structural filtering."
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
        default=30,
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

    records = build_harder_round_v1_records(
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
        "dataset_type": "harder_round_v1",
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
            {"n": 40, "p": 0.12},
            {"n": 60, "p": 0.06},
            {"n": 60, "p": 0.10},
            {"n": 80, "p": 0.05},
            {"n": 80, "p": 0.08},
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

    print(f"[OK] Generated harder dataset with {len(records)} graphs.")
    print(f"[OK] Output dir: {output_dir}")


if __name__ == "__main__":
    main()