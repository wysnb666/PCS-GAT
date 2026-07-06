from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MPC4plus.experiments.datasets.graph_dataset_io import (
    GraphSplitManifest,
    load_graph_records_json,
    save_graph_records_json,
    save_split_manifest_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a graph subset by graph_id from an existing graphs.json."
    )
    parser.add_argument("--graphs-json", type=str, required=True)
    parser.add_argument(
        "--graph-ids",
        nargs="+",
        required=True,
        help="One or more graph ids to extract.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = load_graph_records_json(args.graphs_json)
    wanted_ids = list(args.graph_ids)
    wanted_set = set(wanted_ids)

    by_id = {r.graph_id: r for r in records}
    missing = [gid for gid in wanted_ids if gid not in by_id]
    if missing:
        raise ValueError(f"Missing graph ids in graphs.json: {missing}")

    selected_records = [by_id[gid] for gid in wanted_ids]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_graph_records_json(str(output_dir / "graphs.json"), selected_records)
    save_split_manifest_json(
        str(output_dir / "split.json"),
        GraphSplitManifest(train_ids=[], val_ids=[], test_ids=wanted_ids),
    )

    info = {
        "num_graphs": len(selected_records),
        "graph_ids": wanted_ids,
        "source_graphs_json": args.graphs_json,
    }
    (output_dir / "subset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Extracted {len(selected_records)} graph(s).")
    print(f"[OK] Output dir: {output_dir}")
    print(f"[OK] Graph ids: {wanted_ids}")


if __name__ == "__main__":
    main()