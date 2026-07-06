from __future__ import annotations

import json

import pytest

from MPC4plus.experiments.datasets.graph_dataset_io import (
    graph_to_record,
    make_graph_split_manifest,
    save_graph_records_json,
    save_split_manifest_json,
)
from MPC4plus.experiments.export.export_phase4_split_data import (
    export_phase4_split_data,
)


def _build_demo_graph():
    import networkx as nx

    G = nx.Graph()
    G.add_edges_from(
        [
            (1, 2),
            (3, 4),
            (5, 6),
            (7, 8),
            (1, 6),
            (2, 5),
            (3, 8),
            (4, 7),
            (2, 9),
            (9, 10),
            (4, 11),
            (11, 12),
            (6, 13),
            (8, 14),
        ]
    )
    return G


def test_export_phase4_split_data_outputs_files(tmp_path) -> None:
    record = graph_to_record(_build_demo_graph(), graph_id="g_demo")

    graphs_json = tmp_path / "graphs.json"
    split_json = tmp_path / "split.json"
    out_dir = tmp_path / "exports"

    save_graph_records_json(str(graphs_json), [record])

    manifest = make_graph_split_manifest(
        ["g_demo"],
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=42,
    )
    save_split_manifest_json(str(split_json), manifest)

    info = export_phase4_split_data(
        graphs_json=str(graphs_json),
        split_json=str(split_json),
        split_name="train",
        output_dir=str(out_dir),
        backend="ilp",
    )

    assert info["split_name"] == "train"
    assert info["num_graphs"] == 1

    trace_path = out_dir / "phase4_trace.jsonl"
    cand_jsonl_path = out_dir / "phase4_candidates.jsonl"
    cand_csv_path = out_dir / "phase4_candidates.csv"
    export_info_path = out_dir / "export_info.json"

    assert trace_path.exists()
    assert cand_jsonl_path.exists()
    assert cand_csv_path.exists()
    assert export_info_path.exists()


def test_export_phase4_split_data_export_info_roundtrip(tmp_path) -> None:
    record = graph_to_record(_build_demo_graph(), graph_id="g_demo")

    graphs_json = tmp_path / "graphs.json"
    split_json = tmp_path / "split.json"
    out_dir = tmp_path / "exports"

    save_graph_records_json(str(graphs_json), [record])

    manifest = make_graph_split_manifest(
        ["g_demo"],
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=42,
    )
    save_split_manifest_json(str(split_json), manifest)

    export_phase4_split_data(
        graphs_json=str(graphs_json),
        split_json=str(split_json),
        split_name="train",
        output_dir=str(out_dir),
        backend="ilp",
    )

    export_info = json.loads((out_dir / "export_info.json").read_text(encoding="utf-8"))
    assert export_info["split_name"] == "train"
    assert export_info["num_graphs"] == 1
    assert "num_trace_samples" in export_info
    assert "num_candidate_rows" in export_info


def test_export_phase4_split_data_rejects_bad_split_name(tmp_path) -> None:
    record = graph_to_record(_build_demo_graph(), graph_id="g_demo")

    graphs_json = tmp_path / "graphs.json"
    split_json = tmp_path / "split.json"
    out_dir = tmp_path / "exports"

    save_graph_records_json(str(graphs_json), [record])

    manifest = make_graph_split_manifest(
        ["g_demo"],
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=42,
    )
    save_split_manifest_json(str(split_json), manifest)

    with pytest.raises(ValueError):
        export_phase4_split_data(
            graphs_json=str(graphs_json),
            split_json=str(split_json),
            split_name="bad_split",
            output_dir=str(out_dir),
            backend="ilp",
        )