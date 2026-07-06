from __future__ import annotations

import networkx as nx

from MPC4plus.experiments.datasets.graph_dataset_io import (
    GraphRecord,
    build_edge_map,
    graph_to_record,
    load_edge_map_json,
    load_graph_records_json,
    load_split_manifest_json,
    make_graph_split_manifest,
    record_to_graph,
    save_edge_map_json,
    save_graph_records_json,
    save_split_manifest_json,
    select_records_by_ids,
)


def _build_demo_graph_1() -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4)])
    return G


def _build_demo_graph_2() -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from([(5, 6), (6, 7), (7, 8), (5, 8)])
    return G


def test_graph_to_record_and_back() -> None:
    G = _build_demo_graph_1()
    record = graph_to_record(G, graph_id="g1")
    G2 = record_to_graph(record)

    assert record.graph_id == "g1"
    assert set(G.edges()) == set(G2.edges())


def test_save_and_load_graph_records_json(tmp_path) -> None:
    records = [
        graph_to_record(_build_demo_graph_1(), graph_id="g1"),
        graph_to_record(_build_demo_graph_2(), graph_id="g2"),
    ]
    path = tmp_path / "graphs.json"

    save_graph_records_json(str(path), records)
    loaded = load_graph_records_json(str(path))

    assert len(loaded) == 2
    assert loaded[0].graph_id == "g1"
    assert loaded[1].graph_id == "g2"


def test_build_and_save_edge_map_json(tmp_path) -> None:
    records = [
        graph_to_record(_build_demo_graph_1(), graph_id="g1"),
        graph_to_record(_build_demo_graph_2(), graph_id="g2"),
    ]
    edge_map = build_edge_map(records)

    path = tmp_path / "edge_map.json"
    save_edge_map_json(str(path), edge_map)
    loaded = load_edge_map_json(str(path))

    assert set(loaded.keys()) == {"g1", "g2"}
    assert len(loaded["g1"]) == len(records[0].edges)
    assert len(loaded["g2"]) == len(records[1].edges)


def test_make_graph_split_manifest() -> None:
    graph_ids = [f"g{i}" for i in range(10)]
    manifest = make_graph_split_manifest(
        graph_ids,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=123,
    )

    assert len(manifest.train_ids) + len(manifest.val_ids) + len(manifest.test_ids) == 10
    assert set(manifest.train_ids).isdisjoint(set(manifest.val_ids))
    assert set(manifest.train_ids).isdisjoint(set(manifest.test_ids))
    assert set(manifest.val_ids).isdisjoint(set(manifest.test_ids))


def test_save_and_load_split_manifest_json(tmp_path) -> None:
    graph_ids = [f"g{i}" for i in range(6)]
    manifest = make_graph_split_manifest(
        graph_ids,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=7,
    )

    path = tmp_path / "manifest.json"
    save_split_manifest_json(str(path), manifest)
    loaded = load_split_manifest_json(str(path))

    assert loaded.train_ids == manifest.train_ids
    assert loaded.val_ids == manifest.val_ids
    assert loaded.test_ids == manifest.test_ids


def test_select_records_by_ids() -> None:
    records = [
        GraphRecord(graph_id="g1", edges=[(1, 2)]),
        GraphRecord(graph_id="g2", edges=[(2, 3)]),
        GraphRecord(graph_id="g3", edges=[(3, 4)]),
    ]
    selected = select_records_by_ids(records, ["g1", "g3"])

    assert [r.graph_id for r in selected] == ["g1", "g3"]