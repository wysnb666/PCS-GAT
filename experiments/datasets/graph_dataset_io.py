from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx


Edge = tuple[int, int]


@dataclass(frozen=True)
class GraphRecord:
    graph_id: str
    edges: list[Edge]


@dataclass(frozen=True)
class GraphSplitManifest:
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]


def normalize_edge(u: int, v: int) -> Edge:
    return (u, v) if u <= v else (v, u)


def graph_to_record(G: nx.Graph, *, graph_id: str) -> GraphRecord:
    edges = [normalize_edge(int(u), int(v)) for u, v in G.edges()]
    edges = sorted(set(edges))
    return GraphRecord(graph_id=graph_id, edges=edges)


def record_to_graph(record: GraphRecord | dict) -> nx.Graph:
    if isinstance(record, dict):
        graph_id = str(record["graph_id"])
        edges = [(int(u), int(v)) for (u, v) in record["edges"]]
        record = GraphRecord(graph_id=graph_id, edges=edges)

    G = nx.Graph()
    G.add_edges_from(record.edges)
    return G


def save_graph_records_json(path: str, records: Iterable[GraphRecord]) -> None:
    payload = [asdict(r) for r in records]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_graph_records_json(path: str) -> list[GraphRecord]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[GraphRecord] = []
    for item in raw:
        out.append(
            GraphRecord(
                graph_id=str(item["graph_id"]),
                edges=[normalize_edge(int(u), int(v)) for (u, v) in item["edges"]],
            )
        )
    return out


def build_edge_map(records: Iterable[GraphRecord]) -> dict[str, list[Edge]]:
    edge_map: dict[str, list[Edge]] = {}
    for record in records:
        edge_map[record.graph_id] = list(record.edges)
    return edge_map


def save_edge_map_json(path: str, edge_map: dict[str, list[Edge]]) -> None:
    payload = {
        str(graph_id): [[int(u), int(v)] for (u, v) in edges]
        for graph_id, edges in edge_map.items()
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_edge_map_json(path: str) -> dict[str, list[Edge]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    edge_map: dict[str, list[Edge]] = {}
    for graph_id, edges in raw.items():
        edge_map[str(graph_id)] = [normalize_edge(int(u), int(v)) for (u, v) in edges]
    return edge_map


def make_graph_split_manifest(
    graph_ids: list[str],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> GraphSplitManifest:
    if not graph_ids:
        return GraphSplitManifest(train_ids=[], val_ids=[], test_ids=[])

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    ids = list(graph_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:n_train + n_val + n_test]

    return GraphSplitManifest(
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )


def save_split_manifest_json(path: str, manifest: GraphSplitManifest) -> None:
    payload = asdict(manifest)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_split_manifest_json(path: str) -> GraphSplitManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return GraphSplitManifest(
        train_ids=[str(x) for x in raw["train_ids"]],
        val_ids=[str(x) for x in raw["val_ids"]],
        test_ids=[str(x) for x in raw["test_ids"]],
    )


def select_records_by_ids(
    records: Iterable[GraphRecord],
    graph_ids: Iterable[str],
) -> list[GraphRecord]:
    wanted = set(str(gid) for gid in graph_ids)
    return [r for r in records if r.graph_id in wanted]