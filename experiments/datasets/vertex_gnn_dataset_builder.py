from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

import torch

from MPC4plus.experiments.datasets.vertex_gnn_dataset import (
    VertexGNNTrainingDataset,
    build_vertex_gnn_example_from_live_inputs,
)
from MPC4plus.experiments.models.vertex_gnn_selector import CandidateGraphSpec
from MPC4plus.experiments.models.phase4_state_features import (
    NODE_FEATURE_MODE_LEGACY,
    build_phase4_node_features,
)


def _to_dict(obj: Any) -> dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    return dict(obj)


def _resolve_selected_candidate_idx(
    sample_dict: dict[str, Any],
    *,
    label_name: str,
) -> int | None:
    """
    Resolve positive index from candidate labels.

    label_name examples:
      - "selected_by_rule"
      - "selected_by_lookahead"
    """
    candidates = sample_dict["candidates"]

    positives: list[int] = []
    for idx, cand in enumerate(candidates):
        labels = cand.get("labels", {})
        if int(labels.get(label_name, 0)) == 1:
            positives.append(idx)

    if len(positives) == 1:
        return positives[0]
    if len(positives) > 1:
        raise ValueError(
            f"Multiple positive candidates for label_name={label_name!r} "
            f"in sample_id={sample_dict.get('sample_id')!r}"
        )

    # Backward-compatible fallback for legacy traces/tests.
    if label_name in {"selected_by_rule", "selected_by_lookahead"}:
        decision = sample_dict.get("decision", {})
        if "selected_candidate_idx" in decision and decision["selected_candidate_idx"] is not None:
            return int(decision["selected_candidate_idx"])

    return None


def build_vertex_gnn_example_from_trace_sample(
    sample: dict[str, Any],
    *,
    edge_list: list[tuple[int, int]],
    candidate_feature_names: list[str],
    label_name: str = "selected_by_lookahead",
    node_feature_mode: str = NODE_FEATURE_MODE_LEGACY,
) -> Any:
    """
    Build one VertexGNNTrainingExample from one exported Phase 4 decision sample.
    """
    import networkx as nx

    s = _to_dict(sample)

    G = nx.Graph()
    G.add_edges_from(edge_list)

    components = s["components"]
    candidates = s["candidates"]
    state = s["state"]

    metas = []
    for comp in components:
        satellites = []
        for sat in comp["satellites"]:
            satellites.append(
                _ns(
                    nodes=set(sat["nodes"]),
                    kind=sat["kind"],
                    rescue_edges=tuple(tuple(e) for e in sat["rescue_edges"]),
                    supporting_anchors=set(sat["supporting_anchors"]),
                    rescue_anchor=sat["rescue_anchor"],
                    is_critical_satellite=sat["is_critical_satellite"],
                )
            )

        metas.append(
            _ns(
                nodes=set(comp["nodes"]),
                is_composite=comp["is_composite"],
                center_nodes=set(comp["center_nodes"]),
                center_kind=comp["center_kind"],
                anchors=set(comp["anchors"]),
                j_anchor_map=_int_key_dict(comp["j_anchor_map"]),
                critical_2_anchors=set(comp["critical_2_anchors"]),
                responsible_1_anchors=set(comp["responsible_1_anchors"]),
                s_value=comp["s_value"],
                opt_value=comp["opt_value"],
                critical_ratio=comp["critical_ratio"],
                is_critical=comp["is_critical"],
                critical_case=comp["critical_case"],
                critical_reason=comp["critical_reason"],
                critical_signature=dict(comp["critical_signature"]),
                is_responsible_component=comp["is_responsible_component"],
                responsible_case=comp["responsible_case"],
                responsible_reason=comp["responsible_reason"],
                responsible_signature=dict(comp["responsible_signature"]),
                satellites=satellites,
                H_component_node_sets=[set(xs) for xs in comp.get("H_component_node_sets", [])],
            )
        )

    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    if state.get("H_edges"):
        H.add_edges_from((int(u), int(v)) for u, v in state.get("H_edges", []))
    else:
        # Backward-compatible fallback for old trace files that did not store
        # the full H edge list. New experiments should use H_edges exported by
        # Phase4TraceCollector to keep training and live inference aligned.
        for comp in components:
            center_nodes = list(comp["center_nodes"])
            if len(center_nodes) >= 2:
                for i in range(len(center_nodes) - 1):
                    H.add_edge(center_nodes[i], center_nodes[i + 1])

    C = {tuple(sorted((int(u), int(v)))) for u, v in state.get("C_edges", [])}
    M_C = {tuple(sorted((int(u), int(v)))) for u, v in state.get("M_C_edges", [])}

    fake_state = _ns(
        iteration=int(state["iteration"]),
        G=G,
        H=H,
        C=C,
        M_C=M_C,
        metas=metas,
    )

    node_list = sorted(G.nodes())
    node_to_idx = {u: i for i, u in enumerate(node_list)}

    x = build_phase4_node_features(
        G=fake_state.G,
        H=fake_state.H,
        C=set(fake_state.C),
        M_C=set(fake_state.M_C),
        metas=fake_state.metas,
        node_list=node_list,
        mode=node_feature_mode,
    )

    adj_norm = _build_row_normalized_adjacency(G, node_list)

    candidate_specs = []
    for cand in candidates:
        src_comp_idx = int(cand["source_component_idx"])
        tgt_comp_idx = int(cand["target_component_idx"])
        src_comp_nodes = sorted(components[src_comp_idx]["nodes"])
        tgt_comp_nodes = sorted(components[tgt_comp_idx]["nodes"])

        feat_dict = cand.get("features", {})
        cand_feat_vec = [float(feat_dict.get(name, 0.0)) for name in candidate_feature_names]

        candidate_specs.append(
            CandidateGraphSpec(
                source_idx=node_to_idx[int(cand["source_vertex"])],
                target_idx=node_to_idx[int(cand["target_vertex"])],
                source_component_node_indices=[node_to_idx[u] for u in src_comp_nodes],
                target_component_node_indices=[node_to_idx[u] for u in tgt_comp_nodes],
                candidate_feature_vector=cand_feat_vec,
            )
        )

    live_inputs = {
        "node_list": node_list,
        "node_features": x,
        "adj_norm": adj_norm,
        "candidate_specs": candidate_specs,
        "node_feature_mode": node_feature_mode,
    }

    selected_candidate_idx = _resolve_selected_candidate_idx(
        s,
        label_name=label_name,
    )

    return build_vertex_gnn_example_from_live_inputs(
        graph_id=s["graph_id"],
        sample_id=s["sample_id"],
        iteration=int(state["iteration"]),
        live_inputs=live_inputs,
        selected_candidate_idx=selected_candidate_idx,
    )


def build_vertex_gnn_dataset_from_trace_samples(
    samples: Iterable[dict[str, Any]],
    *,
    edge_map: dict[str, list[tuple[int, int]]],
    candidate_feature_names: list[str],
    label_name: str = "selected_by_lookahead",
    node_feature_mode: str = NODE_FEATURE_MODE_LEGACY,
) -> VertexGNNTrainingDataset:
    examples = []

    for sample in samples:
        s = _to_dict(sample)
        graph_id = str(s["graph_id"])
        if graph_id not in edge_map:
            raise KeyError(f"Missing edge list for graph_id={graph_id!r} in edge_map.")

        example = build_vertex_gnn_example_from_trace_sample(
            s,
            edge_list=edge_map[graph_id],
            candidate_feature_names=candidate_feature_names,
            label_name=label_name,
            node_feature_mode=node_feature_mode,
        )
        examples.append(example)

    return VertexGNNTrainingDataset(examples)


def _int_key_dict(raw: dict) -> dict[int, Any]:
    return {int(k): v for k, v in dict(raw).items()}


def _ns(**kwargs):
    from types import SimpleNamespace
    return SimpleNamespace(**kwargs)


def _build_row_normalized_adjacency(G, node_list: list[int]) -> torch.Tensor:
    n = len(node_list)
    node_to_idx = {u: i for i, u in enumerate(node_list)}

    adj = torch.zeros((n, n), dtype=torch.float32)
    for u in node_list:
        i = node_to_idx[u]
        adj[i, i] = 1.0
    for u, v in G.edges():
        i = node_to_idx[u]
        j = node_to_idx[v]
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    row_sums = adj.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0.0] = 1.0
    return adj / row_sums