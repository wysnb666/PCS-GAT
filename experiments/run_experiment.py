from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import torch

from MPC4plus.experiments.baselines.xgb_ranker import XGBPhase4Ranker
from MPC4plus.experiments.datasets.graph_dataset_io import (
    GraphRecord,
    load_graph_records_json,
    load_split_manifest_json,
    record_to_graph,
    select_records_by_ids,
)
from MPC4plus.experiments.evaluation.end_to_end_compare import (
    add_relative_improvement_vs_rule,
    compare_rule_heuristic_xgb_gnn_end_to_end,
)
from MPC4plus.experiments.models.vertex_gnn_selector import VertexGNNCandidateScorer
from MPC4plus.experiments.models.phase4_state_features import (
    NODE_FEATURE_MODE_LEGACY,
    get_node_feature_dim,
)


def build_default_xgb_ranker() -> XGBPhase4Ranker:
    """
    Build a default unfitted XGBoost ranker.

    This is useful for later `load_xgb_ranker(...)`.
    """
    return XGBPhase4Ranker(
        label_name="label__selected_by_rule",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
    )


def build_default_vertex_gnn_model(
    *,
    candidate_feature_names: list[str] | None = None,
    hidden_dim: int = 32,
    num_layers: int = 2,
    encoder_type: str = "mean",
    node_feature_mode: str = NODE_FEATURE_MODE_LEGACY,
) -> VertexGNNCandidateScorer:
    """
    Build a default vertex-level GNN model instance.
    """
    return VertexGNNCandidateScorer(
        candidate_feature_names=candidate_feature_names or ["expected_g_drop"],
        node_input_dim=get_node_feature_dim(node_feature_mode),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        encoder_type=encoder_type,
        node_feature_mode=node_feature_mode,
    )


def load_xgb_ranker(
    *,
    model_path: str,
    feature_names: list[str],
    label_name: str = "label__selected_by_rule",
) -> XGBPhase4Ranker:
    """
    Load a trained XGBoost ranker from file.

    Important:
    - XGBPhase4Ranker currently stores its model weights via xgboost native save_model
    - feature_names are restored from experiment metadata, not from the model file itself
    """
    ranker = XGBPhase4Ranker(label_name=label_name, feature_names=list(feature_names))
    ranker.load_model(model_path)
    ranker._fitted_feature_names = list(feature_names)
    return ranker


def load_vertex_gnn_model(
    *,
    checkpoint_path: str,
    device: str = "cpu",
) -> VertexGNNCandidateScorer:
    """
    Load a trained vertex-level GNN model from checkpoint created by
    save_vertex_gnn_checkpoint(...).
    """
    payload = torch.load(checkpoint_path, map_location=device)
    metadata = payload.get("metadata", {})

    candidate_feature_names = list(metadata.get("candidate_feature_names", ["expected_g_drop"]))
    hidden_dim = int(metadata.get("hidden_dim", 32))
    num_layers = int(metadata.get("num_layers", 2))
    encoder_type = str(metadata.get("encoder_type", "mean"))
    node_feature_mode = str(metadata.get("node_feature_mode", NODE_FEATURE_MODE_LEGACY))
    node_input_dim = int(metadata.get("node_input_dim", get_node_feature_dim(node_feature_mode)))

    model = VertexGNNCandidateScorer(
        candidate_feature_names=candidate_feature_names,
        node_input_dim=node_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        encoder_type=encoder_type,
        node_feature_mode=node_feature_mode,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def load_xgb_metadata(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_end_to_end_experiment_on_records(
    records: list[GraphRecord],
    *,
    ranker: XGBPhase4Ranker,
    gnn_model: VertexGNNCandidateScorer,
    backend: str = "ilp",
    device: str = "cpu",
    gnn_conservative_margin: float | None = None,
) -> list[dict[str, Any]]:
    """
    Run end-to-end comparison on a list of graph records.

    Returns a flat list of result rows, one per selector per graph.
    """
    rows: list[dict[str, Any]] = []

    for record in records:
        G: nx.Graph = record_to_graph(record)

        metrics = compare_rule_heuristic_xgb_gnn_end_to_end(
            G,
            ranker=ranker,
            gnn_model=gnn_model,
            graph_id=record.graph_id,
            backend=backend,
            device=device,
            gnn_conservative_margin=gnn_conservative_margin,
        )
        rows.extend(add_relative_improvement_vs_rule(metrics))

    return rows


def save_experiment_rows_json(path: str, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run end-to-end selector comparison on a graph dataset split."
    )
    parser.add_argument(
        "--graphs-json",
        type=str,
        required=True,
        help="Path to graph records JSON.",
    )
    parser.add_argument(
        "--split-json",
        type=str,
        required=True,
        help="Path to train/val/test split manifest JSON.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Which split to run.",
    )
    parser.add_argument(
        "--xgb-model",
        type=str,
        required=True,
        help="Path to trained XGBoost model file.",
    )
    parser.add_argument(
        "--xgb-metadata",
        type=str,
        required=True,
        help="Path to XGBoost training metadata JSON.",
    )
    parser.add_argument(
        "--gnn-checkpoint",
        type=str,
        required=True,
        help="Path to trained vertex-level GNN checkpoint.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Where to save experiment result rows.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ilp",
        choices=["ilp", "bruteforce"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--gnn-conservative-margin",
        type=float,
        default=None,
        help=(
            "If set, use ConservativeVertexGNNPhase4Selector and only let GNN "
            "override canonical rule when top1-top2 score gap >= this margin."
        ),
    )

    args = parser.parse_args()

    records = load_graph_records_json(args.graphs_json)
    manifest = load_split_manifest_json(args.split_json)

    if args.split == "train":
        selected_ids = manifest.train_ids
    elif args.split == "val":
        selected_ids = manifest.val_ids
    else:
        selected_ids = manifest.test_ids

    selected_records = select_records_by_ids(records, selected_ids)

    xgb_metadata = load_xgb_metadata(args.xgb_metadata)
    xgb_feature_names = list(xgb_metadata.get("feature_names", []))
    xgb_label_name = str(xgb_metadata.get("label_name", "label__selected_by_rule"))

    if not xgb_feature_names:
        raise ValueError(
            "XGBoost metadata must contain non-empty feature_names to restore the ranker."
        )

    ranker = load_xgb_ranker(
        model_path=args.xgb_model,
        feature_names=xgb_feature_names,
        label_name=xgb_label_name,
    )
    gnn_model = load_vertex_gnn_model(
        checkpoint_path=args.gnn_checkpoint,
        device=args.device,
    )

    rows = run_end_to_end_experiment_on_records(
        selected_records,
        ranker=ranker,
        gnn_model=gnn_model,
        backend=args.backend,
        device=args.device,
        gnn_conservative_margin=args.gnn_conservative_margin,
    )
    save_experiment_rows_json(args.output_json, rows)

    print(f"[OK] Ran experiment on split={args.split} with {len(selected_records)} graphs.")
    print(f"[OK] Result rows saved to: {args.output_json}")


if __name__ == "__main__":
    main()