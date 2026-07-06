from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MPC4plus.experiments.datasets.vertex_gnn_dataset_builder import (
    build_vertex_gnn_dataset_from_trace_samples,
)
from MPC4plus.experiments.models.vertex_gnn_selector import VertexGNNCandidateScorer
from MPC4plus.experiments.models.phase4_state_features import (
    NODE_FEATURE_MODE_LEGACY,
    NODE_FEATURE_MODE_PATH_V1,
    get_node_feature_dim,
    get_node_feature_names,
)


def load_trace_samples_from_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_edge_map_from_json(path: str) -> dict[str, list[tuple[int, int]]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    edge_map: dict[str, list[tuple[int, int]]] = {}
    for graph_id, edges in raw.items():
        edge_map[str(graph_id)] = [(int(u), int(v)) for (u, v) in edges]
    return edge_map


def resolve_device(device_arg: str) -> str:
    device_arg = device_arg.lower().strip()
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested device='cuda' but torch.cuda.is_available() is False.")
        return "cuda"
    if device_arg == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device argument: {device_arg}")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_vertex_gnn_epoch(
    model: VertexGNNCandidateScorer,
    dataset,
    *,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> float:
    model.train()
    model.to(device)

    if len(dataset) == 0:
        return 0.0

    total_loss = 0.0
    num_examples = 0

    for example in dataset.examples():
        if example.selected_candidate_idx is None:
            continue
        if len(example.candidate_specs) == 0:
            continue

        node_features = example.node_features.to(device)
        adj_norm = example.adj_norm.to(device)

        logits = model(
            node_features=node_features,
            adj_norm=adj_norm,
            candidate_specs=example.candidate_specs,
        ).unsqueeze(0)

        target = torch.tensor(
            [example.selected_candidate_idx],
            dtype=torch.long,
            device=device,
        )

        loss = torch.nn.functional.cross_entropy(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_examples += 1

    if num_examples == 0:
        return 0.0
    return total_loss / num_examples


def evaluate_vertex_gnn_top1_accuracy(
    model: VertexGNNCandidateScorer,
    dataset,
    *,
    device: str = "cpu",
) -> float:
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for example in dataset.examples():
            if example.selected_candidate_idx is None:
                continue
            if len(example.candidate_specs) == 0:
                continue

            logits = model(
                node_features=example.node_features.to(device),
                adj_norm=example.adj_norm.to(device),
                candidate_specs=example.candidate_specs,
            )
            pred = int(torch.argmax(logits).item())

            total += 1
            if pred == int(example.selected_candidate_idx):
                correct += 1

    if total == 0:
        return 0.0
    return float(correct) / float(total)


def save_vertex_gnn_checkpoint(
    model: VertexGNNCandidateScorer,
    *,
    path: str,
    metadata: dict,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the vertex-level GNN selector."
    )
    parser.add_argument(
        "--trace-jsonl",
        type=str,
        required=True,
        help="Path to exported structured Phase 4 trace JSONL.",
    )
    parser.add_argument(
        "--edge-map-json",
        type=str,
        required=True,
        help="Path to graph_id -> edge list JSON.",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        required=True,
        help="Output checkpoint path.",
    )
    parser.add_argument(
        "--metadata-out",
        type=str,
        required=True,
        help="Output metadata JSON path.",
    )
    parser.add_argument(
        "--candidate-feature-names",
        type=str,
        nargs="+",
        default=["expected_g_drop"],
        help="Candidate feature names to feed into the GNN scorer.",
    )
    parser.add_argument(
        "--label-name",
        type=str,
        default="selected_by_lookahead",
        help="Trace-level label name. Use selected_by_lookahead for the new training logic.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--node-feature-mode",
        type=str,
        default=NODE_FEATURE_MODE_LEGACY,
        choices=[NODE_FEATURE_MODE_LEGACY, NODE_FEATURE_MODE_PATH_V1],
        help="Vertex-state feature set. Use path_v1 for explicit path-aware features.",
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        default="mean",
        choices=["mean", "gat"],
        help="GNN encoder variant. Use gat for the attention ablation.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device. 'auto' selects cuda if available, else cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for GNN training.",
    )

    args = parser.parse_args()

    device = resolve_device(args.device)
    set_global_seed(args.seed)

    samples = load_trace_samples_from_jsonl(args.trace_jsonl)
    edge_map = load_edge_map_from_json(args.edge_map_json)

    dataset = build_vertex_gnn_dataset_from_trace_samples(
        samples,
        edge_map=edge_map,
        candidate_feature_names=list(args.candidate_feature_names),
        label_name=args.label_name,
        node_feature_mode=args.node_feature_mode,
    )

    model = VertexGNNCandidateScorer(
        candidate_feature_names=list(args.candidate_feature_names),
        node_input_dim=get_node_feature_dim(args.node_feature_mode),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        encoder_type=args.encoder_type,
        node_feature_mode=args.node_feature_mode,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses: list[float] = []
    for _ in range(args.epochs):
        loss = train_vertex_gnn_epoch(
            model,
            dataset,
            optimizer=optimizer,
            device=device,
        )
        losses.append(loss)

    acc = evaluate_vertex_gnn_top1_accuracy(
        model,
        dataset,
        device=device,
    )

    metadata = {
        "num_examples": dataset.num_examples(),
        "num_positive_labels": dataset.num_positive_examples(),
        "candidate_feature_names": list(args.candidate_feature_names),
        "label_name": args.label_name,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "encoder_type": args.encoder_type,
        "node_feature_mode": args.node_feature_mode,
        "node_input_dim": get_node_feature_dim(args.node_feature_mode),
        "node_feature_names": get_node_feature_names(args.node_feature_mode),
        "lr": args.lr,
        "device": device,
        "requested_device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "seed": args.seed,
        "final_train_loss": losses[-1] if losses else 0.0,
        "train_top1_accuracy": acc,
        "dataset_summary": dataset.summary(),
    }

    save_vertex_gnn_checkpoint(
        model,
        path=args.model_out,
        metadata=metadata,
    )

    metadata_out = Path(args.metadata_out)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Trained vertex-level GNN on {dataset.num_examples()} examples.")
    print(f"[OK] Device used: {device}")
    print(f"[OK] Seed used: {args.seed}")
    print(f"[OK] Label used: {args.label_name}")
    print(f"[OK] Model saved to: {args.model_out}")
    print(f"[OK] Metadata saved to: {args.metadata_out}")


if __name__ == "__main__":
    main()