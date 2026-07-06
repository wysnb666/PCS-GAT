from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MPC4plus.experiments.baselines.xgb_ranker import XGBPhase4Ranker
from MPC4plus.experiments.datasets.candidate_table_dataset import CandidateTableDataset


def _load_dataset(path: str) -> CandidateTableDataset:
    if path.endswith(".jsonl"):
        return CandidateTableDataset.from_jsonl(path)
    if path.endswith(".csv"):
        return CandidateTableDataset.from_csv(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train first-stage XGBoost baseline on candidate table data."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to candidate table (.jsonl or .csv).",
    )
    parser.add_argument(
        "--label-name",
        type=str,
        default="label__selected_by_rule",
        help="Training label column. Default: label__selected_by_rule",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        required=True,
        help="Path to save trained XGBoost model.",
    )
    parser.add_argument(
        "--metadata-out",
        type=str,
        required=True,
        help="Path to save training metadata json.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for XGBoost training.",
    )

    args = parser.parse_args()

    set_global_seed(args.seed)

    dataset = _load_dataset(args.input)

    ranker = XGBPhase4Ranker(
        label_name=args.label_name,
        random_state=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )

    train_result = ranker.fit(dataset)
    top1_acc = ranker.evaluate_group_top1_accuracy(dataset)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    ranker.save_model(str(model_out))

    metadata = {
        "label_name": train_result.label_name,
        "feature_names": train_result.feature_names,
        "num_rows": train_result.num_rows,
        "num_groups": train_result.num_groups,
        "positive_rate": train_result.positive_rate,
        "group_top1_accuracy_on_train": top1_acc,
        "seed": args.seed,
    }

    metadata_out = Path(args.metadata_out)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Trained XGBoost baseline on {train_result.num_rows} rows.")
    print(f"[OK] Seed used: {args.seed}")
    print(f"[OK] Model saved to: {model_out}")
    print(f"[OK] Metadata saved to: {metadata_out}")


if __name__ == "__main__":
    main()