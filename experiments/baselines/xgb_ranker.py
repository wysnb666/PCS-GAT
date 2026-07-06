from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from MPC4plus.experiments.datasets.candidate_table_dataset import CandidateTableDataset


def _require_xgboost():
    try:
        from xgboost import XGBClassifier  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for XGBoost baseline. "
            "Please install it in your current environment."
        ) from exc
    return XGBClassifier


@dataclass
class XGBTrainResult:
    label_name: str
    feature_names: list[str]
    num_rows: int
    num_groups: int
    positive_rate: float


class XGBPhase4Ranker:
    """
    First-stage non-GNN learning baseline.

    Current design choice:
    - Use candidate-level flat rows
    - Train with rule label first (e.g. label__selected_by_rule)
    - Use XGBClassifier as a simple binary scorer
    - At inference, score all candidates in a group and choose top-1

    This is intentionally the simplest useful learning baseline before GNN.
    """

    def __init__(
        self,
        *,
        label_name: str = "label__selected_by_rule",
        feature_names: list[str] | None = None,
        random_state: int = 42,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.08,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
    ) -> None:
        XGBClassifier = _require_xgboost()
        self.label_name = label_name
        self.feature_names = feature_names
        self.model = XGBClassifier(
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",

        )
        self._fitted_feature_names: list[str] | None = None
        self._is_fitted = False

    def fit(self, dataset: CandidateTableDataset) -> XGBTrainResult:
        feature_names = self.feature_names or dataset.feature_names()
        if not feature_names:
            raise ValueError("No feature columns found for XGBoost training.")

        X, y = dataset.to_xy(
            label_name=self.label_name,
            feature_names=feature_names,
        )
        if not X:
            raise ValueError("Empty dataset: no candidate rows available for training.")

        self.model.fit(X, y)
        self._fitted_feature_names = list(feature_names)
        self._is_fitted = True

        return XGBTrainResult(
            label_name=self.label_name,
            feature_names=list(feature_names),
            num_rows=dataset.num_rows(),
            num_groups=dataset.num_groups(),
            positive_rate=dataset.positive_rate(self.label_name),
        )

    def predict_scores(self, dataset: CandidateTableDataset) -> list[float]:
        self._check_is_fitted()
        assert self._fitted_feature_names is not None

        X = dataset.feature_matrix(feature_names=self._fitted_feature_names)
        if not X:
            return []

        # Binary classifier -> use positive class probability as candidate score.
        probs = self.model.predict_proba(X)
        return [float(p[1]) for p in probs]

    def predict_rows_with_scores(self, dataset: CandidateTableDataset) -> list[dict[str, Any]]:
        scores = self.predict_scores(dataset)
        rows = dataset.rows()
        if len(scores) != len(rows):
            raise RuntimeError("Score count does not match dataset row count.")

        scored_rows: list[dict[str, Any]] = []
        for row, score in zip(rows, scores):
            new_row = dict(row)
            new_row["pred__score"] = score
            scored_rows.append(new_row)
        return scored_rows

    def predict_top1_by_group(self, dataset: CandidateTableDataset) -> dict[str, int]:
        """
        Return predicted top-1 candidate_idx for each sample_id group.
        """
        scored_rows = self.predict_rows_with_scores(dataset)
        best_by_group: dict[str, tuple[float, int]] = {}

        for row in scored_rows:
            sample_id = str(row["sample_id"])
            candidate_idx = int(row["candidate_idx"])
            score = float(row["pred__score"])

            if sample_id not in best_by_group or score > best_by_group[sample_id][0]:
                best_by_group[sample_id] = (score, candidate_idx)

        return {sample_id: candidate_idx for sample_id, (_, candidate_idx) in best_by_group.items()}

    def evaluate_group_top1_accuracy(self, dataset: CandidateTableDataset) -> float:
        """
        Compare predicted top-1 candidate_idx with the row whose label is 1
        in each group. This is the most natural first metric for the current setup.
        """
        if self.label_name not in dataset.label_names():
            raise ValueError(
                f"Dataset does not contain label column {self.label_name!r}. "
                f"Available labels: {dataset.label_names()}"
            )

        pred = self.predict_top1_by_group(dataset)

        gold: dict[str, int] = {}
        for group in dataset.groups():
            positive_rows = [
                row for row in group.rows
                if float(row.get(self.label_name, 0.0)) > 0.0
            ]
            if len(positive_rows) == 1:
                gold[group.sample_id] = int(positive_rows[0]["candidate_idx"])

        if not gold:
            return 0.0

        correct = 0
        total = 0
        for sample_id, gold_idx in gold.items():
            if sample_id in pred:
                total += 1
                if pred[sample_id] == gold_idx:
                    correct += 1

        if total == 0:
            return 0.0
        return float(correct) / float(total)

    def save_model(self, path: str) -> None:
        self._check_is_fitted()
        self.model.save_model(path)

    def load_model(self, path: str) -> None:
        self.model.load_model(path)
        self._is_fitted = True
        if self._fitted_feature_names is None:
            # Caller should restore feature names separately if loading standalone.
            self._fitted_feature_names = []

    def fitted_feature_names(self) -> list[str]:
        self._check_is_fitted()
        return list(self._fitted_feature_names or [])

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("XGBPhase4Ranker is not fitted yet.")