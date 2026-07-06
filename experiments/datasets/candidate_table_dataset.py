from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CandidateGroup:
    """
    One decision-point group identified by sample_id.

    A group contains multiple candidate rows corresponding to one Phase 4
    decision step.
    """
    sample_id: str
    rows: list[dict[str, Any]]


class CandidateTableDataset:
    """
    Lightweight table dataset wrapper for candidate-level exported data.

    This class is designed for non-GNN baselines (heuristic analysis,
    XGBoost / LightGBM / MLP) before later graph-specific pipelines.

    Expected input rows are the flattened rows produced by:
      - sample_to_candidate_rows(...)
      - samples_to_candidate_rows(...)
      - phase4_candidates.jsonl
      - phase4_candidates.csv

    Column conventions:
      - feature columns: prefix "feature__"
      - label columns:   prefix "label__"
      - metadata columns: everything else
    """

    FEATURE_PREFIX = "feature__"
    LABEL_PREFIX = "label__"

    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        self._rows: list[dict[str, Any]] = [dict(r) for r in rows]
        self._feature_names: list[str] = self._infer_prefixed_columns(self.FEATURE_PREFIX)
        self._label_names: list[str] = self._infer_prefixed_columns(self.LABEL_PREFIX)

    @classmethod
    def from_jsonl(cls, path: str) -> "CandidateTableDataset":
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return cls(rows)

    @classmethod
    def from_csv(cls, path: str) -> "CandidateTableDataset":
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: cls._parse_csv_value(v) for k, v in row.items()})
        return cls(rows)

    @staticmethod
    def _parse_csv_value(value: Any) -> Any:
        """
        Best-effort parser for CSV-loaded values.

        We keep this intentionally conservative:
          - "" -> ""
          - "True"/"False" -> bool
          - integer-looking -> int
          - float-looking -> float
          - otherwise keep as string
        """
        if value is None:
            return None
        if not isinstance(value, str):
            return value

        text = value.strip()
        if text == "":
            return ""

        if text == "True":
            return True
        if text == "False":
            return False

        try:
            return int(text)
        except ValueError:
            pass

        try:
            return float(text)
        except ValueError:
            pass

        return text

    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def num_rows(self) -> int:
        return len(self._rows)

    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def label_names(self) -> list[str]:
        return list(self._label_names)

    def metadata_names(self) -> list[str]:
        all_cols = self.column_names()
        feature_set = set(self._feature_names)
        label_set = set(self._label_names)
        return [c for c in all_cols if c not in feature_set and c not in label_set]

    def column_names(self) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for row in self._rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    names.append(key)
        return names

    def group_ids(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for row in self._rows:
            sample_id = str(row["sample_id"])
            if sample_id not in seen:
                seen.add(sample_id)
                ordered.append(sample_id)
        return ordered

    def groups(self) -> list[CandidateGroup]:
        return [CandidateGroup(sample_id=g, rows=self.rows_by_group(g)) for g in self.group_ids()]

    def rows_by_group(self, sample_id: str) -> list[dict[str, Any]]:
        return [row for row in self._rows if str(row["sample_id"]) == str(sample_id)]

    def num_groups(self) -> int:
        return len(self.group_ids())

    def filter_rows(self, predicate) -> "CandidateTableDataset":
        return CandidateTableDataset([row for row in self._rows if predicate(row)])

    def select_columns(self, columns: list[str]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for row in self._rows:
            selected.append({col: row.get(col) for col in columns})
        return selected

    def feature_matrix(self, feature_names: list[str] | None = None) -> list[list[float]]:
        names = feature_names if feature_names is not None else self._feature_names
        matrix: list[list[float]] = []
        for row in self._rows:
            matrix.append([self._coerce_numeric(row.get(name, 0.0)) for name in names])
        return matrix

    def label_vector(self, label_name: str) -> list[float]:
        if label_name not in self._label_names:
            raise ValueError(
                f"Unknown label column: {label_name}. Available labels: {self._label_names}"
            )
        return [self._coerce_numeric(row.get(label_name, 0.0)) for row in self._rows]

    def to_xy(
        self,
        *,
        label_name: str,
        feature_names: list[str] | None = None,
    ) -> tuple[list[list[float]], list[float]]:
        X = self.feature_matrix(feature_names=feature_names)
        y = self.label_vector(label_name=label_name)
        return X, y

    def group_sizes(self) -> list[int]:
        return [len(self.rows_by_group(gid)) for gid in self.group_ids()]

    def positive_rate(self, label_name: str) -> float:
        y = self.label_vector(label_name)
        if not y:
            return 0.0
        return float(sum(1.0 for v in y if v > 0.0)) / float(len(y))

    def summary(self) -> dict[str, Any]:
        return {
            "num_rows": self.num_rows(),
            "num_groups": self.num_groups(),
            "feature_names": self.feature_names(),
            "label_names": self.label_names(),
            "group_sizes": self.group_sizes(),
        }

    def _infer_prefixed_columns(self, prefix: str) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for row in self._rows:
            for key in row.keys():
                if key.startswith(prefix) and key not in seen:
                    seen.add(key)
                    names.append(key)
        return names

    @staticmethod
    def _coerce_numeric(value: Any) -> float:
        if value is None or value == "":
            return 0.0
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Cannot coerce value to numeric feature/label: {value!r}")