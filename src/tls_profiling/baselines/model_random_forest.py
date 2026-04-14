from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class Config:
    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_leaf: int = 1
    max_features: str | int | float | None = "sqrt"
    class_weight: str | dict[int, float] | None = "balanced_subsample"
    random_state: int = 42
    n_jobs: int = -1
    max_train_samples: int | None = 100_000


class RandomForestBaseline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: Optional[RandomForestClassifier] = None

    def _to_numpy(self, X: np.ndarray) -> np.ndarray:
        X_array = np.asarray(X, dtype=np.float64)
        if X_array.ndim != 2:
            raise ValueError("Expected a 2D feature matrix.")
        if X_array.shape[0] < 1 or X_array.shape[1] < 1:
            raise ValueError("Cannot fit or score an empty feature matrix.")
        return X_array

    def _sample_rows(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.cfg.max_train_samples is None or X.shape[0] <= self.cfg.max_train_samples:
            return X, y

        rng = np.random.default_rng(self.cfg.random_state)
        y_array = np.asarray(y)
        unique_classes, counts = np.unique(y_array, return_counts=True)

        target_total = self.cfg.max_train_samples
        raw_targets = counts / counts.sum() * target_total
        sample_counts = np.floor(raw_targets).astype(int)
        sample_counts = np.maximum(sample_counts, 1)
        sample_counts = np.minimum(sample_counts, counts)

        remainder = target_total - int(sample_counts.sum())
        if remainder > 0:
            fractional = raw_targets - np.floor(raw_targets)
            order = np.argsort(-fractional)
            for idx in order:
                if remainder == 0:
                    break
                if sample_counts[idx] < counts[idx]:
                    sample_counts[idx] += 1
                    remainder -= 1

        sampled_indices: list[np.ndarray] = []
        for class_value, class_count in zip(unique_classes, sample_counts):
            class_indices = np.flatnonzero(y_array == class_value)
            if class_count >= len(class_indices):
                sampled_indices.append(class_indices)
                continue

            chosen = rng.choice(class_indices, size=int(class_count), replace=False)
            sampled_indices.append(np.sort(chosen))

        indices = np.sort(np.concatenate(sampled_indices))
        return X[indices], y_array[indices]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        X_array = self._to_numpy(X)
        y_array = np.asarray(y)
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("Feature matrix and target vector must have the same number of rows.")

        X_sampled, y_sampled = self._sample_rows(X_array, y_array)
        self.model = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            max_features=self.cfg.max_features,
            class_weight=self.cfg.class_weight,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
        )
        self.model.fit(X_sampled, y_sampled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_array = self._to_numpy(X)
        return self.model.predict(X_array)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_array = self._to_numpy(X)
        return self.model.predict_proba(X_array)
