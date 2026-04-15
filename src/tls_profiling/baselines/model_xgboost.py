from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


@dataclass(frozen=True)
class Config:
    n_estimators: int = 400
    max_depth: int = 8
    learning_rate: float = 0.05
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1
    tree_method: str = "hist"
    max_train_samples: int | None = 100_000
    use_balanced_sample_weights: bool = True


class XGBoostBaseline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: Optional[XGBClassifier] = None

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
        X_array = self._to_numpy(X)
        y_array = np.asarray(y)
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("Feature matrix and target vector must have the same number of rows.")

        X_sampled, y_sampled = self._sample_rows(X_array, y_array)
        sample_weight = None
        if self.cfg.use_balanced_sample_weights:
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_sampled)

        num_classes = len(np.unique(y_sampled))
        self.model = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            min_child_weight=self.cfg.min_child_weight,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            reg_lambda=self.cfg.reg_lambda,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
            tree_method=self.cfg.tree_method,
            verbosity=0,
        )
        self.model.fit(X_sampled, y_sampled, sample_weight=sample_weight)
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

    def build_shap_explainer(self, X_background: np.ndarray | None = None) -> Any:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        import shap

        background_array = None
        feature_perturbation = "tree_path_dependent"
        if X_background is not None:
            background_array = self._to_numpy(X_background)
            feature_perturbation = "interventional"

        return shap.TreeExplainer(
            self.model,
            data=background_array,
            feature_perturbation=feature_perturbation,
        )

    def explain(self, X: np.ndarray, explainer: Any | None = None) -> Any:
        X_array = self._to_numpy(X)
        shap_explainer = explainer if explainer is not None else self.build_shap_explainer()
        return shap_explainer(X_array)
