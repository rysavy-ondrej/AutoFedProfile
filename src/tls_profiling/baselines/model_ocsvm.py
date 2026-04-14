from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.svm import OneClassSVM


@dataclass(frozen=True)
class Config:
    kernel: str = "rbf"
    gamma: str | float = "scale"
    nu: float = 0.05
    shrinking: bool = True
    tol: float = 1e-3
    cache_size: float = 512
    max_train_samples: int | None = 5_000
    random_state: int = 42


class OneClassSVMDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: Optional[OneClassSVM] = None

    def _to_numpy(self, X: np.ndarray) -> np.ndarray:
        X_array = np.asarray(X, dtype=np.float64)
        if X_array.ndim != 2:
            raise ValueError("Expected a 2D feature matrix.")
        if X_array.shape[0] < 1 or X_array.shape[1] < 1:
            raise ValueError("Cannot fit or score an empty feature matrix.")
        return X_array

    def _sample_rows(self, X: np.ndarray) -> np.ndarray:
        if self.cfg.max_train_samples is None or X.shape[0] <= self.cfg.max_train_samples:
            return X

        rng = np.random.default_rng(self.cfg.random_state)
        indices = np.sort(rng.choice(X.shape[0], size=self.cfg.max_train_samples, replace=False))
        return X[indices]

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        X_array = self._sample_rows(self._to_numpy(X))
        self.model = OneClassSVM(
            kernel=self.cfg.kernel,
            gamma=self.cfg.gamma,
            nu=self.cfg.nu,
            shrinking=self.cfg.shrinking,
            tol=self.cfg.tol,
            cache_size=self.cfg.cache_size,
        )
        self.model.fit(X_array)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly score where higher means 'more anomalous'.
        OneClassSVM.decision_function returns larger values for more normal
        samples, so we negate it.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_array = self._to_numpy(X)
        normal_score = self.model.decision_function(X_array).reshape(-1)
        return -normal_score

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        threshold is applied on anomaly score from score().
        Returns 1 for anomaly, 0 for normal.
        """
        s = self.score(X)
        return (s >= threshold).astype("int8")
