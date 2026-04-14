from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class Config:
    n_components: int | float = 0.95
    svd_solver: str = "full"
    whiten: bool = False


class PCADetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: Optional[PCA] = None

    def fit(self, X: np.ndarray) -> "PCADetector":
        X_array = np.asarray(X, dtype=np.float64)
        if X_array.ndim != 2:
            raise ValueError("Expected a 2D feature matrix.")

        max_rank = min(X_array.shape[0], X_array.shape[1])
        if max_rank < 1:
            raise ValueError("Cannot fit PCA on an empty feature matrix.")

        model = PCA(
            n_components=self.cfg.n_components,
            svd_solver=self.cfg.svd_solver,
            whiten=self.cfg.whiten,
        )
        model.fit(X_array)

        # Keep at least one component in reserve so reconstruction error stays informative.
        max_allowed_components = max_rank - 1 if max_rank > 1 else 1
        fitted_components = int(model.n_components_)
        if fitted_components > max_allowed_components:
            model = PCA(
                n_components=max_allowed_components,
                svd_solver=self.cfg.svd_solver,
                whiten=self.cfg.whiten,
            )
            model.fit(X_array)

        self.model = model
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly score where higher means 'more anomalous'.
        We use per-sample mean squared reconstruction error.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_array = np.asarray(X, dtype=np.float64)
        reconstructed = self.model.inverse_transform(self.model.transform(X_array))
        reconstruction_error = np.square(X_array - reconstructed).mean(axis=1)
        return reconstruction_error

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        threshold is applied on anomaly score from score().
        Returns 1 for anomaly, 0 for normal.
        """
        s = self.score(X)
        return (s >= threshold).astype("int8")
