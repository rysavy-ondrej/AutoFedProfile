from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class Config:
    n_components: int = 5
    covariance_type: str = "diag"
    reg_covar: float = 1e-5
    max_iter: int = 200
    n_init: int = 2
    init_params: str = "kmeans"
    random_state: int = 42
    max_train_samples: int | None = 20_000


class GaussianMixtureDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: Optional[GaussianMixture] = None

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

    def fit(self, X: np.ndarray) -> "GaussianMixtureDetector":
        X_array = self._sample_rows(self._to_numpy(X))
        n_components = min(self.cfg.n_components, X_array.shape[0])
        if n_components < 1:
            raise ValueError("GaussianMixture requires at least one training sample.")

        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=self.cfg.covariance_type,
            reg_covar=self.cfg.reg_covar,
            max_iter=self.cfg.max_iter,
            n_init=self.cfg.n_init,
            init_params=self.cfg.init_params,
            random_state=self.cfg.random_state,
        )
        self.model.fit(X_array)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly score where higher means 'more anomalous'.
        GaussianMixture.score_samples returns log-likelihood where higher means
        'more normal', so we negate it.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_array = self._to_numpy(X)
        log_likelihood = self.model.score_samples(X_array)
        return -log_likelihood

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        threshold is applied on anomaly score from score().
        Returns 1 for anomaly, 0 for normal.
        """
        s = self.score(X)
        return (s >= threshold).astype("int8")
