from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal 

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class Config:
    # Feature choices
    num_cols: Tuple[str, ...] = ("bs", "ps", "br", "pr", "sp", "dp", "td")
    ja_cols: Tuple[str, ...] = ("tls.ja3", "tls.ja4", "tls.ja3s", "tls.ja4s")
    tls_cols: Tuple[str,...] = ("tls.sext", "tls.cver", "tls.csg", "tls.ccs", "tls.cext", "tls.sver", "tls.ssv", "tls.csv", "tls.scs", "tls.alpn", "tls.sni")
    use_rec_stats: bool = True
    # TLS record sequence
    rec_col: str = "tls.rec"
    rec_maxlen: int = 20  # used if you want fixed-length seq features later

    # Model
    n_estimators: int = 300
    max_samples: float | Literal['auto'] = "auto"
    contamination: float | str = "auto"
    random_state: int = 42
    n_jobs: int = -1

    # Preprocessing
    clip_ports: bool = True
    port_max: int = 65535

class IsolationForestDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: Optional[IsolationForest] = None
    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.model = IsolationForest(
            n_estimators=self.cfg.n_estimators,
            max_samples=self.cfg.max_samples,
            contamination=self.cfg.contamination,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
        )
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly score where higher means 'more anomalous'.
        sklearn's score_samples: higher means 'more normal', so we negate it.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # extract relevant features used to train the model
        normal_score = self.model.score_samples(X)  # higher = more normal
        return -normal_score  # higher = more anomalous

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        threshold is applied on anomaly score from score().
        Returns 1 for anomaly, 0 for normal.
        """
        s = self.score(X)
        return (s >= threshold).astype("int8")