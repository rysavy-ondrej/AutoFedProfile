from dataclasses import dataclass
from typing import Optional, Any
import numpy as np
from tensorflow.keras.models import Model

# -----------------------------
# Results / artifacts containers
# -----------------------------
@dataclass
class AEModels:
    autoencoder: Model
    encoder: Model
    decoder: Model


@dataclass
class AETrainStats:
    avg_error: float
    std_error: float
    min_error: float
    max_error: float
    reconstruction_errors: np.ndarray


@dataclass
class AETrainResult:
    models: AEModels
    stats: AETrainStats
    history: Optional[Any] = None  # Keras History (optional)