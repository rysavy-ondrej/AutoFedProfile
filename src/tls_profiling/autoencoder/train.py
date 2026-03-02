"""
Training utilities for TLS autoencoder models.

This module contains a focused training entrypoint that compiles an already
constructed autoencoder and fits it in self-reconstruction mode:

- input  -> autoencoder -> reconstruction
- target = original input

The helper is intentionally small and reusable from notebooks and scripts.
"""

import importlib
from typing import Any, Literal

import numpy as np

from .types import AEModels


KerasVerbose = Literal["auto", 0, 1, 2]


def _load_tensorflow() -> Any:
    """Load and return the `tensorflow` module lazily."""
    return importlib.import_module("tensorflow")



# -----------------------------
# 2) Training
# -----------------------------
def train_autoencoder_model(
    models: AEModels,
    x_train: np.ndarray,
    x_val: np.ndarray,
    *,
    max_epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    loss: str = "binary_crossentropy",
    early_stopping_patience: int = 10,
    verbose: KerasVerbose = 1,
) -> Any:
    """
    Compile and train an autoencoder with early stopping.

    Parameters
    ----------
    models:
        Container with pre-built Keras models. Only ``models.autoencoder`` is
        compiled and trained by this function.
    x_train:
        Training matrix with shape ``(n_train_samples, n_features)`` used for
        weight updates.
    x_val:
        Validation matrix with shape ``(n_val_samples, n_features)`` used for
        monitoring ``val_loss`` and early stopping. Must have the same feature
        dimension as ``x_train``.
    max_epochs:
        Maximum number of epochs to train.
    batch_size:
        Mini-batch size used in ``model.fit``.
    lr:
        Adam optimizer learning rate.
    loss:
        Reconstruction loss name (for example ``"binary_crossentropy"`` or
        ``"mse"`` depending on your preprocessing and output activation).
    early_stopping_patience:
        Number of epochs with no validation improvement before stopping.
    verbose:
        Keras verbosity level. Accepted values are ``"auto"``, ``0``, ``1``,
        and ``2``.

    Returns
    -------
    Any
        Keras ``History`` object produced by ``model.fit``.

    Raises
    ------
    ValueError
        If inputs are not 2D arrays, have incompatible feature dimensions, or
        if key numeric hyperparameters are invalid.

    Notes
    -----
    Autoencoder training uses self-supervision:

    - inputs: ``x_train``
    - targets: ``x_train``

    and validation targets similarly use ``(x_val, x_val)``.
    """
    if x_train.ndim != 2 or x_val.ndim != 2:
        raise ValueError("x_train and x_val must be 2D arrays")
    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError("x_train and x_val must have the same feature dimension")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0")

    tf = _load_tensorflow()

    models.autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True,
    )

    history = models.autoencoder.fit(
        x_train,
        x_train,
        validation_data=(x_val, x_val),
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[early_stop],
        verbose=verbose,
    )
    return history