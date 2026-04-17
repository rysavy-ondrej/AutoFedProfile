"""
Public top-level API for tls_profiling.

This module re-exports the small set of functions that are intended to be used
directly by notebooks and scripts, so callers can write:

    from tls_profiling import build_and_fit_preprocessor, evaluate_result_csv
"""

from __future__ import annotations

from .evaluation import evaluate_result_csv
from .exploration.connections import get_connection_label, remove_grease_values
from .io import open_tls_parquet_dataset
from .preprocessing import build_and_fit_preprocessor, extract_features, extract_features_ext


def run_autoencoder_experiment(*args, **kwargs):
    """
    Lazy top-level wrapper for the autoencoder experiment helper.

    The import happens inside the function so importing ``tls_profiling`` does
    not immediately pull in heavier optional dependencies like TensorFlow.
    """
    from .autoencoder import run_autoencoder_experiment as _run_autoencoder_experiment

    return _run_autoencoder_experiment(*args, **kwargs)


__all__ = [
    "build_and_fit_preprocessor",
    "evaluate_result_csv",
    "extract_features",
    "extract_features_ext",
    "get_connection_label",
    "open_tls_parquet_dataset",
    "remove_grease_values",
    "run_autoencoder_experiment",
]
