
import numpy as np
from .types import AETrainResult
from .models import build_conv_dense_autoencoder
from .train import train_autoencoder_model
from .calibration import compute_reconstruction_error, summarize_reconstruction_errors, debug_autoencoder


def run_autoencoder_experiment(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    *,
    encoding_dim: int,
    conv_input_size: int = 20,
    intermediate_dim: int = 64,
    max_epochs: int = 50,
    debug: bool = False,
) -> AETrainResult:
    """
    Build, train, and evaluate an autoencoder in one call.

    This helper wraps model construction, training, and reconstruction-error
    summarization to keep notebooks concise while preserving the underlying
    modular pipeline.

    Parameters
    ----------
    x_train : np.ndarray
        Training feature matrix used to optimize autoencoder weights.
        In anomaly-detection settings, this should typically contain mostly
        "normal" traffic so the model learns the baseline reconstruction space.

    x_val : np.ndarray
        Validation feature matrix used during training for monitoring
        generalization (for example, early stopping / model selection in the
        training routine). These samples must be disjoint from ``x_train`` and
        are not used for gradient updates.

    x_test : np.ndarray
        Held-out evaluation feature matrix used only after training to compute
        reconstruction errors and summary statistics. This split should remain
        unseen during optimization and validation to avoid optimistic estimates.

    encoding_dim : int
        Size of the latent bottleneck representation.

    conv_input_size : int, default=20
        Input size used by the convolutional branch of the hybrid architecture.

    intermediate_dim : int, default=64
        Hidden dimensionality used in intermediate dense layers.

    max_epochs : int, default=50
        Maximum number of training epochs.

    debug : bool, default=False
        If ``True``, runs the explicit debug path to inspect encoded/decoded
        outputs and worst reconstructions.

    Returns
    -------
    AETrainResult
        Structured result with trained models, training history, and
        reconstruction-error summary statistics computed on ``x_test``.

    Notes
    -----
    All three inputs are expected to have the same feature dimension
    ``(n_samples, n_features)``. The function infers ``n_features`` from
    ``x_train.shape[1]`` and uses it to build the model.
    """
    input_dim = x_train.shape[1]
    print(f"train shape={x_train.shape}")
    print(f"val shape={x_val.shape}")
    print(f"test shape={x_test.shape}")

    models = build_conv_dense_autoencoder(
        input_dim=input_dim,
        encoding_dim=encoding_dim,
        conv_input_size=conv_input_size,
        intermediate_dim=intermediate_dim,
    )

    history = train_autoencoder_model(
        models=models,
        x_train=x_train,
        x_val=x_val,
        max_epochs=max_epochs,
    )

    errors = compute_reconstruction_error(models.autoencoder, x_test, metric="mse")
    stats = summarize_reconstruction_errors(errors)

    if debug:
        # Use the explicit debug pipeline to also show worst reconstructions
        _encoded, _decoded, _stats, _sigma = debug_autoencoder(models, x_test)

    return AETrainResult(models=models, stats=stats, history=history)