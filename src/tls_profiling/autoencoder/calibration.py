from typing import Tuple, Dict, Any
import numpy as np
from tensorflow.keras.models import Model
from .types import AEModels, AETrainStats
from .vizualization import make_image_from_sample
import matplotlib.pyplot as plt
import seaborn as sns

def compute_reconstruction_error(
    autoencoder: Model,
    X: np.ndarray,
    *,
    metric: str = "mse",
) -> np.ndarray:
    """
    Compute per-sample reconstruction error.
    metric:
      - "mse": mean((x - xhat)^2) per sample
      - "mae": mean(|x - xhat|) per sample
    """
    recon = autoencoder.predict(X, verbose=0)
    if metric == "mse":
        return np.mean((X - recon) ** 2, axis=1)
    if metric == "mae":
        return np.mean(np.abs(X - recon), axis=1)
    raise ValueError(f"Unsupported metric: {metric}")


def summarize_reconstruction_errors(errors: np.ndarray) -> AETrainStats:
    avg = float(np.mean(errors))
    std = float(np.std(errors))
    return AETrainStats(
        avg_error=avg,
        std_error=std,
        min_error=float(np.min(errors)),
        max_error=float(np.max(errors)),
        reconstruction_errors=errors,
    )


def count_sigma_exceedances(errors: np.ndarray, avg: float, std: float) -> Dict[str, Any]:
    """
    Returns counts of samples above avg + k*std for k in {1,2,3}.
    (Useful for quick 'false positive' sanity checks on a benign set.)
    """
    thr1 = avg + 1 * std
    thr2 = avg + 2 * std
    thr3 = avg + 3 * std
    return {
        "thr_1sigma": thr1,
        "thr_2sigma": thr2,
        "thr_3sigma": thr3,
        "n_gt_1sigma": int(np.sum(errors > thr1)),
        "n_gt_2sigma": int(np.sum(errors > thr2)),
        "n_gt_3sigma": int(np.sum(errors > thr3)),
    }


def get_worst_reconstructed_indices(errors: np.ndarray, topk: int = 20) -> np.ndarray:
    """Indices of the worst reconstructed samples (descending error)."""
    topk = min(topk, len(errors))
    return np.argsort(errors)[-topk:][::-1]


def plot_error_violin(errors: np.ndarray) -> None:
    sns.violinplot(data=errors)
    plt.title("Reconstruction error distribution")
    plt.show()


def plot_worst_reconstructions(
    x: np.ndarray,
    recon: np.ndarray,
    errors: np.ndarray,
    *,
    topk: int = 20,
    ncols: int = 20,
) -> None:
    """
    Visualize original vs reconstructed for top-k worst samples.
    Assumes make_image_from_sample() converts vector -> image.
    """
    idxs = get_worst_reconstructed_indices(errors, topk=topk)
    n = min(topk, ncols)

    plt.figure(figsize=(20, 4))
    i = 0
    for j in idxs[:n]:
        orig_img = make_image_from_sample(x[j])
        recon_img = make_image_from_sample(recon[j])

        ax = plt.subplot(2, n, i + 1)
        plt.imshow(orig_img, cmap="gray")
        plt.title("Original")
        plt.axis("off")

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(recon_img, cmap="gray")
        plt.title(f"RE {errors[j]:.3f}")
        plt.axis("off")
        i += 1

    plt.show()


def debug_autoencoder(
    models: AEModels,
    x_test: np.ndarray,
    *,
    metric: str = "mse",
    plot_violin: bool = True,
    plot_worst: bool = True,
    worst_topk: int = 20,
) -> Tuple[np.ndarray, np.ndarray, AETrainStats, Dict[str, Any]]:
    """
    Runs encoder/decoder to reconstruct, computes errors, returns:
      - encoded embeddings
      - reconstructed samples
      - stats
      - sigma exceedance counts
    """
    encoded = models.encoder.predict(x_test, verbose=0)
    decoded = models.decoder.predict(encoded, verbose=0)

    errors = np.mean((x_test - decoded) ** 2, axis=1) if metric == "mse" else np.mean(np.abs(x_test - decoded), axis=1)
    stats = summarize_reconstruction_errors(errors)
    sigma = count_sigma_exceedances(errors, stats.avg_error, stats.std_error)

    print(f"Reconstruction errors: avg={stats.avg_error:.6f} std={stats.std_error:.6f} "
          f"min={stats.min_error:.6f} max={stats.max_error:.6f}")
    print(f"Sigma exceedances: {sigma}")

    if plot_violin:
        plot_error_violin(errors)

    if plot_worst:
        plot_worst_reconstructions(x_test, decoded, errors, topk=worst_topk, ncols=min(worst_topk, 20))

    return encoded, decoded, stats, sigma

