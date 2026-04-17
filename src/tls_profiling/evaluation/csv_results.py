from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    f1_score,
)


LOW_FPR_TARGETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05)
REQUIRED_COLUMNS = {"y_test", "y_pred", "anomaly_score"}


def _load_result_csv(csv_path: str | Path) -> tuple[Path, pd.DataFrame]:
    path = Path(csv_path)
    frame = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns {sorted(missing_columns)} in {path}."
        )

    return path, frame


def _compute_tpr_at_fpr_targets(
    fpr: pd.Series | list[float],
    tpr: pd.Series | list[float],
    targets: tuple[float, ...],
) -> dict[str, float]:
    tpr_at_targets: dict[str, float] = {}
    for target in targets:
        valid_points = [tpr_value for fpr_value, tpr_value in zip(fpr, tpr) if fpr_value <= target]
        tpr_at_targets[f"tpr_at_fpr_{target * 100:.1f}pct"] = float(max(valid_points) if valid_points else 0.0)
    return tpr_at_targets


def _plot_roc_curve(path: Path, fpr, tpr, roc_auc: float) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title(f"ROC Curve: {path.name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_pr_curve(path: Path, precision, recall, pr_auc: float) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}", linewidth=2)
    ax.set_title(f"Precision-Recall Curve: {path.name}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_tpr_at_low_fpr(path: Path, tpr_at_targets: dict[str, float]) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = [metric_name.replace("tpr_at_fpr_", "").replace("pct", "%") for metric_name in tpr_at_targets]
    values = list(tpr_at_targets.values())
    bars = ax.bar(labels, values, color="#1f77b4", edgecolor="black", linewidth=0.6)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.02, 0.98),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    ax.set_ylim(0, 1.0)
    ax.set_title(f"TPR at Low FPR Targets: {path.name}")
    ax.set_xlabel("FPR Target")
    ax.set_ylabel("True Positive Rate")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def evaluate_result_csv(csv_path: str | Path) -> tuple[pd.DataFrame, dict[str, Figure]]:
    """
    Evaluate a binary-result CSV exported by the notebooks.

    The CSV must contain:
    - ``y_test``: binary ground truth (0 normal, 1 anomaly)
    - ``y_pred``: binary predictions
    - ``anomaly_score``: continuous anomaly score, higher means more anomalous

    The function returns:
    - a dataframe with columns ``metric`` and ``value``
    - a dictionary of matplotlib figures for ROC, precision-recall, and TPR at
      low FPR
    """
    path, frame = _load_result_csv(csv_path)

    y_true = frame["y_test"].astype(int)
    y_pred = frame["y_pred"].astype(int)
    y_score = frame["anomaly_score"].astype(float)

    if y_true.nunique() < 2:
        raise ValueError(
            f"Expected both classes in y_test for {path}, got {sorted(y_true.unique())}."
        )

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    pr_auc = float(average_precision_score(y_true, y_score))
    roc_auc = float(roc_auc_score(y_true, y_score))
    ks_statistic = float((tpr - fpr).max())
    mcc = float(matthews_corrcoef(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))

    tpr_at_targets = _compute_tpr_at_fpr_targets(fpr=fpr, tpr=tpr, targets=LOW_FPR_TARGETS)

    metrics_rows = [
        {"metric": "pr_auc", "value": pr_auc},
        *({"metric": metric_name, "value": metric_value} for metric_name, metric_value in tpr_at_targets.items()),
        {"metric": "ks_statistic", "value": ks_statistic},
        {"metric": "mcc", "value": mcc},
        {"metric": "f1", "value": f1},
    ]
    metrics_df = pd.DataFrame(metrics_rows, columns=["metric", "value"])

    figures = {
        "roc_curve": _plot_roc_curve(path=path, fpr=fpr, tpr=tpr, roc_auc=roc_auc),
        "pr_curve": _plot_pr_curve(path=path, precision=precision, recall=recall, pr_auc=pr_auc),
        "tpr_at_low_fpr": _plot_tpr_at_low_fpr(path=path, tpr_at_targets=tpr_at_targets),
    }

    return metrics_df, figures
