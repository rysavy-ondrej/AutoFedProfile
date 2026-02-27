import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_top_n_strings_auto_log_ax(
    df: pd.DataFrame,
    column: str,
    n: int,
    ax,
    title_suffix="",
    skew_ratio_threshold: float = 50.0,
):
    s = df[column].dropna()

    if s.empty:
        ax.set_title(f"{column} (no data)")
        return pd.Series(dtype=int)

    counts = s.astype(str).value_counts().head(n).sort_values()

    # Auto-log decision
    use_log = False
    if len(counts) >= 3:
        max_c = float(counts.max())
        med_c = float(np.median(counts.values))
        if med_c > 0 and (max_c / med_c) >= skew_ratio_threshold:
            use_log = True

    bars = ax.barh(counts.index, counts.values)

    ax.set_title(f"{column} {title_suffix}")
    ax.set_xlabel("Occurrences" + (" (log)" if use_log else ""))

    if use_log:
        ax.set_xscale("log")
        ax.set_xlim(left=max(1, counts.min() * 0.8), right=counts.max() * 1.5)
    else:
        ax.set_xlim(0, counts.max() * 1.15)

    return counts

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

APP_COL = "meta.application.name"
WEEK_COL = "week"
LABEL_COLS = ["system", "application", "unknown"]  # adjust if needed

def plot_3panel_weekly_heatmaps(
    weekly,
    top_n_apps: int = None,
    log_scale: bool = True,
    order_by_volume: bool = False
):
    # Pivot each label into (apps x weeks) matrix
    mats = {
        c: (weekly.pivot(index=APP_COL, columns=WEEK_COL, values=c).fillna(0))
        for c in LABEL_COLS
    }

    total = sum(mats.values())
    if order_by_volume:
        # Consistent app ordering: by total activity across all labels
        app_order = total.sum(axis=1).sort_values(ascending=False).index
    else:
        app_order = total.index.sort_values() # order by applicatoin name

    # Optional: keep only top-N apps
    if top_n_apps is not None:
        app_order = app_order[:top_n_apps]

    # Consistent week ordering
    week_order = total.columns.sort_values()

    # Reindex all matrices consistently
    for c in LABEL_COLS:
        mats[c] = mats[c].reindex(index=app_order, columns=week_order)

    # Optional log transform (recommended for heavy skew)
    def transform(x):
        return np.log1p(x) if log_scale else x

    fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharey=True)

    for ax, c in zip(axes, LABEL_COLS):
        sns.heatmap(
            transform(mats[c]),
            ax=ax,
            cmap="viridis",
            cbar=True
        )
        ax.set_title(f"{c} connections" + (" (log1p)" if log_scale else ""))
        ax.set_xlabel("Week")
        ax.set_ylabel("Application" if ax is axes[0] else "")

        # Make x labels readable
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()
    plt.show()

    return mats  # handy if you want to reuse the matrices