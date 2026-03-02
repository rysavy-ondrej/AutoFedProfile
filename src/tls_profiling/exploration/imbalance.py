import numpy as np
import pandas as pd


def get_class_imbalance(connections_per_x:pd.DataFrame, x_col:str) -> pd.DataFrame:
    """
    Compute compact class-imbalance metrics from a pre-aggregated count table.

    This function expects a dataframe where each row corresponds to one class
    (for example, application, malware family, host, service) and includes the
    number of connections for that class in a column named ``"connections"``.
    It then derives long-tail indicators and returns a compact metrics report.

    Parameters
    ----------
    connections_per_x:
        Pre-aggregated dataframe with at least two columns:
        - ``x_col``: class label column
        - ``"connections"``: number of connections for the class
    x_col:
        Name of the class-label column in ``connections_per_x``.

    Returns
    -------
    pd.DataFrame
        Two-column dataframe ``["metric", "value"]`` with:
        - ``num_classes``
        - ``total_connections``
        - ``top10_share_pct``
        - ``gini_coefficient``
        - ``num_classes_lt_50``
        - ``share_connections_classes_lt_50_pct``

    Raises
    ------
    KeyError
        If required columns are missing.
    ValueError
        If no valid rows remain after cleaning.

    Notes
    -----
    - ``top10_share_pct`` is the percentage of all connections covered by the
      10 largest classes.
    - ``gini_coefficient`` quantifies concentration/inequality of class sizes.
      Higher values indicate stronger long-tail imbalance.
    - "Rare" classes are defined as classes with fewer than 50 connections.
    """
    # Build from the precomputed table created in section 1
    required_cols = [x_col, "connections"]

    missing = [c for c in required_cols if c not in connections_per_x.columns]

    if missing:
        raise KeyError(f"connections_per_x is missing required columns: {missing}")

    counts = (
        connections_per_x[[x_col, "connections"]]
        .dropna(subset=[x_col, "connections"])
        .rename(columns={x_col: "family"})
        .copy()
    )

    counts["connections"] = pd.to_numeric(counts["connections"], errors="coerce")
    counts = counts.dropna(subset=["connections"])
    counts = counts.sort_values("connections", ascending=False).reset_index(drop=True)

    if counts.empty:
        raise ValueError("No valid data in connections_per_x to compute imbalance report.")

    # Long-tail indicators
    counts["rank"] = np.arange(1, len(counts) + 1)
    total_connections = counts["connections"].sum()
    counts["share_pct"] = 100.0 * counts["connections"] / total_connections
    counts["cum_share_pct"] = counts["share_pct"].cumsum()
    top10_share_pct = float(counts.head(10)["connections"].sum() / total_connections * 100.0)

    def gini_coefficient(values: pd.Series) -> float:
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]
        x = x[x >= 0]
        n = len(x)
        if n == 0:
            return np.nan

        if np.all(x == 0):
            return 0.0

        x = np.sort(x)
        i = np.arange(1, n + 1)
        return float((2.0 * np.sum(i * x)) / (n * np.sum(x)) - (n + 1) / n)

    gini = gini_coefficient(counts["connections"])
    # Tables requested

    top_families = counts.head(10).copy()
    rare_families = counts[counts["connections"] < 50].copy()
    rare_share_pct = float(rare_families["connections"].sum() / total_connections * 100.0) if not rare_families.empty else 0.0
    imbalance_report = pd.DataFrame([
        {"metric": "num_classes", "value": int(len(counts))},
        {"metric": "total_connections", "value": int(total_connections)},
        {"metric": "top10_share_pct", "value": top10_share_pct},
        {"metric": "gini_coefficient", "value": gini},
        {"metric": "num_classes_lt_50", "value": int(len(rare_families))},
        {"metric": "share_connections_classes_lt_50_pct", "value": rare_share_pct},
    ])

    return imbalance_report
