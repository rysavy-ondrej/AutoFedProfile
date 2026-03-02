import numpy as np

import pandas as pd

def compute_flow_stats(df:pd.DataFrame) -> pd.DataFrame:
    # Required columns in df:
    # td -- duration in seconds
    # bs, br -- bytes sent / received
    # ps, pr -- packets sent / received
    # ts -- flow start timestamp (used here to estimate flow-level IAT)

    required_cols = ["td", "bs", "br", "ps", "pr", "ts"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in df: {missing}")

    def robust_stats(series: pd.Series) -> dict:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return {
                "mean": np.nan,
                "std": np.nan,
                "median": np.nan,
                "iqr": np.nan,
                "p95": np.nan,
                "p99": np.nan,
            }

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)

        return {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "median": float(s.median()),
            "iqr": float(q3 - q1),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
        }



    # Base numeric series

    td = pd.to_numeric(df["td"], errors="coerce")
    bs = pd.to_numeric(df["bs"], errors="coerce")
    br = pd.to_numeric(df["br"], errors="coerce")
    ps = pd.to_numeric(df["ps"], errors="coerce")
    pr = pd.to_numeric(df["pr"], errors="coerce")



    # Derived metrics

    total_bytes = bs + br
    byte_ratio_up_down = bs / br.replace(0, np.nan)

    # Flow-level IAT estimate from consecutive flow start timestamps
    ts_sorted = pd.to_numeric(df["ts"], errors="coerce").dropna().sort_values()
    iat = ts_sorted.diff().dropna()
    iat = iat[iat >= 0]

    # IAT mean / std (+ robust tails)
    iat_clean = iat.dropna()
    stats_by_property = {
        "duration_s": robust_stats(td),
        "bytes_up": robust_stats(bs),
        "bytes_down": robust_stats(br),
        "bytes_total": robust_stats(total_bytes),
        "packets_up": robust_stats(ps),
        "packets_down": robust_stats(pr),
        "byte_ratio_up_down": robust_stats(byte_ratio_up_down),
        "iat_s": robust_stats(iat_clean),
    }

    # Requested percentages as dedicated properties
    pct_short = float((td < 1.0).mean() * 100) if td.notna().any() else np.nan
    pct_large = float((total_bytes > 1_000_000).mean() * 100) if total_bytes.notna().any() else np.nan

    rows = []
    for prop, metric_values in stats_by_property.items():
        row = {"property": prop}
        row.update(metric_values)
        rows.append(row)

    rows.append({
        "property": "pct_short_connections_lt_1s",
        "mean": pct_short,
        "std": np.nan,
        "median": np.nan,
        "iqr": np.nan,
        "p95": np.nan,
        "p99": np.nan,
    })
    rows.append({
        "property": "pct_large_transfers_gt_1MB",
        "mean": pct_large,
        "std": np.nan,
        "median": np.nan,
        "iqr": np.nan,
        "p95": np.nan,
        "p99": np.nan,
    })

    flow_stats_df = pd.DataFrame(rows)
    flow_stats_df = flow_stats_df[["property", "mean", "std", "median", "iqr", "p95", "p99"]]
    return flow_stats_df