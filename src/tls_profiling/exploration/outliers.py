import numpy as np
import pandas as pd

def get_outliers(df:pd.DataFrame, rare_count:int=10):
    required_cols = ["td", "bs", "br", "tls.rec"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns in df: {missing}")

    audit_df = df.copy()
    audit_df["td"] = pd.to_numeric(audit_df["td"], errors="coerce")
    audit_df["bs"] = pd.to_numeric(audit_df["bs"], errors="coerce")
    audit_df["br"] = pd.to_numeric(audit_df["br"], errors="coerce")
    audit_df["bytes_total"] = audit_df["bs"].fillna(0) + audit_df["br"].fillna(0)

    for c in ["meta.application.name", "meta.application.process", "meta.sample.id"]:
        if c not in audit_df.columns:
            audit_df[c] = np.nan

    # 1) Top 0.1% longest duration connections
    duration_thresh = audit_df["td"].quantile(0.999)
    top_duration_df = (
        audit_df[audit_df["td"] >= duration_thresh]
        .sort_values("td", ascending=False)
        [["meta.sample.id", "meta.application.name", "meta.application.process", "td", "bs", "br", "bytes_total"]]
        .reset_index(drop=True)
    )
    # 2) Largest byte transfers (top 0.1% by total bytes)
    bytes_thresh = audit_df["bytes_total"].quantile(0.999)
    largest_transfers_df = (
        audit_df[audit_df["bytes_total"] >= bytes_thresh]
        .sort_values("bytes_total", ascending=False)
        [["meta.sample.id", "meta.application.name", "meta.application.process", "bytes_total", "bs", "br", "td"]]
        .reset_index(drop=True)
    )

    summary = pd.DataFrame(
        [
                {"metric": "duration_p99.9_threshold_s", "value": float(duration_thresh)},
                {"metric": "bytes_total_p99.9_threshold", "value": float(bytes_thresh)},
                {"metric": "n_top_duration", "value": len(top_duration_df)},
                {"metric": "n_largest_transfer", "value": len(largest_transfers_df)},
            ]
        )

    return {    "summary" : summary, 
                "n_top_duration" : top_duration_df,
                "n_largest_transfer" : largest_transfers_df
            }

from .cardinality import get_df_tls_field_card
def get_rare_values(df: pd.DataFrame, column: str, count: int) -> pd.DataFrame:
    if count <= 0:
        raise ValueError("count must be > 0")

    card_df = get_df_tls_field_card(df, column)
    return (
        card_df
        .sort_values("count", ascending=True)
        .head(count)
        .reset_index(drop=True)
    )
