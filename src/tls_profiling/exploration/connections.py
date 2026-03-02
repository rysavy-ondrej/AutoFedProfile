import pandas as pd
import numpy as np


def _require_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def get_connections_per_x(df: pd.DataFrame, x: str) -> pd.DataFrame:
    _require_columns(df, [x, "connection_label", "meta.application.process"])

    connections_per_applications = (
        df
        .groupby(x)
        .agg(
            connections=(x, "size"),  
            system_connections=("connection_label", lambda s: (s == "system").sum()),
            unknown_connections=("connection_label", lambda s: (s == "unknown").sum()),
            application_connections=("connection_label", lambda s: (s == "application").sum()),
            malware_connections=("connection_label", lambda s: (s == "malware").sum()),

            processes=("meta.application.process", lambda x: list(x.dropna().unique()))
        )
        .reset_index()
    )

    connections_per_applications = connections_per_applications[
        [
            x,
            "connections",
            "system_connections",
            "unknown_connections",
            "application_connections",
            "malware_connections",
            "processes",
        ]
    ]
    return connections_per_applications


def get_weekly_connections_per_x(df: pd.DataFrame, x: str) -> pd.DataFrame:
    _require_columns(df, [x, "ts", "connection_label"])

    work_df = df.copy()
    work_df["dt"] = pd.to_datetime(work_df["ts"], unit="s", utc=True, errors="coerce")
    work_df = work_df.dropna(subset=["dt"])
    work_df["week"] = work_df["dt"].dt.to_period("W").dt.start_time.dt.date


    weekly = (
        work_df.groupby([x, "week", "connection_label"])
        .size()
        .unstack("connection_label", fill_value=0)   # columns: system/unknown/application
        .reset_index()
    )

    # Optional: total connections per week
    weekly["connections"] = weekly.get("system", 0) + weekly.get("unknown", 0) + weekly.get("application", 0)

    return weekly


def get_connection_label(df: pd.DataFrame, method: str, data: dict | None):
    _require_columns(df, ["meta.application.process"])

    if method == "by_application_process":
        if data is None: 
            raise ValueError(f"Parameter 'data' has to be dictionary with system and unkown keys.")
        system_processes = set(data.get("system", []))
        unknown_processes = set(data.get("unknown", []))
        process_col = df["meta.application.process"]

        return np.where(
            process_col.isin(system_processes),
            "system",
            np.where(process_col.isin(unknown_processes), "unknown", "application"),
        )

    if method == "by_malware_family":
        _require_columns(df, ["meta.malware.family", "meta.system.service"])
        return np.where(
            df["meta.malware.family"].notna(),
            "malware",
            np.where(
                df["meta.system.service"].notna(),
                "system",
                "unknown",
            ),
        )

    raise ValueError(
        f"Unsupported method '{method}'. Supported methods: ['by_application_process','by_malware_family']"
    )

def remove_grease_values(df: pd.DataFrame) -> pd.DataFrame:
    # Remove GREASE values from list-like TLS fields
    def _to_int_or_none(val):
        if val is None:
            return None
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            v = val.strip().lower()
            try:
                # supports both "0x...." and plain hex/decimal strings
                return int(v, 16) if v.startswith("0x") else int(v, 16)
            except ValueError:
                return None
        return None

    def is_grease(val) -> bool:
        iv = _to_int_or_none(val)
        return iv is not None and (iv & 0x0F0F) == 0x0A0A

    def remove_grease_from_sequence(seq):
        if isinstance(seq, (list, tuple, np.ndarray)):
            return [e for e in seq if not is_grease(e)]
        return seq

    grease_containing_fields = ["tls.csv", "tls.ssv", "tls.ccs", "tls.cext", "tls.csg", "tls.sext"]
    for f in grease_containing_fields:
        if f in df.columns:
            df[f] = df[f].apply(remove_grease_from_sequence)
    return df