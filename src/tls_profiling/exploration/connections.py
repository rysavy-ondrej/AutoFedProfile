import pandas as pd
import numpy as np


def _require_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _get_category_column_name(df: pd.DataFrame) -> str:
    if "category" in df.columns:
        return "category"
    if "connection_label" in df.columns:
        return "connection_label"
    raise KeyError("Missing required columns: ['category' or 'connection_label']")


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _has_text(series: pd.Series) -> pd.Series:
    cleaned = _clean_text(series)
    return cleaned.notna() & (cleaned != "")


def _resolve_platform(df: pd.DataFrame, os_name: str | None) -> pd.Series:
    if os_name is None:
        _require_columns(df, ["meta.system.os"])
        return _clean_text(df["meta.system.os"])
    return pd.Series(os_name, index=df.index, dtype="string")


def get_connections_per_x(df: pd.DataFrame, x: str) -> pd.DataFrame:
    category_col = _get_category_column_name(df)
    _require_columns(df, [x, "meta.application.process"])

    connections_per_applications = (
        df
        .groupby(x)
        .agg(
            connections=(x, "size"),  
            system_connections=(category_col, lambda s: (s == "system").sum()),
            unknown_connections=(category_col, lambda s: (s == "unknown").sum()),
            application_connections=(category_col, lambda s: (s == "application").sum()),
            malware_connections=(category_col, lambda s: (s == "malware").sum()),

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
    category_col = _get_category_column_name(df)
    _require_columns(df, [x, "ts"])

    work_df = df.copy()
    work_df["dt"] = pd.to_datetime(work_df["ts"], unit="s", utc=True, errors="coerce")
    work_df = work_df.dropna(subset=["dt"])
    work_df["week"] = work_df["dt"].dt.to_period("W").dt.start_time.dt.date


    weekly = (
        work_df.groupby([x, "week", category_col])
        .size()
        .unstack(category_col, fill_value=0)   # columns: system/unknown/application
        .reset_index()
    )

    # Optional: total connections per week
    weekly["connections"] = (
        weekly.get("system", 0)
        + weekly.get("unknown", 0)
        + weekly.get("application", 0)
        + weekly.get("malware", 0)
    )

    return weekly


def get_connection_label(df: pd.DataFrame, os_name: str | None = None) -> pd.DataFrame:
    required = [
        "meta.application.process",
        "meta.application.name",
        "meta.malware.family",
        "meta.system.service",
    ]
    _require_columns(df, required)

    system_processes = {"System", "svchost.exe", "msedge.exe", "backgroundTaskHost.exe", "Explorer.EXE", "explorer.exe", "smartscreen.exe"}
    process_col = _clean_text(df["meta.application.process"])
    application_col = _clean_text(df["meta.application.name"])
    family_col = _clean_text(df["meta.malware.family"])
    service_col = _clean_text(df["meta.system.service"])
    platform_col = _resolve_platform(df, os_name)

    has_family = _has_text(df["meta.malware.family"])
    has_system_service = _has_text(df["meta.system.service"])

    process_is_system = process_col.isin(system_processes)
    process_is_unknown = (
        process_col.isna()
        | (process_col == "")
    )

    category = np.where(
        has_family,
        "malware",
        np.where(
            has_system_service | process_is_system,
            "system",
            np.where(process_is_unknown, "unknown", "application"),
        ),
    )

    label = np.where(
        category == "malware",
        family_col,
        np.where(
            category == "system",
            service_col.where(has_system_service, process_col),
            np.where(
                category == "application",
                application_col.where(_has_text(df["meta.application.name"]), process_col),
                pd.NA,
            ),
        ),
    )

    return pd.DataFrame(
        {
            "category": pd.Series(category, index=df.index, dtype="string"),
            "label": pd.Series(label, index=df.index, dtype="string"),
            "platform": pd.Series(platform_col, index=df.index, dtype="string"),
        },
        index=df.index,
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
