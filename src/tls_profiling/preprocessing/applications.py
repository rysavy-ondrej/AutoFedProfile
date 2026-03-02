import pandas as pd

def get_applications_frequency(df) -> pd.DataFrame:
    freq_df = (
        df["meta.application.name"]
        .dropna()
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={
            "index": "application",
            "meta.application.name": "count"
        })
    )
    return freq_df

def temporal_coverage_per_application(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-application daily sample counts based on meta.sample.id.

    Expected format of meta.sample.id:
        YYYYMMDD_APPLICATION.NAME

    Returns
    -------
    pd.DataFrame
        Columns:
        - application
        - day (datetime64)
        - samples (int)
    """

    # Drop rows without sample id
    df = df.dropna(subset=["meta.sample.id"]).copy()

    # Split meta.sample.id
    parts = df["meta.sample.id"].str.split(r"[_\-\s]+", n=1, expand=True)
    df["day"] = pd.to_datetime(parts[0], format="%Y%m%d", errors="coerce")
    df["application"] = parts[1]

    # Remove malformed rows
    df = df.dropna(subset=["day", "application"])

    # Aggregate
    coverage = (
        df
        .groupby(["application", "day"])
        .size()
        .reset_index(name="samples")
        .sort_values(["application", "day"])
    )

    return coverage

def get_system_processes(os:str = "windows"):
    return {"System","svchost.exe","msedge.exe","backgroundTaskHost.exe","Explorer.EXE","explorer.exe","smartscreen.exe"}
