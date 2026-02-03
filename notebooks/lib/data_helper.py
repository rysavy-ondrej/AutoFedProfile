import pandas as pd
from sklearn.model_selection import train_test_split

def get_families(df) -> list[str]:
    families = sorted(df["meta.malware.family"].dropna().unique())
    return families

def get_families_frequency(df) -> pd.DataFrame:
    freq_df = (
        df["meta.malware.family"]
        .dropna()
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={
            "index": "family",
            "meta.malware.family": "count"
        })
    )
    return freq_df

def per_family_split(df:pd.DataFrame, families: list[str], test_size=0.2, random_state=1234) -> dict[str,tuple[pd.DataFrame,pd.DataFrame]]:
    """
    Perform train/test split independently for each malware family.

    Returns:
        {
            'agenttesla': (train_df, test_df),
            'redline': (train_df, test_df),
            ...
        }
    """

    splits = {}

    for fam in families:
        fam_df = df[df["meta.malware.family"] == fam]

        if len(fam_df) < 2:
            # skip families with only one sample
            continue

        train_df, test_df = train_test_split(
            fam_df,
            test_size=test_size,
            shuffle=True,
            random_state=random_state,
        )
        splits[fam] = (train_df, test_df)

    return splits

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
    parts = df["meta.sample.id"].str.split("_", n=1, expand=True)
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