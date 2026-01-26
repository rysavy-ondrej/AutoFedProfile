import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import load_parquet_files
from preprocessing import get_tls_connections
from typing import List, Optional

# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
def _extract_families(m):
    """Extract malware families from metadata."""
    if m is None or not isinstance(m, dict) or "family" not in m:
        return ["_none_"]
    fams = m["family"]
    return fams.tolist() if hasattr(fams, "tolist") else list(fams)

def _extract_application_name(m):
    """Extract metadata to the simple form -- application name from metadata."""
    if not isinstance(m, dict):
        return pd.NA
    return (m.get("os") or pd.NA)

def _extract_system_name(m):
    """Extract metadata to the simple form -- operating system name from metadata."""
    if not isinstance(m, dict):
        return pd.NA
    return (m.get("os") or pd.NA)

def _extract_service_name(m):
    """Extract metadata to the simple form -- operating service name from metadata."""
    if not isinstance(m, dict):
        return pd.NA
    return (m.get("os") or pd.NA)

# -------------------------------------------------------------
# Malware Dataset
# -------------------------------------------------------------
class MalwareDataset:
    """
    Manager for preprocessing, organizing, and splitting Triage malware datasets.
    """
    def __init__(self, df_tls: pd.DataFrame):
        """
        Initialize the DatasetManager with TLS dataframe.
        This constructor processes the input TLS dataframe by extracting malware families
        from the metadata and expanding the dataframe so that each malware family gets its
        own row.
        Args:
            df_tls (pd.DataFrame): A pandas DataFrame containing TLS connection data with a 
                'meta.malware' column that contains malware family annotation.
        """
        df = df_tls.copy()

        # families: list-like per row (or pd.NA)
        df["meta.malware"] = df["meta.malware"].apply(_extract_families)

        # expand: one row per family
        df_expanded = df.explode("meta.malware", ignore_index=True)

        # keep only annotated families
        df_expanded = df_expanded.loc[
            df_expanded["meta.malware"].notna() & (df_expanded["meta.malware"] != "")
        ].copy()

        # derive columns (extract service BEFORE overwriting meta.system)
        df_expanded["meta.application"] = pd.NA
        df_expanded["meta.service"] = df_expanded["meta.system"].apply(_extract_service_name)
        df_expanded["meta.system"]  = df_expanded["meta.system"].apply(_extract_system_name)
        df_expanded["meta.host"] = pd.NA

        self.df_expanded = df_expanded

    @staticmethod
    def load_from(path:str,max_rows: Optional[int] = None):
        """
        Load malware dataset from parquet files and create a MalwareDataset object.

        This function loads parquet files from the specified path, filters for TLS connections,
        and initializes a MalwareDataset object with the filtered data.

        Args:
            path (str): The file path to the directory containing parquet files.

        Returns:
            MalwareDataset: A MalwareDataset object initialized with TLS connection data.

        Raises:
            FileNotFoundError: If the specified path does not exist.
            ValueError: If no valid parquet files are found at the path.

        Example:
            >>> malware_data = MalwareDataset.load_from('/path/to/malware/data')
        """
        df = load_parquet_files(parquet_folder=path, max_rows=max_rows)
        df_tls = get_tls_connections(df)
        return MalwareDataset(df_tls)

    @property
    def dataframe(self): 
        return self.df_expanded    
        
    def families(self):
        """Return sorted list of malware families."""
        if self.df_expanded is None:
            raise ValueError("Call preprocess() first.")
        return sorted(self.df_expanded["meta.malware"].unique())

    def samples_per_family(self):
        """Return Series with counts per family."""
        if self.df_expanded is None:
            raise ValueError("Call preprocess() first.")
        return self.df_expanded["meta.malware"].value_counts()

    def get_family(self, family: str):
        """Return all rows containing the specified malware family."""
        if self.df_expanded is None:
            raise ValueError("Call preprocess() first.")
        return self.df_expanded[self.df_expanded["meta.malware"] == family]

    def per_family_split(self, test_size=0.2, random_state=42):
        """
        Perform train/test split independently for each malware family.

        Returns:
            {
                'agenttesla': (train_df, test_df),
                'redline': (train_df, test_df),
                ...
            }
        """
        if self.df_expanded is None:
            raise ValueError("Call preprocess() first.")

        splits = {}

        for fam in self.families():
            fam_df = self.df_expanded[self.df_expanded["meta.malware"] == fam]

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

    def leave_one_family_out(self):
        """
        Generator yielding (family, other_df, family_df).

        Family set  = all samples of that family
        Other set = all samples NOT in that family
        """
        for fam in self.families():
            family_df = self.df_expanded[self.df_expanded["meta.malware"] == fam]
            other_df = self.df_expanded[self.df_expanded["meta.malware"] != fam]
            yield fam, other_df, family_df

    def summary(self):
        """Print dataset summary."""
        if self.df_expanded is None:
            raise ValueError("Call preprocess() first.")

        print("Total samples (TLS):", len(self.df_expanded))
        print("Samples with malware labeling:", len(self.df_expanded))
        print("\nFamilies:", self.families())
        print("\nSamples per family:\n", self.samples_per_family())



# -------------------------------------------------------------
# Windows Applications Dataset
# -------------------------------------------------------------
class WinappsDataset:
    """
    Manager for preprocessing, organizing, and splitting windows applications datasets.
    """
    def __init__(self, df_tls: pd.DataFrame):
        self.df_tls = df_tls.copy()                      
        # Extract families
        self.df_tls["meta.application"] = self.df_tls["meta.application"].apply(_extract_application_name)
        self.df_tls["meta.system"] = self.df_tls["meta.system"].apply(_extract_system_name)
        self.df_tls["meta.service"] = pd.NA
        self.df_tls["meta.malware"] = pd.NA
        self.df_tls["meta.host"] = pd.NA
    @staticmethod
    def load_from(path:str,max_rows: Optional[int] = None):
        """
        Load application dataset from parquet files and create a WinappsDataset object.

        This function loads parquet files from the specified path, filters for TLS connections,
        and initializes a WinappsDataset object with the filtered data.

        Args:
            path (str): The file path to the directory containing parquet files.

        Returns:
            WinappsDataset: A WinappsDataset object initialized with TLS connection data.

        Raises:
            FileNotFoundError: If the specified path does not exist.
            ValueError: If no valid parquet files are found at the path.

        Example:
            >>> malware_data = Malware.load_from('/path/to/malware/data')
        """
        df = load_parquet_files(parquet_folder=path, max_rows=max_rows)
        df_tls = get_tls_connections(df)
        return WinappsDataset(df_tls)
    # -------------------------------------------------------------
    # Underlying dataframe
    # -------------------------------------------------------------
    @property
    def dataframe(self): 
        return self.df_tls
    # -------------------------------------------------------------
    # Application-centric accessors
    # -------------------------------------------------------------
    def applications(self) -> List[str]:
        """Return sorted list of applications."""
        if self.df_tls is None:
            raise ValueError("Dataset not initialized.")
        return sorted(self.df_tls["meta.application"].dropna().unique().tolist())

    def samples_per_application(self) -> pd.Series:
        """Return Series with counts per application."""
        if self.df_tls is None:
            raise ValueError("Dataset not initialized.")
        return self.df_tls["meta.application"].value_counts(dropna=True)

    def get_application(self, application: str) -> pd.DataFrame:
        """Return all rows for the specified application."""
        if self.df_tls is None:
            raise ValueError("Dataset not initialized.")
        return self.df_tls[self.df_tls["meta.application"] == application]

    def per_application_split(self, test_size=0.2, random_state=42):
        """
        Perform train/test split independently for each application.

        Returns:
            {
                'GOOGLE.CHROME': (train_df, test_df),
                'MICROSOFT.OUTLOOK': (train_df, test_df),
                ...
            }
        """
        if self.df_tls is None:
            raise ValueError("Dataset not initialized.")

        splits = {}
        for app in self.applications():
            app_df = self.df_tls[self.df_tls["meta.application"] == app]

            if len(app_df) < 2:
                continue

            train_df, test_df = train_test_split(
                app_df,
                test_size=test_size,
                shuffle=True,
                random_state=random_state,
            )
            splits[app] = (train_df, test_df)

        return splits

    def leave_one_application_out(self):
        """
        Generator yielding (application, other_df, app_df).

        app_df   = all samples of that application
        other_df = all samples NOT in that application
        """
        if self.df_tls is None:
            raise ValueError("Dataset not initialized.")

        for app in self.applications():
            app_df = self.df_tls[self.df_tls["meta.application"] == app]
            other_df = self.df_tls[self.df_tls["meta.application"] != app]
            yield app, other_df, app_df

    def summary(self):
        """Print dataset summary."""
        if self.df_tls is None:
            raise ValueError("Dataset not initialized.")

        print("Total samples (TLS):", len(self.df_tls))
        print("Samples with application labeling:", len(self.df_tls))
        print("\nApplications:", self.applications())
        print("\nSamples per application:\n", self.samples_per_application())

from collections import defaultdict   
def _get_label_mapping(df_label: pd.DataFrame):
    df = df_label.copy()
    # extract fields from structured column
    df["ts"] = df["system.os_detection"].apply(
        lambda x: x.get("ts") if isinstance(x, dict) else None
    )
    df["confidence"] = df["system.os_detection"].apply(
        lambda x: x.get("confidence") if isinstance(x, dict) else None
    )
    # hour-level grouping key: YYYYMMDDTHH
    df["ts_hour"] = df["ts"].str.slice(0, 11)

    result = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        if pd.isna(row["ts_hour"]) or pd.isna(row["host.ip"]):
            continue

        result[row["ts_hour"]][row["host.ip"]].append(
            (row["system.os"], row["confidence"])
        )
    # convert defaultdicts to dicts
    # result = {k: dict(v) for k, v in result.items()}
    return result 

def _mapping_to_df(label_mapping:defaultdict) -> pd.DataFrame:
    rows = []
    for ts_hour, ip_map in label_mapping.items():
        if not ip_map: continue
        for ip, candidates in ip_map.items():
            if not candidates:
                continue
            for os_name, conf in candidates:
                if os_name is None or conf is None:
                    continue
                rows.append((ts_hour, ip, os_name, float(conf)))
    return pd.DataFrame(rows, columns=["ts_hour", "ip", "os", "conf"])   
def _best_os_per_key(map_df: pd.DataFrame) -> pd.DataFrame:
    # sum weights per OS within (ts_hour, ip)
    agg = (map_df
           .groupby(["ts_hour", "ip", "os"], as_index=False)["conf"]
           .sum())
    # pick OS with max summed confidence for each (ts_hour, ip)
    idx = agg.groupby(["ts_hour", "ip"])["conf"].idxmax()
    best = agg.loc[idx, ["ts_hour", "ip", "os"]].rename(columns={"os": "meta.system_ext"})
    return best    
# -------------------------------------------------------------
# SOHO Dataset
# -------------------------------------------------------------
class SohoDataset:
    """
    Manager for preprocessing, organizing, and splitting windows applications datasets.
    """        
    def __init__(self, df_tls: pd.DataFrame, df_label : pd.DataFrame):
        self.df_tls = df_tls.copy()
        label_mapping = _get_label_mapping(df_label)

        # build lookup table once
        map_df = _mapping_to_df(label_mapping)
        best_df = _best_os_per_key(map_df)   # provides meta.system (external)

        # vectorized hour key and ip key
        self.df_tls["ts_hour"] = pd.to_datetime(self.df_tls["ts"], unit="s").dt.strftime("%Y%m%dT%H")
        self.df_tls["ip"] = self.df_tls["sa"]

        # internal OS (from flow itself)
        self.df_tls["meta.system_int"] = self.df_tls["meta.system"].apply(_extract_system_name)
        self.df_tls.drop(columns=["meta.system"], inplace=True)

        # merge external OS labels
        self.df_tls = self.df_tls.merge(best_df, on=["ts_hour", "ip"], how="left")
        # best_df provides: meta.system  -> treat as external
        self.df_tls.rename(columns={"meta.system": "meta.system_ext"}, inplace=True)

        # ---- COALESCE: int has higher priority than ext ----
        self.df_tls["meta.system"] = (
            self.df_tls["meta.system_int"]
            .combine_first(self.df_tls["meta.system_ext"])
        )

        # cleanup
        self.df_tls.drop(columns=["meta.system_int", "meta.system_ext", "ts_hour", "ip"], inplace=True)

        # set other columns
        self.df_tls["meta.application"] = pd.NA
        self.df_tls["meta.service"] = pd.NA
        self.df_tls["meta.malware"] = pd.NA
        self.df_tls["meta.host"] = pd.NA        # ??? 
    @staticmethod
    def load_from(path:str, path_label: str, max_rows: Optional[int] = None):
        """
        Load application dataset from parquet files and create a SohoDataset object.

        This function loads parquet files from the specified path, filters for TLS connections,
        and initializes a SohoDataset object with the filtered data.

        Args:
            path (str): The file path to the directory containing parquet files.

        Returns:
            SohoDataset: A SohoDataset object initialized with TLS connection data.

        Raises:
            FileNotFoundError: If the specified path does not exist.
            ValueError: If no valid parquet files are found at the path.
        """
        df = load_parquet_files(parquet_folder=path, max_rows=max_rows)
        df_tls = get_tls_connections(df)
        df_label = load_parquet_files(parquet_folder=path_label, max_rows=max_rows)
        return SohoDataset(df_tls, df_label)
    # -------------------------------------------------------------
    # Underlying dataframe
    # -------------------------------------------------------------
    @property
    def dataframe(self): 
        return self.df_tls