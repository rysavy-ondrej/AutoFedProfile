import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Malware:
    """
    Manager for preprocessing, organizing, and splitting TLS/malware datasets.

    Features:
    ---------
    ✓ Preprocess raw dataset into TLS-only flows
    ✓ Extract malware families (multi-label)
    ✓ Per-family train/test split
    ✓ Leave-One-Family-Out cross-validation (LOFO-CV)
    ✓ Filtering utilities
    ✓ Dataset statistics

    The class assumes:
        df["meta.malware"] is a dict with keys:
            - "family": numpy array of family names
            - "score": float
    and df["mal_fams"] is a list of malware families per row.
    """

    # -------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------
    def __init__(self, df_tls: pd.DataFrame):
        self.df_tls = df_tls.copy()          
        self.df_expanded = None                

        # Extract families
        self.df_tls["mal_fams"] = self.df_tls["meta.malware"].apply(self._extract_families)

        # Explode: one row per family
        self.df_expanded = self.df_tls.explode("mal_fams")
        self.df_expanded = self.df_expanded[
            self.df_expanded["mal_fams"].notna() & (self.df_expanded["mal_fams"] != "")
        ]

    @staticmethod
    def _extract_families(m):
        """Extract malware families from metadata."""
        if m is None or not isinstance(m, dict) or "family" not in m:
            return ["_none_"]
        fams = m["family"]
        return fams.tolist() if hasattr(fams, "tolist") else list(fams)
    # -------------------------------------------------------------
    # Dataset properties
    # -------------------------------------------------------------
    def families(self):
        """Return sorted list of malware families."""
        if self.df_expanded is None:
            raise ValueError("Call preprocess() first.")
        return sorted(self.df_expanded["mal_fams"].unique())

    def samples_per_family(self):
        """Return Series with counts per family."""
        if self.df_expanded is None:
            raise ValueError("Call preprocess() first.")
        return self.df_expanded["mal_fams"].value_counts()

    # -------------------------------------------------------------
    # Per-family filtering
    # -------------------------------------------------------------
    def get_family(self, family: str):
        """Return all rows containing the specified malware family."""
        if self.df_expanded is None:
            raise ValueError("Call preprocess() first.")
        return self.df_expanded[self.df_expanded["mal_fams"] == family]

    # -------------------------------------------------------------
    # Train/Test Split Per Family
    # -------------------------------------------------------------
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
            fam_df = self.df_expanded[self.df_expanded["mal_fams"] == fam]

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

    # -------------------------------------------------------------
    # Leave-One-Family-Out Cross-Validation (LOFO-CV)
    # -------------------------------------------------------------
    def leave_one_family_out(self):
        """
        Generator yielding (family, train_df, test_df).

        Test set  = all samples of that family
        Train set = all samples NOT in that family
        """
        if self.df_expanded is None:
            raise ValueError("Call preprocess() first.")

        for fam in self.families():
            test_df = self.df_expanded[self.df_expanded["mal_fams"] == fam]
            train_df = self.df_expanded[self.df_expanded["mal_fams"] != fam]
            yield fam, train_df, test_df

    # -------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------
    def summary(self):
        """Print dataset summary."""
        if self.df_expanded is None:
            raise ValueError("Call preprocess() first.")

        print("Total samples (TLS):", len(self.df_tls))
        print("Samples with malware labeling:", len(self.df_expanded))
        print("\nFamilies:", self.families())
        print("\nSamples per family:\n", self.samples_per_family())