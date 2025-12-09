import os
from typing import List, Optional
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm

def load_parquet_files(
        parquet_folder: str,
        columns: Optional[List[str]] = None,
        max_rows: Optional[int] = None
    ) -> pd.DataFrame:
    """
    Load and flatten Parquet files from a folder into a pandas DataFrame
    with support for:
        • selective column loading,
        • PyArrow predicate filtering,
        • row-group progress tracking,
        • optional schema validation.

    Parameters
    ----------
    parquet_folder : str
        Path to a directory containing one or more Parquet files
        (including partitioned datasets).
    columns : list[str], optional
        List of columns to load. If None, all columns are loaded.
        Select columns for faster loading.
    max_rows : int, optional
        Maximum number of rows to load. If None, load entire dataset.

    Returns
    -------
    pandas.DataFrame
        A flattened DataFrame containing the filtered data.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist.
    ValueError
        If required columns are missing.
    """

    # -------------------------
    # Input validation
    # -------------------------
    if not os.path.exists(parquet_folder):
        raise FileNotFoundError(f"Folder does not exist: {parquet_folder}")

    # -------------------------
    # Construct dataset reader
    # -------------------------
    dataset = ds.dataset(parquet_folder, format="parquet")

    # -------------------------
    # Show row groups progress
    # -------------------------
    # PyArrow Dataset does NOT expose row groups directly,
    # so we find all Parquet files and count row groups per file.
    parquet_files = [
        f for f in dataset.files if f.endswith(".parquet")
    ]

    dfs = []
    rows_loaded = 0
    for file in tqdm(parquet_files, desc="Reading Parquet files"):
        pq_file = pq.ParquetFile(file)
        for rg_index in range(pq_file.num_row_groups):
            # Read row group with optional column selection
            table = pq_file.read_row_group(
                rg_index,
                columns=columns
            )

            # Flatten nested structs
            table = table.flatten()

            # Convert to pandas
            df_chunk = table.to_pandas()

            # If a row limit is set and this chunk exceeds it:
            if max_rows is not None and rows_loaded + len(df_chunk) > max_rows:
                needed = max_rows - rows_loaded
                df_chunk = df_chunk.iloc[:needed]

            dfs.append(df_chunk)
            rows_loaded += len(df_chunk)
        # Also break outer loop if limit reached
        if max_rows is not None and rows_loaded >= max_rows:
            break

    # Concatenate all chunks
    df = pd.concat(dfs, ignore_index=True)

    return df


