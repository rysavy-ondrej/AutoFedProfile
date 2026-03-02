from ..io.schema import list_columns_by_role, get_tls_schema
import pandas as pd
import numpy as np


def _normalize_cardinality_value(value):
    """
    Normalize values for cardinality counting.

    - String values are kept as-is.
    - Array-like values are converted to a single string by concatenating
      elements with commas.
    - Null-like values are mapped to `pd.NA`.
    - Other scalar values are returned unchanged.
    """
    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple, np.ndarray)):
        return ",".join(map(str, value))

    if pd.isna(value):
        return pd.NA

    return value

def get_cardinality_stat_per_x(df:pd.DataFrame, x_col:str) -> pd.DataFrame:
    """
    Compute per-group cardinality statistics for categorical schema fields.
    To include only a specific type of connections, filter the input df first, for instance, to 
    get information on system connections only:

    mask = df["connection_label"] == "system"
    get_cardinality_stat_per_x(df[mask], `meta.application.name`)
    
    Parameters
    ----------
    df:
        Input dataframe containing at least the grouping column and all
        categorical columns defined by `get_tls_schema()`.
    x_col:
        Grouping column (for example `meta.application.name`).

    Returns
    -------
    pd.DataFrame
        Dataframe indexed by `x_col` with:
        - cardinality (`nunique`) for each categorical field
        - `connections` column with number of rows in each group
    """
    fields = list_columns_by_role(get_tls_schema(), "tls")  
    if fields is None:
        raise ValueError("No categorical fields available in schema")

    work_df = df.copy()
    for col in fields:
        work_df[col] = work_df[col].map(_normalize_cardinality_value)

    # Cardinality of each categorical field per group.
    cardinality_per_app = work_df.groupby(x_col)[fields].nunique()   # index = APP_COL


    # Total number of rows (connections) per group.
    total_connections_per_app = (
        work_df.groupby(x_col)
        .size()
        .rename("connections")     # Series, index = APP_COL
    )

    cardinality_per_app = (
        cardinality_per_app
        .join(total_connections_per_app)  # joins on index, no overlapping columns
    )
    return cardinality_per_app


def get_cardinality_stat(df:pd.DataFrame)-> pd.DataFrame:
    """
    Compute global cardinality summary for categorical schema fields.

    For each categorical field, this function reports:
    - number of unique values (`cardinality`)
    - total number of rows in the dataframe
    - top 3 most frequent values with frequency and ratio

    Parameters
    ----------
    df:
        Input dataframe containing categorical columns from `get_tls_schema()`.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per field plus a final `TOTAL_ROWS` row.
    """
    fields = list_columns_by_role(get_tls_schema(), "categorical")
    if fields is None:
        raise ValueError("No categorical fields available in schema")

    rows = []
    for col in fields:
        s = df[col].map(_normalize_cardinality_value).dropna()

        nunique = s.nunique()
        total = len(df)

        top3 = s.value_counts().head(3)

        # Build structured row
        row = {
            "field": col,
            "cardinality": nunique,
            "total_rows": total,
        }

        for i in range(3):
            if i < len(top3):
                row[f"top{i+1}_value"] = top3.index[i]
                row[f"top{i+1}_freq"] = int(top3.iloc[i])
                row[f"top{i+1}_ratio"] = round(top3.iloc[i] / total, 4)
            else:
                row[f"top{i+1}_value"] = None
                row[f"top{i+1}_freq"] = 0
                row[f"top{i+1}_ratio"] = 0.0

        rows.append(row)

    cardinality_df = pd.DataFrame(rows)

    # Add overall row count summary
    cardinality_df = pd.concat([
        cardinality_df,
        pd.DataFrame([{
            "field": "TOTAL_ROWS",
            "cardinality": len(df),
            "total_rows": len(df)
        }])
    ], ignore_index=True)

    return cardinality_df


import pandas as pd
import numpy as np

def _is_null(v) -> bool:
    # robust null check for Python/NumPy/Pandas scalars
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except Exception:
        return False

def _is_array_like(v) -> bool:
    # treat list/tuple/ndarray as array-like; exclude strings/bytes
    return isinstance(v, (list, tuple, np.ndarray))

def get_df_tls_array_field_cardinality(df, column:str):
    size = len(df)
    r = (
        df.assign(
            tls_field_str=df[column].apply(
                lambda x: ",".join(x) if x is not None else None
            )
        )
        .groupby("tls_field_str")
        .size()
        .reset_index(name="count")
    )

    r["ratio"] = r["count"] / size
    r.rename(columns={"tls_field_str": column}, inplace=True)
    return r

def get_df_tls_scalar_field_cardinality(df, column):
   size = len(df)
   df_res = (
      df.groupby(column)
         .size()
         .reset_index(name="count")
   )    
   df_res["ratio"] = df_res["count"] / size
   return df_res

def get_df_tls_field_card(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Returns a dataframe with [value, count, ratio] for the given column.
    Automatically handles scalar vs array-like (list/tuple/np.ndarray) columns.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in df")

    s = df[column]

    # infer from first non-null value (works for mixed columns)
    first = next((v for v in s if not _is_null(v)), None)

    if first is None:
        # all nulls -> scalar grouping will yield a single NaN group
        return get_df_tls_scalar_field_cardinality(df, column)

    if _is_array_like(first):
        return get_df_tls_array_field_cardinality(df, column)

    return get_df_tls_scalar_field_cardinality(df, column)