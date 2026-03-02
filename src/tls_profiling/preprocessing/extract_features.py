import pandas as pd
import numpy as np
from typing import Any
from sklearn.preprocessing import MultiLabelBinarizer

def _resize_row(row, maxlen, pad_value=0):
    """
    Resize a 1D sequence to a fixed length.

    Shorter sequences are right-padded with `pad_value`; longer sequences
    are truncated from the end.
    """
    current_length = len(row)
    if current_length < maxlen:
        pad_width = maxlen - current_length
        row = np.pad(row, pad_width=(0, pad_width), mode='constant', constant_values=pad_value)
    else:
        row = row[:maxlen]
    return row


def _pad_sequences(rows, maxlen, pad_value=0):
    """
    Apply fixed-length resizing to each sequence in `rows`.

    Parameters
    ----------
    rows:
        Iterable of sequence-like values.
    maxlen:
        Target sequence length.
    pad_value:
        Value used to right-pad shorter sequences.
    """
    resized_rows = [_resize_row(row, maxlen, pad_value=pad_value) for row in rows]
    return resized_rows


def _normalize_to_array(x):
    """
    Normalize a scalar or sequence value into a 1D NumPy array.

    Null-like values become an empty array; scalars become a single-item array.
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x)
    if pd.isna(x):
        return np.array([])
    return np.array([x])


def _build_mlb_frame(series: pd.Series, classes: list[str], suffix: str = "") -> pd.DataFrame:
    """
    Multi-hot encode list-like values with fixed class ordering.

    Parameters
    ----------
    series:
        Input pandas Series where each item is a sequence or scalar value.
    classes:
        Ordered category values used by `MultiLabelBinarizer`.
    suffix:
        Optional suffix appended to output column names.

    Returns
    -------
    pd.DataFrame
        Dense DataFrame with one column per category.
    """
    normalized = series.map(_normalize_to_array).tolist()
    mlb = MultiLabelBinarizer(classes=classes, sparse_output=False)
    mlb.fit([])
    transformed = mlb.transform(normalized)
    transformed_any: Any = transformed
    if hasattr(transformed_any, "toarray"):
        transformed = transformed_any.toarray()
    else:
        transformed = np.asarray(transformed)
    encoded = pd.DataFrame(transformed, columns=mlb.classes_)
    if suffix:
        encoded = encoded.add_suffix(suffix)
    return encoded

def extract_features(df: pd.DataFrame, tls_rec_len:int=20) -> pd.DataFrame:
    """
    Extract model-ready features from a raw TLS-flow dataframe.

    The output contains:
    - selected numeric flow fields
    - selected scalar TLS fields
    - fixed-class multi-hot encodings for selected list-like TLS fields
    - padded/truncated `tls.rec` sequence columns

    Parameters
    ----------
    df:
        Input dataframe with raw flow/TLS columns.
    tls_rec_len:
        Number of `tls.rec.*` columns to create.

    Returns
    -------
    pd.DataFrame
        Feature dataframe suitable as input to downstream preprocessing.
    """
    df = df.copy()
    tls_rec_columns_names = np.array([f"tls.rec.{i}" for i in range(tls_rec_len)])
    flow_data = df[['bs', 'ps', 'br', 'pr', 'td']].astype(float)
    tls_data = df[['tls.cver','tls.sver','tls.scs']].fillna(0).astype(str) 

    sext_possible_values = ['0000','0005','0010','0017','0023','0033','000B','002B','FF01']
    tls_sext_mlb = _build_mlb_frame(df['tls.sext'], sext_possible_values)

    ccs_possible_values = ['0004','0005','0032','0033','0035','0038','0039','1301','1302','1303','000A',
                           '002F','003C','003D','009C','009D','009E','009F','00FF','C007','C009','C00A','C011','C013',
                           'C014','C023','C024','C027','C028','C02B','C02C','C02F','C030','CC13','CC14','CC15','CCA8','CCA9']
    tls_ccs_mlb_renamed = _build_mlb_frame(df['tls.ccs'], ccs_possible_values, '_ccs')

    cext_possible_values = ['0000','0005','0010','0012','0015','0017','0023','0033','3374',
                            '4469','000A','000B','000D','001B','002B','002D','FE0D','FF01']
    tls_cext_mlb_renamed = _build_mlb_frame(df['tls.cext'], cext_possible_values, '_cext')

    csg_possible_values = ['0201','0202','0203','0301','0303','0401','0403','0501','0503',
                           '0601','0603','0804','0805','0806','0302','0402','0502','0602']
    tls_csg_mlb_renamed = _build_mlb_frame(df['tls.csg'], csg_possible_values, '_csg')

    alpn_possible_values = ['h2','http/1.1','']
    tls_alpn_mlb_renamed = _build_mlb_frame(df['tls.alpn'], alpn_possible_values, '_alpn')
    
    csv_possible_values = ['0303','0304','']
    tls_csv_mlb_renamed = _build_mlb_frame(df['tls.csv'], csv_possible_values, '_csv')
    
    ssv_possible_values = ['0304','']
    tls_ssv_mlb_renamed = _build_mlb_frame(df['tls.ssv'], ssv_possible_values, '_ssv')
    
    records_sequences = df['tls.rec'].map(_normalize_to_array)
    records_data = pd.DataFrame(_pad_sequences(records_sequences.values, maxlen=tls_rec_len), columns=tls_rec_columns_names)
    dataset = pd.concat([flow_data, tls_data, tls_sext_mlb, tls_ccs_mlb_renamed, tls_cext_mlb_renamed, 
                         tls_csg_mlb_renamed, tls_alpn_mlb_renamed, tls_csv_mlb_renamed, tls_ssv_mlb_renamed, records_data], axis=1).fillna(0)
    return dataset
