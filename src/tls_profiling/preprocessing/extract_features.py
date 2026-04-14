import pandas as pd
import numpy as np
import warnings
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


def _normalize_tls_token(value: Any) -> str:
    """
    Normalize TLS token values to a canonical uppercase string without `0x`.
    """
    token = str(value).strip()
    if not token or token.lower() == "nan":
        return ""
    if token.lower().startswith("0x"):
        token = token[2:]
    return token.upper()


def _build_list_metadata_frame(
    series: pd.Series,
    classes: list[str],
    prefix: str,
    grease_values: set[str] | None = None,
    unusual_values: set[str] | None = None,
    obsolete_values: set[str] | None = None,
) -> pd.DataFrame:
    """
    Build derived metadata for list-valued TLS fields.

    The output can include:
    - `<prefix>grease`: binary flag set when any GREASE token is present
    - `<prefix>unusual`: binary flag set when any token is outside known/allowed sets
    - `<prefix>obsolete`: binary flag set when any obsolete token is present
    - `<prefix>other`: count of tokens not listed in `classes`
    """
    normalized_rows = series.map(_normalize_to_array).map(
        lambda values: [_normalize_tls_token(value) for value in values]
    )
    known_classes = {_normalize_tls_token(value) for value in classes}
    grease_values = {_normalize_tls_token(value) for value in (grease_values or set())}
    unusual_values = {_normalize_tls_token(value) for value in (unusual_values or set())}
    obsolete_values = {_normalize_tls_token(value) for value in (obsolete_values or set())}

    metadata = pd.DataFrame(index=series.index)

    if grease_values:
        metadata[f"{prefix}grease"] = normalized_rows.map(
            lambda values: int(any(value in grease_values for value in values))
        )

    if obsolete_values:
        metadata[f"{prefix}obsolete"] = normalized_rows.map(
            lambda values: int(any(value in obsolete_values for value in values))
        )

    metadata[f"{prefix}other"] = normalized_rows.map(
        lambda values: int(sum(value not in known_classes for value in values))
    )
    metadata[f"{prefix}unusual"] = normalized_rows.map(
        lambda values: int(
            any(
                value in unusual_values
                or (
                    value not in known_classes
                and value not in grease_values
                and value not in obsolete_values
                )
                for value in values
            )
        )
    )
    return metadata


def _build_mlb_frame(series: pd.Series, classes: list[str], prefix: str = "", suffix: str = "") -> pd.DataFrame:
    """
    Multi-hot encode list-like values with fixed class ordering.

    Parameters
    ----------
    series:
        Input pandas Series where each item is a sequence or scalar value.
    classes:
        Ordered category values used by `MultiLabelBinarizer`.
    prefix:
        Optional prefix prepended to the output column names.
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
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        transformed = mlb.transform(normalized)

    for warning in caught_warnings:
        message = str(warning.message)
        if "unknown class(es)" not in message:
            continue

        unknown_classes = message.split("[", 1)[-1].rsplit("]", 1)[0]
        for unknown_class in [item.strip().strip("'\"") for item in unknown_classes.split(",") if item.strip()]:
            print(f"{prefix}={unknown_class}")

    transformed_any: Any = transformed
    if hasattr(transformed_any, "toarray"):
        transformed = transformed_any.toarray()
    else:
        transformed = np.asarray(transformed)
    column_names = [f"{prefix}{column}{suffix}" for column in mlb.classes_]
    encoded = pd.DataFrame(transformed, columns=column_names, index=series.index)
    return encoded


GREASE = {
    "0A0A","1A1A","2A2A","3A3A","4A4A","5A5A","6A6A","7A7A",
    "8A8A","9A9A","AAAA","BABA","CACA","DADA","EAEA","FAFA"
}

UNUSUAL_SEXT = {
    "4469", "7550" }
UNUSUAL_CEXT = {
    "44CD", "7550" }
UNUSUAL_CCS = {
    "00FB",
    "00FC",
    "00FD",
}
OBSOLETE_CCS = {
    # NULL / no encryption
    "0000","0001","0002","003B","003C","003D",

    # EXPORT-grade
    "0003","0006","0008","000B","000C","000E","0010","0011","0012",
    "0013","0014","0015","0016",

    # RC4
    "0004","0005","0007","0009","000A","0017","0018","0019",
    "C011","C012","C007","C002",

    # DES (single DES)
    "0008","0009","000A","0012","0013","0014",

    # 3DES
    "000A","0016","001B","001E","0023",
    "C012","C008","C013",

    # MD5-based
    "0001","0004","0017","0018",

    # Static DH (no forward secrecy)
    "0030","0031","0032","0033","0036","0037","0038","0039",
    "003E","003F","0040","0041","0042","0043","0044","0045",

    # Additional weak / legacy CBC (non-AEAD RSA)
    "002F","0035","003C","003D"
}


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
    tls_sext_mlb = _build_mlb_frame(df['tls.sext'], sext_possible_values, prefix='tls.sext.')

    ccs_possible_values = ['0004','0005','0032','0033','0035','0038','0039','1301','1302','1303','000A',
                           '002F','003C','003D','009C','009D','009E','009F','00FF','C007','C009','C00A','C011','C013',
                           'C014','C023','C024','C027','C028','C02B','C02C','C02F','C030','CC13','CC14','CC15','CCA8','CCA9']
    tls_ccs_mlb_renamed = _build_mlb_frame(df['tls.ccs'], ccs_possible_values, prefix='tls.ccs.')

    cext_possible_values = ['0000','0005','0010','0012','0015','0017','0023','0033','3374',
                            '4469','000A','000B','000D','001B','002B','002D','FE0D','FF01']
    tls_cext_mlb_renamed = _build_mlb_frame(df['tls.cext'], cext_possible_values, prefix='tls.cext.')

    csg_possible_values = ['0201','0202','0203','0301','0303','0401','0403','0501','0503',
                           '0601','0603','0804','0805','0806','0302','0402','0502','0602']
    tls_csg_mlb_renamed = _build_mlb_frame(df['tls.csg'], csg_possible_values, prefix='tls.csg.')

    alpn_possible_values = ['h2','http/1.1','']
    tls_alpn_mlb_renamed = _build_mlb_frame(df['tls.alpn'], alpn_possible_values, prefix='tls.alpn.')
    
    csv_possible_values = ['0303','0304','']
    tls_csv_mlb_renamed = _build_mlb_frame(df['tls.csv'], csv_possible_values, prefix='tls.csv.')
    
    ssv_possible_values = ['0304','']
    tls_ssv_mlb_renamed = _build_mlb_frame(df['tls.ssv'], ssv_possible_values, prefix='tls.ssv.')
    
    records_sequences = df['tls.rec'].map(_normalize_to_array)
    records_data = pd.DataFrame(_pad_sequences(records_sequences.values, maxlen=tls_rec_len), columns=tls_rec_columns_names)
    dataset = pd.concat([flow_data, tls_data, tls_sext_mlb, tls_ccs_mlb_renamed, tls_cext_mlb_renamed, 
                         tls_csg_mlb_renamed, tls_alpn_mlb_renamed, tls_csv_mlb_renamed, tls_ssv_mlb_renamed, records_data], axis=1).fillna(0)
    return dataset


def extract_features_ext(df: pd.DataFrame, tls_rec_len: int = 20) -> pd.DataFrame:
    """
    Extended feature extraction with derived list metadata.

    In addition to the base encoded features, this adds:
    - GREASE usage flags, e.g. `tls.ccs.grease`
    - unusual-value flags, e.g. `tls.sext.unusual`
    - obsolete cipher-suite usage, e.g. `tls.ccs.obsolete`
    - `other` counters for values outside the configured class lists
    """
    df = df.copy()
    dataset = extract_features(df, tls_rec_len=tls_rec_len)

    grease_values = {
        '0A0A', '1A1A', '2A2A', '3A3A', '4A4A', '5A5A', '6A6A', '7A7A',
        '8A8A', '9A9A', 'AAAA', 'BABA', 'CACA', 'DADA', 'EAEA', 'FAFA',
    }

    sext_possible_values = ['0000','0005','0010','0017','0023','0033','000B','002B','FF01']
    ccs_possible_values = ['0004','0005','0032','0033','0035','0038','0039','1301','1302','1303','000A',
                           '002F','003C','003D','009C','009D','009E','009F','00FF','C007','C009','C00A','C011','C013',
                           'C014','C023','C024','C027','C028','C02B','C02C','C02F','C030','CC13','CC14','CC15','CCA8','CCA9']
    cext_possible_values = ['0000','0005','0010','0012','0015','0017','0023','0033','3374',
                            '4469','000A','000B','000D','001B','002B','002D','FE0D','FF01']
    csg_possible_values = ['0201','0202','0203','0301','0303','0401','0403','0501','0503',
                           '0601','0603','0804','0805','0806','0302','0402','0502','0602']
    alpn_possible_values = ['h2','http/1.1','']
    csv_possible_values = ['0303','0304','']
    ssv_possible_values = ['0304','']
    derived_frames = [
        _build_list_metadata_frame(
            df['tls.sext'],
            sext_possible_values,
            prefix='tls.sext.',
            grease_values=grease_values,
            unusual_values=UNUSUAL_SEXT,
        ),
        _build_list_metadata_frame(
            df['tls.ccs'],
            ccs_possible_values,
            prefix='tls.ccs.',
            grease_values=grease_values,
            unusual_values=UNUSUAL_CCS,
            obsolete_values=OBSOLETE_CCS,
        ),
        _build_list_metadata_frame(
            df['tls.cext'],
            cext_possible_values,
            prefix='tls.cext.',
            grease_values=grease_values,
            unusual_values=UNUSUAL_CEXT,
        ),
        _build_list_metadata_frame(df['tls.csg'], csg_possible_values, prefix='tls.csg.', grease_values=grease_values),
        _build_list_metadata_frame(df['tls.alpn'], alpn_possible_values, prefix='tls.alpn.'),
        _build_list_metadata_frame(df['tls.csv'], csv_possible_values, prefix='tls.csv.', grease_values=grease_values),
        _build_list_metadata_frame(df['tls.ssv'], ssv_possible_values, prefix='tls.ssv.', grease_values=grease_values),
    ]

    return pd.concat([dataset, *derived_frames], axis=1).fillna(0)
