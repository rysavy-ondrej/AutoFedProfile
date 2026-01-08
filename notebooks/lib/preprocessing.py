import os
from typing import List, Optional

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

RECORD_SEQUENCE_SIZE=20
tls_columns_names = np.array([f"tls.rec.{i}" for i in range(RECORD_SEQUENCE_SIZE)])

# Resize row in the array
def resize_row(row, maxlen, pad_value=0):
    current_length = len(row)
    if current_length < maxlen:
        # Calculate the amount of padding needed
        pad_width = maxlen - current_length
        # Pad at the end (you can also pad at the beginning or both sides)
        row = np.pad(row, pad_width=(0, pad_width), mode='constant', constant_values=pad_value)
    else:
        # If the row is longer than the target length, slice it
        row = row[:maxlen]
    return row
# Resize the matrix by padding or removing columns
def pad_sequences(rows, maxlen, pad_value=0):
    resized_rows = [resize_row(row, maxlen) for row in rows]
    return resized_rows

# Extracts features from raw dataset. This will provide suitable output to the preprocessing pipeline.
# Flow related columns: 'BytesOut', 'PacketsOut', 'BytesIn', 'PacketsIn', 'Duration'
# TLS handshake columns: 'TlsClientVersion','TlsServerVersion','TlsServerCipherSuite'
# TLS handshake columns transformed with MultiLabelBinarizer: TlsServerExtensions, TlsClientCipherSuites,
#     TlsClientExtensions, TlsClientSupportedGroups, TlsALPN, TlsClinetSupportedVersions,
#     TlsServerSupportedVerions, 
# TLS record sizes: 'RecordSequence' mapped as 'TlsRecord_X'
#
# The output is a DataFrame with the above specified columns. This dataframe can be used as the input to next
# processing block (preprocessor).
#
def extract_features(df):
    # Flow data
    flow_data = df[['bs', 'ps', 'br', 'pr', 'td']].astype(float)
    # TLS handshake data
    tls_data = df[['tls.cver','tls.sver','tls.scs']].fillna(0).astype(str) 
    # Other TLS attributes - fields of values transformed with MiltiLabelBinarizer
    #selected values are based on the most frequent values in tested datasets
    #tls.sext
    df['tls.sext'] = df['tls.sext'].apply(
    lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
    )
    sext_possible_values = ['0000','0005','0010','0017','0023','0033','000B','002B','FF01']
    mlb = MultiLabelBinarizer(classes = sext_possible_values)
    mlb.fit([])
    transformed = mlb.transform(df['tls.sext'])
    tls_sext_mlb = pd.DataFrame(transformed,columns=mlb.classes_)
    # print(mlb.classes_)
   
    # tls.ccs
    df['tls.ccs'] = df['tls.ccs'].apply(
    lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
    )
    ccs_possible_values = ['0004','0005','0032','0033','0035','0038','0039','1301','1302','1303','000A',
                           '002F','003C','003D','009C','009D','009E','009F','00FF','C007','C009','C00A','C011','C013',
                           'C014','C023','C024','C027','C028','C02B','C02C','C02F','C030','CC13','CC14','CC15','CCA8','CCA9']
    mlb2 = MultiLabelBinarizer(classes = ccs_possible_values)
    mlb2.fit([])
    transformed2 = mlb2.transform(df['tls.ccs'])
    tls_ccs_mlb = pd.DataFrame(transformed2,columns=mlb2.classes_)
    tls_ccs_mlb_renamed = tls_ccs_mlb.add_suffix('_ccs')

    # tls.cext
    df['tls.cext'] = df['tls.cext'].apply(
    lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
    )
    cext_possible_values = ['0000','0005','0010','0012','0015','0017','0023','0033','3374',
                            '4469','000A','000B','000D','001B','002B','002D','FE0D','FF01']
    mlb2 = MultiLabelBinarizer(classes = cext_possible_values)
    mlb2.fit([])
    transformed2 = mlb2.transform(df['tls.cext'])
    tls_cext_mlb = pd.DataFrame(transformed2,columns=mlb2.classes_)
    tls_cext_mlb_renamed = tls_cext_mlb.add_suffix('_cext')

    #  tls.csg
    df['tls.csg'] = df['tls.csg'].apply(
    lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
    )
    csg_possible_values = ['0201','0202','0203','0301','0303','0401','0403','0501','0503',
                           '0601','0603','0804','0805','0806','0302','0402','0502','0602']
    mlb2 = MultiLabelBinarizer(classes = csg_possible_values)
    mlb2.fit([])
    transformed2 = mlb2.transform(df['tls.csg'])
    tls_csg_mlb = pd.DataFrame(transformed2,columns=mlb2.classes_)
    tls_csg_mlb_renamed = tls_csg_mlb.add_suffix('_csg')

    # tls.alpn
    df['tls.alpn'] = df['tls.alpn'].apply(
    lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
    )
    alpn_possible_values = ['h2','http/1.1','']
    mlb2 = MultiLabelBinarizer(classes = alpn_possible_values)
    mlb2.fit([])
    transformed2 = mlb2.transform(df['tls.alpn'])
    tls_alpn_mlb = pd.DataFrame(transformed2,columns=mlb2.classes_)
    tls_alpn_mlb_renamed = tls_alpn_mlb.add_suffix('_alpn')
    
    # tls.csv
    df['tls.csv'] = df['tls.csv'].apply(
    lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
    )
    csv_possible_values = ['0303','0304','']
    mlb2 = MultiLabelBinarizer(classes = csv_possible_values)
    mlb2.fit([])
    transformed2 = mlb2.transform(df['tls.csv'])
    tls_csv_mlb = pd.DataFrame(transformed2,columns=mlb2.classes_)
    tls_csv_mlb_renamed = tls_csv_mlb.add_suffix('_csv')
    
    # tls.ssv
    df['tls.ssv'] = df['tls.ssv'].apply(
    lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
    )
    ssv_possible_values = ['0304','']
    mlb2 = MultiLabelBinarizer(classes = ssv_possible_values)
    mlb2.fit([])
    transformed2 = mlb2.transform(df['tls.ssv'])
    tls_ssv_mlb = pd.DataFrame(transformed2,columns=mlb2.classes_)
    tls_ssv_mlb_renamed = tls_ssv_mlb.add_suffix('_ssv')
    
    #zde pridat sloupce i pro dalsi seznamy hodnot
    # TLS records 
    records_data = pd.DataFrame( pad_sequences(df['tls.rec'].values, maxlen=RECORD_SEQUENCE_SIZE), columns=tls_columns_names)
    dataset = pd.concat([flow_data, tls_data, tls_sext_mlb, tls_ccs_mlb_renamed, tls_cext_mlb_renamed, 
                         tls_csg_mlb_renamed, tls_alpn_mlb_renamed, tls_csv_mlb_renamed, tls_ssv_mlb_renamed, records_data], axis=1).fillna(0)
    # print(dataset.shape)    
    return dataset
#
# Fits the preprocessor that contains scalers for numerical features and OneHotEncoder for categorical data.
# The result is the Pipeline that can be used for further data processing before they are fed in the Autoencoder.
# 
def fit_preprocessor(df):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_tls', MinMaxScaler(), tls_columns_names),
            ('num_flow', MinMaxScaler(), ['bs', 'ps', 'br', 'pr', 'td']),           
            ('cat1', OneHotEncoder(categories = [['0303','0301','0300','0302','']], sparse_output=False, handle_unknown='ignore'), ['tls.cver']),
            ('cat2', OneHotEncoder(categories = [['0303','0301','0300','0302','']], sparse_output=False, handle_unknown='ignore'),['tls.sver']),
            ('cat3', OneHotEncoder(categories = [['', '0xc02b', '0xc02f', '0xcc14', '0xcc13', '0xc030', '0xc014',
                                                       '0x0033', '0x0035', '0x009c', '0xc013', '0x0005', '0x009e',
                                                       '0x002f', '0x000a', '0xcca9', '0xcca8', '0x009f', '0xc011',
                                                       '0xc028', '0x009d', '0x0039', '0x1301', '0x1302', '0xc02c',
                                                       '0x003c', '0xc009', '0x0004', '0xc027', '0x003d', '0x0067']], 
                                    sparse_output=False, handle_unknown='ignore'),['tls.scs']),
            ('remaining', 'passthrough', [])         
        ], remainder='passthrough')
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    pipeline.fit(df)
    return pipeline

def normalize_to_list(x):
    # Case 1: already a list
    if isinstance(x, list):
        return x

    # Case 3: numpy array or pyarrow list 
    if isinstance(x, (np.ndarray,)):
        if x.size == 0:
            return []
        return list(x)

    # Case 2: missing values (None, NaN, pd.NA)
    if x is None or pd.isna(x):
        return []

    # Case 4: arrow arrays (pyarrow.ListScalar / ListArray)
    try:
        import pyarrow as pa
        if isinstance(x, (pa.ListArray, pa.ListScalar)):
            return [] if len(x) == 0 else list(x.to_pylist())
    except ImportError:
        pass

    # Case 5: anything else — treat as scalar
    return [x]

def extract_rec_seq(recs):
    """
    Convert a list of TLS record dicts into a sequence
    of signed lengths (dir * len).

    Parameters
    ----------
    recs : list[dict] or None
        Input TLS record list. Each record must contain:
        - rec["dir"]
        - rec["len"]
    maxlen : int
        Desired output sequence length.
    pad_value : int, optional
        Value to use for padding if sequence is shorter than maxlen.

    Returns
    -------
    ndarray[int]
        Sequence of signed TLS record lengths.
    """

    # Handle missing or invalid values
    if (recs is None or not isinstance(recs, (list, np.ndarray))):
        return np.array([], dtype=np.int32)

    # Extract signed lengths: dir * len
    seq = []
    for rec in recs:
        try:
            signed_len = int(rec["dir"]) * int(rec["len"])
            seq.append(signed_len)
        except Exception:
            # Invalid record → treat as zero
            seq.append(0)
    return np.array(seq, dtype=np.int32)

def get_tls_connections(df: pd.DataFrame, rec_seq_len: int = 20) -> pd.DataFrame:
    """
    Convert the input DataFrame into a TLS-only connection collection.

    Steps:
    -------
    1. Keep only rows where protocol/tls indicator exists (tls.ja3, tls.ciphers, etc.).
    2. Rename IP counter columns:
           ip.bsent → bs
           ip.psent → ps
           ip.brecv → br
           ip.precv → pr
    3. Normalize list-like TLS attributes using normalize_to_list:
           tls.sexts → tls.sext
           tls.csigs → tls.cgs
           tls.cciphers → tls.ccs
           tls.cexts → tls.cext
           tls.ssvers → tls.ssv
           tls.cvers → tls.csv
    4. Convert tls.recs into fixed-length sequences using extract_rec_seq().

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing TLS and non-TLS connections.
    rec_seq_len : int
        Length of the padded/truncated TLS record-length sequence.

    Returns
    -------
    pd.DataFrame
        Cleaned and normalized TLS-only dataset.
    """

    df = df.copy()

    # ----------------------------------------------------------
    # 1. Keep only TLS connections (require:// pick one reliable TLS field)
    # ----------------------------------------------------------
    if "tls.ja3" in df.columns:
        df = df[df["tls.ja3"].notna()]

    # ----------------------------------------------------------
    # 2. Rename base IP stats
    # ----------------------------------------------------------
    rename_map = {
        "ip.bsent": "bs",
        "ip.psent": "ps",
        "ip.brecv": "br",
        "ip.precv": "pr",
        "tcp.dstport" : "dp",
        "tcp.srcport" : "sp",
        "ip.dst" : "da",
        "ip.src" : "sa",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ----------------------------------------------------------
    # 3. Normalize TLS lists & rename
    # ----------------------------------------------------------
    list_cols = {
        "tls.sexts": "tls.sext",
        "tls.csigs": "tls.csg",
        "tls.cciphers": "tls.ccs",
        "tls.cexts": "tls.cext",
        "tls.ssvers": "tls.ssv",
        "tls.csvers": "tls.csv",
        "tls.scipher" : "tls.scs",
        "tls.alpn" : "tls.alpn",
    }

    for old, new in list_cols.items():
        if old in df.columns:
            df[new] = df[old].apply(normalize_to_list)

    # ----------------------------------------------------------
    # 4. Transform TLS record sequences (tls.recs)
    # ----------------------------------------------------------
    if "tls.recs" in df.columns:
        df["tls.rec"] = df["tls.recs"].apply(lambda r: extract_rec_seq(r))
    else:
        df["tls.rec"] = np.zeros((len(df), rec_seq_len), dtype=np.int32)

    # ---------------------------------------------------------
    # 5. Define final allowed schema (keep all meta.* too)
    # ---------------------------------------------------------
    BASE_KEEP = [
        "pt", "sa", "sp", "da", "dp",
        "ps", "pr", "bs", "br",
        "ts", "td",
        "tls.cver", "tls.ccs", "tls.cext", "tls.csg", "tls.csv",
        "tls.alpn", "tls.sni",
        "tls.sver", "tls.scs", "tls.sext", "tls.ssv",
        "tls.ja3", "tls.ja3s", "tls.ja4", "tls.ja4s",
        "tls.rec",
    ]

    # Columns that actually exist in this dataset
    existing_base = [c for c in BASE_KEEP if c in df.columns]

    # Collect all meta.* columns dynamically
    meta_columns = [c for c in df.columns if c.startswith("meta.")]

    # Final column selection
    final_cols = existing_base + meta_columns

    df = df[final_cols]
    
    return df