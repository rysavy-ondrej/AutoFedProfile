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

def normalize_to_array(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x)
    if pd.isna(x):
        return np.array([])
    return np.array([x])

# Extracts features from raw dataset. This will provide suitable output to the preprocessing pipeline.
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
    df['tls.sext'] = df['tls.sext'].apply(normalize_to_array)
    sext_possible_values = ['0000','0005','0010','0017','0023','0033','000B','002B','FF01']
    mlb = MultiLabelBinarizer(classes = sext_possible_values)
    mlb.fit([])
    transformed = mlb.transform(df['tls.sext'])
    tls_sext_mlb = pd.DataFrame(transformed,columns=mlb.classes_)
    # print(mlb.classes_)
   
    # tls.ccs
    df['tls.ccs'] = df['tls.ccs'].apply(normalize_to_array)
    ccs_possible_values = ['0004','0005','0032','0033','0035','0038','0039','1301','1302','1303','000A',
                           '002F','003C','003D','009C','009D','009E','009F','00FF','C007','C009','C00A','C011','C013',
                           'C014','C023','C024','C027','C028','C02B','C02C','C02F','C030','CC13','CC14','CC15','CCA8','CCA9']
    mlb2 = MultiLabelBinarizer(classes = ccs_possible_values)
    mlb2.fit([])
    transformed2 = mlb2.transform(df['tls.ccs'])
    tls_ccs_mlb = pd.DataFrame(transformed2,columns=mlb2.classes_)
    tls_ccs_mlb_renamed = tls_ccs_mlb.add_suffix('_ccs')

    # tls.cext
    df['tls.cext'] = df['tls.cext'].apply(normalize_to_array)
    cext_possible_values = ['0000','0005','0010','0012','0015','0017','0023','0033','3374',
                            '4469','000A','000B','000D','001B','002B','002D','FE0D','FF01']
    mlb2 = MultiLabelBinarizer(classes = cext_possible_values)
    mlb2.fit([])
    transformed2 = mlb2.transform(df['tls.cext'])
    tls_cext_mlb = pd.DataFrame(transformed2,columns=mlb2.classes_)
    tls_cext_mlb_renamed = tls_cext_mlb.add_suffix('_cext')

    #  tls.csg
    df['tls.csg'] = df['tls.csg'].apply(normalize_to_array)
    csg_possible_values = ['0201','0202','0203','0301','0303','0401','0403','0501','0503',
                           '0601','0603','0804','0805','0806','0302','0402','0502','0602']
    mlb2 = MultiLabelBinarizer(classes = csg_possible_values)
    mlb2.fit([])
    transformed2 = mlb2.transform(df['tls.csg'])
    tls_csg_mlb = pd.DataFrame(transformed2,columns=mlb2.classes_)
    tls_csg_mlb_renamed = tls_csg_mlb.add_suffix('_csg')

    # tls.alpn
    df['tls.alpn'] = df['tls.alpn'].apply(normalize_to_array)
    alpn_possible_values = ['h2','http/1.1','']
    mlb2 = MultiLabelBinarizer(classes = alpn_possible_values)
    mlb2.fit([])
    transformed2 = mlb2.transform(df['tls.alpn'])
    tls_alpn_mlb = pd.DataFrame(transformed2,columns=mlb2.classes_)
    tls_alpn_mlb_renamed = tls_alpn_mlb.add_suffix('_alpn')
    
    # tls.csv
    df['tls.csv'] = df['tls.csv'].apply(normalize_to_array)
    csv_possible_values = ['0303','0304','']
    mlb2 = MultiLabelBinarizer(classes = csv_possible_values)
    mlb2.fit([])
    transformed2 = mlb2.transform(df['tls.csv'])
    tls_csv_mlb = pd.DataFrame(transformed2,columns=mlb2.classes_)
    tls_csv_mlb_renamed = tls_csv_mlb.add_suffix('_csv')
    
    # tls.ssv
    df['tls.ssv'] = df['tls.ssv'].apply(normalize_to_array)
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
