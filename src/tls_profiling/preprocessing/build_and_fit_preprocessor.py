from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
def build_and_fit_preprocessor(df):
    """
    Fit and return a preprocessing pipeline for TLS/flow features.

    The pipeline applies:
    - Min-max scaling to selected TLS numeric columns (`tls_columns_names`)
    - Min-max scaling to selected flow numeric columns
    - Fixed-category one-hot encoding for selected TLS categorical columns
    - Passthrough for remaining columns

    Parameters
    ----------
    df:
        Input training dataframe used to fit all transformers.

    Returns
    -------
    Pipeline
        A fitted scikit-learn pipeline with one step named `preprocessor`.
    """
    tls_rec_columns = [c for c in df.columns if c.startswith("tls.rec.")]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_tls', MinMaxScaler(), tls_rec_columns),
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
