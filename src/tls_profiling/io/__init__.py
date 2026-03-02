import pyarrow.dataset as pa
from .readers import open_parquet_dataset 

def open_tls_parquet_dataset(path:str) -> pa.Dataset:
    return open_parquet_dataset(path)
