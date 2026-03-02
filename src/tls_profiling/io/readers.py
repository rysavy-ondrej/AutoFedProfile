import pyarrow.dataset as ds
import pyarrow as pa

from .schema import get_tls_schema 

def open_parquet_dataset(path:str, schema: pa.Schema = None) -> ds.Dataset:
    if schema is None:
        schema=get_tls_schema()
    dataset = ds.dataset(path, format="parquet", schema=schema)
    return dataset

def print_dataset_summary(dataset:ds.Dataset) -> None:
    print("=== DATASET SUMMARY ===")
    print(f"Format: {dataset.format}")
    print(f"Columns: {len(dataset.schema)}")
    print(f"Column names: {dataset.schema.names}")
    fragments = list(dataset.get_fragments())
    print(f"Files: {len(fragments)}")        
    total_rows = dataset.count_rows()
    print(f"Total rows: {total_rows:,}")