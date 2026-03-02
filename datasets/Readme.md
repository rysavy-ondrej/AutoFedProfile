# The Dataset

The purpose of this dataset is to provide **multi-source, multi-level annotations of TLS connections**, enabling research on encrypted traffic analysis, OS fingerprinting, application profiling, and malware behavior characterization.  
All packet captures are preprocessed into normalized **connection-level JSON records**, enriched with structured metadata describing sample origin, system attributes, and (when applicable) malware classification.

The data format is defined in [Data.md](Data.md).

The dataset is described in [Datasets.md](Datasets.md).

## Usage

All datasets are stored in the Apache Parquet format, a columnar storage format optimized for efficient analytics and fast data loading. You can load an entire dataset (i.e., a directory containing multiple Parquet files) using the following code:

```py
import pyarrow.dataset as ds

dataset = ds.dataset(parquet_data_folder, format="parquet")
df = dataset.to_table().to_pandas()
```

This snippet:

* loads all Parquet files in the specified directory as a unified dataset,
* converts the Arrow table into a flattened representation (useful if nested fields are present),
* and returns the result as a Pandas DataFrame.

The data can be filtered using [Parquet Compute Expression](https://arrow.apache.org/docs/python/compute.html#filtering-by-expressions), for instance:

```py
import pyarrow.dataset as ds
import pyarrow.compute as pc

dataset = ds.dataset(parquet_data_folder, format="parquet")

filt = (
    pc.is_null(pc.field("meta.system.service")) &           # no service, that is pure malware connection
    (pc.field("meta.malware.family") == "asyncrat") &       # select single family (asyncrat)
    (pc.field("td") > 10) &                                 # connection duration > 10s
    pc.starts_with(pc.field("meta.sample.id"), "250901")    # use meta.sample.id to get only a single day 2025-09-01
)

df = dataset.to_table(
    filter=filt
).to_pandas()
df
```



