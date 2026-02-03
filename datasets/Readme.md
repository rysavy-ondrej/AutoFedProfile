# The Dataset

The purpose of this dataset is to provide **multi-source, multi-level annotations of TLS connections**, enabling research on encrypted traffic analysis, OS fingerprinting, application profiling, and malware behavior characterization.  
All packet captures are preprocessed into normalized **connection-level JSON records**, enriched with structured metadata describing sample origin, system attributes, and (when applicable) malware classification.

The data format is defined in [Data.md](Data.md).

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

## Collections

### 1. `soho`
This dataset contains TLS flows collected from an operational small-office/home-office (SOHO) network.
It provides realistic, heterogeneous encrypted traffic suitable for research on device profiling, OS fingerprinting, and behavioral modeling.

**Files:**

* soho.parquet/ – directory containing TLS connection data.

**Characteristics**

* Traffic originates from desktops, laptops, smartphones, tablets, IoT devices, and network infrastructure.
* Raw PCAPs do not contain explicit labels; therefore no direct ground-truth OS is available.

**Metadata includes:**
- `meta.sample.id` -- source batch identifier, eg., `20250926T08_SOHO`.

### 2. `malware`
TLS connections from **Triage sandbox executions** of malware and benign samples.

**Characteristics:**
- Each sample is executed in a controlled environment (Windows virtual machine).
- The sandbox provides **ground-truth labels**:
  - malware family
  - severity/score
  - OS version of the sandbox
  - system-level connections triggered indirectly by malware runtime
- Both malware-driven and OS-driven TLS connections are included.

**Metadata includes:**
- `meta.sample.id` is a reference to the malware sample in the Triage format, e.g., `250901-1aw26afr2z_behavioral1`
- `meta.system.os` denotes the sandbox used OS, for instance, `windows10-2004-x64` or `windows11-21h2-x64`
- `meta.system.service` (if the connection matches known system services)
- `meta.malware.family` for malicious samples

### 3. `winapps`
TLS connections captured during execution of **Windows applications** in a Windows Sandbox environment.

**Characteristics:**
- Each run focuses on a specific application (e.g., iTunes, browsers, updaters).
- Ground-truth for the **application generating the TLS connection** is known.
- OS is fixed and known (Windows 10/11 sandbox).
- Useful for supervised application-level TLS classification and fingerprinting.

**Metadata includes:**
- `meta.sample.id` identifies the source collection of the samples, eg., `20240716_Adamant.Messenger`
- `meta.system.os` specifies the Sandboxed OS, eg., `windows10-2004-x64`
- `meta.application.name` application associated with the TLS connection, eg., `Adamant.Messenger`

# Summary

This dataset provides three complementary sources of annotated TLS traffic:

| Dataset       | Source Environment | Size | Number of connections | Interval |
|---------------|--------------------|------|------------|-----------|
| `soho`        | Real network       | 104 files / 12.8MB | 108,036  | 2024-07-16 -- 2024-11-22 |
| `malware`     | Malware sandbox    | 916 files / 63.4 MB | 828,171 | 2025-09-10 -- 2025-09-30 |
| `winapps`     | Windows Sandbox    | 5892 files / 63.2 MB | 29,526 | 2025-09-26 -- 2025-10-01 |



