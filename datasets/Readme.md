# The Dataset

The purpose of this dataset is to provide **multi-source, multi-level annotations of TLS connections**, enabling research on encrypted traffic analysis, OS fingerprinting, application profiling, and malware behavior characterization.  
All packet captures are preprocessed into normalized **connection-level JSON records**, enriched with structured metadata describing sample origin, system attributes, and (when applicable) malware classification.

The data format is defined in [Data.md](Data.md).

## Usage

All datasets are stored in the Apache Parquet format, a columnar storage format optimized for efficient analytics and fast data loading. You can load an entire dataset (i.e., a directory containing multiple Parquet files) using the following code:

```py
import pyarrow.dataset as ds

dataset = ds.dataset(parquet_folder, format="parquet")
df = dataset.to_table().flatten().to_pandas()
```

This snippet:

* loads all Parquet files in the specified directory as a unified dataset,
* converts the Arrow table into a flattened representation (useful if nested fields are present),
* and returns the result as a Pandas DataFrame.

The data can be filtered using Parquet Compute, for instance:

```py
import pyarrow.dataset as ds
import pyarrow.compute as pc
# See more on pyarrow.compute at: 
dataset = ds.dataset("../malware.parquet", format="parquet")

filt = (
     
    pc.is_null(pc.field("meta.system.service")) &           # no service, that is pure malware connection
    (pc.field("meta.malware.family") == "asyncrat") &       # select single family (asyncrat)
    (pc.field("td") > 10) &                                 # connection druation > 10s
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

This dataset does not have any metadata.

### 2. `triage`
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
- `sample.id`
- `system.os`
- `system.service` (if the connection matches known system services)
- `malware.family` for malicious samples
- Differentiation between malware-initiated and system-generated connections.


### 3. `winapps`
TLS connections captured during execution of **Windows applications** in a Windows Sandbox environment.

**Characteristics:**
- Each run focuses on a specific application (e.g., iTunes, browsers, updaters).
- Ground-truth for the **application generating the TLS connection** is known.
- OS is fixed and known (Windows 10/11 sandbox).
- Useful for supervised application-level TLS classification and fingerprinting.

**Metadata includes:**
- `sample.id`
- `system.os`
- `application.name` (process associated with the TLS connection)


# Summary

This dataset provides three complementary sources of annotated TLS traffic:

| Dataset       | Source Environment | Annotation Level | Use Cases |
|---------------|--------------------|------------------|-----------|
| `homelan` | Real network       | OS (inferred via SNI) | Passive OS fingerprinting, LAN modeling |
| `triage`  | Malware sandbox    | OS, malware family, score | Malware TLS profiling, behavioral analysis |
| `winapps` | Windows Sandbox    | Application, OS | Application TLS classification, software fingerprinting |

