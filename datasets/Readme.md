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

## Collections

### 1. `soho`
This dataset contains TLS flows collected from an operational small-office/home-office (SOHO) network.
It provides realistic, heterogeneous encrypted traffic suitable for research on device profiling, OS fingerprinting, and behavioral modeling.

**Files:**

* soho.parquet/ – directory containing TLS connection data.

* soho.meta.parquet – file-level metadata describing devices and inferred system properties.

**Characteristics**

* Traffic originates from desktops, laptops, smartphones, tablets, IoT devices, and network infrastructure.
* Raw PCAPs do not contain explicit labels; therefore no direct ground-truth OS is available.
* Operating system annotations are inferred using SNI-based OS fingerprinting, including mappings to OS-unique domains (high-confidence indicators).
* Metadata is available at the file level and can be propagated to all flows belonging to the same source IP within the day.

**Metadata includes:**
- `host.ip`
- `system.os` (inferred)
- `system.os_detection.{method,pattern,confidence,timestamp}`

The metadata can be used to label the communication of each IP address throughout the day.
Because OS detection is based on heuristics and may be ambiguous for some hosts, additional disambiguation strategies (e.g., majority voting, confidence weighting, or temporal smoothing) are recommended to determine the most probable OS identity.

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
- `sample.{src,hash}`
- `system.os`
- `system.service` (if the connection matches known system services)
- `malware.{family,score}` for malicious samples
- Differentiation between malware-initiated and system-generated connections.


### 3. `winapps`
TLS connections captured during execution of **Windows applications** in a Windows Sandbox environment.

**Characteristics:**
- Each run focuses on a specific application (e.g., iTunes, browsers, updaters).
- Ground-truth for the **application generating the TLS connection** is known.
- OS is fixed and known (Windows 10/11 sandbox).
- Useful for supervised application-level TLS classification and fingerprinting.

**Metadata includes:**
- `sample.{src,hash}`
- `system.os`
- `application.name` (process associated with the TLS connection)


# Summary

This dataset provides three complementary sources of annotated TLS traffic:

| Dataset       | Source Environment | Annotation Level | Use Cases |
|---------------|--------------------|------------------|-----------|
| `homelan` | Real network       | OS (inferred via SNI) | Passive OS fingerprinting, LAN modeling |
| `triage`  | Malware sandbox    | OS, malware family, score | Malware TLS profiling, behavioral analysis |
| `winapps` | Windows Sandbox    | Application, OS | Application TLS classification, software fingerprinting |
