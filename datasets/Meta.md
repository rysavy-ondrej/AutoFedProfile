# Meta Information Specification

This document defines the structure and semantics of the **meta information** associated with exported network-communication records. Metadata is produced either **per connection** or **per source file/host**, depending on dataset origin. The purpose is to provide a consistent, machine-readable description of the **sample**, **system**, **application**, and **malware** context.

---

## 1. Triage-Sourced Dataset

Triage reports provide **OS**, **malware family**, **malware score**, and sometimes **service identifiers**. Because Triage also reports **benign system connections** triggered indirectly by malware (e.g., system library loads), such connections are annotated to distinguish **system-generated** from **malware-driven** activity.

### 1.1 Malware Samples (basic)

```json
{
  "meta": {
    "sample": { "src": "251120-tw93kssjej_behavioral1", "hash": "..." },
    "system": { "os": "windows10-2004-x64" },
    "malware": { "family": ["agenttesla"], "score": 10 }
  }
}
```

### 1.2 Malware Samples Using a Known Service

```json
{
  "meta": {
    "sample": { "src": "251120-tw93kssjej_behavioral1", "hash": "..." },
    "system": { "os": "windows10-2004-x64", "service": "google_core" },
    "malware": { "family": ["agenttesla"], "score": 10 }
  }
}
```

### 1.3 Non-Malware System Connections

```json
{
  "meta": {
    "sample": { "src": "251120-tw93kssjej_behavioral1", "hash": "..." },
    "system": { "os": "windows10-2004-x64" }
  }
}
```

### 1.4 Non-Malware System Connections to Known Services

```json
{
  "meta": {
    "sample": { "src": "251120-tw93kssjej_behavioral1", "hash": "..." },
    "system": { "os": "windows10-2004-x64", "service": "digicert_pki" }
  }
}
```

⸻

## 2. Windows Sandbox Dataset

Connections originate from controlled execution of specific Windows applications. Both the process/application name and the sandbox OS are known.

### 2.1 Application-Level Metadata

```json
{
  "meta": {
    "sample": { "src": "251120-tw93kssjej_behavioral1", "hash": "..." },
    "system": { "os": "windows10-2004-x64" },
    "application": { "name": "Apple.iTunes" }
  }
}
```

### 2.2 Host-Based Metadata

```json
{
  "meta": {
    "sample": { "src": "251120-e44fml3f2", "hash": "..." },
    "system": { "os": "windows10-2004-x64" }
  }
}
```

⸻

## 3. LAN-Captured Dataset

For LAN traffic captures, limited metadata is available. OS identification is inferred using SNI-based fingerprinting applied per source IP. Metadata is typically attached per host rather than per connection.

### 3.1 Host-Level OS Detection Example

```json
{
  "host": { "ip": "192.168.1.197" },
  "system": {
    "os": "macos",
    "os_detection": {
      "method": "sni_patterns",
      "ts": "20250926T0828",
      "confidence": 0.75,
      "pattern": "(^|\\.)mesu\\.apple\\.com$"
    }
  }
}
```



## 4. Metadata Output Specification

This section defines the required JSON schema elements used in the examples above.

### 4.1 sample Object

| Field | Type | Description |
|-------|------|-------------|
| `src` | string | Unique sample identifier (directory name, execution label, triage run, etc.) |
| `hash` | string | Cryptographic hash of the executed binary or PCAP file |


### 4.2 system Object

| Field | Type | Description |
|-------|------|-------------|
| `os` | string | Operating system name (e.g., `windows10-2004-x64`, `android`, `macos`) |
| `service` | string (optional) | Name of known system/cloud service (e.g., `digicert_pki`, `google_core`) |
| `os_detection` | object (optional) | OS inference produced from LAN SNI fingerprinting |

### 4.3 os_detection Object (LAN only)

| Field | Type | Description |
|-------|------|-------------|
| `method` | string | Detection method used (e.g., `sni_patterns`) |
| `ts` | string (format: `yyyyMMddTHHmm`) | Timestamp when OS was identified |
| `confidence` | number (0–1) | Confidence score of the OS classification |
| `sni` | string | Matched pattern (SNI string) |

### 4.4 malware Object

| Field | Type | Description |
|-------|------|-------------|
| `family` | array of strings | Malware family labels from sandbox (e.g., `"agenttesla"`) |
| `score` | number | Malware severity score assigned by Triage |

### 4.5 application Object (Windows Sandbox)

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Process or application name associated with the connection (e.g., `Apple.iTunes`) |

### 4.6 Host-Level Structure (LAN)

| Field | Type | Description |
|-------|------|-------------|
| `host.ip` | string | Observed source IP address |
| `system` | object | OS and OS detection details |
