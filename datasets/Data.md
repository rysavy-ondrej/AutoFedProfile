# Network Flow Dataset – Flattened Schema Reference

Each record corresponds to a **single network connection** and is stored as a **flat JSON object / table row**, suitable for Parquet storage, large-scale analytics, and machine‑learning pipelines.

## 1. Record Structure Overview

Each record is a flat object composed of the following logical groups:

- Connection and transport statistics  
- Temporal information  
- TLS fingerprints and handshake metadata  
- System and malware annotations  
- Compact TLS record sequence  

All fields are optional unless stated otherwise. Missing values are represented as `null`.

## 2. Connection and Transport Statistics

| Field | Type | Description |
|------|------|-------------|
| `bs` | integer | Bytes sent from source to destination. |
| `ps` | integer | Packets sent from source to destination. |
| `br` | integer | Bytes received from destination to source. |
| `pr` | integer | Packets received from destination to source. |
| `sp` | integer | Source transport port (TCP/UDP). |
| `dp` | integer | Destination transport port (TCP/UDP). |
| `sa` | string | Source IP address. |
| `da` | string | Destination IP address. |

## 3. Temporal Fields

| Field | Type | Description |
|------|------|-------------|
| `ts` | float | Timestamp (UNIX epoch) of the first packet in the connection. |
| `td` | float | Total connection duration in seconds. |

## 4. TLS Handshake and Fingerprints

| Field | Type | Description |
|------|------|-------------|
| `tls.ja3` | string | JA3 fingerprint of the TLS client handshake. |
| `tls.ja4` | string | JA4 fingerprint of the TLS client handshake. |
| `tls.ja3s` | string | JA3S fingerprint of the TLS server handshake. |
| `tls.ja4s` | string | JA4S fingerprint of the TLS server handshake. |
| `tls.sext` | array | ServerHello extensions. |
| `tls.csg` | array | Client-supported signature algorithms. |
| `tls.ccs` | array | Client-offered cipher suites. |
| `tls.cext` | array | ClientHello extensions. |
| `tls.ssv` | array | Server-supported protocol versions. |
| `tls.csv` | array | Client-supported protocol versions. |
| `tls.scs` | string | Server-selected cipher suite. |
| `tls.alpn` | array | ALPN protocol values. |
| `tls.sni` | string | Server Name Indication hostname. |

## 5. TLS Record Sequence

| Field | Type | Description |
|------|------|-------------|
| `tls.rec` | array[int] | Ordered sequence of signed TLS record lengths. |

Positive values denote source → destination, negative values denote destination → source.

## 6. Metadata

| Field | Type | Description |
|------|------|-------------|
| `meta.sample.id` | string | Unique sample identifier. |
| `meta.malware.family` | string \| null | Malware family label (scalar). |
| `meta.system.os` | string | Operating system. |
| `meta.system.service` | string \| null | Application or service. |
| `meta.application.name` | string \| null | Application name. |


