# Towards Federated Autoencoder Profiling of Encrypted Network Communication

This repository accompanies the paper “Towards Federated Autoencoder Profiling of Encrypted Network Communication” (NOMS 2025).
It presents preliminary experiments using autoencoders for profiling encrypted network traffic based on Transport Layer Security (TLS) flows and explores the use of federated learning (FL) for collaborative and privacy-preserving model refinement.

The study demonstrates how autoencoders can capture device-specific communication profiles, distinguish application- and user-level behavior, and identify malicious traffic.
Federated learning is applied to enable distributed model sharing without exposing raw data, paving the way toward scalable, multi-level profiling of encrypted communication.

## Experiments

All experiments are implemented as interactive Python notebooks:
* [Autoencoder Cross-Validation](notebooks/noms.autoencoder_crossval.ipynb) -- Demonstrates local TLS flow profiling and reconstruction error–based evaluation.

## Created Datasets

| Dataset | Connections |	Description |
| ------- | ----------- | ------------- |
| [BUT-WAPP2024](datasets/BUT-WAPP2024/) | 14,962	| Communication of selected Windows applications captured in sandboxed environments. |
| [BUT-SOHO2025](datasets/BUT-SOHO2025/)  |	653,644	 | TLS connections collected from multiple real devices on a SOHO network for device profiling. |
| [BUT-TMF2024](datasets/BUT-TMF2024/)  |	16,742	| Network traces from malware samples analyzed in the Tria.ge sandbox, containing both malicious and benign system traffic. |

## Method Overview
* Feature extraction: Based on extended TLS flow metadata (protocol versions, cipher suites, extensions, record size sequences).
* Local model: Convolutional autoencoder combining sequential and scalar/categorical features.
* Federated setup: Implemented using the Flower framework with a cross-silo architecture and FedAvg aggregation.
* Evaluation: Conducted on device profiling and malware detection tasks, including federated learning scenarios.


## Dataset Format

The datasets were created from the source PCAP files using the [Enjoy tool](https://github.com/rysavy-ondrej/shark-tools).
Each document contains JSON records (NDJSON) for TLS connections. The following JSON fields are presented:

| Field      | Description                                                                  |
|------------|------------------------------------------------------------------------------|
| pt         | protocol type (e.g., 6 for TCP)                                              |
| sa         | source address (client)                                                      |
| sp         | source port (client)                                                         |
| da         | destination address (server)                                                 |
| dp         | destination port (server)                                                    |
| ps         | packets from client to server                                                |
| pr         | packets from server to client                                                |
| bs         | octets (bytes) from client to server                                         |
| br         | octets (bytes) from server to client                                         |
| ts         | timestamp of the first packet (connection start)                             |
| td         | connection duration (last packet timestamp minus ts)                         |
| tls.cver   | TLS client version (from Client Hello), e.g., "0x0303"                       |
| tls.ccs    | TLS client cipher suites as an array (from Client Hello)                     |
| tls.cext   | TLS client extensions as an array                                            |
| tls.csg    | TLS client supported groups as an array                                      |
| tls.csv    | TLS client supported versions                                                |
| tls.alpn   | TLS ALPN protocols as an array (e.g., ["h2", "http/1.1"])                    |
| tls.sni    | TLS Server Name Indication (SNI) from Client Hello                           |
| tls.sver   | TLS server version (from Server Hello), e.g., "0x0303"                       |
| tls.scs    | TLS server cipher suite (from Server Hello), e.g., "0xc030"                  |
| tls.sext   | TLS server extensions as an array                                            |
| tls.ssv    | TLS server supported versions (for TLS 1.3)                                  |
| tls.ja3    | JA3 fingerprint for the TLS client                                           |
| tls.ja3s   | JA3S fingerprint for the TLS server                                          |
| tls.ja4    | JA4 fingerprint for the TLS client (alternative fingerprint)                 |
| tls.ja4s   | JA4S fingerprint for the TLS server (alternative fingerprint)                |
| tls.rec    | array of individual TLS record lengths observed in the communication         |
| sample     | sample field (for testing), value can be "nil" if not provided               |

The TLS record array contains TLS record sizes. A negative number represents communication from client to server, while a positive number represents communication in the opposite direction.

Example:

```
{
    "pt": 6,
    "sa": "172.17.147.231",
    "sp": 49687,
    "da": "3.211.225.17",
    "dp": 443,
    "ps": 3,
    "pr": 4,
    "bs": 587,
    "br": 3418,
    "ts": 1721272491.362,
    "td": 0.701,
    "tls.cver": "0x0303",
    "tls.ccs": [
        "C02C",
        "C02B",
        "C030",
        "C02F",
        "C024",
        "C023",
        "C028",
        "C027",
        "C00A",
        "C009",
        "C014",
        "C013",
        "009D",
        "009C",
        "003D",
        "003C",
        "0035",
        "002F",
        "000A"
    ],
    "tls.cext": [
        "0000",
        "0005",
        "000A",
        "000B",
        "000D",
        "0023",
        "0017",
        "FF01"
    ],
    "tls.csg": [
        "0804",
        "0805",
        "0806",
        "0401",
        "0501",
        "0201",
        "0403",
        "0503",
        "0203",
        "0202",
        "0601",
        "0603"
    ],
    "tls.csv": [],
    "tls.alpn": [],
    "tls.sni": "www.biglybt.com",
    "tls.sver": "0x0303",
    "tls.scs": "0xc02f",
    "tls.sext": [
        "0000",
        "000B",
        "FF01",
        "0023",
        "0017"
    ],
    "tls.ssv": [],
    "tls.ja3": "a0e9f5d64349fb13191bc781f81f42e1",
    "tls.ja3s": "2ab44dd8c27bdce434a961463587356a",
    "tls.ja4": "t12d190800_d83cc789557e_7af1ed941c26",
    "tls.ja4s": "t120500_c02f_7c777c72cf5b",
    "tls.rec": [
        -177,
        99,
        4953,
        333,
        4,
        -70,
        -1,
        -40,
        115,
        1,
        40,
        -112,
        537
    ],
    "sample": "BiglySoftware.BiglyBT"
}
```