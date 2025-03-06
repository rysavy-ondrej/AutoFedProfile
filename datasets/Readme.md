# Datasets

This folder contains several datasets for network communication analysis. The datasets are divided into two main categories: **Normal Datasets** and **Malicious Datasets**.

## Normal Datasets

Normal datasets are used to establish the baseline profile of regular, benign network communication. These datasets should not contain any malicious or attack-related traffic.

| **Name**            | **Description**                                                                              |
| ------------------- | -------------------------------------------------------------------------------------------- |
| `desktop.tls`       | A collection of communications from various Windows applications.                          |
| `iscx.tls`          | Various communications from applications, including background noise.                      |
| `mobile.tls`        | A collection of communications from selected mobile applications.                            |
| `windows.tls`       | A comprehensive collection of communications from a wide range of Windows applications.        |
| `cic-aa.normal.tls` | Benign Android communication (1,500 apps) from CICAAGM dataset.                          |

## Malicious Datasets

Malicious datasets contain both benign and attack communications. They are used to analyze, detect, and study malicious network activity.

| **Name**                | **Description**                                                                              |
| ----------------------- | -------------------------------------------------------------------------------------------- |
| `cic-aa.adware.tls`     | Android Adware communication from CICAAGM dataset.                                           |
| `cis-aa.malware.tls`    | Android general malware communication from CICAAGM dataset.                                  |


## Format

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