# Network Flow JSON Reference Manual

This document describes the JSON structure used for representing parsed network connections (flows) in the dataset.
Each JSON object corresponds to a single connection and may contain protocol-specific sections depending on the captured traffic.

This manual defines:

* Field semantics
* Data types
* Optional/conditional sections
* Protocol-specific structures (TCP/UDP/DNS/HTTP/TLS)

## 1. Top-Level Structure

Each record is a JSON object with the following top-level fields:

| Field | Type | Required | Description |
|-------|--------|----------|-------------|
| `id`  | string | yes | Unique identifier of the connection (e.g., `tcp.144`). |
| `ts`  | float  | yes | Timestamp (UNIX epoch) of the first packet in the connection. |
| `td`  | float  | yes | Total connection duration in seconds. |
| `ip`  | object | yes | IP-layer summary. |
| `tcp` | object/null | conditional | Present only for TCP flows. |
| `udp` | object/null | conditional | Present only for UDP flows. |
| `dns` | object/null | optional | Present for DNS traffic (UDP/53 or TCP/53). |
| `http` | object/null | optional | Present where HTTP metadata is extracted. |
| `tls` | object/null | optional | Present for TLS connections. |
| `meta` | object | yes | Metadata linking to the sample and system annotation. |

## 2. IP Section

The ip object provides general transport-agnostic connection information.

| Field | Type | Description |
|--------|--------|-------------|
| `proto` | string | Protocol name (`TCP` or `UDP`). |
| `src` | string | Source IP address. |
| `dst` | string | Destination IP address. |
| `bsent` | integer | Total bytes sent from source to destination. |
| `brecv` | integer | Total bytes received. |
| `psent` | integer | Number of packets sent. |
| `precv` | integer | Number of packets received. |

Example:

```json
"ip": {
    "proto": "TCP",
    "src": "192.168.1.197",
    "dst": "17.253.57.215",
    "bsent": 2535,
    "brecv": 10228,
    "psent": 20,
    "precv": 15
}
```

## 3. TCP Section

The tcp object describes all segments captured during a TCP connection.

| Field | Type | Description |
|-------|------|-------------|
| `srcport` | integer | Source TCP port. |
| `dstport` | integer | Destination TCP port. |
| `segs` | array | List of TCP segments in time order. |

Each segment is represented by the following structure:

| Field | Type | Description |
|--------|--------|-------------|
| `ts` | float | Timestamp of the segment. |
| `dir` | integer | Direction: `1` = src→dst, `-1` = dst→src. |
| `len` | integer | TCP payload length. |
| `flags.str` | string | Flag names in human-readable form. |
| `flags.val` | integer | Numeric representation of TCP flags. |

Example:

```json
"tcp": {
  "srcport": 55565,
  "dstport": 80,
  "segs": [
    {"ts": 1758877659.4376, "dir": -1, "len": 0, "flags": {"str":"uaprSf","val":2}},
    ...
  ]
}
```

## 4. UDP Section

The udp object describes UDP datagrams.

| Field | Type | Description |
|-------|------|-------------|
| `srcport` | integer | Source UDP port. |
| `dstport` | integer | Destination UDP port. |
| `dgms` | array | List of captured datagrams. |

Each datagram is represented by the following structure:

| Field | Type | Description |
|--------|--------|-------------|
| `ts` | float | Timestamp. |
| `dir` | integer | Direction (`1` or `-1`). |
| `len` | integer | Datagram length. |

Example:

```json
"udp": {
  "srcport": 59401,
  "dstport": 53,
  "dgms": [
    {"ts": 1758877477.4276, "dir": -1, "len": 49},
    {"ts": 1758877477.4447, "dir": 1, "len": 98}
  ]
}
```

## 5. DNS Section

The dns object appears when DNS parsing is successful.

| Field | Type | Description |
|--------|--------|-------------|
| `queries` | array | DNS queries. |
| `rcode` | integer | Response code (0 = NOERROR). |
| `responses` | array | DNS answers and related records. |

A query is represented as follows:

| Field | Type | Description |
|--------|--------|-------------|
| `qn` | string | Queried domain (FQDN). |
| `qt` | string | Query type (A, AAAA, CNAME, TXT…). |

A response is represented as follows:

| Field | Type | Description |
|--------|--------|-------------|
| `rr` | string | Record section (e.g., "answer"). |
| `qn` | string | Queried name or owner name. |
| `rt` | string | Record type. |
| `ttl` | integer | TTL in seconds. |
| `rv` | string | Resource value (IP, CNAME, etc.). |

Example:

```json
"dns": {
    "queries": [{"qn": "spclient.wg.spotify.com", "qt": "A"}],
    "rcode": 0,
    "responses": [
        {"rr": "answer", "qn": "spclient.wg.spotify.com", "rt": "CNAME", "ttl": 240, "rv": "edge-web.dual-gslb.spotify.com"},
        {"rr": "answer", "qn": "edge-web.dual-gslb.spotify.com", "rt": "A", "ttl": 120, "rv": "35.186.224.24"}
    ]
}
```

## 6. HTTP/HTTP2 Section

The http/http2 object summarizes HTTP activity on the connection (e.g., proxying, CONNECT method, plaintext HTTP on port 80).

| Field | Type | Description |
|--------|--------|-------------|
| `req` | array | Extracted HTTP requests. |
| `res` | array | Extracted HTTP responses. |

Requests:

| Field | Type | Description |
|--------|--------|-------------|
| `method` | string | HTTP method (`GET`, `POST`, `CONNECT`, …). |
| `uri` | string | URI or CONNECT target. |
| `rnum` | integer | Internal request ID used to link to response. |

Responses:

| Field | Type | Description |
|--------|--------|-------------|
| `code` | string | HTTP response code. |
| `rnum` | integer | Corresponding request number. |

Example:

```json
"http": {
    "req": [
        {"method": "CONNECT", "uri": "proxy-safebrowsing.googleapis.com:443", "rnum": 16183}
    ],
    "res": [
        {"code": "200", "rnum": 16183}
    ]
}
```

## 7. TLS Section (Extended)

The tls block captures parsed TLS handshake information, including fingerprints.


| Field | Type | Description |
|--------|--------|-------------|
| `recs` | array | Parsed TLS records with version, content type, length, direction. |
| `cver` | string | ClientHello advertised version. |
| `cciphers` | array | Cipher suites offered by client. |
| `cexts` | array | ClientHello extensions. |
| `sni` | string | Server Name Indication value. |
| `alpn` | array | ALPN protocols. |
| `csigs` | array | Supported signature algorithms. |
| `csvers` | array | Client-supported protocol versions. |
| `ja3` | string | JA3 fingerprint of the client handshake. |
| `ja4` | string | JA4 fingerprint (extended). |
| `sver` | string | TLS version selected by server. |
| `scipher` | string | Server-selected cipher suite. |
| `sexts` | array | ServerHello extensions. |
| `ssvers` | array | Server-supported versions (TLS 1.3 only). |
| `ja3s` | string | JA3S fingerprint of the server handshake. |
| `ja4s` | string | JA4S fingerprint. |

TLS Records:

| Field | Type | Description |
|--------|--------|-------------|
| `ver` | string | TLS record protocol version. |
| `ct` | integer | Content type (22=handshake, 23=app data, etc.). |
| `len` | integer | Length of TLS record. |
| `dir` | integer | Direction (`1` or `-1`). |

Example:

```json
"tls": {
    "recs": [{ "ver": "0301", "ct":22, "len":512, "dir":-1 }, ...],
    "cver": "0303",
    "cciphers": ["1301","1302", ... ],
    "cexts": ["0010","000b", ... ],
    "sni": "proxy-safebrowsing.googleapis.com",
    "alpn": ["h2", "http/1.1"],
    "csigs": ["0403","0804", ... ],
    "csvers": ["0304","0303","0302"],
    "ja3": "773906b0efdefa24a7f2b8eb6985bf37",
    "ja4": "t13d2014h2_a09f3c656075_e42f34c56612",
    "sver": "0303",
    "scipher": "1301",
    "sexts": ["0033","002B"],
    "ssvers": ["0304"],
    "ja3s": "eb1d94daa7e0344597e756a1fb6e7054",
    "ja4s": "t130200_1301_234ea6891581"
}
```

## 8. Metadata Section

The meta block provides contextual information about dataset provenance (see [Meta.md](Meta.md)).


## Example

```json
{
    "id": "tcp.18",
    "ts": 1757390490.3281,
    "td": 0.23727202415466,
    "ip": {
        "proto": "TCP",
        "src": "10.127.1.153",
        "dst": "52.111.236.22",
        "bsent": 1581,
        "brecv": 20400,
        "psent": 16,
        "precv": 21
    },
    "tcp": {
        "srcport": 49925,
        "dstport": 443,
        "segs": [
            {
                "ts": 1757390490.3281,
                "dir": -1,
                "len": 0,
                "flags": {
                    "str": "uaprSf",
                    "val": 2
                }
            },
            {
                "ts": 1757390490.3835,
                "dir": 1,
                "len": 0,
                "flags": {
                    "str": "uAprSf",
                    "val": 18
                }
            },
            {
                "ts": 1757390490.3838,
                "dir": -1,
                "len": 0,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.3844,
                "dir": -1,
                "len": 215,
                "flags": {
                    "str": "uAPrsf",
                    "val": 24
                }
            },
            {
                "ts": 1757390490.4429,
                "dir": 1,
                "len": 1330,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.4429,
                "dir": 1,
                "len": 1330,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.4429,
                "dir": 1,
                "len": 1330,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.4429,
                "dir": 1,
                "len": 1330,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.443,
                "dir": 1,
                "len": 783,
                "flags": {
                    "str": "uAPrsf",
                    "val": 24
                }
            },
            {
                "ts": 1757390490.4433,
                "dir": -1,
                "len": 0,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.4435,
                "dir": -1,
                "len": 0,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.4484,
                "dir": -1,
                "len": 158,
                "flags": {
                    "str": "uAPrsf",
                    "val": 24
                }
            },
            {
                "ts": 1757390490.5042,
                "dir": 1,
                "len": 51,
                "flags": {
                    "str": "uAPrsf",
                    "val": 24
                }
            },
            {
                "ts": 1757390490.5043,
                "dir": 1,
                "len": 69,
                "flags": {
                    "str": "uAPrsf",
                    "val": 24
                }
            },
            {
                "ts": 1757390490.5046,
                "dir": -1,
                "len": 0,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.5057,
                "dir": -1,
                "len": 87,
                "flags": {
                    "str": "uAPrsf",
                    "val": 24
                }
            },
            {
                "ts": 1757390490.5058,
                "dir": -1,
                "len": 38,
                "flags": {
                    "str": "uAPrsf",
                    "val": 24
                }
            },
            {
                "ts": 1757390490.5058,
                "dir": -1,
                "len": 431,
                "flags": {
                    "str": "uAPrsf",
                    "val": 24
                }
            },
            {
                "ts": 1757390490.562,
                "dir": 1,
                "len": 0,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.562,
                "dir": 1,
                "len": 38,
                "flags": {
                    "str": "uAPrsf",
                    "val": 24
                }
            },
            {
                "ts": 1757390490.5645,
                "dir": 1,
                "len": 1330,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.5646,
                "dir": 1,
                "len": 1330,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.5647,
                "dir": 1,
                "len": 1330,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.5647,
                "dir": 1,
                "len": 1330,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.5647,
                "dir": 1,
                "len": 1330,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.5647,
                "dir": 1,
                "len": 1330,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.5647,
                "dir": 1,
                "len": 1330,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.5647,
                "dir": 1,
                "len": 1330,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.5647,
                "dir": 1,
                "len": 1330,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.5647,
                "dir": 1,
                "len": 1279,
                "flags": {
                    "str": "uAPrsf",
                    "val": 24
                }
            },
            {
                "ts": 1757390490.5647,
                "dir": 1,
                "len": 38,
                "flags": {
                    "str": "uAPrsf",
                    "val": 24
                }
            },
            {
                "ts": 1757390490.5648,
                "dir": -1,
                "len": 0,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.5649,
                "dir": -1,
                "len": 0,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.565,
                "dir": -1,
                "len": 0,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.5652,
                "dir": -1,
                "len": 0,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.5654,
                "dir": -1,
                "len": 0,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            },
            {
                "ts": 1757390490.5654,
                "dir": -1,
                "len": 0,
                "flags": {
                    "str": "uAprsf",
                    "val": 16
                }
            }
        ]
    },
    "tls": {
        "recs": [
            {
                "ver": "0303",
                "ct": 22,
                "len": 210,
                "dir": -1
            },
            {
                "ver": "0303",
                "ct": 22,
                "len": 6098,
                "dir": 1
            },
            {
                "ver": "0303",
                "ct": 22,
                "len": 102,
                "dir": -1
            },
            {
                "ver": "0303",
                "ct": 20,
                "len": 1,
                "dir": -1
            },
            {
                "ver": "0303",
                "ct": 22,
                "len": 40,
                "dir": -1
            },
            {
                "ver": "0303",
                "ct": 20,
                "len": 1,
                "dir": 1
            },
            {
                "ver": "0303",
                "ct": 22,
                "len": 40,
                "dir": 1
            },
            {
                "ver": "0303",
                "ct": 23,
                "len": 64,
                "dir": 1
            },
            {
                "ver": "0303",
                "ct": 23,
                "len": 82,
                "dir": -1
            },
            {
                "ver": "0303",
                "ct": 23,
                "len": 33,
                "dir": -1
            },
            {
                "ver": "0303",
                "ct": 23,
                "len": 426,
                "dir": -1
            },
            {
                "ver": "0303",
                "ct": 23,
                "len": 33,
                "dir": 1
            },
            {
                "ver": "0303",
                "ct": 23,
                "len": 13244,
                "dir": 1
            },
            {
                "ver": "0303",
                "ct": 23,
                "len": 33,
                "dir": 1
            }
        ],
        "cver": "0303",
        "cciphers": [
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
        "cexts": [
            "0000",
            "0005",
            "000A",
            "000B",
            "000D",
            "0023",
            "0010",
            "0017",
            "FF01"
        ],
        "sni": "nexusrules.officeapps.live.com",
        "alpn": [
            "h2",
            "http/1.1"
        ],
        "csigs": [
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
        "csvers": [],
        "ja3": "28a2c9bd18a11de089ef85a160da29e4",
        "ja4": "t12d1909h2_d83cc789557e_7af1ed941c26",
        "sver": "0303",
        "scipher": "C030",
        "sexts": [
            "0005",
            "0010",
            "0017",
            "FF01"
        ],
        "ssvers": [],
        "ja3s": "67bfe5d15ae567fb35fd7837f0116eec",
        "ja4s": "t1204h2_c030_09f674154ab3"
    },
    "http2": {
        "req": [
            {
                "method": "GET",
                "uri": "https://nexusrules.officeapps.live.com/nexus/rules?Application=officeclicktorun.exe&Version=16.0.12527.20470&ClientId=%7bE85B5876-DDC3-4056-93A3-A451C959DABA%7d&OSEnvironment=10&MsoAppId=37&AudienceName=Production&AudienceGroup=Production&AppVersion=16.0.12527.20470&",
                "agent": "Microsoft Office/16.0 (Windows NT 10.0;  16.0.12527; Pro)",
                "rnum": 1023
            }
        ],
        "res": [
            {
                "code": "200",
                "server": "Microsoft-IIS/10.0",
                "content_type": "application/vnd.ms-nexus-rules-v15+xml; charset=utf-8",
                "rnum": 1023
            },
            {
                "code": "200",
                "server": "Microsoft-IIS/10.0",
                "content_type": "application/vnd.ms-nexus-rules-v15+xml; charset=utf-8",
                "rnum": 1023
            }
        ]
    },
    "meta": {
        "sample": {
            "id": "250909-ekfx2agl71_behavioral1"
        },
        "system": {
            "os": "windows10-2004-x64",
            "service": "microsoft_office"
        }
    }
}
```