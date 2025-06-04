# Home Network Connection Dataset

This dataset captures network connections within a home network environment, organized into 1-minute time windows. Each file contains connection logs that have been segmented for analysis and processing.

* Time Windowing: Network connections are recorded in 1-minute intervals. If a connection spans multiple minutes, it is split across multiple windows and appears as several connection records.
* File Generation: Input files are generated every 10 minutes, each containing logs from 10 consecutive 1-minute windows.
* File Organization: Connection records are grouped by the source IP address, representing individual hosts within the network. Each host has a separate subfolder containing its own connection logs.

## Host Information

| Host | Address | Active Interval | Total connections | 
|------| -------| ------------------- | ------ | 
| MacMini | 192.168.1.197 | 5/6/2025 2:48:11 PM +00:00 - 5/27/2025 3:02:29 PM +00:00 |  |
| Google ChromeCast | 192.168.1.185 | 5/6/2025 2:51:22 PM +00:00 - 5/27/2025 3:02:32 PM +00:00 |  |
| Sonos Wifi Speaker  | 192.168.1.190 | 5/6/2025 2:48:14 PM +00:00 - 5/27/2025 3:02:38 PM +00:00  |  |
| iRobot | 192.168.1.188 | 5/6/2025 2:48:19 PM +00:00 - 5/27/2025 3:02:13 PM +00:00 |  |
| Samsung GalaxyTab | 192.168.1.198 | 5/6/2025 2:48:21 PM +00:00 - 5/27/2025 3:00:14 PM +00:00 |  |


The data objects stands for TLS connections. They have the following format:

| Field | Meaning |
|-------|--------|
| `"pt": 6` | Protocol type → **TCP** |
| `"sa": "172.17.145.66"` | Source IP address |
| `"sp": 49759` | Source port |
| `"da": "78.47.205.206"` | Destination IP address (likely a server) |
| `"dp": 443` | Destination port (**HTTPS**) |
| `"ps": 4` | Number of packets sent by the source |
| `"pr": 6` | Number of packets received by the source |
| `"bs": 1973` | Bytes sent by the source |
| `"br": 5093` | Bytes received by the source |
| `"ts": 1721122982.375` | Timestamp (UNIX epoch time) |
| `"td": 30.074` | Duration of the connection in seconds |
| `"tls.cver": "0x0303"` | Client TLS version → **TLS 1.2** |
| `"tls.sver": "0x0303"` | Server TLS version → **TLS 1.2** |
| `"tls.scs": "0x1301"` | Server-selected cipher suite → TLS_AES_128_GCM_SHA256 (TLS 1.3) |
| `"tls.ccs"` | Client cipher suites (hex identifiers) |
| `"tls.cext"` | Client TLS extensions |
| `"tls.csg"` | Supported groups (elliptic curves / finite fields) |
| `"tls.csv"` | Supported versions: `0303` = TLS 1.2, `0304` = TLS 1.3 |
| `"tls.alpn"` | Application Layer Protocols → **HTTP/2**, **HTTP/1.1** |
| `"tls.sni"` | Server Name Indication → `liskservice4.adamant.im` |
| `"tls.sext"` | Server TLS extensions |
| `"tls.ssv"` | Server supported versions (e.g., `0304` = TLS 1.3) |
| `"tls.ja3"` | JA3 fingerprint (client) |
| `"tls.ja3s"` | JA3S fingerprint (server) |
| `"tls.ja4"` | JA4 fingerprint (client behavior) |
| `"tls.ja4s"` | JA4S fingerprint (server behavior) |

In addition to TLS connection features, the lenght of first records are provided in `tls.rec` field:

```json
"tls.rec": [-538, 122, 1, 42, 2606, 281, 53, -1, -53, -570, 74, 74, 733, -570, 733]
