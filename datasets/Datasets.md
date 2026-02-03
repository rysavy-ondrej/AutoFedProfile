


CESNET Idle OS Traffic

https://zenodo.org/records/15004766


## Complete dataset

The complete dataset contains data for the following time periods: A checkmark (✓) indicates that the data is ready, while a gear symbol (⚙) indicates that the data is available, but processing is in progress. A (↻) indicates that the data needs to be captured.

| Dataset  | Year | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|----------|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| malware  | 2024 | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   |
| winapps  | 2024 |     |     |     |     |     |     | ✓   | ✓   | ✓   | ✓   | ✓   |     |
| soho     | 2024 |     |     |     |     |     |     |     |     |     |     |     |     |
| malware  | 2025 | ✓   | ✓   | ✓   |     |     |     |     |     | ✓   | ⚙   | ⚙   | ⚙   |
| winapps  | 2025 |     | ⚙   |     |     |     |     |     |     |     |     |     |     |
| soho     | 2025 |     |     |     |     |     |     |     |     | ✓   | ⚙   | ⚙   | ⚙   |
| malware  | 2026 |     | ↻   | ↻   | ↻   |     |     |     |     |     |     |     |     |
| winapps  | 2026 |     | ↻   | ↻   | ↻   |     |     |     |     |     |     |     |     |
| soho     | 2026 |     | ↻   | ↻   | ↻   |     |     |     |     |     |     |     |     |

TODO:

```
SET E:\0000_COMMON\Datasets\2025_CUS_PCAP_TriageMalware AS malware

FROM malware WINDOW 2024[07-11] -> JSON -> PARQUET INTO malware.parquet
FROM malware WINDOW 2025[02-03] -> JSON -> PARQUET INTO malware.parquet
FROM malware WINDOW 2025[09-11] -> JSON -> PARQUET INTO malware.parquet

SET E:\0000_COMMON\Datasets\2025_CUS_MIXED_HOME-LAN\raw AS soho
FROM soho WINDOW 2025[09-12] -> JSON -> PARQUET INTO soho.parquet

SET E:\0000_COMMON\Datasets\2024_CUS_PCAP_WindowsApplications as winapps
FROM winapps WINDOW 2025[02-03] -> JSON -> PARQUET INTO winapps.parquet
```