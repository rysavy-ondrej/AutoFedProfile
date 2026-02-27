
# Dataset-Level Summary (High-Level Overview)

For each dataset (D1–D5), report:

### A. Size & structure

* Number of TLS connections
* Number of unique hosts (if available)
* Number of unique IPs
* Number of unique SNI / eTLD+1
* Number of unique JA3 / JA4 fingerprints
* Number of unique malware families (D2, D3)
* Time span (start–end date)
* Average connections per day

This gives users:

* Scale
* Diversity
* Temporal coverage


# Temporal Statistics

Very important for drift understanding.

For each dataset:

* Connections per day (mean, min, max)
* Daily variance
* Monthly distribution
* For malware: samples per family over time

Plots to include:

* Daily connection counts
* Score distributions over time (later, after modeling)

This reveals:

* Collection bias
* Bursty behavior
* Time imbalance


# TLS Protocol Characteristics

These are extremely informative.

### Distribution of:

* TLS version (1.0–1.3)
* ALPN values (http/1.1, h2, etc.)
* Cipher suites (top 10 + tail size)
* SNI presence rate
* Resumed vs full handshake ratio

For malware vs benign:

* Compare proportions side-by-side.

This gives:

* Behavioral fingerprinting insight
* Early understanding of separability
* Drift indicators


# Flow-Level Statistics

For each dataset, report summary statistics:

* Duration (mean, median, p95, p99)
* Bytes up/down
* Packet count up/down
* Byte ratio up/down
* IAT mean / std

Prefer:

* Median
* IQR
* p95/p99
  Not only mean (TLS traffic is heavy-tailed).

Also:

* % of short connections (<1s)
* % of large transfers (>1MB)

# TLS Record Sequence Characteristics (Unique Feature of the datasets)

This is critical and often ignored.

For each dataset:

* Average sequence length (how many records until close)
* Distribution of first record size
* Direction of first record (client/server)
* Mean signed record value per position (1–20)
* Variance per position

You can provide:

* Heatmap: position vs mean signed size
* Boxplots per position

This gives:

* Structural signature insight
* Family-level differences
* Drift patterns

---

# Diversity Metrics

These give insight into heterogeneity.

For each dataset:

* Shannon entropy of:

  * SNI
  * JA3
  * Cipher
* % of connections from top 5 SNIs
* % of connections from top 5 JA3s

Why?

* Malware often has low diversity per family
* Real networks have high diversity


# Cross-Dataset Comparisons (Very Valuable)

Provide comparison tables:

| Feature          | malware | winapps | soho | 
| ---------------- | -- | -- | -- | 
| TLS 1.3 %        |    |    |    | 
| Mean duration    |    |    |    | 
| Median bytes     |    |    |    | 
| Unique JA3 count |    |    |    | 
| SNI presence %   |    |    |    | 

This immediately shows:

* Sandbox artifacts
* Malware homogeneity
* Real-network variability


# Class Imbalance Overview

For malware datasets:

* Connections per family
* Families with < 50 connections
* Long-tail distribution

Report:

* Gini coefficient or simply % from top 10 families

This affects:

* Per-family TPR interpretation
* Model fairness

# Outlier Audit

Identify:

* Top 0.1% longest duration connections
* Largest byte transfers
* Rare TLS versions
* Extremely long record sequences

This helps:

* Detect data corruption
* Identify edge cases
* Understand tails

# Domain-Shift Diagnostics (Before Modeling)

Since your work heavily relies on OOD:

Measure divergence between:

* D1 vs D4
* D1 vs D2
* D1 vs D5

Simple metrics:

* KS statistic (for numeric features)
* χ² distance (for categorical)
* Jensen–Shannon divergence (for distributions)

Even reporting:

* “Cipher distribution overlap = 72%”
  is already valuable.


# Minimal Strong EDA Set (If You Want a Lean Version)

If you want a compact but strong EDA section, include:

1. Dataset sizes & time spans
2. TLS version + ALPN distributions
3. Flow duration + bytes percentiles
4. First 5 TLS record position means
5. Top-10 SNI / JA3 distribution
6. Cross-dataset comparison table

