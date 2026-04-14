# Common Evaluation Mistakes in IDS and Network Anomaly Detection

In IDS and network anomaly-detection research, several systematic evaluation errors repeatedly appear in papers—even in well-known venues. These mistakes often produce inflated performance metrics that do not reflect real deployment conditions.

Below are three of the most common problems.

---

## 1. Random Train/Test Split Instead of Temporal Split

### Problem

Many IDS papers randomly split flows into training and testing sets:

```python
train_test_split(dataset, test_size=0.2, shuffle=True)
````

This creates **temporal leakage**.

In network traffic datasets (NetFlow, TLS flows, CICIDS), attacks typically appear in time bursts:

```
t1  t2  t3  t4  t5  t6
scan scan scan scan scan scan
```

If the split is random:

| Flow | Attack | Dataset |
| ---- | ------ | ------- |
| 1    | scan   | train   |
| 2    | scan   | train   |
| 3    | scan   | test    |
| 4    | scan   | test    |

The model has already seen almost identical attack samples in training.

### Result

* Artificially high performance
* Poor real-world generalization

This is particularly problematic for:

* Flow features
* TLS fingerprints
* JA3/JA4 fingerprints
* Statistical packet features

because attack flows are often highly repetitive.

### Correct Approach

Use **temporal splits**.

| Dataset    | Time Range |
| ---------- | ---------- |
| Train      | Day 1–3    |
| Validation | Day 4      |
| Test       | Day 5      |

This simulates real IDS deployment:

* Model trained on historical traffic
* Model evaluated on future traffic

> This change alone often reduces reported accuracy significantly.

---

## 2. Mixing Flows From the Same Attack Instance

### Problem

Many attacks generate thousands of nearly identical flows.

Example: port scanning

```
attacker -> target ports 1..1000
```

Dataset representation:

| Flow | Label  |
| ---- | ------ |
| scan | attack |
| scan | attack |
| scan | attack |
| ...  | ...    |

If these flows are randomly split:

| Flow  | Dataset |
| ----- | ------- |
| scan1 | train   |
| scan2 | train   |
| scan3 | test    |
| scan4 | test    |

The classifier effectively **memorizes the attack pattern**.

### Result

Inflated metrics such as:

* ROC-AUC
* Accuracy
* F1-score

But no ability to detect **new attacks**.

### Correct Approach

Split by **attack instance**, not by flows.

| Attack Instance | Dataset |
| --------------- | ------- |
| Scan #1         | train   |
| Scan #2         | train   |
| Scan #3         | test    |
| Scan #4         | test    |

Or evaluate **event-level detection**:

| Attack Event | Detected |
| ------------ | -------- |
| scan         | yes      |
| botnet       | yes      |
| exfiltration | no       |

> IDS goal: detect attack campaigns, not individual flows.

---

## 3. Evaluating Only ROC-AUC on Extremely Imbalanced Data

### Problem

Many IDS papers report only:

```
ROC-AUC = 0.99
```

With extreme imbalance:

| Class  | Ratio  |
| ------ | ------ |
| Normal | 99.99% |
| Attack | 0.01%  |

ROC can hide massive false alarm rates.

Example:

| Metric          | Value     |
| --------------- | --------- |
| True positives  | 900       |
| False positives | 50,000    |
| True negatives  | 9,950,000 |

[
FPR = \frac{50000}{10,000,000} = 0.005
]

ROC interpretation:

* FPR = 0.5% → appears excellent

Operational reality:

* **50,000 alerts** → SOC overload

### Correct Approach

Include **operational metrics**.

| Metric               | Meaning                     |
| -------------------- | --------------------------- |
| PR-AUC               | Imbalance-aware performance |
| TPR@FPR=0.1%         | Strict false alarm control  |
| Alerts per day       | Operational workload        |
| Precision            | False alarm rate            |
| Per-attack detection | Robustness                  |

Example:

| Metric       | Value |
| ------------ | ----- |
| ROC-AUC      | 0.97  |
| PR-AUC       | 0.42  |
| TPR@FPR=0.1% | 0.81  |
| Alerts/day   | 120   |

> Reflects real deployment constraints.

---

## Summary (General IDS Issues)

| Mistake                       | Consequence                       |
| ----------------------------- | --------------------------------- |
| Random train/test split       | Temporal leakage                  |
| Mixing flows from same attack | Memorization instead of detection |
| Using only ROC-AUC            | Hidden false alarm problem        |

---

# Additional Pitfalls in NetFlow / TLS Anomaly Detection

In NetFlow/IPFIX and TLS-metadata anomaly detection, several **flow-specific methodological issues** invalidate results or inflate performance.

---

## 1. Feature Leakage Through Identifiers or Dataset Artifacts

### Problem

Flow datasets often contain fields that implicitly encode labels.

| Field           | Issue                     |
| --------------- | ------------------------- |
| Source IP       | Fixed attack hosts        |
| Destination IP  | Fixed targets             |
| Port numbers    | Tool-specific patterns    |
| Timestamp       | Attack windows            |
| TLS fingerprint | Static malware signatures |

Example:

| Feature   | Benign          | Malware        |
| --------- | --------------- | -------------- |
| Source IP | enterprise host | sandbox IP     |
| JA3       | Chrome          | malware client |
| Server IP | CDN             | C2 server      |

Model learns:

```
IP address → attack
```

instead of behavior.

### Symptoms

| Metric   | Value |
| -------- | ----- |
| Accuracy | 99.9% |
| ROC-AUC  | 0.999 |

Fails completely on new data.

### Correct Approach

* Remove identifiers (IP, MAC, domain)
* Group-based splitting (by host)
* Cross-network evaluation

---

## 2. Ignoring Flow Aggregation Effects

### Problem

NetFlow/IPFIX records depend on exporter configuration.

| Parameter        | Meaning               |
| ---------------- | --------------------- |
| Active timeout   | Max flow duration     |
| Inactive timeout | Idle termination      |
| Sampling         | Packet sampling ratio |
| Aggregation      | Flow granularity      |

Example:

| Exporter | Timeout | Flows |
| -------- | ------- | ----- |
| A        | 30 s    | 10    |
| B        | 300 s   | 1     |

Same traffic → different representations.

### Result

Models fail across networks.

### Correct Approach

* Document exporter configuration
* Normalize features (e.g., bytes/sec)
* Cross-exporter evaluation

---

## 3. Ignoring TLS Session Context

### Problem

TLS behavior is **session-based**, not flow-based.

Example session:

```
ClientHello → ServerHello → Certificate → AppData
```

Flow-level modeling misses:

* Handshake structure
* Session reuse
* Protocol negotiation

### Additional Issue: TLS Fingerprint Overfitting

| Fingerprint  | Label   |
| ------------ | ------- |
| Chrome JA3   | benign  |
| Trickbot JA3 | malware |

Model learns fingerprint, not behavior.

Malware can easily change fingerprints.

### Correct Approach

Use **session-level modeling**.

| Method              | Purpose                    |
| ------------------- | -------------------------- |
| Sequence models     | Handshake patterns         |
| Graph models        | Host interaction structure |
| Session aggregation | Merge related flows        |

---

## Summary (NetFlow/TLS Issues)

| Problem                         | Consequence                    |
| ------------------------------- | ------------------------------ |
| Feature leakage                 | Model learns dataset artifacts |
| Ignoring exporter configuration | Poor transferability           |
| Ignoring TLS session structure  | Incorrect protocol modeling    |

---

## Practical Guidelines

For robust NetFlow/TLS anomaly detection:

| Design Principle        | Reason                     |
| ----------------------- | -------------------------- |
| Remove identifiers      | Prevent leakage            |
| Temporal splits         | Avoid repetition bias      |
| Exporter-aware features | Ensure transferability     |
| Session-level modeling  | Capture protocol semantics |
| Operational metrics     | Realistic evaluation       |

```
