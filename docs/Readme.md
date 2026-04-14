# Approach

The main motivation behind this project is that direct connection
classification is difficult to do well in practice. Building a suitable
labeled dataset for a standard multiclass classifier is a complex task: it
requires collecting representative traffic for many categories, maintaining
label quality over time, and covering the large variability of real TLS
communication.

Instead of treating the problem only as a multiclass classification task, this
repository focuses primarily on anomaly detection. In this setting, an anomaly
means that a sample does not belong to the selected connection category. For
example, a detector trained for `system` traffic should mark `malware`,
`application`, and unrelated traffic as anomalous with respect to the system
profile.

This approach is practical because training data for a single category are much
easier to obtain than a complete balanced multiclass dataset. Category-specific
traffic can be generated in a controlled way:

- `application` traffic can be collected by running the application in a
  sandbox or test environment
- `malware` traffic can be collected in an analytical or detonation
  environment
- `system` traffic can be collected from clean operating-system activity

The current approach considers the three basic categories:

- `system`
- `malware`
- `application`

These are intentionally high-level categories and can be refined further.
Examples include:

- for `system`: `linux`, `windows`, `macos`, or other operating-system
  profiles
- for `malware`: specific malware families
- for `application`: individual applications or application groups

In addition to these profiled categories, the approach also works with
`unknown` traffic. This category does not have its own dedicated detector.
Instead, `unknown` denotes traffic that is marked as anomalous by all other
category detectors. In other words, if a connection does not fit any learned
profile, it is treated as unknown.

This repository studies TLS connection categorization using the MUSA
(`malware`-`unknown`-`system`-`application`) view of the problem. It contains:

- baseline methods that illustrate how difficult it is to separate these
  traffic categories using classical models
- more sophisticated methods that aim at a state-of-the-art solution to the
  problem

The more advanced approaches rely on deep-learning models and related
representation-learning techniques for TLS traffic profiling and classification.
