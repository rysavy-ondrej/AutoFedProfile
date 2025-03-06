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
