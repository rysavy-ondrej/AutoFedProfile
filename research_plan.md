# Federated Autoencoder Profiling of Encrypted Communication Patterns

## Motivation

The motivation for integrating autoencoders with federated learning lies in the need to understand and characterize encrypted network communication beyond simple anomaly detection. Modern networks consist of diverse devices, operating systems, and users whose behavior patterns vary across applications and contexts. Autoencoders can learn compact, latent representations of normal communication at different semantic levels—ranging from individual devices to user groups or application classes—enabling fine-grained profiling and identification. Such multi-level modeling allows the system to distinguish, for example, between a legitimate user’s workstation, an IoT device, or malware operating within the same network, even when traffic content is encrypted. However, developing these models requires access to rich and distributed data, often subject to privacy and ownership constraints. Federated learning addresses this limitation by allowing local models to be collaboratively trained and refined without sharing raw traffic data. The combination of AE and FL thus provides a scalable, privacy-preserving foundation for cross-domain network profiling that supports diverse use cases including device type identification, OS fingerprinting, user or application recognition, and malware family characterization.

## Research Plan


| **Step** | **Description** |
|-----------|-----------------|
| **1. Dataset Collection and Definition** | Identify and capture relevant traffic domains: (i) benign (user devices, IoT, enterprise services), (ii) malicious (sandboxed malware families or public repositories such as CTU-13, Stratosphere IPS). Collect TLS-based traffic with extended flow metadata (TLS version, cipher suites, JA3/JA4, SNI, timing features). Annotate samples with device, OS, user, application, or malware family labels. |
| **2. Data Preprocessing and Normalization** | Extract flow-level features from PCAP using custom tool (Enjoy). Aggregate by session or host, normalize numeric fields, and encode categorical attributes. Segment data temporally to capture behavioral stability and balance class representation. |
| **3. Dataset Structuring for Hierarchical Profiling** | Organize data by profiling granularity: Level 1 – device/OS identification, Level 2 – user/application profiling, Level 3 – malware family characterization. Prepare subsets for each level and for multi-task learning experiments. |
| **4. Model Training and FL Integration** | Train local AE models on domain-specific data (e.g., per organization or device class). Aggregate models using FL to generalize across domains. Investigate hierarchical AE architectures combining latent representations from multiple levels. |
| **5. Evaluation and Validation** | Evaluate using reconstruction error, clustering accuracy, silhouette score, and F1 score. Test intra-domain, cross-domain, and mixed benign/malicious scenarios. Analyze robustness to data drift and unseen devices. Compare FL and centralized training for privacy–utility trade-offs. |
| **6. Dataset Release and Reproducibility** | Curate anonymized subsets (with sanitized metadata and SNI). Publish dataset documentation, labeling methods, and feature schema. Provide reproducible benchmarks for future AE + FL profiling research. |

### Expected Outcomes

The research will deliver a comprehensive set of artifacts supporting multi-level and cross-domain profiling:

- **Datasets:** Annotated and anonymized TLS-based network flow datasets covering devices, operating systems, users, applications, and malware families.  
- **Methods and Tools:** Reusable **Jupyter notebooks** for feature extraction, AE model training, and federated aggregation workflows.  
- **Hierarchical Profiling Models:** Trained AE and FL models representing device, user, and application profiles with explainable outputs.  
- **Evaluation Framework:** Scripts for automated benchmarking of reconstruction accuracy, clustering performance, and cross-domain generalization.  
- **Reproducibility Package:** Public release (where possible) including dataset schema, notebooks, and experiment configurations enabling replication and extension of results.
