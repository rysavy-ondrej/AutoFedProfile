
# Dataset Folder Organization

The dataset should follow a clear and reproducible directory structure to separate raw data, processed outputs, metadata, and tools.
This organization supports scalability, traceability, and compatibility with multi-level labeling and federated learning experiments.

## Recommended Directory Structure

```
dataset/
├── README.md
├── schema/
│   ├── label_schema.json            # JSON schema definition for validation
│   ├── feature_schema.json          # Feature field definitions (flow metadata)
│   └── enums.yaml                   # Controlled vocabularies (OS, applications, malware)
│
├── metadata/
│   ├── dataset_manifest.yaml        # Summary of dataset versions, samples, and splits
│   ├── sources_list.yaml            # Mapping of data_origin values (e.g., Tria.ge, CTU-13)
│   └── label_maps/                  # Label normalization and alias files
│
├── raw/
│   ├── benign/
│   │   ├── enterprise_lab/          # PCAPs or extracted JSONs from local lab
│   │   ├── home_network/
│   │   └── testbed/
│   └── malware/
│       ├── sandbox/                 # Captures from Tria.ge or similar environments
│       ├── public_datasets/         # Imported CTU-13, CIC-IDS2017, Stratosphere-IPS, etc.
│       └── other_sources/
│
├── processed/
│   ├── flows_json/                  # Final labeled JSON records (per-connection)
│   │   ├── benign/
│   │   └── malware/
│   ├── features_parquet/            # Vectorized features for ML/AE models
│   └── statistics/                  # Flow counts, class distributions, histograms
│
├── splits/
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── federated_nodes/             # FL-specific data partitions
│
├── tools/
│   ├── extractor/                   # Scripts for feature extraction from PCAPs
│   ├── labeler/                     # Annotation pipeline following schema definitions
│   ├── validator/                   # Schema validation and consistency checks
│   └── converters/                  # PCAP → JSON → Parquet conversion utilities
│
└── docs/
    ├── dataset_description.md       # Human-readable dataset overview
    ├── taxonomy_table.md            # Level definitions and examples
    ├── changelog.md                 # Version history
    └── license.txt
```

## Folder Descriptions

| Folder |	Purpose	| Typical Contents |
| --- | --- | --- |
| schema/	| Defines and validates the dataset structure.	| JSON schema, enumerations, controlled vocabularies. |
| metadata/ |	Maintains provenance and version tracking.|	Manifest files, data source registry, label mappings. |
| raw/ | 	Stores original or minimally processed data.	| PCAPs or extracted flow-level JSON files. |
| processed/	|  Contains cleaned and labeled JSON samples ready for analysis.	| Final labeled connections, Parquet features, statistics. |
| splits/	 | Defines training, validation, and federated subsets.	 | Train/test partitions and node-specific FL splits. |
| tools/	| Includes scripts for data extraction, labeling, and validation.	| Feature extractors, validators, converters. |
| docs/ |	Documentation and supporting materials.	| Dataset description, taxonomy table, changelog, license. |


## Federated Learning Extension

For federated learning experiments, data can be organized into node-specific partitions:`

```
splits/federated_nodes/
├── node_1_enterprise/
│   └── data.json
├── node_2_iot/
│   └── data.json
├── node_3_sandbox/
│   └── data.json
└── manifest.yaml
```
Each node folder contains:
* Local dataset subset (data.json)
* Node-level metadata (e.g., device types, environment)
* Log or manifest for reproducibility of FL experiments


## Best Practices
* Keep raw data immutable — transformations should produce new outputs in processed/.
* Use consistent naming conventions (e.g., YYYYMMDD_<source>_<type>.json).
* Version-control all schemas and scripts alongside the dataset.
* Include dataset_manifest.yaml to document dataset version, structure, and statistics.
* Generate regular summary reports (in processed/statistics/) for monitoring dataset health.
