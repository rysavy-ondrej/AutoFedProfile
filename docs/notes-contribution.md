### Contribution

Sources:

* Malware data -- large number of samples, variety --> direct AE training
* Application data -- smaller number of samples, repeating patterns --> few shot learing

Ingerdients:

* Device-level AE models
* Multi-level profiling
* Federated learning
* Extended TLS metadata

Convincing evaluation would:

* Show AE >> PCA
* Show AE >> Isolation Forest
* Show AE + contextual enrichment >> JA3 baseline
* Show Federated AE ≈ Centralized AE

### Evaluation&comparison

| Category   | Method           | Purpose                        |
| ---------- | ---------------- | ------------------------------ |
| Linear     | PCA              | Linear reconstruction baseline |
| Tree-based | Isolation Forest | Nonlinear partitioning         |
| Density    | GMM              | Multi-modal modeling           |
| Boundary   | OC-SVM           | Decision boundary              |
| Deep       | VAE              | Probabilistic deep baseline    |
| Domain     | JA3 frequency    | TLS fingerprint baseline       |




### Evaluation Metrics 

Malware/applications:

* AUROC
* AUPRC (important for imbalance)
* FPR@95%TPR
* TPR@1%FPR
* Per-family detection rate

For host profiling:
* Confusion matrix across hosts
* Cross-host generalization error