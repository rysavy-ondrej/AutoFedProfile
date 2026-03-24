# Baselines

The experiments in this folder establish classical baselines for profiling TLS
network communication and for detecting traffic that deviates from the learned
profile. The baselines are compared with the autoencoder-based approach used in
the rest of the project.

The input data are derived mainly from the `winapps` and `malware` datasets. In
the intended evaluation setting, models are trained on one reference category
(typically benign `winapp` traffic) and then tested on other categories to
measure how well they separate in-distribution communication from anomalous or
out-of-distribution traffic.

## Experiments

The datasets are split into three disjoint parts:

- train
- validation
- test

Across the datasets, the traffic may be grouped into the following labels:

- `winapp`
- `malware`
- `unknown`
- `system`

The goal is to learn a profile of the selected category and then test whether
the resulting model can distinguish that profile from other types of
communication. In practice, this is closer to anomaly detection or
out-of-distribution detection than to standard multi-class classification,
because the model is usually fitted only on the reference behavior and not on a
balanced set of all classes.

## Feature Sets

The following categories of input data are used:

- `FLOW`: basic bidirectional flow statistics: `bs`, `ps`, `br`, `pr`, `td`
- `CTLS`: client TLS metadata: `cver`, `ccs`, `cext`, `csg`, `ALPN`, `csv`
- `STLS`: server TLS metadata: `sver`, `scs`, `sext`, `ssv`
- `REC`: ordered sequence of signed TLS record lengths (first 20 records)

Combinations used in the ablation study:

- `FLOW` (5)
- `FLOW+REC` (25)
- `FLOW+CTLS` (90)
- `FLOW+STLS` (52)
- `FLOW+CTLS+STLS` (137)
- `FLOW+CTLS+REC` (110)
- `FLOW+STLS+REC` (72)
- `Full` (157)

## Methods

The following baseline methods are relevant for the profiling task:

### Isolation Forest

Isolation Forest is a tree-based unsupervised anomaly detector. It isolates
rare or unusual samples by recursively partitioning the feature space; samples
that can be isolated with fewer splits receive higher anomaly scores. It is a
strong baseline for this task because it does not assume any class labels,
handles mixed numeric feature spaces reasonably well, and can model nonlinear
structure better than purely linear methods.

### Gaussian Mixture Model (GMM)

GMM is a density-based probabilistic model that approximates normal traffic as a
mixture of several Gaussian components. A flow is considered anomalous when it
has low likelihood under the fitted benign distribution. This is useful when
normal communication is multi-modal, for example when several applications or
traffic patterns share the same training set. Its main limitation is that it
assumes a parametric density and may be sensitive to feature scaling and
high-dimensional sparse representations.

### JA3/JA4 Fingerprinting

JA3 and JA4 are rule-based TLS fingerprinting approaches derived from the Client
Hello and related handshake properties. For this project they serve as a domain
baseline rather than a learned anomaly detector: the basic idea is to compare
whether a TLS flow uses a known fingerprint seen during training or a new,
previously unseen one. This baseline is attractive because it is interpretable
and directly tied to TLS behavior, but it is limited because many different
flows may share the same fingerprint and benign software updates can change the
fingerprint without representing malicious behavior.

### PCA Model

PCA provides a linear reconstruction baseline. It learns the principal
subspace of the training data and measures anomaly by reconstruction error or by
distance from the learned subspace. In this setting, PCA answers whether simple
linear structure is already sufficient to capture the profile of benign TLS
communication. It is easy to interpret and fast to train, but it cannot model
complex nonlinear relationships in the same way as tree-based or neural models.

### Random Forest

Random Forest is not a native anomaly detector; it is primarily a supervised
classifier. Therefore, it is only meaningful here if labeled anomalous traffic
is also provided during training. In that case it can be used as a supervised
reference or approximate upper bound, but it should not be treated as directly
comparable to one-class or unsupervised profiling methods. For the intended
task, it is secondary to anomaly-detection baselines such as Isolation Forest,
GMM, PCA, or fingerprint frequency methods.

## Metrics

The most important evaluation principle is that anomaly scores should be judged
by how well they rank abnormal traffic above normal traffic and by how useful
the resulting operating point is in practice.

### AUC-ROC

The area under the ROC curve measures how well a method ranks positive samples
above negative samples across all possible thresholds. It is useful because many
baselines produce only anomaly scores, not calibrated probabilities, and the
best decision threshold may vary across datasets. In this project, AUC-ROC is a
good threshold-independent summary of separability between the reference traffic
and out-of-distribution traffic.

### F1-score

F1-score summarizes the balance between precision and recall at a selected
threshold. It is suitable when the experiment ultimately requires a binary
decision, for example whether a connection should be flagged as anomalous or
not. However, F1-score should be interpreted carefully in one-class settings,
because it depends strongly on the chosen threshold and on the class balance in
the evaluation set. For that reason, F1 is best reported together with
threshold-independent metrics such as AUC-ROC.

## Interpretation

For this repository, the baseline results should be interpreted primarily as
anomaly-detection results:

- training uses only the reference or normal traffic profile
- validation is used to calibrate an anomaly threshold
- testing measures how strongly other traffic types deviate from the learned
  profile

If a method is trained with both normal and anomalous labels, then the task
changes from anomaly detection to supervised classification, and the results
should be reported separately from the one-class baselines.
