# Autoencoder Models

This folder contains the TensorFlow/Keras autoencoder implementations used for
TLS profiling experiments in this repository. The code is organized so notebook
experiments can either use a simple end-to-end helper or directly compose model
building, training, and calibration steps.

## Implemented Builders

### `build_conv_dense_autoencoder`

This is the default hybrid model currently used by
`tls_profiling.autoencoder.run_autoencoder_experiment`.

Architecture:

- the first `conv_input_size` features are treated as an ordered sequence
- that sequence branch is processed by `Conv1D -> MaxPooling1D -> Flatten`
- the remaining features are processed by a dense branch
- both branches are concatenated into a shared latent bottleneck
- a dense decoder reconstructs the full input vector

This model is a good fit when the first block of features represents ordered
TLS record statistics and the rest are tabular flow or metadata features.

Typical usage:

```python
from tls_profiling.autoencoder import build_conv_dense_autoencoder
from tls_profiling.autoencoder import train_autoencoder_model
from tls_profiling.autoencoder import compute_reconstruction_error

models = build_conv_dense_autoencoder(
    input_dim=X_train.shape[1],
    encoding_dim=16,
    conv_input_size=20,
)

history = train_autoencoder_model(models, X_train, X_val, loss="mse")
errors = compute_reconstruction_error(models.autoencoder, X_test, metric="mse")
```

### `build_combined_autoencoder`

This is a more configurable hybrid architecture for experiments that need finer
control over model capacity.

Key characteristics:

- multiple convolutional blocks on the sequence-like prefix
- multiple dense blocks on the metadata/tabular suffix
- optional batch or layer normalization
- optional dropout and L2 regularization
- configurable pooling for the sequence branch
- split decoder heads for sequence and metadata reconstruction

Use this builder when you want to tune architecture depth, regularization, or
decoder structure beyond the default baseline model.

Typical usage:

```python
from tls_profiling.autoencoder.models import build_combined_autoencoder

models = build_combined_autoencoder(
    input_dim=X_train.shape[1],
    encoding_dim=32,
    conv_input_size=20,
    conv_filters=(32, 64),
    dense_branch_units=(64, 64),
    norm_type="batch",
    dropout_rate=0.1,
)
```

### `build_dense_autoencoder`

This model is a pure dense autoencoder for fully tabular data.

It is appropriate when:

- feature order does not carry sequential meaning
- you want a simpler baseline than the hybrid convolution+dense models
- all inputs should be treated uniformly as standard tabular features

Typical usage:

```python
from tls_profiling.autoencoder.models import build_dense_autoencoder

models = build_dense_autoencoder(
    input_dim=X_train.shape[1],
    encoding_dim=16,
    encoder_units=(128, 64),
    decoder_units=(64, 128),
    dropout_rate=0.1,
)
```

## Recommended Usage Patterns

### 1. Simple experiment path

Use `run_autoencoder_experiment` when you want the repository's default
workflow:

- build the default hybrid autoencoder
- train with early stopping
- compute reconstruction errors on a held-out set
- return models, history, and summary statistics

This is the easiest option for notebooks and baseline-style experiments.

### 2. Manual training path

Use the lower-level pieces when you need more control:

1. build a model with one of the builders in `models.py`
2. train it with `train_autoencoder_model`
3. score samples with `compute_reconstruction_error`
4. choose thresholds or compare distributions in notebook code

This pattern is better when you need custom thresholding, multiple validation
splits, ablation studies, or model debugging.

### 3. Encoding / decoder inspection

Each builder returns an `AEModels` object with:

- `autoencoder`: full reconstruction model
- `encoder`: input-to-latent projection
- `decoder`: latent-to-reconstruction model

This makes it possible to:

- inspect latent embeddings
- reconstruct samples from latent vectors
- debug reconstruction failures
- reuse the encoder in downstream tasks

## Input Assumptions

The autoencoders in this folder expect dense numeric matrices, typically
produced after preprocessing in `tls_profiling.preprocessing`.

In the current TLS setup:

- the first `conv_input_size` values usually correspond to ordered TLS record
  features
- the remaining values correspond to flow statistics and encoded TLS metadata

If your feature layout differs, prefer `build_dense_autoencoder` or adjust the
hybrid builder parameters accordingly.

## Practical Notes

- `train_autoencoder_model` compiles and trains `models.autoencoder`; the
  builders themselves mainly define architecture
- reconstruction error is currently computed with `mse` or `mae`
- threshold selection is intentionally left to experiment code because anomaly
  operating points depend on the dataset and task
- `build_conv_dense_autoencoder` is the most stable starting point for this
  repository's malware notebooks
