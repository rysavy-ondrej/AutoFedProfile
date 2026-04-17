"""Model builders for TLS autoencoders."""

from typing import Literal, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    Lambda,
    LayerNormalization,
    MaxPooling1D,
)
from tensorflow.keras.models import Model

from .types import AEModels

# -----------------------------
# 1) Model build
# -----------------------------
def build_conv_dense_autoencoder(
    input_dim: int,
    encoding_dim: int,
    conv_input_size: int = 20,
    intermediate_dim: int = 64,
    activation: str = "relu",
    output_activation: str = "sigmoid",
) -> AEModels:
    """
    Build an autoencoder that:
      - applies Conv1D over the first `conv_input_size` features
      - applies Dense over the remaining features
      - concatenates both branches into an encoder
      - decodes back to `input_dim`
    Returns autoencoder + encoder + decoder models.
    """

    if conv_input_size <= 0 or conv_input_size > input_dim:
        raise ValueError(f"conv_input_size must be in [1, {input_dim}], got {conv_input_size}")

    # Input placeholder
    input_x = Input(shape=(input_dim,), name="ae_input")

    # Slice the first conv_input_size values for convolution and reshape to (steps, channels=1)
    def slice_first(x):
        return tf.reshape(x[:, :conv_input_size], (-1, conv_input_size, 1))

    sliced = Lambda(slice_first, output_shape=(conv_input_size, 1), name="slice_first")(input_x)

    # Convolutional processing
    conv = Conv1D(filters=32, kernel_size=3, activation=activation, name="conv1")(sliced)
    pool = MaxPooling1D(pool_size=2, name="pool1")(conv)
    flat = Flatten(name="flat_conv")(pool)

    # Remaining input
    def slice_remaining(x):
        return x[:, conv_input_size:]

    remaining = Lambda(
        slice_remaining,
        output_shape=(input_dim - conv_input_size,),
        name="slice_remaining",
    )(input_x)

    dense_branch = Dense(intermediate_dim, activation=activation, name="dense_branch")(remaining)

    # Combine
    combined = Concatenate(name="concat")([flat, dense_branch])

    # Encoder
    hidden = Dense(intermediate_dim, activation=activation, name="enc_hidden")(combined)
    encoded = Dense(encoding_dim, activation=activation, name="latent")(hidden)

    # Decoder
    hidden_dec = Dense(intermediate_dim, activation=activation, name="dec_hidden")(encoded)
    decoded = Dense(input_dim, activation=output_activation, name="recon")(hidden_dec)

    autoencoder = Model(input_x, decoded, name="autoencoder")
    encoder = Model(input_x, encoded, name="encoder")

    # Decoder model uses the last decoder layers from autoencoder
    encoded_input = Input(shape=(encoding_dim,), name="decoder_input")
    dec_hidden_layer = autoencoder.get_layer("dec_hidden")(encoded_input)
    dec_out_layer = autoencoder.get_layer("recon")(dec_hidden_layer)
    decoder = Model(encoded_input, dec_out_layer, name="decoder")

    return AEModels(autoencoder=autoencoder, encoder=encoder, decoder=decoder)

NormType = Optional[Literal["batch", "layer"]]
PoolingType = Literal["flatten", "global_max", "global_avg"]
LatentActivationType = Optional[str]


def _kernel_regularizer(l2_reg: float):
    """Return an L2 regularizer when requested, otherwise ``None``."""
    return regularizers.l2(l2_reg) if l2_reg > 0 else None


def _maybe_add_norm(x, norm_type: NormType, name: str):
    if norm_type is None:
        return x
    if norm_type == "batch":
        return BatchNormalization(name=name)(x)
    if norm_type == "layer":
        return LayerNormalization(name=name)(x)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def _dense_block(
    x,
    units: int,
    activation: str,
    norm_type: NormType = None,
    dropout_rate: float = 0.0,
    l2_reg: float = 0.0,
    name_prefix: str = "dense",
):
    x = Dense(
        units,
        activation=None,
        kernel_regularizer=_kernel_regularizer(l2_reg),
        name=f"{name_prefix}_dense",
    )(x)
    x = _maybe_add_norm(x, norm_type, name=f"{name_prefix}_norm")
    x = tf.keras.layers.Activation(activation, name=f"{name_prefix}_act")(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name=f"{name_prefix}_drop")(x)
    return x


def _conv_block(
    x,
    filters: int,
    kernel_size: int,
    activation: str,
    norm_type: NormType = None,
    dropout_rate: float = 0.0,
    l2_reg: float = 0.0,
    pool_size: Optional[int] = None,
    name_prefix: str = "conv",
):
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=None,
        kernel_regularizer=_kernel_regularizer(l2_reg),
        name=f"{name_prefix}_conv",
    )(x)
    x = _maybe_add_norm(x, norm_type, name=f"{name_prefix}_norm")
    x = tf.keras.layers.Activation(activation, name=f"{name_prefix}_act")(x)
    if pool_size is not None and pool_size > 1:
        x = MaxPooling1D(pool_size=pool_size, name=f"{name_prefix}_pool")(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name=f"{name_prefix}_drop")(x)
    return x


# -----------------------------
# Model build
# -----------------------------
def build_combined_autoencoder(
    input_dim: int,
    encoding_dim: int,
    conv_input_size: int = 20,
    conv_filters: Tuple[int, ...] = (32, 64),
    conv_kernel_sizes: Tuple[int, ...] = (3, 3),
    conv_pool_sizes: Tuple[Optional[int], ...] = (2, None),
    dense_branch_units: Tuple[int, ...] = (64, 64),
    shared_encoder_units: Tuple[int, ...] = (64,),
    seq_decoder_units: Tuple[int, ...] = (64,),
    meta_decoder_units: Tuple[int, ...] = (64,),
    activation: str = "relu",
    latent_activation: LatentActivationType = None,
    seq_output_activation: Optional[str] = "sigmoid",
    meta_output_activation: Optional[str] = "sigmoid",
    pooling: PoolingType = "global_max",
    norm_type: NormType = None,
    dropout_rate: float = 0.0,
    l2_reg: float = 0.0,
    sequence_loss_weight: float = 1.0,
    metadata_loss_weight: float = 1.0,
    compile_model: bool = False,
    optimizer: str | tf.keras.optimizers.Optimizer = "adam",
    sequence_loss: str | tf.keras.losses.Loss = "mse",
    metadata_loss: str | tf.keras.losses.Loss = "mse",
) -> AEModels:
    """
    Build a hybrid autoencoder with:
      - encoder branch 1: Conv1D-based processing of the first `conv_input_size` features
      - encoder branch 2: Dense processing of the remaining features
      - shared latent bottleneck
      - decoder branch 1: reconstructs the sequence-like part
      - decoder branch 2: reconstructs the remaining/tabular part

    Parameters
    ----------
    input_dim : int
        Total input feature dimension.

    encoding_dim : int
        Latent space dimension.

    conv_input_size : int
        Number of first features treated as ordered sequence.

    conv_filters, conv_kernel_sizes, conv_pool_sizes : tuple
        Configuration of convolutional blocks in sequence encoder branch.

    dense_branch_units : tuple
        Dense layers in metadata/tabular encoder branch.

    shared_encoder_units : tuple
        Dense layers after concatenation before the latent bottleneck.

    seq_decoder_units, meta_decoder_units : tuple
        Dense layers in decoder branches.

    activation : str
        Activation used in hidden layers.

    latent_activation : Optional[str]
        Activation in latent layer. Use None for linear latent layer.

    seq_output_activation, meta_output_activation : Optional[str]
        Output activations for the two decoder heads.

    pooling : {"flatten", "global_max", "global_avg"}
        How to summarize the sequence encoder branch before fusion.

    norm_type : {None, "batch", "layer"}
        Optional normalization after hidden layers.

    dropout_rate : float
        Dropout applied in blocks.

    l2_reg : float
        L2 kernel regularization strength.

    sequence_loss_weight, metadata_loss_weight : float
        Loss weights for both reconstructed parts.

    compile_model : bool
        If True, compile the autoencoder using two output heads.

    optimizer, sequence_loss, metadata_loss
        Compilation parameters.

    Returns
    -------
    AEModels
        Contains:
        - autoencoder: Model(input -> full reconstruction)
        - encoder: Model(input -> latent)
        - decoder: Model(latent -> full reconstruction)
    """

    if input_dim <= 1:
        raise ValueError(f"input_dim must be > 1, got {input_dim}")
    if encoding_dim <= 0:
        raise ValueError(f"encoding_dim must be > 0, got {encoding_dim}")
    if conv_input_size <= 0 or conv_input_size >= input_dim:
        raise ValueError(
            f"conv_input_size must be in [1, {input_dim - 1}], got {conv_input_size}"
        )

    if not (
        len(conv_filters) == len(conv_kernel_sizes) == len(conv_pool_sizes)
    ):
        raise ValueError(
            "conv_filters, conv_kernel_sizes, and conv_pool_sizes must have the same length"
        )

    meta_input_size = input_dim - conv_input_size

    # -----------------------------
    # Input and slicing
    # -----------------------------
    input_x = Input(shape=(input_dim,), name="ae_input")

    def slice_first(x):
        return tf.reshape(x[:, :conv_input_size], (-1, conv_input_size, 1))

    seq_input = Lambda(
        slice_first,
        output_shape=(conv_input_size, 1),
        name="slice_first",
    )(input_x)

    def slice_remaining(x):
        return x[:, conv_input_size:]

    meta_input = Lambda(
        slice_remaining,
        output_shape=(meta_input_size,),
        name="slice_remaining",
    )(input_x)

    # -----------------------------
    # Sequence encoder branch
    # -----------------------------
    x_seq = seq_input
    for i, (filters, kernel_size, pool_size) in enumerate(
        zip(conv_filters, conv_kernel_sizes, conv_pool_sizes), start=1
    ):
        x_seq = _conv_block(
            x_seq,
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            pool_size=pool_size,
            name_prefix=f"seq_enc_block{i}",
        )

    if pooling == "flatten":
        seq_encoded_features = Flatten(name="seq_flatten")(x_seq)
    elif pooling == "global_max":
        seq_encoded_features = GlobalMaxPooling1D(name="seq_global_max")(x_seq)
    elif pooling == "global_avg":
        seq_encoded_features = GlobalAveragePooling1D(name="seq_global_avg")(x_seq)
    else:
        raise ValueError(f"Unsupported pooling mode: {pooling}")

    # -----------------------------
    # Metadata/tabular encoder branch
    # -----------------------------
    x_meta = meta_input
    for i, units in enumerate(dense_branch_units, start=1):
        x_meta = _dense_block(
            x_meta,
            units=units,
            activation=activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name_prefix=f"meta_enc_block{i}",
        )
    meta_encoded_features = x_meta

    # -----------------------------
    # Fusion + shared encoder
    # -----------------------------
    combined = Concatenate(name="concat_features")(
        [seq_encoded_features, meta_encoded_features]
    )

    x_shared = combined
    for i, units in enumerate(shared_encoder_units, start=1):
        x_shared = _dense_block(
            x_shared,
            units=units,
            activation=activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name_prefix=f"shared_enc_block{i}",
        )

    latent = Dense(
        encoding_dim,
        activation=latent_activation,
        kernel_regularizer=_kernel_regularizer(l2_reg),
        name="latent",
    )(x_shared)

    # -----------------------------
    # Shared latent -> two decoder branches
    # -----------------------------
    # Sequence decoder head
    x_seq_dec = latent
    for i, units in enumerate(seq_decoder_units, start=1):
        x_seq_dec = _dense_block(
            x_seq_dec,
            units=units,
            activation=activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name_prefix=f"seq_dec_block{i}",
        )

    seq_recon = Dense(
        conv_input_size,
        activation=seq_output_activation,
        kernel_regularizer=_kernel_regularizer(l2_reg),
        name="seq_recon",
    )(x_seq_dec)

    # Metadata decoder head
    x_meta_dec = latent
    for i, units in enumerate(meta_decoder_units, start=1):
        x_meta_dec = _dense_block(
            x_meta_dec,
            units=units,
            activation=activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name_prefix=f"meta_dec_block{i}",
        )

    meta_recon = Dense(
        meta_input_size,
        activation=meta_output_activation,
        kernel_regularizer=_kernel_regularizer(l2_reg),
        name="meta_recon",
    )(x_meta_dec)

    # Full reconstruction
    full_recon = Concatenate(name="recon")([seq_recon, meta_recon])

    # -----------------------------
    # Models
    # -----------------------------
    autoencoder = Model(input_x, full_recon, name="autoencoder")
    encoder = Model(input_x, latent, name="encoder")

    # Standalone decoder
    decoder_input = Input(shape=(encoding_dim,), name="decoder_input")

    x_seq_dec_d = decoder_input
    for i in range(1, len(seq_decoder_units) + 1):
        x_seq_dec_d = autoencoder.get_layer(f"seq_dec_block{i}_dense")(x_seq_dec_d)
        if norm_type is not None:
            x_seq_dec_d = autoencoder.get_layer(f"seq_dec_block{i}_norm")(x_seq_dec_d)
        x_seq_dec_d = autoencoder.get_layer(f"seq_dec_block{i}_act")(x_seq_dec_d)
        if dropout_rate > 0:
            x_seq_dec_d = autoencoder.get_layer(f"seq_dec_block{i}_drop")(x_seq_dec_d)
    seq_recon_d = autoencoder.get_layer("seq_recon")(x_seq_dec_d)

    x_meta_dec_d = decoder_input
    for i in range(1, len(meta_decoder_units) + 1):
        x_meta_dec_d = autoencoder.get_layer(f"meta_dec_block{i}_dense")(x_meta_dec_d)
        if norm_type is not None:
            x_meta_dec_d = autoencoder.get_layer(f"meta_dec_block{i}_norm")(x_meta_dec_d)
        x_meta_dec_d = autoencoder.get_layer(f"meta_dec_block{i}_act")(x_meta_dec_d)
        if dropout_rate > 0:
            x_meta_dec_d = autoencoder.get_layer(f"meta_dec_block{i}_drop")(x_meta_dec_d)
    meta_recon_d = autoencoder.get_layer("meta_recon")(x_meta_dec_d)

    decoder_out = autoencoder.get_layer("recon")([seq_recon_d, meta_recon_d])
    decoder = Model(decoder_input, decoder_out, name="decoder")

    # -----------------------------
    # Optional compile
    # -----------------------------
    if compile_model:
        # We compile a multi-output training model so losses can be weighted cleanly.
        train_model = Model(
            input_x,
            {"seq_recon": seq_recon, "meta_recon": meta_recon},
            name="autoencoder_train",
        )
        train_model.compile(
            optimizer=optimizer,
            loss={
                "seq_recon": sequence_loss,
                "meta_recon": metadata_loss,
            },
            loss_weights={
                "seq_recon": sequence_loss_weight,
                "meta_recon": metadata_loss_weight,
            },
        )
        # Attach it for convenience if your AEModels class allows extra attrs.
        autoencoder.train_model = train_model

    return AEModels(autoencoder=autoencoder, encoder=encoder, decoder=decoder)

# -----------------------------
# Model build
# -----------------------------
def build_dense_autoencoder(
    input_dim: int,
    encoding_dim: int,
    encoder_units: Tuple[int, ...] = (128, 64),
    decoder_units: Tuple[int, ...] = (64, 128),
    activation: str = "relu",
    latent_activation: LatentActivationType = None,
    output_activation: Optional[str] = "sigmoid",
    norm_type: NormType = None,
    dropout_rate: float = 0.0,
    l2_reg: float = 0.0,
    compile_model: bool = False,
    optimizer: str | tf.keras.optimizers.Optimizer = "adam",
    loss: str | tf.keras.losses.Loss = "mse",
) -> AEModels:
    """
    Build a dense autoencoder for purely tabular input data.

    Parameters
    ----------
    input_dim : int
        Total input feature dimension.

    encoding_dim : int
        Latent space dimension.

    encoder_units : tuple
        Hidden layer sizes of the encoder before the latent layer.

    decoder_units : tuple
        Hidden layer sizes of the decoder after the latent layer.

    activation : str
        Activation function used in hidden layers.

    latent_activation : Optional[str]
        Activation for latent layer. Use None for linear latent space.

    output_activation : Optional[str]
        Activation of the reconstruction layer.

    norm_type : {None, "batch", "layer"}
        Optional normalization type.

    dropout_rate : float
        Dropout applied after activations.

    l2_reg : float
        L2 regularization strength.

    compile_model : bool
        If True, compile the autoencoder.

    optimizer : str or optimizer
        Optimizer used for compilation.

    loss : str or loss
        Reconstruction loss.

    Returns
    -------
    AEModels
        Contains:
        - autoencoder: Model(input -> reconstruction)
        - encoder: Model(input -> latent)
        - decoder: Model(latent -> reconstruction)
    """

    if input_dim <= 0:
        raise ValueError(f"input_dim must be > 0, got {input_dim}")
    if encoding_dim <= 0:
        raise ValueError(f"encoding_dim must be > 0, got {encoding_dim}")

    # -----------------------------
    # Encoder
    # -----------------------------
    input_x = Input(shape=(input_dim,), name="ae_input")

    x = input_x
    for i, units in enumerate(encoder_units, start=1):
        x = _dense_block(
            x,
            units=units,
            activation=activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name_prefix=f"enc_block{i}",
        )

    latent = Dense(
        encoding_dim,
        activation=latent_activation,
        kernel_regularizer=_kernel_regularizer(l2_reg),
        name="latent",
    )(x)

    # -----------------------------
    # Decoder
    # -----------------------------
    y = latent
    for i, units in enumerate(decoder_units, start=1):
        y = _dense_block(
            y,
            units=units,
            activation=activation,
            norm_type=norm_type,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name_prefix=f"dec_block{i}",
        )

    recon = Dense(
        input_dim,
        activation=output_activation,
        kernel_regularizer=_kernel_regularizer(l2_reg),
        name="recon",
    )(y)

    autoencoder = Model(input_x, recon, name="dense_autoencoder")
    encoder = Model(input_x, latent, name="dense_encoder")

    # -----------------------------
    # Standalone decoder
    # -----------------------------
    decoder_input = Input(shape=(encoding_dim,), name="decoder_input")

    z = decoder_input
    for i in range(1, len(decoder_units) + 1):
        z = autoencoder.get_layer(f"dec_block{i}_dense")(z)
        if norm_type is not None:
            z = autoencoder.get_layer(f"dec_block{i}_norm")(z)
        z = autoencoder.get_layer(f"dec_block{i}_act")(z)
        if dropout_rate > 0:
            z = autoencoder.get_layer(f"dec_block{i}_drop")(z)

    decoder_out = autoencoder.get_layer("recon")(z)
    decoder = Model(decoder_input, decoder_out, name="dense_decoder")

    if compile_model:
        autoencoder.compile(optimizer=optimizer, loss=loss)

    return AEModels(autoencoder=autoencoder, encoder=encoder, decoder=decoder)
