from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Lambda, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
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

