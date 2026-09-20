"""The end-to-end 1D-CNN for raw-EEG motor-imagery classification.

A deliberately small, fully convolutional network operating directly on the raw
(band-passed, referenced, z-scored) time signal. Global Average Pooling makes the
model invariant to the exact temporal alignment of a crop -- the natural inductive
bias for cropped training -- and keeps the parameter count tiny (~30k) for low
latency at inference time.
"""
from __future__ import annotations

from typing import Tuple

from . import config


def build_model(
    input_shape: Tuple[int, int] = config.INPUT_SHAPE,
    n_classes: int = config.N_CLASSES,
    train_cfg: config.TrainConfig = config.TRAIN,
):
    """Build and compile the 1D-CNN.

    Architecture (per :class:`~config.TrainConfig`)::

        Input (250, 8)
        [Conv1D -> BatchNorm -> ReLU -> (MaxPool)] x len(conv_blocks)
        GlobalAveragePooling1D
        Dense -> ReLU -> Dropout
        Dense -> Softmax
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape, name="eeg_crop")
    x = inputs
    for block_idx, (filters, kernel, pool) in enumerate(train_cfg.conv_blocks):
        x = layers.Conv1D(
            filters,
            kernel_size=kernel,
            padding="same",
            use_bias=False,
            name=f"conv{block_idx + 1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn{block_idx + 1}")(x)
        x = layers.Activation("relu", name=f"relu{block_idx + 1}")(x)
        if pool and pool > 1:
            x = layers.MaxPooling1D(pool_size=pool, name=f"pool{block_idx + 1}")(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(train_cfg.dense_units, activation="relu", name="dense")(x)
    x = layers.Dropout(train_cfg.dropout, name="dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="approach_a_cnn1d")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=train_cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
