"""The single, shared EEGNet implementation used by every experiment.

Manual TensorFlow/Keras re-implementation of EEGNet (Lawhern et al., 2018,
*"EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer
Interfaces"*) built exclusively from ``Conv2D``, ``DepthwiseConv2D``,
``SeparableConv2D``, ``BatchNormalization``, ``ELU``, ``AveragePooling2D``,
``Dropout``, ``Dense`` and ``Softmax`` -- no third-party EEGNet package is
imported or vendored.

Both the Running-Exponential-Standardization experiment and the Euclidean-
Alignment experiment call :func:`build_eegnet` with the exact same
:class:`~first_ml.src.study.config.EEGNetConfig`, so the architecture can
never drift between them.
"""
from __future__ import annotations

from typing import Tuple

from . import config as study_config


def build_eegnet(
    input_shape: Tuple[int, int, int] = study_config.EEGNET_INPUT_SHAPE,
    n_classes: int = study_config.N_CLASSES,
    cfg: study_config.EEGNetConfig = study_config.EEGNET,
):
    """Build (uncompiled) EEGNet.

    Parameters
    ----------
    input_shape:
        ``(channels, samples, 1)`` -- channel-major, single-"image-channel"
        input, matching the original EEGNet convention.
    n_classes:
        Number of output classes (2: left/right motor imagery).
    cfg:
        Architecture hyper-parameters, shared by both experiments.

    Returns
    -------
    keras.Model
        Uncompiled model; the optimizer/loss are attached identically for
        both experiments in :mod:`first_ml.src.study.training`.
    """
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.constraints import max_norm

    n_channels, n_samples, _ = input_shape
    inputs = keras.Input(shape=input_shape, name="eeg_crop")

    # --- Block 1: temporal convolution + depthwise spatial filtering ----- #
    x = layers.Conv2D(
        cfg.f1,
        kernel_size=(1, cfg.temporal_kernel_length),
        padding="same",
        use_bias=False,
        name="temporal_conv",
    )(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.DepthwiseConv2D(
        kernel_size=(n_channels, 1),
        depth_multiplier=cfg.depth_multiplier,
        use_bias=False,
        padding="valid",
        depthwise_constraint=max_norm(cfg.norm_max_depthwise),
        name="depthwise_conv",
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ELU(name="elu1")(x)
    x = layers.AveragePooling2D(pool_size=(1, cfg.pool1_size), name="avgpool1")(x)
    x = layers.Dropout(cfg.dropout_rate, name="dropout1")(x)

    # --- Block 2: separable convolution ----------------------------------- #
    x = layers.SeparableConv2D(
        cfg.f2,
        kernel_size=(1, cfg.separable_kernel_length),
        padding="same",
        use_bias=False,
        name="separable_conv",
    )(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ELU(name="elu2")(x)
    x = layers.AveragePooling2D(pool_size=(1, cfg.pool2_size), name="avgpool2")(x)
    x = layers.Dropout(cfg.dropout_rate, name="dropout2")(x)

    # --- Classification head ---------------------------------------------- #
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(
        n_classes, kernel_constraint=max_norm(cfg.norm_max_dense), name="dense"
    )(x)
    outputs = layers.Softmax(name="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="eegnet")
