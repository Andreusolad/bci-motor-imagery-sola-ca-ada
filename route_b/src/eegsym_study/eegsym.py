"""EEGSym, manually re-implemented in TensorFlow/Keras.

Faithful adaptation of *EEGSym: Overcoming Inter-Subject Variability in
Motor Imagery Based BCIs With Deep Learning* (Perez-Velasco et al., IEEE
TNSRE 2022). The paper's reference implementation targets 128 Hz / 384-sample
(3 s) crops over an 8- or 16-channel montage; this project's crops are
250 Hz / 250 samples (1 s) over the 8 motor channels FC3/FCZ/FC4/C3/CZ/C4/
CP3/CP4. Every kernel/pooling size below is **proportionally re-derived**
for that sampling rate and crop length using the same design rule the paper
used (temporal scales in *milliseconds*, converted to samples at the
project's own ``FS_TARGET``), exactly as EEGNet's own study does for its
own temporal kernel. No code from the authors' repository is copied; the
public GitHub repo (``Serpeve/EEGSym``) and the paper were only consulted to
recover the architectural description (block order, filter counts, default
hyper-parameters).

Core ideas reproduced:

* **Hemisphere symmetry** -- the 8 channels are split into a *left* branch
  (``FC3, C3, CP3`` + the two central channels ``FCZ, CZ``) and a *right*
  branch (``FC4, C4, CP4`` + the same two central channels), and the
  **same** Keras layer objects (true weight sharing) process both branches,
  exactly as the paper's ``symmetric=True`` design does.
* **Multi-scale temporal inception** -- three parallel temporal
  convolutions per stage (long/medium/short kernels, derived from the
  paper's 500/250/125 ms scales).
* **Residual connections** at every stage (inception blocks, reduction
  blocks, channel-merge, temporal-merge and the classification head).
* **Progressive channel/temporal merging** -- the two hemisphere branches
  are concatenated and merged with a dedicated convolution, and the
  remaining temporal extent is collapsed by a final large-kernel
  convolution before the dense classification head.

Built exclusively from ``Conv2D``, ``DepthwiseConv2D``, ``BatchNormalization``,
``ELU``, ``AveragePooling2D``, ``Dropout``, ``Add``, ``Concatenate``,
``Flatten``, ``Dense`` and ``Softmax`` -- the same layer family EEGNet uses,
plus ``Add``/``Concatenate`` which are structurally required for residual
connections and hemisphere merging.

This module belongs to the fully separate ``first_ml/src/eegsym_study/``
package: it only imports its own local config, never anything from
``first_ml/src/study/``.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from . import config as study_config


def _channel_indices(names: Sequence[str], motor_channels: Sequence[str]) -> List[int]:
    upper = [c.upper() for c in motor_channels]
    return [upper.index(n.upper()) for n in names]


def _ms_to_samples(scales_ms: Sequence[int], fs: int) -> List[int]:
    return [max(1, int(round(ms * fs / 1000))) for ms in scales_ms]


def build_eegsym(
    input_shape: Tuple[int, int, int] = study_config.INPUT_SHAPE,
    n_classes: int = study_config.N_CLASSES,
    cfg: study_config.EEGSymConfig = study_config.EEGSYM,
):
    """Build (uncompiled) EEGSym.

    Parameters
    ----------
    input_shape:
        ``(channels, samples, 1)`` in the project's standard motor-channel
        order (``config.MOTOR_CHANNELS``) -- identical input convention to
        EEGNet, so the shared cropping/normalization pipeline needs no
        changes for this architecture. The hemisphere split happens
        *inside* the model via channel-gathering layers.
    n_classes:
        Number of output classes (2: left/right motor imagery).
    cfg:
        Architecture hyper-parameters (see :class:`study_config.EEGSymConfig`).
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    n_channels, n_samples, _ = input_shape
    fs = study_config.FS_TARGET

    left_idx = _channel_indices(cfg.left_lateral + cfg.central, study_config.MOTOR_CHANNELS)
    right_idx = _channel_indices(cfg.right_lateral + cfg.central, study_config.MOTOR_CHANNELS)
    n_branch_channels = len(left_idx)  # lateral (3) + central (2) = 5

    scales_stage_a = _ms_to_samples(cfg.scales_ms, fs)
    scales_stage_b = [max(1, round(s / 4)) for s in scales_stage_a]

    inputs = keras.Input(shape=input_shape, name="eeg_crop")

    def gather_branch(name: str, indices: List[int]):
        return layers.Lambda(
            lambda x, idx=indices: tf.gather(x, idx, axis=1),
            name=name,
            output_shape=(n_branch_channels, n_samples, 1),
        )(inputs)

    left_branch = gather_branch("left_hemisphere", left_idx)
    right_branch = gather_branch("right_hemisphere", right_idx)

    # ------------------------------------------------------------------ #
    # Shared (weight-tied) sub-network, built once and applied to both
    # hemisphere branches so the two calls share parameters exactly like
    # the paper's symmetric design.
    # ------------------------------------------------------------------ #
    inception_a_temporal = [
        layers.Conv2D(cfg.filters_per_branch, (1, k), padding="same", use_bias=False,
                       name=f"inceptionA_t{i}")
        for i, k in enumerate(scales_stage_a)
    ]
    inception_a_concat_bn = layers.BatchNormalization(name="inceptionA_bn1")
    inception_a_spatial = layers.DepthwiseConv2D(
        (n_branch_channels, 1), use_bias=False, name="inceptionA_spatial"
    )
    inception_a_spatial_bn = layers.BatchNormalization(name="inceptionA_bn2")
    inception_a_skip = layers.Conv2D(
        cfg.filters_per_branch * len(scales_stage_a), (n_branch_channels, 1),
        padding="valid", use_bias=False, name="inceptionA_skip",
    )
    inception_a_pool = layers.AveragePooling2D((1, 2), name="inceptionA_pool")
    inception_a_skip_pool = layers.AveragePooling2D((1, 2), name="inceptionA_skip_pool")
    inception_a_dropout = layers.Dropout(cfg.dropout_rate, name="inceptionA_dropout")

    inception_b_temporal = [
        layers.Conv2D(cfg.filters_per_branch, (1, k), padding="same", use_bias=False,
                       name=f"inceptionB_t{i}")
        for i, k in enumerate(scales_stage_b)
    ]
    inception_b_bn = layers.BatchNormalization(name="inceptionB_bn")
    inception_b_skip = layers.Conv2D(
        cfg.filters_per_branch * len(scales_stage_b), (1, 1), padding="same",
        use_bias=False, name="inceptionB_skip",
    )
    inception_b_pool = layers.AveragePooling2D((1, 2), name="inceptionB_pool")
    inception_b_skip_pool = layers.AveragePooling2D((1, 2), name="inceptionB_skip_pool")
    inception_b_dropout = layers.Dropout(cfg.dropout_rate, name="inceptionB_dropout")

    reduction_blocks = []
    for i, (filters, kernel) in enumerate(zip(cfg.reduction_filters, cfg.reduction_kernels)):
        reduction_blocks.append({
            "conv": layers.Conv2D(filters, (1, kernel), padding="same", use_bias=False,
                                   name=f"reduction{i}_conv"),
            "bn": layers.BatchNormalization(name=f"reduction{i}_bn"),
            "skip": layers.Conv2D(filters, (1, 1), padding="same", use_bias=False,
                                   name=f"reduction{i}_skip"),
            "pool": layers.AveragePooling2D((1, 2), name=f"reduction{i}_pool"),
            "skip_pool": layers.AveragePooling2D((1, 2), name=f"reduction{i}_skip_pool"),
            "dropout": layers.Dropout(cfg.dropout_rate, name=f"reduction{i}_dropout"),
        })

    def symmetric_subnetwork(branch):
        # --- Stage A: multi-scale inception + spatial (channel) collapse --- #
        temporal = [conv(branch) for conv in inception_a_temporal]
        x = layers.Concatenate(axis=-1, name=f"{branch.name}_incA_concat")(temporal) \
            if len(temporal) > 1 else temporal[0]
        x = inception_a_concat_bn(x)
        x = layers.ELU()(x)
        x = inception_a_spatial(x)
        x = inception_a_spatial_bn(x)
        x = layers.ELU()(x)
        x = inception_a_pool(x)
        x = inception_a_dropout(x)
        skip = inception_a_skip(branch)
        skip = inception_a_skip_pool(skip)
        x = layers.Add()([x, skip])

        # --- Stage B: quarter-scale inception (purely temporal now) ------- #
        temporal_b = [conv(x) for conv in inception_b_temporal]
        y = layers.Concatenate(axis=-1)(temporal_b) if len(temporal_b) > 1 else temporal_b[0]
        y = inception_b_bn(y)
        y = layers.ELU()(y)
        y = inception_b_pool(y)
        y = inception_b_dropout(y)
        skip_b = inception_b_skip(x)
        skip_b = inception_b_skip_pool(skip_b)
        x = layers.Add()([y, skip_b])

        # --- Stage C: cascaded residual reduction blocks ------------------ #
        for block in reduction_blocks:
            conv = block["conv"](x)
            conv = block["bn"](conv)
            conv = layers.ELU()(conv)
            conv = block["dropout"](conv)
            skip_c = block["skip"](x)
            merged = layers.Add()([conv, skip_c])
            x = block["pool"](merged)
            # the skip path already matches `merged`'s pre-pool shape, so the
            # same pool is reused for the next iteration's input.
        return x

    left_feat = symmetric_subnetwork(left_branch)
    right_feat = symmetric_subnetwork(right_branch)

    # ------------------------------------------------------------------ #
    # Channel merge: concatenate the two hemispheres then merge them with
    # a dedicated residual convolution (kernel spans the 2-branch axis).
    # ------------------------------------------------------------------ #
    merged = layers.Concatenate(axis=1, name="hemisphere_concat")([left_feat, right_feat])

    cm_conv = layers.Conv2D(cfg.merge_filters, (2, 1), padding="valid", use_bias=False,
                             name="channel_merge_conv")(merged)
    cm_conv = layers.BatchNormalization(name="channel_merge_bn")(cm_conv)
    cm_conv = layers.ELU()(cm_conv)
    cm_conv = layers.Dropout(cfg.dropout_rate)(cm_conv)
    cm_skip = layers.Conv2D(cfg.merge_filters, (2, 1), padding="valid", use_bias=False,
                             name="channel_merge_skip")(merged)
    x = layers.Add(name="channel_merge_add")([cm_conv, cm_skip])

    # ------------------------------------------------------------------ #
    # Temporal merge: collapse whatever time extent remains with one
    # large-kernel convolution (kernel == remaining temporal length).
    # ------------------------------------------------------------------ #
    remaining_t = x.shape[2]
    tm_conv = layers.Conv2D(cfg.merge_filters, (1, remaining_t), padding="valid",
                             use_bias=False, name="temporal_merge_conv")(x)
    tm_conv = layers.BatchNormalization(name="temporal_merge_bn")(tm_conv)
    tm_conv = layers.ELU()(tm_conv)
    tm_conv = layers.Dropout(cfg.dropout_rate)(tm_conv)
    tm_skip = layers.Conv2D(cfg.merge_filters, (1, remaining_t), padding="valid",
                             use_bias=False, name="temporal_merge_skip")(x)
    x = layers.Add(name="temporal_merge_add")([tm_conv, tm_skip])

    # ------------------------------------------------------------------ #
    # Classification head: cascaded 1x1 residual blocks + dense softmax.
    # ------------------------------------------------------------------ #
    for i in range(cfg.head_blocks):
        c = layers.Conv2D(cfg.head_filters, (1, 1), padding="same", use_bias=False,
                           name=f"head{i}_conv")(x)
        c = layers.BatchNormalization(name=f"head{i}_bn")(c)
        c = layers.ELU()(c)
        c = layers.Dropout(cfg.dropout_rate)(c)
        if i == 0 and x.shape[-1] != cfg.head_filters:
            x = layers.Conv2D(cfg.head_filters, (1, 1), padding="same", use_bias=False,
                               name="head_input_proj")(x)
        x = layers.Add(name=f"head{i}_add")([c, x])

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(n_classes, name="dense")(x)
    outputs = layers.Softmax(name="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="eegsym")
