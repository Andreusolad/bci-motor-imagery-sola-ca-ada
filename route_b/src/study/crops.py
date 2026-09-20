"""Dynamic, in-RAM cropping for the study -- EEGNet channel-major layout.

Reuses :func:`first_ml.src.crops.iter_crop_bounds` for the crop-boundary
geometry (same 1 s / 50 % overlap windows as the rest of the project) but
targets EEGNet's ``(channels, samples, 1)`` input instead of the 1D-CNN's
time-major layout, and additionally tracks the parent ``trial_id`` of every
crop so probabilities can later be aggregated back to trial level.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence

import numpy as np

from .. import crops as base_crops
from ..utils import get_logger
from .data_loading import LoadedTrial

logger = get_logger()

NormalizeFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class StudyCropArrays:
    """Materialised EEGNet crops for one split (RAM-only, never persisted)."""

    x: np.ndarray                 # (n_crops, n_channels, crop_samples, 1) float32
    y: np.ndarray                 # (n_crops,) int64
    crop_ids: List[str] = field(default_factory=list)
    trial_ids: List[str] = field(default_factory=list)   # parallel to crop_ids

    def __len__(self) -> int:
        return int(self.x.shape[0])


def build_crops(trials: Sequence[LoadedTrial], normalize_fn: NormalizeFn) -> StudyCropArrays:
    """Normalize every trial then slice it into crops, all held in memory.

    ``normalize_fn`` is applied to each trial's ``(n_channels, n_samples)``
    signal *before* cropping -- this is the single point where the two
    experiments diverge (Running Exponential Standardization vs. Euclidean
    Alignment); everything else in this function is shared.
    """
    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    crop_ids: List[str] = []
    trial_ids: List[str] = []

    for trial in trials:
        normalized = normalize_fn(trial.signal)  # (n_channels, T)
        for start, stop in base_crops.iter_crop_bounds(normalized.shape[1]):
            window = normalized[:, start:stop]              # (n_channels, crop_samples)
            x_list.append(window[:, :, None].astype(np.float32))  # (chans, samples, 1)
            y_list.append(trial.label)
            crop_ids.append(f"{trial.trial_id}#crop{start}")
            trial_ids.append(trial.trial_id)

    if not x_list:
        raise RuntimeError("No crops generated for the given trials.")

    x = np.stack(x_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    logger.info("Built %d crops from %d trials.", len(crop_ids), len(trials))
    return StudyCropArrays(x=x, y=y, crop_ids=crop_ids, trial_ids=trial_ids)


def make_tf_dataset(crops: StudyCropArrays, batch_size: int, shuffle: bool, seed: int):
    """Wrap :class:`StudyCropArrays` in a batched, prefetched ``tf.data.Dataset``.

    Labels are one-hot encoded here (needed for label-smoothed categorical
    cross-entropy, shared by both experiments).
    """
    import tensorflow as tf

    from .. import config as base_config

    y_onehot = tf.one_hot(crops.y, depth=base_config.N_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((crops.x, y_onehot))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(crops), 20_000), seed=seed,
                        reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
