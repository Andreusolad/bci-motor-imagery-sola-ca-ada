"""Dynamic cropped-training windows and ``tf.data`` input pipelines.

Crops are generated at runtime from the in-memory preprocessed trials and are
never written to disk. Each 1 s (250-sample) window is produced with a 125-sample
stride (50 % overlap). Every crop carries a globally unique id
(``sessionkey#t{trial}#c{start}``) so the training entry point can prove no crop
is shared between splits.

Model input is time-major ``(CROP_SAMPLES, N_CHANNELS)``; trials are stored
channel-major ``(N_CHANNELS, T)``, so crops are transposed on the way out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from . import config
from .dataset import SessionRef, load_session_cache
from .preprocessing import ZScoreNormalizer
from .utils import get_logger

logger = get_logger()


def iter_crop_bounds(n_samples: int) -> List[Tuple[int, int]]:
    """Return ``(start, stop)`` indices of every crop fully inside ``n_samples``.

    Uses a fixed window (``CROP_SAMPLES``) and stride (``CROP_STRIDE_SAMPLES``).
    Trailing samples that cannot form a full window are dropped (never padded).
    """
    bounds: List[Tuple[int, int]] = []
    start = 0
    while start + config.CROP_SAMPLES <= n_samples:
        bounds.append((start, start + config.CROP_SAMPLES))
        start += config.CROP_STRIDE_SAMPLES
    return bounds


def _session_cache_path(session_key: str):
    return config.CACHE_DIR / f"{session_key}.npz"


def fit_normalizer(train_session_keys: Sequence[str]) -> ZScoreNormalizer:
    """Fit the per-channel z-score using only the training sessions' trials."""
    train_trials: List[np.ndarray] = []
    for key in train_session_keys:
        cached = load_session_cache(_session_cache_path(key))
        train_trials.extend(cached.trials)
    logger.info("Fitting z-score on %d training trials.", len(train_trials))
    return ZScoreNormalizer.fit(train_trials)


@dataclass
class CropArrays:
    """Materialised crops for one split (held in RAM, never persisted)."""

    x: np.ndarray          # (n_crops, CROP_SAMPLES, N_CHANNELS) float32
    y: np.ndarray          # (n_crops,) int64
    ids: List[str]         # globally unique crop identifiers

    def __len__(self) -> int:
        return int(self.x.shape[0])


def build_split_crops(
    session_keys: Sequence[str],
    normalizer: ZScoreNormalizer,
) -> CropArrays:
    """Generate all normalised crops for the given sessions, in memory.

    The crops are computed on the fly and returned as a single array; nothing is
    written to disk, satisfying the "generate crops dynamically" requirement
    while remaining fast for this small dataset.
    """
    crops: List[np.ndarray] = []
    labels: List[int] = []
    ids: List[str] = []

    for key in session_keys:
        cached = load_session_cache(_session_cache_path(key))
        for trial_idx, (trial, label) in enumerate(zip(cached.trials, cached.labels)):
            normalized = normalizer.transform(trial)  # (channels, T)
            for start, stop in iter_crop_bounds(normalized.shape[1]):
                window = normalized[:, start:stop]          # (channels, CROP_SAMPLES)
                crops.append(window.T.astype(np.float32))   # (CROP_SAMPLES, channels)
                labels.append(int(label))
                ids.append(f"{key}#t{trial_idx}#c{start}")

    if not crops:
        raise RuntimeError(f"No crops generated for sessions {list(session_keys)}.")

    x = np.stack(crops).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    logger.info(
        "Built %d crops from %d sessions (left=%d right=%d).",
        len(ids), len(session_keys),
        int(np.sum(y == config.LABEL_TO_ID["left"])),
        int(np.sum(y == config.LABEL_TO_ID["right"])),
    )
    return CropArrays(x=x, y=y, ids=ids)


def verify_no_crop_overlap(splits: "dict[str, CropArrays]") -> None:
    """Raise if any crop id is shared between two splits.

    This is the final, crop-level anti-leakage guard required before training.
    With a subject-level split it can never trigger, but it fails loudly if an
    upstream bug ever breaks that guarantee.
    """
    names = list(splits.keys())
    id_sets = {name: set(splits[name].ids) for name in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = id_sets[a] & id_sets[b]
            if shared:
                example = sorted(shared)[:3]
                raise RuntimeError(
                    f"{len(shared)} crop(s) shared between '{a}' and '{b}', "
                    f"e.g. {example}. Aborting to prevent data leakage."
                )
    logger.info("Crop-level anti-leakage check passed across %s.", names)


def make_tf_dataset(
    crops: CropArrays,
    batch_size: int,
    shuffle: bool,
    seed: int = config.RANDOM_SEED,
):
    """Wrap a :class:`CropArrays` in a batched, prefetched ``tf.data.Dataset``."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((crops.x, crops.y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(crops), 20_000), seed=seed,
                        reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
