"""1 s / 50 %-overlap cropping and REST balancing for the 3-class problem.

Same crop geometry as the whole project (250 samples, stride 125). Euclidean
Alignment is applied to each *whole* segment before cropping -- identical to how
first_ml aligns a whole trial then crops it (EA is linear per-sample, so this
equals aligning each crop). REST is balanced against LEFT/RIGHT at the segment
level (dropping whole REST segments, seeded) so trial-level aggregation stays
clean and no crop ever straddles a balancing decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from . import config3 as C
from .segments import Segment

NormalizeFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class CropArrays:
    x: np.ndarray                       # (n, C, 250, 1) float32
    y: np.ndarray                       # (n,) int64  (0/1/2)
    trial_ids: List[str] = field(default_factory=list)
    crop_ids: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.x.shape[0])


def iter_crop_bounds(n_samples: int,
                     window: int = C.CROP_SAMPLES,
                     stride: int = C.CROP_STRIDE_SAMPLES) -> List[Tuple[int, int]]:
    bounds: List[Tuple[int, int]] = []
    start = 0
    while start + window <= n_samples:
        bounds.append((start, start + window))
        start += stride
    return bounds


def _n_crops(sig: np.ndarray) -> int:
    return len(iter_crop_bounds(sig.shape[1]))


def balance_rest(segments: Sequence[Segment], seed: int) -> Tuple[List[Segment], Dict[str, int]]:
    """Keep all MI segments; sub-sample whole REST segments so that
    n_rest_crops ~= mean(n_left_crops, n_right_crops)."""
    mi_left = [s for s in segments if s.label == C.LEFT_ID]
    mi_right = [s for s in segments if s.label == C.RIGHT_ID]
    rest = [s for s in segments if s.label == C.REST_ID]

    left_crops = sum(_n_crops(s.signal) for s in mi_left)
    right_crops = sum(_n_crops(s.signal) for s in mi_right)
    target_rest_crops = round((left_crops + right_crops) / 2)

    # Every REST segment is 2 s -> exactly 3 crops.
    crops_per_rest = 3
    n_rest_keep = min(len(rest), max(1, round(target_rest_crops / crops_per_rest)))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rest))
    kept_rest = [rest[i] for i in order[:n_rest_keep]]

    balanced = mi_left + mi_right + kept_rest
    info = {
        "n_left_seg": len(mi_left), "n_right_seg": len(mi_right),
        "n_rest_seg_available": len(rest), "n_rest_seg_kept": n_rest_keep,
        "n_left_crops": left_crops, "n_right_crops": right_crops,
        "target_rest_crops": target_rest_crops,
        "n_rest_crops_kept": sum(_n_crops(s.signal) for s in kept_rest),
    }
    return balanced, info


def build_crops(segments: Sequence[Segment], normalize_fn: NormalizeFn) -> CropArrays:
    """Align (EA) each whole segment, then slice into crops."""
    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    trial_ids: List[str] = []
    crop_ids: List[str] = []
    for seg in segments:
        aligned = normalize_fn(seg.signal)               # (C, T)
        for start, stop in iter_crop_bounds(aligned.shape[1]):
            win = aligned[:, start:stop]                  # (C, 250)
            x_list.append(win[:, :, None].astype(np.float32))
            y_list.append(seg.label)
            trial_ids.append(seg.trial_id)
            crop_ids.append(f"{seg.trial_id}#crop{start}")
    if not x_list:
        raise RuntimeError("No crops generated.")
    return CropArrays(
        x=np.stack(x_list).astype(np.float32),
        y=np.asarray(y_list, dtype=np.int64),
        trial_ids=trial_ids, crop_ids=crop_ids,
    )


def make_tf_dataset(crops: CropArrays, batch_size: int, shuffle: bool, seed: int):
    import tensorflow as tf

    y_onehot = tf.one_hot(crops.y, depth=C.N_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((crops.x, y_onehot))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(crops), 20_000), seed=seed,
                        reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def aggregate_by_trial(crop_probs: np.ndarray, trial_ids: Sequence[str],
                       crop_y_true: np.ndarray):
    """Mean of crop softmax probabilities per trial (never majority vote)."""
    order: Dict[str, int] = {}
    for tid in trial_ids:
        if tid not in order:
            order[tid] = len(order)
    n = len(order)
    prob_sum = np.zeros((n, crop_probs.shape[1]), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    y_true = np.full(n, -1, dtype=np.int64)
    for p, tid, yt in zip(crop_probs, trial_ids, crop_y_true):
        idx = order[tid]
        prob_sum[idx] += p
        counts[idx] += 1
        y_true[idx] = yt
    y_prob = prob_sum / counts[:, None]
    y_pred = y_prob.argmax(axis=1)
    return y_true, y_pred, y_prob, counts
