"""Trial Aggregation: crop softmax probabilities averaged up to trial level.

Procedure (identical for both experiments):

1. Obtain the softmax probability vector of every crop.
2. Group all crops belonging to the same trial (via ``crop.trial_id``).
3. Average the probabilities within each group (**mean of probabilities**,
   never majority voting).
4. Assign the final class as the ``argmax`` of the averaged probability
   vector.

Trial-level metrics computed from this aggregation are the study's primary
metrics; crop-level metrics are kept alongside them for reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class TrialAggregation:
    """Per-trial aggregated predictions, in first-seen trial order."""

    trial_ids: List[str]
    y_true: np.ndarray        # (n_trials,) int64
    y_prob: np.ndarray        # (n_trials, n_classes) float32 -- mean crop softmax
    n_crops_per_trial: List[int]

    def __len__(self) -> int:
        return len(self.trial_ids)

    @property
    def y_pred(self) -> np.ndarray:
        return self.y_prob.argmax(axis=1)


def aggregate_by_trial(
    crop_probs: np.ndarray,
    crop_trial_ids: Sequence[str],
    crop_labels: np.ndarray,
) -> TrialAggregation:
    """Average crop-level softmax probabilities within each trial.

    Parameters
    ----------
    crop_probs:
        ``(n_crops, n_classes)`` softmax outputs, one row per crop.
    crop_trial_ids:
        Parent ``trial_id`` of every crop, same order/length as ``crop_probs``.
    crop_labels:
        True label of every crop (all crops of one trial must share the same
        label; this is asserted below).

    Returns
    -------
    TrialAggregation
    """
    groups: Dict[str, List[int]] = {}
    for idx, trial_id in enumerate(crop_trial_ids):
        groups.setdefault(trial_id, []).append(idx)

    trial_ids: List[str] = []
    y_true: List[int] = []
    y_prob: List[np.ndarray] = []
    n_crops: List[int] = []

    for trial_id, indices in groups.items():
        labels_in_trial = {int(crop_labels[i]) for i in indices}
        if len(labels_in_trial) != 1:
            raise ValueError(
                f"Trial {trial_id!r} has inconsistent crop labels: {labels_in_trial}."
            )
        mean_prob = crop_probs[indices].mean(axis=0)
        trial_ids.append(trial_id)
        y_true.append(labels_in_trial.pop())
        y_prob.append(mean_prob)
        n_crops.append(len(indices))

    return TrialAggregation(
        trial_ids=trial_ids,
        y_true=np.asarray(y_true, dtype=np.int64),
        y_prob=np.stack(y_prob).astype(np.float32),
        n_crops_per_trial=n_crops,
    )
