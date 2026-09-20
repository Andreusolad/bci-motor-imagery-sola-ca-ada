r"""Continuous-stream inference + official BCI-IV-1 scoring.

Inference protocol (per the spec):
  eval stream -> EA (fit per subject on the eval signal, unsupervised) -> sliding
  1 s window / 0.5 s stride -> tanh prediction per window -> causal EMA smoothing
  (alpha=0.3) -> expand to the 1000 Hz sample grid (nearest window centre) ->
  clip to [-1,1] -> mean squared error against the official per-sample label.

Official scoring is reproduced exactly: the provided true labels carry NaN over
every non-scored sample (the "1 s after the start cue" rule and the transition
gaps are baked into the NaN mask), and subject 'a' is already cropped to the
official 1 759 140 samples in load_eval. Scoring the finite samples therefore IS
the official rule -- confirmed by the constant-zero baseline reproducing 0.509.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from preprocessing import CROP, HOP, DS, EuclideanAlignment, load_eval, make_crops

EMA_ALPHA = 0.3


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 1.0:
        return x
    out = np.empty_like(x); acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def _mse_official(cont_per_sample: np.ndarray, y1000: np.ndarray) -> float:
    keep = np.isfinite(y1000)
    return float(np.mean((np.clip(cont_per_sample[keep], -1, 1) - y1000[keep]) ** 2))


def evaluate(model, subj: str, ema_alpha: float = EMA_ALPHA) -> Dict[str, float]:
    """Return {'mse_raw', 'mse_smooth'} for one subject's eval stream."""
    sig, y1000 = load_eval(subj)                       # (8, M)@250, per-sample label @1000
    ea = EuclideanAlignment().fit([sig])               # EA on the eval signal (unsupervised)
    aligned = ea.transform(sig)
    starts = list(range(0, aligned.shape[1] - CROP + 1, HOP))
    X = make_crops(aligned)
    pred_w = model.predict(X, batch_size=512, verbose=0).ravel().astype(np.float64)
    centres = np.clip(np.array([int(round((s + CROP / 2) * DS)) for s in starts]),
                      0, len(y1000) - 1)
    t = np.arange(len(y1000)); j = np.clip(np.searchsorted(centres, t), 1, len(centres) - 1)
    nearest = np.where((t - centres[j - 1]) <= (centres[j] - t), j - 1, j)
    return {"mse_raw": _mse_official(pred_w[nearest], y1000),
            "mse_smooth": _mse_official(_ema(pred_w, ema_alpha)[nearest], y1000)}


def zero_output_baseline(subj: str) -> float:
    """Constant-zero MSE (sanity gate; should reproduce ~0.509)."""
    _, y1000 = load_eval(subj)
    keep = np.isfinite(y1000)
    return float(np.mean(y1000[keep] ** 2))
