"""Signal-level preprocessing: band-pass, downsample, CAR and z-score.

Ordering for a trial (all operations are per-trial and channel-wise unless
noted): band-pass @1000 Hz -> downsample to 250 Hz -> Common Average Reference.
Z-score is applied later, at crop time, using statistics fitted on the training
split only (see :class:`ZScoreNormalizer`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

from . import config


def _bandpass_coeffs() -> tuple[np.ndarray, np.ndarray]:
    """Return Butterworth SOS-free (b, a) coefficients for the configured band."""
    nyquist = 0.5 * config.FS_ORIGINAL
    low = config.BANDPASS_LOW_HZ / nyquist
    high = config.BANDPASS_HIGH_HZ / nyquist
    b, a = butter(config.BANDPASS_ORDER, [low, high], btype="band")
    return b, a


_B, _A = _bandpass_coeffs()


def bandpass_filter(trial: np.ndarray) -> np.ndarray:
    """Zero-phase Butterworth band-pass on a ``(n_channels, n_samples)`` trial.

    Filtering is done at the original 1000 Hz so the 40 Hz low-pass also acts as
    an anti-alias filter for the subsequent downsampling.
    """
    # filtfilt needs length > 3 * max(len(a), len(b)); trials are always longer.
    return filtfilt(_B, _A, trial, axis=-1).astype(np.float32, copy=False)


def downsample(trial: np.ndarray) -> np.ndarray:
    """Polyphase downsample from ``FS_ORIGINAL`` to ``FS_TARGET`` along time."""
    return resample_poly(trial, up=1, down=config.DOWNSAMPLE_FACTOR, axis=-1).astype(
        np.float32, copy=False
    )


def common_average_reference(trial: np.ndarray) -> np.ndarray:
    """Subtract, at every timepoint, the mean across the selected channels.

    CAR is computed over exactly the eight motor channels, matching what a
    minimal real-time system reading only those electrodes would compute.
    """
    return (trial - trial.mean(axis=0, keepdims=True)).astype(np.float32, copy=False)


def preprocess_trial(trial: np.ndarray) -> np.ndarray:
    """Full per-trial chain: band-pass -> downsample -> CAR.

    Parameters
    ----------
    trial:
        ``(n_channels, n_samples)`` float array at ``FS_ORIGINAL``, already
        restricted to the motor-imagery window and motor channels.

    Returns
    -------
    np.ndarray
        ``(n_channels, n_samples_downsampled)`` float32 array at ``FS_TARGET``.
    """
    trial = bandpass_filter(trial)
    trial = downsample(trial)
    trial = common_average_reference(trial)
    return trial


@dataclass
class ZScoreNormalizer:
    """Per-channel z-score, fitted on training data only.

    ``mean`` and ``std`` have shape ``(n_channels,)`` and are broadcast across
    time. Fitting uses running sums so it never holds all samples at once.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, trials: List[np.ndarray], eps: float = 1e-8) -> "ZScoreNormalizer":
        """Fit per-channel statistics over a list of ``(n_channels, T)`` trials."""
        n_channels = config.N_CHANNELS
        total = np.zeros(n_channels, dtype=np.float64)
        total_sq = np.zeros(n_channels, dtype=np.float64)
        count = 0
        for trial in trials:
            total += trial.sum(axis=1)
            total_sq += np.square(trial, dtype=np.float64).sum(axis=1)
            count += trial.shape[1]
        if count == 0:
            raise ValueError("Cannot fit ZScoreNormalizer on zero samples.")
        mean = total / count
        var = np.maximum(total_sq / count - mean**2, 0.0)
        std = np.sqrt(var) + eps
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, crop: np.ndarray) -> np.ndarray:
        """Normalise a ``(n_channels, T)`` crop with the fitted statistics."""
        return ((crop - self.mean[:, None]) / self.std[:, None]).astype(
            np.float32, copy=False
        )

    def to_dict(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: Dict[str, List[float]]) -> "ZScoreNormalizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )
