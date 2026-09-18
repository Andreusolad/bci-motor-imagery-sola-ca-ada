"""Sliding window over a continuous 8-channel stream.

Online there is no cue that marks where to crop, so the continuous stream is scanned with
a fixed window of W seconds that advances every STRIDE samples. This module keeps the
buffer, applies causal preprocessing and returns, every stride, a window ready for the
model. Input samples are expected at FS_TGT (no resampling is done here).

Online preprocessing (uses past samples only):
    CAR (8 channels) -> causal band-pass (stateful) -> [per-window centering] -> EA -> z-score

Offline windows use a pre-cue baseline and may use a zero-phase band-pass; here the
band-pass is causal and the baseline is replaced by per-window centering, because neither
the cue nor future samples exist online. Training with the same options reduces the
train/online mismatch.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import config as C
from .preprocessing import OnlineBandpass, car_over8, apply_ea, apply_zscore


# ============================================================
# Calibration profile (what a user's calibration fixes)
# ============================================================
@dataclass
class CalibrationProfile:
    """Everything the system needs from a user after calibration."""
    R_inv_sqrt: np.ndarray            # EA matrix (8, 8)
    mu: np.ndarray                    # z-score mean (1, 8, 1)
    sd: np.ndarray                    # z-score std  (1, 8, 1)
    w_samp: int = C.W_SAMP
    car_mode: str = C.CAR_MODE
    center_window: bool = True        # subtract the window mean (causal stand-in for baseline)

    def preprocess_window(self, win_raw: np.ndarray, causal_filtered: bool = True) -> np.ndarray:
        """Apply the post-filter steps to one (8, w) window that is already CAR + band-pass.

        Returns the window ready for the model (8, w). `causal_filtered` is not used.
        """
        x = win_raw
        if self.center_window:
            x = x - x.mean(axis=1, keepdims=True)
        x = apply_ea(x, self.R_inv_sqrt)
        x = apply_zscore(x, self.mu, self.sd)
        return x.astype(np.float32)


# ============================================================
# Sliding-window extractor
# ============================================================
class SlidingWindowStream:
    """Consumes the stream chunk by chunk and returns preprocessed windows every stride.

    Usage:
        stream = SlidingWindowStream(profile)
        for chunk in source():                # chunk: (8, n_new_samples) at 250 Hz
            for win in stream.push(chunk):    # win: (8, w), ready for the model
                ...
    """

    def __init__(self, profile: CalibrationProfile, w_samp: int = None,
                 stride_samp: int = C.STRIDE_SAMP, fs: int = C.FS_TGT):
        self.profile = profile
        self.w = w_samp or profile.w_samp
        self.stride = stride_samp
        self.bp = OnlineBandpass(C.N_CH, fs=fs)
        self.buf = deque(maxlen=self.w)          # filtered samples, one column each
        self.since_last = 0                      # samples since the last emitted window
        self.n_seen = 0

    def push(self, chunk_raw: np.ndarray) -> list[np.ndarray]:
        """chunk_raw: (8, n) new samples at FS_TGT. Returns the windows completed in it."""
        chunk_raw = np.asarray(chunk_raw, dtype=np.float32)
        if chunk_raw.ndim == 1:
            chunk_raw = chunk_raw[:, None]
        # instantaneous CAR(8) + stateful causal band-pass
        x = car_over8(chunk_raw) if self.profile.car_mode == 'car8' else chunk_raw
        x = self.bp(x)                           # (8, n) causally filtered
        out = []
        for j in range(x.shape[1]):
            self.buf.append(x[:, j])
            self.n_seen += 1
            self.since_last += 1
            if len(self.buf) == self.w and self.since_last >= self.stride:
                self.since_last = 0
                win = np.stack(self.buf, axis=1)     # (8, w)
                out.append(self.profile.preprocess_window(win))
        return out

    def reset(self):
        self.buf.clear(); self.since_last = 0; self.n_seen = 0
        self.bp = OnlineBandpass(C.N_CH)


# ============================================================
# Building a profile from calibration data
# ============================================================
def build_profile_from_calib(X_calib_bp: np.ndarray, center_window: bool = True,
                             w_samp: int = C.W_SAMP) -> CalibrationProfile:
    """Fit EA and z-score statistics on a user's band-passed calibration windows.

    X_calib_bp: (n, 8, w) calibration windows of the user (LH + RH + IDLE), already
    filtered. They are used to fit R (EA) and then the z-score statistics after EA.
    """
    from .preprocessing import fit_ea, zscore_stats
    R = fit_ea(X_calib_bp)
    Xa = apply_ea(X_calib_bp, R)
    mu, sd = zscore_stats(Xa)
    return CalibrationProfile(R_inv_sqrt=R, mu=mu, sd=sd, w_samp=w_samp,
                              center_window=center_window)
