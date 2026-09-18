"""Preprocessing primitives shared by dataset building, training and streaming.

Offline steps: common average reference, channel selection, anti-alias downsampling,
baseline correction and window cropping. Two pieces are needed for online use:

  1. Causal band-pass. bandpass(causal=False) uses sosfiltfilt (zero-phase, uses future
     samples, offline only); bandpass(causal=True) and OnlineBandpass use sosfilt, which
     can run in real time. Training with causal=True makes the training filter identical
     to the online one.

  2. Euclidean Alignment (EA): fit_ea() on calibration windows, apply_ea() (R^-1/2 @ X)
     on any new window.
"""
from __future__ import annotations
from math import gcd

import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt, sosfilt_zi, resample_poly

from .config import (CYTON_8, FS_SRC, FS_TGT, T0_MS, MI_ONSET_MS, BASE_MS,
                     BP_LO, BP_HI, BP_ORDER, AMP_THR_UV, CAR_MODE)


# ============================================================
# Common average reference (CAR)
# ============================================================
def car(trial: np.ndarray) -> np.ndarray:
    """Common average reference: subtract the instantaneous mean across channels.

    trial: (n_ch, n_samples). The Cyton provides only 8 channels, so the reference that
    can be used online is car_over8() (CAR_MODE='car8').
    """
    return trial - trial.mean(axis=0, keepdims=True)


def car_over8(trial_8: np.ndarray) -> np.ndarray:
    """CAR computed over the 8 Cyton channels only (reproducible on the device)."""
    return trial_8 - trial_8.mean(axis=0, keepdims=True)


def aplicar_car(trial_62: np.ndarray, labels: list[str], mode: str = CAR_MODE) -> np.ndarray:
    """Apply the reference selected by `mode` and return the 8 Cyton channels.

      'car62' -> CAR over the 62 channels, then pick 8
      'car8'  -> pick 8, then CAR over the 8 (reproducible on the Cyton)
      'none'  -> pick 8, no reference (same for any other value)
    """
    if mode == 'car62':
        return pick(car(trial_62), labels)
    x8 = pick(trial_62, labels)
    if mode == 'car8':
        return car_over8(x8)
    return x8


# ============================================================
# Channel selection
# ============================================================
def pick(trial: np.ndarray, labels: list[str], subset: list[str] = CYTON_8) -> np.ndarray:
    """Select the channels in `subset` from an (n_ch, n_samples) array with `labels`."""
    idx = [labels.index(c) for c in subset]
    return trial[idx, :]


# ============================================================
# Anti-alias downsampling
# ============================================================
def downsample(x: np.ndarray, fs_src: int = FS_SRC, fs_tgt: int = FS_TGT,
               axis: int = -1) -> np.ndarray:
    """Polyphase resampling with anti-alias filtering (plain x[..., ::n] would alias)."""
    if fs_src == fs_tgt:
        return x
    g = gcd(fs_src, fs_tgt)
    return resample_poly(x, fs_tgt // g, fs_src // g, axis=axis)


# ============================================================
# Time axis and baseline
# ============================================================
def ds_time(n: int, fs_tgt: int = FS_TGT, t0: int = T0_MS) -> np.ndarray:
    """Time vector in ms for an array of n samples whose first sample is at t0."""
    return np.arange(n) * (1000.0 / fs_tgt) + t0


def baseline(x: np.ndarray, t_ms: np.ndarray, base: tuple = BASE_MS) -> np.ndarray:
    """Subtract the per-channel mean over the baseline interval `base` (ms)."""
    m = (t_ms >= base[0]) & (t_ms < base[1])
    return x - x[:, m].mean(axis=1, keepdims=True)


# ============================================================
# Window cropping (MI and IDLE)
# ============================================================
def crop_desde(x: np.ndarray, t_ms: np.ndarray, onset_ms: float, w_samp: int) -> np.ndarray:
    """Crop w_samp samples starting at onset_ms."""
    i0 = int(np.searchsorted(t_ms, onset_ms))
    return x[:, i0:i0 + w_samp]


def crop_mi(x: np.ndarray, t_ms: np.ndarray, w_samp: int, onset: float = MI_ONSET_MS) -> np.ndarray:
    """Motor-imagery window: w_samp samples from the MI onset."""
    return crop_desde(x, t_ms, onset, w_samp)


def crop_idle_windows(x: np.ndarray, t_ms: np.ndarray, w_samp: int,
                      region_ms: tuple, hop_samp: int | None = None) -> list[np.ndarray]:
    """Cut IDLE windows by sliding inside `region_ms` (pre-cue rest).

    Returns a list of (n_ch, w_samp) windows. hop_samp sets the overlap (default
    w_samp // 2, i.e. 50%). Only windows that fit entirely inside the region are kept.
    """
    if hop_samp is None:
        hop_samp = max(1, w_samp // 2)
    i0 = int(np.searchsorted(t_ms, region_ms[0]))
    i1 = int(np.searchsorted(t_ms, region_ms[1]))
    out = []
    start = i0
    while start + w_samp <= i1:
        out.append(x[:, start:start + w_samp])
        start += hop_samp
    return out


# ============================================================
# Band-pass: zero-phase (offline) and causal (online)
# ============================================================
def _sos(lo: float = BP_LO, hi: float = BP_HI, fs: int = FS_TGT, order: int = BP_ORDER):
    return butter(order, [lo, hi], btype='band', fs=fs, output='sos')


def bandpass(x: np.ndarray, lo: float = BP_LO, hi: float = BP_HI, fs: int = FS_TGT,
             order: int = BP_ORDER, causal: bool = False) -> np.ndarray:
    """Butterworth band-pass along the last axis.

    causal=False -> sosfiltfilt (zero-phase, uses future samples; offline only).
    causal=True  -> sosfilt (causal, usable in real time; introduces phase delay).
    """
    sos = _sos(lo, hi, fs, order)
    if causal:
        return sosfilt(sos, x, axis=-1).astype(np.float32)
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


class OnlineBandpass:
    """Stateful causal band-pass for filtering a continuous stream chunk by chunk.

    The filter state `zi` is kept between calls, so filtering in blocks of N samples gives
    the same result as filtering the whole stream at once (calling sosfilt without state on
    each block would restart the transient every time).
    """

    def __init__(self, n_ch: int, lo: float = BP_LO, hi: float = BP_HI,
                 fs: int = FS_TGT, order: int = BP_ORDER):
        self.sos = _sos(lo, hi, fs, order)
        zi = sosfilt_zi(self.sos)                      # (n_sections, 2)
        # per-channel state: (n_sections, n_ch, 2)
        self.zi = np.repeat(zi[:, None, :], n_ch, axis=1)
        self._primed = False

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        """chunk: (n_ch, n_new_samples) -> filtered (n_ch, n_new_samples)."""
        if not self._primed:
            # scale the initial state by the first sample to avoid a start-up transient
            self.zi = self.zi * chunk[None, :, 0:1]
            self._primed = True
        y, self.zi = sosfilt(self.sos, chunk, axis=-1, zi=self.zi)
        return y.astype(np.float32)


def flag_amplitud(win: np.ndarray, thr: float = AMP_THR_UV) -> bool:
    """True if the band-passed window exceeds the amplitude threshold (artifact)."""
    return bool(np.abs(bandpass(win, causal=False)).max() > thr)


# ============================================================
# Euclidean Alignment (EA)
# ============================================================
def matrix_inv_sqrt(R: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Symmetric R^{-1/2} via eigendecomposition (R = covariance matrix)."""
    w, V = np.linalg.eigh(R)
    w = np.maximum(w, eps)
    return (V @ np.diag(1.0 / np.sqrt(w)) @ V.T).astype(np.float32)


def fit_ea(X_calib: np.ndarray) -> np.ndarray:
    """Fit the EA whitening matrix R^{-1/2} on calibration windows.

    X_calib: (n, n_ch, w_samp). Covariances use the OAS estimator. Returns R_inv_sqrt
    (n_ch, n_ch), to be applied to new windows of the same subject.
    """
    from pyriemann.estimation import Covariances
    covs = Covariances('oas').fit_transform(X_calib.astype(np.float64))
    return matrix_inv_sqrt(covs.mean(axis=0))


def apply_ea(X: np.ndarray, R_inv_sqrt: np.ndarray) -> np.ndarray:
    """Apply EA: R^{-1/2} @ X. Accepts (n_ch, w) or (n, n_ch, w)."""
    if X.ndim == 2:
        return (R_inv_sqrt @ X).astype(np.float32)
    return np.einsum('ij,njk->nik', R_inv_sqrt, X).astype(np.float32)


def zscore_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std, shape (1, n_ch, 1), for z-scoring with fixed stats."""
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_zscore(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """z-score with precomputed stats. Accepts (n_ch, w) or (n, n_ch, w)."""
    if X.ndim == 2:
        return ((X - mu[0]) / sd[0]).astype(np.float32)
    return ((X - mu) / sd).astype(np.float32)
