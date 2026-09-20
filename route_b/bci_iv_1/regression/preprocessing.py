r"""Preprocessing pipeline for the EEGNet-regression BCI-IV-1 decoder.

Self-contained (numpy + scipy only). Reproducible pipeline, in this order:
  raw .mat (1000 Hz, int16) -> select 8 motor channels -> scale to uV
  -> band-pass 0.5-40 Hz (Butterworth order 4, zero-phase filtfilt) @1000 Hz
  -> downsample to 250 Hz (polyphase, factor 4; the 40 Hz low-pass anti-aliases)
  -> Common Average Reference over the 8 channels
  -> Euclidean Alignment (per subject; fit on the recording, apply to it)
  -> 1 s / 0.5 s crops (250-sample windows, 125-sample stride).

Targets are already on the competition scale: left = -1, right = +1, rest = 0.

Data sources (BCI Competition IV, Dataset 1, 1000 Hz .mat):
  calib -> BCICIV_1calib_1000Hz_mat  (cued left/right, mrk.y in {-1,+1})
  eval  -> BCICIV_1eval_1000Hz_mat   (continuous stream)
  labels-> true_labels_official/mat  (per-sample -1/0/+1 + NaN, verified)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample_poly

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA = Path(os.environ.get("BCI_DATA", Path(__file__).resolve().parents[3] / "data")) / "bci_iv_1"
CALIB_DIR = DATA / "BCICIV_1calib_1000Hz_mat"
EVAL_DIR = DATA / "BCICIV_1eval_1000Hz_mat"
LABEL_DIR = DATA / "true_labels_official" / "mat"

MOTOR_CHANNELS = ["FC3", "FCZ", "FC4", "C3", "CZ", "C4", "CP3", "CP4"]
FS_ORIG, FS_TARGET, DS = 1000, 250, 4
BP_LOW, BP_HIGH, BP_ORDER = 0.5, 40.0, 4
CROP, HOP = 250, 125                     # 1 s window, 0.5 s stride @250 Hz
MI_WIN_MS = (500, 3500)                  # calib MI window: [cue+0.5s, cue+3.5s]
REST_WIN_MS = (-2000, 0)                 # calib rest: pre-cue baseline
REAL_SUBJECTS = ["a", "b", "f", "g"]     # c,d,e are artificially generated

_BP_B, _BP_A = butter(BP_ORDER, [BP_LOW / (FS_ORIG / 2), BP_HIGH / (FS_ORIG / 2)], btype="band")


# --------------------------------------------------------------------------- #
# Signal-level preprocessing
# --------------------------------------------------------------------------- #
def preprocess(sig8_1000: np.ndarray) -> np.ndarray:
    """(8, N)@1000 uV -> band-pass -> downsample to 250 -> CAR -> (8, M)@250."""
    x = filtfilt(_BP_B, _BP_A, sig8_1000, axis=-1)
    x = resample_poly(x, up=1, down=DS, axis=-1)
    x = x - x.mean(axis=0, keepdims=True)
    return x.astype(np.float32)


@dataclass
class EuclideanAlignment:
    """Unsupervised spatial whitening (He & Wu 2020). Reference from the domain's
    own trials; whitens each trial by R^{-1/2}. Uses no labels."""

    eps: float = 1e-6
    _w: np.ndarray = None

    def fit(self, trials: List[np.ndarray]) -> "EuclideanAlignment":
        n_ch = trials[0].shape[0]
        acc = np.zeros((n_ch, n_ch), np.float64)
        for x in trials:
            acc += (x @ x.T) / x.shape[1]
        ref = acc / len(trials)
        vals, vecs = np.linalg.eigh(ref)
        vals = np.maximum(vals, self.eps)
        self._w = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (self._w @ x).astype(np.float32)


def make_crops(sig: np.ndarray) -> np.ndarray:
    """(8, M) -> (n_crops, 8, 250, 1) with 1 s windows and 50 % overlap."""
    out = [sig[:, s:s + CROP][:, :, None] for s in range(0, sig.shape[1] - CROP + 1, HOP)]
    return np.stack(out).astype(np.float32) if out else np.empty((0, sig.shape[0], CROP, 1), np.float32)


# --------------------------------------------------------------------------- #
# Data loaders (BCI-IV-1)
# --------------------------------------------------------------------------- #
def _channel_idx(clab: List[str]) -> List[int]:
    low = {c.lower(): i for i, c in enumerate(clab)}
    return [low[c.lower()] for c in MOTOR_CHANNELS]


def load_calib(subj: str) -> Tuple[List[np.ndarray], List[float]]:
    """Return preprocessed (8, T)@250 MI/rest segments + targets {-1,+1,0}."""
    m = loadmat(str(CALIB_DIR / f"BCICIV_calib_ds1{subj}_1000Hz.mat"),
                struct_as_record=False, squeeze_me=True)
    clab = [str(c) for c in np.asarray(m["nfo"].clab).ravel()]
    idx = _channel_idx(clab)
    cnt = np.asarray(m["cnt"])[:, idx].astype(np.float32).T * 0.1        # (8, N) uV @1000
    pos = np.asarray(m["mrk"].pos).astype(int).ravel()
    y = np.asarray(m["mrk"].y).astype(int).ravel()                       # -1 (left) / +1 (right)
    segs, tgts = [], []
    for p, cls in zip(pos, y):
        a, b = p + MI_WIN_MS[0], p + MI_WIN_MS[1]
        if b > cnt.shape[1] or p + REST_WIN_MS[0] < 0:
            continue
        segs.append(preprocess(cnt[:, a:b])); tgts.append(float(cls))    # MI target = mrk.y
        segs.append(preprocess(cnt[:, p + REST_WIN_MS[0]:p])); tgts.append(0.0)  # rest -> 0
    return segs, tgts


def load_eval(subj: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return preprocessed continuous signal (8, M)@250 and per-sample label @1000."""
    m = loadmat(str(EVAL_DIR / f"BCICIV_eval_ds1{subj}_1000Hz.mat"),
                struct_as_record=False, squeeze_me=True)
    clab = [str(c) for c in np.asarray(m["nfo"].clab).ravel()]
    idx = _channel_idx(clab)
    cnt = np.asarray(m["cnt"])[:, idx].astype(np.float32).T * 0.1        # (8, N) @1000
    lab = loadmat(str(LABEL_DIR / f"BCICIV_eval_ds1{subj}_1000Hz_true_y.mat"), squeeze_me=True)
    yk = max((k for k in lab if not k.startswith("__")), key=lambda k: np.asarray(lab[k]).size)
    y1000 = np.asarray(lab[yk]).astype(np.float64).ravel()[:cnt.shape[1]]  # crop to cnt (subj 'a')
    return preprocess(cnt), y1000


def build_calib_crops(subj: str) -> Tuple[np.ndarray, np.ndarray, EuclideanAlignment]:
    """Calib -> EA (fit on calib) -> crops. Returns (X, targets, fitted_EA)."""
    segs, tgts = load_calib(subj)
    ea = EuclideanAlignment().fit(segs)
    xs, ys = [], []
    for seg, t in zip(segs, tgts):
        c = make_crops(ea.transform(seg))
        xs.append(c); ys.append(np.full(len(c), t, np.float32))
    return np.concatenate(xs), np.concatenate(ys), ea
