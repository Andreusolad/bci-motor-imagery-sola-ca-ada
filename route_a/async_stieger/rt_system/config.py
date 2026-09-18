"""Constants shared by the asynchronous decoding package.

Dataset, models, detection, streaming and accumulator read their constants from here:
paths, montage, sampling rates, window length, filter band, thresholds and training
hyperparameters.

Paths are derived from the location of this file: PROJ is the repository root, raw data
are read from $BCI_DATA/stieger2021 (default <repo>/data/stieger2021) and caches are
written to async_stieger/cache.
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import torch


# ============================================================
# Paths (derived from the location of this file)
# ============================================================
# <repo>/route_a/async_stieger/rt_system/config.py
#   parents[0]=rt_system  [1]=async_stieger  [2]=route_a  [3]=<repo>
PROJ   = Path(__file__).resolve().parents[3]
DATA   = Path(os.environ.get('BCI_DATA', PROJ / 'data')) / 'stieger2021'
PKGDIR = Path(__file__).resolve().parents[1]
CACHED = PKGDIR / 'cache'
OUTDIR = PKGDIR / 'outputs'

# 3-class (LH/RH/IDLE) window cache written by dataset.construir_cache_3clase()
NPZ_3CLASE = CACHED / 'trials_cyton8_3clase.npz'


# ============================================================
# Acquisition and montage (target hardware: OpenBCI Cyton, 8 channels)
# ============================================================
CYTON_8 = ['FC3', 'FCZ', 'FC4', 'C3', 'CZ', 'C4', 'CP3', 'CP4']
FS_SRC  = 1000      # sampling rate of the Stieger recordings (Hz)
FS_TGT  = 250       # Cyton sampling rate (Hz); recordings are downsampled to it
N_CH    = 8

# The Cyton records only 8 channels, so a common average reference computed over the
# 62 lab channels cannot be reproduced online. CAR_MODE selects the reference used by
# dataset.py and streaming.py:
#   'car8'  -> CAR over the 8 Cyton channels (reproducible on the device)
#   'car62' -> CAR over all 62 channels, then pick 8 (not reproducible on the device)
#   'none'  -> no common reference
CAR_MODE = 'car8'


# ============================================================
# Time windows
# ============================================================
# Stieger trial layout: first sample at t = -2000 ms (t = 0 is the target cue); the
# cursor appears and motor imagery starts at +2000 ms; nsamp = 5001 + triallen*1000.
T0_MS       = -2000       # time of the first sample (ms)
MI_ONSET_MS = 2000        # cursor onset = start of motor imagery (ms)
BASE_MS     = (-1000, 0)  # baseline interval: last 1 s before the cue (ms)

# Decision window length (s). The same length is used for the MI crop, the IDLE windows
# and the sliding inference window; changing it requires retraining.
W_SEC   = 2.0
W_SAMP  = int(round(W_SEC * FS_TGT))   # samples per window (2.0 s -> 500)

# Sliding window over the continuous stream: re-classification step.
STRIDE_MS   = 125                          # step (ms) -> about 8 decisions/s
STRIDE_SAMP = int(round(STRIDE_MS / 1000 * FS_TGT))

# Region from which IDLE windows are cut: before the cue, i.e. awake rest with no
# intention of control.
IDLE_REGION_MS = (-2000, 0)


# ============================================================
# Filtering
# ============================================================
BP_LO, BP_HI = 8.0, 30.0     # mu + beta band (Hz)
BP_ORDER     = 4
AMP_THR_UV   = 150.0         # amplitude artifact flag after band-pass (uV)


# ============================================================
# Data / split
# ============================================================
# Stieger sessions included in the 3-class cache.
SESSIONS  = [5, 6]
N_TEST    = 12                        # held-out subjects (stratified by online hit rate)
ILLIT_THR = 0.50                      # online hit rate < 0.5 -> BCI-illiterate

# Class labels of the asynchronous system
LABELS   = {0: 'left_hand', 1: 'right_hand', 2: 'idle'}
CLASES   = ['left_hand', 'right_hand', 'idle']
N_CLASES = 3
IDLE_ID  = 2                          # id of the no-control class


# ============================================================
# Training hyperparameters
# ============================================================
SEED            = 42
BATCH           = 64
BATCH_FT        = 32
EPOCHS_PRETRAIN = 80
EPOCHS_FT       = 30
LR_PRETRAIN     = 5e-4
LR_FT           = 1e-4
WD              = 1e-4


# ============================================================
# Detection and accumulation defaults. The confidence threshold can be re-fitted with
# calibration.select_conf_threshold() and the OOD thresholds with ood.*.select_threshold().
# ============================================================
# Confidence rejection (temperature-calibrated softmax)
CONF_THR = 0.60          # max(P(LH), P(RH)) below this -> no command
# OOD rejection: not read by the package; the detectors keep their own threshold_
OOD_RIEMANN_THR = None
OOD_MAHAL_THR   = None
# Explicit idle class
IDLE_PROB_THR = 0.50     # P(idle) above this -> no command

# Temporal accumulation
DWELL_MS       = 600     # time a class must be sustained before a command is emitted
EMA_ALPHA      = 0.30    # exponential smoothing factor for the probabilities
REFRACTORY_MS  = 1000    # lockout after a command is emitted
# Sticky Bayesian filter: probability of staying in the same state
HMM_STAY = 0.90


# ============================================================
# Torch / reproducibility
# ============================================================
def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gpu_mem() -> str:
    if device.type == 'cuda':
        free = torch.cuda.mem_get_info(0)[0] / 1024**3
        alloc = torch.cuda.memory_allocated() / 1024**2
        return f'{alloc:.0f}MB alloc / {free:.1f}GB free'
    return 'cpu'


def log(*a) -> None:
    print(*a, flush=True)
