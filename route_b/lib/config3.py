"""Central configuration for the 3-class (REST/LEFT/RIGHT) study.

Everything that must stay identical to the winning 2-class model
(EEGSym + Euclidean Alignment, corrected window) is imported from
``first_ml`` rather than redefined; only the genuinely new constants
(the REST class, the REST source window, and the continuous-EEG
generation parameters) live here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import lib  # noqa: F401  (side-effect: puts the route_b root on sys.path)

# first_ml reused pieces -------------------------------------------------- #
from src import config as base_config  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SECOND_ML: Path = base_config.PROJECT_ROOT
EXPERIMENTS: Path = SECOND_ML / "experiments"
REST3_DIR: Path = EXPERIMENTS / "rest_3class"
CONTINUOUS_DIR: Path = EXPERIMENTS / "continuous_eval"
SEGMENT_CACHE: Path = EXPERIMENTS / "segment_cache"

# The already-trained winning 2-class model (for reference / comparison only;
# Part 1 trains the 3-class model *from scratch*, it does not fine-tune this).
WINNER_2CLASS_DIR: Path = (
    base_config.PROJECT_ROOT / "experiments" / "corrected_window"
    / "normalization" / "eegsym" / "euclidean_alignment"
)
WINNER_2CLASS_METRICS: Path = WINNER_2CLASS_DIR / "metrics.json"

# Canonical, shared subject-level split (seed 42) -- reused verbatim.
SPLIT_JSON: Path = base_config.SPLIT_JSON
# Data root (BCI_DATA env var, or <repo>/data by default); re-exported so the
# 3-class scripts can locate external datasets under it.
BCI_DATA: Path = base_config.BCI_DATA

# --------------------------------------------------------------------------- #
# Signal / preprocessing (identical to first_ml)
# --------------------------------------------------------------------------- #
FS_ORIGINAL: int = base_config.FS_ORIGINAL            # 1000
FS_TARGET: int = base_config.FS_TARGET                # 250
DOWNSAMPLE_FACTOR: int = base_config.DOWNSAMPLE_FACTOR  # 4
MOTOR_CHANNELS: Tuple[str, ...] = base_config.MOTOR_CHANNELS
N_CHANNELS: int = base_config.N_CHANNELS              # 8
FEEDBACK_ONSET_MS: int = base_config.FEEDBACK_ONSET_MS  # 2000

CROP_SAMPLES: int = base_config.CROP_SAMPLES          # 250 (1 s)
CROP_STRIDE_SAMPLES: int = base_config.CROP_STRIDE_SAMPLES  # 125 (50 % overlap)
CROP_OVERLAP: float = 0.5
INPUT_SHAPE: Tuple[int, int, int] = (N_CHANNELS, CROP_SAMPLES, 1)

RANDOM_SEED: int = base_config.RANDOM_SEED            # 42

# --------------------------------------------------------------------------- #
# Classes (NOTE: 3-class ids differ from the 2-class model on purpose,
# exactly as the brief specifies: 0=REST, 1=LEFT, 2=RIGHT)
# --------------------------------------------------------------------------- #
REST_ID: int = 0
LEFT_ID: int = 1
RIGHT_ID: int = 2
N_CLASSES: int = 3
CLASS_NAMES: Tuple[str, ...] = ("REST", "LEFT", "RIGHT")

# Map the raw dataset's ``targetnumber`` to our 3-class MI ids.
# (targetnumber 1 == right, 2 == left, validated in first_ml/config.py)
TARGET_TO_ID3: Dict[int, int] = {1: RIGHT_ID, 2: LEFT_ID}

# --------------------------------------------------------------------------- #
# REST source window: the pre-cue baseline [-2000, 0) ms.
# Verified empirically: exactly 2000 samples @1000 Hz -> 500 @250 Hz -> 3 crops.
# This is genuine idle (fixation before the cue), with no motor imagery.
# --------------------------------------------------------------------------- #
REST_WINDOW_MS: Tuple[int, int] = (-2000, 0)

# --------------------------------------------------------------------------- #
# Continuous-EEG evaluation (Part 2)
# --------------------------------------------------------------------------- #
CONT_N_SUBJECTS: int = 10
CONT_N_TRIALS_PER_SUBJECT: int = 5
CONT_TRIAL_SECONDS: int = 300
CONT_TRIAL_SAMPLES: int = CONT_TRIAL_SECONDS * FS_TARGET   # 75000
CONT_N_MI_PER_TRIAL: int = 6
# REST periods bracket and separate the MI periods:
#   R M R M R M R M R M R M R  ->  (N_MI + 1) REST periods.
CONT_N_REST_PERIODS: int = CONT_N_MI_PER_TRIAL + 1
# Each REST period must be at least this long (>= 1 crop, and realistic).
CONT_MIN_REST_SAMPLES: int = CROP_SAMPLES                  # 250 (1 s)
CONT_SEEDS: Tuple[int, ...] = (42, 123, 256, 512, 1024)
CONT_SUBJECT_SELECTION_SEED: int = 42   # fixed choice of the 10 subjects across all seeds


# --------------------------------------------------------------------------- #
# Model variants: (architecture, normalization). The already-trained EEGSym+EA
# run keeps its original flat directories; every other variant gets its own.
# --------------------------------------------------------------------------- #
NORM_SHORT: Dict[str, str] = {"euclidean_alignment": "ea", "running_exponential": "rest"}


def model_dir(arch: str, method: str) -> Path:
    if (arch, method) == ("eegsym", "euclidean_alignment"):
        return REST3_DIR
    return EXPERIMENTS / f"rest_3class_{arch}_{NORM_SHORT[method]}"


def continuous_dir(arch: str, method: str) -> Path:
    if (arch, method) == ("eegsym", "euclidean_alignment"):
        return CONTINUOUS_DIR
    return EXPERIMENTS / f"continuous_eval_{arch}_{NORM_SHORT[method]}"


def ensure_dirs() -> None:
    for d in (EXPERIMENTS, REST3_DIR, CONTINUOUS_DIR, SEGMENT_CACHE):
        d.mkdir(parents=True, exist_ok=True)
