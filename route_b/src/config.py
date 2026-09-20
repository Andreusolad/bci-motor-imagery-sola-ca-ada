"""Central configuration for the Approach A (raw-EEG 1D-CNN, cropped training) project.

Every tunable constant, path and preprocessing decision lives here so the rest of
the code base stays declarative and the experiment is reproducible. Nothing in
this project ever writes outside ``PROJECT_ROOT``; the raw ``.mat`` files under
``DATA_DIR`` are only ever read.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Route B project root (this file lives in route_b/src/).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
# Data root: the BCI_DATA environment variable, or <repo>/data by default.
# The read-only Stieger2021 .mat files live under <BCI_DATA>/stieger2021/ and
# are only ever read, never modified.
BCI_DATA: Path = Path(os.environ.get("BCI_DATA", PROJECT_ROOT.parent / "data"))
DATA_DIR: Path = BCI_DATA / "stieger2021"

CACHE_DIR: Path = PROJECT_ROOT / "data" / "cache"
# Approach A (raw-EEG 1D-CNN baseline) writes into its own experiments/ folder,
# alongside every other study, instead of the project root.
APPROACH_A_DIR: Path = PROJECT_ROOT / "experiments" / "legacy_window" / "approach_a_baseline"
FIGURES_DIR: Path = APPROACH_A_DIR / "figures"
LOGS_DIR: Path = APPROACH_A_DIR / "logs"
MODELS_DIR: Path = APPROACH_A_DIR / "models"

# Output artefacts (paths only; created lazily by the code that writes them).
# NOTE: dataset_split.json is intentionally still resolved from PROJECT_ROOT --
# it is the one shared, canonical split (seed 42) that every downstream study
# (EEGNet, EEGSym, both windows) reuses verbatim.
SPLIT_JSON: Path = PROJECT_ROOT / "dataset_split.json"
DATASET_STATS_JSON: Path = APPROACH_A_DIR / "dataset_statistics.json"
METRICS_JSON: Path = APPROACH_A_DIR / "metrics.json"
HISTORY_PKL: Path = APPROACH_A_DIR / "history.pkl"
MODEL_KERAS: Path = MODELS_DIR / "modelo.keras"
MODEL_LAST_KERAS: Path = MODELS_DIR / "modelo_last.keras"
WEIGHTS_H5: Path = MODELS_DIR / "weights.h5"
NORMALIZATION_JSON: Path = MODELS_DIR / "normalization.json"
TRAINING_CURVES_PNG: Path = FIGURES_DIR / "training_curves.png"
CONFUSION_MATRIX_PNG: Path = FIGURES_DIR / "confusion_matrix.png"
ROC_CURVE_PNG: Path = FIGURES_DIR / "roc_curve.png"

# --------------------------------------------------------------------------- #
# Signal / acquisition
# --------------------------------------------------------------------------- #
FS_ORIGINAL: int = 1000          # BCI['SRATE'] in every inspected file.
FS_TARGET: int = 250             # Target rate for 1 s == 250 sample crops.
DOWNSAMPLE_FACTOR: int = FS_ORIGINAL // FS_TARGET  # 4

# Feedback onset: the cursor appears (positionx stops being NaN) at exactly
# +2000 ms in 100 % of trials, and triallength is measured from there. The
# preceding [0, 2000) ms is cue presentation only. See
# ``dataset.imagery_window_from_feedback``.
FEEDBACK_ONSET_MS: int = 2000

# The eight sensorimotor channels requested for this study.
MOTOR_CHANNELS: Tuple[str, ...] = ("FC3", "FCZ", "FC4", "C3", "CZ", "C4", "CP3", "CP4")

# --------------------------------------------------------------------------- #
# Trial selection (validated against the raw data, see APPROACH_A.md)
# --------------------------------------------------------------------------- #
# tasknumber == 1 is the pure Left/Right motor-imagery task.
TASK_FILTER: Tuple[int, ...] = (1,)
# targetnumber 1 == right, 2 == left (validated from cursor trajectory positionx).
TARGET_TO_LABEL: Dict[int, str] = {1: "right", 2: "left"}
LABEL_TO_ID: Dict[str, int] = {"left": 0, "right": 1}
ID_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}
CLASS_NAMES: Tuple[str, ...] = tuple(ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL)))
# Only successfully completed trials.
VALID_RESULT: int = 1

# --------------------------------------------------------------------------- #
# Cropping (see the overlap discussion in APPROACH_A.md)
# --------------------------------------------------------------------------- #
CROP_SECONDS: float = 1.0
CROP_SAMPLES: int = int(round(CROP_SECONDS * FS_TARGET))            # 250
# 50 % overlap -> stride of 0.5 s. The 50 ms hop mentioned in the brief is an
# *inference-time* control-loop parameter, not a training augmentation stride.
CROP_OVERLAP: float = 0.5
CROP_STRIDE_SAMPLES: int = int(round(CROP_SAMPLES * (1.0 - CROP_OVERLAP)))  # 125
# A trial must be at least one crop long (in samples at FS_TARGET) to be usable.
MIN_TRIAL_SAMPLES: int = CROP_SAMPLES

# --------------------------------------------------------------------------- #
# Band-pass filter
# --------------------------------------------------------------------------- #
BANDPASS_LOW_HZ: float = 0.5
BANDPASS_HIGH_HZ: float = 40.0
BANDPASS_ORDER: int = 4          # Butterworth order, applied zero-phase (filtfilt).

# --------------------------------------------------------------------------- #
# Dataset split (subject-independent: whole subjects go to a single split)
# --------------------------------------------------------------------------- #
SPLIT_FRACTIONS: Dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    """Hyper-parameters for the training loop."""

    batch_size: int = 256
    epochs: int = 100
    learning_rate: float = 1e-3
    dropout: float = 0.5
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 6
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-5
    # Convolutional stack: (filters, kernel_size, pool_size). pool_size == 0 -> no pool.
    conv_blocks: Tuple[Tuple[int, int, int], ...] = (
        (16, 25, 2),
        (32, 15, 2),
        (64, 7, 0),
        (64, 7, 0),
    )
    dense_units: int = 32


TRAIN: TrainConfig = TrainConfig()


# --------------------------------------------------------------------------- #
# Derived / convenience
# --------------------------------------------------------------------------- #
N_CHANNELS: int = len(MOTOR_CHANNELS)
INPUT_SHAPE: Tuple[int, int] = (CROP_SAMPLES, N_CHANNELS)  # (time, channels)
N_CLASSES: int = len(LABEL_TO_ID)


def ensure_output_dirs() -> None:
    """Create the writable output directories if they do not exist yet."""
    for directory in (CACHE_DIR, FIGURES_DIR, LOGS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def as_dict() -> Dict[str, object]:
    """Return a JSON-serialisable snapshot of the key configuration values."""
    return {
        "fs_original": FS_ORIGINAL,
        "fs_target": FS_TARGET,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "motor_channels": list(MOTOR_CHANNELS),
        "task_filter": list(TASK_FILTER),
        "target_to_label": TARGET_TO_LABEL,
        "label_to_id": LABEL_TO_ID,
        "valid_result": VALID_RESULT,
        "crop_seconds": CROP_SECONDS,
        "crop_samples": CROP_SAMPLES,
        "crop_overlap": CROP_OVERLAP,
        "crop_stride_samples": CROP_STRIDE_SAMPLES,
        "bandpass_hz": [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
        "bandpass_order": BANDPASS_ORDER,
        "split_fractions": SPLIT_FRACTIONS,
        "random_seed": RANDOM_SEED,
    }
