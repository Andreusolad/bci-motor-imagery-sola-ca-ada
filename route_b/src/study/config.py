"""Configuration for the normalization-comparison study (EEGNet only).

Everything that must be *identical* across the three experiments (Z-score,
Running Exponential Standardization, Euclidean Alignment) lives here as a
single source of truth: architecture hyper-parameters, optimizer settings,
training schedule, callbacks configuration and the random seed. Only the
normalization method itself is selected per-experiment (see
``run_experiment.py``).

Paths, channels, the sampling rate, trial-selection rules and the crop
geometry are intentionally re-imported from ``first_ml.src.config`` (the
existing, validated project configuration) rather than redefined, so both
studies stay consistent with the rest of the project.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from .. import config as base_config

# --------------------------------------------------------------------------- #
# Re-exported, unchanged from the base project
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = base_config.PROJECT_ROOT
DATA_DIR: Path = base_config.DATA_DIR
MOTOR_CHANNELS: Tuple[str, ...] = base_config.MOTOR_CHANNELS
FS_ORIGINAL: int = base_config.FS_ORIGINAL
FS_TARGET: int = base_config.FS_TARGET
TASK_FILTER: Tuple[int, ...] = base_config.TASK_FILTER
TARGET_TO_LABEL: Dict[int, str] = base_config.TARGET_TO_LABEL
LABEL_TO_ID: Dict[str, int] = base_config.LABEL_TO_ID
ID_TO_LABEL: Dict[int, str] = base_config.ID_TO_LABEL
CLASS_NAMES: Tuple[str, ...] = base_config.CLASS_NAMES
VALID_RESULT: int = base_config.VALID_RESULT
CROP_SAMPLES: int = base_config.CROP_SAMPLES
CROP_STRIDE_SAMPLES: int = base_config.CROP_STRIDE_SAMPLES
MIN_TRIAL_SAMPLES: int = base_config.MIN_TRIAL_SAMPLES
N_CHANNELS: int = base_config.N_CHANNELS
N_CLASSES: int = base_config.N_CLASSES

# The existing subject-level split (seed 42) -- reused verbatim, never
# regenerated, per the study's leakage requirements.
SPLIT_JSON: Path = base_config.SPLIT_JSON

# EEGNet works on channel-major input: (channels, samples, 1).
EEGNET_INPUT_SHAPE: Tuple[int, int, int] = (N_CHANNELS, CROP_SAMPLES, 1)

# --------------------------------------------------------------------------- #
# Random seed (identical for both experiments)
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42
assert RANDOM_SEED == base_config.RANDOM_SEED, "Study seed must match the split seed."

# --------------------------------------------------------------------------- #
# EEGNet architecture (Lawhern et al., 2018) -- one definition, shared
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EEGNetConfig:
    """Architecture hyper-parameters for the single, shared EEGNet model."""

    f1: int = 8                    # Number of temporal filters.
    depth_multiplier: int = 2      # D: depthwise spatial filters per temporal filter.
    f2: int = 16                   # Number of pointwise filters (== f1 * depth_multiplier).
    temporal_kernel_length: int = FS_TARGET // 2   # ~half a second at FS_TARGET.
    separable_kernel_length: int = 16
    pool1_size: int = 4
    pool2_size: int = 8
    dropout_rate: float = 0.5      # 0.5 is the paper's recommendation for cross-subject setups.
    norm_max_depthwise: float = 1.0
    norm_max_dense: float = 0.25


EEGNET: EEGNetConfig = EEGNetConfig()

# --------------------------------------------------------------------------- #
# Optimizer: AdamW + weight decay (mandatory for both experiments)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OptimizerConfig:
    """AdamW settings, identical for both experiments."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


OPTIMIZER: OptimizerConfig = OptimizerConfig()

# Label smoothing (mandatory for both experiments).
LABEL_SMOOTHING: float = 0.1

# --------------------------------------------------------------------------- #
# Training schedule / callbacks (mandatory identical for both experiments)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    """Training loop hyper-parameters, identical for both experiments."""

    batch_size: int = 256
    epochs: int = 100
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 6
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-5


TRAIN: TrainConfig = TrainConfig()

# --------------------------------------------------------------------------- #
# Running Exponential Standardization (Braindecode-style, causal)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunningExponentialConfig:
    """Hyper-parameters for the causal exponential moving standardization."""

    factor_new: float = 1e-3       # Exponential decay factor for the running stats.
    init_block_size: int = FS_TARGET   # 1 s warm-start window used to seed mean/var.
    eps: float = 1e-4              # Numerical floor added to the running variance.


RUNNING_EXPONENTIAL: RunningExponentialConfig = RunningExponentialConfig()

# --------------------------------------------------------------------------- #
# Euclidean Alignment (NumPy/SciPy, reference fit on training data only)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EuclideanAlignmentConfig:
    """Hyper-parameters for Euclidean Alignment."""

    eps: float = 1e-6              # Eigenvalue floor for the inverse square root.


EUCLIDEAN_ALIGNMENT: EuclideanAlignmentConfig = EuclideanAlignmentConfig()

# --------------------------------------------------------------------------- #
# Z-score (classic per-channel standardization, reference/controlled baseline)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ZScoreConfig:
    """Hyper-parameters for classic per-channel Z-score normalization."""

    eps: float = 1e-8              # Numerical floor added to the standard deviation.


Z_SCORE: ZScoreConfig = ZScoreConfig()

# --------------------------------------------------------------------------- #
# Normalization methods
# --------------------------------------------------------------------------- #
NORMALIZATION_METHODS: Tuple[str, ...] = ("z_score", "running_exponential", "euclidean_alignment")

# --------------------------------------------------------------------------- #
# Per-experiment output directories
# --------------------------------------------------------------------------- #
# Two windows share the exact same architecture/training protocol and only
# differ in where the crop is taken from the trial:
#   "legacy"    -> [0, triallength]              (cue-included, published results)
#   "corrected" -> [2000, 2000+triallength] ms    (true imagery, see
#                  ``dataset.imagery_window_from_feedback``)
WINDOWS: Tuple[str, ...] = ("legacy", "corrected")

_NORMALIZATION_BASE = PROJECT_ROOT / "experiments" / "legacy_window" / "normalization" / "eegnet"
_NORMALIZATION_BASE_CORRECTED = PROJECT_ROOT / "experiments" / "corrected_window" / "normalization" / "eegnet"

EXPERIMENT_DIRS: Dict[str, Dict[str, Path]] = {
    "legacy": {
        "z_score": _NORMALIZATION_BASE / "z_score",
        "running_exponential": _NORMALIZATION_BASE / "running_exponential",
        "euclidean_alignment": _NORMALIZATION_BASE / "euclidean_alignment",
    },
    "corrected": {
        "z_score": _NORMALIZATION_BASE_CORRECTED / "z_score",
        "running_exponential": _NORMALIZATION_BASE_CORRECTED / "running_exponential",
        "euclidean_alignment": _NORMALIZATION_BASE_CORRECTED / "euclidean_alignment",
    },
}

COMPARISON_DIRS: Dict[str, Path] = {
    "legacy": _NORMALIZATION_BASE / "comparison",
    "corrected": _NORMALIZATION_BASE_CORRECTED / "comparison",
}
# Backwards-compatible default (legacy window, matches every published result).
COMPARISON_DIR: Path = COMPARISON_DIRS["legacy"]


def experiment_dir(method: str, window: str = "legacy") -> Path:
    """Return (and create) the output directory for ``method`` on ``window``."""
    if window not in EXPERIMENT_DIRS:
        raise ValueError(f"Unknown window: {window!r} (expected one of {WINDOWS})")
    if method not in EXPERIMENT_DIRS[window]:
        raise ValueError(f"Unknown normalization method: {method!r}")
    directory = EXPERIMENT_DIRS[window][method]
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def comparison_dir(window: str = "legacy") -> Path:
    """Return (and create) the comparison output directory for ``window``."""
    if window not in COMPARISON_DIRS:
        raise ValueError(f"Unknown window: {window!r} (expected one of {WINDOWS})")
    directory = COMPARISON_DIRS[window]
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def eegnet_as_dict() -> Dict[str, object]:
    """JSON-serialisable snapshot of the shared EEGNet architecture."""
    return {
        "f1": EEGNET.f1,
        "depth_multiplier": EEGNET.depth_multiplier,
        "f2": EEGNET.f2,
        "temporal_kernel_length": EEGNET.temporal_kernel_length,
        "separable_kernel_length": EEGNET.separable_kernel_length,
        "pool1_size": EEGNET.pool1_size,
        "pool2_size": EEGNET.pool2_size,
        "dropout_rate": EEGNET.dropout_rate,
        "norm_max_depthwise": EEGNET.norm_max_depthwise,
        "norm_max_dense": EEGNET.norm_max_dense,
        "input_shape": list(EEGNET_INPUT_SHAPE),
    }


def training_as_dict() -> Dict[str, object]:
    """JSON-serialisable snapshot of the shared training configuration."""
    return {
        "optimizer": "AdamW",
        "learning_rate": OPTIMIZER.learning_rate,
        "weight_decay": OPTIMIZER.weight_decay,
        "label_smoothing": LABEL_SMOOTHING,
        "batch_size": TRAIN.batch_size,
        "epochs_max": TRAIN.epochs,
        "early_stopping_patience": TRAIN.early_stopping_patience,
        "reduce_lr_patience": TRAIN.reduce_lr_patience,
        "reduce_lr_factor": TRAIN.reduce_lr_factor,
        "min_lr": TRAIN.min_lr,
        "random_seed": RANDOM_SEED,
        "callbacks": [
            "EarlyStopping(monitor=val_loss)",
            "ReduceLROnPlateau(monitor=val_loss)",
            "ModelCheckpoint(monitor=val_loss, save_best_only=True)",
            "CSVLogger",
        ],
        "loss": "CategoricalCrossentropy(label_smoothing=%.3f)" % LABEL_SMOOTHING,
    }


def dataset_as_dict() -> Dict[str, object]:
    """JSON-serialisable snapshot of the shared dataset/crop configuration."""
    return {
        "motor_channels": list(MOTOR_CHANNELS),
        "task_filter": list(TASK_FILTER),
        "target_to_label": TARGET_TO_LABEL,
        "valid_result": VALID_RESULT,
        "fs_target": FS_TARGET,
        "crop_samples": CROP_SAMPLES,
        "crop_stride_samples": CROP_STRIDE_SAMPLES,
        "split_json": str(SPLIT_JSON),
        "split_seed": RANDOM_SEED,
    }
