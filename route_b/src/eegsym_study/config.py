"""Configuration for the EEGSym normalization-comparison study.

Deliberately isolated from ``first_ml/src/study/`` (the EEGNet study): this
module only *reads* (imports, never edits) the architecture-agnostic pieces
of that study's config -- paths, channels, sampling rate, crop geometry, the
optimizer/label-smoothing/training-schedule and the reused
``dataset_split.json`` -- so both studies share an identical training
protocol without this file ever writing back into ``src/study/``. The only
thing genuinely new here is the EEGSym architecture definition and this
study's own (``eegsym_``-prefixed) output directories.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from ..study import config as eegnet_study_config

# --------------------------------------------------------------------------- #
# Re-exported, unchanged -- architecture-agnostic, read-only reuse of the
# EEGNet study's config so both studies share paths, channels, split, seed,
# optimizer, label smoothing and training schedule exactly.
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = eegnet_study_config.PROJECT_ROOT
MOTOR_CHANNELS: Tuple[str, ...] = eegnet_study_config.MOTOR_CHANNELS
FS_TARGET: int = eegnet_study_config.FS_TARGET
N_CHANNELS: int = eegnet_study_config.N_CHANNELS
N_CLASSES: int = eegnet_study_config.N_CLASSES
CROP_SAMPLES: int = eegnet_study_config.CROP_SAMPLES
SPLIT_JSON: Path = eegnet_study_config.SPLIT_JSON
RANDOM_SEED: int = eegnet_study_config.RANDOM_SEED
assert RANDOM_SEED == eegnet_study_config.RANDOM_SEED, "Study seed must match the split seed."

# Channel-major input, identical convention to EEGNet: (channels, samples, 1).
INPUT_SHAPE: Tuple[int, int, int] = eegnet_study_config.EEGNET_INPUT_SHAPE

OPTIMIZER = eegnet_study_config.OPTIMIZER
TRAIN = eegnet_study_config.TRAIN
LABEL_SMOOTHING: float = eegnet_study_config.LABEL_SMOOTHING
NORMALIZATION_METHODS: Tuple[str, ...] = eegnet_study_config.NORMALIZATION_METHODS


# --------------------------------------------------------------------------- #
# EEGSym architecture (Perez-Velasco et al., 2022) -- one definition, shared
# by all three EEGSym experiments.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EEGSymConfig:
    """Architecture hyper-parameters for the single, shared EEGSym model.

    Kernel/pooling sizes are derived at build time from ``scales_ms`` and
    ``FS_TARGET`` (see :mod:`eegsym`), the same way the paper defines them
    relative to sampling rate rather than as fixed sample counts.
    """

    filters_per_branch: int = 8
    scales_ms: Tuple[int, ...] = (500, 250, 125)   # Multi-scale temporal inception, per hemisphere.
    dropout_rate: float = 0.25
    left_lateral: Tuple[str, ...] = ("FC3", "C3", "CP3")
    right_lateral: Tuple[str, ...] = ("FC4", "C4", "CP4")
    central: Tuple[str, ...] = ("FCZ", "CZ")               # Shared by both hemisphere branches.
    reduction_filters: Tuple[int, ...] = (12, 12, 6)       # Stage-C cascaded residual blocks.
    reduction_kernels: Tuple[int, ...] = (8, 4, 2)
    merge_filters: int = 12
    head_filters: int = 12
    head_blocks: int = 4


EEGSYM: EEGSymConfig = EEGSymConfig()

# --------------------------------------------------------------------------- #
# Per-experiment output directories, split by window (see
# ``first_ml.src.study.config.WINDOWS`` for what "legacy"/"corrected" mean).
# --------------------------------------------------------------------------- #
WINDOWS: Tuple[str, ...] = eegnet_study_config.WINDOWS

_NORMALIZATION_BASE = PROJECT_ROOT / "experiments" / "legacy_window" / "normalization" / "eegsym"
_NORMALIZATION_BASE_CORRECTED = PROJECT_ROOT / "experiments" / "corrected_window" / "normalization" / "eegsym"

EXPERIMENT_DIRS: Dict[str, Dict[str, Path]] = {
    "legacy": {
        "z_score": _NORMALIZATION_BASE / "z_score",
        "running_exponential": _NORMALIZATION_BASE / "running_exponential",
        "euclidean_alignment": _NORMALIZATION_BASE / "euclidean_alignment",
    },
    "corrected": {
        "z_score": _NORMALIZATION_BASE_CORRECTED / "z_score",
        # Already trained (formerly ``eegsym_rest_imagery_from2s/``).
        "running_exponential": _NORMALIZATION_BASE_CORRECTED / "running_exponential",
        "euclidean_alignment": _NORMALIZATION_BASE_CORRECTED / "euclidean_alignment",
    },
}

COMPARISON_DIRS: Dict[str, Path] = {
    "legacy": _NORMALIZATION_BASE / "comparison",
    "corrected": _NORMALIZATION_BASE_CORRECTED / "comparison",
}
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


def eegsym_as_dict() -> Dict[str, object]:
    """JSON-serialisable snapshot of the shared EEGSym architecture."""
    return {
        "filters_per_branch": EEGSYM.filters_per_branch,
        "scales_ms": list(EEGSYM.scales_ms),
        "dropout_rate": EEGSYM.dropout_rate,
        "left_lateral": list(EEGSYM.left_lateral),
        "right_lateral": list(EEGSYM.right_lateral),
        "central": list(EEGSYM.central),
        "reduction_filters": list(EEGSYM.reduction_filters),
        "reduction_kernels": list(EEGSYM.reduction_kernels),
        "merge_filters": EEGSYM.merge_filters,
        "head_filters": EEGSYM.head_filters,
        "head_blocks": EEGSYM.head_blocks,
        "input_shape": list(INPUT_SHAPE),
    }


def training_as_dict() -> Dict[str, object]:
    """Delegates to the EEGNet study's identical (read-only) snapshot."""
    return eegnet_study_config.training_as_dict()


def dataset_as_dict() -> Dict[str, object]:
    """Delegates to the EEGNet study's identical (read-only) snapshot."""
    return eegnet_study_config.dataset_as_dict()
