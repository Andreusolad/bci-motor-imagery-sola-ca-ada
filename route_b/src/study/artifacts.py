"""Per-experiment artifact writing and completeness verification.

Each experiment directory (``running_exponential/`` or
``euclidean_alignment/``) must end up with exactly the files listed in
``REQUIRED_FILES``. :func:`verify_experiment_outputs` is called at the end of
every experiment run and raises if anything is missing, so a broken run
cannot silently masquerade as complete.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from ..utils import get_logger, save_json, to_native

logger = get_logger()

REQUIRED_FILES: Sequence[str] = (
    "modelo.keras",
    "weights.weights.h5",
    "history.pkl",
    "metrics.json",
    "dataset_split.json",
    "dataset_statistics.json",
    "training_curves.png",
    "confusion_matrix.png",
    "roc_curve.png",
    "classification_report.txt",
    "experiment_config.json",
    "training.log",
)


def write_experiment_config(output_dir: Path, experiment_config: Dict[str, object]) -> None:
    save_json(output_dir / "experiment_config.json", to_native(experiment_config))
    logger.info("Saved %s", output_dir / "experiment_config.json")


def write_split_copy(output_dir: Path, split_payload: Dict[str, object]) -> None:
    save_json(output_dir / "dataset_split.json", to_native(split_payload))
    logger.info("Saved %s", output_dir / "dataset_split.json")


def write_dataset_statistics(output_dir: Path, statistics: Dict[str, object]) -> None:
    save_json(output_dir / "dataset_statistics.json", to_native(statistics))
    logger.info("Saved %s", output_dir / "dataset_statistics.json")


def write_metrics(output_dir: Path, metrics: Dict[str, object]) -> None:
    save_json(output_dir / "metrics.json", to_native(metrics))
    logger.info("Saved %s", output_dir / "metrics.json")


def verify_experiment_outputs(output_dir: Path) -> None:
    """Raise ``FileNotFoundError`` if any required artifact is missing."""
    missing: List[str] = [name for name in REQUIRED_FILES if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Experiment output directory {output_dir} is missing required file(s): {missing}"
        )
    logger.info("All %d required artifacts present in %s.", len(REQUIRED_FILES), output_dir)
