"""End-to-end entry point for the plain 1-D CNN baseline.

Pipeline::

    build cache -> dataset statistics -> subject split -> fit z-score
    -> build crops (train/val/test) -> anti-leakage guard -> train -> evaluate

Usage (run from route_b/two_class/, with BCI_DATA set)::

    python main.py --stage all

Individual stages can be run in isolation (``--stage cache``, ``split``,
``train``) for iterative work.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.crops import (
    CropArrays,
    build_split_crops,
    fit_normalizer,
    verify_no_crop_overlap,
)
from src.dataset import build_cache, cache_exists
from src.evaluate import evaluate_model, plot_training_curves
from src.preprocessing import ZScoreNormalizer
from src.split import build_split, load_split, session_keys_for
from src.train import train_model
from src.utils import get_logger, probe_gpu, save_json, set_global_seed, to_native

logger = get_logger()


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_cache(force: bool = False) -> List[Dict[str, object]]:
    """Build the per-session cache and write ``dataset_statistics.json``."""
    logger.info("=== STAGE: cache ===")
    summaries = build_cache(force=force)
    _write_dataset_statistics(summaries)
    return summaries


def stage_split() -> Dict[str, object]:
    """Build and verify the subject-level split -> ``dataset_split.json``."""
    logger.info("=== STAGE: split ===")
    if not cache_exists():
        raise RuntimeError("Cache is empty. Run the 'cache' stage first.")
    return build_split()


def stage_train_eval() -> Dict[str, object]:
    """Fit normaliser, build crops, guard against leakage, train and evaluate."""
    logger.info("=== STAGE: train + evaluate ===")
    probe_gpu()
    split = load_split()

    train_keys = session_keys_for(split, "train")
    val_keys = session_keys_for(split, "val")
    test_keys = session_keys_for(split, "test")

    normalizer = fit_normalizer(train_keys)
    save_json(config.NORMALIZATION_JSON, normalizer.to_dict())

    crops = {
        "train": build_split_crops(train_keys, normalizer),
        "val": build_split_crops(val_keys, normalizer),
        "test": build_split_crops(test_keys, normalizer),
    }
    # Final crop-level anti-leakage guard: abort before training on any overlap.
    verify_no_crop_overlap(crops)

    model, history = train_model(crops["train"], crops["val"])
    plot_training_curves(history)
    metrics = evaluate_model(model, crops["test"])
    return metrics


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _write_dataset_statistics(summaries: List[Dict[str, object]]) -> None:
    """Aggregate per-session summaries into ``dataset_statistics.json``."""
    n_sessions = len(summaries)
    subjects = sorted({s["subject"] for s in summaries}, key=lambda x: int(x.lstrip("S")))
    total_trials = int(sum(s["n_kept"] for s in summaries))
    total_left = int(sum(s["n_left"] for s in summaries))
    total_right = int(sum(s["n_right"] for s in summaries))

    payload = {
        "config": config.as_dict(),
        "n_subjects": len(subjects),
        "n_sessions_cached": n_sessions,
        "total_trials": total_trials,
        "total_left": total_left,
        "total_right": total_right,
        "class_balance": {
            "left_pct": 100 * total_left / total_trials if total_trials else 0,
            "right_pct": 100 * total_right / total_trials if total_trials else 0,
        },
        "per_session": summaries,
    }
    save_json(config.DATASET_STATS_JSON, to_native(payload))
    logger.info(
        "Dataset statistics: %d subjects, %d sessions, %d trials (L=%d R=%d) -> %s",
        len(subjects), n_sessions, total_trials, total_left, total_right,
        config.DATASET_STATS_JSON,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="1-D CNN baseline pipeline.")
    parser.add_argument(
        "--stage",
        choices=["all", "cache", "split", "train"],
        default="all",
        help="Which stage(s) to run.",
    )
    parser.add_argument(
        "--force-cache",
        action="store_true",
        help="Rebuild every session cache file even if it already exists.",
    )
    args = parser.parse_args()

    config.ensure_output_dirs()
    set_global_seed(config.RANDOM_SEED)

    if args.stage in ("all", "cache"):
        stage_cache(force=args.force_cache)
    if args.stage in ("all", "split"):
        stage_split()
    if args.stage in ("all", "train"):
        stage_train_eval()

    logger.info("Done.")


if __name__ == "__main__":
    main()
