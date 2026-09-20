"""CLI entry point: run one full normalization-comparison experiment.

Pipeline (identical structure for all three experiments; only the
normalization step differs)::

    reuse dataset_split.json (seed 42, subject-level)
    -> load train/val/test trials directly from raw .mat (no NPZ)
    -> fit normalizer (Z-score/EA: train-only; REST: stateless/causal)
    -> build crops in RAM
    -> full anti-leakage guard (subject/session/trial/crop) -- abort on failure
    -> build EEGNet, train (AdamW + weight decay + label smoothing)
    -> evaluate (crop-level + trial-level via trial aggregation)
    -> write all required artifacts, then verify they all exist

Usage::

    .venv\\Scripts\\python.exe -m src.study.run_experiment --method z_score
    .venv\\Scripts\\python.exe -m src.study.run_experiment --method running_exponential
    .venv\\Scripts\\python.exe -m src.study.run_experiment --method euclidean_alignment
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

from .. import config as base_config
from ..dataset import imagery_window_from_feedback
from ..split import load_split, session_keys_for
from ..utils import get_logger, probe_gpu, set_global_seed
from . import config as study_config
from .artifacts import (
    verify_experiment_outputs,
    write_dataset_statistics,
    write_experiment_config,
    write_metrics,
    write_split_copy,
)
from .comparability import build_experiment_config
from .crops import StudyCropArrays, build_crops
from .data_loading import LoadedTrial, load_trials_for_sessions
from .evaluation import evaluate_experiment, plot_training_curves
from .leakage_guard import verify_full_leakage
from .normalization import EuclideanAlignment, RunningExponentialStandardizer, ZScoreNormalizer
from .training import train_eegnet

logger = get_logger()


def _attach_file_log(output_dir: Path) -> logging.FileHandler:
    """Attach a per-experiment ``training.log`` file handler to the shared logger."""
    handler = logging.FileHandler(output_dir / "training.log", mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    return handler


def _build_normalizer(method: str, train_trials: List[LoadedTrial]):
    """Return ``(normalize_fn, normalization_params)`` for the requested method."""
    if method == "z_score":
        zs = ZScoreNormalizer()
        zs.fit([t.signal for t in train_trials])
        stats = zs.to_dict()
        params = {
            "method": "z_score",
            "eps": zs.eps,
            "fit_on": "training trials only (per-channel mean/std)",
            "n_trials_used_for_stats": len(train_trials),
            # Fitted statistics, saved verbatim so inference can reproduce the
            # exact same normalization without recomputing it from data.
            "mean": stats["mean"],
            "std": stats["std"],
        }
        return zs.transform, params

    if method == "running_exponential":
        normalizer = RunningExponentialStandardizer()
        params = {
            "method": "running_exponential",
            "factor_new": normalizer.factor_new,
            "init_block_size": normalizer.init_block_size,
            "eps": normalizer.eps,
            "causal": True,
            "fit_on": "none (stateless, purely causal per-trial transform)",
        }
        return normalizer.transform, params

    if method == "euclidean_alignment":
        ea = EuclideanAlignment()
        ea.fit([t.signal for t in train_trials])
        params = {
            "method": "euclidean_alignment",
            "eps": ea.eps,
            "fit_on": "training trials only (single pooled global reference)",
            "n_trials_used_for_reference": len(train_trials),
        }
        return ea.transform, params

    raise ValueError(f"Unknown normalization method: {method!r}")


def _dataset_statistics(
    trials_by_split: Dict[str, List[LoadedTrial]],
    crops_by_split: Dict[str, StudyCropArrays],
) -> Dict[str, object]:
    """Per-split subject/session/trial/crop counts and class balance."""
    stats: Dict[str, object] = {"config": study_config.dataset_as_dict(), "splits": {}}
    for name, trials in trials_by_split.items():
        crops = crops_by_split[name]
        n_left = sum(1 for t in trials if t.label == base_config.LABEL_TO_ID["left"])
        n_right = sum(1 for t in trials if t.label == base_config.LABEL_TO_ID["right"])
        subjects = sorted({t.subject for t in trials}, key=lambda s: int(s.lstrip("S")))
        sessions = sorted({t.session_key for t in trials})
        stats["splits"][name] = {
            "n_subjects": len(subjects),
            "n_sessions": len(sessions),
            "n_trials": len(trials),
            "n_left_trials": int(n_left),
            "n_right_trials": int(n_right),
            "n_crops": len(crops),
        }
    return stats


def run_experiment(method: str, window: str = "legacy") -> Dict[str, object]:
    """Run one full experiment end-to-end and return its metrics dict.

    ``window`` selects which part of each trial is cropped:
      * ``"legacy"``    -- ``[0, triallength]`` (cue-included, published results).
      * ``"corrected"`` -- ``[2000, 2000+triallength]`` ms, i.e. feedback onset
        onwards (see ``dataset.imagery_window_from_feedback``). Same trial
        duration, same trials kept, same number of crops -- only the window's
        anchor point moves.
    """
    output_dir = study_config.experiment_dir(method, window)
    file_handler = _attach_file_log(output_dir)
    try:
        logger.info("=== EXPERIMENT: %s (window=%s) ===", method, window)
        set_global_seed(study_config.RANDOM_SEED)
        gpu_info = probe_gpu()

        window_fn = imagery_window_from_feedback if window == "corrected" else None

        split_payload = load_split()
        train_keys = session_keys_for(split_payload, "train")
        val_keys = session_keys_for(split_payload, "val")
        test_keys = session_keys_for(split_payload, "test")

        trials_by_split = {
            "train": load_trials_for_sessions(train_keys, window_fn=window_fn),
            "val": load_trials_for_sessions(val_keys, window_fn=window_fn),
            "test": load_trials_for_sessions(test_keys, window_fn=window_fn),
        }

        normalize_fn, normalization_params = _build_normalizer(method, trials_by_split["train"])

        crops_by_split = {
            name: build_crops(trials, normalize_fn) for name, trials in trials_by_split.items()
        }

        verify_full_leakage(split_payload, trials_by_split, crops_by_split)

        experiment_config = build_experiment_config(method, normalization_params)
        experiment_config["window"] = {
            "name": window,
            "start_ms": 0 if window == "legacy" else base_config.FEEDBACK_ONSET_MS,
            "stop": "triallength*1000" if window == "legacy" else f"{base_config.FEEDBACK_ONSET_MS} + triallength*1000",
        }
        write_experiment_config(output_dir, experiment_config)
        write_split_copy(output_dir, split_payload)
        write_dataset_statistics(output_dir, _dataset_statistics(trials_by_split, crops_by_split))

        result = train_eegnet(crops_by_split["train"], crops_by_split["val"], output_dir)
        plot_training_curves(result.history, output_dir)
        eval_metrics = evaluate_experiment(result.model, crops_by_split["test"], output_dir)

        metrics: Dict[str, object] = {
            "normalization_method": method,
            "architecture": "eegnet",
            "window": window,
            "gpu": gpu_info,
            "timing": {
                "total_train_time_s": result.total_train_time_s,
                "n_epochs_run": result.n_epochs_run,
                "mean_epoch_time_s": sum(result.epoch_times_s) / max(len(result.epoch_times_s), 1),
                "epoch_times_s": result.epoch_times_s,
            },
            "model": {
                "n_params": result.n_params,
                "model_size_bytes": result.model_size_bytes,
            },
            "crop": eval_metrics["crop"],
            "trial": eval_metrics["trial"],
        }
        write_metrics(output_dir, metrics)
        verify_experiment_outputs(output_dir)
        logger.info("=== EXPERIMENT %s (window=%s) COMPLETE: all artifacts verified in %s ===",
                    method, window, output_dir)
        return metrics
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one normalization-comparison experiment.")
    parser.add_argument("--method", choices=study_config.NORMALIZATION_METHODS, required=True)
    parser.add_argument("--window", choices=study_config.WINDOWS, default="legacy")
    args = parser.parse_args()
    run_experiment(args.method, args.window)


if __name__ == "__main__":
    main()
