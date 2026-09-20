"""Generalized crop-length sensitivity study (any architecture / method / window).

Generalizes ``first_ml/src/eegsym_study/crop_size.py`` (which was hardcoded to
EEGSym + Running Exponential Standardization on the legacy window) to any
``(architecture, method, window)`` combination, so the crop-length question
("0.5 / 1 / 2 s, which is best?") can be re-asked for whichever architecture
and normalization method wins the corrected-window comparison
(``run_corrected_window_study.py``), on the corrected window.

Same protocol as every other experiment in this project (same optimizer,
loss, callbacks, seed, reused ``dataset_split.json``); the *only* things that
change across runs are the crop length (window/stride geometry + model input
shape) and, via the parameters below, the architecture/method/window.

Everything here is additive and read-only w.r.t. every existing module: it
imports (never edits) the generic pieces of ``first_ml/src/study`` and the
two architecture-specific model builders, and writes only to its own output
directories.

Usage::

    .venv\\Scripts\\python.exe -m src.crop_size_study --arch eegsym --method running_exponential --window corrected --crop 0.5
    .venv\\Scripts\\python.exe -m src.crop_size_study --arch eegsym --method running_exponential --window corrected --crop 2.0
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from . import config as base_config
from .dataset import imagery_window_from_feedback
from .split import load_split, session_keys_for
from .study import config as eegnet_study_config
from .study.artifacts import (
    verify_experiment_outputs,
    write_dataset_statistics,
    write_experiment_config,
    write_metrics,
    write_split_copy,
)
from .study.crops import StudyCropArrays, make_tf_dataset
from .study.data_loading import LoadedTrial, load_trials_for_sessions
from .study.evaluation import evaluate_experiment, plot_training_curves
from .study.leakage_guard import verify_full_leakage
from .study.normalization import EuclideanAlignment, RunningExponentialStandardizer, ZScoreNormalizer
from .utils import get_logger, probe_gpu, save_pickle, set_global_seed

logger = get_logger()

NormalizeFn = Callable[[np.ndarray], np.ndarray]
CROP_OVERLAP = 0.5  # identical 50% overlap to the rest of the project

ARCHITECTURES: Tuple[str, ...] = ("eegnet", "eegsym")


@dataclass(frozen=True)
class CropSizeSpec:
    architecture: str
    method: str
    window: str
    crop_seconds: float

    @property
    def window_samples(self) -> int:
        return int(round(self.crop_seconds * eegnet_study_config.FS_TARGET))

    @property
    def stride_samples(self) -> int:
        return int(round(self.window_samples * (1.0 - CROP_OVERLAP)))

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (eegnet_study_config.N_CHANNELS, self.window_samples, 1)

    @property
    def output_dir(self) -> Path:
        crop_tag = f"crop_{str(self.crop_seconds).replace('.', '_')}s"
        window_dir = "corrected_window" if self.window == "corrected" else "legacy_window"
        return base_config.PROJECT_ROOT / "experiments" / window_dir / "crop_size" / crop_tag


def _build_normalizer(method: str) -> Tuple[NormalizeFn, Dict[str, object]]:
    if method == "z_score":
        raise ValueError(
            "z_score needs train-only fitting; use _build_normalizer_fitted() instead."
        )
    if method == "running_exponential":
        n = RunningExponentialStandardizer()
        params = {
            "method": "running_exponential", "factor_new": n.factor_new,
            "init_block_size": n.init_block_size, "eps": n.eps, "causal": True,
            "fit_on": "none (stateless, purely causal per-trial transform)",
        }
        return n.transform, params
    raise ValueError(f"Unknown normalization method: {method!r}")


def _build_normalizer_fitted(method: str, train_trials: List[LoadedTrial]) -> Tuple[NormalizeFn, Dict[str, object]]:
    """Normalizers requiring a fit on training trials (z_score, euclidean_alignment)."""
    if method == "z_score":
        zs = ZScoreNormalizer()
        zs.fit([t.signal for t in train_trials])
        stats = zs.to_dict()
        params = {
            "method": "z_score", "eps": zs.eps,
            "fit_on": "training trials only (per-channel mean/std)",
            "n_trials_used_for_stats": len(train_trials),
            "mean": stats["mean"], "std": stats["std"],
        }
        return zs.transform, params
    if method == "euclidean_alignment":
        ea = EuclideanAlignment()
        ea.fit([t.signal for t in train_trials])
        params = {
            "method": "euclidean_alignment", "eps": ea.eps,
            "fit_on": "training trials only (single pooled global reference)",
            "n_trials_used_for_reference": len(train_trials),
        }
        return ea.transform, params
    if method == "running_exponential":
        return _build_normalizer(method)
    raise ValueError(f"Unknown normalization method: {method!r}")


def iter_crop_bounds_sized(n_samples: int, window: int, stride: int) -> List[Tuple[int, int]]:
    bounds: List[Tuple[int, int]] = []
    start = 0
    while start + window <= n_samples:
        bounds.append((start, start + window))
        start += stride
    return bounds


def build_crops_sized(
    trials: Sequence[LoadedTrial], normalize_fn: NormalizeFn, window: int, stride: int,
) -> StudyCropArrays:
    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    crop_ids: List[str] = []
    trial_ids: List[str] = []
    for trial in trials:
        normalized = normalize_fn(trial.signal)
        for start, stop in iter_crop_bounds_sized(normalized.shape[1], window, stride):
            win = normalized[:, start:stop]
            x_list.append(win[:, :, None].astype(np.float32))
            y_list.append(trial.label)
            crop_ids.append(f"{trial.trial_id}#crop{start}")
            trial_ids.append(trial.trial_id)
    if not x_list:
        raise RuntimeError("No crops generated for the given trials (window too long?).")
    x = np.stack(x_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    logger.info("Built %d crops from %d trials (window=%d, stride=%d).",
                len(crop_ids), len(trials), window, stride)
    return StudyCropArrays(x=x, y=y, crop_ids=crop_ids, trial_ids=trial_ids)


@dataclass
class SizedTrainingResult:
    model: object
    history: Dict[str, List[float]]
    epoch_times_s: List[float] = field(default_factory=list)
    total_train_time_s: float = 0.0
    n_epochs_run: int = 0
    n_params: int = 0
    model_size_bytes: int = 0


def _epoch_timer(epoch_times: List[float]):
    from tensorflow import keras

    class _Impl(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self._t0 = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            elapsed = time.perf_counter() - self._t0
            epoch_times.append(elapsed)
            logs = logs or {}
            logger.info(
                "epoch %3d | %.2fs | loss=%.4f acc=%.4f val_loss=%.4f val_acc=%.4f",
                epoch + 1, elapsed,
                logs.get("loss", float("nan")), logs.get("accuracy", float("nan")),
                logs.get("val_loss", float("nan")), logs.get("val_accuracy", float("nan")),
            )

    return _Impl()


def _build_callbacks(output_dir: Path, epoch_times: List[float]):
    from tensorflow import keras

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=eegnet_study_config.TRAIN.early_stopping_patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=eegnet_study_config.TRAIN.reduce_lr_factor,
            patience=eegnet_study_config.TRAIN.reduce_lr_patience,
            min_lr=eegnet_study_config.TRAIN.min_lr, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "modelo.keras"), monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.CSVLogger(str(output_dir / "training_epochs.csv")),
        _epoch_timer(epoch_times),
    ]


def _build_model(architecture: str, input_shape: Tuple[int, int, int]):
    if architecture == "eegnet":
        from .study.eegnet import build_eegnet
        return build_eegnet(input_shape=input_shape)
    if architecture == "eegsym":
        from .eegsym_study.eegsym import build_eegsym
        return build_eegsym(input_shape=input_shape)
    raise ValueError(f"Unknown architecture: {architecture!r}")


def train_sized(
    architecture: str, train_crops: StudyCropArrays, val_crops: StudyCropArrays,
    output_dir: Path, input_shape: Tuple[int, int, int],
) -> SizedTrainingResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(eegnet_study_config.RANDOM_SEED)

    from tensorflow import keras

    model = _build_model(architecture, input_shape)
    optimizer = keras.optimizers.AdamW(
        learning_rate=eegnet_study_config.OPTIMIZER.learning_rate,
        weight_decay=eegnet_study_config.OPTIMIZER.weight_decay,
    )
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=eegnet_study_config.LABEL_SMOOTHING)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    model.summary(print_fn=logger.info)
    n_params = int(sum(w.numpy().size for w in model.trainable_weights))

    train_ds = make_tf_dataset(
        train_crops, eegnet_study_config.TRAIN.batch_size, shuffle=True,
        seed=eegnet_study_config.RANDOM_SEED,
    )
    val_ds = make_tf_dataset(
        val_crops, eegnet_study_config.TRAIN.batch_size, shuffle=False,
        seed=eegnet_study_config.RANDOM_SEED,
    )

    epoch_times: List[float] = []
    t_start = time.perf_counter()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=eegnet_study_config.TRAIN.epochs,
        callbacks=_build_callbacks(output_dir, epoch_times), shuffle=False, verbose=2,
    )
    total_time = time.perf_counter() - t_start

    model.save_weights(output_dir / "weights.weights.h5")
    save_pickle(output_dir / "history.pkl", history.history)
    model_size_bytes = (output_dir / "modelo.keras").stat().st_size if \
        (output_dir / "modelo.keras").exists() else 0

    logger.info("Training done in %.1fs (%d epochs, %.2fs/epoch avg, %d params).",
                total_time, len(epoch_times), sum(epoch_times) / max(len(epoch_times), 1), n_params)

    return SizedTrainingResult(
        model=model, history=history.history, epoch_times_s=epoch_times,
        total_train_time_s=total_time, n_epochs_run=len(epoch_times),
        n_params=n_params, model_size_bytes=model_size_bytes,
    )


def _attach_file_log(output_dir: Path) -> logging.FileHandler:
    handler = logging.FileHandler(output_dir / "training.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    return handler


def _dataset_statistics(
    spec: CropSizeSpec, trials_by_split: Dict[str, List[LoadedTrial]],
    crops_by_split: Dict[str, StudyCropArrays],
) -> Dict[str, object]:
    stats: Dict[str, object] = {
        "config": {
            "architecture": spec.architecture, "normalization_method": spec.method,
            "window": spec.window, "motor_channels": list(eegnet_study_config.MOTOR_CHANNELS),
            "fs_target": eegnet_study_config.FS_TARGET, "crop_seconds": spec.crop_seconds,
            "crop_samples": spec.window_samples, "crop_stride_samples": spec.stride_samples,
            "crop_overlap": CROP_OVERLAP, "split_json": str(eegnet_study_config.SPLIT_JSON),
            "split_seed": eegnet_study_config.RANDOM_SEED,
        },
        "splits": {},
    }
    for name, trials in trials_by_split.items():
        crops = crops_by_split[name]
        n_left = sum(1 for t in trials if t.label == base_config.LABEL_TO_ID["left"])
        n_right = sum(1 for t in trials if t.label == base_config.LABEL_TO_ID["right"])
        trials_with_crops = len(set(crops.trial_ids))
        subjects = sorted({t.subject for t in trials}, key=lambda s: int(s.lstrip("S")))
        sessions = sorted({t.session_key for t in trials})
        stats["splits"][name] = {
            "n_subjects": len(subjects), "n_sessions": len(sessions), "n_trials": len(trials),
            "n_trials_with_crops": trials_with_crops,
            "n_trials_dropped_too_short": len(trials) - trials_with_crops,
            "n_left_trials": int(n_left), "n_right_trials": int(n_right), "n_crops": len(crops),
        }
    return stats


def run_crop_experiment(spec: CropSizeSpec) -> Dict[str, object]:
    output_dir = spec.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = _attach_file_log(output_dir)
    try:
        return _run_body(spec, output_dir)
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()


def _run_body(spec: CropSizeSpec, output_dir: Path) -> Dict[str, object]:
    logger.info(
        "=== %s + %s CROP STUDY (window=%s): %.2fs crops (window=%d, stride=%d) -> %s ===",
        spec.architecture, spec.method, spec.window, spec.crop_seconds,
        spec.window_samples, spec.stride_samples, output_dir,
    )
    set_global_seed(eegnet_study_config.RANDOM_SEED)
    gpu_info = probe_gpu()

    window_fn = imagery_window_from_feedback if spec.window == "corrected" else None

    split_payload = load_split()
    trials_by_split = {
        name: load_trials_for_sessions(session_keys_for(split_payload, name), window_fn=window_fn)
        for name in ("train", "val", "test")
    }

    normalize_fn, normalization_params = _build_normalizer_fitted(spec.method, trials_by_split["train"])

    crops_by_split = {
        name: build_crops_sized(trials, normalize_fn, spec.window_samples, spec.stride_samples)
        for name, trials in trials_by_split.items()
    }

    verify_full_leakage(split_payload, trials_by_split, crops_by_split)

    experiment_config = {
        "architecture": spec.architecture,
        "normalization_method": spec.method,
        "normalization_params": normalization_params,
        "window": spec.window,
        "study": "crop_length_sensitivity_generalized",
        "crop": {
            "crop_seconds": spec.crop_seconds, "crop_samples": spec.window_samples,
            "crop_overlap": CROP_OVERLAP, "crop_stride_samples": spec.stride_samples,
            "fs_target": eegnet_study_config.FS_TARGET,
        },
        "training": eegnet_study_config.training_as_dict(),
    }
    write_experiment_config(output_dir, experiment_config)
    write_split_copy(output_dir, split_payload)
    write_dataset_statistics(output_dir, _dataset_statistics(spec, trials_by_split, crops_by_split))

    result = train_sized(spec.architecture, crops_by_split["train"], crops_by_split["val"],
                          output_dir, spec.input_shape)
    plot_training_curves(result.history, output_dir)
    eval_metrics = evaluate_experiment(result.model, crops_by_split["test"], output_dir)

    metrics: Dict[str, object] = {
        "architecture": spec.architecture,
        "normalization_method": spec.method,
        "window": spec.window,
        "crop_seconds": spec.crop_seconds,
        "crop_samples": spec.window_samples,
        "crop_stride_samples": spec.stride_samples,
        "gpu": gpu_info,
        "timing": {
            "total_train_time_s": result.total_train_time_s,
            "n_epochs_run": result.n_epochs_run,
            "mean_epoch_time_s": sum(result.epoch_times_s) / max(len(result.epoch_times_s), 1),
            "median_epoch_time_s": float(np.median(result.epoch_times_s)) if result.epoch_times_s else 0.0,
            "epoch_times_s": result.epoch_times_s,
        },
        "model": {"n_params": result.n_params, "model_size_bytes": result.model_size_bytes},
        "crop": eval_metrics["crop"],
        "trial": eval_metrics["trial"],
    }
    write_metrics(output_dir, metrics)
    verify_experiment_outputs(output_dir)
    logger.info("=== CROP STUDY %.2fs (%s+%s, window=%s) COMPLETE: verified in %s ===",
                spec.crop_seconds, spec.architecture, spec.method, spec.window, output_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalized crop-length sensitivity study.")
    parser.add_argument("--arch", choices=ARCHITECTURES, required=True)
    parser.add_argument("--method", choices=eegnet_study_config.NORMALIZATION_METHODS, required=True)
    parser.add_argument("--window", choices=eegnet_study_config.WINDOWS, default="corrected")
    parser.add_argument("--crop", type=float, required=True, help="Crop length in seconds (e.g. 0.5, 2.0)")
    args = parser.parse_args()
    spec = CropSizeSpec(architecture=args.arch, method=args.method, window=args.window, crop_seconds=args.crop)
    run_crop_experiment(spec)


if __name__ == "__main__":
    main()
