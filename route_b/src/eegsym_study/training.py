"""Training harness for EEGSym: AdamW + weight decay + label smoothing.

Deliberately duplicated (not imported) from the EEGNet study's
``first_ml/src/study/training.py`` so this package never has to import or
touch that file; the training *protocol* itself (optimizer, loss, batch
size, epoch budget, callbacks, seed) is still identical because every value
below is read from :mod:`first_ml.src.eegsym_study.config`, which in turn
re-exports (read-only) the exact same ``OPTIMIZER`` / ``TRAIN`` /
``LABEL_SMOOTHING`` / ``RANDOM_SEED`` objects the EEGNet study uses. The only
thing that differs between the two studies is which model gets built.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from ..study.crops import StudyCropArrays, make_tf_dataset
from ..utils import get_logger, save_pickle
from . import config as study_config
from .eegsym import build_eegsym

logger = get_logger()


@dataclass
class TrainingResult:
    """Everything downstream evaluation/reporting needs about the run."""

    model: object
    history: Dict[str, List[float]]
    epoch_times_s: List[float] = field(default_factory=list)
    total_train_time_s: float = 0.0
    n_epochs_run: int = 0
    n_params: int = 0
    model_size_bytes: int = 0


class _EpochTimerCallback:
    """Keras callback recording wall-clock time per epoch and logging it."""

    def __new__(cls, epoch_times: List[float]):
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
    """Callbacks shared verbatim by all three EEGSym experiments."""
    from tensorflow import keras

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=study_config.TRAIN.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=study_config.TRAIN.reduce_lr_factor,
            patience=study_config.TRAIN.reduce_lr_patience,
            min_lr=study_config.TRAIN.min_lr,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "modelo.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(output_dir / "training_epochs.csv")),
        _EpochTimerCallback(epoch_times),
    ]


def _compile_model(model) -> None:
    from tensorflow import keras

    optimizer = keras.optimizers.AdamW(
        learning_rate=study_config.OPTIMIZER.learning_rate,
        weight_decay=study_config.OPTIMIZER.weight_decay,
    )
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=study_config.LABEL_SMOOTHING)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])


def train_eegsym(
    train_crops: StudyCropArrays,
    val_crops: StudyCropArrays,
    output_dir: Path,
) -> TrainingResult:
    """Build, compile and fit EEGSym identically for all three experiments.

    Persists ``modelo.keras`` (best weights, via ``ModelCheckpoint``),
    ``weights.weights.h5`` (final weights) and ``history.pkl`` into
    ``output_dir``. Returns a :class:`TrainingResult` with timing and model
    size, used later for the cross-experiment comparison report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    from ..utils import set_global_seed
    set_global_seed(study_config.RANDOM_SEED)

    model = build_eegsym()
    _compile_model(model)
    model.summary(print_fn=logger.info)
    n_params = int(sum(w.numpy().size for w in model.trainable_weights))

    train_ds = make_tf_dataset(
        train_crops, study_config.TRAIN.batch_size, shuffle=True, seed=study_config.RANDOM_SEED
    )
    val_ds = make_tf_dataset(
        val_crops, study_config.TRAIN.batch_size, shuffle=False, seed=study_config.RANDOM_SEED
    )

    epoch_times: List[float] = []
    t_start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=study_config.TRAIN.epochs,
        callbacks=_build_callbacks(output_dir, epoch_times),
        shuffle=False,  # the tf.data pipeline already shuffles each epoch
        verbose=2,
    )
    total_time = time.perf_counter() - t_start

    weights_path = output_dir / "weights.weights.h5"
    model.save_weights(weights_path)
    save_pickle(output_dir / "history.pkl", history.history)

    model_size_bytes = (output_dir / "modelo.keras").stat().st_size if \
        (output_dir / "modelo.keras").exists() else 0

    logger.info(
        "Training done in %.1fs (%d epochs, %.2fs/epoch avg, %d params).",
        total_time, len(epoch_times),
        sum(epoch_times) / max(len(epoch_times), 1), n_params,
    )

    return TrainingResult(
        model=model,
        history=history.history,
        epoch_times_s=epoch_times,
        total_train_time_s=total_time,
        n_epochs_run=len(epoch_times),
        n_params=n_params,
        model_size_bytes=model_size_bytes,
    )
