"""3-class EEGSym: the *same* architecture as the winner, only n_classes=3.

Reuses ``src.eegsym_study.eegsym.build_eegsym`` verbatim (same layers, same
weight-tied hemisphere design, same kernel derivation) and merely sets the
softmax width to 3. The optimizer, loss (categorical cross-entropy with the
same 0.1 label smoothing), and callbacks are the winner's, taken from the
shared first_ml study config, so the *only* differences from the 2-class model
are the output width and the 3-class loss -- exactly as the brief requires.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import lib  # noqa: F401
from . import config3 as C

from src.eegsym_study import config as eegsym_config  # noqa: E402
from src.eegsym_study.eegsym import build_eegsym  # noqa: E402
from src.study.eegnet import build_eegnet  # noqa: E402

ARCHITECTURES = ("eegsym", "eegnet")


def build_model3(architecture: str = "eegsym"):
    """Build the uncompiled 3-class model on (8, 250, 1) inputs."""
    if architecture == "eegsym":
        return build_eegsym(input_shape=C.INPUT_SHAPE, n_classes=C.N_CLASSES)
    if architecture == "eegnet":
        return build_eegnet(input_shape=C.INPUT_SHAPE, n_classes=C.N_CLASSES)
    raise ValueError(f"Unknown architecture: {architecture!r}")


def compile_model3(model):
    """Compile with the winner's exact optimizer/loss (softmax width aside)."""
    from tensorflow import keras

    optimizer = keras.optimizers.AdamW(
        learning_rate=eegsym_config.OPTIMIZER.learning_rate,   # 1e-3
        weight_decay=eegsym_config.OPTIMIZER.weight_decay,     # 1e-4
    )
    loss = keras.losses.CategoricalCrossentropy(
        label_smoothing=eegsym_config.LABEL_SMOOTHING,          # 0.1
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def build_callbacks(output_dir: Path, epoch_times: List[float]):
    """Winner's callback stack (EarlyStopping / ReduceLROnPlateau / checkpoint)."""
    import time as _time

    from tensorflow import keras

    from src.utils import get_logger
    logger = get_logger()
    train = eegsym_config.TRAIN

    class _EpochTimer(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self._t0 = _time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            dt = _time.perf_counter() - self._t0
            epoch_times.append(dt)
            logs = logs or {}
            logger.info(
                "epoch %3d | %.2fs | loss=%.4f acc=%.4f val_loss=%.4f val_acc=%.4f",
                epoch + 1, dt, logs.get("loss", float("nan")),
                logs.get("accuracy", float("nan")), logs.get("val_loss", float("nan")),
                logs.get("val_accuracy", float("nan")),
            )

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=train.early_stopping_patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=train.reduce_lr_factor,
            patience=train.reduce_lr_patience, min_lr=train.min_lr, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "modelo.keras"), monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.CSVLogger(str(output_dir / "training_epochs.csv")),
        _EpochTimer(),
    ]


def training_hparams(architecture: str = "eegsym") -> dict:
    """JSON snapshot of the (winner-identical) training hyper-parameters."""
    from src.study import config as eegnet_config
    t = eegsym_config.TRAIN
    hp = {
        "optimizer": "AdamW",
        "learning_rate": eegsym_config.OPTIMIZER.learning_rate,
        "weight_decay": eegsym_config.OPTIMIZER.weight_decay,
        "label_smoothing": eegsym_config.LABEL_SMOOTHING,
        "loss": "CategoricalCrossentropy(label_smoothing=0.1)",
        "batch_size": t.batch_size,
        "epochs_max": t.epochs,
        "early_stopping_patience": t.early_stopping_patience,
        "reduce_lr_patience": t.reduce_lr_patience,
        "reduce_lr_factor": t.reduce_lr_factor,
        "min_lr": t.min_lr,
        "random_seed": C.RANDOM_SEED,
    }
    if architecture == "eegsym":
        hp["architecture"] = "EEGSym (Perez-Velasco et al. 2022), 3-class head"
        hp["eegsym"] = eegsym_config.eegsym_as_dict()
    elif architecture == "eegnet":
        hp["architecture"] = "EEGNet (Lawhern et al. 2018), 3-class head"
        hp["eegnet"] = eegnet_config.eegnet_as_dict()
    return hp
