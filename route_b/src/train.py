"""Training loop: callbacks, class weighting, artefact persistence."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from . import config
from .crops import CropArrays, make_tf_dataset
from .model import build_model
from .utils import get_logger, save_pickle

logger = get_logger()


def _save_weights(model) -> None:
    """Save weights as ``models/weights.h5`` (the requested deliverable name).

    Keras 3 only lets ``save_weights`` write a path ending in ``.weights.h5``, so
    we write to that name first and rename to ``weights.h5``. To reload the raw
    weights, rename back to ``*.weights.h5`` or (recommended) load
    ``models/modelo.keras`` directly, which is self-contained.
    """
    tmp = config.MODELS_DIR / "weights.weights.h5"
    model.save_weights(tmp)
    if config.WEIGHTS_H5.exists():
        config.WEIGHTS_H5.unlink()
    tmp.rename(config.WEIGHTS_H5)


def _class_weights(y: np.ndarray) -> Dict[int, float]:
    """Inverse-frequency class weights to counter any residual class imbalance."""
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    return {int(c): float(total / (len(classes) * n)) for c, n in zip(classes, counts)}


def _build_callbacks():
    from tensorflow import keras

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.TRAIN.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.TRAIN.reduce_lr_factor,
            patience=config.TRAIN.reduce_lr_patience,
            min_lr=config.TRAIN.min_lr,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODEL_KERAS),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(config.LOGS_DIR / "training_log.csv")),
    ]


def train_model(
    train_crops: CropArrays,
    val_crops: CropArrays,
) -> Tuple[object, Dict[str, list]]:
    """Train the CNN and persist the best model, last model, weights and history.

    Returns the (best-weights-restored) model and the Keras history dict.
    """
    config.ensure_output_dirs()

    model = build_model()
    model.summary(print_fn=logger.info)

    train_ds = make_tf_dataset(train_crops, config.TRAIN.batch_size, shuffle=True)
    val_ds = make_tf_dataset(val_crops, config.TRAIN.batch_size, shuffle=False)

    class_weight = _class_weights(train_crops.y)
    logger.info("Class weights: %s", class_weight)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.TRAIN.epochs,
        class_weight=class_weight,
        callbacks=_build_callbacks(),
        shuffle=False,  # the tf.data pipeline already shuffles each epoch
        verbose=2,
    )

    # Persist artefacts. ModelCheckpoint already wrote the best model to
    # MODEL_KERAS; here we also save the last-epoch model and portable weights.
    model.save(config.MODEL_LAST_KERAS)
    _save_weights(model)
    save_pickle(config.HISTORY_PKL, history.history)
    logger.info("Saved best model -> %s", config.MODEL_KERAS)
    logger.info("Saved last model -> %s", config.MODEL_LAST_KERAS)
    logger.info("Saved weights    -> %s", config.WEIGHTS_H5)
    return model, history.history
