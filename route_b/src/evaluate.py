"""Evaluation: metrics, confusion matrix, ROC/AUC and all figures.

Uses the non-interactive Agg backend so it runs head-less. All numbers land in
``metrics.json`` and all plots in ``figures/``.
"""
from __future__ import annotations

from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from . import config  # noqa: E402
from .crops import CropArrays, make_tf_dataset  # noqa: E402
from .utils import get_logger, save_json, to_native  # noqa: E402

logger = get_logger()

_POSITIVE_ID = config.LABEL_TO_ID["right"]  # ROC "positive" class.


def _predict(model, crops: CropArrays) -> np.ndarray:
    """Return softmax probabilities for the crops."""
    ds = make_tf_dataset(crops, config.TRAIN.batch_size, shuffle=False)
    return model.predict(ds, verbose=0)


def evaluate_model(model, test_crops: CropArrays) -> Dict[str, object]:
    """Compute all metrics and figures on the held-out test crops."""
    config.ensure_output_dirs()
    probs = _predict(model, test_crops)
    y_true = test_crops.y
    y_pred = probs.argmax(axis=1)
    y_score = probs[:, _POSITIVE_ID]

    metrics: Dict[str, object] = {
        "n_test_crops": int(len(test_crops)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_per_class": precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "class_names": list(config.CLASS_NAMES),
    }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.N_CLASSES)))
    metrics["confusion_matrix"] = cm.tolist()
    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=config.CLASS_NAMES, zero_division=0, output_dict=True
    )

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=_POSITIVE_ID)
    roc_auc = float(auc(fpr, tpr))
    metrics["roc_auc"] = roc_auc

    save_json(config.METRICS_JSON, to_native(metrics))
    logger.info("Test accuracy=%.4f  f1_macro=%.4f  AUC=%.4f",
                metrics["accuracy"], metrics["f1_macro"], roc_auc)

    _plot_confusion_matrix(cm)
    _plot_roc(fpr, tpr, roc_auc)
    return metrics


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_training_curves(history: Dict[str, List[float]]) -> None:
    """Plot accuracy and loss (train vs val) to ``training_curves.png``."""
    config.ensure_output_dirs()
    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_acc.plot(epochs, history["accuracy"], label="train")
    ax_acc.plot(epochs, history["val_accuracy"], label="val")
    ax_acc.set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    ax_loss.plot(epochs, history["loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set(title="Loss", xlabel="epoch", ylabel="loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(config.TRAINING_CURVES_PNG, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", config.TRAINING_CURVES_PNG)


def _plot_confusion_matrix(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ticks = range(config.N_CLASSES)
    ax.set(
        xticks=ticks, yticks=ticks,
        xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
        xlabel="predicted", ylabel="true", title="Confusion matrix (test crops)",
    )
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(config.N_CLASSES):
        for j in range(config.N_CLASSES):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(config.CONFUSION_MATRIX_PNG, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", config.CONFUSION_MATRIX_PNG)


def _plot_roc(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.6)
    ax.set(
        xlim=(0, 1), ylim=(0, 1.02),
        xlabel="False positive rate", ylabel="True positive rate",
        title=f"ROC -- positive class: {config.ID_TO_LABEL[_POSITIVE_ID]}",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.ROC_CURVE_PNG, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", config.ROC_CURVE_PNG)
