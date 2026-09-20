"""Evaluation: crop-level metrics (kept) + trial-level metrics (primary).

Trial-level metrics are computed via :mod:`first_ml.src.study.trial_aggregation`
(mean of crop softmax probabilities, never majority voting) and are the
study's primary metrics, per the brief. Crop-level metrics are preserved
alongside them, matching what the existing project already reports.
"""
from __future__ import annotations

from pathlib import Path
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

from .. import config as base_config  # noqa: E402
from ..utils import get_logger, to_native  # noqa: E402
from .crops import StudyCropArrays, make_tf_dataset  # noqa: E402
from .trial_aggregation import TrialAggregation, aggregate_by_trial  # noqa: E402

logger = get_logger()

_POSITIVE_ID = base_config.LABEL_TO_ID["right"]


def _predict_crop_probs(model, crops: StudyCropArrays) -> np.ndarray:
    ds = make_tf_dataset(crops, batch_size=512, shuffle=False, seed=0)
    return model.predict(ds, verbose=0)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(base_config.N_CLASSES)))
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=_POSITIVE_ID)
    roc_auc = float(auc(fpr, tpr))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_per_class": precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "confusion_matrix": cm.tolist(),
        "roc_auc": roc_auc,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "n_samples": int(len(y_true)),
    }


def evaluate_experiment(model, test_crops: StudyCropArrays, output_dir: Path) -> Dict[str, object]:
    """Compute crop-level and trial-level metrics; save figures + report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_probs = _predict_crop_probs(model, test_crops)
    crop_y_true = test_crops.y
    crop_y_pred = crop_probs.argmax(axis=1)
    crop_y_score = crop_probs[:, _POSITIVE_ID]
    crop_metrics = _classification_metrics(crop_y_true, crop_y_pred, crop_y_score)

    trial_agg: TrialAggregation = aggregate_by_trial(crop_probs, test_crops.trial_ids, crop_y_true)
    trial_y_true = trial_agg.y_true
    trial_y_pred = trial_agg.y_pred
    trial_y_score = trial_agg.y_prob[:, _POSITIVE_ID]
    trial_metrics = _classification_metrics(trial_y_true, trial_y_pred, trial_y_score)
    trial_metrics["n_trials"] = len(trial_agg)
    trial_metrics["mean_crops_per_trial"] = float(np.mean(trial_agg.n_crops_per_trial))

    logger.info(
        "CROP  accuracy=%.4f f1=%.4f AUC=%.4f | TRIAL accuracy=%.4f f1=%.4f AUC=%.4f",
        crop_metrics["accuracy"], crop_metrics["f1_macro"], crop_metrics["roc_auc"],
        trial_metrics["accuracy"], trial_metrics["f1_macro"], trial_metrics["roc_auc"],
    )

    _plot_confusion_matrix(np.asarray(trial_metrics["confusion_matrix"]), output_dir)
    _plot_roc(
        np.asarray(trial_metrics["roc_curve"]["fpr"]),
        np.asarray(trial_metrics["roc_curve"]["tpr"]),
        trial_metrics["roc_auc"],
        output_dir,
    )
    _write_classification_report(
        output_dir, crop_y_true, crop_y_pred, trial_y_true, trial_y_pred
    )

    return {"crop": to_native(crop_metrics), "trial": to_native(trial_metrics)}


def plot_training_curves(history: Dict[str, List[float]], output_dir: Path) -> None:
    """Plot accuracy and loss (train vs val) to ``training_curves.png``."""
    output_dir.mkdir(parents=True, exist_ok=True)
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
    fig.savefig(output_dir / "training_curves.png", dpi=120)
    plt.close(fig)
    logger.info("Saved %s", output_dir / "training_curves.png")


def _plot_confusion_matrix(cm: np.ndarray, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ticks = range(base_config.N_CLASSES)
    ax.set(
        xticks=ticks, yticks=ticks,
        xticklabels=base_config.CLASS_NAMES, yticklabels=base_config.CLASS_NAMES,
        xlabel="predicted", ylabel="true", title="Confusion matrix (test trials)",
    )
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(base_config.N_CLASSES):
        for j in range(base_config.N_CLASSES):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    logger.info("Saved %s", output_dir / "confusion_matrix.png")


def _plot_roc(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.6)
    ax.set(
        xlim=(0, 1), ylim=(0, 1.02),
        xlabel="False positive rate", ylabel="True positive rate",
        title=f"ROC (test trials) -- positive class: {base_config.ID_TO_LABEL[_POSITIVE_ID]}",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=120)
    plt.close(fig)
    logger.info("Saved %s", output_dir / "roc_curve.png")


def _write_classification_report(
    output_dir: Path,
    crop_y_true: np.ndarray, crop_y_pred: np.ndarray,
    trial_y_true: np.ndarray, trial_y_pred: np.ndarray,
) -> None:
    lines = [
        "=" * 70,
        "TRIAL-LEVEL classification report (PRIMARY METRICS)",
        "=" * 70,
        classification_report(
            trial_y_true, trial_y_pred, target_names=base_config.CLASS_NAMES, zero_division=0
        ),
        "",
        "=" * 70,
        "CROP-LEVEL classification report (secondary / reference metrics)",
        "=" * 70,
        classification_report(
            crop_y_true, crop_y_pred, target_names=base_config.CLASS_NAMES, zero_division=0
        ),
    ]
    (output_dir / "classification_report.txt").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved %s", output_dir / "classification_report.txt")
