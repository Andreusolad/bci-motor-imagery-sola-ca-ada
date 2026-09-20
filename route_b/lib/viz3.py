"""Plotting helpers (3-class aware). All figures are built from real arrays."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config3 as C  # noqa: E402

_CLASS_COLORS = {0: "#9aa0a6", 1: "#1a73e8", 2: "#e8710a"}  # REST grey, LEFT blue, RIGHT orange


def plot_confusion_matrix(cm: np.ndarray, path: Path,
                          names: Sequence[str] = C.CLASS_NAMES,
                          title: str = "Confusion matrix",
                          normalize: bool = False) -> None:
    cm = np.asarray(cm, dtype=float)
    disp = cm.copy()
    if normalize:
        rs = cm.sum(axis=1, keepdims=True)
        disp = np.divide(cm, rs, out=np.zeros_like(cm), where=rs > 0)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=disp.max() if disp.max() else 1)
    fig.colorbar(im, ax=ax)
    ticks = range(len(names))
    ax.set(xticks=ticks, yticks=ticks, xticklabels=names, yticklabels=names,
           xlabel="predicted", ylabel="true", title=title)
    thr = disp.max() / 2 if disp.max() else 0
    for i in range(len(names)):
        for j in range(len(names)):
            txt = f"{disp[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if disp[i, j] > thr else "black", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_training_curves(history: Dict[str, List[float]], path: Path) -> None:
    epochs = range(1, len(history["loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(epochs, history["accuracy"], label="train")
    a1.plot(epochs, history["val_accuracy"], label="val")
    a1.set(title="Accuracy", xlabel="epoch", ylabel="accuracy"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(epochs, history["loss"], label="train")
    a2.plot(epochs, history["val_loss"], label="val")
    a2.set(title="Loss", xlabel="epoch", ylabel="loss"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_timeline(gt: np.ndarray, pred: np.ndarray, times_s: np.ndarray,
                  path: Path, title: str) -> None:
    """Ground-truth vs prediction over time, with correct/incorrect shading."""
    fig, (ax_gt, ax_pr, ax_ok) = plt.subplots(
        3, 1, figsize=(13, 4.4), sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 1]})

    def _step(ax, series, label):
        for cls in range(C.N_CLASSES):
            mask = series == cls
            ax.fill_between(times_s, cls - 0.4, cls + 0.4, where=mask,
                            color=_CLASS_COLORS[cls], step="mid")
        ax.set(yticks=range(C.N_CLASSES), yticklabels=C.CLASS_NAMES,
               ylabel=label, ylim=(-0.6, C.N_CLASSES - 0.4))
        ax.grid(alpha=0.2, axis="x")

    _step(ax_gt, gt, "Ground truth")
    _step(ax_pr, pred, "Prediction")

    correct = gt == pred
    ax_ok.fill_between(times_s, 0, 1, where=correct, color="#188038", step="mid", label="correct")
    ax_ok.fill_between(times_s, 0, 1, where=~correct, color="#d93025", step="mid", label="incorrect")
    ax_ok.set(yticks=[], ylabel="match", xlabel="time (s)", ylim=(0, 1))
    ax_ok.legend(loc="upper right", ncol=2, fontsize=8, framealpha=0.9)

    ax_gt.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_prob_histograms(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> None:
    """Predicted-probability distribution of each class, split by true class."""
    fig, axes = plt.subplots(1, C.N_CLASSES, figsize=(13, 3.8), sharey=True)
    bins = np.linspace(0, 1, 26)
    for cls in range(C.N_CLASSES):
        ax = axes[cls]
        for true_cls in range(C.N_CLASSES):
            vals = y_prob[np.asarray(y_true) == true_cls, cls]
            ax.hist(vals, bins=bins, alpha=0.55, color=_CLASS_COLORS[true_cls],
                    label=f"true {C.CLASS_NAMES[true_cls]}")
        ax.set(title=f"P({C.CLASS_NAMES[cls]})", xlabel="probability")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_bars(labels: Sequence[str], values: Sequence[float], path: Path,
              title: str, ylabel: str, err: Optional[Sequence[float]] = None,
              hline: Optional[float] = None) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.7), 4.5))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=err, capsize=3, color="#0e7c74")
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    if hline is not None:
        ax.axhline(hline, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set(title=title, ylabel=ylabel, xticks=x, ylim=(0, 1.02))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
