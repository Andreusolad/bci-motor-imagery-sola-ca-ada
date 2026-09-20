r"""Diagnostic experiment: can the model separate REST from MI at all?

Same winning pipeline as the 3-class REST model (EEGSym + Euclidean Alignment,
corrected window, same channels / filtering / downsampling / CAR / crops /
subject split / seed / architecture / hyper-parameters / training) -- the only
change is the labels:  REST = 0,  MI = 1  (LEFT and RIGHT merged).

LEFT vs RIGHT is not distinguished here. The goal is purely to locate the
bottleneck: can the model tell "there is imagery" from "rest"? We look beyond
the predicted class -- per-sample probabilities, ROC/PR curves, and the
penultimate-layer feature space (UMAP + t-SNE) -- to tell apart three
possibilities: the classes overlap intrinsically, the model is just using a bad
threshold, or the representation separates them cleanly.

No architecture/preprocessing/hyper-parameter changes. No improvements applied.

Usage:  python binary_rest_mi.py   (run from route_b/detection/, BCI_DATA set)
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import config3 as C                        # noqa: E402
from lib import crops3, ea_io                       # noqa: E402
from lib.segments import Segment, extract_segments_for_sessions  # noqa: E402

from src.eegsym_study.eegsym import build_eegsym     # noqa: E402
from src.split import load_split, session_keys_for   # noqa: E402
from src.utils import get_logger, save_json, set_global_seed  # noqa: E402

from sklearn.metrics import (                         # noqa: E402
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

logger = get_logger()
OUT = Path(__file__).resolve().parents[1] / "experiments" / "binary_rest_mi"
REST, MI = 0, 1
EMBED_MAX = 6000          # points for t-SNE / UMAP (subsampled for tractability)


# --------------------------------------------------------------------------- #
def _n_crops(sig) -> int:
    return len(crops3.iter_crop_bounds(sig.shape[1]))


def balance_binary(segments: List[Segment], seed: int) -> List[Segment]:
    """Keep all MI segments; subsample whole REST segments so REST crops ~= MI crops (1:1)."""
    mi = [s for s in segments if s.kind == "mi"]
    rest = [s for s in segments if s.kind == "rest"]
    n_mi_crops = sum(_n_crops(s.signal) for s in mi)
    n_keep = min(len(rest), max(1, round(n_mi_crops / 3)))   # each REST (2 s) = 3 crops
    rng = np.random.default_rng(seed)
    kept = [rest[i] for i in rng.permutation(len(rest))[:n_keep]]
    return mi + kept


def build_crops(segments: List[Segment], W: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, tids = [], [], []
    for s in segments:
        aligned = ea_io.apply_ea(W, s.signal)
        for a, b in crops3.iter_crop_bounds(aligned.shape[1]):
            X.append(aligned[:, a:b][:, :, None])
            y.append(REST if s.kind == "rest" else MI)
            tids.append(s.trial_id)
    return np.stack(X).astype(np.float32), np.asarray(y, np.int64), tids


def make_ds(X, y, batch, shuffle, seed):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((X, tf.one_hot(y, 2)))
    if shuffle:
        ds = ds.shuffle(min(len(X), 20000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def aggregate_by_trial(p_mi, tids, y):
    order, sums, cnt, yt = {}, {}, {}, {}
    for p, t, yy in zip(p_mi, tids, y):
        if t not in order:
            order[t] = len(order); sums[t] = 0.0; cnt[t] = 0
        sums[t] += p; cnt[t] += 1; yt[t] = yy
    ts = list(order)
    return (np.array([yt[t] for t in ts]),
            np.array([sums[t] / cnt[t] for t in ts]))


def binary_metrics(y, p_mi, tag):
    pred = (p_mi >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[REST, MI])
    return {
        "tag": tag, "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p_mi)),
        "pr_auc": float(average_precision_score(y, p_mi)),
        "recall_REST": float(recall_score(y, pred, pos_label=REST, zero_division=0)),
        "recall_MI": float(recall_score(y, pred, pos_label=MI, zero_division=0)),
        "precision_REST": float(precision_score(y, pred, pos_label=REST, zero_division=0)),
        "precision_MI": float(precision_score(y, pred, pos_label=MI, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


# --------------------------------------------------------------------------- #
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "binary_rest_mi.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        from tensorflow import keras
        set_global_seed(C.RANDOM_SEED)
        split = load_split(C.SPLIT_JSON)

        logger.info("Extracting segments per split ...")
        tr = balance_binary(extract_segments_for_sessions(session_keys_for(split, "train")), C.RANDOM_SEED)
        va = balance_binary(extract_segments_for_sessions(session_keys_for(split, "val")), C.RANDOM_SEED)
        te = balance_binary(extract_segments_for_sessions(session_keys_for(split, "test")), C.RANDOM_SEED)

        # EA on TRAIN only (fit on balanced train MI+REST) -- identical to the winner
        ea = ea_io.fit_ea([s.signal for s in tr])
        ea_io.save_ea(ea, OUT / "ea_reference.json")
        W = ea_io.load_ea_matrix(OUT / "ea_reference.json")

        Xtr, ytr, _ = build_crops(tr, W)
        Xva, yva, _ = build_crops(va, W)
        Xte, yte, tids = build_crops(te, W)
        logger.info("crops -> train %d (REST %d/MI %d) | val %d | test %d",
                    len(ytr), int((ytr == REST).sum()), int((ytr == MI).sum()), len(yva), len(yte))

        # --- same architecture / optimizer / loss / callbacks as the winner --- #
        from lib import model3
        model = build_eegsym(input_shape=C.INPUT_SHAPE, n_classes=2)
        model3.compile_model3(model)
        epoch_times: List[float] = []
        cbs = model3.build_callbacks(OUT, epoch_times)
        from src.eegsym_study import config as eg
        hist = model.fit(make_ds(Xtr, ytr, eg.TRAIN.batch_size, True, C.RANDOM_SEED),
                         validation_data=make_ds(Xva, yva, eg.TRAIN.batch_size, False, C.RANDOM_SEED),
                         epochs=eg.TRAIN.epochs, callbacks=cbs, verbose=0)
        model.save_weights(OUT / "weights.weights.h5")
        n_ep = len(hist.history["loss"])
        logger.info("trained %d epochs", n_ep)

        # --- predictions + probabilities on TEST --- #
        prob = model.predict(Xte, batch_size=512, verbose=0)     # (N, 2)
        p_mi = prob[:, MI]
        crop_m = binary_metrics(yte, p_mi, "crop")
        yt_trial, p_trial = aggregate_by_trial(p_mi, tids, yte)
        trial_m = binary_metrics(yt_trial, p_trial, "trial")

        np.savez_compressed(OUT / "test_probabilities.npz",
                            p_rest=prob[:, REST], p_mi=p_mi, y_true=yte,
                            trial_ids=np.array(tids))

        # --- penultimate-layer features = input to the final Dense classifier --- #
        dense = [l for l in model.layers if isinstance(l, keras.layers.Dense)]
        feat_model = keras.Model(model.input, dense[-1].input)
        feats = feat_model.predict(Xte, batch_size=512, verbose=0)
        feats = feats.reshape(len(feats), -1)
        np.savez_compressed(OUT / "test_features.npz", features=feats.astype(np.float32), y_true=yte)
        logger.info("penultimate features: %s", feats.shape)

        save_json(OUT / "metrics.json", {
            "design": "binary REST(0) vs MI(1=LEFT+RIGHT); winner pipeline, labels-only change",
            "epochs": n_ep, "n_params": int(model.count_params()),
            "crop": crop_m, "trial": trial_m,
        })
        _figures(yte, prob[:, REST], p_mi, feats)
        _report(crop_m, trial_m)
        logger.info("=== DONE in %.1fs | crop bal_acc %.3f roc_auc %.3f pr_auc %.3f | "
                    "REST recall %.3f MI recall %.3f ===",
                    time.perf_counter() - t0, crop_m["balanced_accuracy"], crop_m["roc_auc"],
                    crop_m["pr_auc"], crop_m["recall_REST"], crop_m["recall_MI"])
    finally:
        logger.removeHandler(h)
        h.close()


def _figures(y, p_rest, p_mi, feats):
    # (1) probability histogram
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(p_mi[y == REST], bins=40, alpha=0.6, label="true REST", color="#8a8f98", density=True)
    ax.hist(p_mi[y == MI], bins=40, alpha=0.6, label="true MI", color="#0e7c74", density=True)
    ax.axvline(0.5, color="k", ls="--", lw=0.8, label="threshold 0.5")
    ax.set(xlabel="P(MI)", ylabel="density", title="P(MI) probability histogram by true class")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "prob_histogram.png", dpi=130); plt.close(fig)

    # (2) ROC
    fpr, tpr, _ = roc_curve(y, p_mi); auc = roc_auc_score(y, p_mi)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}"); ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC curve (REST vs MI)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "roc_curve.png", dpi=130); plt.close(fig)

    # (3) PR
    prec, rec, _ = precision_recall_curve(y, p_mi); ap = average_precision_score(y, p_mi)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, lw=2, label=f"AP={ap:.3f}")
    ax.axhline((y == MI).mean(), color="grey", ls="--", lw=0.8, label="baseline (MI prevalence)")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "pr_curve.png", dpi=130); plt.close(fig)

    # (4) embeddings (subsampled)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(feats))[:EMBED_MAX]
    Xe, ye = feats[idx], y[idx]
    try:
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(Xe)
        from sklearn.manifold import TSNE
        ts = TSNE(n_components=2, init="pca", perplexity=30, random_state=42).fit_transform(Xs)
        _scatter(ts, ye, "t-SNE of the feature space (test)", OUT / "tsne.png")
    except Exception as e:  # noqa: BLE001
        logger.warning("t-SNE failed: %s", e)
    try:
        import umap
        um = umap.UMAP(n_components=2, random_state=42).fit_transform(Xe)
        _scatter(um, ye, "UMAP of the feature space (test)", OUT / "umap.png")
    except Exception as e:  # noqa: BLE001
        logger.warning("UMAP failed: %s", e)


def _scatter(emb, y, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for lab, name, col in [(REST, "REST", "#8a8f98"), (MI, "MI", "#0e7c74")]:
        m = y == lab
        ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.4, label=name, color=col)
    ax.set(title=title); ax.legend(markerscale=3); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _report(crop, trial):
    def block(m):
        cm = m["confusion_matrix"]
        return [f"### {m['tag']} level (n={m['n']})", "",
                "| Metric | Value |", "|---|---|",
                f"| Accuracy | {m['accuracy']:.3f} |",
                f"| Balanced accuracy | {m['balanced_accuracy']:.3f} |",
                f"| Precision (macro) | {m['precision_macro']:.3f} |",
                f"| Recall (macro) | {m['recall_macro']:.3f} |",
                f"| F1 (macro) | {m['f1_macro']:.3f} |",
                f"| ROC-AUC | {m['roc_auc']:.3f} |",
                f"| PR-AUC | {m['pr_auc']:.3f} |",
                f"| Recall REST | {m['recall_REST']:.3f} |",
                f"| Recall MI | {m['recall_MI']:.3f} |",
                f"| Precision REST | {m['precision_REST']:.3f} |",
                f"| Precision MI | {m['precision_MI']:.3f} |", "",
                "Confusion matrix (rows=truth, cols=prediction):", "",
                "| | pred REST | pred MI |", "|---|---|---|",
                f"| REST | {cm[0][0]} | {cm[0][1]} |",
                f"| MI | {cm[1][0]} | {cm[1][1]} |", ""]
    lines = ["# Binary diagnostic: REST vs MI (LEFT+RIGHT)", "",
             "Same winning pipeline (EEGSym+EA, corrected window); only the labels change.", ""]
    lines += block(crop) + block(trial)
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
