r"""Temporal post-filtering of the continuous 3-class predictions with a Hidden
Markov Model used as a CAUSAL forward filter (deployable online).

WHY AN HMM, AND WHY CAUSAL FORWARD FILTERING
--------------------------------------------
Failure mode being fixed: *base-rate false alarms*. Only ~6 % of continuous
windows are truly MI, yet the memoryless classifier fires "MI" on ~18 % of REST
windows, so when it says MI it is right only ~15 % of the time. Isolated one-
window MI spikes inside long REST blocks are almost always false. Fixing this
REQUIRES exploiting the temporal structure (states persist in long blocks),
which per-window argmax ignores.

Alternatives considered:
  * Majority vote / mode filter: crude special case (fixed uniform window, no
    class priors, no asymmetric persistence). Kept only as a baseline to beat.
  * EMA of posteriors: smooths but does not model discrete state transitions,
    and lags.
  * Dwell/debounce ("N consecutive MI windows"): ad-hoc hard thresholds; an HMM
    subsumes them (a high self-transition IS a soft dwell requirement).
  * HMM: models the label sequence as a Markov chain (transition matrix A =
    temporal inertia) with the classifier as emissions -- the principled
    generalisation and the classic tool for continuous / asynchronous MI-BCI
    (Obermaier et al. 2001 IEEE TRE; Rezek & Roberts 2005; asynchronous-BCI line
    Mason & Birch 2000; Millan et al.). The transition prior directly penalises
    an isolated MI window surrounded by REST -> exactly our false-alarm problem.

We use the CAUSAL forward algorithm (belief at t uses only past+present), which
is what a real-time drone can compute. Viterbi (full-sequence, non-causal) is
reported only as an OFFLINE upper bound, not as a deployable result.

The HMM defines a FAMILY of operating points via a persistence-strength knob m
(scales temporal inertia). Pure maximum-likelihood A (m=1) over-smooths the rare
MI class, so -- exactly as with the gate threshold in combined_eval.py -- we
SELECT the operating point on HELD-OUT VALIDATION data (max MI-detection F1) and
apply it BLIND to test. The naive majority-vote window W is selected the same way
for a fair comparison.

LEAKAGE CONTROL
---------------
* Test posteriors come from the FROZEN flat 3-class model already evaluated on
  the TEST streams (this script only post-processes the saved `prob`).
* The HMM transition matrix, the emission prior, AND the operating point (m, W)
  are all estimated/selected on HELD-OUT VALIDATION subjects (disjoint from the
  classifier's training set and from the test streams). The frozen model is run
  on validation streams built from validation subjects only. No test label or
  test neural signal ever sets a parameter.

Usage:  python temporal_filter_eval.py   (run from route_b/continuous/, BCI_DATA set)
"""
from __future__ import annotations
import sys

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import config3 as C
from lib import continuous_gen as gen
from lib import crops3, ea_io, metrics3, model3
from lib.segments import load_or_build_subject_pool

from src.split import load_split  # noqa: E402
from src.utils import get_logger, save_json  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = get_logger()
OUT = C.EXPERIMENTS / "continuous_eval_temporal"
FLAT_DIR = C.EXPERIMENTS / "continuous_eval"
REST, LEFT, RIGHT = C.REST_ID, C.LEFT_ID, C.RIGHT_ID

N_VAL_SUBJECTS = 8          # held-out validation subjects for param/operating-point fitting
VAL_SEED = 11               # stream-construction seed for the validation set
EPS = 1e-6
M_GRID = np.round(np.geomspace(0.1, 20.0, 25), 4)     # HMM persistence-strength grid
W_GRID = (1, 3, 5, 7, 9, 11, 15, 21, 31)              # majority-vote window grid


# --------------------------------------------------------------------------- #
# filters
# --------------------------------------------------------------------------- #
def _rescale_persistence(A: np.ndarray, m: float) -> np.ndarray:
    """m>1 = stickier (more inertia); m<1 = looser. m=1 = maximum-likelihood A."""
    if m == 1.0:
        return A
    B = A.copy()
    for i in range(C.N_CLASSES):
        off = 1.0 - A[i, i]
        new_off = min(off / m, 0.999999) if off > 0 else 0.0
        scale = (new_off / off) if off > 0 else 0.0
        B[i] = A[i] * scale
        B[i, i] = 1.0 - new_off
    return B


def hmm_forward(P: np.ndarray, A: np.ndarray, prior: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Causal online filtering: label_t = argmax_s P(state_t=s | o_1..o_t)."""
    E = (P + EPS) / prior                    # scaled likelihood (posterior / prior)
    alpha = pi * E[0]
    alpha /= alpha.sum()
    out = np.empty(len(P), dtype=np.int64)
    out[0] = int(alpha.argmax())
    for t in range(1, len(P)):
        alpha = (alpha @ A) * E[t]
        alpha /= alpha.sum()
        out[t] = int(alpha.argmax())
    return out


def hmm_viterbi(P: np.ndarray, A: np.ndarray, prior: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Offline MAP path (non-causal) -- upper bound only, NOT deployable online."""
    logE = np.log((P + EPS) / prior)
    logA = np.log(A)
    T = len(P)
    d = np.log(pi) + logE[0]
    back = np.zeros((T, C.N_CLASSES), dtype=np.int64)
    for t in range(1, T):
        mm = d[:, None] + logA
        back[t] = mm.argmax(axis=0)
        d = mm.max(axis=0) + logE[t]
    out = np.empty(T, dtype=np.int64)
    out[-1] = int(d.argmax())
    for t in range(T - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return out


def majority_vote(labels: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return labels
    h = window // 2
    out = labels.copy()
    for i in range(len(labels)):
        lo, hi = max(0, i - h), min(len(labels), i + h + 1)
        out[i] = int(np.bincount(labels[lo:hi], minlength=C.N_CLASSES).argmax())
    return out


# --------------------------------------------------------------------------- #
# streams
# --------------------------------------------------------------------------- #
def _window_gt(sample_gt: np.ndarray) -> np.ndarray:
    bounds = crops3.iter_crop_bounds(sample_gt.shape[0])
    return np.array([int(np.bincount(sample_gt[s:e], minlength=C.N_CLASSES).argmax())
                     for s, e in bounds])


def build_val_streams(split, W3, model) -> Tuple[List[Dict], np.ndarray, Dict]:
    """Val streams built from validation subjects; run frozen model -> posteriors.
    Also returns the maximum-likelihood transition matrix from val ground truth."""
    val_subjects = sorted(split["splits"]["val"]["subjects"], key=lambda s: int(s[1:]))
    streams, counts, used = [], np.zeros((C.N_CLASSES, C.N_CLASSES)), []
    for subject in val_subjects:
        if len(used) >= N_VAL_SUBJECTS:
            break
        pool = load_or_build_subject_pool(subject, split)
        if not (pool.mi_left and pool.mi_right and pool.rest):
            continue
        used.append(subject)
        for ti in range(C.CONT_N_TRIALS_PER_SUBJECT):
            rng = np.random.default_rng([VAL_SEED, int(subject[1:]), ti, 3])
            trial = gen.build_continuous_trial(pool, ti, rng, seed=VAL_SEED)
            normalized = ea_io.apply_ea(W3, trial.signal)
            bounds = crops3.iter_crop_bounds(normalized.shape[1])
            X = np.stack([normalized[:, s:e][:, :, None] for s, e in bounds]).astype(np.float32)
            prob = model.predict(X, batch_size=512, verbose=0).astype(np.float64)
            wgt = _window_gt(trial.ground_truth)
            streams.append({"gt": wgt, "raw": prob.argmax(1).astype(np.int64), "prob": prob})
            for a, b in zip(wgt[:-1], wgt[1:]):
                counts[a, b] += 1.0
    A = (counts + EPS) / (counts.sum(axis=1, keepdims=True) + C.N_CLASSES * EPS)
    dwell = {C.CLASS_NAMES[i]: float(1.0 / (1.0 - A[i, i])) for i in range(C.N_CLASSES)}
    logger.info("Val transition matrix on %s | dwell(windows)=%s", used, dwell)
    return streams, A, {"val_subjects_used": used, "transition_matrix": A.tolist(),
                        "dwell_windows": dwell}


def load_test_streams(seed: int) -> List[Dict]:
    d = np.load(FLAT_DIR / f"seed_{seed}" / "predictions.npz", allow_pickle=True)
    subj = d["subject"].astype(str); tkey = d["trial_key"].astype(str)
    keys = np.array([f"{s}|{t}" for s, t in zip(subj, tkey)])
    streams = []
    for k in dict.fromkeys(keys.tolist()):
        idx = np.where(keys == k)[0]
        streams.append({"gt": d["gt"][idx].astype(np.int64),
                        "raw": d["pred"][idx].astype(np.int64),
                        "prob": d["prob"][idx].astype(np.float64)})
    return streams


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _mi_f1(gt: np.ndarray, pred: np.ndarray) -> float:
    mi_t, mi_p = gt != REST, pred != REST
    tp = (mi_t & mi_p).sum(); fp = (~mi_t & mi_p).sum(); fn = (mi_t & ~mi_p).sum()
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def full_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict:
    m = metrics3.compute_metrics(gt, pred)
    b = metrics3.bci_continuous_metrics(gt, pred)
    mi_t, mi_p = gt != REST, pred != REST
    tp = int((mi_t & mi_p).sum()); fp = int((~mi_t & mi_p).sum()); fn = int((mi_t & ~mi_p).sum())
    return {
        "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
        "precision_macro": m["precision_macro"], "recall_macro": m["recall_macro"],
        "f1_macro": m["f1_macro"], "per_class_accuracy": m["per_class_accuracy"],
        "fpr": b["false_positive_rate_rest_as_mi"], "fnr": b["false_negative_rate_mi_as_rest"],
        "rest_vs_mi_accuracy": b["rest_vs_mi_accuracy"],
        "mi_detection_precision": (tp / (tp + fp)) if (tp + fp) else float("nan"),
        "mi_detection_recall": (tp / (tp + fn)) if (tp + fn) else float("nan"),
        "mi_detection_f1": _mi_f1(gt, pred),
        "confusion_matrix": m["confusion_matrix"],
    }


def _apply(streams, method, A, prior, pi, m=1.0, W=5):
    Am = _rescale_persistence(A, m)
    gts, preds = [], []
    for s in streams:
        gts.append(s["gt"])
        if method == "raw":
            preds.append(s["raw"])
        elif method == "majority":
            preds.append(majority_vote(s["raw"], W))
        elif method == "hmm_causal":
            preds.append(hmm_forward(s["prob"], Am, prior, pi))
        elif method == "hmm_viterbi":
            preds.append(hmm_viterbi(s["prob"], Am, prior, pi))
    return np.concatenate(gts), np.concatenate(preds)


# --------------------------------------------------------------------------- #
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "temporal_filter_eval.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        split = load_split(C.SPLIT_JSON)
        crop_support = json.loads((C.EXPERIMENTS / "rest_3class" / "metrics.json").read_text()
                                  )["crop"]["support_per_class"]
        prior = np.array([crop_support["REST"], crop_support["LEFT"], crop_support["RIGHT"]], float)
        prior = prior / prior.sum()
        pi = np.array([0.90, 0.05, 0.05])

        # frozen flat 3-class model + its EA, to run on validation streams
        W3 = ea_io.load_ea_matrix(C.EXPERIMENTS / "rest_3class" / "ea_reference.json")
        model = model3.build_model3("eegsym")
        model.load_weights(C.EXPERIMENTS / "rest_3class" / "weights.weights.h5")

        val_streams, A, trans_info = build_val_streams(split, W3, model)

        # --- select operating point on VALIDATION (max MI-detection F1) --- #
        def val_curve(method, grid, arg):
            rows = []
            for g in grid:
                gt, pred = _apply(val_streams, method, A, prior, pi, **{arg: g})
                fm = full_metrics(gt, pred)
                rows.append({arg: float(g), "mi_f1": fm["mi_detection_f1"],
                             "mi_precision": fm["mi_detection_precision"],
                             "mi_recall": fm["mi_detection_recall"],
                             "balanced_accuracy": fm["balanced_accuracy"], "fpr": fm["fpr"]})
            return rows

        hmm_val = val_curve("hmm_causal", M_GRID, "m")
        maj_val = val_curve("majority", W_GRID, "W")
        m_star = max(hmm_val, key=lambda r: r["mi_f1"])["m"]
        W_star = max(maj_val, key=lambda r: r["mi_f1"])["W"]
        logger.info("Selected on val: HMM m*=%.3f | majority W*=%d", m_star, int(W_star))

        # --- apply BLIND to TEST (5 seeds) --- #
        seeds = C.CONT_SEEDS
        methods = {"raw": {}, "majority": {"W": int(W_star)},
                   "hmm_causal": {"m": m_star}, "hmm_viterbi": {"m": m_star}}
        per_seed = {k: [] for k in methods}
        pooled = {k: {"gt": [], "pred": []} for k in methods}
        for seed in seeds:
            ts = load_test_streams(seed)
            for name, kw in methods.items():
                gt, pred = _apply(ts, name, A, prior, pi, **kw)
                per_seed[name].append(full_metrics(gt, pred))
                pooled[name]["gt"].append(gt); pooled[name]["pred"].append(pred)
            logger.info("test seed %d done", seed)

        SCAL = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
                "fpr", "fnr", "rest_vs_mi_accuracy", "mi_detection_precision",
                "mi_detection_recall", "mi_detection_f1"]
        agg = {}
        for name in methods:
            rows = per_seed[name]
            gt_all = np.concatenate(pooled[name]["gt"]); pr_all = np.concatenate(pooled[name]["pred"])
            pm = full_metrics(gt_all, pr_all)
            agg[name] = {"mean": {k: float(np.mean([r[k] for r in rows])) for k in SCAL},
                         "std": {k: float(np.std([r[k] for r in rows], ddof=1)) for k in SCAL},
                         "pooled_per_class_accuracy": pm["per_class_accuracy"],
                         "pooled_confusion_matrix": pm["confusion_matrix"]}

        summary = {
            "design": "HMM causal forward filter; operating point (m,W) selected on held-out val",
            "seeds": list(seeds), "operating_point": {"hmm_m_star": m_star, "majority_W_star": int(W_star)},
            "emission_prior_train": prior.tolist(), "initial_distribution_pi": pi.tolist(),
            "hmm_parameters": trans_info,
            "val_selection": {"hmm_curve": hmm_val, "majority_curve": maj_val},
            "aggregate": agg,
        }
        save_json(OUT / "summary.json", summary)
        _write_report(agg, trans_info, m_star, int(W_star), hmm_val)
        _figures(agg, val_streams, A, prior, pi, m_star, hmm_val)

        r = agg["raw"]["mean"]; hc = agg["hmm_causal"]["mean"]
        logger.info("=== DONE in %.1fs | bal_acc %.3f->%.3f | FPR %.3f->%.3f | "
                    "MI-precision %.3f->%.3f | MI-recall %.3f->%.3f | MI-F1 %.3f->%.3f ===",
                    time.perf_counter() - t0, r["balanced_accuracy"], hc["balanced_accuracy"],
                    r["fpr"], hc["fpr"], r["mi_detection_precision"], hc["mi_detection_precision"],
                    r["mi_detection_recall"], hc["mi_detection_recall"],
                    r["mi_detection_f1"], hc["mi_detection_f1"])
    finally:
        logger.removeHandler(h)
        h.close()


NAME = {"raw": "No filter (argmax)", "majority": "Majority vote (W*)",
        "hmm_causal": "Causal HMM (deployable)", "hmm_viterbi": "HMM Viterbi (offline)"}


def _write_report(agg, trans_info, m_star, W_star, hmm_val):
    lines = ["# Temporal HMM filter over the continuous predictions (3 classes)", "",
             f"- Transition matrix (MLE) estimated on validation: {trans_info['val_subjects_used']}",
             f"- Dwell (0.5 s windows): " +
             ", ".join(f"{k}={v:.1f}" for k, v in trans_info["dwell_windows"].items()),
             f"- Operating point chosen on validation (max MI-detection F1): "
             f"HMM m*={m_star:.3f}, vote W*={W_star}", "",
             "## Global metrics on TEST (mean +/- std, 5 seeds)", "",
             "| Metric | " + " | ".join(NAME[m] for m in agg) + " |",
             "|---|" + "---|" * len(agg)]
    for key, lab in [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
                     ("precision_macro", "Precision (macro)"), ("recall_macro", "Recall (macro)"),
                     ("f1_macro", "F1 (macro)"), ("fpr", "FPR (rest->MI)"),
                     ("fnr", "FNR (MI->rest)"), ("rest_vs_mi_accuracy", "Acc REST-vs-MI"),
                     ("mi_detection_precision", "MI detection precision"),
                     ("mi_detection_recall", "MI detection recall"),
                     ("mi_detection_f1", "MI detection F1")]:
        lines.append(f"| {lab} | " +
                     " | ".join(f"{agg[m]['mean'][key]:.3f}+-{agg[m]['std'][key]:.3f}" for m in agg) + " |")
    lines += ["", "## Per-class accuracy (pooled)", "",
              "| Class | " + " | ".join(NAME[m] for m in agg) + " |", "|---|" + "---|" * len(agg)]
    for cls in C.CLASS_NAMES:
        lines.append(f"| {cls} | " +
                     " | ".join(f"{agg[m]['pooled_per_class_accuracy'][cls]:.3f}" for m in agg) + " |")
    lines += ["", "## Confusion matrices (pooled; rows=truth, cols=prediction)"]
    for m in agg:
        cm = agg[m]["pooled_confusion_matrix"]
        lines += ["", f"{NAME[m]}", "", "| | pred REST | pred LEFT | pred RIGHT |", "|---|---|---|---|"]
        for i, cls in enumerate(C.CLASS_NAMES):
            lines.append(f"| {cls} | {cm[i][0]} | {cm[i][1]} | {cm[i][2]} |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _figures(agg, val_streams, A, prior, pi, m_star, hmm_val):
    # confusion raw vs hmm
    for mth, tag in [("raw", "no filter"), ("hmm_causal", "causal HMM")]:
        cm = np.array(agg[mth]["pooled_confusion_matrix"], float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(4.6, 4))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set(xticks=range(3), yticks=range(3), xticklabels=C.CLASS_NAMES,
               yticklabels=C.CLASS_NAMES, xlabel="predicted", ylabel="truth",
               title=f"Confusion ({tag})")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=10)
        fig.colorbar(im, fraction=0.046); fig.tight_layout()
        fig.savefig(OUT / f"confusion_{mth}.png", dpi=130); plt.close(fig)

    # FPR & MI-precision & MI-recall bars
    methods = ["raw", "majority", "hmm_causal", "hmm_viterbi"]
    labs = ["no filter", "vote W*", "causal HMM", "HMM Viterbi"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, key, title in [(axes[0], "fpr", "FPR (false activations)"),
                           (axes[1], "mi_detection_precision", "MI detection precision"),
                           (axes[2], "mi_detection_recall", "MI detection recall")]:
        vals = [agg[m]["mean"][key] for m in methods]
        ax.bar(range(4), vals, color=["#8a8f98", "#c0873a", "#0e7c74", "#5aa9a0"])
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
        ax.set(xticks=range(4), title=title, ylim=(0, max(vals) * 1.3 + 0.05))
        ax.set_xticklabels(labs, rotation=15); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fpr_precision_recall.png", dpi=130); plt.close(fig)

    # validation selection curve (precision/recall/F1 vs m)
    ms = [r["m"] for r in hmm_val]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, lab in [("mi_f1", "F1 MI"), ("mi_precision", "Precision MI"),
                     ("mi_recall", "Recall MI"), ("fpr", "FPR")]:
        ax.plot(ms, [r[key] for r in hmm_val], "-o", ms=3, label=lab)
    ax.axvline(m_star, color="grey", ls="--", lw=0.8, label=f"m*={m_star:.2f}")
    ax.set(xscale="log", xlabel="persistence strength m", ylabel="value",
           title="Operating-point selection on validation")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "val_selection_curve.png", dpi=130); plt.close(fig)

    # timeline example (val stream 0)
    s = val_streams[0]
    gt = s["gt"]; raw = s["raw"]; hmm = hmm_forward(s["prob"], _rescale_persistence(A, m_star), prior, pi)
    t = np.arange(len(gt)) * (C.CROP_STRIDE_SAMPLES / C.FS_TARGET)
    fig, axes = plt.subplots(3, 1, figsize=(11, 5), sharex=True)
    for ax, seq, nm in [(axes[0], gt, "Truth"), (axes[1], raw, "No filter"), (axes[2], hmm, "Causal HMM")]:
        ax.step(t, seq, where="post", lw=1.0)
        ax.set(ylabel=nm, yticks=[0, 1, 2], yticklabels=["REST", "LEFT", "RIGHT"], ylim=(-0.3, 2.3))
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Example trial: truth vs. raw vs. causal HMM")
    fig.tight_layout(); fig.savefig(OUT / "timeline_example.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
