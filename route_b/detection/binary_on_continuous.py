r"""Evaluate the dedicated binary REST-vs-MI model on the 5-minute continuous
streams, and compare it head-to-head with the 3-class model's REST-vs-MI gate
(P_LEFT+P_RIGHT) on identical streams.

Question: is a detector trained specifically for REST-vs-MI a better gate than
the incidental gate of the 3-class model, in the real (94 % REST) deployment
scenario? The threshold-free ROC-AUC / PR-AUC answer separability; FPR/FNR at a
validation-selected threshold give a deployable operating point.

Leakage control: both models are frozen; each EA is train-fit; the binary
model's decision threshold is selected on held-out validation streams and applied
blind to test. Test streams use the same rng as continuous_eval.

Usage:  python binary_on_continuous.py   (run from route_b/detection/, BCI_DATA set)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import config3 as C                          # noqa: E402
from lib import continuous_gen as gen                 # noqa: E402
from lib import crops3, ea_io, model3                 # noqa: E402
from lib.segments import load_or_build_subject_pool   # noqa: E402

from src.eegsym_study.eegsym import build_eegsym       # noqa: E402
from src.split import load_split                       # noqa: E402
from src.utils import get_logger, save_json            # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve  # noqa: E402

import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402

logger = get_logger()
OUT = Path(__file__).resolve().parents[1] / "experiments" / "binary_on_continuous"
BIN_DIR = Path(__file__).resolve().parents[1] / "experiments" / "binary_rest_mi"
FLAT_DIR = C.EXPERIMENTS / "rest_3class"
REST = C.REST_ID


def p_mi_stream(model, W, trial, is_binary):
    norm = ea_io.apply_ea(W, trial.signal)
    bounds = crops3.iter_crop_bounds(norm.shape[1])
    X = np.stack([norm[:, s:e][:, :, None] for s, e in bounds]).astype(np.float32)
    prob = model.predict(X, batch_size=512, verbose=0)
    p_mi = prob[:, 1] if is_binary else (prob[:, C.LEFT_ID] + prob[:, C.RIGHT_ID])
    gt = np.array([int(np.bincount(trial.ground_truth[s:e], minlength=C.N_CLASSES).argmax()) for s, e in bounds])
    return p_mi, (gt != REST).astype(int)


def collect(model, W, pools, subjects, seeds, is_binary):
    P, Y = [], []
    for seed in seeds:
        for s in subjects:
            for ti in range(C.CONT_N_TRIALS_PER_SUBJECT):
                rng = np.random.default_rng([seed, int(s[1:]), ti])
                trial = gen.build_continuous_trial(pools[s], ti, rng, seed=seed)
                p, y = p_mi_stream(model, W, trial, is_binary)
                P.append(p); Y.append(y)
    return np.concatenate(P), np.concatenate(Y)


def at_threshold(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    rec_mi = tp / (tp + fn) if (tp + fn) else 0.0
    rec_rest = tn / (tn + fp) if (tn + fp) else 0.0
    return {"threshold": float(t), "fpr": 1 - rec_rest, "fnr": 1 - rec_mi,
            "recall_REST": rec_rest, "recall_MI": rec_mi,
            "precision_MI": tp / (tp + fp) if (tp + fp) else 0.0,
            "balanced_accuracy": 0.5 * (rec_mi + rec_rest),
            "confusion_matrix": [[tn, fp], [fn, tp]]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "binary_on_continuous.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        split = load_split(C.SPLIT_JSON)
        binm = build_eegsym(input_shape=C.INPUT_SHAPE, n_classes=2)
        binm.load_weights(BIN_DIR / "weights.weights.h5")
        Wb = ea_io.load_ea_matrix(BIN_DIR / "ea_reference.json")
        flat = model3.build_model3("eegsym"); flat.load_weights(FLAT_DIR / "weights.weights.h5")
        Wf = ea_io.load_ea_matrix(FLAT_DIR / "ea_reference.json")

        test_subs = gen.select_subjects(split)
        pools = {s: load_or_build_subject_pool(s, split) for s in test_subs}

        # threshold for the binary model selected on VALIDATION streams
        val_subs = sorted(split["splits"]["val"]["subjects"], key=lambda s: int(s[1:]))[:8]
        vpools = {s: load_or_build_subject_pool(s, split) for s in val_subs}
        pv, yv = collect(binm, Wb, vpools, val_subs, [11], True)
        grid = np.linspace(0.1, 0.9, 81)
        t_star = float(grid[np.argmax([at_threshold(yv, pv, t)["balanced_accuracy"] for t in grid])])
        # threshold that matches the 3-class gate's known continuous FPR (0.179), for a fair FNR compare
        logger.info("val-selected binary threshold t*=%.3f", t_star)

        # test (5 seeds)
        pb, yb = collect(binm, Wb, pools, test_subs, C.CONT_SEEDS, True)
        pf, yf = collect(flat, Wf, pools, test_subs, C.CONT_SEEDS, False)
        assert np.array_equal(yb, yf)                    # identical streams -> identical GT

        # binary threshold matched to the 3-class FPR at 0.5
        flat_05 = at_threshold(yf, pf, 0.5)
        # find binary threshold giving FPR ~ flat_05 fpr
        fprs = np.array([at_threshold(yb, pb, t)["fpr"] for t in grid])
        t_match = float(grid[np.argmin(np.abs(fprs - flat_05["fpr"]))])

        results = {
            "design": "dedicated binary REST-vs-MI vs 3-class gate on identical 5-min streams",
            "n_windows": int(len(yb)), "mi_prevalence": float(yb.mean()),
            "binary": {
                "roc_auc": float(roc_auc_score(yb, pb)), "pr_auc": float(average_precision_score(yb, pb)),
                "at_0.5": at_threshold(yb, pb, 0.5),
                "at_val_selected": at_threshold(yb, pb, t_star),
                "at_matched_flat_fpr": at_threshold(yb, pb, t_match),
            },
            "flat_3class_gate": {
                "roc_auc": float(roc_auc_score(yf, pf)), "pr_auc": float(average_precision_score(yf, pf)),
                "at_0.5": flat_05,
            },
        }
        save_json(OUT / "summary.json", results)
        _report(results)
        _fig(yb, pb, yf, pf)
        b, f = results["binary"], results["flat_3class_gate"]
        logger.info("=== DONE in %.1fs | ROC-AUC binary %.3f vs 3class %.3f | "
                    "binary@0.5 FPR %.3f FNR %.3f | 3class@0.5 FPR %.3f FNR %.3f ===",
                    time.perf_counter() - t0, b["roc_auc"], f["roc_auc"],
                    b["at_0.5"]["fpr"], b["at_0.5"]["fnr"], f["at_0.5"]["fpr"], f["at_0.5"]["fnr"])
    finally:
        logger.removeHandler(h)
        h.close()


def _report(r):
    b, f = r["binary"], r["flat_3class_gate"]
    lines = ["# Dedicated binary model vs. 3-class gate, on the 5-min continuous streams", "",
             f"{r['n_windows']} windows | MI prevalence = {r['mi_prevalence']:.2f} "
             f"(rest {1-r['mi_prevalence']:.0%})", "",
             "## Separability (threshold-independent)", "",
             "| | Dedicated binary | 3-class gate |", "|---|---|---|",
             f"| ROC-AUC | {b['roc_auc']:.3f} | {f['roc_auc']:.3f} |",
             f"| PR-AUC | {b['pr_auc']:.3f} | {f['pr_auc']:.3f} |", "",
             "## Operating point (FPR = false alarms, FNR = missed MI)", "",
             "| Config | FPR | FNR | MI recall | Balanced acc |", "|---|---|---|---|---|"]
    for name, m in [("Binary @0.5", b["at_0.5"]), ("Binary @val-sel", b["at_val_selected"]),
                    ("Binary @matched-FPR", b["at_matched_flat_fpr"]),
                    ("3-class @0.5", f["at_0.5"])]:
        lines.append(f"| {name} | {m['fpr']:.3f} | {m['fnr']:.3f} | {m['recall_MI']:.3f} | "
                     f"{m['balanced_accuracy']:.3f} |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fig(yb, pb, yf, pf):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for y, p, name in [(yb, pb, "Dedicated binary"), (yf, pf, "3-class gate")]:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y,p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="FPR (rest->MI)", ylabel="TPR (MI detected)",
           title="REST-vs-MI continuous: binary vs 3-class")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "roc_comparison.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
