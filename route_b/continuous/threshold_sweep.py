r"""Gate-threshold sweep for the hierarchical model on the continuous streams.

The hierarchical decision is: predict MI iff P(LEFT)+P(RIGHT) > tau (the gate),
and on those windows the 2-class winner picks the side. Because the L/R decision
does NOT depend on tau, one pass over the streams (collecting per-window
p_mi, the L/R side, and the ground truth) lets us sweep tau entirely offline and
trace the full FPR<->FNR operating curve WITHOUT re-running the models per tau.

Answers: can we keep the hierarchical's good, balanced L/R AND recover a low
false-positive rate (idle rejection) by moving the gate threshold?

Usage (after hierarchical_eval.py, which caches the 2-class EA):
    python threshold_sweep.py
"""
from __future__ import annotations
import sys

import logging
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import config3 as C
from lib import continuous_gen as gen
from lib import ea_io, metrics3, viz3
from lib.segments import load_or_build_subject_pool

import hierarchical_eval as H

from src.eegsym_study.eegsym import build_eegsym  # noqa: E402
from src.split import load_split  # noqa: E402
from src.utils import get_logger, save_json  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = get_logger()
OUT = C.EXPERIMENTS / "continuous_eval_hierarchical" / "threshold_sweep"


def collect() -> dict:
    """One pass over all streams: gather per-window p_mi, L/R side, ground truth."""
    W2 = H.ensure_2class_ea()
    W3 = ea_io.load_ea_matrix(H.GATE_DIR / "ea_reference.json")
    gate = build_eegsym(input_shape=C.INPUT_SHAPE, n_classes=3)
    gate.load_weights(H.GATE_DIR / "weights.weights.h5")
    lr = build_eegsym(input_shape=C.INPUT_SHAPE, n_classes=2)
    lr.load_weights(H.LR_DIR / "weights.weights.h5")

    split = load_split(C.SPLIT_JSON)
    subjects = gen.select_subjects(split)
    pools = {s: load_or_build_subject_pool(s, split) for s in subjects}

    p_mi_all, lr_side_all, gt_all = [], [], []
    for seed in C.CONT_SEEDS:
        for subject in subjects:
            for ti in range(C.CONT_N_TRIALS_PER_SUBJECT):
                rng = np.random.default_rng([seed, int(subject[1:]), ti])
                trial = gen.build_continuous_trial(pools[subject], ti, rng, seed=seed)
                _, gate_prob, lr_prob = H.hierarchical_predict(gate, lr, W3, W2, trial.signal)
                p_mi_all.append(gate_prob[:, C.LEFT_ID] + gate_prob[:, C.RIGHT_ID])
                lr_side_all.append(np.where(lr_prob.argmax(1) == 0, C.LEFT_ID, C.RIGHT_ID))
                gt_all.append(H._window_gt(trial.ground_truth))
        logger.info("collected seed %d", seed)
    return {"p_mi": np.concatenate(p_mi_all),
            "lr_side": np.concatenate(lr_side_all),
            "gt": np.concatenate(gt_all)}


def sweep(data: dict) -> list:
    p_mi, lr_side, gt = data["p_mi"], data["lr_side"], data["gt"]
    rows = []
    for tau in np.round(np.linspace(0.30, 0.95, 66), 4):
        pred = np.where(p_mi > tau, lr_side, C.REST_ID)
        m = metrics3.compute_metrics(gt, pred)
        b = metrics3.bci_continuous_metrics(gt, pred)
        rows.append({
            "tau": float(tau), "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"], "f1_macro": m["f1_macro"],
            "rest_recall": m["per_class_accuracy"]["REST"],
            "left_recall": m["per_class_accuracy"]["LEFT"],
            "right_recall": m["per_class_accuracy"]["RIGHT"],
            "fpr": b["false_positive_rate_rest_as_mi"],
            "fnr": b["false_negative_rate_mi_as_rest"],
            "rest_vs_mi_accuracy": b["rest_vs_mi_accuracy"],
        })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "threshold_sweep.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        data = collect()
        rows = sweep(data)
        taus = np.array([r["tau"] for r in rows])

        def at(tau):
            return rows[int(np.argmin(np.abs(taus - tau)))]

        best_bal = max(rows, key=lambda r: r["balanced_accuracy"])
        # lowest FNR among rows whose FPR <= 0.18 (match flat EEGSym idle rejection)
        low_fpr = [r for r in rows if r["fpr"] <= 0.18]
        fpr_target = min(low_fpr, key=lambda r: r["fnr"]) if low_fpr else None

        summary = {
            "design": "hierarchical gate threshold sweep (tau on P_MI), pooled over 5 seeds",
            "n_windows": int(len(data["gt"])),
            "operating_points": {
                "default_tau_0.50": at(0.50),
                "max_balanced_accuracy": best_bal,
                "fpr_le_0.18_min_fnr": fpr_target,
            },
            "curve": rows,
        }
        save_json(OUT / "threshold_sweep.json", summary)

        # --- figures --- #
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([r["fpr"] for r in rows], [1 - r["fnr"] for r in rows], "-o", ms=3)
        for r in (at(0.50), best_bal):
            ax.annotate(f"tau={r['tau']:.2f}", (r["fpr"], 1 - r["fnr"]), fontsize=8)
        ax.set(xlabel="False positive rate (idle -> MI)",
               ylabel="MI detection rate (1 - FNR)",
               title="Hierarchical gate: idle-rejection vs MI-detection trade-off")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / "operating_curve.png", dpi=130); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for key, lab in [("balanced_accuracy", "balanced acc"), ("fpr", "FPR"),
                         ("fnr", "FNR"), ("left_recall", "LEFT recall"),
                         ("rest_recall", "REST recall")]:
            ax.plot(taus, [r[key] for r in rows], label=lab)
        ax.axvline(0.5, color="grey", ls="--", lw=0.8, alpha=0.7)
        ax.set(xlabel="gate threshold tau (predict MI iff P_MI > tau)", ylabel="value",
               title="Metrics vs gate threshold")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / "metrics_vs_threshold.png", dpi=130); plt.close(fig)

        logger.info("default(0.5): bal_acc=%.3f FPR=%.3f FNR=%.3f LEFT=%.3f",
                    at(0.5)["balanced_accuracy"], at(0.5)["fpr"], at(0.5)["fnr"], at(0.5)["left_recall"])
        logger.info("max bal_acc: tau=%.2f bal_acc=%.3f FPR=%.3f LEFT=%.3f",
                    best_bal["tau"], best_bal["balanced_accuracy"], best_bal["fpr"], best_bal["left_recall"])
        if fpr_target:
            logger.info("FPR<=0.18 best: tau=%.2f FPR=%.3f FNR=%.3f LEFT=%.3f bal_acc=%.3f",
                        fpr_target["tau"], fpr_target["fpr"], fpr_target["fnr"],
                        fpr_target["left_recall"], fpr_target["balanced_accuracy"])
        logger.info("=== THRESHOLD SWEEP DONE in %.1fs -> %s ===", time.perf_counter() - t0, OUT)
    finally:
        logger.removeHandler(h)
        h.close()


if __name__ == "__main__":
    main()
