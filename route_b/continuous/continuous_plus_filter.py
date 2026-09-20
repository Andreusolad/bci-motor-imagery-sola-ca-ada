r"""Combine the continuous-trained model with the temporal HMM filter.

Rationale (from the two prior findings):
  * The temporal HMM filter reduces false alarms but needs recall headroom; it
    failed on the trial-trained flat model (MI-recall 0.514 -> 0.084).
  * The continuous-trained model is a high-recall detector (MI-recall 0.775) but
    over-fires (FPR 0.424).
  => Their combination should let the filter suppress the isolated false alarms
     while keeping the abundant true MI. This script tests that on identical test
     streams, with a fair 2x2 matrix {flat, continuous} x {raw, +HMM}.

Leakage control: identical to temporal_filter_eval.py -- the HMM transition
matrix and each model's filter operating point (m, W) are estimated/selected on
held-out validation subjects (disjoint from training and from the test streams);
the frozen models only ever produce posteriors.

Usage:  python continuous_plus_filter.py   (run from route_b/continuous/, BCI_DATA set)
"""
from __future__ import annotations
import sys

import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import config3 as C
from lib import continuous_gen as gen
from lib import ea_io, model3
from lib.segments import load_or_build_subject_pool

import temporal_filter_eval as TF
from train_continuous import infer_stream

from src.split import load_split  # noqa: E402
from src.utils import get_logger, save_json  # noqa: E402

logger = get_logger()
OUT = C.EXPERIMENTS / "continuous_trained" / "with_temporal_filter"
CONT_DIR = C.EXPERIMENTS / "continuous_trained"
FLAT_DIR = C.EXPERIMENTS / "rest_3class"
PRIOR = np.array([1 / 3, 1 / 3, 1 / 3])          # both models trained ~balanced
PI = np.array([0.90, 0.05, 0.05])


def build_test_streams(model, W3, pools, test_subs, seed) -> List[Dict]:
    streams = []
    for subject in test_subs:
        for ti in range(C.CONT_N_TRIALS_PER_SUBJECT):
            rng = np.random.default_rng([seed, int(subject[1:]), ti])
            trial = gen.build_continuous_trial(pools[subject], ti, rng, seed=seed)
            gt, prob = infer_stream(model, W3, trial)
            streams.append({"gt": gt, "raw": prob.argmax(1).astype(np.int64), "prob": prob.astype(np.float64)})
    return streams


def select_op(val_streams, A) -> Dict:
    m_star = max(TF.M_GRID, key=lambda m: TF._mi_f1(*TF._apply(val_streams, "hmm_causal", A, PRIOR, PI, m=m)))
    W_star = int(max(TF.W_GRID, key=lambda W: TF._mi_f1(*TF._apply(val_streams, "majority", A, PRIOR, PI, W=W))))
    return {"m": float(m_star), "W": W_star}


def eval_model(tag, model, W3, split, pools, test_subs, A) -> Dict:
    # per-model operating point selected on val
    val_streams, _, _ = TF.build_val_streams(split, W3, model)
    op = select_op(val_streams, A)
    logger.info("[%s] operating point on val: m*=%.3f W*=%d", tag, op["m"], op["W"])

    conds = {"raw": {"method": "raw"}, "majority": {"method": "majority", "W": op["W"]},
             "hmm": {"method": "hmm_causal", "m": op["m"]}}
    per_seed = {c: [] for c in conds}
    pooled = {c: {"gt": [], "pred": []} for c in conds}
    for seed in C.CONT_SEEDS:
        streams = build_test_streams(model, W3, pools, test_subs, seed)
        for c, kw in conds.items():
            method = kw["method"]
            extra = {k: v for k, v in kw.items() if k != "method"}
            gt, pred = TF._apply(streams, method, A, PRIOR, PI, **extra)
            per_seed[c].append(TF.full_metrics(gt, pred))
            pooled[c]["gt"].append(gt); pooled[c]["pred"].append(pred)
        logger.info("[%s] test seed %d done", tag, seed)

    KEYS = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
            "fpr", "fnr", "mi_detection_precision", "mi_detection_recall", "mi_detection_f1"]
    out = {"operating_point": op}
    for c in conds:
        rows = per_seed[c]
        gt_all = np.concatenate(pooled[c]["gt"]); pr_all = np.concatenate(pooled[c]["pred"])
        pm = TF.full_metrics(gt_all, pr_all)
        out[c] = {"mean": {k: float(np.mean([r[k] for r in rows])) for k in KEYS},
                  "std": {k: float(np.std([r[k] for r in rows], ddof=1)) for k in KEYS},
                  "pooled_per_class_accuracy": pm["per_class_accuracy"],
                  "pooled_confusion_matrix": pm["confusion_matrix"]}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "continuous_plus_filter.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        split = load_split(C.SPLIT_JSON)
        test_subs = gen.select_subjects(split)
        pools = {s: load_or_build_subject_pool(s, split) for s in test_subs}

        # transition matrix (model-agnostic, from val GT) via the flat model call
        flat = model3.build_model3("eegsym"); flat.load_weights(FLAT_DIR / "weights.weights.h5")
        Wflat = ea_io.load_ea_matrix(FLAT_DIR / "ea_reference.json")
        _, A, trans_info = TF.build_val_streams(split, Wflat, flat)
        logger.info("transition dwell: %s", trans_info["dwell_windows"])

        cont = model3.build_model3("eegsym"); cont.load_weights(CONT_DIR / "weights.weights.h5")
        Wcont = ea_io.load_ea_matrix(CONT_DIR / "ea_reference.json")

        results = {
            "flat": eval_model("flat", flat, Wflat, split, pools, test_subs, A),
            "continuous": eval_model("continuous", cont, Wcont, split, pools, test_subs, A),
        }
        summary = {"design": "2x2: {flat, continuous} model x {raw, majority, HMM} temporal filter, "
                             "identical test streams, per-model op-point selected on val",
                   "transition_dwell": trans_info["dwell_windows"], "results": results}
        save_json(OUT / "summary.json", summary)
        _report(results)
        logger.info("=== DONE in %.1fs ===", time.perf_counter() - t0)
    finally:
        logger.removeHandler(h)
        h.close()


def _report(results):
    KEYS = [("balanced_accuracy", "Balanced acc"), ("fpr", "FPR"), ("fnr", "FNR"),
            ("mi_detection_precision", "MI precision"), ("mi_detection_recall", "MI recall"),
            ("mi_detection_f1", "MI F1"), ("accuracy", "Accuracy")]
    cols = [("flat", "raw", "Flat raw"), ("flat", "hmm", "Flat+HMM"),
            ("continuous", "raw", "Continuous raw"), ("continuous", "majority", "Continuous+vote"),
            ("continuous", "hmm", "Continuous+HMM")]
    lines = ["# Continuous model + temporal filter (2x2 on identical streams)", "",
             f"- Flat op-point: {results['flat']['operating_point']}",
             f"- Continuous op-point: {results['continuous']['operating_point']}", "",
             "| Metric | " + " | ".join(c[2] for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for key, name in KEYS:
        row = " | ".join(f"{results[m][c]['mean'][key]:.3f}" for m, c, _ in cols)
        lines.append(f"| {name} | {row} |")
    lines += ["", "## Per-class accuracy (pooled)", "",
              "| Class | " + " | ".join(c[2] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for cls in C.CLASS_NAMES:
        row = " | ".join(f"{results[m][c]['pooled_per_class_accuracy'][cls]:.3f}" for m, c, _ in cols)
        lines.append(f"| {cls} | {row} |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
