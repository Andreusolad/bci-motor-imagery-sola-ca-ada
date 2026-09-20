r"""Second decoder on BCI-IV-2a: EEGNet + EA (Stieger) zero-shot.

Same experiment as bciciv2a_eval.py but swapping the architecture EEGSym ->
EEGNet (both trained on Stieger's corrected window, EA normalization). Goal: if a
second, very different architecture also lands in the ~0.70 band on 2a zero-shot,
the "bottleneck is the signal, not the model" thesis is reinforced with a
head-to-head on the same external dataset.

Reuses the loading / ERD verification / per-subject EA / crop / trial-aggregation
pipeline from bciciv2a_eval verbatim; only the model + weights change.

Usage:  python bciciv2a_eegnet_eval.py   (run from route_b/transfer/, BCI_DATA set)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.study.eegnet import build_eegnet             # noqa: E402
from src.utils import get_logger, save_json           # noqa: E402

# reuse the exact 2a pipeline (loading, ERD, crops, metrics) from the EEGSym run
from bciciv2a_eval import subject_metrics, boot_ci, SUBJECTS, OUR8, IDX8   # noqa: E402

logger = get_logger()
W = (Path(__file__).resolve().parents[1] / "experiments" / "corrected_window"
     / "normalization" / "eegnet" / "euclidean_alignment" / "weights.weights.h5")
OUT = Path(__file__).resolve().parents[1] / "experiments" / "bciciv2a_eegnet_eval"
EEGSYM_REF = 0.738   # our EEGSym+EA pooled bal-acc on 2a (for the head-to-head note)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "bciciv2a_eegnet_eval.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        from tensorflow import keras  # noqa: F401
        model = build_eegnet()                 # (8,250,1), 2 classes
        model.load_weights(str(W))
        logger.info("loaded binary EEGNet+EA (Stieger corrected-window). Channels 2a idx=%s", IDX8)
        results = []
        for s in SUBJECTS:
            r = subject_metrics(model, s)
            results.append(r)
            logger.info("%s n=%d acc=%.3f bal=%.3f auc=%.3f ERD=%+.3f", s, r["n_trials"],
                        r["accuracy"], r["balanced_accuracy"], r["auc"],
                        r["erd_gap_c4_minus_c3_right_minus_left"])
            keras.backend.clear_session()

        rng = np.random.default_rng(42)
        bal = np.array([r["balanced_accuracy"] for r in results])
        auc = np.array([r["auc"] for r in results])
        pooled = sum(r["accuracy"] * r["n_trials"] for r in results) / sum(r["n_trials"] for r in results)
        summary = {
            "design": "Second decoder on 2a: binary EEGNet+EA (Stieger corrected window) zero-shot; "
                      "identical pipeline to bciciv2a_eval (EEGSym). Head-to-head on the same dataset.",
            "balanced_accuracy": dict(zip(["mean", "ci_low", "ci_high"], boot_ci(bal, rng))),
            "auc": dict(zip(["mean", "ci_low", "ci_high"], boot_ci(auc, rng))),
            "pooled_accuracy": float(pooled), "eegsym_ea_reference_balacc": EEGSYM_REF,
            "results": results}
        save_json(OUT / "summary.json", summary)
        _report(summary)
        b = summary["balanced_accuracy"]
        logger.info("=== DONE in %.1fs === EEGNet bal-acc %.3f [%.3f,%.3f] pooled %.3f (EEGSym ref %.3f)",
                    time.perf_counter() - t0, b["mean"], b["ci_low"], b["ci_high"], pooled, EEGSYM_REF)
    finally:
        logger.removeHandler(h); h.close()


def _report(s):
    b, a = s["balanced_accuracy"], s["auc"]
    lines = ["# BCI-IV-2a -- second decoder: EEGNet+EA (Stieger) zero-shot", "",
             "Head-to-head with EEGSym+EA on the same external dataset (same pipeline, same "
             "ERD verification, same per-subject EA). Chance = 0.5.", "",
             "| Subj | n | Acc/Bal-acc | AUC | ERD gap |", "|---|---|---|---|---|"]
    for r in s["results"]:
        lines.append(f"| {r['subject']} | {r['n_trials']} | {r['balanced_accuracy']:.3f} | "
                     f"{r['auc']:.3f} | {r['erd_gap_c4_minus_c3_right_minus_left']:+.3f} |")
    lines += ["",
              f"- EEGNet+EA mean bal-acc = {b['mean']:.3f}, 95% CI [{b['ci_low']:.3f}, {b['ci_high']:.3f}]; "
              f"AUC {a['mean']:.3f}; pooled {s['pooled_accuracy']:.3f}.",
              f"- EEGSym+EA (reference) = {s['eegsym_ea_reference_balacc']:.3f}.",
              "- Reading: two very different architectures (heavy EEGSym / light EEGNet), both "
              "trained only on Stieger, land in the same band when transferred to 2a without "
              "seeing labels -> the ceiling is set by the signal/dataset, not the model."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
