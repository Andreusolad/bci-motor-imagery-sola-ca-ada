r"""Combined pipeline: per-subject calibration + tunable REST/MI gate + temporal
smoothing, evaluated leakage-free on the subject's own continuous streams.

Three stacked layers, each on a different axis:
  1. Calibration (weights)      -- fine-tune the frozen 3-class EEGSym+EA on the
                                   subject's OWN cal-train trials  (WHAT it decides).
  2. Gate threshold tau (rule)  -- predict MI iff P(LEFT)+P(RIGHT) > tau, else REST,
                                   L/R by argmax of the two              (WHEN it fires).
  3. Temporal smoothing (W)     -- majority vote over W consecutive 1 s windows,
                                   applied per trial                     (kills flicker).

LEAKAGE CONTROL (the whole point):
  * Each subject's trials are split by ORIGINAL trial (MI + its REST baseline stay
    together) into cal_train / cal_val / eval. Fine-tuning uses cal_train, early
    stopping uses cal_val, and the continuous EVAL streams use ONLY eval segments.
  * The gate threshold tau and the smoothing window W are chosen to maximise
    balanced accuracy on CAL-VAL streams (held out from weight fitting AND from
    eval) and then applied *blind* to the eval streams -- exactly what a real
    deployment does (calibrate the user, freeze their operating point, run).
  * The global 3-class EA reference is kept fixed (never recomputed on test data).

Reports an ablation on the SAME eval streams: baseline (global argmax) ->
calibrated (argmax) -> calibrated+gate -> calibrated+gate+smoothing (full).

Usage (after train_3class.py --arch eegsym has produced the base model):
    python combined_eval.py
"""
from __future__ import annotations
import sys

import gc
import logging
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import config3 as C
from lib import continuous_gen as gen
from lib import crops3, ea_io, metrics3, viz3
from lib.segments import Segment, SubjectPool, extract_segments_for_sessions

# Reuse the leakage-safe split + fine-tuning already validated in Step 2.
import calibration_eval as CAL
from calibration_eval import _split_subject, _pool_from_segments, _finetune, _pack

from src.split import load_split, session_keys_for  # noqa: E402
from src.utils import get_logger, save_json  # noqa: E402

logger = get_logger()
OUT = C.EXPERIMENTS / "continuous_eval_combined"
BASE_DIR = C.model_dir("eegsym", "euclidean_alignment")

EVAL_SEEDS = (42, 123, 256)          # continuous-stream construction seeds for eval
TUNE_SEEDS = (777,)                  # separate seed for the cal-val tuning streams
EVAL_SALT = 7                        # rng salt (matches calibration_eval eval streams)
TUNE_SALT = 99
TAUS = np.round(np.arange(0.30, 0.81, 0.05), 2)   # gate-threshold grid
WINS = (1, 3, 5, 7, 9)                            # smoothing windows (#1 s crops, odd)

VARIANTS = ("baseline", "calibrated", "cal_gate", "full")
AGG_FIELDS = ("balanced_accuracy", "left_recall", "right_recall",
              "rest_recall", "fpr", "fnr", "accuracy")


# --------------------------------------------------------------------------- #
# gate + smoothing (pure post-processing on window-level softmax)
# --------------------------------------------------------------------------- #
def _gate(probs: np.ndarray, tau: float) -> np.ndarray:
    """Predict MI iff P(LEFT)+P(RIGHT) > tau; the side is the larger of the two."""
    p_mi = probs[:, C.LEFT_ID] + probs[:, C.RIGHT_ID]
    lr = np.where(probs[:, C.LEFT_ID] >= probs[:, C.RIGHT_ID], C.LEFT_ID, C.RIGHT_ID)
    return np.where(p_mi > tau, lr, C.REST_ID).astype(np.int64)


def _smooth(pred: np.ndarray, window: int) -> np.ndarray:
    """Centered majority vote over `window` consecutive predictions (odd window)."""
    if window <= 1:
        return pred
    h = window // 2
    out = pred.copy()
    n = len(pred)
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        out[i] = np.bincount(pred[lo:hi], minlength=C.N_CLASSES).argmax()
    return out


# --------------------------------------------------------------------------- #
# continuous streams -> per-trial (ground truth, softmax per model)
# --------------------------------------------------------------------------- #
def _stream(models: Sequence, pool: SubjectPool, W3: np.ndarray,
            seeds: Sequence[int], salt: int) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
    """One entry per (seed, trial): (window ground-truth, [softmax per model]).

    The trial (hence the streams and their ground truth) is built ONCE, so every
    model is scored on identical windows -- an apples-to-apples ablation.
    """
    out: List[Tuple[np.ndarray, List[np.ndarray]]] = []
    for seed in seeds:
        for ti in range(C.CONT_N_TRIALS_PER_SUBJECT):
            rng = np.random.default_rng([seed, int(pool.subject[1:]), ti, salt])
            trial = gen.build_continuous_trial(pool, ti, rng, seed=seed)
            normalized = ea_io.apply_ea(W3, trial.signal)
            bounds = crops3.iter_crop_bounds(normalized.shape[1])
            X = np.stack([normalized[:, s:e][:, :, None] for s, e in bounds]).astype(np.float32)
            gt = np.array([int(np.bincount(trial.ground_truth[s:e], minlength=C.N_CLASSES).argmax())
                           for s, e in bounds])
            probs = [m.predict(X, batch_size=512, verbose=0) for m in models]
            out.append((gt, probs))
    return out


def _tune_operating_point(ft_model, tune_pool: SubjectPool, W3: np.ndarray) -> Tuple[float, int, float]:
    """Pick (tau, W) maximising balanced accuracy on CAL-VAL streams only."""
    data = _stream([ft_model], tune_pool, W3, TUNE_SEEDS, TUNE_SALT)
    gt = np.concatenate([g for g, _ in data])
    best = (-1.0, 0.5, 1)
    for tau in TAUS:
        gated = [_gate(p[0], tau) for _, p in data]        # per-trial gate
        for w in WINS:
            pred = np.concatenate([_smooth(g_, w) for g_ in gated])
            ba = balanced_accuracy_score(gt, pred)
            if ba > best[0]:
                best = (float(ba), float(tau), int(w))
    return best[1], best[2], best[0]


def _apply_variants(eval_data, tau: float, window: int) -> Dict[str, Dict]:
    """Score the 4 ablation variants on the shared eval streams."""
    gt = np.concatenate([g for g, _ in eval_data])
    preds = {
        "baseline":   np.concatenate([p[0].argmax(1) for _, p in eval_data]),        # global argmax
        "calibrated": np.concatenate([p[1].argmax(1) for _, p in eval_data]),        # finetuned argmax
        "cal_gate":   np.concatenate([_gate(p[1], tau) for _, p in eval_data]),      # + gate
        "full":       np.concatenate([_smooth(_gate(p[1], tau), window)              # + gate + smoothing
                                       for _, p in eval_data]),
    }
    return {v: _pack(gt, preds[v]) for v in VARIANTS}


# --------------------------------------------------------------------------- #
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "combined_eval.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        from lib import model3
        W3 = ea_io.load_ea_matrix(BASE_DIR / "ea_reference.json")
        normalize_fn = lambda sig: ea_io.apply_ea(W3, sig)  # noqa: E731
        base_weights = BASE_DIR / "weights.weights.h5"

        split = load_split(C.SPLIT_JSON)
        subjects = gen.select_subjects(split)
        logger.info("Combined pipeline on %d subjects: %s", len(subjects), subjects)

        base_model = model3.build_model3("eegsym")
        base_model.load_weights(base_weights)

        per_subject = []
        for subject in subjects:
            keys = [k for k in session_keys_for(split, "test") if k.startswith(f"{subject}_")]
            segs = extract_segments_for_sessions(keys)
            cal_tr, cal_va, ev = _split_subject(segs, seed=C.RANDOM_SEED + int(subject[1:]))
            eval_pool = _pool_from_segments(subject, ev)
            if not (eval_pool.mi_left and eval_pool.mi_right and eval_pool.rest):
                logger.warning("%s: eval pool missing a class, skipping.", subject)
                continue

            # tuning pool = cal_val; fall back to cal_train+cal_val if a class is absent
            tune_pool = _pool_from_segments(subject, cal_va)
            if not (tune_pool.mi_left and tune_pool.mi_right and tune_pool.rest):
                tune_pool = _pool_from_segments(subject, cal_tr + cal_va)
                logger.info("%s: cal_val lacked a class -> tuning on cal_train+cal_val.", subject)

            ft_model, n_ep = _finetune(base_weights, cal_tr, cal_va, normalize_fn)
            tau, window, tune_ba = _tune_operating_point(ft_model, tune_pool, W3)

            eval_data = _stream([base_model, ft_model], eval_pool, W3, EVAL_SEEDS, EVAL_SALT)
            variants = _apply_variants(eval_data, tau, window)

            row = {"subject": subject, "tau": tau, "smooth_window": window,
                   "tune_balanced_accuracy": tune_ba, "finetune_epochs": int(n_ep),
                   "n_cal_train_seg": len(cal_tr), "n_cal_val_seg": len(cal_va), "n_eval_seg": len(ev),
                   **{v: variants[v] for v in VARIANTS}}
            per_subject.append(row)
            logger.info("%s: bal_acc base %.3f -> full %.3f | LEFT %.3f -> %.3f | "
                        "FPR %.3f -> %.3f | tau=%.2f W=%d ep=%d",
                        subject, variants["baseline"]["balanced_accuracy"],
                        variants["full"]["balanced_accuracy"], variants["baseline"]["left_recall"],
                        variants["full"]["left_recall"], variants["baseline"]["fpr"],
                        variants["full"]["fpr"], tau, window, n_ep)
            del ft_model
            gc.collect()

        def agg(variant, field):
            return float(np.mean([r[variant][field] for r in per_subject]))

        def agg_delta(field, a="baseline", b="full"):
            d = np.array([r[b][field] - r[a][field] for r in per_subject])
            return {"mean": float(d.mean()), "std": float(d.std(ddof=1)),
                    "n_improved": int((d > 0).sum())}

        summary = {
            "design": "per-subject calibration + tunable REST/MI gate + temporal smoothing; "
                      "tau & W chosen on cal-val, applied blind to eval",
            "subjects": [r["subject"] for r in per_subject],
            "eval_seeds": list(EVAL_SEEDS), "tune_seeds": list(TUNE_SEEDS),
            "grids": {"tau": TAUS.tolist(), "smooth_window": list(WINS)},
            "aggregate": {v: {f: agg(v, f) for f in AGG_FIELDS} for v in VARIANTS},
            "delta_full_vs_baseline": {f: agg_delta(f) for f in AGG_FIELDS},
            "per_subject": per_subject,
        }
        save_json(OUT / "summary.json", summary)

        # ---- report ---- #
        a = summary["aggregate"]
        lines = ["# Combined pipeline: calibration + gate threshold + temporal smoothing", "",
                 f"Subjects: {', '.join(summary['subjects'])} | eval seeds: {EVAL_SEEDS}", "",
                 "Ablation on identical held-out continuous streams (means over subjects):", "",
                 "| Metric | Baseline | Calibrated | +Gate | +Smoothing (full) |",
                 "|---|---|---|---|---|"]
        for f in ("balanced_accuracy", "left_recall", "right_recall", "rest_recall", "fpr", "fnr"):
            lines.append(f"| {f} | " + " | ".join(f"{a[v][f]:.4f}" for v in VARIANTS) + " |")
        d = summary["delta_full_vs_baseline"]
        lines += ["", "Full vs baseline (per-subject deltas):", "",
                  "| Metric | delta mean | delta std | #improved |", "|---|---|---|---|"]
        for f in ("balanced_accuracy", "left_recall", "fpr", "fnr"):
            lines.append(f"| {f} | {d[f]['mean']:+.4f} | {d[f]['std']:.4f} | {d[f]['n_improved']}/{len(per_subject)} |")
        lines += ["", "Per-subject chosen operating point (tuned on cal-val only):", "",
                  "| Subject | tau | W | bal_acc base | bal_acc full |", "|---|---|---|---|---|"]
        for r in per_subject:
            lines.append(f"| {r['subject']} | {r['tau']:.2f} | {r['smooth_window']} | "
                         f"{r['baseline']['balanced_accuracy']:.3f} | {r['full']['balanced_accuracy']:.3f} |")
        (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        # ---- figures ---- #
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # (1) aggregate balanced accuracy across the 4 stages
        fig, ax = plt.subplots(figsize=(7, 5))
        vals = [a[v]["balanced_accuracy"] for v in VARIANTS]
        ax.bar(range(4), vals, color=["#8a8f98", "#4c9aa5", "#1f7a8c", "#0e7c74"])
        for i, val in enumerate(vals):
            ax.text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=9)
        ax.axhline(1 / 3, color="grey", ls="--", lw=0.8, label="chance")
        ax.set(xticks=range(4), ylim=(0, 1), ylabel="balanced accuracy",
               title="Stacked pipeline: balanced accuracy by stage")
        ax.set_xticklabels(["baseline", "calibrated", "+gate", "+smoothing"], rotation=15)
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(OUT / "ablation_balanced_accuracy.png", dpi=130); plt.close(fig)

        # (2) per-subject baseline vs full
        subs = [r["subject"] for r in per_subject]
        x = np.arange(len(subs)); w = 0.38
        fig, ax = plt.subplots(figsize=(max(7, len(subs) * 0.8), 5))
        ax.bar(x - w / 2, [r["baseline"]["balanced_accuracy"] for r in per_subject], w,
               label="baseline (global)", color="#8a8f98")
        ax.bar(x + w / 2, [r["full"]["balanced_accuracy"] for r in per_subject], w,
               label="full pipeline", color="#0e7c74")
        ax.axhline(1 / 3, color="grey", ls="--", lw=0.8)
        ax.set(title="Per-subject: baseline vs full combined pipeline (continuous)",
               ylabel="balanced accuracy", xticks=x, ylim=(0, 1))
        ax.set_xticklabels(subs, rotation=45, ha="right"); ax.legend(); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(OUT / "baseline_vs_full_per_subject.png", dpi=130); plt.close(fig)

        logger.info("=== COMBINED DONE in %.1fs | bal_acc base %.3f -> full %.3f (delta%+.3f) | "
                    "LEFT %.3f -> %.3f | FPR %.3f -> %.3f | %d/%d improved ===",
                    time.perf_counter() - t0, a["baseline"]["balanced_accuracy"],
                    a["full"]["balanced_accuracy"], d["balanced_accuracy"]["mean"],
                    a["baseline"]["left_recall"], a["full"]["left_recall"],
                    a["baseline"]["fpr"], a["full"]["fpr"],
                    d["balanced_accuracy"]["n_improved"], len(per_subject))
    finally:
        logger.removeHandler(h)
        h.close()


if __name__ == "__main__":
    main()
