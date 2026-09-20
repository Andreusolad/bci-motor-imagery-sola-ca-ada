r"""Step 2 --- per-subject calibration (fine-tuning) of the 3-class EEGSym+EA
model, evaluated on the subject's OWN continuous 5-minute streams.

Tests the real bottleneck (cross-subject domain shift): give the frozen 3-class
model a few of the subject's own trials to calibrate, then see if it decodes
that subject's continuous stream better.

Leakage control: each subject's trials are split (by ORIGINAL trial, so a
trial's MI and its REST baseline stay together) into cal-train / cal-val / eval.
Fine-tuning uses only cal-*; the continuous streams are built ONLY from eval
segments. The global 3-class EA reference is kept fixed (not recomputed per
subject) to isolate the weight-adaptation effect, exactly as in the prior
subject fine-tuning study.

Usage (after train_3class.py --arch eegsym has produced the base model):
    python calibration_eval.py
"""
from __future__ import annotations
import sys

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import config3 as C
from lib import continuous_gen as gen
from lib import crops3, ea_io, metrics3, model3, viz3
from lib.segments import Segment, SubjectPool, extract_segments_for_sessions

from src.split import load_split, session_keys_for  # noqa: E402
from src.utils import get_logger, save_json  # noqa: E402

logger = get_logger()
OUT = C.EXPERIMENTS / "continuous_eval_calibration"
BASE_DIR = C.model_dir("eegsym", "euclidean_alignment")   # frozen 3-class EEGSym+EA

CAL_SEEDS = (42, 123, 256)          # fewer seeds than Part 2 (this is a 2x-model sweep)
SPLIT_FRACTIONS = (0.55, 0.15, 0.30)  # cal_train / cal_val / eval, by original trial
FT_LR = 1e-4
FT_WD = 1e-5
FT_LABEL_SMOOTHING = 0.1
FT_BATCH = 64
FT_MAX_EPOCHS = 40
FT_PATIENCE = 8


def _underlying_trial(seg: Segment) -> str:
    """Original-trial key shared by a trial's MI and its REST baseline."""
    tag = seg.trial_id.split("#")[1]            # "trial123" or "rest123"
    idx = tag.replace("trial", "").replace("rest", "")
    return f"{seg.session_key}#{idx}"


def _split_subject(segs: List[Segment], seed: int) -> Tuple[List[Segment], List[Segment], List[Segment]]:
    keys = sorted({_underlying_trial(s) for s in segs})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    n = len(keys)
    n_tr = int(round(SPLIT_FRACTIONS[0] * n))
    n_va = int(round(SPLIT_FRACTIONS[1] * n))
    tr = {keys[i] for i in order[:n_tr]}
    va = {keys[i] for i in order[n_tr:n_tr + n_va]}
    buckets = ([], [], [])
    for s in segs:
        k = _underlying_trial(s)
        (buckets[0] if k in tr else buckets[1] if k in va else buckets[2]).append(s)
    return buckets


def _pool_from_segments(subject: str, segs: List[Segment]) -> SubjectPool:
    return SubjectPool(
        subject=subject,
        mi_left=[s.signal for s in segs if s.label == C.LEFT_ID],
        mi_right=[s.signal for s in segs if s.label == C.RIGHT_ID],
        rest=[s.signal for s in segs if s.label == C.REST_ID],
    )


def _finetune(base_weights: Path, cal_tr: List[Segment], cal_va: List[Segment],
              normalize_fn) -> object:
    from tensorflow import keras
    from src.utils import set_global_seed
    set_global_seed(C.RANDOM_SEED)

    tr_bal, _ = crops3.balance_rest(cal_tr, seed=C.RANDOM_SEED)
    va_bal, _ = crops3.balance_rest(cal_va, seed=C.RANDOM_SEED)
    tr_crops = crops3.build_crops(tr_bal, normalize_fn)
    va_crops = crops3.build_crops(va_bal, normalize_fn)

    net = model3.build_model3("eegsym")
    net.load_weights(base_weights)
    net.compile(optimizer=keras.optimizers.AdamW(learning_rate=FT_LR, weight_decay=FT_WD),
                loss=keras.losses.CategoricalCrossentropy(label_smoothing=FT_LABEL_SMOOTHING),
                metrics=["accuracy"])
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=FT_PATIENCE,
                                         restore_best_weights=True, verbose=0),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5,
                                             min_lr=1e-6, verbose=0)]
    tr_ds = crops3.make_tf_dataset(tr_crops, FT_BATCH, True, C.RANDOM_SEED)
    va_ds = crops3.make_tf_dataset(va_crops, FT_BATCH, False, C.RANDOM_SEED)
    hist = net.fit(tr_ds, validation_data=va_ds, epochs=FT_MAX_EPOCHS, callbacks=cbs, verbose=0)
    return net, len(hist.history["loss"])


def _eval_streams(model, pool: SubjectPool, W3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Continuous windows over CAL_SEEDS x 5 trials from this subject's eval pool."""
    gts, preds = [], []
    for seed in CAL_SEEDS:
        for ti in range(C.CONT_N_TRIALS_PER_SUBJECT):
            rng = np.random.default_rng([seed, int(pool.subject[1:]), ti, 7])
            trial = gen.build_continuous_trial(pool, ti, rng, seed=seed)
            normalized = ea_io.apply_ea(W3, trial.signal)
            bounds = crops3.iter_crop_bounds(normalized.shape[1])
            X = np.stack([normalized[:, s:e][:, :, None] for s, e in bounds]).astype(np.float32)
            probs = model.predict(X, batch_size=512, verbose=0)
            preds.append(probs.argmax(1))
            gts.append(np.array([int(np.bincount(trial.ground_truth[s:e], minlength=3).argmax())
                                 for s, e in bounds]))
    return np.concatenate(gts), np.concatenate(preds)


def _pack(gt, pred) -> Dict:
    m = metrics3.compute_metrics(gt, pred)
    b = metrics3.bci_continuous_metrics(gt, pred)
    return {"balanced_accuracy": m["balanced_accuracy"], "accuracy": m["accuracy"],
            "rest_recall": m["per_class_accuracy"]["REST"],
            "left_recall": m["per_class_accuracy"]["LEFT"],
            "right_recall": m["per_class_accuracy"]["RIGHT"],
            "fpr": b["false_positive_rate_rest_as_mi"], "fnr": b["false_negative_rate_mi_as_rest"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "calibration_eval.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        W3 = ea_io.load_ea_matrix(BASE_DIR / "ea_reference.json")
        normalize_fn = lambda sig: ea_io.apply_ea(W3, sig)  # noqa: E731
        base_weights = BASE_DIR / "weights.weights.h5"

        split = load_split(C.SPLIT_JSON)
        subjects = gen.select_subjects(split)
        logger.info("Calibration on %d subjects: %s", len(subjects), subjects)

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

            gt_b, pred_b = _eval_streams(base_model, eval_pool, W3)
            ft_model, n_ep = _finetune(base_weights, cal_tr, cal_va, normalize_fn)
            gt_f, pred_f = _eval_streams(ft_model, eval_pool, W3)

            base_m = _pack(gt_b, pred_b)
            ft_m = _pack(gt_f, pred_f)
            row = {"subject": subject,
                   "n_cal_train_seg": len(cal_tr), "n_cal_val_seg": len(cal_va),
                   "n_eval_seg": len(ev), "finetune_epochs": int(n_ep),
                   "baseline": base_m, "finetuned": ft_m,
                   "delta": {k: ft_m[k] - base_m[k] for k in base_m}}
            per_subject.append(row)
            logger.info("%s: bal_acc %.3f->%.3f (delta%+.3f) | LEFT %.3f->%.3f | FPR %.3f->%.3f | ep=%d",
                        subject, base_m["balanced_accuracy"], ft_m["balanced_accuracy"],
                        row["delta"]["balanced_accuracy"], base_m["left_recall"], ft_m["left_recall"],
                        base_m["fpr"], ft_m["fpr"], n_ep)
            import gc
            del ft_model
            gc.collect()

        def agg(field):
            b = np.array([r["baseline"][field] for r in per_subject])
            f = np.array([r["finetuned"][field] for r in per_subject])
            return {"baseline_mean": float(b.mean()), "finetuned_mean": float(f.mean()),
                    "mean_delta": float((f - b).mean()), "std_delta": float((f - b).std(ddof=1)),
                    "n_improved": int((f > b).sum())}

        summary = {
            "design": "per-subject fine-tuning of 3-class EEGSym+EA, eval on held-out continuous streams",
            "subjects": [r["subject"] for r in per_subject], "cal_seeds": list(CAL_SEEDS),
            "finetune": {"lr": FT_LR, "weight_decay": FT_WD, "label_smoothing": FT_LABEL_SMOOTHING,
                         "batch": FT_BATCH, "max_epochs": FT_MAX_EPOCHS, "patience": FT_PATIENCE,
                         "split_fractions_caltr_calva_eval": list(SPLIT_FRACTIONS),
                         "ea": "global 3-class reference kept fixed (not recomputed per subject)"},
            "aggregate": {k: agg(k) for k in ("balanced_accuracy", "left_recall", "right_recall",
                                              "rest_recall", "fpr", "fnr", "accuracy")},
            "per_subject": per_subject,
        }
        save_json(OUT / "summary.json", summary)

        # report + figures
        a = summary["aggregate"]
        lines = ["# Per-subject calibration of the 3-class EEGSym+EA model (continuous eval)", "",
                 f"Subjects: {', '.join(summary['subjects'])} | seeds: {CAL_SEEDS}", "",
                 "| Metric | Baseline | Fine-tuned | delta (mean) | std delta | #improved |",
                 "|---|---|---|---|---|---|"]
        for k in ("balanced_accuracy", "left_recall", "right_recall", "rest_recall", "fpr", "fnr"):
            g = a[k]
            lines.append(f"| {k} | {g['baseline_mean']:.4f} | {g['finetuned_mean']:.4f} | "
                         f"{g['mean_delta']:+.4f} | {g['std_delta']:.4f} | {g['n_improved']}/{len(per_subject)} |")
        (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        subs = [r["subject"] for r in per_subject]
        viz3.plot_bars(subs, [r["finetuned"]["balanced_accuracy"] for r in per_subject],
                       OUT / "balanced_acc_finetuned_per_subject.png",
                       title="Fine-tuned balanced accuracy per subject (continuous)",
                       ylabel="balanced accuracy", hline=1 / 3)
        # baseline vs finetuned grouped bar (balanced acc)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(subs)); w = 0.38
        fig, ax = plt.subplots(figsize=(max(7, len(subs) * 0.8), 5))
        ax.bar(x - w / 2, [r["baseline"]["balanced_accuracy"] for r in per_subject], w,
               label="baseline", color="#8a8f98")
        ax.bar(x + w / 2, [r["finetuned"]["balanced_accuracy"] for r in per_subject], w,
               label="fine-tuned", color="#0e7c74")
        ax.axhline(1 / 3, color="grey", ls="--", lw=0.8)
        ax.set(title="Per-subject calibration: balanced accuracy (continuous streams)",
               ylabel="balanced accuracy", xticks=x, ylim=(0, 1))
        ax.set_xticklabels(subs, rotation=45, ha="right"); ax.legend(); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(OUT / "baseline_vs_finetuned_balacc.png", dpi=130); plt.close(fig)

        logger.info("=== CALIBRATION DONE in %.1fs | bal_acc %.3f->%.3f (delta%+.3f) | LEFT %.3f->%.3f | %d/%d improved ===",
                    time.perf_counter() - t0, a["balanced_accuracy"]["baseline_mean"],
                    a["balanced_accuracy"]["finetuned_mean"], a["balanced_accuracy"]["mean_delta"],
                    a["left_recall"]["baseline_mean"], a["left_recall"]["finetuned_mean"],
                    a["balanced_accuracy"]["n_improved"], len(per_subject))
    finally:
        logger.removeHandler(h)
        h.close()


if __name__ == "__main__":
    main()
