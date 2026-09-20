"""Subject-specific fine-tuning of the winning model (EEGSym+EA, corrected window).

Tests whether a short per-subject calibration improves on the
subject-independent model, on 5 randomly chosen test subjects (never seen
during the original training -- test subjects were held out at the subject
level, so this is a legitimate personalization scenario, not leakage).

Protocol, per subject
----------------------
1. Take every one of that subject's test-split trials (corrected window,
   same preprocessing as everywhere else in the project).
2. Split them 60/20/20 at the *trial* level (deterministic, subject-specific
   seed): calibration-train / calibration-val (early stopping) / held-out
   eval. Splitting at the trial level (before cropping) means no crop can
   straddle two subsets.
3. Evaluate the ORIGINAL winning weights on the held-out eval crops
   (baseline, no fine-tuning) -- this is the number the subject would get
   with zero calibration.
4. Fine-tune a *fresh copy* of the original weights on calibration-train,
   early-stopping on calibration-val, with a much lower learning rate than
   the original training (1e-4 vs 1e-3) and lower weight decay (1e-5 vs
   1e-4), everything else identical (AdamW, label smoothing 0.1, same
   architecture).
5. Evaluate the fine-tuned model on the *same* held-out eval crops used in
   step 3 -- apples-to-apples comparison.

The Euclidean Alignment reference is NOT recomputed per subject (kept as
the original, globally-fitted-on-training-subjects reference): this
isolates the effect of fine-tuning the model weights from the effect of
changing the normalization, which is a different, separate question.

Writes into ``experiments/corrected_window/subject_finetuning/``.

Usage:  python finetune_subject_corrected.py   (run from route_b/two_class/, BCI_DATA set)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as base_config
from src.crop_size_study import build_crops_sized
from src.dataset import imagery_window_from_feedback
from src.eegsym_study.eegsym import build_eegsym
from src.split import load_split, session_keys_for
from src.study.data_loading import LoadedTrial, load_trials_for_sessions
from src.study.evaluation import evaluate_experiment
from src.study.normalization import EuclideanAlignment
from src.utils import get_logger, set_global_seed

logger = get_logger()

ROOT = Path(__file__).resolve().parents[1]
BASE_WEIGHTS = (
    ROOT / "experiments" / "corrected_window" / "normalization" / "eegsym"
    / "euclidean_alignment" / "weights.weights.h5"
)
OUT_DIR = ROOT / "experiments" / "corrected_window" / "subject_finetuning"

RANDOM_SEED = 42
N_SUBJECTS = 5
FS = 250
WINDOW = 250
STRIDE = 125
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)  # cal-train, cal-val, eval

# Fine-tuning schedule -- much gentler than the original training.
FT_LEARNING_RATE = 1e-4
FT_WEIGHT_DECAY = 1e-5
FT_LABEL_SMOOTHING = 0.1
FT_BATCH_SIZE = 64
FT_MAX_EPOCHS = 60
FT_PATIENCE = 8


def _subject_trial_split(trials: List[LoadedTrial], seed: int):
    n = len(trials)
    order = np.random.default_rng(seed).permutation(n)
    n_cal_train = int(round(n * SPLIT_FRACTIONS[0]))
    n_cal_val = int(round(n * SPLIT_FRACTIONS[1]))
    idx_cal_train = order[:n_cal_train]
    idx_cal_val = order[n_cal_train:n_cal_train + n_cal_val]
    idx_eval = order[n_cal_train + n_cal_val:]
    return (
        [trials[i] for i in idx_cal_train],
        [trials[i] for i in idx_cal_val],
        [trials[i] for i in idx_eval],
    )


def _assert_disjoint(*groups: List[LoadedTrial]) -> None:
    seen: set = set()
    for g in groups:
        ids = {t.trial_id for t in g}
        if seen & ids:
            raise RuntimeError(f"Trial leakage between subject subsets: {seen & ids}")
        seen |= ids


def _make_tf_dataset(crops, batch_size: int, shuffle: bool, seed: int):
    from src.study.crops import make_tf_dataset
    return make_tf_dataset(crops, batch_size, shuffle=shuffle, seed=seed)


def _finetune_one_subject(subject: str, trials: List[LoadedTrial], ea: EuclideanAlignment,
                           seed: int) -> Dict[str, object]:
    out_dir = OUT_DIR / f"subject_{subject}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_train, cal_val, eval_trials = _subject_trial_split(trials, seed)
    _assert_disjoint(cal_train, cal_val, eval_trials)
    logger.info("[%s] trials: cal_train=%d cal_val=%d eval=%d (total=%d)",
                subject, len(cal_train), len(cal_val), len(eval_trials), len(trials))

    crops_cal_train = build_crops_sized(cal_train, ea.transform, WINDOW, STRIDE)
    crops_cal_val = build_crops_sized(cal_val, ea.transform, WINDOW, STRIDE)
    crops_eval = build_crops_sized(eval_trials, ea.transform, WINDOW, STRIDE)
    logger.info("[%s] crops: cal_train=%d cal_val=%d eval=%d",
                subject, len(crops_cal_train), len(crops_cal_val), len(crops_eval))

    from tensorflow import keras

    # --- Baseline: original weights, zero calibration --------------------- #
    set_global_seed(seed)
    model = build_eegsym()
    model.load_weights(BASE_WEIGHTS)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=FT_LEARNING_RATE, weight_decay=FT_WEIGHT_DECAY),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=FT_LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    baseline_eval = evaluate_experiment(model, crops_eval, out_dir / "baseline")
    logger.info("[%s] BASELINE (no fine-tuning) trial acc=%.4f f1=%.4f auc=%.4f",
                subject, baseline_eval["trial"]["accuracy"], baseline_eval["trial"]["f1_macro"],
                baseline_eval["trial"]["roc_auc"])

    # --- Fine-tune on this subject's calibration data ---------------------- #
    set_global_seed(seed)
    ft_model = build_eegsym()
    ft_model.load_weights(BASE_WEIGHTS)
    ft_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=FT_LEARNING_RATE, weight_decay=FT_WEIGHT_DECAY),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=FT_LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    train_ds = _make_tf_dataset(crops_cal_train, FT_BATCH_SIZE, shuffle=True, seed=seed)
    val_ds = _make_tf_dataset(crops_cal_val, FT_BATCH_SIZE, shuffle=False, seed=seed)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=FT_PATIENCE, restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0),
    ]
    t0 = time.perf_counter()
    history = ft_model.fit(
        train_ds, validation_data=val_ds, epochs=FT_MAX_EPOCHS,
        callbacks=callbacks, shuffle=False, verbose=0,
    )
    ft_time = time.perf_counter() - t0
    n_epochs = len(history.history.get("loss", []))
    logger.info("[%s] fine-tuned for %d epochs in %.1fs", subject, n_epochs, ft_time)

    ft_eval = evaluate_experiment(ft_model, crops_eval, out_dir / "finetuned")
    logger.info("[%s] FINE-TUNED trial acc=%.4f f1=%.4f auc=%.4f",
                subject, ft_eval["trial"]["accuracy"], ft_eval["trial"]["f1_macro"],
                ft_eval["trial"]["roc_auc"])

    ft_model.save_weights(out_dir / "finetuned_weights.weights.h5")

    result = {
        "subject": subject,
        "n_trials_total": len(trials),
        "n_trials_cal_train": len(cal_train),
        "n_trials_cal_val": len(cal_val),
        "n_trials_eval": len(eval_trials),
        "n_crops_cal_train": len(crops_cal_train),
        "n_crops_cal_val": len(crops_cal_val),
        "n_crops_eval": len(crops_eval),
        "finetune_epochs_run": n_epochs,
        "finetune_time_s": ft_time,
        "baseline": {
            "trial_accuracy": baseline_eval["trial"]["accuracy"],
            "trial_f1_macro": baseline_eval["trial"]["f1_macro"],
            "trial_roc_auc": baseline_eval["trial"]["roc_auc"],
        },
        "finetuned": {
            "trial_accuracy": ft_eval["trial"]["accuracy"],
            "trial_f1_macro": ft_eval["trial"]["f1_macro"],
            "trial_roc_auc": ft_eval["trial"]["roc_auc"],
        },
        "delta": {
            "trial_accuracy": ft_eval["trial"]["accuracy"] - baseline_eval["trial"]["accuracy"],
            "trial_f1_macro": ft_eval["trial"]["f1_macro"] - baseline_eval["trial"]["f1_macro"],
            "trial_roc_auc": ft_eval["trial"]["roc_auc"] - baseline_eval["trial"]["roc_auc"],
        },
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    set_global_seed(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split_payload = load_split()
    test_subjects = sorted(split_payload["splits"]["test"]["subjects"], key=lambda s: int(s.lstrip("S")))
    selected = list(np.random.default_rng(RANDOM_SEED).choice(test_subjects, size=N_SUBJECTS, replace=False))
    logger.info("Test subjects available: %s", test_subjects)
    logger.info("Selected %d subjects (seed=%d): %s", N_SUBJECTS, RANDOM_SEED, selected)

    logger.info("Loading train trials for the EA reference (reused, not recomputed per subject)...")
    train_trials = load_trials_for_sessions(
        session_keys_for(split_payload, "train"), window_fn=imagery_window_from_feedback
    )
    ea = EuclideanAlignment()
    ea.fit([t.signal for t in train_trials])
    del train_trials  # free memory, no longer needed

    logger.info("Loading all test trials...")
    test_trials = load_trials_for_sessions(
        session_keys_for(split_payload, "test"), window_fn=imagery_window_from_feedback
    )
    by_subject: Dict[str, List[LoadedTrial]] = {}
    for t in test_trials:
        by_subject.setdefault(t.subject, []).append(t)

    results = []
    for i, subject in enumerate(selected):
        subj_trials = by_subject.get(subject, [])
        if len(subj_trials) < 20:
            logger.warning("Skipping %s: only %d trials (too few).", subject, len(subj_trials))
            continue
        result = _finetune_one_subject(subject, subj_trials, ea, seed=RANDOM_SEED + i + 1)
        results.append(result)

    deltas_acc = [r["delta"]["trial_accuracy"] for r in results]
    summary = {
        "random_seed": RANDOM_SEED,
        "n_subjects_requested": N_SUBJECTS,
        "test_subjects_available": test_subjects,
        "selected_subjects": selected,
        "base_weights": str(BASE_WEIGHTS),
        "finetune_config": {
            "learning_rate": FT_LEARNING_RATE,
            "weight_decay": FT_WEIGHT_DECAY,
            "label_smoothing": FT_LABEL_SMOOTHING,
            "batch_size": FT_BATCH_SIZE,
            "max_epochs": FT_MAX_EPOCHS,
            "early_stopping_patience": FT_PATIENCE,
            "split_fractions_cal_train_cal_val_eval": SPLIT_FRACTIONS,
        },
        "per_subject": results,
        "mean_delta_trial_accuracy": float(np.mean(deltas_acc)) if deltas_acc else None,
        "std_delta_trial_accuracy": float(np.std(deltas_acc)) if deltas_acc else None,
        "n_improved": int(sum(1 for d in deltas_acc if d > 0)),
        "n_subjects_evaluated": len(results),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %s", OUT_DIR / "summary.json")
    logger.info("=== SUMMARY: mean delta trial accuracy = %+.4f (std %.4f), %d/%d subjects improved ===",
                summary["mean_delta_trial_accuracy"] or 0.0, summary["std_delta_trial_accuracy"] or 0.0,
                summary["n_improved"], summary["n_subjects_evaluated"])


if __name__ == "__main__":
    main()
