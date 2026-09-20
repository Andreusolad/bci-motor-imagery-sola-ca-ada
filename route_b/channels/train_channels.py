r"""Channel-count diagnostic on Stieger: does more channels raise the ceiling?

Trains the winner pipeline (EEGNet + Euclidean Alignment, corrected window,
binary LH/RH) at three NESTED, bibliographically-grounded, bilaterally-symmetric
montages, holding architecture + everything else fixed so the only variable is
the electrode count:

  * 8  : FC3 FCz FC4 C3 Cz C4 CP3 CP4  -- sensorimotor strip (Cyton deploy set).
  * 16 : + FC1 FC2 C5 C1 C2 C6 CP1 CP2 -- full FC/C/CP sensorimotor grid = the
         central block of the BCI-IV-2a montage (Brunner et al. 2008;
         Pfurtscheller & Neuper 2001).
  * 32 : + Fz Pz F3 F4 FC5 FC6 FT7 FT8 T7 T8 CP5 CP6 TP7 TP8 P3 P4 -- a
         bilaterally symmetric frontal-temporal-parietal ring covering the wider
         motor-imagery network (premotor/SMA + posterior parietal; Hanakawa
         et al. 2003); standard 10-10 32-channel research montage.

Why EEGNet (not EEGSym): EEGSym hard-codes an 8-channel hemisphere split and does
not scale; EEGNet is a 2-D conv over channels x time and scales cleanly, keeping
the architecture identical across 8/16/32 (the correct controlled design). We
already showed EEGNet+EA == EEGSym+EA on 2a (0.736 vs 0.738), so no loss of
validity.

Methodology is the winner's, verbatim: same subject split, corrected window
[2000, 2000+triallength] ms, EA global reference fit on TRAIN only (leakage-safe),
1 s / 0.5 s crops, AdamW, early stopping, trial-level aggregation. Reports
per-subject trial accuracy + subject-level bootstrap CI (Phase-1 framework).

Usage:  python train_channels.py --set 16   (run from route_b/channels/, BCI_DATA set)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as base_config                            # noqa: E402

# ---- nested, symmetric, literature-grounded channel sets (all present in Stieger) ----
SET8 = ["FC3", "FCZ", "FC4", "C3", "CZ", "C4", "CP3", "CP4"]
SET16 = ["FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
         "CP3", "CP1", "CP2", "CP4"]
SET32 = SET16 + ["FZ", "PZ", "F3", "F4", "FC5", "FC6", "FT7", "FT8", "T7", "T8",
                 "CP5", "CP6", "TP7", "TP8", "P3", "P4"]
SETS = {8: SET8, 16: SET16, 32: SET32}
_REPO = Path(__file__).resolve().parents[1]
EXISTING_8_WEIGHTS = (_REPO / "experiments" / "corrected_window" / "normalization"
                      / "eegnet" / "euclidean_alignment" / "weights.weights.h5")
OUT_ROOT = _REPO / "experiments"


def _patch_channels(chans: List[str]) -> None:
    """Monkey-patch the global motor-channel set so the whole loader uses it."""
    base_config.MOTOR_CHANNELS = tuple(chans)
    base_config.N_CHANNELS = len(chans)


def per_subject_accuracy(agg, tid2subj) -> Dict[str, Tuple[int, int]]:
    acc: Dict[str, List[int]] = {}
    for tid, yt, yh in zip(agg.trial_ids, agg.y_true, agg.y_pred):
        c, n = acc.get(tid2subj[tid], (0, 0))
        acc[tid2subj[tid]] = (c + int(yt == yh), n + 1)
    return acc


def boot_ci(v: np.ndarray, rng, b=10_000) -> Tuple[float, float, float]:
    n = len(v); m = v[rng.integers(0, n, size=(b, n))].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _auc(y, score) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    r = np.empty(len(order)); r[order] = np.arange(1, len(order) + 1)
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run(nch: int):
    chans = SETS[nch]
    _patch_channels(chans)
    out = OUT_ROOT / f"eegnet_ea_{nch}ch"; out.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("first_ml")
    h = logging.FileHandler(out / "train.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h); logger.setLevel(logging.INFO)
    t0 = time.perf_counter()

    # imports AFTER patching channels (they read base_config at call time)
    from src.split import load_split, session_keys_for
    from src.dataset import imagery_window_from_feedback
    from src.study.data_loading import load_trials_for_sessions
    from src.study.crops import build_crops, make_tf_dataset
    from src.study.run_experiment import _build_normalizer
    from src.study.eegnet import build_eegnet
    from src.study.training import _compile_model, _build_callbacks
    from src.study.trial_aggregation import aggregate_by_trial
    from src.utils import set_global_seed, save_json
    from tensorflow import keras

    set_global_seed(base_config.RANDOM_SEED)
    logger.info("=== CHANNEL STUDY: %d channels === %s", nch, chans)
    wf = imagery_window_from_feedback
    split = load_split()
    train = load_trials_for_sessions(session_keys_for(split, "train"), window_fn=wf)
    val = load_trials_for_sessions(session_keys_for(split, "val"), window_fn=wf)
    test = load_trials_for_sessions(session_keys_for(split, "test"), window_fn=wf)
    assert train[0].signal.shape[0] == nch, f"expected {nch} channels, got {train[0].signal.shape[0]}"
    tid2subj = {t.trial_id: t.subject for t in test}
    subjects = sorted({t.subject for t in test}, key=lambda s: int(s.lstrip("S")))
    logger.info("trials train/val/test = %d/%d/%d ; test subjects=%d", len(train), len(val), len(test), len(subjects))

    normalize_fn, ea_params = _build_normalizer("euclidean_alignment", train)  # EA global on TRAIN
    tr_c, va_c, te_c = (build_crops(train, normalize_fn), build_crops(val, normalize_fn),
                        build_crops(test, normalize_fn))

    model = build_eegnet(input_shape=(nch, base_config.CROP_SAMPLES, 1), n_classes=2)
    _compile_model(model)
    n_params = int(sum(w.numpy().size for w in model.trainable_weights))

    if nch == 8 and EXISTING_8_WEIGHTS.exists():
        model.load_weights(str(EXISTING_8_WEIGHTS))
        logger.info("8ch: loaded existing winner weights (no retrain); params=%d", n_params)
        epochs_run = 0
    else:
        tr_ds = make_tf_dataset(tr_c, base_config.TRAIN.batch_size, shuffle=True, seed=base_config.RANDOM_SEED)
        va_ds = make_tf_dataset(va_c, base_config.TRAIN.batch_size, shuffle=False, seed=base_config.RANDOM_SEED)
        et: List[float] = []
        hist = model.fit(tr_ds, validation_data=va_ds, epochs=base_config.TRAIN.epochs,
                         callbacks=_build_callbacks(out, et), shuffle=False, verbose=2)
        epochs_run = len(hist.history["loss"])
        model.save_weights(str(out / "weights.weights.h5"))
        logger.info("trained %d epochs; params=%d", epochs_run, n_params)

    # ---- evaluate held-out test at trial level ----
    probs = model.predict(te_c.x, batch_size=512, verbose=0)
    agg = aggregate_by_trial(probs, te_c.trial_ids, te_c.y)
    acc = per_subject_accuracy(agg, tid2subj)
    vec = np.array([acc[s][0] / acc[s][1] for s in subjects])
    pooled = sum(acc[s][0] for s in subjects) / sum(acc[s][1] for s in subjects)
    auc = _auc(agg.y_true, agg.y_prob[:, 1])
    rng = np.random.default_rng(base_config.RANDOM_SEED)
    pt, lo, hi = boot_ci(vec, rng)

    summary = {"n_channels": nch, "channels": chans, "n_params": n_params, "epochs_run": epochs_run,
               "n_test_subjects": len(subjects), "pooled_trial_accuracy": float(pooled),
               "subject_mean_accuracy": pt, "ci_low": lo, "ci_high": hi, "trial_auc": auc,
               "per_subject": {s: vec[i] for i, s in enumerate(subjects)},
               "ea_params": ea_params, "train_time_s": time.perf_counter() - t0}
    save_json(out / "summary.json", summary)
    logger.info("=== %dch DONE in %.1fs === pooled-acc=%.4f subj-mean=%.4f [%.4f,%.4f] AUC=%.4f",
                nch, summary["train_time_s"], pooled, pt, lo, hi, auc)
    logger.removeHandler(h); h.close()
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=int, choices=[8, 16, 32], required=True)
    run(ap.parse_args().set)
