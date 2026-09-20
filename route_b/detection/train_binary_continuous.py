r"""Train a binary REST-vs-MI detector directly on continuous-derived windows
(including boundary/transition windows), then evaluate it as a gate on the
5-minute continuous test streams -- head-to-head with the two gates we already
have: the crop-trained binary (AUC 0.759) and the 3-class gate (AUC 0.732).

Hypothesis: the crop-trained binary drops from AUC 0.865 (clean crops) to 0.759
(continuous) because of transition windows it never saw. Training on continuous
windows should recover part of that gap -> a better gate. The base-rate ceiling
(94 % REST) is expected to remain.

Reuses the train_continuous.py machinery (leakage-safe: subject split first,
records built only from a set's subjects, EA on train, threshold selected on
val). Only the label mapping changes: REST=0, MI(=LEFT or RIGHT)=1.

Usage:  python train_binary_continuous.py   (run from route_b/detection/, BCI_DATA set)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "continuous"))

from lib import config3 as C                          # noqa: E402
from lib import continuous_gen as gen                 # noqa: E402
from lib import crops3, ea_io                         # noqa: E402
from lib.segments import load_or_build_subject_pool   # noqa: E402

import train_continuous as TC                         # noqa: E402

from src.eegsym_study.eegsym import build_eegsym       # noqa: E402
from src.eegsym_study import config as eg              # noqa: E402
from src.split import load_split                       # noqa: E402
from src.utils import get_logger, save_json, set_global_seed  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402

logger = get_logger()
OUT = Path(__file__).resolve().parents[1] / "experiments" / "binary_continuous"
REST = C.REST_ID


def at_threshold(y, p, t):
    pred = (p >= t).astype(int)
    tp = ((y == 1) & (pred == 1)).sum(); fn = ((y == 1) & (pred == 0)).sum()
    fp = ((y == 0) & (pred == 1)).sum(); tn = ((y == 0) & (pred == 0)).sum()
    rec_mi = tp / (tp + fn) if (tp + fn) else 0.0
    rec_rest = tn / (tn + fp) if (tn + fp) else 0.0
    return {"threshold": float(t), "fpr": float(1 - rec_rest), "fnr": float(1 - rec_mi),
            "recall_MI": float(rec_mi), "balanced_accuracy": float(0.5 * (rec_mi + rec_rest))}


def collect(model, W, pools, subjects, seeds):
    P, Y = [], []
    for seed in seeds:
        for s in subjects:
            for ti in range(C.CONT_N_TRIALS_PER_SUBJECT):
                rng = np.random.default_rng([seed, int(s[1:]), ti])
                trial = gen.build_continuous_trial(pools[s], ti, rng, seed=seed)
                norm = ea_io.apply_ea(W, trial.signal)
                bounds = crops3.iter_crop_bounds(norm.shape[1])
                X = np.stack([norm[:, a:b][:, :, None] for a, b in bounds]).astype(np.float32)
                P.append(model.predict(X, batch_size=512, verbose=0)[:, 1])
                gt = np.array([int(np.bincount(trial.ground_truth[a:b], minlength=C.N_CLASSES).argmax())
                               for a, b in bounds])
                Y.append((gt != REST).astype(int))
    return np.concatenate(P), np.concatenate(Y)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "train_binary_continuous.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        from tensorflow import keras
        set_global_seed(C.RANDOM_SEED)
        split = load_split(C.SPLIT_JSON)
        tr_subs = sorted(split["splits"]["train"]["subjects"], key=lambda s: int(s[1:]))[:TC.N_TRAIN_SUBJECTS]
        va_subs = sorted(split["splits"]["val"]["subjects"], key=lambda s: int(s[1:]))[:TC.N_VAL_SUBJECTS]
        pools = {s: load_or_build_subject_pool(s, split) for s in tr_subs + va_subs}

        train_sig = []
        for s in tr_subs:
            train_sig += pools[s].mi_left + pools[s].mi_right + pools[s].rest
        ea = ea_io.fit_ea(train_sig); ea_io.save_ea(ea, OUT / "ea_reference.json")
        W = ea_io.load_ea_matrix(OUT / "ea_reference.json")

        # continuous-derived windows (3-class labels) -> remap to binary
        Xtr, ytr3 = TC.build_windows(pools, tr_subs, TC.RECORDS_PER_TRAIN, W, TC.GEN_SEED, 1, True)
        Xva, yva3 = TC.build_windows(pools, va_subs, TC.RECORDS_PER_VAL, W, TC.GEN_SEED + 1, 2, True)
        ytr = (ytr3 != REST).astype(np.int64); yva = (yva3 != REST).astype(np.int64)
        cnt = np.bincount(ytr, minlength=2)
        logger.info("train windows %d (REST %d/MI %d) | val %d", len(ytr), cnt[0], cnt[1], len(yva))

        cw = len(ytr) / (2 * np.maximum(cnt, 1)); wtr = cw[ytr].astype(np.float32)

        model = build_eegsym(input_shape=C.INPUT_SHAPE, n_classes=2)
        from lib import model3
        model3.compile_model3(model)

        def ds(X, y, w, shuffle):
            import tensorflow as tf
            d = tf.data.Dataset.from_tensor_slices((X, tf.one_hot(y, 2), w))
            if shuffle:
                d = d.shuffle(min(len(X), 20000), seed=C.RANDOM_SEED, reshuffle_each_iteration=True)
            return d.batch(eg.TRAIN.batch_size).prefetch(tf.data.AUTOTUNE)

        cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=eg.TRAIN.early_stopping_patience,
                                             restore_best_weights=True, verbose=1),
               keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=eg.TRAIN.reduce_lr_factor,
                                                 patience=eg.TRAIN.reduce_lr_patience, min_lr=eg.TRAIN.min_lr)]
        model.fit(ds(Xtr, ytr, wtr, True),
                  validation_data=ds(Xva, yva, np.ones(len(yva), np.float32), False),
                  epochs=eg.TRAIN.epochs, callbacks=cbs, verbose=0)
        model.save_weights(OUT / "weights.weights.h5")

        # threshold on VAL continuous streams
        pv, yv = collect(model, W, pools, va_subs, [11])
        grid = np.linspace(0.1, 0.9, 81)
        t_star = float(grid[np.argmax([at_threshold(yv, pv, t)["balanced_accuracy"] for t in grid])])

        # test streams (5 seeds)
        test_subs = gen.select_subjects(split)
        tpools = {s: load_or_build_subject_pool(s, split) for s in test_subs}
        pt, yt = collect(model, W, tpools, test_subs, C.CONT_SEEDS)
        fprs = np.array([at_threshold(yt, pt, t)["fpr"] for t in grid])
        t_match = float(grid[np.argmin(np.abs(fprs - 0.425))])   # match the 3-class gate FPR

        results = {
            "design": "binary REST-vs-MI trained ON continuous windows; gate eval on 5-min test streams",
            "n_windows": int(len(yt)), "mi_prevalence": float(yt.mean()),
            "roc_auc": float(roc_auc_score(yt, pt)), "pr_auc": float(average_precision_score(yt, pt)),
            "val_selected_threshold": t_star,
            "at_0.5": at_threshold(yt, pt, 0.5),
            "at_val_selected": at_threshold(yt, pt, t_star),
            "at_matched_fpr_0.425": at_threshold(yt, pt, t_match),
            "reference": {"crop_trained_binary_auc": 0.759, "three_class_gate_auc": 0.732},
        }
        save_json(OUT / "summary.json", results)
        _report(results)
        logger.info("=== DONE in %.1fs | AUC %.3f (crop-bin 0.759, 3class 0.732) | "
                    "@0.5 FPR %.3f FNR %.3f | @val FPR %.3f FNR %.3f ===",
                    time.perf_counter() - t0, results["roc_auc"],
                    results["at_0.5"]["fpr"], results["at_0.5"]["fnr"],
                    results["at_val_selected"]["fpr"], results["at_val_selected"]["fnr"])
    finally:
        logger.removeHandler(h)
        h.close()


def _report(r):
    lines = ["# Binary REST-vs-MI trained on continuous windows -- gate over the 5 min", "",
             f"{r['n_windows']} windows | MI prevalence {r['mi_prevalence']:.2f}", "",
             "## Separability (AUC), compared with the other two gates", "",
             "| Gate | ROC-AUC |", "|---|---|",
             f"| Binary trained on continuous (new) | {r['roc_auc']:.3f} |",
             f"| Binary trained on crops | {r['reference']['crop_trained_binary_auc']:.3f} |",
             f"| 3-class gate | {r['reference']['three_class_gate_auc']:.3f} |",
             f"| PR-AUC (new) | {r['pr_auc']:.3f} |", "",
             "## Operating points", "",
             "| Config | FPR | FNR | MI recall | Bal. acc |", "|---|---|---|---|---|"]
    for name, key in [("@0.5", "at_0.5"), ("@val-sel", "at_val_selected"),
                      ("@FPR=0.425 (match 3-class)", "at_matched_fpr_0.425")]:
        m = r[key]
        lines.append(f"| {name} | {m['fpr']:.3f} | {m['fnr']:.3f} | {m['recall_MI']:.3f} | "
                     f"{m['balanced_accuracy']:.3f} |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
