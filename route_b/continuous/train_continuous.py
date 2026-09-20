r"""NEW PIPELINE (variant a): train the 3-class EEGSym on windows derived from
CONTINUOUS records (concatenated real segments), then evaluate on the same
continuous test streams -- a paired A/B against the trial-crop-trained model.

MOTIVATION (see the domain-shift analysis): the current model is trained on
isolated MI/REST trial crops but deployed on continuous EEG. Two genuine
train/deploy mismatches: (1) it never sees BOUNDARY windows that straddle a
REST<->MI transition, and (2) its training prior is balanced while deployment is
~94 % REST. Training on continuous-derived windows exposes the model to the
boundary content and continuous statistics (the asynchronous-BCI paradigm:
Mason & Birch 2000; Millan; Scherer/Pfurtscheller).

COMPUTE CHOICE (documented, honest): the full ~94 %-REST prior with adequate MI
coverage is infeasible on CPU (would need ~10^5 windows/epoch). We therefore
train on continuous-derived windows but SUBSAMPLE pure-REST windows so the
training set is class-balanced (cost-sensitive), keeping ALL MI and ALL boundary
windows. This isolates the "continuous-derived + boundary windows" factor at
feasible cost; the deployment prior remains a decision-time concern (threshold /
HMM), exactly as for the trial model (which was also balanced).

LEAKAGE CONTROL (the hard requirement)
--------------------------------------
Separation is enforced at the SUBJECT level BEFORE any record is generated:
  * The subject split (36 train / 13 val / 13 test, seed 42) is fixed first.
  * Every continuous record is built EXCLUSIVELY from the segment pool of subjects
    in ONE set -> no segment, trial, session or window can cross sets, because a
    subject belongs to exactly one set and a record never mixes subjects across
    sets.
  * EA is fit on TRAIN pool segments only. Early stopping uses VAL records only.
  * The test streams are built only from TEST subjects and are regenerated with
    the SAME rng as continuous_eval.py, so the flat and continuous models are
    compared on IDENTICAL streams.
An explicit assertion checks the three subject sets are disjoint.

Usage:  python train_continuous.py   (run from route_b/continuous/, BCI_DATA set)
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
from lib.segments import SubjectPool, load_or_build_subject_pool

from src.split import load_split  # noqa: E402
from src.utils import get_logger, save_json, set_global_seed  # noqa: E402

logger = get_logger()
OUT = C.EXPERIMENTS / "continuous_trained"
FLAT_DIR = C.EXPERIMENTS / "rest_3class"
REST, LEFT, RIGHT = C.REST_ID, C.LEFT_ID, C.RIGHT_ID

N_TRAIN_SUBJECTS = 15
N_VAL_SUBJECTS = 6
RECORDS_PER_TRAIN = 8
RECORDS_PER_VAL = 4
GEN_SEED = 2024
MAX_EPOCHS = 45
PATIENCE = 10
BATCH = 256


# --------------------------------------------------------------------------- #
def _window_label(sample_gt: np.ndarray, s: int, e: int) -> Tuple[int, bool]:
    seg = sample_gt[s:e]
    lab = int(np.bincount(seg, minlength=C.N_CLASSES).argmax())
    pure_rest = bool(np.all(seg == REST))
    return lab, pure_rest


def build_windows(pools: Dict[str, SubjectPool], subjects: List[str], n_records: int,
                  W3: np.ndarray, seed_base: int, rng_seed: int, balance_rest: bool
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Slide windows over continuous records; optionally subsample pure-REST windows."""
    X_info, y_info, X_rest = [], [], []
    for subject in subjects:
        pool = pools[subject]
        for r in range(n_records):
            rrng = np.random.default_rng([seed_base, int(subject[1:]), r])
            trial = gen.build_continuous_trial(pool, r, rrng, seed=seed_base)
            norm = ea_io.apply_ea(W3, trial.signal)
            for s, e in crops3.iter_crop_bounds(norm.shape[1]):
                lab, pure_rest = _window_label(trial.ground_truth, s, e)
                win = norm[:, s:e][:, :, None].astype(np.float32)
                if pure_rest:
                    X_rest.append(win)
                else:
                    X_info.append(win); y_info.append(lab)
    X_info = np.asarray(X_info, np.float32); y_info = np.asarray(y_info, np.int64)
    X_rest = np.asarray(X_rest, np.float32)
    n_keep = len(X_info) if balance_rest else len(X_rest)
    n_keep = min(n_keep, len(X_rest))
    rng = np.random.default_rng(rng_seed)
    keep = rng.permutation(len(X_rest))[:n_keep]
    X = np.concatenate([X_info, X_rest[keep]], axis=0)
    y = np.concatenate([y_info, np.full(n_keep, REST, np.int64)], axis=0)
    order = rng.permutation(len(X))
    return X[order], y[order]


def make_ds(X, y, sample_w, batch, shuffle, seed):
    import tensorflow as tf
    y1 = tf.one_hot(y, depth=C.N_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((X, y1, sample_w))
    if shuffle:
        ds = ds.shuffle(min(len(X), 20000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def infer_stream(model, W3, trial) -> Tuple[np.ndarray, np.ndarray]:
    norm = ea_io.apply_ea(W3, trial.signal)
    bounds = crops3.iter_crop_bounds(norm.shape[1])
    X = np.stack([norm[:, s:e][:, :, None] for s, e in bounds]).astype(np.float32)
    prob = model.predict(X, batch_size=512, verbose=0)
    gt = np.array([int(np.bincount(trial.ground_truth[s:e], minlength=C.N_CLASSES).argmax())
                   for s, e in bounds])
    return gt, prob


def eval_metrics(gt, pred) -> Dict:
    m = metrics3.compute_metrics(gt, pred)
    b = metrics3.bci_continuous_metrics(gt, pred)
    mi_t, mi_p = gt != REST, pred != REST
    tp = int((mi_t & mi_p).sum()); fp = int((~mi_t & mi_p).sum()); fn = int((mi_t & ~mi_p).sum())
    return {"accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
            "precision_macro": m["precision_macro"], "recall_macro": m["recall_macro"],
            "f1_macro": m["f1_macro"], "per_class_accuracy": m["per_class_accuracy"],
            "fpr": b["false_positive_rate_rest_as_mi"], "fnr": b["false_negative_rate_mi_as_rest"],
            "mi_precision": (tp / (tp + fp)) if (tp + fp) else float("nan"),
            "mi_recall": (tp / (tp + fn)) if (tp + fn) else float("nan"),
            "confusion_matrix": m["confusion_matrix"]}


def _agg(rows, keys):
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "std": float(np.std([r[k] for r in rows], ddof=1))} for k in keys}


# --------------------------------------------------------------------------- #
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "train_continuous.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        import tensorflow as tf  # noqa: F401
        set_global_seed(C.RANDOM_SEED)
        split = load_split(C.SPLIT_JSON)
        tr_all = sorted(split["splits"]["train"]["subjects"], key=lambda s: int(s[1:]))
        va_all = sorted(split["splits"]["val"]["subjects"], key=lambda s: int(s[1:]))
        te_all = sorted(split["splits"]["test"]["subjects"], key=lambda s: int(s[1:]))
        # leakage guard: subject sets disjoint
        assert set(tr_all).isdisjoint(va_all) and set(tr_all).isdisjoint(te_all) \
            and set(va_all).isdisjoint(te_all), "subject sets overlap!"
        train_subs = tr_all[:N_TRAIN_SUBJECTS]
        val_subs = va_all[:N_VAL_SUBJECTS]
        logger.info("train subjects (%d): %s", len(train_subs), train_subs)
        logger.info("val subjects (%d): %s", len(val_subs), val_subs)

        pools = {s: load_or_build_subject_pool(s, split) for s in train_subs + val_subs}

        # EA on TRAIN pool segments only
        train_sig = []
        for s in train_subs:
            train_sig += pools[s].mi_left + pools[s].mi_right + pools[s].rest
        ea = ea_io.fit_ea(train_sig)
        ea_io.save_ea(ea, OUT / "ea_reference.json")
        W3 = ea_io.load_ea_matrix(OUT / "ea_reference.json")
        logger.info("EA fit on %d train segments.", len(train_sig))

        Xtr, ytr = build_windows(pools, train_subs, RECORDS_PER_TRAIN, W3, GEN_SEED, 1, True)
        Xva, yva = build_windows(pools, val_subs, RECORDS_PER_VAL, W3, GEN_SEED + 1, 2, True)
        counts = np.bincount(ytr, minlength=C.N_CLASSES)
        logger.info("train windows: %d %s | val windows: %d", len(Xtr),
                    {C.CLASS_NAMES[i]: int(counts[i]) for i in range(3)}, len(Xva))

        # cost-sensitive sample weights (balanced)
        cw = len(ytr) / (C.N_CLASSES * np.maximum(counts, 1))
        wtr = cw[ytr].astype(np.float32)
        wva = np.ones(len(yva), np.float32)

        model = model3.build_model3("eegsym")
        model3.compile_model3(model)
        epoch_times: List[float] = []
        cbs = model3.build_callbacks(OUT, epoch_times)
        tr_ds = make_ds(Xtr, ytr, wtr, BATCH, True, C.RANDOM_SEED)
        va_ds = make_ds(Xva, yva, wva, BATCH, False, C.RANDOM_SEED)
        hist = model.fit(tr_ds, validation_data=va_ds, epochs=MAX_EPOCHS, callbacks=cbs, verbose=0)
        model.save_weights(OUT / "weights.weights.h5")
        n_ep = len(hist.history["loss"])
        logger.info("trained %d epochs.", n_ep)

        # ---- paired eval on IDENTICAL test streams: flat vs continuous ---- #
        flat = model3.build_model3("eegsym")
        flat.load_weights(FLAT_DIR / "weights.weights.h5")
        Wflat = ea_io.load_ea_matrix(FLAT_DIR / "ea_reference.json")

        test_subs = gen.select_subjects(split)
        test_pools = {s: load_or_build_subject_pool(s, split) for s in test_subs}
        KEYS = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
                "fpr", "fnr", "mi_precision", "mi_recall"]
        rows = {"flat": [], "continuous": []}
        save_pred = {"subject": [], "trial_key": [], "gt": [], "pred": [], "prob": []}
        for seed in C.CONT_SEEDS:
            g_c, p_c, g_f, p_f = [], [], [], []
            for subject in test_subs:
                for ti in range(C.CONT_N_TRIALS_PER_SUBJECT):
                    rng = np.random.default_rng([seed, int(subject[1:]), ti])
                    trial = gen.build_continuous_trial(test_pools[subject], ti, rng, seed=seed)
                    gt_c, pr_c = infer_stream(model, W3, trial)
                    gt_f, pr_f = infer_stream(flat, Wflat, trial)
                    g_c.append(gt_c); p_c.append(pr_c.argmax(1))
                    g_f.append(gt_f); p_f.append(pr_f.argmax(1))
                    if seed == 42:
                        n = len(gt_c)
                        save_pred["subject"] += [subject] * n
                        save_pred["trial_key"] += [f"t{ti}"] * n
                        save_pred["gt"].append(gt_c); save_pred["pred"].append(pr_c.argmax(1))
                        save_pred["prob"].append(pr_c)
            rows["continuous"].append(eval_metrics(np.concatenate(g_c), np.concatenate(p_c)))
            rows["flat"].append(eval_metrics(np.concatenate(g_f), np.concatenate(p_f)))
            logger.info("eval seed %d done", seed)

        np.savez_compressed(OUT / "predictions_seed42.npz",
                            subject=np.array(save_pred["subject"]),
                            trial_key=np.array(save_pred["trial_key"]),
                            gt=np.concatenate(save_pred["gt"]),
                            pred=np.concatenate(save_pred["pred"]),
                            prob=np.concatenate(save_pred["prob"]))

        agg = {m: _agg(rows[m], KEYS) for m in rows}
        # pooled confusion for the continuous model
        summary = {
            "design": "3-class EEGSym trained on continuous-derived windows (balanced), "
                      "evaluated on identical continuous test streams vs the trial-trained flat model",
            "train_subjects": train_subs, "val_subjects": val_subs, "test_subjects": test_subs,
            "records_per_train": RECORDS_PER_TRAIN, "n_train_windows": int(len(Xtr)),
            "epochs": n_ep, "seeds": list(C.CONT_SEEDS),
            "flat_vs_continuous": agg,
            "hparams": model3.training_hparams("eegsym"),
        }
        save_json(OUT / "summary.json", summary)
        _report(agg)
        logger.info("=== CONTINUOUS TRAINING DONE in %.1fs | flat bal_acc %.3f -> continuous %.3f | "
                    "flat MI-prec %.3f -> continuous %.3f | flat FPR %.3f -> continuous %.3f ===",
                    time.perf_counter() - t0,
                    agg["flat"]["balanced_accuracy"]["mean"], agg["continuous"]["balanced_accuracy"]["mean"],
                    agg["flat"]["mi_precision"]["mean"], agg["continuous"]["mi_precision"]["mean"],
                    agg["flat"]["fpr"]["mean"], agg["continuous"]["fpr"]["mean"])
    finally:
        logger.removeHandler(h)
        h.close()


def _report(agg):
    lab = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
           ("precision_macro", "Precision (macro)"), ("recall_macro", "Recall (macro)"),
           ("f1_macro", "F1 (macro)"), ("fpr", "FPR (rest->MI)"), ("fnr", "FNR (MI->rest)"),
           ("mi_precision", "MI detection precision"), ("mi_recall", "MI detection recall")]
    lines = ["# Training on continuous EEG vs. training on trials", "",
             "Paired evaluation on identical test streams (mean +/- std, 5 seeds).", "",
             "| Metric | Trial-trained (baseline) | Continuous-trained (new) |", "|---|---|---|"]
    for key, name in lab:
        f = agg["flat"][key]; c = agg["continuous"][key]
        lines.append(f"| {name} | {f['mean']:.3f}+-{f['std']:.3f} | {c['mean']:.3f}+-{c['std']:.3f} |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
