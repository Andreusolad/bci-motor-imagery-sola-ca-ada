r"""Closing the delta confound -- a controlled 2x2 double dissociation.

The feature-importance analysis (feature_importance.py) suggested that the
REST-vs-MI separation leans on the delta band (0.5-4 Hz) -- a low-frequency
difference between the pre-cue baseline (REST source) and the feedback window
(MI source) -- rather than on the mu/beta sensorimotor rhythms that are the
genuine signature of motor imagery. That is a hypothesis from band-ablation at
eval time. Here it is tested with a controlled retrain, complementing the
negative-class audit.

Design: the same winning pipeline (EEGSym + Euclidean Alignment, corrected
window, same channels/downsample/CAR/crops/subject-split/seed/architecture/
hyper-parameters). The only change is an extra zero-phase high-pass at 4 Hz that
removes the delta band, applied identically to every segment before EA. Four
models are trained -- {REST-vs-MI detection, LEFT-vs-RIGHT discrimination} x
{full-band, delta-removed} -- and the AUC is read.

Prediction (double dissociation):
  * REST-vs-MI  : AUC drops when delta is removed  -> detection was a LF confound.
  * LEFT-vs-RIGHT: AUC ~unchanged when delta is removed -> direction is real
    mu/beta physiology.

REST-vs-MI is run first (primary result) so a partial run already closes it.

Usage:  python delta_confound_control.py   (run from route_b/detection/, BCI_DATA set)
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import config3 as C                              # noqa: E402
from lib import crops3, ea_io, model3                     # noqa: E402
from lib.segments import Segment, extract_segments_for_sessions  # noqa: E402

from src.eegsym_study.eegsym import build_eegsym           # noqa: E402
from src.eegsym_study import config as eg                  # noqa: E402
from src.split import load_split, session_keys_for         # noqa: E402
from src.utils import get_logger, save_json, set_global_seed  # noqa: E402

from binary_rest_mi import (balance_binary, build_crops, make_ds,   # noqa: E402
                            aggregate_by_trial, binary_metrics)

import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402

logger = get_logger()
OUT = Path(__file__).resolve().parents[1] / "experiments" / "delta_confound"
HP_CUTOFF = 4.0                       # remove delta (0.5-4 Hz); keep theta/mu/beta/gamma
FS = C.FS_TARGET
LEFT, RIGHT = C.LEFT_ID, C.RIGHT_ID

# Full-band baselines from the identical winning pipeline (for reference in the
# report; the script also RE-RUNS full-band in-house so every pair is controlled).
REF = {"rvm_crop_auc": 0.865, "rvm_trial_auc": 0.887, "lr_trial_auc": 0.834}


def highpass(sig: np.ndarray) -> np.ndarray:
    b, a = butter(4, HP_CUTOFF / (FS / 2), btype="highpass")
    return filtfilt(b, a, sig, axis=1).astype(np.float32)


def remove_delta(segs: List[Segment]) -> List[Segment]:
    return [replace(s, signal=highpass(s.signal)) for s in segs]


# ---- LEFT-vs-RIGHT helpers (MI only; LEFT=0, RIGHT=1) ---------------------- #
def _n_crops(sig) -> int:
    return len(crops3.iter_crop_bounds(sig.shape[1]))


def balance_lr(segs: List[Segment], seed: int) -> List[Segment]:
    """Keep MI segments; subsample the majority direction so LEFT crops ~= RIGHT crops."""
    left = [s for s in segs if s.kind == "mi" and s.label == LEFT]
    right = [s for s in segs if s.kind == "mi" and s.label == RIGHT]
    cl = sum(_n_crops(s.signal) for s in left)
    cr = sum(_n_crops(s.signal) for s in right)
    rng = np.random.default_rng(seed)
    if cl > cr:
        keep = max(1, round(len(left) * cr / cl))
        left = [left[i] for i in rng.permutation(len(left))[:keep]]
    elif cr > cl:
        keep = max(1, round(len(right) * cl / cr))
        right = [right[i] for i in rng.permutation(len(right))[:keep]]
    return left + right


def build_crops_lr(segs: List[Segment], W: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, tids = [], [], []
    for s in segs:
        aligned = ea_io.apply_ea(W, s.signal)
        lab = 0 if s.label == LEFT else 1
        for a, b in crops3.iter_crop_bounds(aligned.shape[1]):
            X.append(aligned[:, a:b][:, :, None]); y.append(lab); tids.append(s.trial_id)
    return np.stack(X).astype(np.float32), np.asarray(y, np.int64), tids


# --------------------------------------------------------------------------- #
def fit_and_eval(tr, va, te, build_fn, tag: str) -> Dict:
    """Fit EA on train, build crops via build_fn, train EEGSym(2), eval crop+trial AUC."""
    from tensorflow import keras
    set_global_seed(C.RANDOM_SEED)
    d = OUT / tag; d.mkdir(parents=True, exist_ok=True)

    ea = ea_io.fit_ea([s.signal for s in tr]); ea_io.save_ea(ea, d / "ea_reference.json")
    W = ea_io.load_ea_matrix(d / "ea_reference.json")

    Xtr, ytr, _ = build_fn(tr, W)
    Xva, yva, _ = build_fn(va, W)
    Xte, yte, tids = build_fn(te, W)
    logger.info("[%s] crops train %d (cls %s) | val %d | test %d",
                tag, len(ytr), np.bincount(ytr).tolist(), len(yva), len(yte))

    model = build_eegsym(input_shape=C.INPUT_SHAPE, n_classes=2)
    model3.compile_model3(model)
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss",
               patience=eg.TRAIN.early_stopping_patience, restore_best_weights=True, verbose=0),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=eg.TRAIN.reduce_lr_factor,
               patience=eg.TRAIN.reduce_lr_patience, min_lr=eg.TRAIN.min_lr)]
    hist = model.fit(make_ds(Xtr, ytr, eg.TRAIN.batch_size, True, C.RANDOM_SEED),
                     validation_data=make_ds(Xva, yva, eg.TRAIN.batch_size, False, C.RANDOM_SEED),
                     epochs=eg.TRAIN.epochs, callbacks=cbs, verbose=0)
    model.save_weights(d / "weights.weights.h5")

    p = model.predict(Xte, batch_size=512, verbose=0)[:, 1]
    crop_m = binary_metrics(yte, p, "crop")
    yt, pt = aggregate_by_trial(p, tids, yte)
    trial_m = binary_metrics(yt, pt, "trial")
    logger.info("[%s] epochs %d | crop AUC %.3f | trial AUC %.3f",
                tag, len(hist.history["loss"]), crop_m["roc_auc"], trial_m["roc_auc"])
    return {"epochs": len(hist.history["loss"]), "crop": crop_m, "trial": trial_m}


def _save(results: Dict):
    save_json(OUT / "summary.json", {"design": "delta-confound control 2x2 (retrain, high-pass 4 Hz)",
                                     "hp_cutoff_hz": HP_CUTOFF, "reference_fullband": REF,
                                     "results": results})
    _report(results); _figure(results)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "delta_confound.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    results: Dict = {}
    try:
        split = load_split(C.SPLIT_JSON)
        logger.info("Extracting segments (once) ...")
        base = {sp: extract_segments_for_sessions(session_keys_for(split, sp))
                for sp in ("train", "val", "test")}
        nod = {sp: remove_delta(base[sp]) for sp in base}
        logger.info("delta removed (high-pass %.1f Hz) on all splits", HP_CUTOFF)

        # ---- TASK 1 (PRIMARY): REST vs MI detection -- full then no-delta ---- #
        for cond, src in (("full", base), ("nodelta", nod)):
            tr = balance_binary(src["train"], C.RANDOM_SEED)
            va = balance_binary(src["val"], C.RANDOM_SEED)
            te = balance_binary(src["test"], C.RANDOM_SEED)
            results[f"rvm_{cond}"] = fit_and_eval(tr, va, te, build_crops, f"rvm_{cond}")
            _save(results)   # incremental: primary result survives a partial run

        # ---- TASK 2 (CONTROL): LEFT vs RIGHT discrimination ----------------- #
        for cond, src in (("full", base), ("nodelta", nod)):
            tr = balance_lr(src["train"], C.RANDOM_SEED)
            va = balance_lr(src["val"], C.RANDOM_SEED)
            te = balance_lr(src["test"], C.RANDOM_SEED)
            results[f"lr_{cond}"] = fit_and_eval(tr, va, te, build_crops_lr, f"lr_{cond}")
            _save(results)

        logger.info("=== DONE in %.1fs ===", time.perf_counter() - t0)
    finally:
        logger.removeHandler(h)
        h.close()


def _delta(results, key):
    if f"{key}_full" in results and f"{key}_nodelta" in results:
        return (results[f"{key}_full"]["trial"]["roc_auc"],
                results[f"{key}_nodelta"]["trial"]["roc_auc"])
    return (None, None)


def _report(results):
    def row(name, key):
        f, n = _delta(results, key)
        if f is None:
            return f"| {name} | -- | -- | -- |"
        return f"| {name} | {f:.3f} | {n:.3f} | {n - f:+.3f} |"
    lines = ["# Closing the delta confound -- 2x2 control (retrain, high-pass 4 Hz)", "",
             "Same winning pipeline (EEGSym+EA, corrected window); the only change is a "
             f"zero-phase high-pass at {HP_CUTOFF:.0f} Hz that removes the delta band (0.5-4 Hz).", "",
             "## Trial AUC: full-band vs delta-removed", "",
             "| Task | Full-band | No delta | delta |", "|---|---|---|---|",
             row("REST-vs-MI (detection)", "rvm"),
             row("LEFT-vs-RIGHT (discrimination)", "lr"), "",
             "Prediction: REST-vs-MI detection should drop (it was a low-frequency "
             "confound) and L/R direction should hold (genuine mu/beta). If so, the delta "
             "confound is shown by double dissociation.", ""]
    for key, name in (("rvm_full", "REST-vs-MI full-band"), ("rvm_nodelta", "REST-vs-MI no-delta"),
                      ("lr_full", "L/R full-band"), ("lr_nodelta", "L/R no-delta")):
        if key in results:
            m = results[key]
            lines += [f"### {name} (n_ep {m['epochs']})",
                      f"- crop: AUC {m['crop']['roc_auc']:.3f}, bal-acc {m['crop']['balanced_accuracy']:.3f}",
                      f"- trial: AUC {m['trial']['roc_auc']:.3f}, bal-acc {m['trial']['balanced_accuracy']:.3f}", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _figure(results):
    tasks = [("REST-vs-MI\n(detection)", "rvm"), ("LEFT-vs-RIGHT\n(discrimination)", "lr")]
    full = [_delta(results, k)[0] for _, k in tasks]
    nod = [_delta(results, k)[1] for _, k in tasks]
    if any(v is None for v in full + nod):
        return
    x = np.arange(len(tasks)); w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.bar(x - w / 2, full, w, label="full-band (0.5-40 Hz)", color="#1f3a5f")
    ax.bar(x + w / 2, nod, w, label="no delta (high-pass 4 Hz)", color="#c0873a")
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance")
    ax.set(xticks=x, ylabel="ROC-AUC (trial)", ylim=(0.45, 0.95),
           title="Delta confound: detection vs discrimination double dissociation")
    ax.set_xticklabels([t for t, _ in tasks]); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "delta_dissociation.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
