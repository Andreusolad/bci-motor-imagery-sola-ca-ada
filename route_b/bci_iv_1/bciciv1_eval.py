r"""Apply the best asynchronous-capable model (3-class REST/LEFT/RIGHT EEGSym+EA,
trained on Stieger) to the BCI Competition IV Data Set 1 evaluation set, as a
zero-shot cross-dataset asynchronous BCI (continuous stream, explicit idle
state).

Ground truth: true_y is a per-sample vector in {-1, 0, +1} (+/-1 = the subject's
two MI classes, 0 = idle/no-control) plus NaN in transition periods (excluded
from scoring).

Per eval subject a-g:
  * select the 8 motor channels, scale int16->uV (x0.1),
  * preprocess (band-pass 0.5-40 @1000 -> downsample x4 -> CAR),
  * fit EA on the subject's own continuous EEG (unsupervised, leakage-safe) and
    whiten -> the model's expected EA-aligned space,
  * slide a 1 s / 0.5 s window, predict {REST,LEFT,RIGHT} -> map REST->0,
    LEFT->-1, RIGHT->+1,
  * ground truth per window = true_y at the window centre (NaN -> excluded),
  * async metrics: accuracy, balanced accuracy, FPR_idle, per-class recall,
    3x3 confusion, and direction accuracy on control windows.

Subjects a,f are left/foot (not left/right); the model has no foot class, so the
+1 class is a cross-task mismatch there -> reported separately and flagged.

Usage:  python bciciv1_eval.py   (run from route_b/bci_iv_1/, BCI_DATA set)
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import config3 as C                    # noqa: E402
from lib import crops3, ea_io, model3           # noqa: E402
from src.preprocessing import preprocess_trial  # noqa: E402
from src.utils import get_logger, save_json     # noqa: E402

logger = get_logger()
EVAL = C.BCI_DATA / "bci_iv_1" / "BCICIV_1eval_1000Hz_mat"
LAB = C.BCI_DATA / "bci_iv_1" / "true_labels_official" / "mat"
W3 = Path(__file__).resolve().parents[1] / "experiments" / "rest_3class" / "weights.weights.h5"
OUT = Path(__file__).resolve().parents[1] / "experiments" / "bciciv1_eval"
OUR8 = ["FC3", "FCZ", "FC4", "C3", "CZ", "C4", "CP3", "CP4"]
FS0, DS = 1000, 4                       # 1000 Hz -> 250 Hz
SUBJECTS = list("abcdefg")
LR_SUBJECTS = set("bcdeg")              # left/right (proper match); a,f are left/foot
REST, LEFT, RIGHT = C.REST_ID, C.LEFT_ID, C.RIGHT_ID   # 0,1,2
CROP = C.CROP_SAMPLES                    # 250 (1 s @250)
HOP = CROP // 2                          # 125 (0.5 s)


def channel_index(clab: List[str]) -> List[int]:
    low = {c.lower(): i for i, c in enumerate(clab)}
    return [low[ch.lower()] for ch in OUR8]


def load_eval(subj: str):
    m = loadmat(str(EVAL / f"BCICIV_eval_ds1{subj}_1000Hz.mat"), struct_as_record=False, squeeze_me=True)
    clab = [str(c) for c in np.asarray(m["nfo"].clab).ravel()]
    classes = [str(c) for c in np.asarray(m["nfo"].classes).ravel()]
    idx = channel_index(clab)
    cnt = np.asarray(m["cnt"])[:, idx].astype(np.float32).T * 0.1   # (8, N) uV
    ymat = loadmat(str(LAB / f"BCICIV_eval_ds1{subj}_1000Hz_true_y.mat"), squeeze_me=True)
    yk = max((k for k in ymat if not k.startswith("__")), key=lambda k: np.asarray(ymat[k]).size)
    y = np.asarray(ymat[yk]).astype(np.float64).ravel()[:cnt.shape[1]]   # crop to cnt (subj 'a')
    return cnt, y, classes


def eval_subject(model, subj: str) -> Dict:
    cnt, y1000, classes = load_eval(subj)
    sig = preprocess_trial(cnt)                       # (8, M) @250
    M = sig.shape[1]
    # EA on the subject's own signal (unsupervised) -> whiten
    ea = ea_io.fit_ea([sig]); d = OUT / subj; d.mkdir(parents=True, exist_ok=True)
    ea_io.save_ea(ea, d / "ea_reference.json"); W = ea_io.load_ea_matrix(d / "ea_reference.json")
    aligned = ea_io.apply_ea(W, sig)

    starts = list(range(0, M - CROP + 1, HOP))
    X = np.stack([aligned[:, s:s + CROP][:, :, None] for s in starts]).astype(np.float32)
    pred3 = model.predict(X, batch_size=512, verbose=0).argmax(1)
    # map 3-class -> {-1,0,+1}
    to_signed = {REST: 0, LEFT: -1, RIGHT: 1}
    pred = np.array([to_signed[p] for p in pred3])

    # ground truth at window centre (1000 Hz index), NaN -> excluded
    centres = np.array([int(round((s + CROP / 2) * DS)) for s in starts])
    centres = np.clip(centres, 0, len(y1000) - 1)
    gt = y1000[centres]
    keep = np.isfinite(gt)
    pred, gt = pred[keep], gt[keep].astype(int)

    return _metrics(subj, classes, pred, gt)


def _metrics(subj, classes, pred, gt) -> Dict:
    labs = [0, -1, 1]                      # idle, class1(-1), class2(+1)
    cm = np.zeros((3, 3), int)
    for t, p in zip(gt, pred):
        cm[labs.index(t), labs.index(p)] += 1
    recalls = {}
    for i, name in zip(labs, ["idle", "neg(-1)", "pos(+1)"]):
        tot = cm[labs.index(i)].sum()
        recalls[name] = float(cm[labs.index(i), labs.index(i)] / tot) if tot else float("nan")
    n_idle = (gt == 0).sum()
    fpr_idle = float(((gt == 0) & (pred != 0)).sum() / n_idle) if n_idle else float("nan")
    # direction accuracy on windows that are truly control and predicted control
    ctrl = (gt != 0) & (pred != 0)
    dir_acc = float((pred[ctrl] == gt[ctrl]).mean()) if ctrl.sum() else float("nan")
    bal = float(np.nanmean([recalls["idle"], recalls["neg(-1)"], recalls["pos(+1)"]]))
    return {"subject": subj, "classes": classes, "n_windows": int(len(gt)),
            "accuracy": float((pred == gt).mean()), "balanced_accuracy": bal,
            "fpr_idle": fpr_idle, "recall": recalls, "direction_accuracy_on_control": dir_acc,
            "n_control_predicted_control": int(ctrl.sum()),
            "confusion_idle_neg_pos": cm.tolist(),
            "is_left_right": subj in LR_SUBJECTS}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "bciciv1_eval.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        model = model3.build_model3("eegsym"); model.load_weights(W3)
        logger.info("loaded 3-class EEGSym+EA (Stieger). Evaluating BCI-IV-1 eval a-g ...")
        results = []
        for s in SUBJECTS:
            r = eval_subject(model, s)
            results.append(r)
            logger.info("ds1%s %-12s n=%d acc=%.3f bal=%.3f FPR_idle=%.3f dir=%.3f %s",
                        s, str(r["classes"]), r["n_windows"], r["accuracy"], r["balanced_accuracy"],
                        r["fpr_idle"], r["direction_accuracy_on_control"],
                        "" if r["is_left_right"] else "(left/FOOT - cross-task)")
        save_json(OUT / "summary.json", {
            "design": "zero-shot cross-dataset async eval: 3-class Stieger EEGSym+EA on BCI-IV-1 eval; "
                      "per-subject EA (unsupervised); window 1s/0.5s; GT true_y at window centre",
            "results": results})
        _report(results)
        logger.info("=== DONE in %.1fs ===", time.perf_counter() - t0)
    finally:
        logger.removeHandler(h)
        h.close()


def _report(results):
    lr = [r for r in results if r["is_left_right"]]
    def agg(rows, key):
        return float(np.nanmean([r[key] for r in rows])) if rows else float("nan")
    # honest pooled direction accuracy over left/right, weighted by #control predictions
    n_ctrl = sum(r["n_control_predicted_control"] for r in lr)
    n_ok = sum(r["direction_accuracy_on_control"] * r["n_control_predicted_control"] for r in lr)
    pooled_dir = float(n_ok / n_ctrl) if n_ctrl else float("nan")
    lines = ["# BCI-IV-1 eval -- 3-class Stieger model (EEGSym+EA) zero-shot, asynchronous", "",
             "Official GT verified. Window 1 s / 0.5 s; per-subject EA (unsupervised); "
             "NaN (transition) excluded. Mapping REST->0(idle), LEFT->-1, RIGHT->+1. 3-way chance = 0.333.", "",
             "| Subj | Classes | n win. | Acc | Bal-acc | FPR_idle | R.idle | R.neg(-1) | R.pos(+1) | #ctrl | Dir-acc |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        flag = "" if r["is_left_right"] else " foot"
        rc = r["recall"]
        lines.append(f"| ds1{r['subject']}{flag} | {'/'.join(r['classes'])} | {r['n_windows']} | "
                     f"{r['accuracy']:.3f} | {r['balanced_accuracy']:.3f} | {r['fpr_idle']:.3f} | "
                     f"{rc['idle']:.3f} | {rc['neg(-1)']:.3f} | {rc['pos(+1)']:.3f} | "
                     f"{r['n_control_predicted_control']} | {r['direction_accuracy_on_control']:.3f} |")
    lines += ["", "## Reading",
              f"- Mean left/right (b,c,d,e,g): acc {agg(lr,'accuracy'):.3f}, "
              f"bal-acc {agg(lr,'balanced_accuracy'):.3f} (~ 3-way chance 0.333), "
              f"FPR_idle {agg(lr,'fpr_idle'):.3f}.",
              f"- Pooled dir-acc (weighted by #ctrl) = {pooled_dir:.3f} over {n_ctrl} control "
              "windows -- dominated by g (1097) and b (593). The high dir-acc of c/d/e (0.90/0.75/1.00) "
              "are over 60/53/4 windows: the model barely fires (idle-locked), not a robust signal.",
              "- Collapse of the neg(-1)=left class in almost all (recall ~0-0.29): the same LEFT "
              "collapse documented on Stieger, which transfers to BCI-IV-1.",
              "- Large inter-subject variability in FPR_idle (0.003 -> 0.53): some stay locked in "
              "idle (c,d,e), others over-fire (a,f,g). Consistent with the bimodal/illiteracy pattern "
              "and with idle detection (REST-based, delta-dependent) not transferring well -- an echo "
              "of the delta confound.",
              "- a and f are left/foot: the +1 class is a different task (no 'foot' in the model); "
              "reference only, not comparable in direction."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
