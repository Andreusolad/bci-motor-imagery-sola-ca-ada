r"""Steps 1+2: score the decoder with the official BCI-IV-1 competition metric.

The BCI-IV-1 competition is scored not by accuracy but by mean squared error
between a continuous output in [-1,+1] and the per-sample true label
(-1 / 0 / +1, NaN = transition, excluded). Lower is better. Fixed references
(official): output 0 = 0.509; 2009 champion = 0.382.

The 3-class EEGSym+EA decoder (Stieger, zero-shot) is reused exactly as in
bciciv1_eval.py, changing only the output and the metric:

  * soft output (step 1): y_cont = P(right) - P(left) in [-1,+1] per window.
  * hard output (step 2): argmax -> {REST:0, LEFT:-1, RIGHT:+1}.

Each per-window value is expanded to the 1000 Hz sample grid (nearest window
centre) and MSE is computed over non-NaN samples, per subject and pooled.

Correctness gate (reproduces 0.5094 vs the official 0.509): the zero-output MSE
= mean(label^2) over non-NaN samples. If the pooled zero-output MSE is not
~0.509 the metric was misunderstood.

Subjects a,b,f,g are the real subjects; c,d,e are artificially generated in the
official BCI-IV-1 description (reported separately). a,f are left/foot; the MSE
metric scores +/-1 regardless of body part, as the competition did.

Usage:  python bciciv1_mse_eval.py   (run from route_b/bci_iv_1/, BCI_DATA set)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import config3 as C                    # noqa: E402
from lib import ea_io, model3                    # noqa: E402
from src.preprocessing import preprocess_trial   # noqa: E402
from src.utils import get_logger, save_json      # noqa: E402

logger = get_logger()
EVAL = C.BCI_DATA / "bci_iv_1" / "BCICIV_1eval_1000Hz_mat"
LAB = C.BCI_DATA / "bci_iv_1" / "true_labels_official" / "mat"
W3 = Path(__file__).resolve().parents[1] / "experiments" / "rest_3class" / "weights.weights.h5"
OUT = Path(__file__).resolve().parents[1] / "experiments" / "bciciv1_mse"
OUR8 = ["FC3", "FCZ", "FC4", "C3", "CZ", "C4", "CP3", "CP4"]
FS0, DS = 1000, 4
SUBJECTS = list("abcdefg")
REAL = set("abfg")                        # real subjects (c,d,e generated)
CROP = C.CROP_SAMPLES                      # 250
HOP = CROP // 2                            # 125
REST, LEFT, RIGHT = C.REST_ID, C.LEFT_ID, C.RIGHT_ID


def channel_index(clab: List[str]) -> List[int]:
    low = {c.lower(): i for i, c in enumerate(clab)}
    return [low[ch.lower()] for ch in OUR8]


def load_eval(subj: str):
    m = loadmat(str(EVAL / f"BCICIV_eval_ds1{subj}_1000Hz.mat"), struct_as_record=False, squeeze_me=True)
    clab = [str(c) for c in np.asarray(m["nfo"].clab).ravel()]
    idx = channel_index(clab)
    cnt = np.asarray(m["cnt"])[:, idx].astype(np.float32).T * 0.1        # (8, N) uV @1000
    ymat = loadmat(str(LAB / f"BCICIV_eval_ds1{subj}_1000Hz_true_y.mat"), squeeze_me=True)
    yk = max((k for k in ymat if not k.startswith("__")), key=lambda k: np.asarray(ymat[k]).size)
    y = np.asarray(ymat[yk]).astype(np.float64).ravel()[:cnt.shape[1]]   # crop to cnt (subj 'a')
    return cnt, y


def window_outputs(model, subj: str):
    """Return per-1000Hz-sample soft & hard outputs + the true label vector."""
    cnt, y1000 = load_eval(subj)
    sig = preprocess_trial(cnt)                       # (8, M) @250
    M = sig.shape[1]
    # EA per subject (unsupervised), same helper/flow as bciciv1_eval.py
    ea = ea_io.fit_ea([sig]); d = OUT / subj; d.mkdir(parents=True, exist_ok=True)
    ea_io.save_ea(ea, d / "ea_reference.json"); Wm = ea_io.load_ea_matrix(d / "ea_reference.json")
    aligned = ea_io.apply_ea(Wm, sig)

    starts = list(range(0, M - CROP + 1, HOP))
    X = np.stack([aligned[:, s:s + CROP][:, :, None] for s in starts]).astype(np.float32)
    probs = model.predict(X, batch_size=512, verbose=0)          # (n, 3) softmax
    soft_w = probs[:, RIGHT] - probs[:, LEFT]                     # in [-1, 1]
    hard_map = np.array([0.0, -1.0, 1.0])                        # REST,LEFT,RIGHT -> 0,-1,+1
    hard_w = hard_map[probs.argmax(1)]
    centres = np.clip(np.array([int(round((s + CROP / 2) * DS)) for s in starts]),
                      0, len(y1000) - 1)
    # expand per-window value to every 1000Hz sample via nearest window centre
    t = np.arange(len(y1000))
    j = np.searchsorted(centres, t)
    j = np.clip(j, 1, len(centres) - 1)
    left_closer = (t - centres[j - 1]) <= (centres[j] - t)
    nearest = np.where(left_closer, j - 1, j)
    return soft_w[nearest], hard_w[nearest], y1000


def mse_nonan(pred: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    keep = np.isfinite(y)
    return float(np.mean((pred[keep] - y[keep]) ** 2)), int(keep.sum())


def subject_mse(model, subj: str) -> Dict:
    soft, hard, y = window_outputs(model, subj)
    zero = np.zeros_like(soft)
    mse_soft, n = mse_nonan(soft, y)
    mse_hard, _ = mse_nonan(hard, y)
    mse_zero, _ = mse_nonan(zero, y)
    return {"subject": subj, "real": subj in REAL, "n_scored": n,
            "mse_zero": mse_zero, "mse_soft": mse_soft, "mse_hard": mse_hard,
            "beats_baseline_soft": mse_soft < mse_zero, "beats_baseline_hard": mse_hard < mse_zero}


def _pooled(rows, key, ref="mse_zero"):
    """Sample-weighted mean of a per-subject MSE (each subject weighted by n_scored)."""
    w = np.array([r["n_scored"] for r in rows], float)
    v = np.array([r[key] for r in rows], float)
    return float((w * v).sum() / w.sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "bciciv1_mse.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        model = model3.build_model3("eegsym"); model.load_weights(str(W3))
        logger.info("loaded 3-class EEGSym+EA (Stieger). Scoring BCI-IV-1 eval with competition MSE...")
        results = []
        for s in SUBJECTS:
            r = subject_mse(model, s)
            results.append(r)
            logger.info("ds1%s %-9s n=%d  MSE zero=%.4f  soft=%.4f  hard=%.4f",
                        s, "(real)" if r["real"] else "(gen.)", r["n_scored"],
                        r["mse_zero"], r["mse_soft"], r["mse_hard"])
        real = [r for r in results if r["real"]]
        allr = results
        summary = {
            "design": "3-class EEGSym+EA (Stieger) scored with the official BCI-IV-1 competition "
                      "MSE (continuous output vs -1/0/+1 label, NaN excluded). Soft=P(R)-P(L), "
                      "hard=argmax->{0,-1,+1}. References: baseline 0.509, champion 0.382.",
            "gate_zero_output_mse": {"real_subjects": _pooled(real, "mse_zero"),
                                     "all7": _pooled(allr, "mse_zero"),
                                     "expected_official": 0.509},
            "pooled_real_subjects": {"soft": _pooled(real, "mse_soft"), "hard": _pooled(real, "mse_hard"),
                                     "zero": _pooled(real, "mse_zero")},
            "pooled_all7": {"soft": _pooled(allr, "mse_soft"), "hard": _pooled(allr, "mse_hard"),
                            "zero": _pooled(allr, "mse_zero")},
            "results": results}
        save_json(OUT / "summary.json", summary)
        _report(summary)
        g = summary["gate_zero_output_mse"]["real_subjects"]
        logger.info("GATE zero-output MSE (real) = %.4f (official 0.509) %s",
                    g, "OK" if abs(g - 0.509) < 0.02 else "CHECK")
        logger.info("POOLED real: soft=%.4f hard=%.4f (baseline %.4f)",
                    summary["pooled_real_subjects"]["soft"], summary["pooled_real_subjects"]["hard"],
                    summary["pooled_real_subjects"]["zero"])
        logger.info("=== DONE in %.1fs ===", time.perf_counter() - t0)
    finally:
        logger.removeHandler(h); h.close()


def _report(s: Dict):
    g = s["gate_zero_output_mse"]; pr = s["pooled_real_subjects"]
    lines = ["# BCI-IV-1 eval -- decoder with the official metric (competition MSE)", "",
             "MSE between a continuous output in [-1,+1] and the -1/0/+1 label (NaN excluded); "
             "lower is better. References: silence (0-output)=0.509, 2009 champion=0.382.", "",
             f"Correctness gate (0-output = mean(label^2)): real = {g['real_subjects']:.4f} "
             f"(official 0.509), 7 subjects = {g['all7']:.4f}. "
             + ("Matches -> metric understood." if abs(g['real_subjects'] - 0.509) < 0.02
                else "Does not match 0.509 -> check."), "",
             "| Subj | type | n | MSE 0-output | MSE soft | MSE hard | beats silence? |",
             "|---|---|---|---|---|---|---|"]
    for r in s["results"]:
        tip = "real" if r["real"] else "gen."
        bt = "yes (soft)" if r["beats_baseline_soft"] else ("yes (hard only)" if r["beats_baseline_hard"] else "no")
        lines.append(f"| ds1{r['subject']} | {tip} | {r['n_scored']} | {r['mse_zero']:.4f} | "
                     f"{r['mse_soft']:.4f} | {r['mse_hard']:.4f} | {bt} |")
    lines += ["",
              f"- Pooled (real subjects a,b,f,g): soft = {pr['soft']:.4f}, "
              f"hard = {pr['hard']:.4f}, silence = {pr['zero']:.4f}.",
              f"- Soft vs hard: {'soft wins (lower MSE); committing hard pays 4x under squared error' if pr['soft'] < pr['hard'] else 'hard wins (unexpected)'}.",
              f"- Beats 0.509? {'yes' if pr['soft'] < 0.509 else 'no'} with soft output "
              f"({pr['soft']:.4f}). {'Cross-architecture corroboration on the external arbiter.' if pr['soft'] < 0.509 else ''}",
              "",
              "Note: a,f are left/foot; the MSE scores +-1 regardless of body part (as the "
              "competition did). c,d,e are artificially generated subjects (official BCI-IV-1 "
              "description); they are listed but the honest pooled figure uses only real subjects."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
