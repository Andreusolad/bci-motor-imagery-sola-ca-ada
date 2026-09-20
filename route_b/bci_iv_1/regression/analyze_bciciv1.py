r"""BCI-IV-1 rigour add-ons for the EEGNet-regression decoder.

Three analyses the scoring discussion needs, all on this decoder's own numbers,
reusing the validated pipeline:

  1. Per-subject block-bootstrap CI on the official MSE.  Blocks = maximal runs
     of constant finite label (the natural exchangeable unit of a continuous
     stream); resample blocks with replacement, recompute MSE = sum(SSE)/sum(n).
  2. Detection-vs-direction decomposition.  The metric identity
        MSE = E[y^2] - E[2*o*y - o^2]
     is exact, so the gain over silence (MSE_zero - MSE) splits additively into
     a MI part and a rest part; and the MSE itself splits into rest cost,
     MI-magnitude cost (sign right, too timid) and MI-sign cost (sign wrong).
     Reported next to detection AUC (|o| separating MI from rest) and direction
     accuracy (sign(o)==sign(y) on MI samples).
  3. Causality note is textual (EA is fit on the eval signal, unsupervised but
     non-causal) -- printed here for the record.

Uses the Stieger-pretrained regressor fine-tuned per subject (best arm, MSE
~0.411).  Deterministic (seed 42); EMA alpha 0.3 as in evaluate.py.

Usage:  python analyze_bciciv1.py   (run from route_b/bci_iv_1/regression/, BCI_DATA set)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from preprocessing import CROP, HOP, DS, EuclideanAlignment, load_eval, make_crops, REAL_SUBJECTS
from evaluate import _ema, EMA_ALPHA
import train as T

OUT = Path(__file__).with_name("results")
B_BOOT = 5000
RNG = np.random.default_rng(42)


def continuous_output(model, subj: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (cont_per_sample_@1000, y1000) restricted to finite-label samples.

    Mirrors evaluate.evaluate exactly: EA on the eval signal (unsupervised,
    non-causal), sliding 1 s / 0.5 s windows, tanh prediction, causal EMA,
    nearest-window expansion to the 1000 Hz grid, clip to [-1, 1].
    """
    sig, y1000 = load_eval(subj)
    ea = EuclideanAlignment().fit([sig])
    aligned = ea.transform(sig)
    starts = list(range(0, aligned.shape[1] - CROP + 1, HOP))
    X = make_crops(aligned)
    pred_w = model.predict(X, batch_size=512, verbose=0).ravel().astype(np.float64)
    centres = np.clip(np.array([int(round((s + CROP / 2) * DS)) for s in starts]),
                      0, len(y1000) - 1)
    t = np.arange(len(y1000)); j = np.clip(np.searchsorted(centres, t), 1, len(centres) - 1)
    nearest = np.where((t - centres[j - 1]) <= (centres[j] - t), j - 1, j)
    cont = np.clip(_ema(pred_w, EMA_ALPHA)[nearest], -1.0, 1.0)
    keep = np.isfinite(y1000)
    return cont[keep], y1000[keep]


def constant_label_blocks(y: np.ndarray) -> list[np.ndarray]:
    """Indices grouped into maximal runs of identical label value."""
    change = np.flatnonzero(np.diff(y) != 0) + 1
    return np.split(np.arange(len(y)), change)


def block_bootstrap_ci(o: np.ndarray, y: np.ndarray, blocks: list[np.ndarray]) -> tuple[float, float]:
    """95% CI of the MSE by resampling constant-label blocks with replacement."""
    sse = np.array([np.sum((o[b] - y[b]) ** 2) for b in blocks])
    n = np.array([len(b) for b in blocks], dtype=np.float64)
    idx = RNG.integers(0, len(blocks), size=(B_BOOT, len(blocks)))
    boot = sse[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _auc(score: np.ndarray, pos: np.ndarray) -> float:
    """AUC via mean rank of the positive class (Mann-Whitney)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s = score[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = int(pos.sum()); n_neg = len(pos) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def analyze(o: np.ndarray, y: np.ndarray) -> dict:
    mse = float(np.mean((o - y) ** 2))
    mse_zero = float(np.mean(y ** 2))
    is_mi = np.abs(y) == 1
    is_rest = y == 0
    frac_mi = float(is_mi.mean()); frac_rest = float(is_rest.mean())
    # gain over silence, split additively (E[2oy - o^2])
    gain_mi = float(np.mean((2 * o * y - o ** 2)[is_mi]) * frac_mi)
    gain_rest = float(np.mean((-o ** 2)[is_rest]) * frac_rest)          # y=0 => -o^2
    # cost split: MSE = frac_rest*E[o^2|rest] + frac_mi*E[(o-y)^2|MI]
    cost_rest = float(np.mean((o ** 2)[is_rest]) * frac_rest)
    err_mi = (o - y)[is_mi]
    sign_ok = np.sign(o[is_mi]) == np.sign(y[is_mi])
    cost_mi_mag = float(np.mean((err_mi[sign_ok]) ** 2) * sign_ok.mean() * frac_mi) if sign_ok.any() else 0.0
    cost_mi_sign = float(np.mean((err_mi[~sign_ok]) ** 2) * (~sign_ok).mean() * frac_mi) if (~sign_ok).any() else 0.0
    # detection / direction
    auc_det = _auc(np.abs(o), is_mi)
    acc_dir = float((np.sign(o[is_mi]) == np.sign(y[is_mi])).mean())
    mag_mi = float(np.mean(np.abs(o[is_mi]))); mag_rest = float(np.mean(np.abs(o[is_rest])))
    return dict(mse=mse, mse_zero=mse_zero, frac_mi=frac_mi,
                gain_mi=gain_mi, gain_rest=gain_rest,
                cost_rest=cost_rest, cost_mi_mag=cost_mi_mag, cost_mi_sign=cost_mi_sign,
                auc_det=auc_det, acc_dir=acc_dir, mag_mi=mag_mi, mag_rest=mag_rest)


def main():
    from tensorflow import keras
    t0 = time.perf_counter()
    rows = []
    for s in REAL_SUBJECTS:
        model, _ = T.finetune(s)
        o, y = continuous_output(model, s)
        keras.backend.clear_session()
        a = analyze(o, y)
        lo, hi = block_bootstrap_ci(o, y, constant_label_blocks(y))
        a.update(subject=s, ci95=[round(lo, 4), round(hi, 4)])
        rows.append(a)
        print(f"ds1{s}: MSE={a['mse']:.4f} CI[{lo:.4f},{hi:.4f}] "
              f"AUCdet={a['auc_det']:.3f} accdir={a['acc_dir']:.3f} "
              f"gainMI={a['gain_mi']:+.4f} gainRest={a['gain_rest']:+.4f}")

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    summary = dict(
        arm="EEGNet-regression pretrain(Stieger)+fine-tune, EMA a=0.3",
        note_ea="EA fitted on the eval signal per subject: unsupervised (no labels) "
                "but non-causal (uses the whole recording's covariance). Disclosed for scoring.",
        b_boot=B_BOOT, seed=42,
        mean_mse=round(m("mse"), 4), mean_auc_det=round(m("auc_det"), 4),
        mean_acc_dir=round(m("acc_dir"), 4), mean_gain_mi=round(m("gain_mi"), 4),
        mean_gain_rest=round(m("gain_rest"), 4),
        mean_cost_rest=round(m("cost_rest"), 4), mean_cost_mi_mag=round(m("cost_mi_mag"), 4),
        mean_cost_mi_sign=round(m("cost_mi_sign"), 4),
        per_subject=rows, runtime_s=round(time.perf_counter() - t0, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bciciv1_rigour.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== means ===")
    print(f"MSE {summary['mean_mse']:.4f} | AUCdet {summary['mean_auc_det']:.3f} | "
          f"accdir {summary['mean_acc_dir']:.3f}")
    print(f"gain MI {summary['mean_gain_mi']:+.4f} | gain rest {summary['mean_gain_rest']:+.4f}")
    print(f"cost: rest {summary['mean_cost_rest']:.4f} | MI-mag {summary['mean_cost_mi_mag']:.4f} | "
          f"MI-sign {summary['mean_cost_mi_sign']:.4f}")
    print(f"saved -> {OUT / 'bciciv1_rigour.json'}")


if __name__ == "__main__":
    main()
