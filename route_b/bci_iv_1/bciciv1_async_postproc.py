r"""Lower the BCI-IV-1 competition MSE by post-processing the output.

Starting point: the 3-class EEGSym+EA (Stieger) applied to BCI-IV-1 eval gives a
per-window soft output P(right)-P(left); scored with the official MSE it reaches
0.4795 on the real subjects (beats the 0.509 silence baseline; the 2009 champion
got 0.382). This script does not retrain anything -- it only post-processes that
continuous output the way competition winners do, to minimize MSE:

  (a) temporal smoothing  -- causal EMA over the window sequence (removes jitter;
      the true label is smooth, so a smooth output scores much better).
  (b) hedging gain        -- multiply the output by g<=1 (a confident wrong answer
      costs 4x under MSE, so shrinking amplitude toward 0 when unsure pays off).
  (idle is handled for free: the 3-class model already outputs ~0 during rest.)

The (alpha, gain) hyper-parameters are chosen by leave-one-subject-out over the 4
real subjects (tune on 3, test on the 4th), so nothing is tuned on the subject it
is scored on. The oracle (best single setting on all real subjects) is also
reported as an upper bound / headroom.

Real subjects a,b,f,g; c,d,e are generated (reported apart). a,f are left/foot.

Usage:  python bciciv1_async_postproc.py   (run from route_b/bci_iv_1/, BCI_DATA set)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import model3, ea_io                        # noqa: E402
from lib import config3 as C                          # noqa: E402
from src.preprocessing import preprocess_trial        # noqa: E402
from src.utils import get_logger, save_json           # noqa: E402
from bciciv1_mse_eval import load_eval, CROP, HOP, DS, REST, LEFT, RIGHT  # noqa: E402

logger = get_logger()
W3 = Path(__file__).resolve().parents[1] / "experiments" / "rest_3class" / "weights.weights.h5"
OUT = Path(__file__).resolve().parents[1] / "experiments" / "bciciv1_postproc"
SUBJECTS = list("abcdefg")
REAL = list("abfg")
BASELINE_ZERO, CHAMPION = 0.509, 0.382
ALPHAS = [0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0]   # 1.0 = no smoothing
GAINS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]                # <1 = hedge toward 0


def per_window_output(model, subj: str):
    """Return (soft_w per window, window-centre sample index, per-sample label)."""
    cnt, y1000 = load_eval(subj)
    sig = preprocess_trial(cnt); M = sig.shape[1]
    ea = ea_io.fit_ea([sig]); d = OUT / subj; d.mkdir(parents=True, exist_ok=True)
    ea_io.save_ea(ea, d / "ea.json"); Wm = ea_io.load_ea_matrix(d / "ea.json")
    aligned = ea_io.apply_ea(Wm, sig)
    starts = list(range(0, M - CROP + 1, HOP))
    X = np.stack([aligned[:, s:s + CROP][:, :, None] for s in starts]).astype(np.float32)
    probs = model.predict(X, batch_size=512, verbose=0)
    soft_w = probs[:, RIGHT] - probs[:, LEFT]
    centres = np.clip(np.array([int(round((s + CROP / 2) * DS)) for s in starts]), 0, len(y1000) - 1)
    return soft_w.astype(np.float64), centres, y1000


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 1.0:
        return x
    out = np.empty_like(x); acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def mse_for(cache, subj: str, alpha: float, gain: float) -> Tuple[float, int]:
    soft_w, centres, y1000 = cache[subj]
    s = np.clip(_ema(soft_w, alpha) * gain, -1.0, 1.0)
    t = np.arange(len(y1000)); j = np.clip(np.searchsorted(centres, t), 1, len(centres) - 1)
    nearest = np.where((t - centres[j - 1]) <= (centres[j] - t), j - 1, j)
    keep = np.isfinite(y1000)
    return float(np.mean((s[nearest][keep] - y1000[keep]) ** 2)), int(keep.sum())


def pooled(cache, subs, alpha, gain) -> float:
    num = den = 0.0
    for s in subs:
        m, n = mse_for(cache, s, alpha, gain); num += m * n; den += n
    return num / den


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "postproc.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h); t0 = time.perf_counter()
    try:
        model = model3.build_model3("eegsym"); model.load_weights(str(W3))
        logger.info("3-class EEGSym+EA loaded. Computing per-window outputs for BCI-IV-1 eval...")
        cache = {s: per_window_output(model, s) for s in SUBJECTS}

        # baseline (no post-processing): alpha=1, gain=1
        base = pooled(cache, REAL, 1.0, 1.0)
        logger.info("baseline (no smoothing, no hedge) pooled-real MSE = %.4f", base)

        # ---- LOSO over the 4 real subjects ----
        loso: Dict[str, Dict] = {}
        for held in REAL:
            train = [s for s in REAL if s != held]
            best = min(((a, g) for a in ALPHAS for g in GAINS),
                       key=lambda ag: pooled(cache, train, ag[0], ag[1]))
            m, _ = mse_for(cache, held, best[0], best[1])
            loso[held] = {"alpha": best[0], "gain": best[1], "mse": m}
            logger.info("LOSO %s: best(alpha=%.2f,gain=%.2f on others) -> MSE %.4f (base %.4f)",
                        held, best[0], best[1], m, mse_for(cache, held, 1.0, 1.0)[0])
        # pooled LOSO (weight by scored samples)
        num = den = 0.0
        for held in REAL:
            _, n = mse_for(cache, held, 1.0, 1.0)
            num += loso[held]["mse"] * n; den += n
        loso_pooled = num / den

        # ---- oracle: single best (alpha,gain) over all real subjects (headroom) ----
        oracle_ag = min(((a, g) for a in ALPHAS for g in GAINS),
                        key=lambda ag: pooled(cache, REAL, ag[0], ag[1]))
        oracle = pooled(cache, REAL, *oracle_ag)

        # per-subject at the oracle setting (incl. generated, for context)
        per_subj = {s: {"mse_base": mse_for(cache, s, 1.0, 1.0)[0],
                        "mse_oracle": mse_for(cache, s, *oracle_ag)[0],
                        "real": s in REAL} for s in SUBJECTS}

        summary = {"baseline_do_nothing": BASELINE_ZERO, "champion_2009": CHAMPION,
                   "our_raw_soft": round(base, 4),
                   "loso_pooled_real": round(loso_pooled, 4), "loso_per_subject": loso,
                   "oracle_setting": {"alpha": oracle_ag[0], "gain": oracle_ag[1]},
                   "oracle_pooled_real": round(oracle, 4),
                   "per_subject": per_subj,
                   "grid": {"alphas": ALPHAS, "gains": GAINS}}
        save_json(OUT / "summary.json", summary)
        _report(summary)
        logger.info("=== DONE in %.1fs === raw=%.4f  LOSO=%.4f  oracle=%.4f (champion %.3f, baseline %.3f)",
                    time.perf_counter() - t0, base, loso_pooled, oracle, CHAMPION, BASELINE_ZERO)
    finally:
        logger.removeHandler(h); h.close()


def _report(s: Dict):
    lines = ["# BCI-IV-1 -- lowering the MSE with post-processing (smoothing + hedging)", "",
             "No retraining: the 3-class continuous output is post-processed. Lower MSE = better. "
             f"References: silence={s['baseline_do_nothing']}, 2009 champion={s['champion_2009']}.", "",
             f"- Raw output (no post-processing): {s['our_raw_soft']}",
             f"- With post-processing, leave-one-subject-out: {s['loso_pooled_real']}",
             f"- Oracle (best single setting, upper bound): {s['oracle_pooled_real']} "
             f"(alpha={s['oracle_setting']['alpha']}, gain={s['oracle_setting']['gain']})", "",
             "| Subj | type | raw MSE | post-processed MSE (oracle) |", "|---|---|---|---|"]
    for k, v in s["per_subject"].items():
        lines.append(f"| ds1{k} | {'real' if v['real'] else 'gen.'} | {v['mse_base']:.4f} | {v['mse_oracle']:.4f} |")
    lines += ["", "## LOSO per subject (setting chosen on the other 3 real subjects)",
              "| Subj | alpha | gain | MSE |", "|---|---|---|---|"]
    for k, v in s["loso_per_subject"].items():
        lines.append(f"| ds1{k} | {v['alpha']} | {v['gain']} | {v['mse']:.4f} |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
