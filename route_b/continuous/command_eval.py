r"""Command-level (per-episode) evaluation for the drone use case.

Per-window recall is too harsh: each 300 s trial contains 6 separate MI episodes,
each a distinct command the drone must execute in the correct direction, without
spurious turns during the REST periods in between. This script scores the system
the way a drone would actually use it, with a realistic dwell rule.

Command model (dwell): a command fires when >= DWELL consecutive windows are
predicted MI (non-REST); its direction = the dominant L/R within the run. Each
command is attributed to the ground-truth event containing its midpoint. Then,
per 300 s trial (6 MI episodes, 7 REST periods):
  * correct  = episode with a command in the right direction
  * wrong    = episode with a command in the wrong direction
  * missed   = episode with no command
  * false    = commands landing in a REST period (spurious turns)

Evaluated on the same test streams as everything else (10 test subjects x 5
trials x 5 seeds), for flat-raw (baseline), continuous-raw, and continuous+HMM.
No leakage: frozen models; HMM transition matrix from held-out val GT.

Usage:  python command_eval.py   (run from route_b/continuous/, BCI_DATA set)
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
from lib import crops3, ea_io, model3
from lib.segments import load_or_build_subject_pool

import temporal_filter_eval as TF

from src.split import load_split  # noqa: E402
from src.utils import get_logger, save_json  # noqa: E402

logger = get_logger()
OUT = C.EXPERIMENTS / "command_eval"
FLAT_DIR = C.EXPERIMENTS / "rest_3class"
CONT_DIR = C.EXPERIMENTS / "continuous_trained"
REST, LEFT, RIGHT = C.REST_ID, C.LEFT_ID, C.RIGHT_ID
PRIOR = np.array([1 / 3, 1 / 3, 1 / 3])
PI = np.array([0.90, 0.05, 0.05])
HMM_M = 0.5848              # continuous model's val-selected operating point
DWELLS = (2, 3, 4)


def estimate_A(split) -> np.ndarray:
    val = sorted(split["splits"]["val"]["subjects"], key=lambda s: int(s[1:]))[:8]
    counts = np.zeros((3, 3))
    for subject in val:
        pool = load_or_build_subject_pool(subject, split)
        if not (pool.mi_left and pool.mi_right and pool.rest):
            continue
        for ti in range(C.CONT_N_TRIALS_PER_SUBJECT):
            rng = np.random.default_rng([11, int(subject[1:]), ti, 3])
            trial = gen.build_continuous_trial(pool, ti, rng, seed=11)
            w = TF._window_gt(trial.ground_truth)
            for a, b in zip(w[:-1], w[1:]):
                counts[a, b] += 1
    return (counts + 1e-6) / (counts.sum(1, keepdims=True) + 3e-6)


def eval_commands(pred: np.ndarray, centers: np.ndarray, events: List[Dict], dwell: int
                  ) -> Tuple[int, int, int, int]:
    runs = []
    i, n = 0, len(pred)
    while i < n:
        if pred[i] != REST:
            j = i
            while j < n and pred[j] != REST:
                j += 1
            if j - i >= dwell:
                seg = pred[i:j]
                direction = LEFT if (seg == LEFT).sum() >= (seg == RIGHT).sum() else RIGHT
                runs.append((int(centers[(i + j) // 2]), direction))
            i = j
        else:
            i += 1
    commanded: Dict[int, int] = {}
    false_cmd = 0
    for csample, direction in runs:
        ev_idx = next((k for k, ev in enumerate(events)
                       if ev["start_sample"] <= csample < ev["end_sample"]), None)
        if ev_idx is None:
            continue
        if events[ev_idx]["class_id"] == REST:
            false_cmd += 1
        elif ev_idx not in commanded:
            commanded[ev_idx] = direction
    correct = wrong = missed = 0
    for k, ev in enumerate(events):
        if ev["class_id"] == REST:
            continue
        if k in commanded:
            if commanded[k] == ev["class_id"]:
                correct += 1
            else:
                wrong += 1
        else:
            missed += 1
    return correct, wrong, missed, false_cmd


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "command_eval.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        split = load_split(C.SPLIT_JSON)
        A = estimate_A(split)
        Ahmm = TF._rescale_persistence(A, HMM_M)

        flat = model3.build_model3("eegsym"); flat.load_weights(FLAT_DIR / "weights.weights.h5")
        Wf = ea_io.load_ea_matrix(FLAT_DIR / "ea_reference.json")
        cont = model3.build_model3("eegsym"); cont.load_weights(CONT_DIR / "weights.weights.h5")
        Wc = ea_io.load_ea_matrix(CONT_DIR / "ea_reference.json")

        test_subs = gen.select_subjects(split)
        pools = {s: load_or_build_subject_pool(s, split) for s in test_subs}
        bounds = crops3.iter_crop_bounds(C.CONT_TRIAL_SAMPLES)
        centers = np.array([(s + e) // 2 for s, e in bounds])

        conds = ["flat_raw", "cont_raw", "cont_hmm"]
        acc = {d: {c: {"correct": [], "wrong": [], "missed": [], "false": []} for c in conds} for d in DWELLS}
        n_trials = 0
        for seed in C.CONT_SEEDS:
            for subject in test_subs:
                for ti in range(C.CONT_N_TRIALS_PER_SUBJECT):
                    rng = np.random.default_rng([seed, int(subject[1:]), ti])
                    trial = gen.build_continuous_trial(pools[subject], ti, rng, seed=seed)
                    Xc = np.stack([ea_io.apply_ea(Wc, trial.signal)[:, s:e][:, :, None]
                                   for s, e in bounds]).astype(np.float32)
                    Xf = np.stack([ea_io.apply_ea(Wf, trial.signal)[:, s:e][:, :, None]
                                   for s, e in bounds]).astype(np.float32)
                    prob_c = cont.predict(Xc, batch_size=512, verbose=0)
                    prob_f = flat.predict(Xf, batch_size=512, verbose=0)
                    preds = {"flat_raw": prob_f.argmax(1),
                             "cont_raw": prob_c.argmax(1),
                             "cont_hmm": TF.hmm_forward(prob_c.astype(np.float64), Ahmm, PRIOR, PI)}
                    for d in DWELLS:
                        for c in conds:
                            co, wr, mi, fa = eval_commands(preds[c], centers, trial.events, d)
                            acc[d][c]["correct"].append(co); acc[d][c]["wrong"].append(wr)
                            acc[d][c]["missed"].append(mi); acc[d][c]["false"].append(fa)
                    n_trials += 1
            logger.info("seed %d done (%d trials)", seed, n_trials)

        summary = {"design": "per-command drone eval (dwell rule); 6 MI episodes/trial",
                   "n_trials": n_trials, "dwells": list(DWELLS),
                   "results": {}}
        for d in DWELLS:
            summary["results"][d] = {}
            for c in conds:
                r = acc[d][c]
                summary["results"][d][c] = {k: {"per_trial_mean": float(np.mean(v)),
                                                 "per_trial_std": float(np.std(v, ddof=1))}
                                            for k, v in r.items()}
        save_json(OUT / "summary.json", summary)
        _report(summary)
        d = 3
        r = summary["results"][d]
        logger.info("=== DWELL=3 | of 6 episodes -> correct: flat %.2f, cont %.2f, cont+HMM %.2f | "
                    "false/trial: flat %.2f, cont %.2f, cont+HMM %.2f ===",
                    r["flat_raw"]["correct"]["per_trial_mean"], r["cont_raw"]["correct"]["per_trial_mean"],
                    r["cont_hmm"]["correct"]["per_trial_mean"], r["flat_raw"]["false"]["per_trial_mean"],
                    r["cont_raw"]["false"]["per_trial_mean"], r["cont_hmm"]["false"]["per_trial_mean"])
        logger.info("DONE in %.1fs", time.perf_counter() - t0)
    finally:
        logger.removeHandler(h)
        h.close()


def _report(summary):
    NAME = {"flat_raw": "Flat (baseline)", "cont_raw": "Continuous", "cont_hmm": "Continuous+HMM"}
    lines = ["# Per-command evaluation (drone use) -- 6 MI episodes per 5-min trial", "",
             f"Mean over {summary['n_trials']} trials (10 test subjects x 5 x 5 seeds). "
             "Dwell rule = MI sustained >= D windows.", ""]
    for d in summary["dwells"]:
        r = summary["results"][d]
        lines += [f"## Dwell = {d} windows (~{d*0.5:.1f} s sustained)", "",
                  "| Metric (of 6 episodes/trial) | " + " | ".join(NAME[c] for c in NAME) + " |",
                  "|---|---|---|---|"]
        for key, lab in [("correct", "Correct commands (right direction)"),
                         ("wrong", "Reversed commands"), ("missed", "Missed episodes"),
                         ("false", "False turns in rest (per trial)")]:
            row = " | ".join(f"{r[c][key]['per_trial_mean']:.2f}" for c in NAME)
            lines.append(f"| {lab} | {row} |")
        # correct as %
        row = " | ".join(f"{r[c]['correct']['per_trial_mean']/6*100:.0f}\\%" for c in NAME)
        lines.append(f"| % episodes executed correctly | {row} |")
        # directional error WHEN ACTING = wrong / (correct+wrong)  -- the safety-critical metric
        def dir_err(c):
            co = r[c]["correct"]["per_trial_mean"]; wr = r[c]["wrong"]["per_trial_mean"]
            return (wr / (co + wr) * 100) if (co + wr) > 0 else float("nan")
        row = " | ".join(f"{dir_err(c):.0f}\\%" for c in NAME)
        lines.append(f"| Directional error when acting (critical) | {row} |")
        lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
