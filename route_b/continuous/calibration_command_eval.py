r"""Does per-subject fine-tuning reduce the directional error (the drone's
safety-critical metric)?

The command-level eval showed the binding constraint is not detection but that,
when the system acts during MI, it turns the wrong way ~34 % of the time -- a
property of the L/R decoder that continuous training and the temporal filter do
not touch. Per-subject calibration was the only lever that improved L/R decoding,
so it is tested directly on the command-level directional error.

Reuses the leakage-safe machinery of calibration_eval.py (split by original
trial, fine-tune on cal_train, eval on the held-out eval pool only; global EA
fixed) and scores baseline vs fine-tuned with command_eval.eval_commands
(dwell rule). No test segment used in fine-tuning; eval streams built only from
each subject's held-out eval segments.

Usage:  python calibration_command_eval.py   (run from route_b/continuous/, BCI_DATA set)
"""
from __future__ import annotations
import sys

import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import config3 as C
from lib import continuous_gen as gen
from lib import crops3, ea_io, model3
from lib.segments import extract_segments_for_sessions

import calibration_eval as CAL
from calibration_eval import _split_subject, _pool_from_segments, _finetune
from command_eval import eval_commands

from src.split import load_split, session_keys_for  # noqa: E402
from src.utils import get_logger, save_json  # noqa: E402

logger = get_logger()
OUT = C.EXPERIMENTS / "calibration_command"
BASE_DIR = CAL.BASE_DIR
SEEDS = (42, 123, 256)
DWELL = 3


def _score_model(model, W3, eval_pool, subject, centers) -> Dict[str, int]:
    tot = {"correct": 0, "wrong": 0, "missed": 0, "false": 0}
    for seed in SEEDS:
        for ti in range(C.CONT_N_TRIALS_PER_SUBJECT):
            rng = np.random.default_rng([seed, int(subject[1:]), ti, 7])
            trial = gen.build_continuous_trial(eval_pool, ti, rng, seed=seed)
            X = np.stack([ea_io.apply_ea(W3, trial.signal)[:, s:e][:, :, None]
                          for s, e in crops3.iter_crop_bounds(trial.signal.shape[1])]).astype(np.float32)
            pred = model.predict(X, batch_size=512, verbose=0).argmax(1)
            co, wr, mi, fa = eval_commands(pred, centers, trial.events, DWELL)
            tot["correct"] += co; tot["wrong"] += wr; tot["missed"] += mi; tot["false"] += fa
    return tot


def _dir_err(t):
    d = t["correct"] + t["wrong"]
    return (t["wrong"] / d) if d else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "calibration_command.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        W3 = ea_io.load_ea_matrix(BASE_DIR / "ea_reference.json")
        normalize_fn = lambda sig: ea_io.apply_ea(W3, sig)  # noqa: E731
        base_weights = BASE_DIR / "weights.weights.h5"
        centers = np.array([(s + e) // 2 for s, e in crops3.iter_crop_bounds(C.CONT_TRIAL_SAMPLES)])

        split = load_split(C.SPLIT_JSON)
        subjects = gen.select_subjects(split)
        base = model3.build_model3("eegsym"); base.load_weights(base_weights)

        per_subject = []
        for subject in subjects:
            keys = [k for k in session_keys_for(split, "test") if k.startswith(f"{subject}_")]
            segs = extract_segments_for_sessions(keys)
            cal_tr, cal_va, ev = _split_subject(segs, seed=C.RANDOM_SEED + int(subject[1:]))
            eval_pool = _pool_from_segments(subject, ev)
            if not (eval_pool.mi_left and eval_pool.mi_right and eval_pool.rest):
                logger.warning("%s: eval pool missing a class, skipping.", subject)
                continue
            base_t = _score_model(base, W3, eval_pool, subject, centers)
            ft, n_ep = _finetune(base_weights, cal_tr, cal_va, normalize_fn)
            ft_t = _score_model(ft, W3, eval_pool, subject, centers)
            row = {"subject": subject, "epochs": int(n_ep),
                   "baseline": base_t, "finetuned": ft_t,
                   "dir_err_baseline": _dir_err(base_t), "dir_err_finetuned": _dir_err(ft_t)}
            per_subject.append(row)
            logger.info("%s: dir-err %.3f -> %.3f | correct %d->%d wrong %d->%d missed %d->%d false %d->%d",
                        subject, row["dir_err_baseline"], row["dir_err_finetuned"],
                        base_t["correct"], ft_t["correct"], base_t["wrong"], ft_t["wrong"],
                        base_t["missed"], ft_t["missed"], base_t["false"], ft_t["false"])
            import gc; del ft; gc.collect()

        # aggregate (pool all subjects' counts) + per-subject dir-err mean
        def pool(kind):
            return {k: int(np.sum([r[kind][k] for r in per_subject]))
                    for k in ("correct", "wrong", "missed", "false")}
        agg = {"baseline": pool("baseline"), "finetuned": pool("finetuned")}
        n_streams = len(per_subject) * len(SEEDS) * C.CONT_N_TRIALS_PER_SUBJECT
        summary = {
            "design": "per-subject fine-tuning evaluated at COMMAND level (dwell=3); "
                      "focus = directional error when acting",
            "seeds": list(SEEDS), "dwell": DWELL, "n_streams": n_streams,
            "subjects": [r["subject"] for r in per_subject],
            "pooled_directional_error": {"baseline": _dir_err(agg["baseline"]),
                                         "finetuned": _dir_err(agg["finetuned"])},
            "per_subject_dir_err_mean": {
                "baseline": float(np.mean([r["dir_err_baseline"] for r in per_subject])),
                "finetuned": float(np.mean([r["dir_err_finetuned"] for r in per_subject]))},
            "pooled_counts": agg,
            "per_trial": {m: {k: agg[m][k] / n_streams for k in agg[m]} for m in agg},
            "per_subject": per_subject,
        }
        save_json(OUT / "summary.json", summary)
        _report(summary)
        logger.info("=== DONE in %.1fs | pooled dir-err %.3f -> %.3f | per-subj mean %.3f -> %.3f ===",
                    time.perf_counter() - t0, summary["pooled_directional_error"]["baseline"],
                    summary["pooled_directional_error"]["finetuned"],
                    summary["per_subject_dir_err_mean"]["baseline"],
                    summary["per_subject_dir_err_mean"]["finetuned"])
    finally:
        logger.removeHandler(h)
        h.close()


def _report(s):
    pt = s["per_trial"]
    lines = ["# Per-subject fine-tuning -> directional error (per-command eval, dwell=3)", "",
             f"Subjects: {', '.join(s['subjects'])} | seeds: {s['seeds']} | {s['n_streams']} streams", "",
             "## Critical metric: directional error when acting (reversed / commands issued)", "",
             "| | Baseline | Calibrated |", "|---|---|---|",
             f"| Pooled (all commands) | {s['pooled_directional_error']['baseline']*100:.0f}\\% | "
             f"{s['pooled_directional_error']['finetuned']*100:.0f}\\% |",
             f"| Per-subject mean | {s['per_subject_dir_err_mean']['baseline']*100:.0f}\\% | "
             f"{s['per_subject_dir_err_mean']['finetuned']*100:.0f}\\% |", "",
             "## Commands per trial (of 6 episodes)", "",
             "| Metric | Baseline | Calibrated |", "|---|---|---|"]
    for k, lab in [("correct", "Correct"), ("wrong", "Reversed"),
                   ("missed", "Missed"), ("false", "False turns in rest")]:
        lines.append(f"| {lab} | {pt['baseline'][k]:.2f} | {pt['finetuned'][k]:.2f} |")
    lines += ["", "## Directional error per subject", "",
              "| Subject | Baseline | Calibrated |", "|---|---|---|"]
    for r in s["per_subject"]:
        lines.append(f"| {r['subject']} | {r['dir_err_baseline']*100:.0f}\\% | {r['dir_err_finetuned']*100:.0f}\\% |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
