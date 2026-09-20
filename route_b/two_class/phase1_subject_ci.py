r"""Subject-level bootstrap confidence intervals for the six corrected-window
two-class models (Table S3), and paired per-subject contrasts between
architectures and normalizations (Table 3, Fig. 2).

For each of the six models {EEGNet, EEGSym} x {z-score, running-exponential,
Euclidean Alignment}:
  - reload the test trials (corrected window) directly from the .mat files,
  - refit the normalizer on the training split only (deterministic, so it
    reproduces the exact statistics used at train time; running-exponential is
    stateless),
  - build crops, load the trained weights, predict, aggregate to trial level,
  - compute per-subject trial accuracy,
  - bootstrap over the 13 test subjects (B=10,000) for a 95% CI of the
    subject-mean accuracy,
  - paired per-subject contrasts (EEGSym vs EEGNet; EA vs z-score; EA vs
    running-exponential) with 95% CIs and the number of subjects improving.

Correctness check: the trial-weighted (pooled) accuracy per model matches the
model's stored trial accuracy, asserted within 0.006.

Usage:  python phase1_subject_ci.py   (run from route_b/two_class/, BCI_DATA set)

Reads:  ../experiments/corrected_window/normalization/<arch>/<method>/weights.weights.h5
Writes: ../experiments/subject_ci/{summary.json, REPORT.md, subject_ci_forest.png}
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.split import load_split, session_keys_for            # noqa: E402
from src.dataset import imagery_window_from_feedback          # noqa: E402
from src.study.data_loading import load_trials_for_sessions   # noqa: E402
from src.study.crops import build_crops                       # noqa: E402
from src.study.run_experiment import _build_normalizer        # noqa: E402
from src.study.trial_aggregation import aggregate_by_trial    # noqa: E402
from src.study.eegnet import build_eegnet                     # noqa: E402
from src.eegsym_study.eegsym import build_eegsym              # noqa: E402
from src.utils import get_logger, save_json, set_global_seed  # noqa: E402

logger = get_logger()
_ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT = _ROOT_DIR / "experiments" / "corrected_window" / "normalization"
OUT = _ROOT_DIR / "experiments" / "subject_ci"
B = 10_000
SEED = 42
METHODS = ["z_score", "running_exponential", "euclidean_alignment"]
ARCHS = {"eegnet": build_eegnet, "eegsym": build_eegsym}
# published trial accuracy (main.tex comparativa-corrected-6) for the correctness check
PUBLISHED = {("eegnet", "z_score"): 0.6814, ("eegnet", "running_exponential"): 0.7040,
             ("eegnet", "euclidean_alignment"): 0.7068, ("eegsym", "z_score"): 0.7015,
             ("eegsym", "running_exponential"): 0.7260, ("eegsym", "euclidean_alignment"): 0.7450}
NICE = {"z_score": "Z-score", "running_exponential": "running-exp", "euclidean_alignment": "EA"}


def per_subject_accuracy(agg, tid2subj) -> Dict[str, Tuple[int, int]]:
    """subject -> (n_correct, n_trials) at trial level."""
    acc: Dict[str, List[int]] = {}
    yp = agg.y_pred
    for tid, yt, yhat in zip(agg.trial_ids, agg.y_true, yp):
        s = tid2subj[tid]
        c, n = acc.get(s, (0, 0))
        acc[s] = (c + int(yt == yhat), n + 1)
    return acc


def boot_ci(vals: np.ndarray, rng, b=B) -> Tuple[float, float, float]:
    """Point (mean over subjects) + 95% CI resampling subjects with replacement."""
    n = len(vals)
    means = vals[rng.integers(0, n, size=(b, n))].mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_ci(a: np.ndarray, b: np.ndarray, rng, nb=B) -> Dict:
    """Paired per-subject contrast a-b (same subject order), bootstrap over subjects."""
    d = a - b
    n = len(d)
    means = d[rng.integers(0, n, size=(nb, n))].mean(axis=1)
    return {"mean_diff": float(d.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)),
            "n_up": int((d > 0).sum()), "n_down": int((d < 0).sum()), "n": n}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "phase1.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        from tensorflow import keras  # noqa: F401
        set_global_seed(SEED)
        split = load_split()
        logger.info("Loading train + test trials (corrected window) from .mat ...")
        wf = imagery_window_from_feedback
        train_trials = load_trials_for_sessions(session_keys_for(split, "train"), window_fn=wf)
        test_trials = load_trials_for_sessions(session_keys_for(split, "test"), window_fn=wf)
        tid2subj = {t.trial_id: t.subject for t in test_trials}
        subjects = sorted({t.subject for t in test_trials}, key=lambda s: int(s.lstrip("S")))
        logger.info("test subjects (%d): %s", len(subjects), subjects)

        # per-subject accuracy matrix: rows=models, aligned columns=subjects
        per_model: Dict[Tuple[str, str], np.ndarray] = {}
        pooled_acc: Dict[Tuple[str, str], float] = {}

        for method in METHODS:
            normalize_fn, _ = _build_normalizer(method, train_trials)
            crops = build_crops(test_trials, normalize_fn)
            for arch, builder in ARCHS.items():
                w = ROOT / arch / method / "weights.weights.h5"
                model = builder()  # (8,250,1), 2 classes
                model.load_weights(w)
                probs = model.predict(crops.x, batch_size=512, verbose=0)
                agg = aggregate_by_trial(probs, crops.trial_ids, crops.y)
                acc = per_subject_accuracy(agg, tid2subj)
                vec = np.array([acc[s][0] / acc[s][1] for s in subjects])  # subject-aligned
                pooled = sum(acc[s][0] for s in subjects) / sum(acc[s][1] for s in subjects)
                per_model[(arch, method)] = vec
                pooled_acc[(arch, method)] = pooled
                pub = PUBLISHED[(arch, method)]
                ok = abs(pooled - pub) <= 0.006
                logger.info("%-6s %-20s pooled=%.4f (published %.4f) %s | subj-mean=%.4f",
                            arch, method, pooled, pub, "OK" if ok else "MISMATCH!", vec.mean())
                if not ok:
                    logger.warning("  pooled acc mismatch for %s/%s -> check pipeline", arch, method)
                keras.backend.clear_session()

        # --- bootstrap CIs per model --- #
        rng = np.random.default_rng(SEED)
        models = []
        for arch in ARCHS:
            for method in METHODS:
                vec = per_model[(arch, method)]
                pt, lo, hi = boot_ci(vec, rng)
                models.append({"arch": arch, "method": method, "nice": f"{arch}+{NICE[method]}",
                               "pooled_accuracy": pooled_acc[(arch, method)],
                               "subject_mean": pt, "ci_low": lo, "ci_high": hi,
                               "published_trial_acc": PUBLISHED[(arch, method)],
                               "per_subject": per_model[(arch, method)].tolist()})

        # --- paired contrasts --- #
        contrasts = {}
        # (a) EEGSym vs EEGNet, per method + pooled-over-methods
        for method in METHODS:
            contrasts[f"eegsym_vs_eegnet__{method}"] = paired_ci(
                per_model[("eegsym", method)], per_model[("eegnet", method)], rng)
        # architecture, averaged across the 3 normalizations (per subject)
        sym = np.mean([per_model[("eegsym", m)] for m in METHODS], axis=0)
        net = np.mean([per_model[("eegnet", m)] for m in METHODS], axis=0)
        contrasts["eegsym_vs_eegnet__avg"] = paired_ci(sym, net, rng)
        # (b) normalization: EA vs z-score, EA vs running-exp (averaged across the 2 arches)
        def avg_over_arch(method):
            return np.mean([per_model[(a, method)] for a in ARCHS], axis=0)
        contrasts["EA_vs_zscore__avg"] = paired_ci(avg_over_arch("euclidean_alignment"),
                                                   avg_over_arch("z_score"), rng)
        contrasts["EA_vs_runningexp__avg"] = paired_ci(avg_over_arch("euclidean_alignment"),
                                                       avg_over_arch("running_exponential"), rng)

        save_json(OUT / "summary.json", {
            "design": "subject-level bootstrap CIs (B=%d, resample subjects) for the 6 "
                      "corrected-window models; paired per-subject contrasts" % B,
            "n_test_subjects": len(subjects), "subjects": subjects,
            "models": models, "contrasts": contrasts})
        _report(subjects, models, contrasts)
        _figure(models)
        logger.info("=== DONE in %.1fs ===", time.perf_counter() - t0)
    finally:
        logger.removeHandler(h)
        h.close()


def _report(subjects, models, contrasts):
    lines = [f"# Subject-level 95% CIs (bootstrap, B={B}) for the six models", "",
             f"{len(subjects)} test subjects. Statistic = mean accuracy over subjects; "
             "CI by resampling subjects. Check: pooled accuracy matches the stored value.", "",
             "## Accuracy per model (trial)", "",
             "| Model | Pooled | Subject mean | 95% CI | Stored |", "|---|---|---|---|---|"]
    for m in models:
        lines.append(f"| {m['nice']} | {m['pooled_accuracy']:.3f} | {m['subject_mean']:.3f} | "
                     f"[{m['ci_low']:.3f}, {m['ci_high']:.3f}] | {m['published_trial_acc']:.3f} |")
    lines += ["", "## Paired per-subject contrasts (delta accuracy, 95% CI)", "",
              "| Contrast | Mean delta | 95% CI | Up/Down |", "|---|---|---|---|"]
    label = {"eegsym_vs_eegnet__z_score": "EEGSym-EEGNet (z-score)",
             "eegsym_vs_eegnet__running_exponential": "EEGSym-EEGNet (running-exp)",
             "eegsym_vs_eegnet__euclidean_alignment": "EEGSym-EEGNet (EA)",
             "eegsym_vs_eegnet__avg": "EEGSym-EEGNet (mean over norms)",
             "EA_vs_zscore__avg": "EA-Z-score (mean over archs)",
             "EA_vs_runningexp__avg": "EA-running-exp (mean over archs)"}
    for k, lab in label.items():
        c = contrasts[k]
        lines.append(f"| {lab} | {c['mean_diff']:+.3f} | [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | "
                     f"{c['n_up']}/{c['n_down']} |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _figure(models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = sorted(models, key=lambda m: m["subject_mean"])
    y = np.arange(len(models))
    pt = [m["subject_mean"] for m in models]
    lo = [m["subject_mean"] - m["ci_low"] for m in models]
    hi = [m["ci_high"] - m["subject_mean"] for m in models]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.errorbar(pt, y, xerr=[lo, hi], fmt="o", color="#1f3a5f", capsize=4, lw=1.5)
    ax.axvline(0.5, color="k", ls="--", lw=0.8, label="chance")
    ax.set(yticks=y, xlabel="trial accuracy, mean over subjects", ylim=(-0.6, len(models) - 0.4),
           title="subject-level 95% CI (bootstrap) -- corrected window")
    ax.set_yticklabels([m["nice"] for m in models]); ax.grid(alpha=0.3, axis="x"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "subject_ci_forest.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
