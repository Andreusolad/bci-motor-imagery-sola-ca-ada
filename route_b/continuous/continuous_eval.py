r"""Part 2 --- evaluate a frozen 3-class model on artificial CONTINUOUS EEG.

Generalized over ``(architecture, normalization)`` so it can score the winner
(EEGSym + Euclidean Alignment) or EEGNet + Running Exponential Standardization,
each loaded from its Part 1 directory. Builds continuous 5-minute recordings by
concatenating real Stieger segments (10 test subjects x 5 trials, 6 MI + REST
each), runs the model exactly as in real time (1 s windows, 50 % overlap, the
normalization fixed at training: EA reuses the saved reference; running
exponential runs causally over the stream), and scores every window against a
per-sample ground truth. Repeated over 5 seeds for robustness.

Usage (after the matching train_3class.py has produced the model):
    python continuous_eval.py --arch eegnet --method running_exponential
    python continuous_eval.py --arch eegsym  --method euclidean_alignment
"""
from __future__ import annotations
import sys

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import config3 as C
from lib import continuous_gen as gen
from lib import crops3, metrics3, model3, normalize3, viz3
from lib.segments import load_or_build_subject_pool

from src.split import load_split  # noqa: E402
from src.utils import get_logger, save_json  # noqa: E402

logger = get_logger()


# --------------------------------------------------------------------------- #
# Sliding-window inference
# --------------------------------------------------------------------------- #
def _window_gt(gt: np.ndarray, start: int, stop: int) -> int:
    return int(np.bincount(gt[start:stop], minlength=C.N_CLASSES).argmax())


def infer_trial(model, normalized: np.ndarray, gt: np.ndarray):
    """Return per-window (center_time_s, gt_label, pred_label, probs)."""
    bounds = crops3.iter_crop_bounds(normalized.shape[1])
    X = np.stack([normalized[:, s:e][:, :, None] for s, e in bounds]).astype(np.float32)
    probs = model.predict(X, batch_size=512, verbose=0)
    pred = probs.argmax(axis=1)
    centers = np.array([(s + C.CROP_SAMPLES / 2) / C.FS_TARGET for s, _ in bounds])
    gt_win = np.array([_window_gt(gt, s, e) for s, e in bounds], dtype=np.int64)
    return centers, gt_win, pred, probs


# --------------------------------------------------------------------------- #
# One seed = 50 continuous trials
# --------------------------------------------------------------------------- #
def run_seed(seed: int, subjects: List[str], pools: Dict[str, object], model,
             normalize_stream: Callable[[np.ndarray], np.ndarray],
             out: Path, is_primary: bool) -> Dict[str, object]:
    seed_dir = out / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    tl_dir = seed_dir / "timelines"
    if is_primary:
        tl_dir.mkdir(exist_ok=True)

    rec_subject: List[str] = []
    rec_trialkey: List[str] = []
    rec_time: List[float] = []
    rec_gt: List[int] = []
    rec_pred: List[int] = []
    rec_prob: List[np.ndarray] = []
    per_trial_acc: List[Dict] = []
    primary_signals: Dict[str, np.ndarray] = {}
    primary_gt: Dict[str, np.ndarray] = {}
    all_events: Dict[str, List] = {}

    for subject in subjects:
        pool = pools[subject]
        subj_num = int(subject[1:])
        for ti in range(C.CONT_N_TRIALS_PER_SUBJECT):
            rng = np.random.default_rng([seed, subj_num, ti])
            trial = gen.build_continuous_trial(pool, ti, rng, seed=seed)
            normalized = normalize_stream(trial.signal)
            centers, gt_win, pred, probs = infer_trial(model, normalized, trial.ground_truth)

            trial_key = f"{subject}_t{ti}"
            n = len(centers)
            rec_subject.extend([subject] * n)
            rec_trialkey.extend([trial_key] * n)
            rec_time.extend(centers.tolist())
            rec_gt.extend(gt_win.tolist())
            rec_pred.extend(pred.tolist())
            rec_prob.append(probs)

            acc = float(np.mean(gt_win == pred))
            per_trial_acc.append({"trial_key": trial_key, "subject": subject,
                                   "accuracy": acc, "n_windows": int(n)})
            all_events[trial_key] = trial.events
            if is_primary:
                viz3.plot_timeline(gt_win, pred, centers, tl_dir / f"{trial_key}.png",
                                   title=f"{trial_key} (seed {seed}) -- acc={acc:.3f}")
                primary_signals[trial_key] = trial.signal.astype(np.float32)
                primary_gt[trial_key] = trial.ground_truth
        logger.info("[seed %d] %s done (%d trials).", seed, subject, C.CONT_N_TRIALS_PER_SUBJECT)

    gt = np.asarray(rec_gt)
    pred = np.asarray(rec_pred)
    prob = np.concatenate(rec_prob, axis=0)
    subj_arr = np.asarray(rec_subject)

    global_m = metrics3.compute_metrics(gt, pred, prob)
    bci_m = metrics3.bci_continuous_metrics(gt, pred)
    per_subject_acc = {s: float(np.mean(gt[subj_arr == s] == pred[subj_arr == s])) for s in subjects}
    trial_accs = np.array([t["accuracy"] for t in per_trial_acc])
    subj_accs = np.array(list(per_subject_acc.values()))

    seed_metrics = {
        "seed": seed, "n_trials": len(per_trial_acc), "n_windows": int(len(gt)),
        "global": global_m, "bci": bci_m,
        "per_subject_accuracy": per_subject_acc, "per_trial_accuracy": per_trial_acc,
        "across_subjects": {"mean": float(subj_accs.mean()), "std": float(subj_accs.std(ddof=1))},
        "across_trials": {"mean": float(trial_accs.mean()), "std": float(trial_accs.std(ddof=1))},
    }
    save_json(seed_dir / "metrics.json", seed_metrics)
    np.savez_compressed(seed_dir / "predictions.npz",
                        subject=subj_arr, trial_key=np.asarray(rec_trialkey),
                        time_s=np.asarray(rec_time, dtype=np.float32),
                        gt=gt.astype(np.int8), pred=pred.astype(np.int8), prob=prob.astype(np.float32))
    (seed_dir / "events.json").write_text(json.dumps(all_events, indent=1), encoding="utf-8")

    viz3.plot_confusion_matrix(np.asarray(global_m["confusion_matrix"]),
                               seed_dir / "confusion_matrix.png",
                               title=f"Continuous confusion (seed {seed})")
    viz3.plot_confusion_matrix(np.asarray(global_m["confusion_matrix"]),
                               seed_dir / "confusion_matrix_normalized.png",
                               title=f"Continuous confusion (seed {seed}, normalized)", normalize=True)
    viz3.plot_prob_histograms(gt, prob, seed_dir / "prob_histograms.png")
    viz3.plot_bars(subjects, [per_subject_acc[s] for s in subjects],
                   seed_dir / "accuracy_per_subject.png",
                   title=f"Window accuracy per subject (seed {seed})", ylabel="accuracy", hline=1 / 3)
    tkeys = [t["trial_key"] for t in per_trial_acc]
    viz3.plot_bars(tkeys, [t["accuracy"] for t in per_trial_acc],
                   seed_dir / "accuracy_per_trial.png",
                   title=f"Window accuracy per trial (seed {seed})", ylabel="accuracy", hline=1 / 3)

    if is_primary and primary_signals:
        def _obj(d):
            keys = list(d)
            arr = np.empty(len(keys), dtype=object)
            for i, k in enumerate(keys):
                arr[i] = d[k]
            return np.asarray(keys), arr
        k1, s1 = _obj(primary_signals)
        _, g1 = _obj(primary_gt)
        np.savez_compressed(seed_dir / "continuous_eeg.npz", trial_key=k1, signal=s1,
                            ground_truth=g1, fs=C.FS_TARGET,
                            note="preprocessed pre-normalization continuous EEG @250Hz")
        logger.info("[seed %d] saved continuous EEG for %d trials (primary seed).", seed, len(k1))

    logger.info("[seed %d] global acc=%.4f bal_acc=%.4f f1=%.4f | FPR=%.3f FNR=%.3f",
                seed, global_m["accuracy"], global_m["balanced_accuracy"], global_m["f1_macro"],
                bci_m["false_positive_rate_rest_as_mi"], bci_m["false_negative_rate_mi_as_rest"])
    return seed_metrics


# --------------------------------------------------------------------------- #
# Robustness aggregation across seeds
# --------------------------------------------------------------------------- #
def _agg(values: List[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and std > 0:
        lo, hi = sps.t.interval(0.95, df=n - 1, loc=mean, scale=std / np.sqrt(n))
        lo, hi = float(lo), float(hi)
    else:
        lo = hi = mean
    return {"mean": mean, "std": std, "ci95_low": lo, "ci95_high": hi, "n": n}


def aggregate(seed_results: List[Dict]) -> Dict[str, object]:
    def col(path):
        vals = []
        for r in seed_results:
            d = r
            for k in path:
                d = d[k]
            vals.append(d)
        return vals
    metrics = {
        "accuracy": col(["global", "accuracy"]),
        "balanced_accuracy": col(["global", "balanced_accuracy"]),
        "precision_macro": col(["global", "precision_macro"]),
        "recall_macro": col(["global", "recall_macro"]),
        "f1_macro": col(["global", "f1_macro"]),
        "rest_accuracy": [r["global"]["per_class_accuracy"]["REST"] for r in seed_results],
        "left_accuracy": [r["global"]["per_class_accuracy"]["LEFT"] for r in seed_results],
        "right_accuracy": [r["global"]["per_class_accuracy"]["RIGHT"] for r in seed_results],
        "false_positive_rate_rest_as_mi": col(["bci", "false_positive_rate_rest_as_mi"]),
        "false_negative_rate_mi_as_rest": col(["bci", "false_negative_rate_mi_as_rest"]),
        "rest_vs_mi_accuracy": col(["bci", "rest_vs_mi_accuracy"]),
        "across_subjects_mean_accuracy": col(["across_subjects", "mean"]),
        "across_trials_mean_accuracy": col(["across_trials", "mean"]),
    }
    return {name: {"per_seed": vals, **_agg(vals)} for name, vals in metrics.items()}


def _write_report(path: Path, summary: Dict, seeds, subjects, arch, method) -> None:
    lines = [
        f"# Continuous-EEG evaluation -- {arch} + {method}", "",
        f"- Subjects (test, unseen): {', '.join(subjects)}",
        f"- Seeds: {', '.join(map(str, seeds))}",
        f"- Per seed: {C.CONT_N_SUBJECTS} subjects x {C.CONT_N_TRIALS_PER_SUBJECT} trials "
        f"x {C.CONT_TRIAL_SECONDS}s = 50 continuous trials (250 min)", "",
        "## Metrics across seeds (mean +/- std [95% CI])", "",
        "| Metric | Mean | Std | 95% CI |", "|---|---|---|---|",
    ]
    for name, agg in summary.items():
        lines.append(f"| {name} | {agg['mean']:.4f} | {agg['std']:.4f} | "
                     f"[{agg['ci95_low']:.4f}, {agg['ci95_high']:.4f}] |")
    lines += ["", "Chance for 3 balanced classes is 0.333; REST-vs-MI chance is 0.5.",
              "Global accuracy is inflated because a 5-min idle-heavy stream is ~90% REST; "
              "balanced accuracy and FPR/FNR are the meaningful metrics.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(arch: str, method: str) -> None:
    out = C.continuous_dir(arch, method)
    mdir = C.model_dir(arch, method)
    C.ensure_dirs()
    out.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(out / "continuous_eval.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        weights = mdir / "weights.weights.h5"
        if not weights.exists():
            raise FileNotFoundError(f"Missing Part 1 weights at {weights}. Train first.")
        model = model3.build_model3(arch)
        model.load_weights(weights)
        normalize_stream = normalize3.load_normalizer(method, mdir)
        logger.info("Loaded %s + %s from %s.", arch, method, mdir)

        split = load_split(C.SPLIT_JSON)
        subjects = gen.select_subjects(split)
        logger.info("Selected %d test subjects: %s", len(subjects), subjects)
        pools = {s: load_or_build_subject_pool(s, split) for s in subjects}

        seed_results: List[Dict] = []
        for seed in C.CONT_SEEDS:
            logger.info("==================== SEED %d ====================", seed)
            seed_results.append(run_seed(seed, subjects, pools, model, normalize_stream,
                                         out, is_primary=(seed == C.CONT_SEEDS[0])))

        summary = aggregate(seed_results)
        save_json(out / "summary.json", {
            "architecture": arch, "normalization": method,
            "seeds": list(C.CONT_SEEDS), "subjects": subjects,
            "design": {"n_subjects": C.CONT_N_SUBJECTS,
                       "n_trials_per_subject": C.CONT_N_TRIALS_PER_SUBJECT,
                       "trial_seconds": C.CONT_TRIAL_SECONDS, "n_mi_per_trial": C.CONT_N_MI_PER_TRIAL,
                       "window_s": 1.0, "overlap": C.CROP_OVERLAP},
            "robustness": summary,
        })
        _write_report(out / "REPORT.md", summary, C.CONT_SEEDS, subjects, arch, method)

        keys = ["accuracy", "balanced_accuracy", "f1_macro", "rest_accuracy",
                "left_accuracy", "right_accuracy", "rest_vs_mi_accuracy"]
        viz3.plot_bars(keys, [summary[k]["mean"] for k in keys], out / "robustness_summary.png",
                       title=f"{arch}+{C.NORM_SHORT[method]} continuous metrics across 5 seeds",
                       ylabel="value", err=[summary[k]["std"] for k in keys], hline=1 / 3)

        logger.info("=== PART 2 COMPLETE in %.1fs -> %s ===", time.perf_counter() - t0, out)
        logger.info("Mean acc=%.4f bal_acc=%.4f (FPR=%.3f FNR=%.3f)",
                    summary["accuracy"]["mean"], summary["balanced_accuracy"]["mean"],
                    summary["false_positive_rate_rest_as_mi"]["mean"],
                    summary["false_negative_rate_mi_as_rest"]["mean"])
    finally:
        logger.removeHandler(h)
        h.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Continuous-EEG evaluation of a 3-class model.")
    p.add_argument("--arch", choices=model3.ARCHITECTURES, default="eegsym")
    p.add_argument("--method", choices=("euclidean_alignment", "running_exponential"),
                   default="euclidean_alignment")
    args = p.parse_args()
    run(args.arch, args.method)


if __name__ == "__main__":
    main()
