r"""Step 1 --- hierarchical REST-gate + 2-class L/R decoder, on the SAME
continuous 5-minute streams, to avoid the LEFT-collapse of the flat 3-class head.

Two proven models are combined at inference (nothing is retrained):

  Stage 1 (gate): the 3-class EEGSym+EA model decides REST vs MI as the binary
                  P(LEFT)+P(RIGHT) vs P(REST) (its REST detection is strong).
  Stage 2 (L/R):  on windows gated as MI, the winning 2-class EEGSym+EA model
                  (left/right, which never saw REST, ~0.745) picks the side.

Each model is fed the continuous signal normalized with its OWN training-fit EA
reference: the 3-class one for the gate, the 2-class one for the L/R model. The
2-class winner never persisted its EA matrix, so it is re-fit here on the exact
same training L/R trials (EA is deterministic -> identical matrix).

Usage (run from route_b/continuous/, BCI_DATA set):
    python hierarchical_eval.py
"""
from __future__ import annotations
import sys

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import config3 as C
from lib import continuous_gen as gen
from lib import crops3, ea_io, metrics3, viz3
from lib.segments import load_or_build_subject_pool

from src.dataset import imagery_window_from_feedback  # noqa: E402
from src.eegsym_study.eegsym import build_eegsym  # noqa: E402
from src.split import load_split, session_keys_for  # noqa: E402
from src.study.data_loading import load_trials_for_sessions  # noqa: E402
from src.study.normalization.euclidean_alignment import EuclideanAlignment  # noqa: E402
from src.utils import get_logger, save_json  # noqa: E402

logger = get_logger()

OUT = C.EXPERIMENTS / "continuous_eval_hierarchical"
GATE_DIR = C.model_dir("eegsym", "euclidean_alignment")        # 3-class EEGSym+EA
LR_DIR = (Path(__file__).resolve().parents[1] / "experiments" / "corrected_window"
          / "normalization" / "eegsym" / "euclidean_alignment")  # 2-class winner
EA2_JSON = OUT / "ea_2class_lr_reference.json"


def ensure_2class_ea() -> np.ndarray:
    """Re-fit (once) the 2-class winner's EA reference on training L/R trials."""
    if EA2_JSON.exists():
        return ea_io.load_ea_matrix(EA2_JSON)
    logger.info("Re-fitting 2-class L/R EA on training trials (corrected window)...")
    split = load_split(C.SPLIT_JSON)
    train_trials = load_trials_for_sessions(
        session_keys_for(split, "train"), window_fn=imagery_window_from_feedback)
    ea = EuclideanAlignment()
    ea.fit([t.signal for t in train_trials])
    ea_io.save_ea(ea, EA2_JSON)
    logger.info("Saved 2-class EA reference (%d L/R trials) -> %s",
                len(train_trials), EA2_JSON)
    return ea_io.load_ea_matrix(EA2_JSON)


def _windows(signal: np.ndarray) -> np.ndarray:
    bounds = crops3.iter_crop_bounds(signal.shape[1])
    return np.stack([signal[:, s:e][:, :, None] for s, e in bounds]).astype(np.float32)


def _window_gt(gt: np.ndarray) -> np.ndarray:
    bounds = crops3.iter_crop_bounds(len(gt))
    return np.array([int(np.bincount(gt[s:e], minlength=C.N_CLASSES).argmax())
                     for s, e in bounds], dtype=np.int64)


def hierarchical_predict(gate_model, lr_model, W3: np.ndarray, W2: np.ndarray,
                         signal: np.ndarray):
    """Return (pred, gate_prob3, lr_prob2) per window for one continuous trial."""
    X3 = _windows(ea_io.apply_ea(W3, signal))
    X2 = _windows(ea_io.apply_ea(W2, signal))
    gate_prob = gate_model.predict(X3, batch_size=512, verbose=0)   # (n, 3)
    lr_prob = lr_model.predict(X2, batch_size=512, verbose=0)       # (n, 2): 0=left,1=right

    p_rest = gate_prob[:, C.REST_ID]
    p_mi = gate_prob[:, C.LEFT_ID] + gate_prob[:, C.RIGHT_ID]
    is_mi = p_mi > p_rest
    # 2-class ids: 0=left->LEFT_ID(1), 1=right->RIGHT_ID(2)
    lr_side = np.where(lr_prob.argmax(1) == 0, C.LEFT_ID, C.RIGHT_ID)
    pred = np.where(is_mi, lr_side, C.REST_ID).astype(np.int64)
    return pred, gate_prob, lr_prob


def _agg(vals: List[float]) -> Dict[str, float]:
    a = np.asarray(vals, float); n = len(a)
    mean = float(a.mean()); std = float(a.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and std > 0:
        lo, hi = sps.t.interval(0.95, df=n - 1, loc=mean, scale=std / np.sqrt(n))
        lo, hi = float(lo), float(hi)
    else:
        lo = hi = mean
    return {"mean": mean, "std": std, "ci95_low": lo, "ci95_high": hi, "n": n}


def run_seed(seed, subjects, pools, gate_model, lr_model, W3, W2, is_primary):
    seed_dir = OUT / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    tl_dir = seed_dir / "timelines"
    if is_primary:
        tl_dir.mkdir(exist_ok=True)

    rec_subj, rec_gt, rec_pred = [], [], []
    per_trial: List[Dict] = []
    for subject in subjects:
        pool = pools[subject]
        for ti in range(C.CONT_N_TRIALS_PER_SUBJECT):
            rng = np.random.default_rng([seed, int(subject[1:]), ti])
            trial = gen.build_continuous_trial(pool, ti, rng, seed=seed)
            gt = _window_gt(trial.ground_truth)
            pred, _, _ = hierarchical_predict(gate_model, lr_model, W3, W2, trial.signal)
            rec_subj.extend([subject] * len(gt)); rec_gt.extend(gt.tolist()); rec_pred.extend(pred.tolist())
            acc = float(np.mean(gt == pred))
            per_trial.append({"trial_key": f"{subject}_t{ti}", "subject": subject,
                              "accuracy": acc, "n_windows": int(len(gt))})
            if is_primary:
                centers = np.array([(s + C.CROP_SAMPLES / 2) / C.FS_TARGET
                                    for s, _ in crops3.iter_crop_bounds(trial.signal.shape[1])])
                viz3.plot_timeline(gt, pred, centers, tl_dir / f"{subject}_t{ti}.png",
                                   title=f"{subject}_t{ti} hierarchical (seed {seed}) -- acc={acc:.3f}")
        logger.info("[seed %d] %s done.", seed, subject)

    gt = np.asarray(rec_gt); pred = np.asarray(rec_pred); subj = np.asarray(rec_subj)
    global_m = metrics3.compute_metrics(gt, pred)
    bci_m = metrics3.bci_continuous_metrics(gt, pred)
    per_subject = {s: float(np.mean(gt[subj == s] == pred[subj == s])) for s in subjects}
    trial_accs = np.array([t["accuracy"] for t in per_trial])
    subj_accs = np.array(list(per_subject.values()))
    seed_metrics = {
        "seed": seed, "n_trials": len(per_trial), "n_windows": int(len(gt)),
        "global": global_m, "bci": bci_m, "per_subject_accuracy": per_subject,
        "per_trial_accuracy": per_trial,
        "across_subjects": {"mean": float(subj_accs.mean()), "std": float(subj_accs.std(ddof=1))},
        "across_trials": {"mean": float(trial_accs.mean()), "std": float(trial_accs.std(ddof=1))},
    }
    save_json(seed_dir / "metrics.json", seed_metrics)
    viz3.plot_confusion_matrix(np.asarray(global_m["confusion_matrix"]),
                               seed_dir / "confusion_matrix.png",
                               title=f"Hierarchical continuous confusion (seed {seed})")
    viz3.plot_confusion_matrix(np.asarray(global_m["confusion_matrix"]),
                               seed_dir / "confusion_matrix_normalized.png",
                               title=f"Hierarchical (seed {seed}, normalized)", normalize=True)
    viz3.plot_bars(subjects, [per_subject[s] for s in subjects],
                   seed_dir / "accuracy_per_subject.png",
                   title=f"Window accuracy per subject (seed {seed})", ylabel="accuracy", hline=1 / 3)
    logger.info("[seed %d] acc=%.4f bal_acc=%.4f | REST=%.3f LEFT=%.3f RIGHT=%.3f | FPR=%.3f FNR=%.3f",
                seed, global_m["accuracy"], global_m["balanced_accuracy"],
                global_m["per_class_accuracy"]["REST"], global_m["per_class_accuracy"]["LEFT"],
                global_m["per_class_accuracy"]["RIGHT"],
                bci_m["false_positive_rate_rest_as_mi"], bci_m["false_negative_rate_mi_as_rest"])
    return seed_metrics


def aggregate(seed_results):
    def col(path):
        out = []
        for r in seed_results:
            d = r
            for k in path:
                d = d[k]
            out.append(d)
        return out
    m = {
        "accuracy": col(["global", "accuracy"]),
        "balanced_accuracy": col(["global", "balanced_accuracy"]),
        "f1_macro": col(["global", "f1_macro"]),
        "rest_accuracy": [r["global"]["per_class_accuracy"]["REST"] for r in seed_results],
        "left_accuracy": [r["global"]["per_class_accuracy"]["LEFT"] for r in seed_results],
        "right_accuracy": [r["global"]["per_class_accuracy"]["RIGHT"] for r in seed_results],
        "false_positive_rate_rest_as_mi": col(["bci", "false_positive_rate_rest_as_mi"]),
        "false_negative_rate_mi_as_rest": col(["bci", "false_negative_rate_mi_as_rest"]),
        "rest_vs_mi_accuracy": col(["bci", "rest_vs_mi_accuracy"]),
    }
    return {k: {"per_seed": v, **_agg(v)} for k, v in m.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "hierarchical_eval.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        W2 = ensure_2class_ea()
        W3 = ea_io.load_ea_matrix(GATE_DIR / "ea_reference.json")

        gate_model = build_eegsym(input_shape=C.INPUT_SHAPE, n_classes=3)
        gate_model.load_weights(GATE_DIR / "weights.weights.h5")
        lr_model = build_eegsym(input_shape=C.INPUT_SHAPE, n_classes=2)
        lr_model.load_weights(LR_DIR / "weights.weights.h5")
        logger.info("Loaded gate (3-class EEGSym+EA) and L/R (2-class EEGSym+EA) models.")

        split = load_split(C.SPLIT_JSON)
        subjects = gen.select_subjects(split)
        pools = {s: load_or_build_subject_pool(s, split) for s in subjects}

        seed_results = []
        for seed in C.CONT_SEEDS:
            logger.info("========== SEED %d ==========", seed)
            seed_results.append(run_seed(seed, subjects, pools, gate_model, lr_model,
                                         W3, W2, is_primary=(seed == C.CONT_SEEDS[0])))
        summary = aggregate(seed_results)
        save_json(OUT / "summary.json", {
            "design": "hierarchical: gate=3class-EEGSym+EA (REST vs MI), LR=2class-EEGSym+EA",
            "seeds": list(C.CONT_SEEDS), "subjects": subjects, "robustness": summary})

        lines = ["# Continuous-EEG evaluation -- HIERARCHICAL (REST-gate + 2-class L/R)", "",
                 f"- Subjects: {', '.join(subjects)}", f"- Seeds: {', '.join(map(str, C.CONT_SEEDS))}",
                 "", "| Metric | Mean | Std | 95% CI |", "|---|---|---|---|"]
        for k, a in summary.items():
            lines.append(f"| {k} | {a['mean']:.4f} | {a['std']:.4f} | [{a['ci95_low']:.4f}, {a['ci95_high']:.4f}] |")
        (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        keys = ["accuracy", "balanced_accuracy", "rest_accuracy", "left_accuracy",
                "right_accuracy", "rest_vs_mi_accuracy"]
        viz3.plot_bars(keys, [summary[k]["mean"] for k in keys], OUT / "robustness_summary.png",
                       title="Hierarchical continuous metrics across 5 seeds", ylabel="value",
                       err=[summary[k]["std"] for k in keys], hline=1 / 3)
        logger.info("=== HIERARCHICAL COMPLETE in %.1fs | acc=%.4f bal_acc=%.4f LEFT=%.4f FPR=%.4f ===",
                    time.perf_counter() - t0, summary["accuracy"]["mean"],
                    summary["balanced_accuracy"]["mean"], summary["left_accuracy"]["mean"],
                    summary["false_positive_rate_rest_as_mi"]["mean"])
    finally:
        logger.removeHandler(h)
        h.close()


if __name__ == "__main__":
    main()
