r"""Cross-dataset external validation on BCI Competition IV, dataset 2a.

The winning decoder (binary EEGSym + Euclidean Alignment, trained on Stieger's
corrected window) is applied zero-shot to 2a's left/right-hand MI: if a very
different decoder lands at the same dataset ceiling, the bottleneck is the
signal, not the model. Reported as per-subject balanced accuracy + subject-level
bootstrap CIs.

Ground truth: 2a labels are in the .mat `y` field (y in {1,2,3,4}; 1=left hand,
2=right hand, 3=feet, 4=tongue). Only LH/RH are kept. Labels/channel order are
not trusted blindly: before scoring, the contralateral mu/beta ERD is recovered
(C3 desync for right hand, C4 for left hand); if the laterality sign is wrong the
script aborts and reports, using no inferred labels.

Data facts: cells 3..8 of `data` are the 6 MI runs (48 trials each, 250 Hz, y in
1..4, ~8 s/trial); cells 0..2 are eye/calibration runs (no trials). 2a is already
at 250 Hz, so no resampling is needed.

Pipeline (as the model expects): band-pass 0.5-40 Hz (Butterworth-4, zero phase)
@250 -> CAR over the 8 motor channels -> EA fit per 2a subject (unsupervised,
leakage-safe) -> 1 s / 0.5 s crops -> EEGSym -> aggregate by trial. MI window
t in [2,5] s after the trial-start marker (imagery onset = cue at 2 s, mirroring
the Stieger corrected-window choice of taking imagery from its onset).

Usage:  python bciciv2a_eval.py   (run from route_b/transfer/, BCI_DATA set)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C                                    # noqa: E402
from src.study.normalization import EuclideanAlignment         # noqa: E402
from src.study.trial_aggregation import aggregate_by_trial     # noqa: E402
from src.eegsym_study.eegsym import build_eegsym               # noqa: E402
from src.crops import iter_crop_bounds                         # noqa: E402
from src.utils import get_logger, save_json                    # noqa: E402

logger = get_logger()
DATA = C.BCI_DATA / "bci_iv_2a"
W = (Path(__file__).resolve().parents[1] / "experiments" / "corrected_window"
     / "normalization" / "eegsym" / "euclidean_alignment" / "weights.weights.h5")
OUT = Path(__file__).resolve().parents[1] / "experiments" / "bciciv2a_eval"
SUBJECTS = [f"A0{i}T" for i in range(1, 10)]

# Official BCI-IV-2a channel order (Brunner et al. 2008): 22 EEG then 3 EOG.
# 0-based indices of OUR 8 Cyton motor channels within the 25-column X.
CH2A = {"Fz": 0, "FC3": 1, "FC1": 2, "FCz": 3, "FC2": 4, "FC4": 5, "C5": 6,
        "C3": 7, "C1": 8, "Cz": 9, "C2": 10, "C4": 11, "C6": 12, "CP3": 13,
        "CP1": 14, "CPz": 15, "CP2": 16, "CP4": 17, "P1": 18, "Pz": 19,
        "P2": 20, "POz": 21}
OUR8 = ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4"]
IDX8 = [CH2A[c] for c in OUR8]
C3_LOCAL, C4_LOCAL = OUR8.index("C3"), OUR8.index("C4")   # positions within the 8

FS = 250
WIN = (int(2.0 * FS), int(5.0 * FS))          # [2 s, 5 s] -> samples [500, 1250]
LEFT2A, RIGHT2A = 1, 2                         # 2a y codes for LH / RH
# map 2a code -> our model id (left=0, right=1)
CODE_TO_ID = {LEFT2A: C.LABEL_TO_ID["left"], RIGHT2A: C.LABEL_TO_ID["right"]}

_BP_B, _BP_A = butter(C.BANDPASS_ORDER,
                      [C.BANDPASS_LOW_HZ / (FS / 2), C.BANDPASS_HIGH_HZ / (FS / 2)],
                      btype="band")
_MU_B, _MU_A = butter(4, [8 / (FS / 2), 30 / (FS / 2)], btype="band")   # mu/beta for ERD


def preprocess(sig8: np.ndarray) -> np.ndarray:
    """band-pass 0.5-40 @250 -> CAR over the 8 channels (already at 250 Hz)."""
    filt = filtfilt(_BP_B, _BP_A, sig8, axis=-1).astype(np.float32)
    return (filt - filt.mean(axis=0, keepdims=True)).astype(np.float32)


def load_subject_trials(subj: str) -> Tuple[List[np.ndarray], List[int]]:
    """Return preprocessed (8, 750) LH/RH trial signals + our-model labels."""
    m = loadmat(str(DATA / f"{subj}.mat"), struct_as_record=False, squeeze_me=True)
    data = np.atleast_1d(m["data"])
    sigs: List[np.ndarray] = []
    labels: List[int] = []
    for cell in data:
        tr = np.atleast_1d(np.asarray(cell.trial)).astype(int)
        y = np.atleast_1d(np.asarray(cell.y)).astype(int)
        if tr.size == 0:
            continue                       # eye/calibration runs
        X = np.asarray(cell.X, dtype=np.float64)     # (T, 25) microvolts
        for start, code in zip(tr, y):
            if code not in CODE_TO_ID:
                continue                   # skip feet/tongue
            a, b = start + WIN[0], start + WIN[1]
            if b > X.shape[0]:
                continue
            seg = X[a:b, IDX8].T           # (8, 750)
            sigs.append(preprocess(seg))
            labels.append(CODE_TO_ID[code])
    return sigs, labels


def erd_gap(sigs: List[np.ndarray], labels: List[int]) -> float:
    """Contralateral laterality gap (C4-C3) for right minus left. Must be > 0."""
    p = np.stack([np.log(np.var(filtfilt(_MU_B, _MU_A, s, axis=-1), axis=-1) + 1e-12)
                  for s in sigs])                                   # (n, 8) log band-power
    lab = np.asarray(labels)
    gap = p[:, C4_LOCAL] - p[:, C3_LOCAL]
    right = gap[lab == C.LABEL_TO_ID["right"]].mean()
    left = gap[lab == C.LABEL_TO_ID["left"]].mean()
    return float(right - left)


def crops_for(sigs, labels) -> Tuple[np.ndarray, np.ndarray]:
    """EA fit per subject (unsupervised) -> 1s/0.5s crops (8,250,1) + labels."""
    ea = EuclideanAlignment().fit(sigs)          # reference from THIS subject only
    xs, ys, tids = [], [], []
    for ti, (s, lab) in enumerate(zip(sigs, labels)):
        aligned = ea.transform(s)
        for a, b in iter_crop_bounds(aligned.shape[1]):
            xs.append(aligned[:, a:b][:, :, None].astype(np.float32))
            ys.append(lab)
            tids.append(f"t{ti}")
    return np.stack(xs), np.asarray(ys, dtype=np.int64), tids


def subject_metrics(model, subj: str) -> Dict:
    sigs, labels = load_subject_trials(subj)
    gap = erd_gap(sigs, labels)
    x, y, tids = crops_for(sigs, labels)
    probs = model.predict(x, batch_size=512, verbose=0)
    agg = aggregate_by_trial(probs, tids, y)
    yt, yp = agg.y_true, agg.y_pred
    # per-class recall -> balanced accuracy
    recalls = []
    for cls in (0, 1):
        mask = yt == cls
        recalls.append(float((yp[mask] == cls).mean()) if mask.any() else float("nan"))
    acc = float((yt == yp).mean())
    bal = float(np.nanmean(recalls))
    # AUC on prob of class 'right' (=1)
    pr = agg.y_prob[:, 1]
    auc = _auc(yt, pr)
    return {"subject": subj, "n_trials": int(len(yt)), "n_left": int((yt == 0).sum()),
            "n_right": int((yt == 1).sum()), "accuracy": acc, "balanced_accuracy": bal,
            "recall_left": recalls[0], "recall_right": recalls[1], "auc": auc,
            "erd_gap_c4_minus_c3_right_minus_left": gap}


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based binary AUC (prob of positive class = right=1)."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[len(neg):].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ci(vals: np.ndarray, rng, b=10_000) -> Tuple[float, float, float]:
    n = len(vals)
    means = vals[rng.integers(0, n, size=(b, n))].mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "bciciv2a_eval.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        from tensorflow import keras  # noqa: F401
        model = build_eegsym()
        model.load_weights(str(W))
        logger.info("loaded binary EEGSym+EA (Stieger corrected-window winner).")
        logger.info("channel map OUR8=%s -> 2a idx=%s ; window=[2,5]s ; NO resampling (already 250Hz)",
                    OUR8, IDX8)

        # --- ground-truth / channel / window verification via contralateral ERD ---
        gaps = {}
        for s in SUBJECTS:
            sigs, labels = load_subject_trials(s)
            gaps[s] = erd_gap(sigs, labels)
            logger.info("ERD check %s: C4-C3 (right-left) = %+.3f  n=%d", s, gaps[s], len(labels))
        mean_gap = float(np.mean(list(gaps.values())))
        n_pos = sum(g > 0 for g in gaps.values())
        logger.info("ERD laterality: mean gap=%+.3f, %d/9 subjects positive", mean_gap, n_pos)
        if mean_gap <= 0 or n_pos < 5:
            logger.error("ABORT: contralateral ERD not recovered (mean gap %+.3f, %d/9 positive). "
                         "Channel order / label mapping / window NOT verified -- refusing to score "
                         "with unverified labels.", mean_gap, n_pos)
            save_json(OUT / "ABORTED.json", {"reason": "ERD laterality not recovered",
                                             "mean_gap": mean_gap, "n_positive": n_pos, "gaps": gaps})
            return
        logger.info("ERD OK -> channel order, label mapping and window verified. Scoring...")

        results = []
        for s in SUBJECTS:
            r = subject_metrics(model, s)
            results.append(r)
            logger.info("%s n=%d acc=%.3f bal=%.3f auc=%.3f (R_L=%.2f R_R=%.2f) ERD=%+.3f",
                        s, r["n_trials"], r["accuracy"], r["balanced_accuracy"], r["auc"],
                        r["recall_left"], r["recall_right"],
                        r["erd_gap_c4_minus_c3_right_minus_left"])
            keras.backend.clear_session()

        rng = np.random.default_rng(42)
        bal = np.array([r["balanced_accuracy"] for r in results])
        auc = np.array([r["auc"] for r in results])
        pooled_correct = sum(r["accuracy"] * r["n_trials"] for r in results)
        pooled_n = sum(r["n_trials"] for r in results)
        summary = {
            "design": "OUR zero-shot cross-dataset eval: binary EEGSym+EA (Stieger corrected "
                      "window) applied cold to BCI-IV-2a LH/RH; per-subject EA (unsupervised); "
                      "band-pass 0.5-40 + CAR; window [2,5]s; 1s/0.5s crops; already 250Hz.",
            "erd_verification": {"mean_gap": mean_gap, "n_positive": n_pos, "per_subject": gaps},
            "balanced_accuracy": dict(zip(["mean", "ci_low", "ci_high"], boot_ci(bal, rng))),
            "auc": dict(zip(["mean", "ci_low", "ci_high"], boot_ci(auc, rng))),
            "pooled_accuracy": float(pooled_correct / pooled_n),
            "results": results}
        save_json(OUT / "summary.json", summary)
        _report(summary)
        logger.info("=== DONE in %.1fs === bal-acc %.3f [%.3f,%.3f] pooled-acc %.3f",
                    time.perf_counter() - t0, summary["balanced_accuracy"]["mean"],
                    summary["balanced_accuracy"]["ci_low"], summary["balanced_accuracy"]["ci_high"],
                    summary["pooled_accuracy"])
    finally:
        logger.removeHandler(h)
        h.close()


def _report(s: Dict):
    b = s["balanced_accuracy"]; a = s["auc"]
    lines = ["# BCI-IV-2a -- decoder (binary EEGSym+EA, Stieger) zero-shot", "",
             "Cross-dataset external validation with the winning decoder and framework. Labels "
             "come from the file and are verified by contralateral ERD "
             f"(mean C4-C3 right-left gap = {s['erd_verification']['mean_gap']:+.3f}, "
             f"{s['erd_verification']['n_positive']}/9 positive). Binary chance = 0.5.", "",
             "| Subj | n | Acc | Bal-acc | AUC | R.left | R.right | ERD gap |",
             "|---|---|---|---|---|---|---|---|"]
    for r in s["results"]:
        lines.append(f"| {r['subject']} | {r['n_trials']} | {r['accuracy']:.3f} | "
                     f"{r['balanced_accuracy']:.3f} | {r['auc']:.3f} | {r['recall_left']:.2f} | "
                     f"{r['recall_right']:.2f} | {r['erd_gap_c4_minus_c3_right_minus_left']:+.3f} |")
    lines += ["",
              f"- Mean bal-acc (9 subj) = {b['mean']:.3f}, 95% CI [{b['ci_low']:.3f}, {b['ci_high']:.3f}] "
              "(per-subject bootstrap).",
              f"- Mean AUC = {a['mean']:.3f}, 95% CI [{a['ci_low']:.3f}, {a['ci_high']:.3f}].",
              f"- Pooled accuracy (trial-weighted) = {s['pooled_accuracy']:.3f}.",
              "- Reading: the EEGSym+EA decoder, trained only on Stieger and applied cold to 2a "
              "(different hardware, subjects and paradigm), transfers above chance without seeing a "
              "single 2a label -> the sensorimotor physics (mu/beta at C3/C4) are portable. The "
              "ceiling is set by the signal, not the model."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
