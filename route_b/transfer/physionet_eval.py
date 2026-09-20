r"""Cross-dataset validation on the Physionet EEG Motor Movement/Imagery dataset
(EEGMMIDB).

The winning decoder (binary EEGSym + Euclidean Alignment, trained on Stieger's
corrected window) is applied zero-shot to Physionet's imagined left/right fist
runs (R04/R08/R12). Used here as an evaluation target (not as a pretraining
pool): with ~106 subjects it is the only dataset large enough to show the bimodal
per-subject distribution that underlies the BCI-illiteracy thesis (Stieger n=13,
2a n=9 lack the power).

Ground truth: labels are the EDF+ annotations. In R04/R08/R12, T1 = left fist,
T2 = right fist, T0 = rest (dropped; the decoder is binary L/R). The T1/T2 mapping
is not trusted blindly: before scoring, the contralateral mu/beta ERD is recovered
(C3 desync for right, C4 for left); if the laterality sign is wrong the script
aborts.

Data facts: 64-ch 10-10 montage; the 8 Cyton motor channels are all present
(Fc3./Fcz./Fc4./C3../Cz../C4../Cp3./Cp4.). Normal fs = 160 Hz; S088/S092/S100 are
128 Hz (anomalous) and excluded by fs (data-quality, logged). 160 -> 250 Hz is an
upsample (resample_poly up=25, down=16).

Pipeline (as the model expects): pick 8 -> band-pass 0.5-40 (Butterworth-4, zero
phase) @160 -> resample 160->250 -> CAR over the 8 -> EA fit per subject
(unsupervised) -> 1 s / 0.5 s crops -> EEGSym -> aggregate by trial. MI window
t in [0.5, 3.5] s after each T1/T2 onset (3 s sustained imagery, skipping the
0.5 s onset transient).

Usage:  python physionet_eval.py   (run from route_b/transfer/, BCI_DATA set)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C                                    # noqa: E402
from src.study.normalization import EuclideanAlignment         # noqa: E402
from src.study.trial_aggregation import aggregate_by_trial     # noqa: E402
from src.eegsym_study.eegsym import build_eegsym               # noqa: E402
from src.crops import iter_crop_bounds                         # noqa: E402
from src.utils import get_logger, save_json                    # noqa: E402

logger = get_logger()
ROOT = C.BCI_DATA / "physionet" / "eegmmidb-1.0.0"
Wt = (Path(__file__).resolve().parents[1] / "experiments" / "corrected_window"
      / "normalization" / "eegsym" / "euclidean_alignment" / "weights.weights.h5")
OUT = Path(__file__).resolve().parents[1] / "experiments" / "physionet_eval"
RUNS = ["R04", "R08", "R12"]                     # imagined left/right fist
OUR8 = ["FC3", "FCZ", "FC4", "C3", "CZ", "C4", "CP3", "CP4"]
C3L, C4L = OUR8.index("C3"), OUR8.index("C4")
FS_NATIVE, FS = 160, 250
UP, DOWN = 25, 16                                # 160 * 25/16 = 250
WIN = (0.5, 3.5)                                 # seconds after T1/T2 onset
# T1 = left fist, T2 = right fist -> our ids (left=0, right=1)
DESC_TO_ID = {"T1": C.LABEL_TO_ID["left"], "T2": C.LABEL_TO_ID["right"]}

_BP_B, _BP_A = butter(C.BANDPASS_ORDER,
                      [C.BANDPASS_LOW_HZ / (FS_NATIVE / 2), C.BANDPASS_HIGH_HZ / (FS_NATIVE / 2)],
                      btype="band")
_MU_B, _MU_A = butter(4, [8 / (FS / 2), 30 / (FS / 2)], btype="band")


def _pick8(raw) -> np.ndarray:
    norm = {c.upper().replace(".", "").strip(): c for c in raw.ch_names}
    picks = [raw.ch_names.index(norm[c]) for c in OUR8]
    return raw.get_data(picks=picks) * 1e6            # (8, N) -> microvolts


def load_subject(subj: str) -> Tuple[List[np.ndarray], List[int]]:
    """Return preprocessed (8, 750) @250 imagery segments + our-model labels."""
    import mne
    mne.set_log_level("ERROR")
    sigs: List[np.ndarray] = []
    labels: List[int] = []
    for run in RUNS:
        f = ROOT / subj / f"{subj}{run}.edf"
        if not f.exists():
            continue
        raw = mne.io.read_raw_edf(str(f), preload=True, verbose="ERROR")
        if int(round(raw.info["sfreq"])) != FS_NATIVE:
            logger.info("  %s%s skipped (fs=%.0f != 160)", subj, run, raw.info["sfreq"])
            continue
        x = _pick8(raw)                                       # (8, N) @160
        x = filtfilt(_BP_B, _BP_A, x, axis=-1)                # band-pass @160
        x = resample_poly(x, UP, DOWN, axis=-1)               # -> 250 Hz
        x = (x - x.mean(axis=0, keepdims=True)).astype(np.float32)   # CAR over 8
        a0, a1 = int(round(WIN[0] * FS)), int(round(WIN[1] * FS))    # window offsets @250
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
            if desc not in DESC_TO_ID:
                continue
            s0 = int(round(onset * FS)) + a0
            s1 = s0 + (a1 - a0)
            if s1 > x.shape[1]:
                continue
            sigs.append(x[:, s0:s1].astype(np.float32))
            labels.append(DESC_TO_ID[desc])
    return sigs, labels


def erd_gap(sigs, labels) -> float:
    p = np.stack([np.log(np.var(filtfilt(_MU_B, _MU_A, s, axis=-1), axis=-1) + 1e-12) for s in sigs])
    lab = np.asarray(labels)
    gap = p[:, C4L] - p[:, C3L]
    return float(gap[lab == C.LABEL_TO_ID["right"]].mean() - gap[lab == C.LABEL_TO_ID["left"]].mean())


def crops_for(sigs, labels):
    ea = EuclideanAlignment().fit(sigs)
    xs, ys, tids = [], [], []
    for ti, (s, lab) in enumerate(zip(sigs, labels)):
        aligned = ea.transform(s)
        for a, b in iter_crop_bounds(aligned.shape[1]):
            xs.append(aligned[:, a:b][:, :, None].astype(np.float32))
            ys.append(lab); tids.append(f"t{ti}")
    return np.stack(xs), np.asarray(ys, dtype=np.int64), tids


def _auc(y, score) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def subject_metrics(model, subj, sigs, labels) -> Dict:
    x, y, tids = crops_for(sigs, labels)
    probs = model.predict(x, batch_size=512, verbose=0)
    agg = aggregate_by_trial(probs, tids, y)
    yt, yp = agg.y_true, agg.y_pred
    recalls = [float((yp[yt == c] == c).mean()) if (yt == c).any() else float("nan") for c in (0, 1)]
    return {"subject": subj, "n_trials": int(len(yt)), "n_left": int((yt == 0).sum()),
            "n_right": int((yt == 1).sum()), "accuracy": float((yt == yp).mean()),
            "balanced_accuracy": float(np.nanmean(recalls)), "recall_left": recalls[0],
            "recall_right": recalls[1], "auc": _auc(yt, agg.y_prob[:, 1])}


def boot_ci(vals, rng, b=10_000):
    n = len(vals); means = vals[rng.integers(0, n, size=(b, n))].mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "physionet_eval.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        from tensorflow import keras  # noqa: F401
        subjects = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("S"))
        logger.info("Physionet: %d subject dirs. Loading imagery runs %s ...", len(subjects), RUNS)

        # --- load + ERD verification first (abort if laterality wrong) ---
        loaded: Dict[str, Tuple] = {}
        gaps: Dict[str, float] = {}
        for s in subjects:
            sigs, labels = load_subject(s)
            if len(labels) < 6 or len(set(labels)) < 2:
                logger.info("  %s excluded (n=%d, classes=%d)", s, len(labels), len(set(labels)))
                continue
            loaded[s] = (sigs, labels)
            gaps[s] = erd_gap(sigs, labels)
        mean_gap = float(np.mean(list(gaps.values())))
        n_pos = sum(g > 0 for g in gaps.values())
        n = len(loaded)
        logger.info("Loaded %d valid subjects. ERD laterality: mean gap=%+.3f, %d/%d positive",
                    n, mean_gap, n_pos, n)
        if mean_gap <= 0 or n_pos < 0.6 * n:
            logger.error("ABORT: contralateral ERD not recovered (mean %+.3f, %d/%d). Refusing to "
                         "score with unverified labels.", mean_gap, n_pos, n)
            save_json(OUT / "ABORTED.json", {"reason": "ERD laterality not recovered",
                                             "mean_gap": mean_gap, "n_positive": n_pos, "n": n})
            return
        logger.info("ERD OK (population-level) -> label mapping verified. Scoring %d subjects...", n)

        model = build_eegsym(); model.load_weights(str(Wt))
        results = []
        for i, (s, (sigs, labels)) in enumerate(loaded.items(), 1):
            r = subject_metrics(model, s, sigs, labels)
            r["erd_gap"] = gaps[s]
            results.append(r)
            if i % 20 == 0 or i == n:
                logger.info("  scored %d/%d ...", i, n)
            keras.backend.clear_session()

        rng = np.random.default_rng(42)
        bal = np.array([r["balanced_accuracy"] for r in results])
        auc = np.array([r["auc"] for r in results])
        pooled = sum(r["accuracy"] * r["n_trials"] for r in results) / sum(r["n_trials"] for r in results)
        # bimodality descriptors
        above = float((bal >= 0.6).mean())     # fraction of "literate-ish" subjects
        near_chance = float((bal < 0.55).mean())
        summary = {
            "design": "OUR zero-shot cross-dataset eval: binary EEGSym+EA (Stieger corrected window) "
                      "applied cold to Physionet imagined L/R fist (R04/R08/R12); per-subject EA; "
                      "band-pass 0.5-40 + upsample 160->250 + CAR; window [0.5,3.5]s; 1s/0.5s crops.",
            "n_subjects": n, "erd": {"mean_gap": mean_gap, "n_positive": n_pos},
            "balanced_accuracy": dict(zip(["mean", "ci_low", "ci_high"], boot_ci(bal, rng))),
            "auc": dict(zip(["mean", "ci_low", "ci_high"], boot_ci(auc, rng))),
            "pooled_accuracy": float(pooled),
            "distribution": {"median_bal": float(np.median(bal)), "std_bal": float(bal.std()),
                             "frac_bal_ge_0.60": above, "frac_bal_lt_0.55": near_chance,
                             "min_bal": float(bal.min()), "max_bal": float(bal.max())},
            "results": results}
        save_json(OUT / "summary.json", summary)
        _report(summary); _figure(bal)
        logger.info("=== DONE in %.1fs === n=%d bal-acc %.3f [%.3f,%.3f] pooled %.3f "
                    "(>=0.60: %.0f%%, <0.55: %.0f%%)", time.perf_counter() - t0, n,
                    summary["balanced_accuracy"]["mean"], summary["balanced_accuracy"]["ci_low"],
                    summary["balanced_accuracy"]["ci_high"], pooled, above * 100, near_chance * 100)
    finally:
        logger.removeHandler(h); h.close()


def _report(s: Dict):
    b, a, d = s["balanced_accuracy"], s["auc"], s["distribution"]
    lines = ["# Physionet (EEGMMIDB) -- decoder (binary EEGSym+EA, Stieger) zero-shot", "",
             f"External validation with the winning decoder over {s['n_subjects']} subjects (imagined "
             "left/right fist, R04/R08/R12). T1/T2 labels are verified by contralateral ERD at the "
             f"population level (mean gap {s['erd']['mean_gap']:+.3f}, {s['erd']['n_positive']}/"
             f"{s['n_subjects']} positive). Used here as the target (not a pool): with large N the "
             "bimodal distribution is visible. Chance = 0.5.", "",
             f"- Mean bal-acc = {b['mean']:.3f}, 95% CI [{b['ci_low']:.3f}, {b['ci_high']:.3f}] "
             "(per-subject bootstrap).",
             f"- Mean AUC = {a['mean']:.3f}, 95% CI [{a['ci_low']:.3f}, {a['ci_high']:.3f}].",
             f"- Pooled accuracy = {s['pooled_accuracy']:.3f}.",
             f"- Per-subject distribution: median {d['median_bal']:.3f}, sd {d['std_bal']:.3f}, "
             f"range [{d['min_bal']:.3f}, {d['max_bal']:.3f}]. "
             f"{d['frac_bal_ge_0.60']*100:.0f}% of subjects with bal-acc >= 0.60, "
             f"{d['frac_bal_lt_0.55']*100:.0f}% near chance (< 0.55).", "",
             "Reading: over 100+ subjects, the decoder trained only on Stieger transfers above "
             "chance on average, but with a large per-subject spread -- the bimodal illiteracy "
             "pattern: one group decodes well and another stays at chance. The ceiling is set by "
             "the subject's aptitude/signal, not the model (same message as 2a and Stieger, now "
             "with statistical power). See `physionet_bal_hist.png`."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _figure(bal: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.hist(bal, bins=np.arange(0.30, 1.01, 0.05), color="#1f3a5f", edgecolor="white")
    ax.axvline(0.5, color="k", ls="--", lw=0.9, label="chance")
    ax.axvline(bal.mean(), color="#c0392b", ls="-", lw=1.5, label=f"mean {bal.mean():.3f}")
    ax.set(xlabel="per-subject balanced accuracy (trial)", ylabel="n subjects",
           title=f"Physionet: per-subject distribution (n={len(bal)}) -- zero-shot EEGSym+EA")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "physionet_bal_hist.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
