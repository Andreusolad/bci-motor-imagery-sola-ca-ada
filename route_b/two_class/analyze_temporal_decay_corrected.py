"""Per-crop discriminability over the trial, on the corrected window.

Takes the winning corrected-window model (``[2000, 2000+triallength]`` ms, no
cue) and groups the test trials into three duration bands -- up to 2 s, 2-4 s
and 4-6 s of imagery (``triallength``) -- then measures whether the model's
discriminative signal (mean softmax mass on the true class, ``p_true``) decays
across the trial within each band.

Method (model-based temporal profiling, no retraining):

1. Rebuild the winning architecture and load its trained corrected-window
   weights.
2. Normalize every test trial (corrected window) with the winning method and
   slice it into the model's own crops (same length/stride it was trained
   with), recording each crop's absolute start time within the corrected
   window (seconds since feedback onset, not since cue onset).
3. Predict every crop's softmax; ``p_true`` = probability mass on the trial's
   true class.
4. Group trials by ``triallength`` into (0,2], (2,4], (4,6] second bands.
   Within each band, average ``p_true`` over crops whose centre falls in each
   0.5 s time-bin of the (corrected) trial.

Writes into ``experiments/corrected_window/temporal_analysis/``.

Usage:  python analyze_temporal_decay_corrected.py   (run from route_b/two_class/, BCI_DATA set)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as base_config
from src.crop_size_study import build_crops_sized
from src.dataset import imagery_window_from_feedback
from src.split import load_split, session_keys_for
from src.study.data_loading import load_trials_for_sessions
from src.study.normalization import EuclideanAlignment, RunningExponentialStandardizer, ZScoreNormalizer
from src.utils import get_logger, set_global_seed

logger = get_logger()

# --------------------------------------------------------------------------- #
# Winning combination -- update these three constants once the corrected-
# window normalization comparison (run_corrected_window_study.py) finishes.
# --------------------------------------------------------------------------- #
ARCHITECTURE = "eegsym"           # "eegnet" | "eegsym"
METHOD = "euclidean_alignment"    # "z_score" | "running_exponential" | "euclidean_alignment"
CROP_SECONDS = 1.0                # crop length the winning model was trained with

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = (
    ROOT / "experiments" / "corrected_window" / "normalization" / ARCHITECTURE / METHOD / "weights.weights.h5"
)
OUT_DIR = ROOT / "experiments" / "corrected_window" / "temporal_analysis"

RANDOM_SEED = 42
FS = 250
WINDOW = int(round(CROP_SECONDS * FS))
STRIDE = WINDOW // 2  # 50% overlap
TIME_BIN_S = 0.5
MIN_TRIALS_PER_BUCKET = 30

# (label, lower_exclusive, upper_inclusive) in seconds of triallength.
_BUCKETS: List[Tuple[str, float, float]] = [
    ("up to 2 s", 0.0, 2.0),
    ("2-4 s", 2.0, 4.0),
    ("4-6 s", 4.0, 6.0),
]


def _bucket_for(duration_s: float) -> str | None:
    for label, lo, hi in _BUCKETS:
        if lo < duration_s <= hi:
            return label
    return None


def _build_model():
    if ARCHITECTURE == "eegnet":
        from src.study.eegnet import build_eegnet
        return build_eegnet()
    if ARCHITECTURE == "eegsym":
        from src.eegsym_study.eegsym import build_eegsym
        return build_eegsym()
    raise ValueError(f"Unknown architecture: {ARCHITECTURE!r}")


def _build_normalizer(method: str, train_trials):
    if method == "running_exponential":
        return RunningExponentialStandardizer()
    if method == "z_score":
        zs = ZScoreNormalizer()
        zs.fit([t.signal for t in train_trials])
        return zs
    if method == "euclidean_alignment":
        ea = EuclideanAlignment()
        ea.fit([t.signal for t in train_trials])
        return ea
    raise ValueError(f"Unknown normalization method: {method!r}")


def main() -> None:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"{WEIGHTS_PATH} not found -- run the corrected-window normalization study "
            "and set ARCHITECTURE/METHOD/CROP_SECONDS at the top of this script first."
        )
    set_global_seed(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = _build_model()
    model.load_weights(WEIGHTS_PATH)
    logger.info("Rebuilt %s and loaded corrected-window weights from %s", ARCHITECTURE, WEIGHTS_PATH)

    split_payload = load_split()
    # z_score / euclidean_alignment need train trials (corrected window) to fit on.
    train_trials = load_trials_for_sessions(
        session_keys_for(split_payload, "train"), window_fn=imagery_window_from_feedback
    )
    test_trials = load_trials_for_sessions(
        session_keys_for(split_payload, "test"), window_fn=imagery_window_from_feedback
    )
    logger.info("Loaded %d train trials (for normalizer fitting) and %d test trials.",
                len(train_trials), len(test_trials))

    duration_by_trial: Dict[str, float] = {t.trial_id: t.signal.shape[1] / FS for t in test_trials}
    bucket_by_trial: Dict[str, str] = {
        tid: b for tid, d in duration_by_trial.items() if (b := _bucket_for(d)) is not None
    }
    bucket_trial_counts: Dict[str, set] = defaultdict(set)
    for tid, b in bucket_by_trial.items():
        bucket_trial_counts[b].add(tid)
    for label, _, _ in _BUCKETS:
        logger.info("Bucket %-10s -> %d test trials", label, len(bucket_trial_counts.get(label, set())))

    normalizer = _build_normalizer(METHOD, train_trials)
    crops = build_crops_sized(test_trials, normalizer.transform, WINDOW, STRIDE)
    probs = model.predict(crops.x, batch_size=512, verbose=0)

    accum: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    for i, crop_id in enumerate(crops.crop_ids):
        trial_id = crops.trial_ids[i]
        bucket = bucket_by_trial.get(trial_id)
        if bucket is None:
            continue
        start = int(crop_id.rsplit("#crop", 1)[1])
        centre_s = (start + WINDOW / 2) / FS
        time_bin = np.floor(centre_s / TIME_BIN_S) * TIME_BIN_S + TIME_BIN_S / 2
        p_true = float(probs[i, int(crops.y[i])])
        accum[(bucket, round(float(time_bin), 3))].append(p_true)

    profiles: Dict[str, object] = {}
    for label, _, _ in _BUCKETS:
        n_trials = len(bucket_trial_counts.get(label, set()))
        if n_trials < MIN_TRIALS_PER_BUCKET:
            logger.warning("Skipping bucket %s: only %d trials (< %d).", label, n_trials, MIN_TRIALS_PER_BUCKET)
            continue
        bins = sorted({tb for (bb, tb) in accum if bb == label})
        points = []
        for tb in bins:
            vals = accum[(label, tb)]
            points.append({
                "time_s": tb, "mean_p_true": float(np.mean(vals)),
                "sem_p_true": float(np.std(vals) / np.sqrt(len(vals))), "n_crops": len(vals),
            })
        peak = max(points, key=lambda p: p["mean_p_true"])
        first_pt, last_pt = points[0], points[-1]
        profiles[label] = {
            "n_trials": n_trials,
            "peak_time_s": peak["time_s"], "peak_mean_p_true": peak["mean_p_true"],
            "first_bin_p_true": first_pt["mean_p_true"], "last_bin_p_true": last_pt["mean_p_true"],
            "delta_first_to_last": last_pt["mean_p_true"] - first_pt["mean_p_true"],
            "points": points,
        }
        logger.info("Bucket %-10s | %4d trials | peak %.2fs (p=%.3f) | first=%.3f last=%.3f",
                    label, n_trials, peak["time_s"], peak["mean_p_true"],
                    first_pt["mean_p_true"], last_pt["mean_p_true"])

    payload = {
        "architecture": ARCHITECTURE, "normalization_method": METHOD,
        "window": "corrected", "model_weights": str(WEIGHTS_PATH),
        "crop_window_samples": WINDOW, "crop_stride_samples": STRIDE, "fs": FS,
        "time_bin_s": TIME_BIN_S,
        "measure": ("mean p_true (softmax mass on the true class) of the crops whose centre falls "
                    "in each absolute time-bin (seconds since feedback onset) of the corrected-window "
                    "trial; higher = more discriminative"),
        "min_trials_per_bucket": MIN_TRIALS_PER_BUCKET,
        "duration_buckets": profiles,
    }
    (OUT_DIR / "temporal_decay_corrected.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved %s", OUT_DIR / "temporal_decay_corrected.json")
    _plot(profiles, OUT_DIR / "temporal_decay_corrected.png")


def _plot(profiles: Dict[str, object], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("viridis")
    labels = list(profiles.keys())
    for k, b in enumerate(labels):
        pts = profiles[b]["points"]
        xs = [p["time_s"] for p in pts]
        ys = [p["mean_p_true"] for p in pts]
        es = [p["sem_p_true"] for p in pts]
        color = cmap(k / max(len(labels) - 1, 1))
        ax.errorbar(xs, ys, yerr=es, marker="o", markersize=4, capsize=2,
                    color=color, label=f"{b} (n={profiles[b]['n_trials']})")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set(
        title=f"{ARCHITECTURE}+{METHOD}, corrected window -- discriminability per second of the trial",
        xlabel="Time since feedback onset (s, crop centre)",
        ylabel="Mean probability on the true class (p_true)",
    )
    ax.grid(alpha=0.3)
    ax.legend(title="Trial duration", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
