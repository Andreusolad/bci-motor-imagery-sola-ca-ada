"""Frequency-band permutation importance for the winning model (EEGSym+EA, corrected window).

Answers: which frequency bands does the model actually rely on to tell
left from right? Motor imagery is expected to modulate mu (~8-12 Hz) and
beta (~13-30 Hz) rhythms over C3/C4 (event-related desynchronization).
This checks whether the trained model's behaviour is consistent with that.

Method: band-permutation importance, the same logic as the channel and
time-segment permutation importance, generalized to the frequency axis:

1. Band-pass filter every test crop (zero-phase Butterworth, order 4) into
   ``in_band`` (content inside the band, all 8 channels) and
   ``out_of_band = crop - in_band`` (everything else, computed once per
   band -- deterministic, not part of the repeated shuffles).
2. Shuffle ``in_band`` across crops (breaking the association between that
   band's content and the label while preserving each crop's own
   out-of-band content and every other crop's marginal distribution),
   reconstruct ``out_of_band + shuffled_in_band``, predict, and measure the
   drop in trial-level accuracy/F1/AUC relative to the unpermuted baseline.
3. Repeat 10 times per band (different random shuffles) and report
   mean +/- std, exactly as for channels/time segments.

Canonical bands (motor-imagery literature): delta, theta, mu/alpha, low
beta, high beta, low gamma -- the last one bounded by the project's own
0.5-40 Hz preprocessing filter.

Writes into ``experiments/corrected_window/frequency_analysis/``.

Usage:  python analyze_frequency_importance_corrected.py   (run from route_b/two_class/, BCI_DATA set)
"""
from __future__ import annotations
import sys

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import butter, filtfilt  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as base_config
from src.crop_size_study import build_crops_sized
from src.dataset import imagery_window_from_feedback
from src.eegsym_study.eegsym import build_eegsym
from src.split import load_split, session_keys_for
from src.study.data_loading import load_trials_for_sessions
from src.study.normalization import EuclideanAlignment
from src.study.trial_aggregation import aggregate_by_trial
from src.utils import get_logger, set_global_seed

logger = get_logger()

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = (
    ROOT / "experiments" / "corrected_window" / "normalization" / "eegsym"
    / "euclidean_alignment" / "weights.weights.h5"
)
OUT_DIR = ROOT / "experiments" / "corrected_window" / "frequency_analysis"

RANDOM_SEED = 42
FS = 250
WINDOW = 250   # 1 s crops, exactly as the winning model was trained
STRIDE = 125   # 50% overlap
N_REPEATS = 10
POSITIVE_ID = base_config.LABEL_TO_ID["right"]

# (label, low_hz, high_hz). Upper edge capped at 40 Hz -- the project's own
# preprocessing band-pass (Section "Metodologia comun") already removes
# everything above that, so a "gamma" band above 40 Hz would be empty.
BANDS: List[Tuple[str, float, float]] = [
    ("delta (0.5-4 Hz)", 0.5, 4.0),
    ("theta (4-8 Hz)", 4.0, 8.0),
    ("mu/alfa (8-12 Hz)", 8.0, 12.0),
    ("beta baja (13-20 Hz)", 13.0, 20.0),
    ("beta alta (20-30 Hz)", 20.0, 30.0),
    ("gamma baja (30-40 Hz)", 30.0, 40.0),
]


def _load_test_and_train_trials():
    split_payload = load_split()
    train_trials = load_trials_for_sessions(
        session_keys_for(split_payload, "train"), window_fn=imagery_window_from_feedback
    )
    test_trials = load_trials_for_sessions(
        session_keys_for(split_payload, "test"), window_fn=imagery_window_from_feedback
    )
    return train_trials, test_trials


def _trial_metrics(model, x: np.ndarray, trial_ids, y: np.ndarray) -> Dict[str, float]:
    crop_probs = model.predict(x, batch_size=512, verbose=0)
    trial_agg = aggregate_by_trial(crop_probs, trial_ids, y)
    y_true, y_pred = trial_agg.y_true, trial_agg.y_pred
    y_score = trial_agg.y_prob[:, POSITIVE_ID]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def _bandpass(x: np.ndarray, low: float, high: float, fs: int) -> np.ndarray:
    """Zero-phase Butterworth band-pass, vectorized over (n_crops, n_channels, n_samples)."""
    nyq = fs / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x, axis=-1, method="gust")


def main() -> None:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"{WEIGHTS_PATH} not found -- train the winning model first.")
    set_global_seed(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_eegsym()
    model.load_weights(WEIGHTS_PATH)
    logger.info("Rebuilt EEGSym and loaded corrected-window EA weights from %s", WEIGHTS_PATH)

    train_trials, test_trials = _load_test_and_train_trials()
    logger.info("Loaded %d train trials (EA reference fit) and %d test trials.",
                len(train_trials), len(test_trials))

    ea = EuclideanAlignment()
    ea.fit([t.signal for t in train_trials])

    crops = build_crops_sized(test_trials, ea.transform, WINDOW, STRIDE)
    x = crops.x[..., 0]  # (n_crops, n_channels, n_samples)
    logger.info("Built %d test crops from %d trials.", len(crops), len(test_trials))

    baseline = _trial_metrics(model, crops.x, crops.trial_ids, crops.y)
    logger.info("Baseline trial metrics: %s", baseline)

    band_importance: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for label, low, high in BANDS:
        logger.info("Filtering all test crops into band %s (%.1f-%.1f Hz)...", label, low, high)
        in_band = _bandpass(x, low, high, FS)
        out_of_band = x - in_band

        drops: Dict[str, List[float]] = {"accuracy": [], "f1_macro": [], "roc_auc": []}
        for r in range(N_REPEATS):
            rng = np.random.default_rng(RANDOM_SEED + hash(label) % 1000 * 1000 + r)
            order = rng.permutation(x.shape[0])
            reconstructed = (out_of_band + in_band[order])[..., None].astype(np.float32)
            permuted_metrics = _trial_metrics(model, reconstructed, crops.trial_ids, crops.y)
            for key in drops:
                drops[key].append(baseline[key] - permuted_metrics[key])

        band_importance[label] = {
            key: (float(np.mean(vals)), float(np.std(vals))) for key, vals in drops.items()
        }
        logger.info("Band %-24s -> accuracy drop = %.4f +/- %.4f",
                    label, *band_importance[label]["accuracy"])

    payload = {
        "architecture": "eegsym",
        "normalization_method": "euclidean_alignment",
        "window": "corrected",
        "model_weights": str(WEIGHTS_PATH),
        "test_crops": len(crops),
        "test_trials": len(set(crops.trial_ids)),
        "n_repeats_per_band": N_REPEATS,
        "method": (
            "Band-permutation importance: for each band, band-pass filter every test crop "
            "(order-4 zero-phase Butterworth) into in-band and out-of-band content, shuffle the "
            "in-band content across crops (all 8 channels together), reconstruct, and measure the "
            f"drop in trial-level metrics vs. the unpermuted baseline. Mean +/- std over {N_REPEATS} "
            "independent shuffles per band."
        ),
        "baseline_trial_metrics": baseline,
        "band_importance": band_importance,
    }
    (OUT_DIR / "importancia_por_frecuencia.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved %s", OUT_DIR / "importancia_por_frecuencia.json")

    _plot(band_importance, OUT_DIR / "frequency_importance.png")


def _plot(band_importance: Dict[str, Dict[str, Tuple[float, float]]], out_path: Path) -> None:
    bands = list(band_importance.keys())
    means = [band_importance[b]["accuracy"][0] * 100 for b in bands]
    stds = [band_importance[b]["accuracy"][1] * 100 for b in bands]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#0e7c74" if m >= 0 else "#b23a48" for m in means]
    ax.bar(bands, means, yerr=stds, capsize=4, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        title="EEGSym+EA, corrected window -- per-band permutation importance",
        ylabel="Trial accuracy drop (percentage points)",
        xlabel="Frequency band",
    )
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
