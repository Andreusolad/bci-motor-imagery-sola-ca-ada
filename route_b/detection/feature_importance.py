r"""Feature importance (channels + frequency bands) for the 3-class EEGSym+EA
model, measured on the test crops.

- Channels: permutation importance. For each of the 8 electrodes (post-CAR,
  pre-EA), shuffle that channel across crops, re-apply the fixed EA, re-predict,
  and measure the drop in balanced accuracy (and per-class recall). Permutation
  keeps the input in-distribution (uses real channel data from other crops).
  EA is applied per-crop (linear per-sample -> identical to whole-segment EA).
- Frequencies: band-ablation importance. For each band, remove it from each test
  segment (subtract its band-pass component; longer segment = clean filtering),
  re-EA, re-crop, re-predict, measure the drop. Bands: delta/theta/mu/low-beta/
  high-beta/gamma.

Per-class drops reveal whether a channel/band matters for REST-vs-MI detection
or for LEFT-vs-RIGHT direction.

Usage:  python feature_importance.py   (run from route_b/detection/, BCI_DATA set)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import config3 as C                          # noqa: E402
from lib import crops3, ea_io, metrics3, model3       # noqa: E402
from lib.segments import Segment, extract_segments_for_sessions  # noqa: E402

from src.split import load_split, session_keys_for    # noqa: E402
from src.utils import get_logger, save_json           # noqa: E402

import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402

logger = get_logger()
OUT = Path(__file__).resolve().parents[1] / "experiments" / "feature_importance"
FLAT_DIR = C.EXPERIMENTS / "rest_3class"
CH = ["FC3", "FCZ", "FC4", "C3", "CZ", "C4", "CP3", "CP4"]
FS = C.FS_TARGET
BANDS = [("delta", 0.5, 4), ("theta", 4, 8), ("mu/alpha", 8, 13),
         ("low-beta", 13, 20), ("high-beta", 20, 30), ("gamma", 30, 40)]


def electrode_crops(segments: List[Segment]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, tids = [], [], []
    for s in segments:
        for a, b in crops3.iter_crop_bounds(s.signal.shape[1]):
            X.append(s.signal[:, a:b]); y.append(s.label); tids.append(s.trial_id)
    return np.stack(X).astype(np.float32), np.asarray(y, np.int64), tids


def bandpass(sig: np.ndarray, lo: float, hi: float) -> np.ndarray:
    ny = FS / 2
    blo, bhi = max(lo, 0.1) / ny, min(hi, ny - 0.1) / ny
    b, a = butter(4, [blo, bhi], btype="band")
    return filtfilt(b, a, sig, axis=1)


def band_removed_crops(segments: List[Segment], lo: float, hi: float) -> np.ndarray:
    X = []
    for s in segments:
        removed = s.signal - bandpass(s.signal, lo, hi)
        for a, b in crops3.iter_crop_bounds(removed.shape[1]):
            X.append(removed[:, a:b])
    return np.stack(X).astype(np.float32)


def score(model, W, Xe: np.ndarray, y: np.ndarray) -> Dict:
    Xea = np.einsum("ij,njt->nit", W, Xe)[..., None].astype(np.float32)
    pred = model.predict(Xea, batch_size=512, verbose=0).argmax(1)
    m = metrics3.compute_metrics(y, pred)
    return {"balanced_accuracy": m["balanced_accuracy"], **m["per_class_accuracy"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "feature_importance.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    t0 = time.perf_counter()
    try:
        split = load_split(C.SPLIT_JSON)
        model = model3.build_model3("eegsym"); model.load_weights(FLAT_DIR / "weights.weights.h5")
        W = ea_io.load_ea_matrix(FLAT_DIR / "ea_reference.json")

        segs = extract_segments_for_sessions(session_keys_for(split, "test"))
        segs, _ = crops3.balance_rest(segs, seed=C.RANDOM_SEED)     # same balancing as its training
        Xe, y, _ = electrode_crops(segs)
        logger.info("test crops: %d | classes %s", len(y), np.bincount(y).tolist())

        base = score(model, W, Xe, y)
        logger.info("baseline: %s", {k: round(v, 3) for k, v in base.items()})

        rng = np.random.default_rng(C.RANDOM_SEED)
        # --- channel permutation importance --- #
        chan = []
        for c in range(8):
            Xp = Xe.copy()
            Xp[:, c, :] = Xe[rng.permutation(len(Xe)), c, :]
            s = score(model, W, Xp, y)
            chan.append({"channel": CH[c],
                         "drop_balanced_accuracy": base["balanced_accuracy"] - s["balanced_accuracy"],
                         "drop_REST": base["REST"] - s["REST"],
                         "drop_LEFT": base["LEFT"] - s["LEFT"],
                         "drop_RIGHT": base["RIGHT"] - s["RIGHT"]})
            logger.info("chan %-4s drop bal_acc=%.3f (REST %.3f LEFT %.3f RIGHT %.3f)",
                        CH[c], chan[-1]["drop_balanced_accuracy"], chan[-1]["drop_REST"],
                        chan[-1]["drop_LEFT"], chan[-1]["drop_RIGHT"])

        # --- frequency band ablation importance --- #
        freq = []
        for name, lo, hi in BANDS:
            Xf = band_removed_crops(segs, lo, hi)
            s = score(model, W, Xf, y)
            freq.append({"band": name, "range_hz": [lo, hi],
                         "drop_balanced_accuracy": base["balanced_accuracy"] - s["balanced_accuracy"],
                         "drop_REST": base["REST"] - s["REST"],
                         "drop_LEFT": base["LEFT"] - s["LEFT"],
                         "drop_RIGHT": base["RIGHT"] - s["RIGHT"]})
            logger.info("band %-9s drop bal_acc=%.3f (REST %.3f LEFT %.3f RIGHT %.3f)",
                        name, freq[-1]["drop_balanced_accuracy"], freq[-1]["drop_REST"],
                        freq[-1]["drop_LEFT"], freq[-1]["drop_RIGHT"])

        save_json(OUT / "summary.json", {"model": "3-class EEGSym+EA", "baseline": base,
                                         "channels": chan, "frequencies": freq})
        _figures(base, chan, freq)
        _report(base, chan, freq)
        logger.info("=== DONE in %.1fs ===", time.perf_counter() - t0)
    finally:
        logger.removeHandler(h)
        h.close()


def _figures(base, chan, freq):
    for data, key, names, title, path in [
        (chan, "channel", [d["channel"] for d in chan], "Channel importance (permutation)", "channels.png"),
        (freq, "band", [d["band"] for d in freq], "Frequency-band importance (ablation)", "frequencies.png")]:
        x = np.arange(len(data)); w = 0.2
        fig, ax = plt.subplots(figsize=(max(7, len(data) * 1.1), 4.6))
        for i, (cls, col) in enumerate([("drop_balanced_accuracy", "#1f3a5f"), ("drop_REST", "#8a8f98"),
                                        ("drop_LEFT", "#0e7c74"), ("drop_RIGHT", "#c0873a")]):
            ax.bar(x + (i - 1.5) * w, [d[cls] for d in data], w, label=cls.replace("drop_", ""), color=col)
        ax.axhline(0, color="k", lw=0.6)
        ax.set(xticks=x, ylabel="performance drop (importance)", title=title)
        ax.set_xticklabels(names, rotation=20); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(OUT / path, dpi=130); plt.close(fig)


def _report(base, chan, freq):
    lines = ["# Channel and frequency importance (3-class EEGSym+EA model)", "",
             f"Baseline balanced accuracy = {base['balanced_accuracy']:.3f} "
             f"(REST {base['REST']:.3f}, LEFT {base['LEFT']:.3f}, RIGHT {base['RIGHT']:.3f}). "
             "Importance = drop when perturbed (larger = more important).", "",
             "## Channels (permutation)", "",
             "| Channel | delta balanced acc | delta REST | delta LEFT | delta RIGHT |", "|---|---|---|---|---|"]
    for d in sorted(chan, key=lambda z: -z["drop_balanced_accuracy"]):
        lines.append(f"| {d['channel']} | {d['drop_balanced_accuracy']:+.3f} | {d['drop_REST']:+.3f} | "
                     f"{d['drop_LEFT']:+.3f} | {d['drop_RIGHT']:+.3f} |")
    lines += ["", "## Frequencies (band ablation)", "",
              "| Band (Hz) | delta balanced acc | delta REST | delta LEFT | delta RIGHT |", "|---|---|---|---|---|"]
    for d in sorted(freq, key=lambda z: -z["drop_balanced_accuracy"]):
        lines.append(f"| {d['band']} ({d['range_hz'][0]}-{d['range_hz'][1]}) | "
                     f"{d['drop_balanced_accuracy']:+.3f} | {d['drop_REST']:+.3f} | "
                     f"{d['drop_LEFT']:+.3f} | {d['drop_RIGHT']:+.3f} |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
