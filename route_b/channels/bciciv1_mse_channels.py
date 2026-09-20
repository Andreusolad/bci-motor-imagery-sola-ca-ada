r"""Transfer of the channel-count models to BCI-IV-1 (official competition MSE).

Applies each binary EEGNet+EA channel model (8/16/32, trained on Stieger) zero-shot
to the BCI-IV-1 eval stream, scored with the official competition MSE on the SOFT
output P(right)-P(left). Same methodology as joint_paper/bciciv1_mse_eval.py, only
the model (binary EEGNet at N channels) and the channel selection change.

Gate: zero-output MSE over real subjects must reproduce ~0.509.
BCI-IV-1 lacks FT7/FT8/TP7/TP8, so the 32-channel model cannot be applied without
a montage gap -> that set is reported as "montage mismatch" (a portability finding),
not zero-filled.

Usage:  python bciciv1_mse_channels.py --set 16   (run from route_b/channels/, BCI_DATA set)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as base_config                       # noqa: E402
from src.preprocessing import preprocess_trial              # noqa: E402
from src.study.normalization import EuclideanAlignment      # noqa: E402
from src.study.eegnet import build_eegnet                   # noqa: E402
from src.utils import get_logger, save_json                 # noqa: E402

from train_channels import SETS                             # noqa: E402

logger = get_logger()
EVAL = base_config.BCI_DATA / "bci_iv_1" / "BCICIV_1eval_1000Hz_mat"
LAB = base_config.BCI_DATA / "bci_iv_1" / "true_labels_official" / "mat"
OUT_ROOT = Path(__file__).resolve().parents[1] / "experiments"
CH = base_config.CROP_SAMPLES        # 250
HOP = CH // 2                        # 125
DS = 4                               # 1000 -> 250
SUBJECTS = list("abcdefg")
REAL = set("abfg")


def weights_for(nch: int) -> Path:
    if nch == 8:
        return (Path(__file__).resolve().parents[1] / "experiments" / "corrected_window"
                / "normalization" / "eegnet" / "euclidean_alignment" / "weights.weights.h5")
    return OUT_ROOT / f"eegnet_ea_{nch}ch" / "weights.weights.h5"


def channel_index(clab: List[str], want: List[str]):
    low = {c.lower(): i for i, c in enumerate(clab)}
    miss = [c for c in want if c.lower() not in low]
    return ([low[c.lower()] for c in want] if not miss else None), miss


def load_eval(subj: str, want: List[str]):
    m = loadmat(str(EVAL / f"BCICIV_eval_ds1{subj}_1000Hz.mat"), struct_as_record=False, squeeze_me=True)
    clab = [str(c) for c in np.asarray(m["nfo"].clab).ravel()]
    idx, miss = channel_index(clab, want)
    if idx is None:
        return None, None, miss
    cnt = np.asarray(m["cnt"])[:, idx].astype(np.float32).T * 0.1
    ymat = loadmat(str(LAB / f"BCICIV_eval_ds1{subj}_1000Hz_true_y.mat"), squeeze_me=True)
    yk = max((k for k in ymat if not k.startswith("__")), key=lambda k: np.asarray(ymat[k]).size)
    y = np.asarray(ymat[yk]).astype(np.float64).ravel()[:cnt.shape[1]]
    return cnt, y, miss


def mse_nonan(pred, y):
    keep = np.isfinite(y)
    return float(np.mean((pred[keep] - y[keep]) ** 2))


def eval_subject(model, subj, want) -> Dict:
    cnt, y1000, _ = load_eval(subj, want)
    sig = preprocess_trial(cnt); M = sig.shape[1]
    ea = EuclideanAlignment().fit([sig]); aligned = ea.transform(sig)
    starts = list(range(0, M - CH + 1, HOP))
    X = np.stack([aligned[:, s:s + CH][:, :, None] for s in starts]).astype(np.float32)
    probs = model.predict(X, batch_size=512, verbose=0)
    soft_w = probs[:, 1] - probs[:, 0]                       # P(right)-P(left)
    centres = np.clip(np.array([int(round((s + CH / 2) * DS)) for s in starts]), 0, len(y1000) - 1)
    t = np.arange(len(y1000)); j = np.clip(np.searchsorted(centres, t), 1, len(centres) - 1)
    nearest = np.where((t - centres[j - 1]) <= (centres[j] - t), j - 1, j)
    return {"subject": subj, "real": subj in REAL, "n_scored": int(np.isfinite(y1000).sum()),
            "mse_zero": mse_nonan(np.zeros_like(soft_w[nearest]), y1000),
            "mse_soft": mse_nonan(soft_w[nearest], y1000)}


def run(nch: int):
    want = SETS[nch]
    out = OUT_ROOT / f"eegnet_ea_{nch}ch"; out.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(out / "bciciv1_mse.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h); t0 = time.perf_counter()
    try:
        # montage-availability check on one subject
        _, _, miss = load_eval("b", want)
        if miss:
            logger.warning("%dch: BCI-IV-1 is MISSING %s -> montage mismatch, cannot transfer.", nch, miss)
            save_json(out / "bciciv1_mse.json", {"n_channels": nch, "status": "montage_mismatch",
                                                 "missing_channels": miss})
            return
        model = build_eegnet(input_shape=(nch, CH, 1), n_classes=2)
        model.load_weights(str(weights_for(nch)))
        logger.info("%dch: loaded model, scoring BCI-IV-1 eval (competition MSE, soft output)...", nch)
        results = [eval_subject(model, s, want) for s in SUBJECTS]
        real = [r for r in results if r["real"]]
        def pooled(rows, k):
            w = np.array([r["n_scored"] for r in rows], float); v = np.array([r[k] for r in rows], float)
            return float((w * v).sum() / w.sum())
        summ = {"n_channels": nch, "status": "ok",
                "gate_zero_real": pooled(real, "mse_zero"),
                "pooled_real_soft": pooled(real, "mse_soft"),
                "per_subject": results}
        save_json(out / "bciciv1_mse.json", summ)
        logger.info("=== %dch BCI-IV-1 DONE in %.1fs === gate=%.4f soft(real)=%.4f",
                    nch, time.perf_counter() - t0, summ["gate_zero_real"], summ["pooled_real_soft"])
    finally:
        logger.removeHandler(h); h.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=int, choices=[8, 16, 32], required=True)
    run(ap.parse_args().set)
