r"""Train a regressor directly on BCI-IV-1 (per subject).

Everything before this used a decoder trained on another dataset (Stieger)
applied zero-shot -> best competition MSE 0.4795 (champion 0.382, baseline
0.509). Here the strategy switches, exactly like the competitors did: train on
BCI-IV-1's own calibration recording, and train a regressor whose loss is the
competition metric (MSE on a continuous [-1,+1] output), instead of a classifier.

Per subject:
  * calib -> crops with regression targets: left MI = -1, right MI = +1, rest = 0
    (reuses bciciv1_calibration.build_calib_set; labels are file ground truth,
    ERD-verified). EA fit on calib.
  * train a small EEGNet-style regressor (tanh output, MSE loss, early stopping).
  * eval stream -> EA per subject -> sliding window -> continuous output -> EMA
    smoothing -> official MSE.

Compared against the zero-shot 0.4795, the champion 0.382 and the 0.509 baseline.
Real subjects a,b,f,g (c,d,e generated; a,f left/foot). Competition-legal: no
external data, each subject trained on its own calib only.

Usage:  python bciciv1_regressor.py   (run from route_b/bci_iv_1/, BCI_DATA set)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import ea_io                                  # noqa: E402
from lib import config3 as C                            # noqa: E402
from src.preprocessing import preprocess_trial          # noqa: E402
from src.utils import get_logger, save_json, set_global_seed  # noqa: E402
from bciciv1_mse_eval import load_eval, CROP, HOP, DS   # noqa: E402
from bciciv1_calibration import build_calib_set         # noqa: E402

logger = get_logger()
OUT = Path(__file__).resolve().parents[1] / "experiments" / "bciciv1_regressor"
SUBJECTS = list("abcdefg")
REAL = list("abfg")
BASELINE, CHAMPION, OUR_ZEROSHOT = 0.509, 0.382, 0.4795
TARGET = np.array([0.0, -1.0, 1.0])       # REST_ID->0, LEFT_ID->-1, RIGHT_ID->+1
EMA_ALPHA = 0.3                            # fixed moderate smoothing at eval time


def build_regressor(n_ch: int = 8, n_samp: int = CROP):
    from tensorflow import keras
    from tensorflow.keras import layers as L
    inp = keras.Input((n_ch, n_samp, 1))
    x = L.Conv2D(8, (1, 64), padding="same", use_bias=False)(inp)
    x = L.BatchNormalization()(x)
    x = L.DepthwiseConv2D((n_ch, 1), use_bias=False, depth_multiplier=2,
                          depthwise_constraint=keras.constraints.MaxNorm(1.0))(x)
    x = L.BatchNormalization()(x); x = L.Activation("elu")(x)
    x = L.AveragePooling2D((1, 4))(x); x = L.Dropout(0.5)(x)
    x = L.SeparableConv2D(16, (1, 16), padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x); x = L.Activation("elu")(x)
    x = L.AveragePooling2D((1, 8))(x); x = L.Dropout(0.5)(x)
    x = L.Flatten()(x)
    out = L.Dense(1, activation="tanh")(x)
    return keras.Model(inp, out)


def _ema(x, a):
    out = np.empty_like(x); acc = x[0]
    for i, v in enumerate(x):
        acc = a * v + (1 - a) * acc; out[i] = acc
    return out


def eval_mse(model, subj: str, alpha: float) -> float:
    cnt, y1000 = load_eval(subj)
    sig = preprocess_trial(cnt); M = sig.shape[1]
    ea = ea_io.fit_ea([sig]); d = OUT / subj; d.mkdir(parents=True, exist_ok=True)
    ea_io.save_ea(ea, d / "ea_eval.json"); Wm = ea_io.load_ea_matrix(d / "ea_eval.json")
    aligned = ea_io.apply_ea(Wm, sig)
    starts = list(range(0, M - CROP + 1, HOP))
    X = np.stack([aligned[:, s:s + CROP][:, :, None] for s in starts]).astype(np.float32)
    pred = model.predict(X, batch_size=512, verbose=0).ravel().astype(np.float64)
    pred = np.clip(_ema(pred, alpha), -1.0, 1.0)
    centres = np.clip(np.array([int(round((s + CROP / 2) * DS)) for s in starts]), 0, len(y1000) - 1)
    t = np.arange(len(y1000)); j = np.clip(np.searchsorted(centres, t), 1, len(centres) - 1)
    nearest = np.where((t - centres[j - 1]) <= (centres[j] - t), j - 1, j)
    keep = np.isfinite(y1000)
    return float(np.mean((pred[nearest][keep] - y1000[keep]) ** 2))


def train_subject(subj: str) -> Dict:
    from tensorflow import keras
    set_global_seed(C.RANDOM_SEED)
    X, ycls, classes, erd = build_calib_set(subj)          # EA'd calib crops + class ids
    y = TARGET[ycls]                                        # -> regression targets
    rng = np.random.default_rng(C.RANDOM_SEED)
    perm = rng.permutation(len(y)); ntr = int(0.85 * len(y))
    tr, va = perm[:ntr], perm[ntr:]
    model = build_regressor()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    rl = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)
    hist = model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=80,
                     batch_size=64, verbose=0, callbacks=[es, rl])
    mse_raw = eval_mse(model, subj, 1.0)
    mse_smooth = eval_mse(model, subj, EMA_ALPHA)
    keras.backend.clear_session()
    return {"subject": subj, "real": subj in REAL, "classes": classes, "erd": erd,
            "n_calib_crops": int(len(y)), "epochs": len(hist.history["loss"]),
            "mse_raw": mse_raw, "mse_smooth": mse_smooth}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "regressor.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h); t0 = time.perf_counter()
    try:
        results = []
        for s in SUBJECTS:
            r = train_subject(s)
            results.append(r)
            logger.info("ds1%s %-4s ep=%d MSE raw=%.4f smooth=%.4f (zero-shot was %.4f)",
                        s, "real" if r["real"] else "gen.", r["epochs"], r["mse_raw"],
                        r["mse_smooth"], OUR_ZEROSHOT)
        real = [r for r in results if r["real"]]

        def pooled(rows, key):   # each subject weighted equally (per-subject competition unit)
            return float(np.mean([r[key] for r in rows]))
        summary = {"baseline_do_nothing": BASELINE, "champion_2009": CHAMPION,
                   "our_zeroshot": OUR_ZEROSHOT, "ema_alpha": EMA_ALPHA,
                   "regressor_real_mean_raw": round(pooled(real, "mse_raw"), 4),
                   "regressor_real_mean_smooth": round(pooled(real, "mse_smooth"), 4),
                   "results": results}
        save_json(OUT / "summary.json", summary)
        _report(summary)
        logger.info("=== DONE in %.1fs === regressor real-mean MSE raw=%.4f smooth=%.4f "
                    "(zero-shot %.4f, champion %.3f, baseline %.3f)", time.perf_counter() - t0,
                    summary["regressor_real_mean_raw"], summary["regressor_real_mean_smooth"],
                    OUR_ZEROSHOT, CHAMPION, BASELINE)
    finally:
        logger.removeHandler(h); h.close()


def _report(s: Dict):
    lines = ["# BCI-IV-1 -- regressor trained on the dataset itself", "",
             "New strategy: instead of zero-shot from Stieger, a regressor (continuous output, MSE "
             "loss) is trained on each subject's `calib` and evaluated on its `eval`. "
             f"Lower = better. References: silence {s['baseline_do_nothing']}, champion "
             f"{s['champion_2009']}, zero-shot {s['our_zeroshot']}.", "",
             f"- Regressor, real mean (raw): {s['regressor_real_mean_raw']}",
             f"- Regressor, real mean (EMA-smoothed {s['ema_alpha']}): "
             f"{s['regressor_real_mean_smooth']}", "",
             "| Subj | type | classes | ep | raw MSE | smoothed MSE |", "|---|---|---|---|---|---|"]
    for r in s["results"]:
        lines.append(f"| ds1{r['subject']} | {'real' if r['real'] else 'gen.'} | "
                     f"{'/'.join(r['classes'])} | {r['epochs']} | {r['mse_raw']:.4f} | "
                     f"{r['mse_smooth']:.4f} |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
