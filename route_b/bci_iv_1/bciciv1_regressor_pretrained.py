r"""Point 2: pretrain the regressor on Stieger, then fine-tune per BCI-IV-1 subject.

The from-scratch per-subject regressor (bciciv1_regressor.py) reached real-mean
MSE 0.4332, beating the champion on the good subjects (g,a) but not on the weak
ones (b,f), which overfit ~200 calib trials. Here it is given a stronger prior:
pretrain one regressor on Stieger (pooled train subjects, MSE loss, targets
left=-1/right=+1/rest=0), then fine-tune it gently on each BCI-IV-1 subject's
calib. Same leakage discipline: train on calib, test on eval; EA unsupervised;
fixed EMA.

Stieger segments come from the cached per-subject pools (segment_cache), so no
slow raw load. Reuses build_regressor / build_calib_set / eval_mse from
bciciv1_regressor.py.

Usage:  python bciciv1_regressor_pretrained.py   (run from route_b/bci_iv_1/, BCI_DATA set)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import config3 as C                          # noqa: E402
from lib import ea_io                                   # noqa: E402
from lib.segments import load_or_build_subject_pool    # noqa: E402
from src.split import load_split, session_keys_for     # noqa: E402
from src.utils import get_logger, save_json, set_global_seed  # noqa: E402
from bciciv1_regressor import (build_regressor, build_calib_set, eval_mse,   # noqa: E402
                               TARGET, EMA_ALPHA, SUBJECTS, REAL,
                               BASELINE, CHAMPION, OUR_ZEROSHOT)

logger = get_logger()
OUT = Path(__file__).resolve().parents[1] / "experiments" / "bciciv1_regressor_pretrained"
PRET_W = OUT / "pretrained.weights.h5"
FROM_SCRATCH_REF = 0.4332          # per-subject from-scratch regressor (smoothed)


def pretrain_on_stieger():
    from tensorflow import keras
    set_global_seed(C.RANDOM_SEED)
    split = load_split(C.SPLIT_JSON)
    train_subjects = sorted(split["splits"]["train"]["subjects"], key=lambda s: int(s[1:]))
    logger.info("Loading cached Stieger pools for %d train subjects...", len(train_subjects))
    # pool.{mi_left,mi_right,rest} are Lists[np.ndarray] (raw signals) -> (signal, target) pairs
    sig_tgt = []
    for subj in train_subjects:
        pool = load_or_build_subject_pool(subj, split)
        if not (pool.mi_left and pool.mi_right and pool.rest):
            continue
        sig_tgt += [(np.asarray(a, np.float32), -1.0) for a in pool.mi_left]   # LEFT -> -1
        sig_tgt += [(np.asarray(a, np.float32), 1.0) for a in pool.mi_right]   # RIGHT -> +1
        sig_tgt += [(np.asarray(a, np.float32), 0.0) for a in pool.rest]       # REST -> 0
    logger.info("collected %d Stieger segments", len(sig_tgt))
    ea = ea_io.fit_ea([s for s, _ in sig_tgt])
    OUT.mkdir(parents=True, exist_ok=True)
    ea_io.save_ea(ea, OUT / "ea_pretrain.json"); W = ea_io.load_ea_matrix(OUT / "ea_pretrain.json")
    CR, HP = C.CROP_SAMPLES, C.CROP_STRIDE_SAMPLES
    xs, ys = [], []
    for sig, tgt in sig_tgt:
        al = ea_io.apply_ea(W, sig)
        for s in range(0, al.shape[1] - CR + 1, HP):
            xs.append(al[:, s:s + CR][:, :, None]); ys.append(tgt)
    X = np.stack(xs).astype(np.float32); y = np.asarray(ys, np.float32)
    logger.info("built %d pretrain crops", len(y))
    rng = np.random.default_rng(C.RANDOM_SEED)
    perm = rng.permutation(len(y)); ntr = int(0.9 * len(y)); tr, va = perm[:ntr], perm[ntr:]
    model = build_regressor()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    rl = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)
    logger.info("pretraining regressor on %d crops...", len(y))
    model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=60, batch_size=256,
              verbose=0, callbacks=[es, rl])
    model.save_weights(str(PRET_W))
    logger.info("pretrained weights saved -> %s", PRET_W)
    keras.backend.clear_session()


def finetune_subject(subj: str) -> Dict:
    from tensorflow import keras
    set_global_seed(C.RANDOM_SEED)
    X, ycls, classes, erd = build_calib_set(subj)
    y = TARGET[ycls]
    rng = np.random.default_rng(C.RANDOM_SEED)
    perm = rng.permutation(len(y)); ntr = int(0.85 * len(y)); tr, va = perm[:ntr], perm[ntr:]
    model = build_regressor(); model.load_weights(str(PRET_W))       # start from Stieger prior
    model.compile(optimizer=keras.optimizers.Adam(2e-4), loss="mse")  # gentle fine-tune
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    rl = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
    hist = model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=60,
                     batch_size=64, verbose=0, callbacks=[es, rl])
    mse_raw = eval_mse(model, subj, 1.0)
    mse_smooth = eval_mse(model, subj, EMA_ALPHA)
    keras.backend.clear_session()
    return {"subject": subj, "real": subj in REAL, "classes": classes, "erd": erd,
            "epochs": len(hist.history["loss"]), "mse_raw": mse_raw, "mse_smooth": mse_smooth}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(OUT / "regressor_pretrained.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h); t0 = time.perf_counter()
    try:
        from tensorflow import keras  # noqa: F401
        pretrain_on_stieger()
        results = []
        for s in SUBJECTS:
            r = finetune_subject(s)
            results.append(r)
            logger.info("ds1%s %-4s ep=%d MSE raw=%.4f smooth=%.4f (scratch was %.4f-mean)",
                        s, "real" if r["real"] else "gen.", r["epochs"], r["mse_raw"],
                        r["mse_smooth"], FROM_SCRATCH_REF)
        real = [r for r in results if r["real"]]
        mean_raw = float(np.mean([r["mse_raw"] for r in real]))
        mean_smooth = float(np.mean([r["mse_smooth"] for r in real]))
        summary = {"baseline": BASELINE, "champion": CHAMPION, "our_zeroshot": OUR_ZEROSHOT,
                   "from_scratch_ref": FROM_SCRATCH_REF, "ema_alpha": EMA_ALPHA,
                   "pretrained_real_mean_raw": round(mean_raw, 4),
                   "pretrained_real_mean_smooth": round(mean_smooth, 4), "results": results}
        save_json(OUT / "summary.json", summary)
        _report(summary)
        logger.info("=== DONE in %.1fs === pretrained+finetune real-mean MSE smooth=%.4f "
                    "(from-scratch %.4f, champion %.3f, baseline %.3f)", time.perf_counter() - t0,
                    mean_smooth, FROM_SCRATCH_REF, CHAMPION, BASELINE)
    finally:
        logger.removeHandler(h); h.close()


def _report(s: Dict):
    lines = ["# BCI-IV-1 -- regressor pretrain(Stieger) + per-subject fine-tune", "",
             "The regressor is given a stronger prior (pretrained on Stieger) and fine-tuned per "
             f"subject on its calib. Lower = better. References: baseline {s['baseline']}, champion "
             f"{s['champion']}, zero-shot {s['our_zeroshot']}, from-scratch regressor {s['from_scratch_ref']}.", "",
             f"- Pretrain+fine-tune, real mean (smoothed): {s['pretrained_real_mean_smooth']} "
             f"(from-scratch was {s['from_scratch_ref']})", "",
             "| Subj | type | ep | raw MSE | smoothed MSE |", "|---|---|---|---|---|"]
    for r in s["results"]:
        lines.append(f"| ds1{r['subject']} | {'real' if r['real'] else 'gen.'} | {r['epochs']} | "
                     f"{r['mse_raw']:.4f} | {r['mse_smooth']:.4f} |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
