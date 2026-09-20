r"""Training for the EEGNet-regression decoder.

Protocol: one model per subject, trained only on that subject's calibration
recording, with MSE loss, Adam, batch 64, EarlyStopping (patience 10) on a
calib-internal validation split, and ReduceLROnPlateau.

Two entry points:
  * train_from_scratch(subj)          -- competition-legal, calib only.
  * pretrain_stieger() + finetune(..) -- give the per-subject model a stronger
    prior by pretraining one regressor on Stieger (pooled), then gently
    fine-tuning it on each subject's calib. This is the best-performing variant.

The Stieger pretraining is the only part that depends on the wider project (it
reads the cached Stieger segment pools); everything else uses only this package.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from model import build_model, compile_model
from preprocessing import EuclideanAlignment, build_calib_crops, make_crops

BATCH, ES_PATIENCE = 64, 10
PRETRAIN_W = Path(__file__).with_name("pretrained.weights.h5")


def _callbacks(patience: int):
    from tensorflow import keras
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=max(3, patience // 2), min_lr=1e-6)]


def _fit(model, X, y, epochs, seed=42):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y)); ntr = int(0.85 * len(y))
    tr, va = perm[:ntr], perm[ntr:]
    hist = model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=epochs,
                     batch_size=BATCH, verbose=0, callbacks=_callbacks(ES_PATIENCE))
    return len(hist.history["loss"])


def train_from_scratch(subj: str):
    """Train a per-subject regressor from scratch on its calib. Returns (model, epochs)."""
    from tensorflow import keras
    keras.utils.set_random_seed(42)
    X, y, _ = build_calib_crops(subj)
    model = compile_model(build_model(), lr=1e-3)
    return model, _fit(model, X, y, epochs=80)


def finetune(subj: str, pretrained_weights: Path = PRETRAIN_W):
    """Fine-tune the Stieger-pretrained regressor on a subject's calib (gentle LR)."""
    from tensorflow import keras
    keras.utils.set_random_seed(42)
    X, y, _ = build_calib_crops(subj)
    model = build_model(); model.load_weights(str(pretrained_weights))
    compile_model(model, lr=2e-4)
    return model, _fit(model, X, y, epochs=60)


def pretrain_stieger(out_weights: Path = PRETRAIN_W):
    """Pretrain ONE regressor on pooled Stieger segments (targets left=-1/right=+1/rest=0).

    Reads the cached Stieger per-subject pools built by the second_ml package.
    """
    from tensorflow import keras
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from lib import config3 as C
    from lib.segments import load_or_build_subject_pool
    from src.split import load_split

    keras.utils.set_random_seed(42)
    split = load_split(C.SPLIT_JSON)
    train_subjects = sorted(split["splits"]["train"]["subjects"], key=lambda s: int(s[1:]))
    sig_tgt = []
    for subj in train_subjects:
        pool = load_or_build_subject_pool(subj, split)
        if not (pool.mi_left and pool.mi_right and pool.rest):
            continue
        sig_tgt += [(np.asarray(a, np.float32), -1.0) for a in pool.mi_left]
        sig_tgt += [(np.asarray(a, np.float32), 1.0) for a in pool.mi_right]
        sig_tgt += [(np.asarray(a, np.float32), 0.0) for a in pool.rest]
    print(f"[pretrain] {len(sig_tgt)} Stieger segments")
    ea = EuclideanAlignment().fit([s for s, _ in sig_tgt])       # global EA on Stieger
    xs, ys = [], []
    for sig, t in sig_tgt:
        c = make_crops(ea.transform(sig))
        xs.append(c); ys.append(np.full(len(c), t, np.float32))
    X = np.concatenate(xs); y = np.concatenate(ys)
    print(f"[pretrain] {len(y)} crops")
    model = compile_model(build_model(), lr=1e-3)
    rng = np.random.default_rng(42); perm = rng.permutation(len(y)); ntr = int(0.9 * len(y))
    model.fit(X[perm[:ntr]], y[perm[:ntr]], validation_data=(X[perm[ntr:]], y[perm[ntr:]]),
              epochs=60, batch_size=256, verbose=0, callbacks=_callbacks(8))
    model.save_weights(str(out_weights))
    keras.backend.clear_session()
    return out_weights
