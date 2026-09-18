"""Training utilities (library functions; this module does not train anything by itself).

train_loop keeps the data on the CPU and moves batches to the GPU, halves the batch size
on CUDA out-of-memory errors, and uses AdamW with a cosine learning-rate schedule and
gradient-norm clipping. It accepts optional class weights (the IDLE class can unbalance
the data) and an optional augmenter callable (x, y) -> (x, y) applied to each batch.
"""
from __future__ import annotations
import time
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

from . import config as C
from .config import device, log, gpu_mem


def _loader(X, y, batch, shuffle):
    ds = TensorDataset(torch.from_numpy(X).float().unsqueeze(1),
                       torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0,
                      pin_memory=(device.type == 'cuda'))


@torch.no_grad()
def predict_batched(model, X, batch=256):
    """Predicted class indices for X (n, n_ch, w), computed in batches."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().unsqueeze(1).to(device)
        out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def class_weights(y: np.ndarray) -> torch.Tensor:
    """Weights inversely proportional to class frequency: n / (n_classes * count)."""
    counts = np.bincount(y, minlength=C.N_CLASES).astype(float)
    w = counts.sum() / (C.N_CLASES * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_loop(model, Xtr, ytr, epochs, lr, batch, val=None, tag='',
               weights: torch.Tensor | None = None, augmenter=None):
    """Train `model`, halving the batch size on out-of-memory errors.

    val: optional (X, y) evaluated after every epoch. Returns the best validation accuracy
    (0.0 when val is None).
    """
    while True:
        try:
            opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=C.WD)
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            dl = _loader(Xtr, ytr, batch, shuffle=True)
            best, t0 = 0.0, time.time()
            for ep in range(epochs):
                model.train(); run = 0.0
                for xb, yb in dl:
                    xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                    if augmenter is not None:
                        xa, yb = augmenter(xb.squeeze(1), yb); xb = xa.unsqueeze(1)
                    opt.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); run += loss.item() * xb.size(0)
                sched.step()
                if val is not None:
                    acc = accuracy_score(val[1], predict_batched(model, val[0]))
                    best = max(best, acc)
                if tag and ((ep + 1) % 10 == 0 or ep == 0):
                    msg = f'    {tag} ep{ep+1:3d}/{epochs} loss={run/len(Xtr):.4f}'
                    if val is not None:
                        msg += f' val_acc={acc:.3f} best={best:.3f}'
                    log(msg + f'  ({time.time()-t0:.0f}s, {gpu_mem()})')
            return best
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and batch > 8:
                torch.cuda.empty_cache(); gc.collect(); batch //= 2
                log(f'    [OOM] reducing batch -> {batch}'); continue
            raise


def make_val_split(X, y, frac=0.1, seed=C.SEED):
    """Random (not stratified) holdout: returns (train, val), with a fraction `frac` in val."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    nv = int(frac * len(idx))
    return (X[idx[nv:]], y[idx[nv:]]), (X[idx[:nv]], y[idx[:nv]])
