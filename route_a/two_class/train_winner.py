#!/usr/bin/env python
"""Retrain the best grid cell (W = 0.5 s, F = 16, CWT downsampling on) and save it.

`run_grid.py` evaluates every cell and discards the model. This script runs the same
training loop for the winning cell (same subject split, seed, optimizer, AMP, batch,
epochs and early stopping on validation trial accuracy) and keeps the best weights
together with the normalization statistics the model was trained with. The saved
checkpoint is what `transfer_2a.py --protocol P3` applies to BCI-IV-2a.

The checkpoint shipped in checkpoints/ was produced this way (best validation trial
accuracy 0.6991 at epoch 24, about 3 minutes on an RTX 4060).

Usage:
    python train_winner.py
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from run_grid import (DualBranch, build_morlet_bank, cwt_power, make_windows,
                      gather_windows, load_data,
                      FS, F_MIN, F_MAX, OVERLAP, DS_FACTOR,
                      CLIP_SIGMA, SEED, BATCH, LR, WD, MAX_EPOCHS, PATIENCE)

HERE = Path(__file__).resolve().parent
CKPT = HERE / 'checkpoints'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_normalized():
    """Subject split and raw z-score with training statistics, clipped (as in run_grid)."""
    X, y, subj, train_subs, val_subs = load_data()
    subj = subj.astype(int)
    tr_mask = np.isin(subj, list(train_subs)); va_mask = np.isin(subj, list(val_subs))
    mu = X[tr_mask].mean(axis=(0, 2), keepdims=True)
    sd = X[tr_mask].std(axis=(0, 2), keepdims=True) + 1e-6
    Xn = np.clip((X - mu) / sd, -CLIP_SIGMA, CLIP_SIGMA).astype(np.float32)
    Xseg = torch.tensor(Xn, device=device)
    y_t = torch.tensor(y, device=device)
    return Xseg, y_t, y, np.where(tr_mask)[0], np.where(va_mask)[0]


def train_config(Xseg, y_t, y, tr_rows, va_rows, w_sec, n_freqs, downsample,
                 max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=SEED, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    w_samp = int(round(w_sec * FS))
    stride = max(1, int(round(w_samp * (1 - OVERLAP))))
    ds = DS_FACTOR if downsample else 1

    tr_row, tr_start, n_win = make_windows(tr_rows, w_samp, stride)
    va_row, va_start, _     = make_windows(va_rows, w_samp, stride)
    tr_row = torch.tensor(tr_row, device=device); tr_start = torch.tensor(tr_start, device=device)
    va_row = torch.tensor(va_row, device=device); va_start = torch.tensor(va_start, device=device)

    freqs = np.logspace(np.log10(F_MIN), np.log10(F_MAX), n_freqs)
    kr, ki, pad = build_morlet_bank(freqs, FS, w_samp)
    with torch.no_grad():
        n_s = min(4096, len(tr_row)); sel = torch.randperm(len(tr_row), device=device)[:n_s]
        cs = cwt_power(gather_windows(Xseg, tr_row[sel], tr_start[sel], w_samp), kr, ki, pad, ds)
        cwt_mu = cs.mean(dim=(0,1,3), keepdim=True); cwt_sd = cs.std(dim=(0,1,3), keepdim=True) + 1e-5
    def make_cwt(xw): return (cwt_power(xw, kr, ki, pad, ds) - cwt_mu) / cwt_sd

    model = DualBranch().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))
    crit = nn.CrossEntropyLoss()
    n_tr = len(tr_row); best=-1; best_state=None; best_ep=-1; since=0; hist=[]
    t0 = time.time()
    for ep in range(1, max_epochs+1):
        model.train(); perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, BATCH):
            b = perm[i:i+BATCH]
            xw = gather_windows(Xseg, tr_row[b], tr_start[b], w_samp); yb = y_t[tr_row[b]]
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                loss = crit(model(xw, make_cwt(xw)), yb)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        # validation, trial level (mean softmax over the crops of each trial)
        model.eval(); wc=wt=0; prob={}
        with torch.no_grad():
            for i in range(0, len(va_row), 512):
                rb = va_row[i:i+512]; xw = gather_windows(Xseg, rb, va_start[i:i+512], w_samp)
                with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                    p = torch.softmax(model(xw, make_cwt(xw)), 1)
                yb = y_t[rb]; wc += (p.argmax(1)==yb).sum().item(); wt += len(rb)
                for r, pr in zip(rb.cpu().numpy(), p.float().cpu().numpy()): prob[r] = prob.get(r,0)+pr
        rows_u = np.array(sorted(prob)); pred = np.array([prob[r].argmax() for r in rows_u])
        ta = float((pred == y[rows_u]).mean()); wa = wc/max(1,wt)
        hist.append({'epoch':ep, 'val_trial_acc':ta, 'val_win_acc':wa})
        if verbose: print(f'  ep {ep:2d}  val trial-acc {ta:.4f}   win-acc {wa:.4f}')
        if ta > best:
            best=ta; best_ep=ep; since=0
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            since += 1
        if since >= patience: break
    model.load_state_dict(best_state)
    dt = time.time()-t0
    bundle = dict(w_sec=w_sec, n_freqs=n_freqs, downsample=bool(downsample), w_samp=w_samp,
                  stride=stride, ds=ds, freqs=freqs, pad=int(pad),
                  cwt_mu=cwt_mu.detach().cpu(), cwt_sd=cwt_sd.detach().cpu(),
                  best_epoch=best_ep, best_val_trial_acc=best, history=hist, train_time_s=dt)
    print(f'  -> best val trial-acc {best:.4f} @ epoch {best_ep}  ({dt:.0f}s)')
    return model, bundle


def save_ckpt(model, bundle, fname):
    CKPT.mkdir(exist_ok=True)
    torch.save({'state_dict': model.state_dict(), **{k:bundle[k] for k in
               ['w_sec','n_freqs','downsample','w_samp','stride','ds','freqs','pad',
                'cwt_mu','cwt_sd','best_epoch','best_val_trial_acc','history']}}, CKPT/fname)
    print('   checkpoint saved ->', CKPT/fname)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    Xseg, y_t, y, tr_rows, va_rows = load_normalized()
    print('Training W=0.5 s  F=16  downsampling on ...')
    model, bundle = train_config(Xseg, y_t, y, tr_rows, va_rows, 0.5, 16, True)
    save_ckpt(model, bundle, 'winner_W0.5_F16_dsON.pt')


if __name__ == '__main__':
    main()
