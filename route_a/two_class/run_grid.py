#!/usr/bin/env python
r"""Grid search for the dual-branch decoder (raw 1D + CWT 2D) with cropped training.

Evaluates 36 cells:
  - crop length W:                    0.5, 1.0, 1.5, 2.0, 2.5, 3.0 s
  - number of CWT frequencies:        16, 24, 32
  - temporal downsampling of the CWT: off / on

Model (DualBranch):
  - raw branch: 4 Conv1D layers on the EEG crop
  - CWT branch: 2 Conv2D layers on the time-frequency scalogram of the same crop
  - concatenation -> dense layer -> 2 logits (left vs right hand)

Input: cache/trials_cyton8_lhrh_v2_W3.npz, built by build_cache.py: 8 channels after a
common average reference over the 62 EEG channels, no band-pass, baseline corrected, 3 s
of motor imagery (750 samples at 250 Hz). The cache keeps only trials that lasted at least
3 s; this subset is the same for every cell.

Split: by subject, from ../split_80_20_subjects.json (50 training / 12 validation subjects).
Crops are cut after the split, so no crop of a validation subject is seen in training.

Preprocessing, per cell:
  - raw: per-channel z-score with training statistics, clipped to +-8 sigma
  - CWT: complex Morlet wavelets (4-40 Hz, log-spaced, 6 cycles) on the same crop ->
    power -> log1p -> standardized per frequency with training statistics

Evaluation on the validation subjects: window accuracy, trial accuracy (softmax summed over
the crops of each trial) and Cohen's kappa at trial level. Training stops after 6 epochs
without improvement of the validation trial accuracy (at most 30 epochs); each cell reports
the metrics of its best epoch.

Output (outputs/):
  - results.csv      one row per cell, appended as each cell finishes
  - results.md       table sorted by validation trial accuracy
  With --smoke: results_smoke.csv and results_smoke.md.

Usage:
  python run_grid.py --smoke     # quick check: 6 + 4 subjects, 4 cells, 2 epochs
  python run_grid.py             # full grid
"""
import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- paths / config
HERE = Path(__file__).resolve().parent
NPZ = HERE / 'cache' / 'trials_cyton8_lhrh_v2_W3.npz'
SPLIT_JSON = HERE.parent / 'split_80_20_subjects.json'
OUTDIR = HERE / 'outputs'

FS = 250
SEG = 750                                   # samples per MI segment (3 s)

# --- grid ---
WINDOWS_SEC = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
N_FREQS_GRID = [16, 24, 32]
DOWNSAMPLE_GRID = [False, True]
F_MIN, F_MAX = 4.0, 40.0                     # CWT band (Hz)
N_CYCLES = 6.0                              # Morlet wavelet width, in cycles
OVERLAP = 0.5                              # overlap between consecutive crops
DS_FACTOR = 8                              # temporal downsampling factor of the CWT (when on)

# --- training ---
MAX_EPOCHS = 30
PATIENCE = 6
BATCH = 128
LR = 1e-3
WD = 1e-4
DROPOUT = 0.5
CLIP_SIGMA = 8.0
SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log(*a):
    print(*a, flush=True)


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- CWT (Morlet, on device)
def build_morlet_bank(freqs, fs, w_samp, n_cycles=N_CYCLES):
    """Bank of complex Morlet wavelets used as conv1d kernels.

    Returns (kernel_real, kernel_imag), each of shape (n_freqs, 1, L), and the padding L//2.
    Each wavelet is truncated at +-3 sigma or at the crop length (w_samp), whichever is
    shorter, and normalized to unit energy.
    """
    kernels_r, kernels_i, max_len = [], [], 0
    tmp = []
    for f in freqs:
        sigma_t = n_cycles / (2 * np.pi * f)          # width in seconds
        half = int(min(3 * sigma_t * fs, (w_samp - 1) / 2))
        half = max(half, 1)
        tw = np.arange(-half, half + 1) / fs          # time axis of the wavelet
        gauss = np.exp(-tw ** 2 / (2 * sigma_t ** 2))
        wav = np.exp(2j * np.pi * f * tw) * gauss
        wav /= np.sqrt(np.sum(np.abs(wav) ** 2))       # unit energy
        tmp.append(wav)
        max_len = max(max_len, len(wav))
    if max_len % 2 == 0:
        max_len += 1
    for wav in tmp:                                    # zero-pad to a common length (centered)
        pad = (max_len - len(wav)) // 2
        w = np.pad(wav, (pad, max_len - len(wav) - pad))
        kernels_r.append(w.real.astype(np.float32))
        kernels_i.append(w.imag.astype(np.float32))
    kr = torch.tensor(np.stack(kernels_r)[:, None, :], device=device)
    ki = torch.tensor(np.stack(kernels_i)[:, None, :], device=device)
    return kr, ki, max_len // 2


def cwt_power(x, kr, ki, pad, ds_factor):
    """CWT of a batch of crops.

    x: (B, C, T) -> (B, C, F, T'), log1p of the power (not yet standardized).
    T' = T, or T // ds_factor when downsampling.
    """
    B, C, T = x.shape
    xf = x.reshape(B * C, 1, T)
    re = F.conv1d(xf, kr, padding=pad)                 # (B*C, F, T)
    im = F.conv1d(xf, ki, padding=pad)
    power = re ** 2 + im ** 2
    F_ = power.shape[1]
    power = power.reshape(B, C, F_, power.shape[-1])
    if ds_factor and ds_factor > 1:
        power = F.avg_pool2d(power, kernel_size=(1, ds_factor))   # time axis only
    return torch.log1p(power)


# ---------------------------------------------------------------- dual-branch model
class DualBranch(nn.Module):
    def __init__(self, n_ch=8, dropout=DROPOUT):
        super().__init__()
        # raw branch: 4 x Conv1D
        self.raw = nn.Sequential(
            nn.Conv1d(n_ch, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        # CWT branch: 2 x Conv2D
        self.cwt = nn.Sequential(
            nn.Conv2d(n_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(64 + 32, 64), nn.ReLU(), nn.Linear(64, 2),
        )

    def forward(self, raw, cwt):
        a = self.raw(raw).flatten(1)
        b = self.cwt(cwt).flatten(1)
        return self.head(torch.cat([a, b], dim=1))


# ---------------------------------------------------------------- data / cropping
def load_data():
    z = np.load(NPZ, allow_pickle=True)
    X = z['X'].astype(np.float32)               # (N, 8, 750)
    y = z['y'].astype(np.int64)
    subj = z['subject'].astype(int)
    sp = json.load(open(SPLIT_JSON, encoding='utf-8'))
    train_subs = set(int(s[1:]) for s in sp['train_subjects'])
    val_subs = set(int(s[1:]) for s in sp['val_subjects'])
    return X, y, subj, train_subs, val_subs


def make_windows(rows, w_samp, stride):
    """(row, start) of every crop of every trial in `rows`, plus the number of crops per trial."""
    starts = list(range(0, SEG - w_samp + 1, stride))
    idx_row, idx_start = [], []
    for r in rows:
        for s in starts:
            idx_row.append(r)
            idx_start.append(s)
    return (np.asarray(idx_row, np.int64), np.asarray(idx_start, np.int64), len(starts))


def gather_windows(Xseg, rows, starts, w_samp):
    """Cut crops (B, 8, w_samp) from Xseg (N, 8, 750) on the device, one start per crop."""
    base = Xseg[rows]                                  # (B, 8, SEG)
    t_idx = starts[:, None, None] + torch.arange(w_samp, device=Xseg.device)[None, None, :]
    t_idx = t_idx.expand(-1, base.shape[1], -1)        # (B, 8, w_samp)
    return torch.gather(base, 2, t_idx)


# ---------------------------------------------------------------- one grid cell
def run_config(Xseg, y_t, subj_t, tr_rows, va_rows, w_sec, n_freqs, downsample,
               max_epochs, patience):
    set_seed(SEED)
    w_samp = int(round(w_sec * FS))
    stride = max(1, int(round(w_samp * (1 - OVERLAP))))
    ds = DS_FACTOR if downsample else 1

    # crop indices
    tr_row, tr_start, n_win_trial = make_windows(tr_rows, w_samp, stride)
    va_row, va_start, _ = make_windows(va_rows, w_samp, stride)
    tr_row = torch.tensor(tr_row, device=device)
    tr_start = torch.tensor(tr_start, device=device)
    va_row = torch.tensor(va_row, device=device)
    va_start = torch.tensor(va_start, device=device)

    # Morlet bank on log-spaced frequencies
    freqs = np.logspace(np.log10(F_MIN), np.log10(F_MAX), n_freqs)
    kr, ki, pad = build_morlet_bank(freqs, FS, w_samp)

    # CWT standardization: per-frequency statistics from a random sample of training crops
    with torch.no_grad():
        n_s = min(4096, len(tr_row))
        sel = torch.randperm(len(tr_row), device=device)[:n_s]
        xs = gather_windows(Xseg, tr_row[sel], tr_start[sel], w_samp)
        cs = cwt_power(xs, kr, ki, pad, ds)            # (n_s, 8, F, T')
        cwt_mu = cs.mean(dim=(0, 1, 3), keepdim=True)  # per frequency
        cwt_sd = cs.std(dim=(0, 1, 3), keepdim=True) + 1e-5

    def make_cwt(xw):
        c = cwt_power(xw, kr, ki, pad, ds)
        return (c - cwt_mu) / cwt_sd

    model = DualBranch().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    crit = nn.CrossEntropyLoss()

    n_tr = len(tr_row)
    best_trial_acc, best_state, best_ep, since = -1.0, None, -1, 0
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, BATCH):
            b = perm[i:i + BATCH]
            xw = gather_windows(Xseg, tr_row[b], tr_start[b], w_samp)
            yb = y_t[tr_row[b]]
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(xw, make_cwt(xw))
                loss = crit(out, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # --- validation (window and trial level) ---
        model.eval()
        win_correct, win_tot = 0, 0
        prob_sum = {}                                  # row -> summed softmax [2]
        with torch.no_grad():
            for i in range(0, len(va_row), 512):
                rows_b = va_row[i:i + 512]
                xw = gather_windows(Xseg, rows_b, va_start[i:i + 512], w_samp)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    p = torch.softmax(model(xw, make_cwt(xw)), dim=1)
                yb = y_t[rows_b]
                win_correct += (p.argmax(1) == yb).sum().item()
                win_tot += len(rows_b)
                rows_np = rows_b.cpu().numpy()
                p_np = p.float().cpu().numpy()
                for r, pr in zip(rows_np, p_np):
                    prob_sum[r] = prob_sum.get(r, 0) + pr
        # trial level
        rows_u = np.array(sorted(prob_sum))
        pred = np.array([prob_sum[r].argmax() for r in rows_u])
        true = y_t.cpu().numpy()[rows_u]
        trial_acc = float((pred == true).mean())
        win_acc = win_correct / max(1, win_tot)
        # Cohen's kappa
        po = trial_acc
        p1t = (true == 1).mean(); p1p = (pred == 1).mean()
        pe = p1t * p1p + (1 - p1t) * (1 - p1p)
        kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else 0.0

        if trial_acc > best_trial_acc:
            best_trial_acc = trial_acc
            best_state = {'win_acc': win_acc, 'trial_acc': trial_acc, 'kappa': kappa}
            best_ep = ep
            since = 0
        else:
            since += 1
        if since >= patience:
            break

    dt = time.time() - t0
    del model, kr, ki
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {
        'window_sec': w_sec, 'n_freqs': n_freqs, 'cwt_downsample': int(downsample),
        'w_samp': w_samp, 'stride': stride, 'wins_per_trial': n_win_trial,
        'n_train_windows': int(n_tr), 'n_val_windows': int(len(va_row)),
        'val_win_acc': round(best_state['win_acc'], 4),
        'val_trial_acc': round(best_state['trial_acc'], 4),
        'val_kappa': round(best_state['kappa'], 4),
        'best_epoch': best_ep, 'train_time_s': round(dt, 1),
    }


# ---------------------------------------------------------------- main / grid
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    windows = WINDOWS_SEC
    nfreqs = N_FREQS_GRID
    downs = DOWNSAMPLE_GRID
    max_epochs, patience = MAX_EPOCHS, PATIENCE

    X, y, subj, train_subs, val_subs = load_data()
    if args.smoke:
        train_subs = set(list(sorted(train_subs))[:6])
        val_subs = set(list(sorted(val_subs))[:4])
        windows, nfreqs, downs = [1.0, 2.0], [16], [False, True]
        max_epochs, patience = 2, 2

    # raw z-score per channel with training statistics (+ clip), over the whole segment
    tr_mask = np.isin(subj, list(train_subs))
    va_mask = np.isin(subj, list(val_subs))
    mu = X[tr_mask].mean(axis=(0, 2), keepdims=True)
    sd = X[tr_mask].std(axis=(0, 2), keepdims=True) + 1e-6
    Xn = np.clip((X - mu) / sd, -CLIP_SIGMA, CLIP_SIGMA).astype(np.float32)

    Xseg = torch.tensor(Xn, device=device)             # (N, 8, 750) on the device
    y_t = torch.tensor(y, device=device)
    subj_t = torch.tensor(subj, device=device)
    tr_rows = np.where(tr_mask)[0]
    va_rows = np.where(va_mask)[0]

    configs = [(w, nf, d) for w in windows for nf in nfreqs for d in downs]
    log('#' * 72)
    log(f'# dual-branch grid | device={device} | configs={len(configs)}')
    log(f'# train {len(tr_rows)} trials ({len(train_subs)} subj) / '
        f'val {len(va_rows)} trials ({len(val_subs)} subj)')
    log(f'# windows={windows} nfreqs={nfreqs} downsample={downs} '
        f'epochs<= {max_epochs} patience={patience}')
    log('#' * 72)

    csv_path = OUTDIR / ('results_smoke.csv' if args.smoke else 'results.csv')
    fields = ['window_sec', 'n_freqs', 'cwt_downsample', 'w_samp', 'stride',
              'wins_per_trial', 'n_train_windows', 'n_val_windows',
              'val_win_acc', 'val_trial_acc', 'val_kappa', 'best_epoch', 'train_time_s']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    rows = []
    for k, (w, nf, d) in enumerate(configs, 1):
        log(f'\n[{k}/{len(configs)}] W={w}s nfreqs={nf} downsample={d} ...')
        r = run_config(Xseg, y_t, subj_t, tr_rows, va_rows, w, nf, d, max_epochs, patience)
        rows.append(r)
        log(f'    -> trial_acc={r["val_trial_acc"]}  win_acc={r["val_win_acc"]}  '
            f'kappa={r["val_kappa"]}  ep*={r["best_epoch"]}  ({r["train_time_s"]}s)  '
            f'[{r["n_train_windows"]} win train]')
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fields).writerow(r)

    # ---- markdown summary ----
    rows_sorted = sorted(rows, key=lambda r: r['val_trial_acc'], reverse=True)
    md = OUTDIR / ('results_smoke.md' if args.smoke else 'results.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('# Dual-branch decoder (raw 1D + CWT 2D): grid results\n\n')
        f.write(f'- Data: `trials_cyton8_lhrh_v2_W3.npz` (LR task, CAR, no band-pass, '
                f'3 s of MI). Split by subject ({len(train_subs)} train / {len(val_subs)} val, '
                f'no subject overlap).\n')
        f.write(f'- Model: 4 x Conv1D (raw) + 2 x Conv2D (CWT), concat -> dense -> softmax.\n')
        f.write(f'- CWT: Morlet {F_MIN:g}-{F_MAX:g} Hz, {N_CYCLES:g} cycles, '
                f'overlap {OVERLAP:.0%}, downsampling factor {DS_FACTOR} when on.\n')
        f.write(f'- Main metric: val trial-acc (softmax summed over the crops of each '
                f'trial). Sorted from best to worst.\n\n')
        best = rows_sorted[0]
        f.write(f'## Best cell\n\n')
        f.write(f'W={best["window_sec"]}s, nfreqs={best["n_freqs"]}, '
                f'downsample={"ON" if best["cwt_downsample"] else "OFF"} -> '
                f'trial-acc {best["val_trial_acc"]}, win-acc {best["val_win_acc"]}, '
                f'kappa {best["val_kappa"]}.\n\n')
        f.write('## Full table\n\n')
        f.write('| W (s) | nfreqs | downsample | win/trial | trial-acc | win-acc | kappa | '
                'ep* | win train | t (s) |\n')
        f.write('|--:|--:|:--:|--:|--:|--:|--:|--:|--:|--:|\n')
        for r in rows_sorted:
            f.write(f'| {r["window_sec"]} | {r["n_freqs"]} | '
                    f'{"ON" if r["cwt_downsample"] else "OFF"} | {r["wins_per_trial"]} | '
                    f'{r["val_trial_acc"]} | {r["val_win_acc"]} | {r["val_kappa"]} | '
                    f'{r["best_epoch"]} | {r["n_train_windows"]} | {r["train_time_s"]} |\n')
        f.write('\n## Caveats\n\n')
        f.write('- The cache keeps only trials with at least 3 s of MI. The subset is the '
                'same for every cell, so it does not confound the comparison of crop '
                'lengths.\n')
        f.write('- Every cell crops the same fixed 3 s segment; a shorter W gives more crops '
                'of that segment. Crop lengths are compared on the same data.\n')
        f.write('- No independent test set: the validation subjects are used to choose the '
                'cell. confirm_test.py evaluates the chosen cell on held-out subjects.\n')
    log(f'\nSaved {csv_path.name} and {md.name}.')


if __name__ == '__main__':
    main()
