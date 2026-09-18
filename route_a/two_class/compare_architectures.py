#!/usr/bin/env python
r"""EEGSym vs the dual-branch decoder under the same protocol, with transfer to BCI-IV-2a.

Both architectures share everything except the model: training pool (all Stieger trials in
the cache), per-trial z-score clipped to +-8 sigma, overlapping crops, Adam (learning rate
1e-3, weight decay 1e-4), early stopping on a stratified 10% inner validation split, and
trial-level evaluation (softmax summed over the crops of a trial). No data augmentation.

Arms: {dualbranch, eegsym} x {W = 0.5 s, W = 3.0 s}. Each arm is trained on the Stieger pool
and evaluated on the 9 BCI-IV-2a subjects, on session E: zero-shot, and after fine-tuning on
the subject's session T (learning rate 1e-4, 20% of T held out for early stopping). One seed
per run (--seed); analyze_architectures.py aggregates the seeds found in outputs/.

EEGSym: the port in eegsym.py with filters_per_branch = 16 by default (--filters; the published
value is 24), dropout 0.4, channels reordered to [left lateral, midline, right lateral].
Dual-branch: 16 CWT frequencies, CWT downsampling on.

Input: cache/trials_cyton8_lhrh_v2_W3.npz (build_cache.py) and cache/cache_2a.npz (built by
transfer_2a.py, on first use if missing).
Output: outputs/results_arch_seed<seed>.csv. The --sanity mode only prints.

Usage:
  python compare_architectures.py --sanity --arch eegsym --window 3.0   # sanity check in Stieger
  python compare_architectures.py --sanity --arch eegsym --window 0.5
  python compare_architectures.py --seed 42                             # 2 x 2 arms, eval on 2a
"""
import os, sys, csv, json, time, copy, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
from run_grid import (DualBranch, build_morlet_bank, cwt_power, make_windows, gather_windows,
                      FS, SEG, OVERLAP, CLIP_SIGMA, set_seed, F_MIN, F_MAX)
from transfer_2a import make_cwt_fn, inner_val_split, load_cache as load_2a, device
from eegsym import EEGSym, EEGSYM_REORDER

STIEGER = HERE / 'cache' / 'trials_cyton8_lhrh_v2_W3.npz'
SPLIT_JSON = HERE.parent / 'split_80_20_subjects.json'
OUTDIR = HERE / 'outputs'
N_FREQS, DS = 16, 8
MAX_EPOCHS, PATIENCE, LR, WD = 30, 6, 1e-3, 1e-4
FT_EPOCHS, FT_PATIENCE, FT_LR = 30, 6, 1e-4
EEGSYM_FILTERS = 16       # EEGSym filters per branch (published value: 24); see --filters
REORDER_T = torch.tensor(EEGSYM_REORDER, device=device)


def log(*a): print(*a, flush=True)


def ztrial(X):
    mu = X.mean(axis=2, keepdims=True); sd = X.std(axis=2, keepdims=True) + 1e-6
    return np.clip((X - mu) / sd, -CLIP_SIGMA, CLIP_SIGMA).astype(np.float32)


def make_model(arch, w_samp):
    if arch == 'dualbranch':
        return DualBranch().to(device)
    elif arch == 'eegsym':
        return EEGSym(ncha=8, input_samples=w_samp, fs=FS, filters_per_branch=EEGSYM_FILTERS, dropout=0.4).to(device)
    raise ValueError(arch)


def fwd(model, arch, xw, cwt_fn):
    """Logits for a batch of raw crops (B, 8, w_samp)."""
    if arch == 'eegsym':
        return model(xw.index_select(1, REORDER_T))
    return model(xw, cwt_fn(xw))


def eval_trial(model, arch, Xseg, y_t, rows, cwt_fn, w_samp, stride, batch=256):
    row_idx, start_idx, _ = make_windows(rows, w_samp, stride)
    row_t = torch.tensor(row_idx, device=device); start_t = torch.tensor(start_idx, device=device)
    prob = {}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(row_t), batch):
            rb = row_t[i:i + batch]
            xw = gather_windows(Xseg, rb, start_t[i:i + batch], w_samp)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                p = torch.softmax(fwd(model, arch, xw, cwt_fn), dim=1)
            for r, pr in zip(rb.cpu().numpy(), p.float().cpu().numpy()):
                prob[int(r)] = prob.get(int(r), 0) + pr
    rows_u = np.array(sorted(prob)); P = np.stack([prob[r] for r in rows_u])
    pred = P.argmax(1); true = y_t.cpu().numpy()[rows_u]
    acc = float((pred == true).mean())
    p1t = (true == 1).mean(); p1p = (pred == 1).mean(); pe = p1t * p1p + (1 - p1t) * (1 - p1p)
    kappa = float((acc - pe) / (1 - pe)) if (1 - pe) > 1e-9 else 0.0
    return acc, kappa, int(len(rows_u))


def build_cwt_fn(Xseg, norm_rows, w_samp, stride, seed=42):
    set_seed(seed)
    freqs = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_FREQS)
    tr_row, tr_start, _ = make_windows(norm_rows, w_samp, stride)
    tr_row = torch.tensor(tr_row, device=device); tr_start = torch.tensor(tr_start, device=device)
    kr, ki, pad = build_morlet_bank(freqs, FS, w_samp)
    with torch.no_grad():
        sel = torch.randperm(len(tr_row), device=device)[:min(4096, len(tr_row))]
        cs = cwt_power(gather_windows(Xseg, tr_row[sel], tr_start[sel], w_samp), kr, ki, pad, DS)
        cwt_mu = cs.mean(dim=(0, 1, 3), keepdim=True); cwt_sd = cs.std(dim=(0, 1, 3), keepdim=True) + 1e-5
    return make_cwt_fn(freqs, w_samp, DS, cwt_mu, cwt_sd)


def train(arch, Xseg, y_t, tr_rows, iv_rows, cwt_fn, w_samp, stride, batch,
          init_state=None, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=42):
    set_seed(seed)
    tr_row, tr_start, _ = make_windows(tr_rows, w_samp, stride)
    tr_row = torch.tensor(tr_row, device=device); tr_start = torch.tensor(tr_start, device=device)
    model = make_model(arch, w_samp)
    if init_state is not None:
        model.load_state_dict(copy.deepcopy(init_state))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda')); crit = nn.CrossEntropyLoss()
    n_tr = len(tr_row); best, best_state, since, ep = -1, None, 0, 0
    for ep in range(1, max_epochs + 1):
        model.train(); perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, batch):
            b = perm[i:i + batch]
            xw = gather_windows(Xseg, tr_row[b], tr_start[b], w_samp)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                loss = crit(fwd(model, arch, xw, cwt_fn), y_t[tr_row[b]])
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        iv_acc, _, _ = eval_trial(model, arch, Xseg, y_t, iv_rows, cwt_fn, w_samp, stride)
        if iv_acc > best:
            best, best_state, since = iv_acc, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
        if since >= patience:
            break
    model.load_state_dict(best_state)
    return model, round(best, 4), ep


def load_stieger():
    z = np.load(STIEGER, allow_pickle=True)
    return ztrial(z['X'].astype(np.float32)), z['y'].astype(np.int64), z['subject'].astype(int)


# ------------------------------------------------------------------ sanity check in Stieger
def sanity(arch, w_sec, batch, seed=42, limit=0, maxep=MAX_EPOCHS):
    """Train on the Stieger training subjects and evaluate on the Stieger validation subjects.
    Checks that the model learns (otherwise the port is broken). `limit` > 0 keeps only the
    first `limit` training subjects; `maxep` caps the number of epochs."""
    X, y, subj = load_stieger()
    sp = json.load(open(SPLIT_JSON, encoding='utf-8'))
    tr_subs = sorted(int(s[1:]) for s in sp['train_subjects']); va_subs = set(int(s[1:]) for s in sp['val_subjects'])
    if limit:
        tr_subs = tr_subs[:limit]
    w_samp = int(round(w_sec * FS)); stride = max(1, int(round(w_samp * (1 - OVERLAP))))
    Xseg = torch.tensor(X, device=device); y_t = torch.tensor(y, device=device)
    tr_all = np.where(np.isin(subj, list(tr_subs)))[0]
    va_rows = np.where(np.isin(subj, list(va_subs)))[0]
    tr_rows, iv_rows = inner_val_split(tr_all, y, frac=0.1, seed=seed)
    cwt_fn = build_cwt_fn(Xseg, tr_rows, w_samp, stride, seed=seed) if arch == 'dualbranch' else None
    t0 = time.time()
    model, iv, ep = train(arch, Xseg, y_t, tr_rows, iv_rows, cwt_fn, w_samp, stride, batch, max_epochs=maxep, seed=seed)
    acc, kappa, n = eval_trial(model, arch, Xseg, y_t, va_rows, cwt_fn, w_samp, stride)
    log(f"# sanity {arch} W={w_sec}s: Stieger val acc={acc:.4f} kappa={kappa:.3f} "
        f"(n={n}, iv={iv}, ep={ep}, {time.time()-t0:.0f}s, batch={batch})")
    log(f"#   -> {'learns (acc > 0.60)' if acc > 0.60 else 'does not learn: check the port'}")
    return acc


# ------------------------------------------------------------------ full 2x2 -> 2a
def run_arm(arch, w_sec, batch, Xst, yst, X2a, y2a, subj2a, sess2a, subjects, seed):
    w_samp = int(round(w_sec * FS)); stride = max(1, int(round(w_samp * (1 - OVERLAP))))
    Npool = len(Xst)
    Xall = np.concatenate([Xst, X2a], axis=0); yall = np.concatenate([yst, y2a], axis=0)
    Xseg = torch.tensor(Xall, device=device); y_t = torch.tensor(yall, device=device)
    pool_rows = np.arange(Npool)
    tr_pool, iv_pool = inner_val_split(pool_rows, yall, frac=0.1, seed=seed)
    cwt_fn = build_cwt_fn(Xseg, tr_pool, w_samp, stride, seed=seed) if arch == 'dualbranch' else None
    t0 = time.time()
    pool_model, iv, ep = train(arch, Xseg, y_t, tr_pool, iv_pool, cwt_fn, w_samp, stride, batch, seed=seed)
    log(f"  [{arch} W={w_sec}] pool: iv={iv} ep={ep} ({time.time()-t0:.0f}s)")
    rows = []
    for s in subjects:
        e_rows = np.where((subj2a == s) & (sess2a == 1))[0] + Npool
        t_rows = np.where((subj2a == s) & (sess2a == 0))[0] + Npool
        zs, zsk, n = eval_trial(pool_model, arch, Xseg, y_t, e_rows, cwt_fn, w_samp, stride)
        tin, tho = inner_val_split(t_rows, yall, frac=0.2, seed=seed + s)
        ft, _, _ = train(arch, Xseg, y_t, tin, tho, cwt_fn, w_samp, stride, batch,
                         init_state=pool_model.state_dict(), lr=FT_LR, max_epochs=FT_EPOCHS,
                         patience=FT_PATIENCE, seed=seed + s)
        cal, calk, _ = eval_trial(ft, arch, Xseg, y_t, e_rows, cwt_fn, w_samp, stride)
        log(f"    A{s:02d}: zero-shot={zs:.4f} -> calibrated={cal:.4f} lift={cal-zs:+.4f}")
        rows.append({'arch': arch, 'window': w_sec, 'subject': s, 'acc_zeroshot': round(zs, 4),
                     'kappa_zeroshot': round(zsk, 4), 'acc_calib': round(cal, 4),
                     'kappa_calib': round(calk, 4), 'lift': round(cal - zs, 4), 'n_test': n})
        del ft
        if device.type == 'cuda': torch.cuda.empty_cache()
    del pool_model, Xseg
    if device.type == 'cuda': torch.cuda.empty_cache()
    return rows


def main():
    global EEGSYM_FILTERS
    ap = argparse.ArgumentParser()
    ap.add_argument('--sanity', action='store_true')
    ap.add_argument('--arch', choices=['dualbranch', 'eegsym'], default='eegsym')
    ap.add_argument('--window', type=float, default=3.0)
    ap.add_argument('--batch', type=int, default=0, help='0 = automatic, by arch and window')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0, help='sanity: training subjects (0 = all)')
    ap.add_argument('--maxep', type=int, default=MAX_EPOCHS, help='sanity: maximum epochs')
    ap.add_argument('--filters', type=int, default=EEGSYM_FILTERS)
    args = ap.parse_args()
    EEGSYM_FILTERS = args.filters
    OUTDIR.mkdir(exist_ok=True)

    def auto_batch(arch, w):
        if arch == 'eegsym' and w >= 2.0: return 64
        return 128

    if args.sanity:
        b = args.batch or auto_batch(args.arch, args.window)
        sanity(args.arch, args.window, b, seed=args.seed, limit=args.limit, maxep=args.maxep); return

    Xst, yst, _ = load_stieger()
    X2a_raw, y2a, subj2a, sess2a, _ = load_2a()
    X2a = ztrial(X2a_raw.astype(np.float32))
    log(f"# Stieger {Xst.shape} | 2a {X2a.shape} | seed={args.seed}")
    subjects = list(range(1, 10))
    # the W = 3.0 s arms run first, so a broken EEGSym port shows up early
    arms = [('dualbranch', 3.0), ('eegsym', 3.0), ('dualbranch', 0.5), ('eegsym', 0.5)]
    all_rows = []
    t0 = time.time()
    for arch, w in arms:
        b = args.batch or auto_batch(arch, w)
        rows = run_arm(arch, w, b, Xst, yst, X2a, y2a, subj2a, sess2a, subjects, args.seed)
        all_rows.extend(rows)
        zs = np.mean([r['acc_zeroshot'] for r in rows]); ca = np.mean([r['acc_calib'] for r in rows])
        log(f"# {arch} W={w}: mean zero-shot={zs:.4f} calibrated={ca:.4f}\n")
    csv_path = OUTDIR / f'results_arch_seed{args.seed}.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); wr.writeheader(); wr.writerows(all_rows)
    log(f"# saved {csv_path.name} ({time.time()-t0:.0f}s)")
    log("# done")


if __name__ == '__main__':
    main()
