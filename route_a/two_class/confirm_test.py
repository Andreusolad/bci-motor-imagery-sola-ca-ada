#!/usr/bin/env python
r"""Held-out test of the cell chosen in run_grid.py: W = 0.5 s, 16 frequencies, CWT
downsampling on.

The cell was chosen on the 12 validation subjects of the 80/20 split, so those subjects
cannot be the test set. This script splits the 62 subjects as follows:
  - test:       10 subjects drawn (seed 42) from the 50 original training subjects,
                never used for model selection
  - validation: the 12 original validation subjects, used only for early stopping
  - training:   the remaining 40 subjects

It trains the cell once with the pipeline of run_grid.py (same cropping, CWT, model and
training settings), stops early on the validation trial accuracy and evaluates the best
epoch once on the test subjects: trial accuracy, window accuracy, kappa and trial accuracy
per test subject.

Output: outputs/confirm_test.json

Usage:
  python confirm_test.py
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_grid as G   # model, CWT, cropping and training settings

# --- chosen cell ---
WIN_SEC = 0.5
N_FREQS = 16
DOWNSAMPLE = True
N_TEST_SUBS = 10          # test subjects, drawn from the 50 original training subjects
SEED = 42


def build_split():
    """train (40) / val (12) / test (10); test comes from the original training subjects."""
    sp = json.load(open(G.SPLIT_JSON, encoding='utf-8'))
    old_train = sorted(int(s[1:]) for s in sp['train_subjects'])   # 50
    old_val = sorted(int(s[1:]) for s in sp['val_subjects'])       # 12
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(old_train))
    test_subs = sorted(old_train[i] for i in perm[:N_TEST_SUBS])
    train_subs = sorted(old_train[i] for i in perm[N_TEST_SUBS:])
    val_subs = old_val
    assert not (set(test_subs) & set(old_val)), 'test overlaps the model-selection subjects'
    assert not (set(test_subs) & set(train_subs))
    return train_subs, val_subs, test_subs


def eval_split(model, Xseg, y_t, rows, starts, w_samp, make_cwt, subj_t):
    """Return (win_acc, trial_acc, kappa, {subject: trial accuracy}) over the given crops."""
    model.eval()
    win_c, win_n = 0, 0
    prob_sum = {}
    with torch.no_grad():
        for i in range(0, len(rows), 512):
            rb = rows[i:i + 512]
            xw = G.gather_windows(Xseg, rb, starts[i:i + 512], w_samp)
            with torch.amp.autocast('cuda', enabled=(G.device.type == 'cuda')):
                p = torch.softmax(model(xw, make_cwt(xw)), dim=1)
            yb = y_t[rb]
            win_c += (p.argmax(1) == yb).sum().item(); win_n += len(rb)
            for r, pr in zip(rb.cpu().numpy(), p.float().cpu().numpy()):
                prob_sum[int(r)] = prob_sum.get(int(r), 0) + pr
    rows_u = np.array(sorted(prob_sum))
    pred = np.array([prob_sum[r].argmax() for r in rows_u])
    true = y_t.cpu().numpy()[rows_u]
    trial_acc = float((pred == true).mean())
    po = trial_acc
    p1t = (true == 1).mean(); p1p = (pred == 1).mean()
    pe = p1t * p1p + (1 - p1t) * (1 - p1p)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else 0.0
    # per subject
    subj_np = subj_t.cpu().numpy()
    per = {}
    for r, pd, tr in zip(rows_u, pred, true):
        s = int(subj_np[r]); per.setdefault(s, [0, 0])
        per[s][1] += 1; per[s][0] += int(pd == tr)
    per_sub = {s: round(v[0] / v[1], 3) for s, v in sorted(per.items())}
    return win_c / max(1, win_n), trial_acc, kappa, per_sub


def main():
    G.set_seed(SEED)
    train_subs, val_subs, test_subs = build_split()
    G.log('#' * 72)
    G.log(f'# held-out test | cell W={WIN_SEC}s nf={N_FREQS} ds={DOWNSAMPLE}')
    G.log(f'# train {len(train_subs)} / val {len(val_subs)} / test {len(test_subs)} subjects')
    G.log(f'# test subjects: {test_subs}')
    G.log('#' * 72)

    X, y, subj, _, _ = G.load_data()
    tr_mask = np.isin(subj, train_subs)
    va_mask = np.isin(subj, val_subs)
    te_mask = np.isin(subj, test_subs)

    # raw z-score with the training statistics of this split, clipped
    mu = X[tr_mask].mean(axis=(0, 2), keepdims=True)
    sd = X[tr_mask].std(axis=(0, 2), keepdims=True) + 1e-6
    Xn = np.clip((X - mu) / sd, -G.CLIP_SIGMA, G.CLIP_SIGMA).astype(np.float32)
    Xseg = torch.tensor(Xn, device=G.device)
    y_t = torch.tensor(y, device=G.device)
    subj_t = torch.tensor(subj, device=G.device)

    w_samp = int(round(WIN_SEC * G.FS))
    stride = max(1, int(round(w_samp * (1 - G.OVERLAP))))
    ds = G.DS_FACTOR if DOWNSAMPLE else 1

    def wins(mask):
        r, s, _ = G.make_windows(np.where(mask)[0], w_samp, stride)
        return torch.tensor(r, device=G.device), torch.tensor(s, device=G.device)
    tr_row, tr_start = wins(tr_mask)
    va_row, va_start = wins(va_mask)
    te_row, te_start = wins(te_mask)

    freqs = np.logspace(np.log10(G.F_MIN), np.log10(G.F_MAX), N_FREQS)
    kr, ki, pad = G.build_morlet_bank(freqs, G.FS, w_samp)
    with torch.no_grad():
        sel = torch.randperm(len(tr_row), device=G.device)[:4096]
        cs = G.cwt_power(G.gather_windows(Xseg, tr_row[sel], tr_start[sel], w_samp), kr, ki, pad, ds)
        cwt_mu = cs.mean(dim=(0, 1, 3), keepdim=True); cwt_sd = cs.std(dim=(0, 1, 3), keepdim=True) + 1e-5

    def make_cwt(xw):
        return (G.cwt_power(xw, kr, ki, pad, ds) - cwt_mu) / cwt_sd

    model = G.DualBranch().to(G.device)
    opt = torch.optim.Adam(model.parameters(), lr=G.LR, weight_decay=G.WD)
    scaler = torch.amp.GradScaler('cuda', enabled=(G.device.type == 'cuda'))
    crit = nn.CrossEntropyLoss()

    best_val, best_state, best_ep, since = -1.0, None, -1, 0
    n_tr = len(tr_row); t0 = time.time()
    for ep in range(1, G.MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_tr, device=G.device)
        for i in range(0, n_tr, G.BATCH):
            b = perm[i:i + G.BATCH]
            xw = G.gather_windows(Xseg, tr_row[b], tr_start[b], w_samp)
            with torch.amp.autocast('cuda', enabled=(G.device.type == 'cuda')):
                loss = crit(model(xw, make_cwt(xw)), y_t[tr_row[b]])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        # early stopping on validation, never on test
        _, val_trial, _, _ = eval_split(model, Xseg, y_t, va_row, va_start, w_samp, make_cwt, subj_t)
        if val_trial > best_val:
            best_val = val_trial; best_ep = ep; since = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        G.log(f'  ep{ep:>2}  val_trial={val_trial:.4f}  (best {best_val:.4f} @ep{best_ep})')
        if since >= G.PATIENCE:
            break

    # restore the best epoch (by validation) and evaluate once on test
    model.load_state_dict(best_state)
    te_win, te_trial, te_kappa, te_per = eval_split(model, Xseg, y_t, te_row, te_start, w_samp, make_cwt, subj_t)
    va_win, va_trial, va_kappa, _ = eval_split(model, Xseg, y_t, va_row, va_start, w_samp, make_cwt, subj_t)

    G.log('\n' + '=' * 60)
    G.log(f'W={WIN_SEC}s nf={N_FREQS} ds={DOWNSAMPLE} (best ep {best_ep}, '
          f'{time.time()-t0:.0f}s)')
    G.log(f'  val  (12 subj): trial={va_trial:.4f}  win={va_win:.4f}  kappa={va_kappa:.4f}')
    G.log(f'  test (10 subj): trial={te_trial:.4f}  win={te_win:.4f}  kappa={te_kappa:.4f}')
    G.log('=' * 60)
    G.log('  test, per subject:')
    for s, a in te_per.items():
        G.log(f'    S{s:>2}: {a}')

    out = {
        'winner': {'window_sec': WIN_SEC, 'n_freqs': N_FREQS, 'downsample': DOWNSAMPLE},
        'split': {'train_subs': train_subs, 'val_subs': val_subs, 'test_subs': test_subs},
        'best_epoch': best_ep,
        'val': {'trial_acc': round(va_trial, 4), 'win_acc': round(va_win, 4), 'kappa': round(va_kappa, 4)},
        'test': {'trial_acc': round(te_trial, 4), 'win_acc': round(te_win, 4), 'kappa': round(te_kappa, 4)},
        'test_per_subject': te_per,
    }
    (G.OUTDIR / 'confirm_test.json').write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    G.log(f'\nSaved {G.OUTDIR / "confirm_test.json"}.')


if __name__ == '__main__':
    main()
