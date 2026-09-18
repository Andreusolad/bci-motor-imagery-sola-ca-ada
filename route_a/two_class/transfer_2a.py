#!/usr/bin/env python
r"""Transfer of the dual-branch decoder (raw + CWT) to BCI-IV-2a.

Uses DualBranch, the Morlet CWT and the cropping of run_grid.py; only the data loading is
specific to BCI-IV-2a. Two protocols, both without subject leakage:
  P2  leave-one-subject-out within 2a: for each subject, train on 7 subjects, stop early on
      the next subject in the list (wrapping around) and test on all trials of the left-out
      subject (both sessions).
  P3  cross-dataset zero-shot: the Stieger checkpoint checkpoints/winner_W0.5_F16_dsON.pt
      (written by train_winner.py) applied to 2a, inference only. Each 2a subject is z-scored
      with the statistics of its own trials (no labels involved); the CWT statistics come
      from the checkpoint.

Preprocessing (--build-cache): CAR over the 22 EEG channels, then the 8 channels FC3, FCz, FC4,
C3, Cz, C4, CP3, CP4 in the order of the Stieger cache; no band-pass; MI window [2, 5] s after
the trial start, i.e. 750 samples from the cue; optional baseline subtraction (mean over
[0.5, 1.5] s, before the cue). In P2, per-channel z-score with training statistics clipped to
+-8 sigma, and CWT with 16 frequencies, 4-40 Hz, downsampling on.

Input: $BCI_DATA/bci_iv_2a/A0<n>T.npz, A0<n>E.npz and true_labels/A0<n>T.mat, A0<n>E.mat.
Output: cache/cache_2a.npz and outputs/results_2a_<protocol>_W<window>[_noart].csv.

Usage:
  python transfer_2a.py --build-cache            # build cache/cache_2a.npz (once)
  python transfer_2a.py --protocol P2 --window 0.5
  python transfer_2a.py --protocol P3            # Stieger checkpoint applied to 2a
  python transfer_2a.py --protocol P2 --window 0.5 --smoke   # evaluate subjects 1 and 2 only
"""
import os, sys, csv, json, time, copy, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
from run_grid import (DualBranch, build_morlet_bank, cwt_power, make_windows,
                      gather_windows, FS, SEG, OVERLAP, CLIP_SIGMA, set_seed,
                      F_MIN, F_MAX, N_CYCLES, DS_FACTOR)

DATA = Path(os.environ.get('BCI_DATA', REPO / 'data')) / 'bci_iv_2a'
LAB = DATA / 'true_labels'
CKPT_STIEGER = HERE / 'checkpoints' / 'winner_W0.5_F16_dsON.pt'
CACHE = HERE / 'cache' / 'cache_2a.npz'
OUTDIR = HERE / 'outputs'

STD22 = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6',
         'CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']
CYTON = ['FC3','FCz','FC4','C3','Cz','C4','CP3','CP4']       # same order as the Stieger cache
CYIDX = [STD22.index(c) for c in CYTON]                       # [1,3,5,7,9,11,13,17]
MI_INI = int(2.0 * FS)          # the cue comes 2 s after the trial start
BASE_INI, BASE_FIN = int(0.5 * FS), int(1.5 * FS)   # baseline before the cue

# training (same settings as for Stieger)
MAX_EPOCHS, PATIENCE, BATCH, LR, WD = 30, 6, 128, 1e-3, 1e-4
N_FREQS, DOWNSAMPLE = 16, True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log(*a): print(*a, flush=True)


# ---------------------------------------------------------------- 2a loader
def build_cache(baseline=True):
    """Extract every left/right-hand trial of the 18 sessions -> cache/cache_2a.npz.

    X (N,8,750) float32, y (0=LH,1=RH), subject (1-9), session (0=T,1=E), artifact (0/1).
    CAR over the 22 EEG channels -> 8 channels; MI window [2,5] s; optional baseline before
    the cue.
    """
    Xs, ys, subs, sess, arts = [], [], [], [], []
    for num in range(1, 10):
        for si, ses in enumerate(['T', 'E']):
            npz = np.load(DATA / f'A{num:02d}{ses}.npz', allow_pickle=True)
            s = npz['s'].astype(np.float64)
            etyp = npz['etyp'].flatten().astype(int)
            epos = npz['epos'].flatten().astype(int)
            art = npz['artifacts'].flatten().astype(int)
            lab = loadmat(LAB / f'A{num:02d}{ses}.mat')['classlabel'].flatten() - 1
            eeg = s[:, :22]
            eeg = eeg - eeg.mean(axis=1, keepdims=True)      # CAR over the 22 EEG channels
            eeg = eeg[:, CYIDX]                               # keep 8 channels
            starts = np.where(etyp == 768)[0]                 # event 768: trial start
            for i, idx in enumerate(starts):
                if lab[i] not in (0, 1):                      # left/right hand only
                    continue
                pos = epos[idx]
                ini, fin = pos + MI_INI, pos + MI_INI + SEG
                if fin > eeg.shape[0]:
                    continue
                seg = eeg[ini:fin, :].copy()                  # (750,8)
                if baseline:
                    b = eeg[pos + BASE_INI:pos + BASE_FIN, :].mean(axis=0, keepdims=True)
                    seg = seg - b
                Xs.append(seg.T.astype(np.float32))           # (8,750)
                ys.append(int(lab[i])); subs.append(num); sess.append(si); arts.append(int(art[i]))
    X = np.stack(Xs); y = np.array(ys, np.int64)
    subj = np.array(subs, np.int64); session = np.array(sess, np.int64); artifact = np.array(arts, np.int64)
    np.savez_compressed(CACHE, X=X, y=y, subject=subj, session=session, artifact=artifact,
                        channels=np.array(CYTON), baseline=baseline)
    log(f'cache -> {CACHE.name}: X{X.shape} y{y.shape} | LH{(y==0).sum()} RH{(y==1).sum()} | '
        f'subs{sorted(set(subs))} | artifacts {artifact.sum()}/{len(artifact)}')
    return CACHE


def load_cache():
    if not CACHE.exists():
        build_cache()
    z = np.load(CACHE, allow_pickle=True)
    return z['X'], z['y'], z['subject'], z['session'], z['artifact']


# ---------------------------------------------------------------- train / evaluate
def make_cwt_fn(freqs, w_samp, ds, cwt_mu, cwt_sd):
    """Return a function mapping crops (B, 8, w_samp) to the standardized CWT."""
    kr, ki, pad = build_morlet_bank(freqs, FS, w_samp)
    def fn(xw):
        c = cwt_power(xw, kr, ki, pad, ds)
        return (c - cwt_mu) / cwt_sd
    return fn


def eval_trial(model, Xseg, y_t, rows, cwt_fn, w_samp, stride):
    row_idx, start_idx, _ = make_windows(rows, w_samp, stride)
    row_t = torch.tensor(row_idx, device=device); start_t = torch.tensor(start_idx, device=device)
    prob = {}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(row_t), 512):
            rb = row_t[i:i+512]
            xw = gather_windows(Xseg, rb, start_t[i:i+512], w_samp)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                p = torch.softmax(model(xw, cwt_fn(xw)), dim=1)
            for r, pr in zip(rb.cpu().numpy(), p.float().cpu().numpy()):
                prob[int(r)] = prob.get(int(r), 0) + pr
    rows_u = np.array(sorted(prob)); P = np.stack([prob[r] for r in rows_u])
    pred = P.argmax(1); true = y_t.cpu().numpy()[rows_u]
    acc = float((pred == true).mean())
    p1t = (true == 1).mean(); p1p = (pred == 1).mean(); pe = p1t*p1p + (1-p1t)*(1-p1p)
    kappa = float((acc-pe)/(1-pe)) if (1-pe) > 1e-9 else 0.0
    return acc, kappa, int(len(rows_u))


def setup_norm(Xn, y, norm_rows, w_sec, seed=42):
    """Raw z-score and CWT statistics from norm_rows, applied to all of X.

    Returns (Xseg, y_t, cwt_fn, w_samp, stride)."""
    set_seed(seed)
    w_samp = int(round(w_sec * FS)); stride = max(1, int(round(w_samp*(1-OVERLAP))))
    ds = DS_FACTOR if DOWNSAMPLE else 1
    mu = Xn[norm_rows].mean(axis=(0, 2), keepdims=True)
    sd = Xn[norm_rows].std(axis=(0, 2), keepdims=True) + 1e-6
    Xz = np.clip((Xn - mu)/sd, -CLIP_SIGMA, CLIP_SIGMA).astype(np.float32)
    Xseg = torch.tensor(Xz, device=device); y_t = torch.tensor(y, device=device)
    freqs = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_FREQS)
    tr_row, tr_start, _ = make_windows(norm_rows, w_samp, stride)
    tr_row = torch.tensor(tr_row, device=device); tr_start = torch.tensor(tr_start, device=device)
    kr, ki, pad = build_morlet_bank(freqs, FS, w_samp)
    with torch.no_grad():
        sel = torch.randperm(len(tr_row), device=device)[:min(4096, len(tr_row))]
        cs = cwt_power(gather_windows(Xseg, tr_row[sel], tr_start[sel], w_samp), kr, ki, pad, ds)
        cwt_mu = cs.mean(dim=(0, 1, 3), keepdim=True); cwt_sd = cs.std(dim=(0, 1, 3), keepdim=True)+1e-5
    return Xseg, y_t, make_cwt_fn(freqs, w_samp, ds, cwt_mu, cwt_sd), w_samp, stride


def train_model(Xseg, y_t, tr_rows, iv_rows, cwt_fn, w_samp, stride,
                init_state=None, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=42):
    """Train on tr_rows (fine-tune if init_state is given), early stopping on iv_rows.

    Returns (model at its best epoch, best inner-validation trial accuracy)."""
    set_seed(seed)
    tr_row, tr_start, _ = make_windows(tr_rows, w_samp, stride)
    tr_row = torch.tensor(tr_row, device=device); tr_start = torch.tensor(tr_start, device=device)
    model = DualBranch().to(device)
    if init_state is not None:
        model.load_state_dict(copy.deepcopy(init_state))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda')); crit = nn.CrossEntropyLoss()
    n_tr = len(tr_row); best, best_state, since = -1, None, 0
    for ep in range(1, max_epochs+1):
        model.train(); perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, BATCH):
            b = perm[i:i+BATCH]
            xw = gather_windows(Xseg, tr_row[b], tr_start[b], w_samp)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                loss = crit(model(xw, cwt_fn(xw)), y_t[tr_row[b]])
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        iv_acc, _, _ = eval_trial(model, Xseg, y_t, iv_rows, cwt_fn, w_samp, stride)
        if iv_acc > best:
            best, best_state, since = iv_acc, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
        if since >= patience:
            break
    model.load_state_dict(best_state)
    return model, round(best, 4)


def train_eval(Xn, y, tr_rows, iv_rows, te_rows, w_sec, seed=42):
    """z-score with tr_rows statistics; train; early stopping on iv_rows; evaluate on te_rows."""
    Xseg, y_t, cwt_fn, w_samp, stride = setup_norm(Xn, y, tr_rows, w_sec, seed)
    model, best = train_model(Xseg, y_t, tr_rows, iv_rows, cwt_fn, w_samp, stride, seed=seed)
    acc, kappa, n = eval_trial(model, Xseg, y_t, te_rows, cwt_fn, w_samp, stride)
    del model, Xseg
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {'acc': acc, 'kappa': kappa, 'n_test': n, 'iv_acc': best}


# ---------------------------------------------------------------- protocols
def inner_val_split(tr_rows, y, frac=0.2, seed=0):
    """Stratified hold-out: `frac` of each class (at least one trial) -> (train, held-out)."""
    rng = np.random.RandomState(seed); tr_rows = np.asarray(tr_rows)
    ho = np.zeros(len(tr_rows), bool)
    for cls in (0, 1):
        idx = np.where(y[tr_rows] == cls)[0]; rng.shuffle(idx)
        ho[idx[:max(1, int(frac*len(idx)))]] = True
    return tr_rows[~ho], tr_rows[ho]


def protocol_P2(X, y, subj, sess, w_sec, subjects):
    rows_out = []
    for s in subjects:
        te = np.where(subj == s)[0]
        tr_all = np.where(subj != s)[0]
        # inner validation: the next subject in the list, used only for early stopping
        others = [o for o in subjects if o != s]
        iv_subj = others[(subjects.index(s)) % len(others)]
        iv = np.where(subj == iv_subj)[0]
        tr = np.where((subj != s) & (subj != iv_subj))[0]
        r = train_eval(X, y, tr, iv, te, w_sec, seed=200+s)
        log(f'  P2 LOSO A{s:02d}: acc={r["acc"]:.4f} kappa={r["kappa"]:.4f} (iv=A{iv_subj:02d}, n_test={r["n_test"]})')
        rows_out.append({'protocol': 'P2', 'subject': s, 'window': w_sec, **r})
    return rows_out


def protocol_P3(X, y, subj, subjects, recompute_cwt=False):
    """Cross-dataset zero-shot: Stieger checkpoint -> 2a. Raw z-score per 2a subject."""
    ck = torch.load(CKPT_STIEGER, weights_only=False)
    w_samp = int(ck['w_samp']); stride = int(ck['stride']); ds = int(ck['ds'])
    freqs = np.asarray(ck['freqs'])
    cwt_mu = ck['cwt_mu'].to(device); cwt_sd = ck['cwt_sd'].to(device)
    cwt_fn = make_cwt_fn(freqs, w_samp, ds, cwt_mu, cwt_sd)
    model = DualBranch().to(device); model.load_state_dict(ck['state_dict'])
    rows_out = []
    for s in subjects:
        rows = np.where(subj == s)[0]
        mu = X[rows].mean(axis=(0, 2), keepdims=True); sd = X[rows].std(axis=(0, 2), keepdims=True)+1e-6
        Xz = np.clip((X - mu)/sd, -CLIP_SIGMA, CLIP_SIGMA).astype(np.float32)
        Xseg = torch.tensor(Xz, device=device); y_t = torch.tensor(y, device=device)
        acc, kappa, n = eval_trial(model, Xseg, y_t, rows, cwt_fn, w_samp, stride)
        log(f'  P3 xdata A{s:02d}: acc={acc:.4f} kappa={kappa:.4f} (n={n})')
        rows_out.append({'protocol': 'P3', 'subject': s, 'window': ck['w_sec'],
                         'acc': acc, 'kappa': kappa, 'n_test': n, 'iv_acc': ''})
        del Xseg
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-cache', action='store_true')
    ap.add_argument('--no-baseline', action='store_true')
    ap.add_argument('--protocol', choices=['P2', 'P3'])
    ap.add_argument('--window', type=float, default=0.5)
    ap.add_argument('--drop-art', action='store_true', help='drop trials flagged as artifacts')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    OUTDIR.mkdir(exist_ok=True)
    CACHE.parent.mkdir(exist_ok=True)

    if args.build_cache:
        build_cache(baseline=not args.no_baseline); return
    if not args.protocol:
        log('nothing to do: pass --build-cache or --protocol {P2,P3}'); return

    X, y, subj, sess, art = load_cache()
    if args.drop_art:
        m = art == 0
        X, y, subj, sess, art = X[m], y[m], subj[m], sess[m], art[m]
        log(f'# drop-art: {m.sum()}/{len(m)} trials kept (no artifact flag)')
    subjects = [1, 2] if args.smoke else list(range(1, 10))
    log(f'# {args.protocol} | W={args.window}s | drop_art={args.drop_art} | subjects={subjects} | X{X.shape}')
    t0 = time.time()
    if args.protocol == 'P2':
        rows = protocol_P2(X, y, subj, sess, args.window, subjects)
    else:
        rows = protocol_P3(X, y, subj, subjects)
    accs = [r['acc'] for r in rows]
    log(f'# {args.protocol} mean acc={np.mean(accs):.4f} '
        f'(min {np.min(accs):.3f} max {np.max(accs):.3f}) in {time.time()-t0:.1f}s')
    tag = f'{args.protocol}_W{args.window}' + ('_noart' if args.drop_art else '')
    csv_path = OUTDIR / f'results_2a_{tag}.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    log(f'# saved {csv_path.name}')


if __name__ == '__main__':
    main()
