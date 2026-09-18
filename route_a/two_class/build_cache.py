#!/usr/bin/env python
"""Build the two-class trial cache used by every script in this folder.

Reads the Stieger2021 .mat files (sessions 5-11) and keeps the left/right-hand trials
of the pure LR task (tasknumber 1, targets 1 and 2) that last at least `--window`
seconds and did not time out. Each trial is processed as follows:

  1. common average reference over the 62 EEG channels
  2. pick the 8 channels FC3, FCz, FC4, C3, Cz, C4, CP3, CP4
  3. resample 1000 -> 250 Hz (polyphase)
  4. subtract the mean of the last second before the cue, [-1, 0) s
  5. crop the motor-imagery window starting at cursor onset (+2 s)

Artifact flags are stored, not used to drop trials. The online hit rate of every
subject (LR task) is stored alongside.

Output: cache/trials_cyton8_lhrh_v2_W3.npz
    X (N, 8, 750) float32, y (0 = left, 1 = right), subject, session,
    f_amp (8-30 Hz amplitude above 150 uV), f_art (artifact flag of the dataset),
    channels, hit_subjects, hit_values

Usage:
    python build_cache.py                 # W = 3 s, the window used in the paper
"""
import os
import glob
import time
import argparse
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from pymatreader import read_mat

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = Path(os.environ.get('BCI_DATA', REPO / 'data')) / 'stieger2021'
CACHE = HERE / 'cache'

CYTON_8 = ['FC3', 'FCZ', 'FC4', 'C3', 'CZ', 'C4', 'CP3', 'CP4']
FS_SRC, FS_TGT = 1000, 250
T0_MS = -2000                 # first sample of the stored trial
MI_ONSET_MS = 2000            # cursor appears: motor imagery starts
BASE_MS = (-1000, 0)          # baseline: last second before the cue
AMP_THR_UV = 150.0
SESSIONS = [5, 6, 7, 8, 9, 10, 11]


def car(trial_62):
    return trial_62 - trial_62.mean(axis=0, keepdims=True)


def pick(trial_62, labels, subset=CYTON_8):
    idx = [labels.index(c) for c in subset]
    return trial_62[idx, :]


def downsample(x, fs_src=FS_SRC, fs_tgt=FS_TGT, axis=-1):
    g = gcd(fs_src, fs_tgt)
    return resample_poly(x, fs_tgt // g, fs_src // g, axis=axis)


def ds_time(n, fs_tgt=FS_TGT, t0=T0_MS):
    return np.arange(n) * (1000.0 / fs_tgt) + t0


def baseline(x, t_ms_axis, base=BASE_MS):
    m = (t_ms_axis >= base[0]) & (t_ms_axis < base[1])
    return x - x[:, m].mean(axis=1, keepdims=True)


def crop_mi(x, t_ms_axis, w, onset=MI_ONSET_MS, fs_tgt=FS_TGT):
    i0 = int(np.searchsorted(t_ms_axis, onset))
    n = int(round(w * fs_tgt))
    return x[:, i0:i0 + n]


def _bandpass(s, lo, hi, fs, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, s, axis=-1)


def flag_amplitud(xc, thr=AMP_THR_UV):
    return bool(np.abs(_bandpass(xc, 8, 30, FS_TGT)).max() > thr)


def parse_name(fp):
    stem = Path(fp).stem            # S4_Session_5
    sid = int(stem.split('_')[0][1:])
    ses = int(stem.split('_')[-1])
    return sid, ses


def build(w):
    npz = CACHE / f'trials_cyton8_lhrh_v2_W{w:g}.npz'
    files = sorted(glob.glob(str(DATA / 'S*_Session_*.mat')))
    files = [f for f in files if parse_name(f)[1] in SESSIONS]
    if not files:
        raise SystemExit(f'no Stieger2021 .mat files found in {DATA}')
    Xs, ys, subs, sess, fa_, fr_ = [], [], [], [], [], []
    hits = {}
    n_target = int(round(w * FS_TGT))
    t0 = time.time()
    for k, fp in enumerate(files, 1):
        sid, ses = parse_name(fp)
        try:
            m = read_mat(str(fp)); bci = m['BCI']
            labels = list(bci['chaninfo']['label']); data = bci['data']; td = bci['TrialData']
            tasknum = np.asarray(td['tasknumber']).flatten().astype(int)
            target = np.asarray(td['targetnumber']).flatten().astype(int)
            result = np.asarray(td['result'], dtype=float).flatten()
            triallen = np.asarray(td['triallength'], dtype=float).flatten()
            artflag = np.asarray(td['artifact']).flatten().astype(int)
            lr = (tasknum == 1) & np.isin(target, [1, 2])
            hh = hits.setdefault(sid, [0, 0])
            hh[0] += int(np.nansum(result[lr] == 1)); hh[1] += int(np.sum(~np.isnan(result[lr])))
            keep = lr & (triallen >= w) & ~np.isnan(result)
            yf = np.where(target == 2, 0, 1)
            for i in np.where(keep)[0]:
                try:
                    tr = np.asarray(data[i], dtype=float)
                    if tr.shape[0] != len(labels):
                        tr = tr.T
                    x = car(tr); x = pick(x, labels); x = downsample(x)
                    t = ds_time(x.shape[1]); x = baseline(x, t)
                    xc = crop_mi(x, t, w=w)
                    if xc.shape[1] != n_target:
                        continue
                    Xs.append(xc.astype(np.float32)); ys.append(int(yf[i]))
                    subs.append(sid); sess.append(ses)
                    fa_.append(flag_amplitud(xc)); fr_.append(bool(artflag[i]))
                except Exception:
                    continue
            del m, bci, data
        except Exception as e:
            print(f'  failed {Path(fp).name}: {e}', flush=True)
            continue
        if k % 50 == 0:
            print(f'  [{k}/{len(files)}] ({time.time()-t0:.0f}s)', flush=True)
    X = np.array(Xs, np.float32); y = np.array(ys, np.int64)
    subject = np.array(subs, np.int16); session = np.array(sess, np.int16)
    hr = {int(s): (v[0] / v[1] if v[1] else np.nan) for s, v in hits.items()}
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(
        npz, X=X, y=y, subject=subject, session=session,
        f_amp=np.array(fa_, bool), f_art=np.array(fr_, bool),
        channels=np.array(CYTON_8),
        hit_subjects=np.array(sorted(hr)), hit_values=np.array([hr[s] for s in sorted(hr)]))
    print(f'X{X.shape} in {time.time()-t0:.0f}s -> {npz}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=float, default=3.0, help='MI window length in seconds')
    args = ap.parse_args()
    build(args.window)


if __name__ == '__main__':
    main()
