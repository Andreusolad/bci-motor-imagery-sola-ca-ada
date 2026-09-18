"""Builds the 3-class window dataset (LH / RH / IDLE) from Stieger et al. (2021).

A left/right classifier always outputs one of its two classes, even when the user is not
imagining anything. An explicit third class, IDLE (no control), trained on awake rest EEG,
lets the model output "no command". IDLE windows come from two sources of the same
dataset:

  Source A: pre-cue rest [-2000, 0] ms of the LR trials (task 1). The user is awake and
     still before the target appears, with the same task, montage and session as the
     LH/RH trials. Windows are cut by sliding inside this region (at most idle_por_trial
     per trial).

  Source B: trials with targetnumber == 4, the REST target of task 2 (UD) and task 3 (2D).
     This is cued rest: the user sees the rest target and performs no motor imagery.

Both are cut to the same length W as the MI windows and go through the same
preprocessing (CAR according to CAR_MODE, downsampling, baseline). The stored windows
are not band-pass filtered.

Reads:  $BCI_DATA/stieger2021/S*_Session_*.mat (sessions in config.SESSIONS)
Writes: async_stieger/cache/trials_cyton8_3clase.npz
        (trials_cyton8_3clase_smoke.npz with smoke=True)

Usage:
    from rt_system.dataset import construir_cache_3clase
    construir_cache_3clase(smoke=True)     # subjects 1-4 only, quick check
    construir_cache_3clase()               # full dataset (reads every .mat file)
"""
from __future__ import annotations
import glob
import time
from pathlib import Path

import numpy as np
from pymatreader import read_mat

from . import config as C
from .config import log
from .preprocessing import (aplicar_car, downsample, ds_time, baseline,
                            crop_mi, crop_idle_windows, flag_amplitud)


# ============================================================
# File name parsing
# ============================================================
def parse_name(fp: str) -> tuple[int, int]:
    stem = Path(fp).stem            # 'S4_Session_5'
    return int(stem.split('_')[0][1:]), int(stem.split('_')[-1])


# ============================================================
# One session -> labelled windows
# ============================================================
def preprocesar_sesion(mat_path, w_samp: int = C.W_SAMP, incluir_rest: bool = True,
                       idle_por_trial: int = 2, car_mode: str = C.CAR_MODE) -> dict | None:
    """Return a dict with the windows (n, 8, w_samp) and metadata of one session.

    Keys: X, y (0=LH, 1=RH, 2=IDLE), source ('mi' | 'precue' | 'rest'), f_amp (amplitude
          artifact flag), hit (n_hit, n_dec) for the subject's online LR hit rate.
    Returns None if the session yields no window.
    """
    m = read_mat(str(mat_path))
    bci = m['BCI']
    labels = list(bci['chaninfo']['label'])
    data = bci['data']
    td = bci['TrialData']
    tasknum = np.asarray(td['tasknumber']).flatten().astype(int)
    target  = np.asarray(td['targetnumber']).flatten().astype(int)
    result  = np.asarray(td['result'], dtype=float).flatten()
    triallen = np.asarray(td['triallength'], dtype=float).flatten()
    artflag = np.asarray(td['artifact']).flatten().astype(int)

    # online hit rate over all LR trials (task 1, target 1/2)
    lr = (tasknum == 1) & np.isin(target, [1, 2])
    n_hit = int(np.nansum(result[lr] == 1))
    n_dec = int(np.sum(~np.isnan(result[lr])))

    w_sec = w_samp / C.FS_TGT
    X, y, source, f_amp = [], [], [], []

    def _prep(trial_raw):
        """CAR (per car_mode) -> 8 ch -> downsample -> baseline. Returns (x8, t)."""
        tr = np.asarray(trial_raw, dtype=float)
        if tr.shape[0] != len(labels):
            tr = tr.T
        x = aplicar_car(tr, labels, mode=car_mode)   # -> (8, n)
        x = downsample(x)
        t = ds_time(x.shape[1])
        return baseline(x, t), t

    # ---- LH/RH: MI window of the valid LR trials ----
    keep_lr = lr & (triallen >= w_sec) & ~np.isnan(result)
    y_lr = np.where(target == 2, 0, 1)               # LH(2)->0, RH(1)->1
    for i in np.where(keep_lr)[0]:
        try:
            x, t = _prep(data[i])
            win = crop_mi(x, t, w_samp)
            if win.shape[1] != w_samp:
                continue
            X.append(win.astype(np.float32)); y.append(int(y_lr[i]))
            source.append('mi'); f_amp.append(flag_amplitud(win))
        except Exception:
            continue

    # ---- IDLE source A: pre-cue rest of the LR trials ----
    for i in np.where(lr)[0]:                         # all LR trials, no length requirement
        try:
            x, t = _prep(data[i])
            wins = crop_idle_windows(x, t, w_samp, C.IDLE_REGION_MS)
            for win in wins[:idle_por_trial]:
                if win.shape[1] != w_samp:
                    continue
                X.append(win.astype(np.float32)); y.append(C.IDLE_ID)
                source.append('precue'); f_amp.append(flag_amplitud(win))
        except Exception:
            continue

    # ---- IDLE source B: REST target (4), window at the same onset as MI ----
    if incluir_rest:
        keep_rest = (target == 4) & (triallen >= w_sec)
        for i in np.where(keep_rest)[0]:
            try:
                x, t = _prep(data[i])
                win = crop_mi(x, t, w_samp)          # same onset as the MI window
                if win.shape[1] != w_samp:
                    continue
                X.append(win.astype(np.float32)); y.append(C.IDLE_ID)
                source.append('rest'); f_amp.append(flag_amplitud(win))
            except Exception:
                continue

    del m, bci, data
    if not X:
        return None
    return dict(X=np.array(X, np.float32), y=np.array(y, np.int64),
                source=np.array(source), f_amp=np.array(f_amp, bool),
                hit=(n_hit, n_dec))


# ============================================================
# Per-session class balancing
# ============================================================
def balancear(d: dict, rng: np.random.RandomState, idle_ratio: float = 1.0) -> dict:
    """Subsample IDLE windows so that they do not dominate.

    idle_ratio = desired n_idle / (n_lh + n_rh). With 1.0, IDLE keeps about as many
    windows as LH and RH together. The kept IDLE windows are drawn uniformly at random
    from both sources (precue and rest).
    """
    y = d['y']
    n_mi = int((y != C.IDLE_ID).sum())
    idle_idx = np.where(y == C.IDLE_ID)[0]
    n_keep = min(len(idle_idx), int(round(idle_ratio * n_mi)))
    if len(idle_idx) > n_keep:
        keep = rng.choice(idle_idx, size=n_keep, replace=False)
        mask = np.ones(len(y), bool); mask[idle_idx] = False; mask[keep] = True
    else:
        mask = np.ones(len(y), bool)
    return {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in d.items()}


# ============================================================
# Full cache
# ============================================================
def construir_cache_3clase(smoke: bool = False, incluir_rest: bool = True,
                           idle_ratio: float = 1.0, out_path: Path | None = None) -> dict:
    """Walk the .mat files, extract LH/RH/IDLE windows and save a reusable npz.

    smoke=True -> subjects 1-4 only (quick check). Returns the arrays and the per-subject
    online hit rate.
    """
    C.set_seed()
    rng = np.random.RandomState(C.SEED)
    files = sorted(glob.glob(str(C.DATA / 'S*_Session_*.mat')))
    files = [f for f in files if parse_name(f)[1] in C.SESSIONS]
    if smoke:
        files = [f for f in files if parse_name(f)[0] in (1, 2, 3, 4)
                 and parse_name(f)[1] in (5, 6)]
    if out_path is None:
        out_path = (C.CACHED / 'trials_cyton8_3clase_smoke.npz') if smoke else C.NPZ_3CLASE

    log('=' * 72)
    log(f'3-class cache build (LH/RH/IDLE) {"[SMOKE]" if smoke else ""}')
    log(f'  files={len(files)}  W={C.W_SEC}s ({C.W_SAMP} samp)  CAR={C.CAR_MODE}  '
        f'rest={incluir_rest}  idle_ratio={idle_ratio}')
    log('=' * 72)

    Xs, ys, subs, sess, srcs, famps = [], [], [], [], [], []
    hits: dict[int, list[int]] = {}
    t0 = time.time()
    for k, fp in enumerate(files, 1):
        sid, ses = parse_name(fp)
        d = preprocesar_sesion(fp, incluir_rest=incluir_rest)
        if d is None:
            log(f'  [{k:3d}/{len(files)}] S{sid}_Ses{ses}: EMPTY'); continue
        d = balancear(d, rng, idle_ratio)
        n = len(d['y'])
        Xs.append(d['X']); ys.append(d['y'])
        subs.append(np.full(n, sid, np.int16)); sess.append(np.full(n, ses, np.int16))
        srcs.append(d['source']); famps.append(d['f_amp'])
        hh = hits.setdefault(sid, [0, 0]); hh[0] += d['hit'][0]; hh[1] += d['hit'][1]
        if k % 10 == 0 or smoke:
            yy = d['y']
            eta = (time.time() - t0) / k * (len(files) - k)
            log(f'  [{k:3d}/{len(files)}] S{sid}_Ses{ses}: '
                f'LH={int((yy==0).sum())} RH={int((yy==1).sum())} IDLE={int((yy==2).sum())} '
                f'({time.time()-t0:.0f}s eta {eta:.0f}s)')

    X = np.concatenate(Xs); y = np.concatenate(ys)
    subject = np.concatenate(subs); session = np.concatenate(sess)
    source = np.concatenate(srcs); f_amp = np.concatenate(famps)
    hit_rate = {int(s): (v[0] / v[1] if v[1] else np.nan) for s, v in hits.items()}

    C.CACHED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, X=X, y=y, subject=subject, session=session, source=source,
        f_amp=f_amp, channels=np.array(C.CYTON_8),
        hit_subjects=np.array(sorted(hit_rate)),
        hit_values=np.array([hit_rate[s] for s in sorted(hit_rate)]),
        meta=np.array([f'W={C.W_SEC}s', f'fs={C.FS_TGT}', f'CAR={C.CAR_MODE}',
                       'clases=LH/RH/IDLE', 'idle=precue+rest']))
    log(f'\nSaved {out_path.name}: X {X.shape}  '
        f'LH={int((y==0).sum())} RH={int((y==1).sum())} IDLE={int((y==2).sum())}  '
        f'subjects={len(hit_rate)}  in {time.time()-t0:.0f}s')
    return dict(path=out_path, X=X, y=y, subject=subject, session=session,
                source=source, f_amp=f_amp, hit_rate=hit_rate)


# ============================================================
# Loading and cross-subject split
# ============================================================
def load_cache(smoke: bool = False, path: Path | None = None) -> dict:
    """Load the npz written by construir_cache_3clase()."""
    if path is None:
        path = (C.CACHED / 'trials_cyton8_3clase_smoke.npz') if smoke else C.NPZ_3CLASE
    z = np.load(path, allow_pickle=True)
    hit_rate = {int(s): float(v) for s, v in zip(z['hit_subjects'], z['hit_values'])}
    return dict(X=z['X'], y=z['y'], subject=z['subject'], session=z['session'],
                source=z['source'], f_amp=z['f_amp'], hit_rate=hit_rate)


def split_cross_subject(subject: np.ndarray, hit_rate: dict, n_test: int = C.N_TEST) -> dict:
    """Held-out subject split stratified by online hit rate.

    Subjects are sorted by hit rate and up to n_test of them (capped at
    max(2, n_subjects // 3)) are taken at evenly spaced ranks, so the test set spans the
    whole skill range. Test subjects are never in `train`. `illit` lists the subjects with
    hit rate < ILLIT_THR.
    """
    subs = sorted(set(int(s) for s in subject))
    by_skill = sorted(subs, key=lambda s: (hit_rate.get(s, 0.5), s))
    n_test = min(n_test, max(2, len(subs) // 3))
    pick = np.unique(np.linspace(0, len(by_skill) - 1, n_test).round().astype(int))
    test = sorted(by_skill[i] for i in pick)
    train = [s for s in subs if s not in test]
    illit = sorted(s for s in subs if hit_rate.get(s, 1.0) < C.ILLIT_THR)
    return dict(train=train, test=test, illit=illit, all=subs)
