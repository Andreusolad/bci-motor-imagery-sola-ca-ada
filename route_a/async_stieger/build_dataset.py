"""Build the four-class window cache (LH / RH / REST1 / REST2) of the asynchronous study.

Reads the Stieger et al. (2021) .mat files of sessions 5 and 6 (C.SESSIONS) from
$BCI_DATA/stieger2021, applies the rt_system preprocessing (8 Cyton channels, CAR over
the 8 channels by default, polyphase downsampling to 250 Hz) and cuts W-second windows
from three sources, stored in `source`:

  'mi'     motor imagery: one window per valid LR trial (tasknumber 1, target 1 = RH,
           2 = LH) from the MI onset (C.MI_ONSET_MS = 2000 ms).
  'rest1'  cued rest (target 4, "down"; tasks 2 and 3): one window from the same onset.
  'rest2'  pre-cue rest: windows inside C.IDLE_REGION_MS = [-2000, 0) ms of the trials
           of the tasks in --rest2-tasks (task 1 only by default).

Per window the cache also stores `tasknumber` (so rest1 can be analyzed per task block),
a globally unique `trial_id` (for the leakage check and per-trial grouping), the
amplitude-artifact flag `f_amp`, and `mi_valido`: True when the window comes from an LR
trial that also yields an MI window (triallength >= W and a non-NaN online result). MI
windows come only from such trials while rest2 windows come from all task-1 trials; the
flag allows equalizing that selection.

Baseline policy (--baseline):
  per_window    (default) subtract the mean of each window. Identical for all sources,
                and the only option available to a sliding online system.
  none          no baseline; the band-pass removes the DC offset.
  trial_precue  subtract the mean of C.BASE_MS = [-1000, 0) ms from the whole trial.
                That interval lies inside the rest2 window, whose second half then has
                zero mean by construction, a marker a classifier can exploit (a 0.5 Hz
                high-pass does not remove an offset within a 2 s window). Kept only to
                quantify that confound.

The cache is band-agnostic: band-pass filtering is applied at training time
(run_experiments.py).

Writes cache/trials_mc4_W{W}_{baseline}[_smoke].npz (or --out) with X (n, 8, W*250)
float32, y, subject, session, source, tasknumber, trial_id, f_amp, mi_valido, channels,
hit_subjects / hit_values (online LR hit rate per subject) and meta.

Usage:
    python build_dataset.py --smoke          # subjects S1-S4, quick check
    python build_dataset.py --W 2.0
    python build_dataset.py --W 1.0 --out trials_mc_W1.npz
"""
from __future__ import annotations
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
from pymatreader import read_mat

from rt_system import config as C                                    # noqa: E402
from rt_system.preprocessing import (aplicar_car, downsample, ds_time,  # noqa: E402
                                     crop_mi, crop_idle_windows, flag_amplitud)

OUTDIR = Path(__file__).resolve().parent / 'cache'

# ---- class ids ---------------------------------------------------------------
# Four-class ids; formulations A and B remap them (see run_experiments.py).
LH, RH, REST1, REST2 = 0, 1, 2, 3
CLASES4 = ['left_hand', 'right_hand', 'rest1_cued', 'rest2_precue']
SOURCES = ['mi', 'rest1', 'rest2']


# ============================================================
# File names
# ============================================================
def parse_name(fp: str) -> tuple[int, int]:
    stem = Path(fp).stem                      # 'S4_Session_5'
    return int(stem.split('_')[0][1:]), int(stem.split('_')[-1])


# ============================================================
# Baseline policy
# ============================================================
def aplicar_baseline(x: np.ndarray, t: np.ndarray, policy: str) -> np.ndarray:
    """Trial-level baseline, applied before cropping (see the module docstring).

    'trial_precue' introduces the rest2 marker. 'none' and 'per_window' leave the trial
    unchanged (per_window acts on each window in centrar_ventana).
    """
    if policy == 'trial_precue':
        m = (t >= C.BASE_MS[0]) & (t < C.BASE_MS[1])
        return x - x[:, m].mean(axis=1, keepdims=True)
    return x


def centrar_ventana(win: np.ndarray, policy: str) -> np.ndarray:
    """Window-level baseline, identical for all sources."""
    if policy == 'per_window':
        return win - win.mean(axis=1, keepdims=True)
    return win


# ============================================================
# One session -> labelled windows
# ============================================================
def preprocesar_sesion(mat_path, w_samp: int, baseline_policy: str,
                       car_mode: str = C.CAR_MODE,
                       idle_region: tuple = C.IDLE_REGION_MS,
                       rest2_tasks: tuple = (1,), solo_rest2: bool = False) -> dict | None:
    """Windows (n, 8, w_samp) and metadata of one session, with the two rest classes apart.

    Returns X, y, source, tasknumber, trial_idx, mi_valido, f_amp and hit = (n_hit, n_dec).
    """
    m = read_mat(str(mat_path))
    bci = m['BCI']
    labels = list(bci['chaninfo']['label'])
    data = bci['data']
    td = bci['TrialData']
    tasknum = np.asarray(td['tasknumber']).flatten().astype(int)
    target = np.asarray(td['targetnumber']).flatten().astype(int)
    result = np.asarray(td['result'], dtype=float).flatten()
    triallen = np.asarray(td['triallength'], dtype=float).flatten()

    # online hit rate over all LR trials (used to stratify subjects by skill)
    lr = (tasknum == 1) & np.isin(target, [1, 2])
    n_hit = int(np.nansum(result[lr] == 1))
    n_dec = int(np.sum(~np.isnan(result[lr])))

    w_sec = w_samp / C.FS_TGT
    X, y, source, tasks, tidx, f_amp, mival = [], [], [], [], [], [], []

    # LR trials that yield an MI window (long enough and with an online decision).
    # rest2 is taken from all task-1 trials, so its windows are flagged with mi_valido.
    keep_lr = lr & (triallen >= w_sec) & ~np.isnan(result)
    set_mi_ok = set(np.where(keep_lr)[0].tolist())

    def _prep(trial_raw):
        tr = np.asarray(trial_raw, dtype=float)
        if tr.shape[0] != len(labels):
            tr = tr.T
        x = aplicar_car(tr, labels, mode=car_mode)      # -> (8, n)
        x = downsample(x)
        t = ds_time(x.shape[1])
        return aplicar_baseline(x, t, baseline_policy), t

    def _push(win, lab, src, i):
        win = centrar_ventana(win, baseline_policy)
        X.append(win.astype(np.float32)); y.append(int(lab))
        source.append(src); tasks.append(int(tasknum[i])); tidx.append(int(i))
        f_amp.append(flag_amplitud(win)); mival.append(i in set_mi_ok)

    # ---- LH / RH: MI window of the valid LR trials ----
    y_lr = np.where(target == 2, LH, RH)                # target 2 -> LH (0), 1 -> RH (1)
    for i in ([] if solo_rest2 else np.where(keep_lr)[0]):
        try:
            x, t = _prep(data[i])
            win = crop_mi(x, t, w_samp)
            if win.shape[1] != w_samp:
                continue
            _push(win, y_lr[i], 'mi', i)
        except Exception:
            continue

    # ---- REST1: cued rest (target 4, "down"), window from the MI onset ----
    keep_r1 = (target == 4) & (triallen >= w_sec)
    for i in ([] if solo_rest2 else np.where(keep_r1)[0]):
        try:
            x, t = _prep(data[i])
            win = crop_mi(x, t, w_samp)                 # same onset as MI
            if win.shape[1] != w_samp:
                continue
            _push(win, REST1, 'rest1', i)
        except Exception:
            continue

    # ---- REST2: pre-cue rest. `rest2_tasks` selects the tasks whose pre-cue interval
    # is used: (1,) (default) uses task 1 only; (1, 2, 3) uses all three tasks.
    m_r2 = np.isin(tasknum, list(rest2_tasks))
    for i in np.where(m_r2)[0]:
        try:
            x, t = _prep(data[i])
            for win in crop_idle_windows(x, t, w_samp, idle_region):
                if win.shape[1] != w_samp:
                    continue
                _push(win, REST2, 'rest2', i)
        except Exception:
            continue

    del m, bci, data
    if not X:
        return None
    return dict(X=np.array(X, np.float32), y=np.array(y, np.int64),
                source=np.array(source), tasknumber=np.array(tasks, np.int16),
                trial_idx=np.array(tidx, np.int32), mi_valido=np.array(mival, bool),
                f_amp=np.array(f_amp, bool), hit=(n_hit, n_dec))


# ============================================================
# Cache construction
# ============================================================
def construir(W: float = C.W_SEC, baseline_policy: str = 'per_window',
              sessions: list | None = None, smoke: bool = False,
              out_name: str | None = None, car_mode: str = C.CAR_MODE,
              rest2_tasks: tuple = (1,), solo_rest2: bool = False) -> Path:
    C.set_seed()
    w_samp = int(round(W * C.FS_TGT))
    sessions = sessions or C.SESSIONS

    # The pre-cue region lasts 2.0 s. With the 50% hop of crop_idle_windows, W = 1 s
    # gives 3 rest2 windows per trial, W = 2 s gives 1 and W > 2 s gives none.
    n_r2 = len(crop_idle_windows(np.zeros((8, 3000)), ds_time(3000), w_samp,
                                 C.IDLE_REGION_MS))
    if n_r2 == 0:
        raise SystemExit(
            f'W={W}s leaves no rest2 window: the pre-cue region '
            f'{C.IDLE_REGION_MS} lasts {(C.IDLE_REGION_MS[1]-C.IDLE_REGION_MS[0])/1000:.1f}s. '
            f'REST2 requires W<=2.0s in Stieger.')

    files = sorted(glob.glob(str(C.DATA / 'S*_Session_*.mat')))
    files = [f for f in files if parse_name(f)[1] in sessions]
    if smoke:
        files = [f for f in files if parse_name(f)[0] in (1, 2, 3, 4)]

    if out_name is None:
        out_name = (f'trials_mc4_W{W:g}_{baseline_policy}'
                    f'{"_smoke" if smoke else ""}.npz')
    out_path = OUTDIR / out_name
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print('=' * 74)
    print(f'Four-class window cache (LH/RH/REST1/REST2){" [SMOKE]" if smoke else ""}')
    print(f'  files={len(files)}  W={W}s ({w_samp} samp)  CAR={car_mode}')
    print(f'  baseline={baseline_policy}   rest2 windows/trial={n_r2}')
    print(f'  sessions={sessions}')
    print('=' * 74, flush=True)

    Xs, ys, subs, sess, srcs, tsks, tids, famps, mivs = [], [], [], [], [], [], [], [], []
    hits: dict[int, list[int]] = {}
    t0 = time.time()
    for k, fp in enumerate(files, 1):
        sid, ses = parse_name(fp)
        try:
            d = preprocesar_sesion(fp, w_samp, baseline_policy, car_mode=car_mode,
                                   rest2_tasks=rest2_tasks, solo_rest2=solo_rest2)
        except Exception as e:
            print(f'  [{k:3d}/{len(files)}] S{sid}_Ses{ses}: ERROR {e}', flush=True)
            continue
        if d is None:
            print(f'  [{k:3d}/{len(files)}] S{sid}_Ses{ses}: EMPTY', flush=True)
            continue
        n = len(d['y'])
        Xs.append(d['X']); ys.append(d['y'])
        subs.append(np.full(n, sid, np.int16)); sess.append(np.full(n, ses, np.int16))
        srcs.append(d['source']); tsks.append(d['tasknumber']); famps.append(d['f_amp'])
        mivs.append(d['mi_valido'])
        # globally unique trial_id: subject / session / trial index
        tids.append(sid * 1_000_000 + ses * 10_000 + d['trial_idx'])
        hh = hits.setdefault(sid, [0, 0]); hh[0] += d['hit'][0]; hh[1] += d['hit'][1]
        if k % 10 == 0 or smoke or k == len(files):
            yy = d['y']; el = time.time() - t0
            print(f'  [{k:3d}/{len(files)}] S{sid}_Ses{ses}: '
                  f'LH={int((yy==LH).sum())} RH={int((yy==RH).sum())} '
                  f'R1={int((yy==REST1).sum())} R2={int((yy==REST2).sum())} '
                  f'({el:.0f}s eta {el/k*(len(files)-k):.0f}s)', flush=True)

    if not Xs:
        raise SystemExit('no window was extracted')

    X = np.concatenate(Xs); y = np.concatenate(ys)
    subject = np.concatenate(subs); session = np.concatenate(sess)
    source = np.concatenate(srcs); tasknumber = np.concatenate(tsks)
    trial_id = np.concatenate(tids); f_amp = np.concatenate(famps)
    mi_valido = np.concatenate(mivs)
    hit_rate = {int(s): (v[0] / v[1] if v[1] else np.nan) for s, v in hits.items()}

    # ---- integrity checks ----
    verificar(X, y, subject, source, trial_id, tasknumber, mi_valido,
              baseline_policy, w_samp, rest2_tasks=rest2_tasks)

    np.savez_compressed(
        out_path, X=X, y=y, subject=subject, session=session, source=source,
        tasknumber=tasknumber, trial_id=trial_id, f_amp=f_amp, mi_valido=mi_valido,
        channels=np.array(C.CYTON_8),
        hit_subjects=np.array(sorted(hit_rate)),
        hit_values=np.array([hit_rate[s] for s in sorted(hit_rate)]),
        meta=np.array([f'W={W}', f'fs={C.FS_TGT}', f'CAR={car_mode}',
                       f'baseline={baseline_policy}',
                       'clases=LH/RH/REST1/REST2', f'sessions={sessions}']))

    print(f'\nSaved {out_path.name}  X{X.shape}')
    for cid, nm in zip([LH, RH, REST1, REST2], CLASES4):
        print(f'    {nm:16s} n={int((y==cid).sum()):7,d}')
    print(f'    subjects={len(hit_rate)}  unique trials={len(set(trial_id.tolist())):,d}  '
          f'{time.time()-t0:.0f}s')
    return out_path


# ============================================================
# Checks: trial/subject leakage and baseline marker
# ============================================================
def verificar(X, y, subject, source, trial_id, tasknumber, mi_valido,
              policy, w_samp, rest2_tasks: tuple = (1,)) -> None:
    print('\n--- checks ---', flush=True)

    # 1. each trial belongs to one subject, so a subject split never splits a trial
    df = {}
    for t, s in zip(trial_id, subject):
        if t in df:
            assert df[t] == s, f'trial {t} in two subjects ({df[t]},{s})'
        else:
            df[t] = s
    print(f'  [OK] {len(df):,d} unique trial_id, each from a single subject '
          f'-> a subject split never splits a trial')

    # 2. mi and rest2 windows of the same trial share trial_id
    both = 0
    by_t: dict[int, set] = {}
    for t, s in zip(trial_id, source):
        by_t.setdefault(t, set()).add(s)
    both = sum(1 for v in by_t.values() if {'mi', 'rest2'} <= v)
    print(f'  [OK] {both:,d} trials contribute both MI and REST2 -> same trial_id, '
          f'always in the same partition')

    # 3. mi only from task 1, rest2 only from rest2_tasks, rest1 expected from tasks 2/3
    t_r1 = set(tasknumber[source == 'rest1'].tolist())
    t_mi = set(tasknumber[source == 'mi'].tolist())
    t_r2 = set(tasknumber[source == 'rest2'].tolist())
    print(f'  [--] tasknumber per source: mi={sorted(t_mi)} rest2={sorted(t_r2)} '
          f'rest1={sorted(t_r1)}')
    assert t_mi <= {1}, 'mi should come from task 1 only'
    assert t_r2 <= set(rest2_tasks), (f'rest2 comes from tasks {sorted(t_r2)} but '
                                      f'{sorted(rest2_tasks)} were requested')
    if t_r1 & {1}:
        print('  [WARN] rest1 contains task 1 (unexpected)')

    # 4. selection asymmetry between MI and REST2
    m2 = source == 'rest2'
    if m2.any():
        frac = float((~mi_valido[m2]).mean())
        print(f'  [--] REST2 from trials without an MI window: {frac*100:.1f}% '
              f'-> use mi_valido to equalize the selection')

    # 5. baseline marker: mean absolute channel mean over the second half of the window,
    # per source; the ratio between sources must stay below 100 unless trial_precue
    if w_samp >= 250:
        half = slice(w_samp // 2, w_samp)
        am = {s: float(np.abs(X[source == s][:, :, half].mean(axis=2)).mean())
              for s in SOURCES if (source == s).any()}
        print(f'  [--] |mean of 2nd half| per source: ' +
              '  '.join(f'{k}={v:.3e}' for k, v in am.items()))
        vals = list(am.values())
        ratio = max(vals) / max(min(vals), 1e-30)
        if policy == 'trial_precue':
            print(f'  [WARN] ratio={ratio:.1e} with trial_precue: the marker is present '
                  f'by design (only to quantify the confound)')
        else:
            assert ratio < 100, (f'baseline marker present (ratio={ratio:.1e}) '
                                 f'with policy {policy}')
            print(f'  [OK] ratio={ratio:.1f}x (<100) -> no deterministic marker')


# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--W', type=float, default=2.0,
                    help='window length in seconds (REST2 requires W<=2.0 in Stieger)')
    ap.add_argument('--baseline', default='per_window',
                    choices=['per_window', 'none', 'trial_precue'],
                    help='baseline policy; per_window avoids the rest2 marker')
    ap.add_argument('--sessions', type=int, nargs='+', default=None)
    ap.add_argument('--car', default=C.CAR_MODE, choices=['car8', 'car62', 'none'])
    ap.add_argument('--smoke', action='store_true', help='subjects S1-S4, quick check')
    ap.add_argument('--out', default=None, help='output npz file name')
    ap.add_argument('--rest2-tasks', type=int, nargs='+', default=[1],
                    help='tasks whose pre-cue interval gives REST2 windows: 1 (default) '
                         'or 1 2 3 (all tasks)')
    ap.add_argument('--solo-rest2', action='store_true',
                    help='extract REST2 windows only (to extend an existing cache '
                         'without duplicating MI/REST1)')
    a = ap.parse_args()
    construir(W=a.W, baseline_policy=a.baseline, sessions=a.sessions,
              smoke=a.smoke, out_name=a.out, car_mode=a.car,
              rest2_tasks=tuple(a.rest2_tasks), solo_rest2=a.solo_rest2)


if __name__ == '__main__':
    main()
