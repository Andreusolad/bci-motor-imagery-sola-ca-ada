"""Stage 1 of 3 of the per-intent evaluation: cache the continuous signal of each trial.

The window cache of build_dataset.py holds one window per trial and source, so it has no
time sequence. The online system accumulates evidence over consecutive windows before it
emits a command, so the per-intent evaluation needs dense sliding windows (125 ms stride).
Reading the .mat files is the slow step, so this script reads them once and stores the
signal of every trial after CAR over the 8 channels and downsampling to 250 Hz. Windowing,
band-pass and z-score are applied in stage 2 (infer_streams.py), which can then run for
several checkpoints without reading the .mat files again.

Trial time line (at 1000 Hz, nsamp = 5001 + triallength*1000, so t spans
[-2000, 3000 + triallength*1000] ms):
    [-2000, 0)                          pre-cue rest (REST2; 2.0 s)
    [0, 2000)                           cue visible, cursor not yet shown
    [2000, 2000 + triallength*1000)     motor imagery / sustained cued rest
    [2000 + triallength*1000, end]      post-trial (result on screen); cached but not
                                        used by accumulate.py

Cached trials (validation subjects of ../split_80_20_subjects.json, sessions 5 and 6):
  - LR trials (tasknumber 1, target 1 or 2) -> LH / RH intents. All of them are kept,
    including trials without an online result (timeouts); the flag `mi_ok` reproduces the
    trial selection of build_dataset.py (triallength >= 2 s and a non-NaN result).
  - REST1 trials (target 4, tasknumber 2 or 3) -> sustained cued rest, the only rest
    segment in the dataset long enough for the accumulator to emit a command.

Writes cache/streams_val.npz (cache/streams_val_smoke.npz with --smoke, or --out): `sig`
(8, total samples) float64 with per-trial `offset`, the per-trial fields subject,
session, trial_idx, kind, y4, triallen, result and mi_ok, and the online hit rate per
subject (hit_subjects / hit_values).

Usage:
    python prep_streams.py                 # 12 validation subjects, sessions 5 and 6
    python prep_streams.py --smoke         # first 2 validation subjects, quick check
"""
from __future__ import annotations
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
from pymatreader import read_mat

HERE = Path(__file__).resolve().parent

from rt_system import config as C                                      # noqa: E402
from rt_system.preprocessing import aplicar_car, downsample, ds_time   # noqa: E402

from build_dataset import parse_name, LH, RH, REST1                    # noqa: E402
from run_experiments import split_canonico                             # noqa: E402

OUTDIR = HERE / 'cache'


def preparar_sesion(mat_path, car_mode: str = C.CAR_MODE) -> dict | None:
    """Continuous signals (8, n) of the LR and REST1 trials of one session."""
    m = read_mat(str(mat_path))
    bci = m['BCI']
    labels = list(bci['chaninfo']['label'])
    data = bci['data']
    td = bci['TrialData']
    tasknum = np.asarray(td['tasknumber']).flatten().astype(int)
    target = np.asarray(td['targetnumber']).flatten().astype(int)
    result = np.asarray(td['result'], dtype=float).flatten()
    triallen = np.asarray(td['triallength'], dtype=float).flatten()

    lr = (tasknum == 1) & np.isin(target, [1, 2])
    r1 = (target == 4) & np.isin(tasknum, [2, 3])

    # online hit rate, computed as in build_dataset.py
    n_hit = int(np.nansum(result[lr] == 1))
    n_dec = int(np.sum(~np.isnan(result[lr])))

    # trial selection of build_dataset.py at W = 2 s: long enough and an online decision
    w_sec = C.W_SAMP / C.FS_TGT
    mi_ok_mask = lr & (triallen >= w_sec) & ~np.isnan(result)

    sigs, meta = [], []
    for i in np.where(lr | r1)[0]:
        tr = np.asarray(data[i], dtype=float)
        if tr.shape[0] != len(labels):
            tr = tr.T
        x = aplicar_car(tr, labels, mode=car_mode)        # -> (8, n) at 1000 Hz
        # Kept in float64: build_dataset.py casts to float32 only after centering each
        # window, and the same order is needed to reproduce its windows exactly.
        x = downsample(x)                                 # -> (8, n) at 250 Hz, float64
        sigs.append(x)
        # y4: LH (target 2) = 0, RH (target 1) = 1, REST1 = 2
        if lr[i]:
            y4 = LH if target[i] == 2 else RH
            kind = 'mi'
        else:
            y4 = REST1
            kind = 'rest1'
        meta.append(dict(trial_idx=int(i), kind=kind, y4=int(y4),
                         triallen=float(triallen[i]),
                         result=float(result[i]),
                         mi_ok=bool(mi_ok_mask[i]),
                         n=int(x.shape[1])))

    del m, bci, data
    if not sigs:
        return None
    return dict(sigs=sigs, meta=meta, hit=(n_hit, n_dec))


def construir(smoke: bool = False, out_name: str | None = None) -> Path:
    _, val_subs = split_canonico()
    if smoke:
        val_subs = val_subs[:2]

    files = sorted(glob.glob(str(C.DATA / 'S*_Session_*.mat')))
    files = [f for f in files
             if parse_name(f)[0] in set(val_subs) and parse_name(f)[1] in C.SESSIONS]

    out_name = out_name or f'streams_val{"_smoke" if smoke else ""}.npz'
    out_path = OUTDIR / out_name
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print('=' * 76)
    print(f'Continuous stream cache{" [SMOKE]" if smoke else ""}')
    print(f'  validation subjects: {val_subs}')
    print(f'  files: {len(files)}   CAR={C.CAR_MODE}   fs={C.FS_TGT} Hz')
    print('=' * 76, flush=True)

    todo_sig, filas = [], []
    hits: dict[int, list[int]] = {}
    t0 = time.time()
    for k, fp in enumerate(files, 1):
        sid, ses = parse_name(fp)
        try:
            d = preparar_sesion(fp)
        except Exception as e:
            print(f'  [{k:2d}/{len(files)}] S{sid}_Ses{ses}: ERROR {e}', flush=True)
            continue
        if d is None:
            print(f'  [{k:2d}/{len(files)}] S{sid}_Ses{ses}: EMPTY', flush=True)
            continue
        for x, mt in zip(d['sigs'], d['meta']):
            todo_sig.append(x)
            filas.append(dict(subject=sid, session=ses, **mt))
        hh = hits.setdefault(sid, [0, 0]); hh[0] += d['hit'][0]; hh[1] += d['hit'][1]
        n_mi = sum(1 for m in d['meta'] if m['kind'] == 'mi')
        n_r1 = sum(1 for m in d['meta'] if m['kind'] == 'rest1')
        el = time.time() - t0
        print(f'  [{k:2d}/{len(files)}] S{sid}_Ses{ses}: LR={n_mi} '
              f'(mi_ok={sum(1 for m in d["meta"] if m["mi_ok"])}) REST1={n_r1} '
              f'({el:.0f}s eta {el/k*(len(files)-k):.0f}s)', flush=True)

    if not todo_sig:
        raise SystemExit('no trial was extracted')

    # trials have different lengths: concatenate along time and keep start offsets
    largos = np.array([x.shape[1] for x in todo_sig], np.int64)
    offset = np.concatenate([[0], np.cumsum(largos)]).astype(np.int64)
    sig = np.concatenate(todo_sig, axis=1).astype(np.float64)          # (8, total)

    def col(k, dt):
        return np.array([f[k] for f in filas], dt)

    hit_rate = {int(s): (v[0] / v[1] if v[1] else np.nan) for s, v in hits.items()}
    np.savez(
        out_path, sig=sig, offset=offset,
        subject=col('subject', np.int16), session=col('session', np.int16),
        trial_idx=col('trial_idx', np.int32), kind=col('kind', '<U6'),
        y4=col('y4', np.int64), triallen=col('triallen', np.float32),
        result=col('result', np.float32), mi_ok=col('mi_ok', bool),
        channels=np.array(C.CYTON_8),
        hit_subjects=np.array(sorted(hit_rate)),
        hit_values=np.array([hit_rate[s] for s in sorted(hit_rate)]),
        meta=np.array([f'fs={C.FS_TGT}', f'CAR={C.CAR_MODE}', f't0_ms={C.T0_MS}',
                       f'mi_onset_ms={C.MI_ONSET_MS}', 'no baseline (centred per window)',
                       f'sessions={C.SESSIONS}', f'val_subs={val_subs}']))

    n_mi = int((col('kind', '<U6') == 'mi').sum())
    n_r1 = int((col('kind', '<U6') == 'rest1').sum())
    seg_mi = float(largos[col('kind', '<U6') == 'mi'].sum() / C.FS_TGT)
    seg_r1 = float(largos[col('kind', '<U6') == 'rest1'].sum() / C.FS_TGT)
    print(f'\nSaved {out_path.name}  sig{sig.shape}  '
          f'({sig.nbytes/1e6:.0f} MB in memory)')
    print(f'    LR trials    = {n_mi:5d}  (mi_ok={int(col("mi_ok", bool).sum())})  '
          f'{seg_mi/60:.1f} min of signal')
    print(f'    REST1 trials = {n_r1:5d}                {seg_r1/60:.1f} min of signal')
    print(f'    subjects={len(hit_rate)}   {time.time()-t0:.0f}s')
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smoke', action='store_true', help='first 2 validation subjects')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    construir(smoke=a.smoke, out_name=a.out)


if __name__ == '__main__':
    main()
