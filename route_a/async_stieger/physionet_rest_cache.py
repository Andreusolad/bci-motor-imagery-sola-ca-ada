"""Cache of passive rest, motor imagery and within-task rest from Physionet EEGMMIDB.

The Stieger dataset has no resting-state runs, so passive rest is taken from Physionet
EEGMMIDB. There all segment types come from the same subject, session and amplifier:

    kind      source                                              y
    mi        motor imagery, runs 4/8/12 (T1 = left fist,         0 = LH, 1 = RH
              T2 = right fist)
    rest_t0   T0 rest between MI trials (task context)            -1
    rest_eo   run R01, 60 s of continuous rest, eyes open         -1
    rest_ec   run R02, 60 s of continuous rest, eyes closed       -1

Preprocessing follows build_dataset.py: pick the 8 Cyton channels by name, convert to uV,
common average reference over those 8 channels (C.CAR_MODE = 'car8', the reference used by
the checkpoints and reproducible on the Cyton), resample 160 -> 250 Hz with the same
anti-aliased polyphase resampler. The signal is stored continuous; per-window centring,
band-pass and z-score are applied at evaluation time (as in infer_streams.py), so the cache
does not depend on the frequency band. MI and T0 segments shorter than one window
(C.W_SAMP) are dropped. Runs not sampled at 160 Hz or missing a channel are skipped.

Reads:  $BCI_DATA/physionet/eegmmidb-1.0.0/S<nnn>/S<nnn>R<rr>.edf
Writes: cache/pn_rest.npz (cache/pn_rest_smoke.npz with --smoke; --out sets another name
        inside cache/). Arrays: sig (8, total samples) float32, offset (n_segments + 1),
        and per segment subject, run, kind, y, onset_s; plus channels and meta.

Usage:
    python physionet_rest_cache.py --smoke      # first 5 subjects
    python physionet_rest_cache.py              # all subjects except PN_EXCLUDE
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from rt_system import config as C                                    # noqa: E402
from rt_system.preprocessing import downsample                       # noqa: E402

DATA = Path(os.environ.get('BCI_DATA', HERE.parents[1] / 'data'))
PN_ROOT = DATA / 'physionet' / 'eegmmidb-1.0.0'
PN_EXCLUDE = {'S088', 'S092', 'S100'}       # recorded at 128 Hz instead of 160 Hz
CACHE = HERE / 'cache'
CYTON = list(C.CYTON_8)
RUNS_MI = [4, 8, 12]                        # imagined left (T1) / right (T2) fist
FS_SRC_PN = 160


def norm(c: str) -> str:
    return c.replace('.', '').strip().upper()


def leer_run(f: Path):
    """EDF -> ((8, n) at 250 Hz in uV with CAR8, annotations, source fs).

    Returns (None, None, fs) if the run is not at 160 Hz or lacks one of the 8 channels.
    """
    import mne
    raw = mne.io.read_raw_edf(f, preload=True, verbose='ERROR')
    fs = float(raw.info['sfreq'])
    if abs(fs - FS_SRC_PN) > 1e-6:
        return None, None, fs
    ch = [norm(c) for c in raw.ch_names]
    if not all(c in ch for c in CYTON):
        return None, None, fs
    X = raw.get_data()[[ch.index(c) for c in CYTON], :] * 1e6      # V -> uV
    X = X - X.mean(axis=0, keepdims=True)                           # CAR over the 8 channels
    X = downsample(X, fs_src=FS_SRC_PN, fs_tgt=C.FS_TGT)            # 160 -> 250 Hz, anti-alias
    return X.astype(np.float64), raw.annotations, fs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    import mne
    mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')

    subs = sorted([d.name for d in PN_ROOT.glob('S[0-9][0-9][0-9]')
                   if d.is_dir() and d.name not in PN_EXCLUDE])
    if a.smoke:
        subs = subs[:5]

    out_name = a.out or ('pn_rest_smoke.npz' if a.smoke else 'pn_rest.npz')
    out_path = CACHE / out_name
    CACHE.mkdir(parents=True, exist_ok=True)

    print('=' * 78)
    print('Physionet cache: passive rest (R01/R02) + MI and within-task rest (runs 4/8/12)')
    print(f'  subjects={len(subs)}  CAR=car8  fs={C.FS_TGT} Hz  units=uV')
    print('=' * 78, flush=True)

    sigs, filas, saltados = [], [], []
    t0 = time.time()
    for k, sub in enumerate(subs, 1):
        d = PN_ROOT / sub
        num = int(sub[1:])
        n_seg0 = len(filas)
        # ---- continuous passive rest: one segment per run ----
        for r, kind in ((1, 'rest_eo'), (2, 'rest_ec')):
            f = d / f'{sub}R{r:02d}.edf'
            if not f.exists():
                saltados.append(f'{sub}R{r:02d}:missing'); continue
            X, _, fs = leer_run(f)
            if X is None:
                saltados.append(f'{sub}R{r:02d}:fs={fs}'); continue
            sigs.append(X)
            filas.append(dict(subject=num, run=r, kind=kind, y=-1,
                              onset_s=0.0, n=int(X.shape[1])))
        # ---- MI + within-task rest: one segment per annotation ----
        for r in RUNS_MI:
            f = d / f'{sub}R{r:02d}.edf'
            if not f.exists():
                saltados.append(f'{sub}R{r:02d}:missing'); continue
            X, ann, fs = leer_run(f)
            if X is None:
                saltados.append(f'{sub}R{r:02d}:fs={fs}'); continue
            for on, du, de in zip(ann.onset, ann.duration, ann.description):
                i0 = int(round(on * C.FS_TGT))
                i1 = int(round((on + du) * C.FS_TGT))
                i1 = min(i1, X.shape[1])
                if i1 - i0 < C.W_SAMP:            # segment shorter than one window
                    continue
                if de == 'T0':
                    kind, y = 'rest_t0', -1
                elif de == 'T1':
                    kind, y = 'mi', 0             # left fist -> LH (0, as in Stieger)
                elif de == 'T2':
                    kind, y = 'mi', 1             # right fist -> RH
                else:
                    continue
                sigs.append(X[:, i0:i1].copy())
                filas.append(dict(subject=num, run=r, kind=kind, y=y,
                                  onset_s=float(on), n=int(i1 - i0)))
        if k % 20 == 0 or a.smoke or k == len(subs):
            el = time.time() - t0
            print(f'  [{k:3d}/{len(subs)}] {sub}: +{len(filas)-n_seg0} segments '
                  f'(total {len(filas)})  ({el:.0f}s eta {el/k*(len(subs)-k):.0f}s)',
                  flush=True)

    largos = np.array([x.shape[1] for x in sigs], np.int64)
    offset = np.concatenate([[0], np.cumsum(largos)]).astype(np.int64)
    sig = np.concatenate(sigs, axis=1).astype(np.float32)

    def col(k, dt):
        return np.array([f[k] for f in filas], dt)

    kind = col('kind', '<U8')
    np.savez(out_path, sig=sig, offset=offset,
             subject=col('subject', np.int16), run=col('run', np.int16),
             kind=kind, y=col('y', np.int64), onset_s=col('onset_s', np.float32),
             channels=np.array(CYTON),
             meta=np.array([f'fs={C.FS_TGT}', 'CAR=car8', 'units=uV',
                            f'fs_src={FS_SRC_PN}', 'no baseline (centred per window)',
                            'R01=eyes open, R02=eyes closed (verified by alpha power)',
                            'T1=left fist->y=0(LH), T2=right fist->y=1(RH)',
                            f'excluded={sorted(PN_EXCLUDE)}']))

    print(f'\nSaved {out_path.name}  sig{sig.shape}  ({sig.nbytes/1e6:.0f} MB)')
    for kk in ['mi', 'rest_t0', 'rest_eo', 'rest_ec']:
        m = kind == kk
        print(f'    {kk:8s} segments={int(m.sum()):5d}  '
              f'{largos[m].sum()/C.FS_TGT/60:7.1f} min')
    print(f'    subjects={len(set(col("subject", int).tolist()))}  '
          f'skipped={len(saltados)}  {time.time()-t0:.0f}s')
    if saltados:
        print(f'    (skipped: {saltados[:8]}{"..." if len(saltados) > 8 else ""})')


if __name__ == '__main__':
    main()
