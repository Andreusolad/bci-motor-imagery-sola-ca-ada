"""Cue-locking control: does the command score follow the mental state or the cue?

REST2 is defined by its position relative to the cue (window [-2000, 0) ms), so
separating it from MI could reduce to detecting that the cue has not appeared yet.

A W-second window slides over the whole trial (step --stride-ms) and the trained model
is evaluated at every position, with no information about the cue onset. P(command) is
averaged per window centre (ms, cue at t = 0) over the trials of each session file and
then over the session files, and the position of the rise is measured:
  - a rise at t ~ 2000 ms (MI onset, when the cursor appears) follows the mental state;
  - a rise at t ~ 0 ms (cue onset) follows the cue: the 2 s before the MI onset, when
    the subject is not imagining yet, are already scored as a command.
`transicion` compares the mean score of the window centres in [-2000, 0), [0, 2000)
and [2000, 5000) ms; `frac_cue` is the fraction of the total rise that occurs between
the first two intervals, i.e. at the cue.

Needs a checkpoint saved by run_experiments.py (ckpt_*.pt). Uses the left/right trials
(tasknumber 1, triallength >= 4 s) of the validation subjects, sessions C.SESSIONS.
Writes outputs/cue_free_control_<checkpoint name>.json (or --out).

Usage:
    python cue_free_control.py --ckpt outputs/main/ckpt_C_mb_W2_r1_s42.pt
    python cue_free_control.py --ckpt outputs/main/ckpt_C_mb_W2_r1_s42.pt --smoke
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

from rt_system import config as C                                  # noqa: E402
from rt_system.models import build_model                           # noqa: E402
from rt_system.preprocessing import (aplicar_car, downsample, ds_time,  # noqa: E402
                                     bandpass)
from build_dataset import parse_name, LH, RH                       # noqa: E402
from run_experiments import BANDAS, split_canonico                 # noqa: E402

from pymatreader import read_mat                                   # noqa: E402


def perfil_temporal(mat_path, ckpt, w_samp, banda, stride_ms=125,
                    max_trials=None) -> dict:
    """Slide the window over each whole trial of one session file.

    Returns {window centre (ms): (mean P(command) over trials, n)}. The model is not
    told where the cue is: every position is evaluated in the same way.
    """
    m = read_mat(str(mat_path))
    bci = m['BCI']
    labels = list(bci['chaninfo']['label'])
    data = bci['data']
    td = bci['TrialData']
    tasknum = np.asarray(td['tasknumber']).flatten().astype(int)
    target = np.asarray(td['targetnumber']).flatten().astype(int)
    triallen = np.asarray(td['triallength'], dtype=float).flatten()

    lr = (tasknum == 1) & np.isin(target, [1, 2])
    idx = np.where(lr & (triallen >= 4.0))[0]
    if max_trials:
        idx = idx[:max_trials]

    stride = int(round(stride_ms / 1000 * C.FS_TGT))
    lo, hi = BANDAS[banda]
    mu, sd = ckpt['mu'], ckpt['sd']

    por_pos: dict[int, list] = {}
    for i in idx:
        tr = np.asarray(data[i], dtype=float)
        if tr.shape[0] != len(labels):
            tr = tr.T
        x = aplicar_car(tr, labels, mode=C.CAR_MODE)
        x = downsample(x)
        t = ds_time(x.shape[1])

        wins, centros = [], []
        s = 0
        while s + w_samp <= x.shape[1]:
            w = x[:, s:s + w_samp]
            w = w - w.mean(axis=1, keepdims=True)          # per-window centring, as in the cache
            wins.append(w); centros.append(float(t[s + w_samp // 2]))
            s += stride
        if not wins:
            continue
        Xw = np.stack(wins).astype(np.float32)
        Xw = bandpass(Xw, lo=lo, hi=hi, fs=C.FS_TGT)
        Xw = ((Xw - mu) / sd).astype(np.float32)

        p = predecir(ckpt, Xw)
        for c, pv in zip(centros, p):
            por_pos.setdefault(int(round(c)), []).append(float(pv))

    del m, bci, data
    return {k: (float(np.mean(v)), len(v)) for k, v in sorted(por_pos.items())}


@torch.no_grad()
def predecir(ckpt, X, batch=256) -> np.ndarray:
    """Command score per window, in the same space as analyze.py."""
    form = ckpt['form']
    if form == 'D':
        mg = build_model(ckpt['arch'], 8, X.shape[2], 2).to(C.device)
        mg.load_state_dict(ckpt['gate']); mg.eval()
        out = []
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float().unsqueeze(1).to(C.device)
            out.append(torch.softmax(mg(xb), 1)[:, 0].cpu().numpy())   # P(MI)
        return np.concatenate(out)

    n = ckpt['n_clases']
    md = build_model(ckpt['arch'], 8, X.shape[2], n).to(C.device)
    md.load_state_dict(ckpt['model']); md.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().unsqueeze(1).to(C.device)
        p = torch.softmax(md(xb), 1).cpu().numpy()
        out.append(p.max(1) if n == 2 else p[:, LH] + p[:, RH])
    return np.concatenate(out)


def transicion(perfil: dict) -> dict:
    """Rise of P(command) at the cue (t = 0) vs at the MI onset (t = 2000 ms).

    The intervals refer to window centres.
    """
    ts = np.array(sorted(perfil))
    ps = np.array([perfil[t][0] for t in ts])
    if len(ts) < 4:
        return {}

    def media(a, b):
        m = (ts >= a) & (ts < b)
        return float(ps[m].mean()) if m.any() else float('nan')

    pre_cue = media(-2000, 0)          # before the cue
    post_cue = media(0, 2000)          # cue visible, MI not started yet
    mi = media(2000, 5000)             # MI with cursor feedback

    salto_cue = post_cue - pre_cue     # rise attributed to the cue onset
    salto_mi = mi - post_cue           # rise attributed to the MI onset
    total = mi - pre_cue
    return {'pre_cue_[-2000,0)': pre_cue, 'post_cue_[0,2000)': post_cue,
            'mi_[2000,5000)': mi, 'salto_cue': salto_cue, 'salto_mi': salto_mi,
            'total': total,
            'frac_cue': float(salto_cue / total) if total not in (0, float('nan')) else float('nan')}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--stride-ms', type=int, default=125)
    ap.add_argument('--max-trials', type=int, default=40,
                    help='trials per session file')
    ap.add_argument('--smoke', action='store_true', help='2 subjects, 5 trials per session')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, weights_only=False, map_location=C.device)
    W = float(ck.get('W', 2.0)); w_samp = int(round(W * C.FS_TGT))
    banda = ck.get('banda', 'mb')
    print(f'ckpt {Path(a.ckpt).name}: form={ck["form"]} band={banda} W={W}s '
          f'n_clases={ck.get("n_clases")}\n')

    _, val_subs = split_canonico()
    if a.smoke:
        val_subs = val_subs[:2]; a.max_trials = 5

    files = sorted(glob.glob(str(C.DATA / 'S*_Session_*.mat')))
    files = [f for f in files
             if parse_name(f)[0] in set(val_subs) and parse_name(f)[1] in C.SESSIONS]
    print(f'{len(files)} validation files ({len(val_subs)} subjects), '
          f'stride={a.stride_ms}ms, {a.max_trials} trials/session\n', flush=True)

    acumulado: dict[int, list] = {}
    por_sujeto = {}
    for k, fp in enumerate(files, 1):
        sid, ses = parse_name(fp)
        perfil = perfil_temporal(fp, ck, w_samp, banda, a.stride_ms, a.max_trials)
        if not perfil:
            continue
        por_sujeto.setdefault(sid, {})
        for t, (p, n) in perfil.items():
            acumulado.setdefault(t, []).append(p)
            por_sujeto[sid].setdefault(t, []).append(p)
        print(f'  [{k}/{len(files)}] S{sid}_Ses{ses}: {len(perfil)} positions', flush=True)

    perfil_med = {t: (float(np.mean(v)), len(v)) for t, v in sorted(acumulado.items())}
    tr = transicion(perfil_med)

    print('\n' + '=' * 76)
    print('Temporal profile of P(command), mean over validation session files')
    print('=' * 76)
    print(f'{"t centre (ms)":>14} {"P(command)":>12} {"n":>6}')
    for t, (p, n) in perfil_med.items():
        marca = ''
        if -60 <= t <= 60:
            marca = '  <-- CUE (t=0)'
        elif 1940 <= t <= 2060:
            marca = '  <-- MI onset (t=2000)'
        print(f'{t:14d} {p:12.4f} {n:6d}{marca}')

    print('\n' + '=' * 76)
    print('Rise at the cue vs at the MI onset')
    print('=' * 76)
    for k, v in tr.items():
        print(f'  {k:22s} {v:+.4f}')
    if tr and tr.get('total', 0) > 0:
        fc = tr['frac_cue']
        print(f'\n  fraction of the total rise that occurs before the MI onset: {fc:.1%}')
        if fc > 0.5:
            print('  -> most of the rise occurs at the cue onset, while the subject is not')
            print('     imagining yet: the model follows the cue rather than the state, and')
            print('     the MI-vs-REST2 separation is inflated by cue-locking.')
        elif fc < 0.25:
            print('  -> the rise is concentrated at the MI onset: the model follows the')
            print('     mental state.')
        else:
            print('  -> mixed effect, partly cue-locked and partly state-related; report')
            print('     both figures separately.')

    out = Path(a.out) if a.out else HERE / 'outputs' / \
        f'cue_free_control_{Path(a.ckpt).stem}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'ckpt': str(a.ckpt), 'perfil': perfil_med,
                               'transicion': tr,
                               'por_sujeto': {s: {t: float(np.mean(v))
                                                  for t, v in d.items()}
                                              for s, d in por_sujeto.items()}},
                              indent=1, default=float))
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
