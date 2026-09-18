"""Stage 2 of 3 of the per-intent evaluation: dense sliding-window inference on the streams.

Reads cache/streams_val.npz (prep_streams.py) and a checkpoint of run_experiments.py,
slides a W-second window with a 125 ms stride (rounded to whole samples) over every trial
and evaluates the model at each position. Writes outputs/streams/win_{tag}.npz with the
probabilities and time stamps (start, center and end, in ms) of every window and a copy
of the trial table, so that stage 3 (accumulate.py, run through cost_per_intent.py) can
be rerun without a GPU.

Canonical probability space: the formulations have different heads (2, 3 or 4 outputs,
or two models). All of them are mapped to the same vector [P(LH), P(RH), P(IDLE)], the
input of rt_system.accumulator.EvidenceAccumulator (C.N_CLASES = 3, C.IDLE_ID = 2):

    A/E (2 classes)  p_cmd = max(pLH, pRH)   p3 = [p_cmd*pLH, p_cmd*pRH, 1 - p_cmd]
    B   (3 classes)  p_cmd = pLH + pRH       p3 = [pLH, pRH, pIDLE]
    C   (4 classes)  p_cmd = pLH + pRH       p3 = [pLH, pRH, pR1 + pR2]
    D   (gate+disc)  p_cmd = P_gate(MI)      p3 = [p_cmd*pdLH, p_cmd*pdRH, 1 - p_cmd]

1 - p3[IDLE] equals the per-window command score of analyze.py (checked with an assert
on the first chunk of trials), so the per-window and per-intent analyses threshold the
same score.

Preprocessing follows the training order: crop -> per-window centering -> zero-phase
band-pass -> z-score with the mu/sd stored in the checkpoint.

Usage:
    python infer_streams.py --ckpt outputs/main/ckpt_B_bb_W2_r1_s42.pt
    python infer_streams.py --todos      # the 10 cells in CELDAS_DEF
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

from rt_system import config as C                            # noqa: E402
from rt_system.models import build_model                     # noqa: E402
from rt_system.preprocessing import bandpass, ds_time        # noqa: E402

from build_dataset import LH, RH                             # noqa: E402
from run_experiments import BANDAS                           # noqa: E402

CACHE = HERE / 'cache'
OUT = HERE / 'outputs' / 'streams'
IDLE3 = 2                                    # index of IDLE in the canonical space


# ============================================================
# Models of a checkpoint
# ============================================================
def cargar_modelos(ckpt, n_samples: int):
    """Return (list of models in eval mode, formulation)."""
    form = ckpt['form']
    if form == 'D':
        mg = build_model(ckpt['arch'], 8, n_samples, 2).to(C.device)
        mg.load_state_dict(ckpt['gate']); mg.eval()
        md = build_model(ckpt['arch'], 8, n_samples, 2).to(C.device)
        md.load_state_dict(ckpt['disc']); md.eval()
        return [mg, md], form
    n = int(ckpt['n_clases'])
    m = build_model(ckpt['arch'], 8, n_samples, n).to(C.device)
    m.load_state_dict(ckpt['model']); m.eval()
    return [m], form


@torch.no_grad()
def _softmax_batches(model, X: np.ndarray, batch: int) -> np.ndarray:
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().unsqueeze(1).to(C.device)
        out.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def probs_canonicas(models, form: str, X: np.ndarray, batch: int = 512) -> np.ndarray:
    """(n, 8, w) -> (n, 3) = [P(LH), P(RH), P(IDLE)] in the canonical space."""
    if form == 'D':
        pg = _softmax_batches(models[0], X, batch)      # [P(MI), P(rest)]
        pd = _softmax_batches(models[1], X, batch)      # [P(LH), P(RH)]
        p_cmd = pg[:, 0]
        p3 = np.stack([p_cmd * pd[:, 0], p_cmd * pd[:, 1], 1.0 - p_cmd], 1)
    else:
        p = _softmax_batches(models[0], X, batch)
        if p.shape[1] == 2:                             # A / E
            p_cmd = p.max(1)
            p3 = np.stack([p_cmd * p[:, LH], p_cmd * p[:, RH], 1.0 - p_cmd], 1)
        elif p.shape[1] == 3:                           # B
            p3 = p.copy()
        else:                                           # C: merge REST1 + REST2
            p3 = np.stack([p[:, LH], p[:, RH], p[:, 2] + p[:, 3]], 1)
    return p3.astype(np.float32)


def score_comando_ref(models, form: str, X: np.ndarray, batch: int = 512) -> np.ndarray:
    """Per-window command score of analyze.py, used to check the invariant."""
    if form == 'D':
        pg = _softmax_batches(models[0], X, batch)
        return 1.0 - pg[:, 1]
    p = _softmax_batches(models[0], X, batch)
    if p.shape[1] == 2:
        return p.max(1)
    return p[:, LH] + p[:, RH]


# ============================================================
# Windowing
# ============================================================
def ventanas_de_trial(x: np.ndarray, w_samp: int, stride_samp: int):
    """Return (wins (k, 8, w), ini_ms, centro_ms, fin_ms), or None if the trial is short."""
    t = ds_time(x.shape[1])
    idx = list(range(0, x.shape[1] - w_samp + 1, stride_samp))
    if not idx:
        return None
    # Center in float64 and cast afterwards, in the same order as build_dataset.py, so
    # that the windows match the training cache exactly.
    wins = np.stack([x[:, s:s + w_samp] for s in idx])
    wins = (wins - wins.mean(axis=2, keepdims=True)).astype(np.float32)
    ini = np.array([t[s] for s in idx], np.float32)
    cen = np.array([t[s + w_samp // 2] for s in idx], np.float32)
    fin = np.array([t[s + w_samp - 1] for s in idx], np.float32)
    return wins, ini, cen, fin


def correr(ckpt_path: Path, streams: Path, stride_ms: float = 125.0,
           batch: int = 512, chunk_trials: int = 150, verificar: bool = True) -> Path:
    z = np.load(streams, allow_pickle=True)
    sig, offset = z['sig'], z['offset']
    n_trials = len(offset) - 1

    ck = torch.load(ckpt_path, weights_only=False, map_location=C.device)
    W = float(ck.get('W', 2.0)); w_samp = int(round(W * C.FS_TGT))
    banda = ck.get('banda', 'mb'); lo, hi = BANDAS[banda]
    mu, sd = ck['mu'], ck['sd']
    models, form = cargar_modelos(ck, w_samp)
    stride_samp = int(round(stride_ms / 1000 * C.FS_TGT))
    stride_real = stride_samp / C.FS_TGT * 1000.0

    tag = ckpt_path.stem.replace('ckpt_', '')
    print(f'[{tag}] form={form} band={banda} W={W}s  '
          f'stride={stride_samp} samp ({stride_real:.1f} ms actual)  '
          f'trials={n_trials}', flush=True)

    P, INI, CEN, FIN, TRI = [], [], [], [], []
    t0 = time.time()
    for c0 in range(0, n_trials, chunk_trials):
        c1 = min(c0 + chunk_trials, n_trials)
        wins, inis, cens, fins, tris = [], [], [], [], []
        for i in range(c0, c1):
            x = sig[:, offset[i]:offset[i + 1]]
            v = ventanas_de_trial(x, w_samp, stride_samp)
            if v is None:
                continue
            w, a, b, c = v
            wins.append(w); inis.append(a); cens.append(b); fins.append(c)
            tris.append(np.full(len(w), i, np.int32))
        if not wins:
            continue
        Xw = np.concatenate(wins)
        Xw = bandpass(Xw, lo=lo, hi=hi, fs=C.FS_TGT)
        Xw = ((Xw - mu) / sd).astype(np.float32)

        p3 = probs_canonicas(models, form, Xw, batch)
        if verificar and c0 == 0:
            sc = score_comando_ref(models, form, Xw, batch)
            err = float(np.abs((1.0 - p3[:, IDLE3]) - sc).max())
            assert err < 2e-6, (f'invariant violated: 1-P(IDLE) != score_comando '
                                f'(max err {err:.2e}); the per-intent analysis would not '
                                f'be comparable with the per-window analysis')
            print(f'  [OK] invariant 1-P(IDLE) == score_comando  (max err {err:.1e})',
                  flush=True)

        P.append(p3); INI.append(np.concatenate(inis)); CEN.append(np.concatenate(cens))
        FIN.append(np.concatenate(fins)); TRI.append(np.concatenate(tris))
        if (c0 // chunk_trials) % 10 == 0:
            el = time.time() - t0
            print(f'  trials {c1}/{n_trials}  windows={sum(len(p) for p in P):,d}  '
                  f'({el:.0f}s eta {el/max(c1,1)*(n_trials-c1):.0f}s)', flush=True)

    probs = np.concatenate(P); ini = np.concatenate(INI)
    cen = np.concatenate(CEN); fin = np.concatenate(FIN); tri = np.concatenate(TRI)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f'win_{tag}.npz'
    np.savez_compressed(
        out_path, probs3=probs, ini_ms=ini, centro_ms=cen, fin_ms=fin, win_trial=tri,
        # trial table copied as is, so that stage 3 needs only this file
        subject=z['subject'], session=z['session'], trial_idx=z['trial_idx'],
        kind=z['kind'], y4=z['y4'], triallen=z['triallen'], result=z['result'],
        mi_ok=z['mi_ok'], hit_subjects=z['hit_subjects'], hit_values=z['hit_values'],
        meta=np.array([f'ckpt={ckpt_path.name}', f'form={form}', f'banda={banda}',
                       f'W={W}', f'stride_ms={stride_real:.2f}',
                       'probs3=[P(LH),P(RH),P(IDLE)] canonical',
                       'invariant: 1-P(IDLE)==score_comando of analyze.py']))
    del models
    if C.device.type == 'cuda':
        torch.cuda.empty_cache()
    print(f'  -> {out_path.name}  {len(probs):,d} windows  '
          f'({time.time()-t0:.0f}s)\n', flush=True)
    return out_path


CELDAS_DEF = ['A_bb_W2_r1_s42', 'A_mb_W2_r1_s42', 'B_bb_W2_r1_s42', 'B_mb_W2_r1_s42',
              'C_bb_W2_r1_s42', 'C_mb_W2_r1_s42', 'D_bb_W2_r1_s42', 'D_mb_W2_r1_s42',
              'B_bb_W2_r1_s43', 'B_mb_W2_r1_s43']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--todos', action='store_true', help='the 10 cells in CELDAS_DEF')
    ap.add_argument('--streams', default=None)
    ap.add_argument('--stride-ms', type=float, default=125.0)
    ap.add_argument('--batch', type=int, default=512)
    a = ap.parse_args()

    streams = Path(a.streams) if a.streams else CACHE / 'streams_val.npz'
    if not streams.exists():
        raise SystemExit(f'missing {streams}. Run first:  python prep_streams.py')

    if a.todos:
        ckpts = [HERE / 'outputs' / 'main' / f'ckpt_{c}.pt' for c in CELDAS_DEF]
    elif a.ckpt:
        ckpts = [Path(a.ckpt)]
    else:
        raise SystemExit('use --ckpt <path> or --todos')

    t0 = time.time()
    for cp in ckpts:
        if not cp.exists():
            print(f'[SKIP] {cp.name} does not exist', flush=True)
            continue
        correr(cp, streams, stride_ms=a.stride_ms, batch=a.batch)
    print(f'total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
