"""Score a system on the BCI Competition IV Data Set 1 evaluation recordings.

A shot trains on the whole calibration recording of each subject, predicts the
continuous evaluation recording and scores it with the official metric (mean squared
error against the per-sample labels, subject mean over a, b, f, g). Three shots are
defined here, the ones reported in the paper:

  D1   the configuration selected on the development bench (calibration data only)
  D12  D1 with four adjustments, averaged over two alignment members (offline)
  D13  D12 made strictly causal (see causality_test.py)

Each subject also gets a block-bootstrap interval whose unit is a constant-label
segment of the evaluation stream.

The winning entry's published per-subject values (Zhang et al. 2012, Table 3) are
a 0.40, b 0.42, c 0.42, d 0.29 (mean 0.38). Their subjects a, b, c, d are our
a, b, f, g: `check_labels.py` matches the silence baseline of each subject to the
published zero-output value of the corresponding subject.

Usage:
    python run_shot.py --shot D12
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np

import data_io as A
import components as K
import pipeline as P

SUJETOS = ['a', 'b', 'f', 'g']
DEV_CFG = K.HERE / 'config' / 'dev_best.json'

# Table 3 of the winning entry, in our subject naming.
GANADOR_POR_SUJETO = {'a': 0.40, 'b': 0.42, 'f': 0.42, 'g': 0.29}
GANADOR_MEDIA = 0.382          # official ranking value
SEGUNDO_OFICIAL = 0.383
SILENCIO_OFICIAL = 0.509

# D12 = D1 + a 1.0 s window, a 320-sample post-processing buffer, feature alignment
# estimated in 300 s blocks, and the average of two alignment members.
D12_EXTRA = {'w_s': 1.0, 'post_taps': 320, 'alineamiento_bloques_s': 300.0}
D12_MIEMBROS = ['centrado', 'euclideo_centrado']
_VARIANTE_DE_MIEMBRO = {
    'centrado': dict(recentrar=False, alineamiento='centrado'),
    'euclideo_centrado': dict(recentrar=True, alineamiento='centrado'),
}


def _sse_por_segmento(out_100: np.ndarray, true_y: np.ndarray):
    """(sse, n) per constant-label segment, the unit of the block bootstrap.

    Same arithmetic as `data_io.mse_oficial`: the 100 Hz output is held for 10
    samples and only finite labels count.
    """
    o = np.repeat(np.asarray(out_100, np.float64), A.FS_LAB // A.FS_IV1)
    t = np.asarray(true_y, np.float64)
    n = min(len(o), len(t))
    o, t = o[:n], t[:n]
    e2 = np.where(np.isfinite(t), (o - np.nan_to_num(t)) ** 2, 0.0)
    sse, cnt = [], []
    for s in A.segmentos_definidos(t):
        a, b = s['ini'], s['fin']
        sse.append(float(e2[a:b].sum()))
        cnt.append(int(np.isfinite(t[a:b]).sum()))
    return np.asarray(sse), np.asarray(cnt, np.int64)


def _un_sujeto(args) -> tuple[str, dict]:
    s, cfg = args
    t1 = time.time()
    # an ensemble arrives as a list of configurations
    if isinstance(cfg, (list, tuple)):
        r = P.correr_eval_ensamblado(s, list(cfg))
        cfg = r['cfg']
    else:
        r = P.correr_eval(s, cfg)
    true_y = A.cargar_true_y(s)
    sse, cnt = _sse_por_segmento(r['salida'], true_y)
    mse, lo, hi = A.boot_bloques(sse, cnt)
    gan = GANADOR_POR_SUJETO[s]
    return s, dict(
        mse=r['mse'], mse_boot=float(mse), ic95=[float(lo), float(hi)],
        mse_cero=r['mse_cero'], n_evaluadas=r['n_evaluadas'],
        n_segmentos=int(len(cnt)), n_features=r['n_features'],
        n_dim_modelo=r.get('n_dim_modelo'), n_train=r['n_train'],
        tam_subestados_nc=r['tam_subestados_nc'],
        reduccion_relativa=float(1.0 - r['mse'] / r['mse_cero']),
        cfg=cfg, hash_cfg=K.hash_cfg(cfg),
        n_miembros=r.get('n_miembros', 1), hash_miembros=r.get('hash_miembros'),
        ganador=gan, dif_vs_ganador=float(r['mse'] - gan),
        bate_al_ganador=bool(r['mse'] < gan),
        ic_excluye_al_ganador=bool(hi < gan or lo > gan),
        segundos=round(time.time() - t1, 1))


def disparar(cfgs: dict, sujetos=SUJETOS, etiqueta: str = '', procesos: int = 2) -> dict:
    """`cfgs` maps subject -> configuration (or list of configurations for an ensemble)."""
    t0 = time.time()
    tareas = [(s, cfgs[s]) for s in sujetos]
    if procesos <= 1:
        res = dict(_un_sujeto(t) for t in tareas)
    else:
        with ProcessPoolExecutor(max_workers=min(procesos, len(tareas))) as ex:
            res = dict(ex.map(_un_sujeto, tareas))
    por_sujeto = {s: res[s] for s in sujetos}
    for s in sujetos:
        v = por_sujeto[s]
        print(f'   {s}: mse={v["mse"]:.4f} CI95[{v["ic95"][0]:.4f},'
              f'{v["ic95"][1]:.4f}] silence={v["mse_cero"]:.4f} '
              f'winner={v["ganador"]:.2f} '
              f'{"beats" if v["bate_al_ganador"] else "does not beat"} '
              f'({v["segundos"]:.0f}s)')

    medias = dict(
        mse=float(np.mean([por_sujeto[s]['mse'] for s in sujetos])),
        mse_cero=float(np.mean([por_sujeto[s]['mse_cero'] for s in sujetos])),
        n_bate=int(sum(por_sujeto[s]['bate_al_ganador'] for s in sujetos)))
    medias['reduccion_relativa'] = float(1.0 - medias['mse'] / medias['mse_cero'])
    medias['dif_vs_ganador'] = float(medias['mse'] - GANADOR_MEDIA)
    medias['bate_al_ganador'] = bool(medias['mse'] < GANADOR_MEDIA)

    es_ens = any(isinstance(cfgs[s], (list, tuple)) for s in sujetos)
    disparo = dict(
        etiqueta=etiqueta,
        utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        es_ensamblado=bool(es_ens),
        cfgs=({s: list(cfgs[s]) for s in sujetos} if es_ens
              else {s: cfgs[s] for s in sujetos}),
        sujetos=list(sujetos),
        por_sujeto=por_sujeto, media=medias,
        referencia=dict(ganador_media=GANADOR_MEDIA,
                        ganador_por_sujeto=GANADOR_POR_SUJETO,
                        segundo=SEGUNDO_OFICIAL, silencio=SILENCIO_OFICIAL),
        segundos=round(time.time() - t0, 1))
    print(f'\n   MEAN {medias["mse"]:.4f} vs winner {GANADOR_MEDIA:.3f} '
          f'({medias["dif_vs_ganador"]:+.4f}); beats it on {medias["n_bate"]}/'
          f'{len(sujetos)} subjects')
    return disparo


def cfg_desarrollo() -> dict:
    """The configuration chosen on the development bench (calibration data only)."""
    dev = json.loads(DEV_CFG.read_text(encoding='utf-8'))
    return P.cfg_con(**dev['cfg'])


def cfgs_d12() -> list:
    """The two members of the D12 ensemble."""
    base = cfg_desarrollo()
    base_c = P.cfg_con(**{**base, 'alineamiento': 'centrado', **D12_EXTRA})
    return [P.cfg_con(**{**base_c, **_VARIANTE_DE_MIEMBRO[m]}) for m in D12_MIEMBROS]


def cfgs_d13() -> list:
    """D12 made causal. The recentering member is dropped (its causal version cost more
    on the calibration bench than removing it) and the band-pass and the feature
    alignment of the remaining member are replaced by their causal versions, using the
    same table the perturbation test uses."""
    import causality_test as CA
    miembros = [c for c in cfgs_d12() if not c['recentrar']]
    assert miembros
    return CA.componer(miembros, {'filtrado', 'alineamiento'})


def cfgs_del_disparo(ident: str) -> tuple[dict, str]:
    if ident == 'D1':
        c = cfg_desarrollo()
        return {s: c for s in SUJETOS}, 'development-bench configuration'
    if ident == 'D12':
        c = cfgs_d12()
        return {s: c for s in SUJETOS}, 'D1 + four adjustments, two-member ensemble (offline)'
    if ident == 'D13':
        c = cfgs_d13()
        return {s: c for s in SUJETOS}, 'D12 made strictly causal'
    raise ValueError(ident)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shot', choices=['D1', 'D12', 'D13'], default='D12')
    ap.add_argument('--procesos', type=int, default=2, help='subjects run in parallel')
    a = ap.parse_args()

    cfgs, desc = cfgs_del_disparo(a.shot)
    print(f'SHOT {a.shot}: {desc}')
    c0 = cfgs[SUJETOS[0]]
    for i, c in enumerate(c0 if isinstance(c0, (list, tuple)) else [c0]):
        if isinstance(c0, (list, tuple)):
            print(f'   --- member {i + 1} of {len(c0)} ---')
        for k, v in c.items():
            print(f'   {k:<24s} {v}')
    print()
    d = disparar(cfgs, etiqueta=f'{a.shot}: {desc}', procesos=a.procesos)
    out = K.OUT / f'shot_{a.shot}.json'
    K.guardar_json(out, d)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
