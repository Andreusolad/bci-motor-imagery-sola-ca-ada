"""Regression check: the development-bench result must reproduce exactly.

Re-runs the five-fold cross-validation inside each calibration recording for the
configuration in config/dev_best.json and compares, subject by subject, against the
values stored there. The code path is deterministic (fixed seeds, no parallelism
inside the computation), so the tolerance is 0 up to JSON round-off. Run it after any
change to the pipeline. It uses calibration data only.

Usage:
    python check_regression.py
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import json
import time
from concurrent.futures import ProcessPoolExecutor

import components as K
import pipeline as P

SUJETOS = ['a', 'b', 'f', 'g']
TOL = 1e-12
DEV_CFG = K.HERE / 'config' / 'dev_best.json'


def _uno(args):
    s, cfg = args
    r = P.correr_cv(s, cfg)
    return s, dict(mse=r['mse'], mse_cero=r['mse_cero'],
                   reduccion=r['reduccion_relativa'])


def main():
    t0 = time.time()
    dev = json.loads(DEV_CFG.read_text(encoding='utf-8'))
    cfg = P.cfg_con(**dev['cfg'])
    ref = dev['cv_reference']

    with ProcessPoolExecutor(max_workers=4) as ex:
        ahora = dict(ex.map(_uno, [(s, cfg) for s in SUJETOS]))

    filas, peor = [], 0.0
    for s in SUJETOS:
        for k in ('mse', 'mse_cero', 'reduccion'):
            a, b = ahora[s][k], ref[s][k]
            d = abs(a - b)
            peor = max(peor, d)
            filas.append(dict(sujeto=s, campo=k, ahora=a, guardado=b, dif=d,
                              ok=bool(d <= TOL)))
    todo_ok = all(f['ok'] for f in filas)
    out = dict(cfg=cfg, hash_cfg=K.hash_cfg(cfg), tolerancia=TOL, max_dif=peor,
               n_comprobaciones=len(filas), todas_ok=todo_ok, filas=filas,
               segundos=round(time.time() - t0, 1))
    K.guardar_json(K.OUT / 'check_regression.json', out)
    for f in filas:
        print(f'  {"ok  " if f["ok"] else "FAIL"} {f["sujeto"]} {f["campo"]:<9s} '
              f'{f["ahora"]!r} vs {f["guardado"]!r}  diff={f["dif"]:.3e}')
    print(f'\n{len(filas)} checks, max diff={peor:.3e}, '
          f'{"all ok" if todo_ok else "REGRESSION"}  ({time.time() - t0:.0f}s)')
    assert todo_ok, 'the stored development-bench result does not reproduce'


if __name__ == '__main__':
    main()
