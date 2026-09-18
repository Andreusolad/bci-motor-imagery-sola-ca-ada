"""Aptitude study: train the five cross-validation folds.

Calls `run_experiments.correr_celda` unchanged (same network, preprocessing and rest
balancing) and only changes which subjects are held out: fold k validates on the
subjects listed for it in outputs/aptitude_folds.json (12-13 subjects per fold). After
the five folds every one of the 62 subjects has out-of-fold predictions.

Split injection: `correr_celda` calls `split_canonico()` when split_mode == 'canon' and
`split_rt()` otherwise, and caches the band-passed, z-scored data under the key
(band, causal, split_mode). This script passes split_mode='fold<k>' and replaces
`run_experiments.split_rt` with a function that returns the split of the fold. The cache
key then includes the fold, so z-score statistics are never reused from the training
set of another fold.

Formulations and bands (--plan "form:band"):
  A  binary LH/RH, trained on MI windows only: direction axis
  B  LH/RH/IDLE: direction and detection from one network
  D  gate (MI vs rest) + LH/RH discriminator: the two axes in two separate networks
  mb = 8-30 Hz, bb = 0.5-40 Hz

Reads:  cache/trials_mc4_W<W>_per_window.npz (build_dataset.py) and
        outputs/aptitude_folds.json (aptitude_folds.py).
Writes: outputs/aptitude/main/fold<k>/preds_*.npz and ckpt_*.pt, and
        outputs/aptitude/main/folds_summary.json (outputs/aptitude/smoke/ with --smoke).

Usage:
    python aptitude_train.py --smoke                 # fold 0, 2 epochs, B:mb: code check
    python aptitude_train.py                         # 5 folds x A,B,D x mb,bb
    python aptitude_train.py --folds 1 2 --plan B:bb
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_experiments as RE                                    # noqa: E402

OUT = HERE / 'outputs' / 'aptitude'
FOLDS_JSON = HERE / 'outputs' / 'aptitude_folds.json'


def cargar_folds() -> dict:
    if not FOLDS_JSON.exists():
        raise SystemExit(f'missing {FOLDS_JSON}; run python aptitude_folds.py first')
    return json.loads(FOLDS_JSON.read_text(encoding='utf-8'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--plan', nargs='+',
                    default=['B:bb', 'B:mb', 'D:bb', 'D:mb', 'A:bb', 'A:mb'],
                    help='cells in priority order, as "form:band". The fold loop is '
                         'the inner loop, so each finished cell covers all 62 subjects '
                         'and the run can be stopped between cells.')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--ratio', type=float, default=1.0)
    ap.add_argument('--W', type=float, default=2.0)
    ap.add_argument('--arch', default='conformer')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--causal', action='store_true')
    ap.add_argument('--smoke', action='store_true',
                    help='fold 0, 2 epochs, cell B:mb: checks the code')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    if a.smoke:
        a.folds = [0]; a.epochs = 2; a.plan = ['B:mb']; a.seeds = [42]
    epochs = a.epochs if a.epochs is not None else RE.C.EPOCHS_PRETRAIN
    celdas = []
    for p in a.plan:
        form, _, banda = p.partition(':')
        if form not in RE.FORMULACIONES or banda not in RE.BANDAS:
            raise SystemExit(f'invalid plan entry: {p!r} (use form:band, e.g. B:bb)')
        celdas.append((form, banda))

    fj = cargar_folds()
    folds = {int(k): v for k, v in fj['folds'].items()}
    train_de = {int(k): v for k, v in fj['train_de_fold'].items()}

    cache = RE.CACHE / f'trials_mc4_W{a.W:g}_per_window.npz'
    if not cache.exists():
        raise SystemExit(f'missing cache {cache}')

    out_root = OUT / ('smoke' if a.smoke else 'main')
    plan = [(k, b, f, s) for (f, b) in celdas for s in a.seeds for k in a.folds]
    print('=' * 74)
    print(f'Aptitude study: training by fold   trainings={len(plan)}')
    print(f'  plan={a.plan}  folds={a.folds}')
    print(f'  seeds={a.seeds}  epochs={epochs}  arch={a.arch}  device={RE.C.device}')
    print(f'  output={out_root}')
    for k in a.folds:
        print(f'   fold {k}: val={folds[k]}  (train n={len(train_de[k])})')
    print('=' * 74, flush=True)
    if a.dry_run:
        return

    d = RE.cargar(cache)
    print(f'cache loaded: X{d["X"].shape}  meta={d["meta"]}', flush=True)

    # Resumable: a training already listed in folds_summary.json whose preds file
    # exists is skipped.
    res_json = out_root / 'folds_summary.json'
    resultados = json.loads(res_json.read_text()) if res_json.exists() else []
    hechas = {(r['fold'], r['form'], r['banda'], r['seed']) for r in resultados}
    t0, n_new = time.time(), 0
    for k, banda, form, seed in plan:
        pred_fp = out_root / f'fold{k}' / f'preds_{form}_{banda}_W{a.W:g}_r{a.ratio:g}_s{seed}.npz'
        if (k, form, banda, seed) in hechas and pred_fp.exists():
            print(f'  [skip] fold{k} {form}_{banda}_s{seed} already trained', flush=True)
            continue
        tr_s, va_s = train_de[k], folds[k]

        # Inject the split of the fold (see the module docstring).
        RE.split_rt = lambda subject, hit_rate, _tr=tr_s, _va=va_s: (_tr, _va)

        out_dir = out_root / f'fold{k}'
        r = RE.correr_celda(d, form, banda, a.ratio, seed, epochs, a.arch,
                            f'fold{k}', a.causal, out_dir, a.W, guardar_ckpt=True)
        r['fold'] = k
        # The validation and training subjects must be exactly those of the fold.
        assert r['subjects_val'] == sorted(va_s), \
            f'fold {k}: val={r["subjects_val"]} != {sorted(va_s)}'
        assert r['subjects_train'] == sorted(tr_s), f'fold {k}: training subjects differ'
        resultados.append(r)
        n_new += 1
        out_root.mkdir(parents=True, exist_ok=True)
        res_json.write_text(json.dumps(resultados, indent=1))
        el = time.time() - t0
        pend = len(plan) - len(resultados)
        print(f'  [{len(resultados)}/{len(plan)}] fold{k} {form}_{banda}_s{seed} '
              f'ok  ({el/60:.1f} min, eta {el/n_new*pend/60:.0f} min)', flush=True)

    print(f'\n{n_new} new trainings ({len(resultados)}/{len(plan)}) in '
          f'{(time.time()-t0)/60:.1f} min -> {out_root}')
    print('next step:  python aptitude_metrics.py')


if __name__ == '__main__':
    main()
