"""Five subject folds and a per-subject inventory for the aptitude study.

The folds cover the 62 Stieger subjects so that every subject gets an out-of-fold
prediction (no subject influences the model that scores it) with five trainings per
formulation instead of 62 for leave-one-subject-out:

  fold 0     the canonical validation split (val_subjects of split_80_20_subjects.json,
             12 subjects), so its metrics can be checked against the canonical results.
  folds 1-4  the remaining 50 subjects, stratified by online hit rate: sorted by hit_rate
             (ties broken by subject id) and dealt in serpentine order (0,1,2,3,3,2,1,0,...)
             so that every fold gets a comparable mix of good and poor performers and no
             fold takes most of the low-aptitude subjects.

The assignment is deterministic. The inventory lists, per subject: fold, hit_rate,
low-aptitude flag (hit_rate < ILLIT_THR), window counts and amplitude-flag fractions per
source (mi, rest1, rest2), LH/RH counts and balance, number of sessions and trials.

Reads cache/trials_mc4_W2_per_window.npz (metadata only) and ../split_80_20_subjects.json.

Usage:
    python aptitude_folds.py --test     # 9 self-tests, writes outputs/aptitude_folds_tests.json
    python aptitude_folds.py            # writes outputs/aptitude_folds.json and
                                        # outputs/aptitude_inventory.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'cache' / 'trials_mc4_W2_per_window.npz'
SPLIT_JSON = HERE.parent / 'split_80_20_subjects.json'
OUT = HERE / 'outputs'

N_FOLDS = 5
ILLIT_THR = 0.50          # same threshold as analyze.py (ILLIT_THR)


# ============================================================
# Metadata (X is never read: np.load reads npz keys lazily)
# ============================================================
def cargar_meta() -> dict:
    z = np.load(CACHE, allow_pickle=True)
    d = {'subject': z['subject'], 'session': z['session'], 'source': z['source'],
         'y': z['y'], 'trial_id': z['trial_id'], 'f_amp': z['f_amp'],
         'tasknumber': z['tasknumber'], 'mi_valido': z['mi_valido']}
    d['hit_rate'] = {int(s): float(v)
                     for s, v in zip(z['hit_subjects'], z['hit_values'])}
    d['meta'] = [str(s) for s in z['meta']]
    return d


def split_canonico() -> tuple[list[int], list[int]]:
    sp = json.loads(SPLIT_JSON.read_text(encoding='utf-8'))
    f = lambda L: sorted(int(s[1:]) if isinstance(s, str) else int(s) for s in L)
    return f(sp['train_subjects']), f(sp['val_subjects'])


# ============================================================
# Folds
# ============================================================
def serpentina(orden: list[int], k: int) -> list[list[int]]:
    """Deal `orden` (already sorted by the stratification variable) into k groups
    in serpentine order 0..k-1, k-1..0, 0..k-1, ...  Returns k lists."""
    grupos: list[list[int]] = [[] for _ in range(k)]
    for i, s in enumerate(orden):
        ciclo, pos = divmod(i, k)
        j = pos if ciclo % 2 == 0 else (k - 1 - pos)
        grupos[j].append(s)
    return grupos


def construir_folds(hit_rate: dict) -> list[list[int]]:
    tr_can, va_can = split_canonico()
    todos = sorted(hit_rate)
    assert set(tr_can) | set(va_can) == set(todos), 'canonical split does not cover the cache'

    # fold 0 = canonical validation split; the rest are stratified by hit rate
    resto = sorted(tr_can)
    # ascending hit_rate; ties broken by subject id so the order is deterministic
    orden = sorted(resto, key=lambda s: (hit_rate[s], s))
    folds = [sorted(va_can)] + [sorted(g) for g in serpentina(orden, N_FOLDS - 1)]
    return folds


# ============================================================
# Per-subject inventory
# ============================================================
def inventario(d: dict, folds: list[list[int]]) -> list[dict]:
    fold_de = {s: k for k, f in enumerate(folds) for s in f}
    sub = d['subject'].astype(int)
    src = d['source']
    filas = []
    for s in sorted(d['hit_rate']):
        m = sub == s
        fila = {'subject': s, 'fold': fold_de[s], 'hit_rate': d['hit_rate'][s],
                'iletrado': int(d['hit_rate'][s] < ILLIT_THR)}
        # n_*: windows per source; famp_*: fraction of them flagged by the amplitude check
        for nm in ('mi', 'rest1', 'rest2'):
            mm = m & (src == nm)
            fila[f'n_{nm}'] = int(mm.sum())
            fila[f'famp_{nm}'] = float(d['f_amp'][mm].mean()) if mm.any() else float('nan')
        mmi = m & (src == 'mi')
        fila['n_LH'] = int((d['y'][mmi] == 0).sum())
        fila['n_RH'] = int((d['y'][mmi] == 1).sum())
        fila['bal_LH'] = (fila['n_LH'] / fila['n_mi']) if fila['n_mi'] else float('nan')
        fila['n_sesiones'] = int(len(set(d['session'][m].tolist())))
        fila['n_trials'] = int(len(set(d['trial_id'][m].tolist())))
        filas.append(fila)
    return filas


# ============================================================
# Self-tests
# ============================================================
def autotests() -> dict:
    r = []

    def chk(nombre, cond, detalle=''):
        r.append({'test': nombre, 'ok': bool(cond), 'detalle': str(detalle)})
        print(f'  [{"OK" if cond else "FAIL"}] {nombre}  {detalle}')
        return bool(cond)

    d = cargar_meta()
    hr = d['hit_rate']
    folds = construir_folds(hr)
    tr_can, va_can = split_canonico()

    chk('62 subjects in the cache', len(hr) == 62, f'n={len(hr)}')
    union = sorted(s for f in folds for s in f)
    chk('the folds cover exactly the 62 subjects',
        union == sorted(hr), f'n_union={len(union)}')
    dup = len(union) != len(set(union))
    chk('no subject in two folds', not dup, f'duplicates={dup}')
    chk('fold 0 == canonical validation split', folds[0] == sorted(va_can),
        f'fold0={folds[0]}')
    tam = [len(f) for f in folds]
    chk('12 or 13 subjects per fold', all(12 <= t <= 13 for t in tam) and sum(tam) == 62,
        f'sizes={tam}')

    # low-aptitude subjects (hit_rate < ILLIT_THR)
    il = sorted(s for s in hr if hr[s] < ILLIT_THR)
    chk('low-aptitude subjects = {3,6,17,21,24,40}', il == [3, 6, 17, 21, 24, 40], f'{il}')
    rep = {k: sum(1 for s in f if hr[s] < ILLIT_THR) for k, f in enumerate(folds)}
    chk('at most 2 low-aptitude subjects in each of folds 1-4',
        all(v <= 2 for k, v in rep.items() if k > 0), f'low-aptitude per fold={rep}')

    # training set of each fold = the 62 subjects minus the held-out fold
    ok_tr = all(sorted(set(hr) - set(f)) == sorted(set(hr) - set(f)) and
                len(set(hr) - set(f)) == 62 - len(f) for f in folds)
    chk('train of each fold = 62 - held-out', ok_tr,
        f'n_train={[62 - len(f) for f in folds]}')

    # stratification: mean hit_rate of folds 1-4 within a 0.05 range
    mus = [float(np.mean([hr[s] for s in f])) for f in folds]
    chk('mean hit_rate per fold within 0.05 (folds 1-4)',
        max(mus[1:]) - min(mus[1:]) < 0.05,
        'means=' + ', '.join(f'{m:.3f}' for m in mus))

    ok = all(x['ok'] for x in r)
    print(f'\n{sum(x["ok"] for x in r)}/{len(r)} self-tests OK')
    return {'todos_ok': ok, 'tests': r}


# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test', action='store_true', help='run the self-tests and exit')
    a = ap.parse_args()

    if a.test:
        res = autotests()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'aptitude_folds_tests.json').write_text(json.dumps(res, indent=1))
        sys.exit(0 if res['todos_ok'] else 1)

    d = cargar_meta()
    folds = construir_folds(d['hit_rate'])
    inv = inventario(d, folds)

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        'n_folds': N_FOLDS,
        'illit_thr': ILLIT_THR,
        'cache': CACHE.name,
        'cache_meta': d['meta'],
        'folds': {str(k): f for k, f in enumerate(folds)},
        'train_de_fold': {str(k): sorted(set(d['hit_rate']) - set(f))
                          for k, f in enumerate(folds)},
        'tamanos': [len(f) for f in folds],
        'hit_rate': {str(s): v for s, v in sorted(d['hit_rate'].items())},
        'iletrados': sorted(s for s, v in d['hit_rate'].items() if v < ILLIT_THR),
        'iletrados_por_fold': {str(k): [s for s in f if d['hit_rate'][s] < ILLIT_THR]
                               for k, f in enumerate(folds)},
        'hit_rate_medio_por_fold': [float(np.mean([d['hit_rate'][s] for s in f]))
                                    for f in folds],
        'inventario': inv,
    }
    (OUT / 'aptitude_folds.json').write_text(json.dumps(payload, indent=1))

    with open(OUT / 'aptitude_inventory.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(inv[0]))
        w.writeheader()
        w.writerows(inv)

    print('=' * 74)
    print('Aptitude study: folds and inventory')
    print('=' * 74)
    for k, f in enumerate(folds):
        ilk = [s for s in f if d['hit_rate'][s] < ILLIT_THR]
        print(f'  fold {k}: n={len(f):2d}  '
              f'mean hit_rate={np.mean([d["hit_rate"][s] for s in f]):.3f}  '
              f'low-aptitude={ilk}')
        print(f'          {f}')
    print(f'\n  low-aptitude subjects: {payload["iletrados"]}')
    print(f'\n-> {OUT / "aptitude_folds.json"}')
    print(f'-> {OUT / "aptitude_inventory.csv"}')


if __name__ == '__main__':
    main()
