"""Aptitude study: per-subject out-of-fold metrics.

Collects the `preds_*.npz` files of the five folds (written by aptitude_train.py) and
produces one row per (cell, subject), with the two axes measured separately.

  Direction (LH vs RH)
    acc_dir      accuracy of the LH/RH decision over all MI windows, without threshold.
                 It equals `acc_control` of analyze.py at thr=0: the command score is
                 >= 0, so at thr=0 every window passes the gate. The thresholded
                 `acc_control_thr` only counts MI windows that pass the gate, so it
                 depends on detection. Both are saved. Wilson 95% CI in acc_dir_lo/hi.
    auc_dir      AUC of the continuous direction score (RH vs LH) within the MI
                 windows. Threshold-free and on the same scale as auc_det.

  Detection (MI vs rest)
    auc_det      AUC of the command score, MI vs all rest windows.
    auc_det_r1   same against REST1 only (cued rest).
    auc_det_r2   same against REST2 only (pre-cue rest).
    pauc02       pAUC (fpr <= 0.2) of the recall vs fpr_idle envelope (analyze.pauc).
    pauc02_r1/r2 the same pAUC with the false-positive rate of that rest source only.
    recall_fpr05/10  best recall with fpr_idle <= 0.05 / 0.10 (recall_fpr10_r1: with
                 fpr_rest1 <= 0.10).

  At the operating threshold --thr (default 0.5): acc_control_thr, recall_thr,
  fpr_idle_thr, fpr_rest1_thr, fpr_rest2_thr, error_critico_thr, coste_thr.
  Per-subject 95% CIs of auc_dir, auc_det and auc_det_r1 come from a window-level
  bootstrap; acc_dir_h1/h2 and auc_det_h1/h2 are split-half values (halves by trial).
  hit_rate, iletrado and n_sesiones are taken from outputs/aptitude_folds.json.

The sweep metrics use the functions of analyze.py unchanged (same threshold grid, same
definitions of pAUC and recall@fpr).

Writes outputs/aptitude_metrics.csv, the same rows in outputs/aptitude_metrics.json and
the per-subject threshold sweeps in outputs/aptitude_metrics_sweeps.npz (the three
paths follow --out).

Usage:
    python aptitude_metrics.py --test        # self-tests (AUC vs sklearn, etc.)
    python aptitude_metrics.py               # outputs/aptitude_metrics.{csv,json}
    python aptitude_metrics.py --dir outputs/aptitude/smoke
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_dataset import LH, RH, REST1, REST2                     # noqa: E402
import analyze as AN                                               # noqa: E402

OUT = HERE / 'outputs'
DIR_DEF = OUT / 'aptitude' / 'main'
FOLDS_JSON = OUT / 'aptitude_folds.json'


# ============================================================
# Rank-based AUC (ties get average ranks)
# ============================================================
def rangos_medios(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with the average rank for ties.

    Vectorized with bincount over groups of equal values, because the per-window
    bootstrap in auc_ci calls it B times per subject and metric.
    """
    o = np.argsort(x, kind='mergesort')
    xs = x[o]
    grp = np.cumsum(np.r_[True, xs[1:] != xs[:-1]]) - 1
    idx = np.arange(1, len(x) + 1, dtype=float)
    avg = np.bincount(grp, weights=idx) / np.bincount(grp)
    r = np.empty(len(x), float)
    r[o] = avg[grp]
    return r


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC. NaN if either class is empty."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rangos_medios(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auc_ci(pos, neg, B: int = 2000, seed: int = 42) -> tuple[float, float]:
    """95% CI of the AUC by stratified bootstrap (positives and negatives resampled
    separately).

    The resampling unit is the window within one subject, not the subject: the CI
    says whether this particular subject is above chance.
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    n1, n0 = len(pos), len(neg)
    if n1 < 2 or n0 < 2:
        return float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    v = np.empty(B)
    for b in range(B):
        v[b] = auc(pos[rng.randint(0, n1, n1)], neg[rng.randint(0, n0, n0)])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% CI of a proportion (k successes out of n)."""
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


# ============================================================
# pAUC against a single rest source
# ============================================================
def pauc_key(sweep: list[dict], key: str, fmax: float = 0.2) -> float:
    """Same as analyze.pauc but with `key` (fpr_rest1 or fpr_rest2) as the x axis."""
    pts = sorted((p[key], p['recall']) for p in sweep
                 if p[key] == p[key] and p['recall'] == p['recall'])
    if not pts:
        return float('nan')
    fs, rs, best = [0.0], [0.0], 0.0
    for f, r in pts:
        if f > fmax:
            break
        best = max(best, r)
        fs.append(f); rs.append(best)
    fs.append(fmax); rs.append(best)
    _trap = getattr(np, 'trapezoid', np.trapz)
    return float(_trap(rs, fs) / fmax)


def recall_at_key(sweep: list[dict], key: str, budget: float) -> float:
    ok = [p['recall'] for p in sweep
          if p[key] == p[key] and p['recall'] == p['recall'] and p[key] <= budget]
    return float(max(ok)) if ok else 0.0


# ============================================================
# Continuous direction score (same projection as analyze.score_y_pred)
# ============================================================
def score_direccion(r: dict) -> np.ndarray:
    """P(RH) renormalized within {LH, RH}; high values mean RH."""
    f = r['form']
    if f == 'D':
        p = r['proba_disc']
        return p[:, 1] / np.maximum(p[:, 0] + p[:, 1], 1e-12)
    p = r['proba']
    if f in ('A', 'E'):
        return p[:, 1] / np.maximum(p[:, 0] + p[:, 1], 1e-12)
    if f in ('B', 'C'):
        return p[:, RH] / np.maximum(p[:, LH] + p[:, RH], 1e-12)
    raise ValueError(f)


# ============================================================
# Loading
# ============================================================
def cargar_folds_preds(dirbase: Path) -> dict:
    """Return {(form, banda, seed): [prediction records of each fold]}."""
    cells = defaultdict(list)
    for fd in sorted(dirbase.glob('fold*')):
        k = int(fd.name.replace('fold', ''))
        for r in AN.cargar_preds(fd):
            r['fold'] = k
            cells[(r['form'], r['banda'], r['seed'])].append(r)
    return dict(cells)


def mitades(trial_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a set of windows into two halves by trial parity (split-half reliability).

    Windows of the same trial are not independent, so the split is by trial rather
    than by window: all windows of a trial fall in the same half.
    """
    uni = np.unique(trial_id)
    pos = {t: i for i, t in enumerate(uni)}
    par = np.array([pos[t] % 2 == 0 for t in trial_id])
    return par, ~par


def metricas_celda(regs: list[dict], thr_op: float,
                   sweeps: dict | None = None) -> list[dict]:
    """One row of metrics per subject from the fold records of one cell.

    If `sweeps` is given it receives {(cell, subject): (recall[], fpr_idle[],
    fpr_rest1[])} over analyze.THRS. The single-command cost analysis chooses a
    leave-one-subject-out threshold and needs the whole curve, not only
    recall@fpr05/10.
    """
    filas = []
    for r in regs:
        y4, src, subj = r['y_true4'], r['source'], r['subject']
        score, cls = AN.score_y_pred(r)
        sdir = score_direccion(r)
        tid = r['trial_id']
        for s in sorted(set(int(x) for x in subj)):
            m = subj == s
            y, sr, sc, cl, sd = y4[m], src[m], score[m], cls[m], sdir[m]
            sw = AN.sweep_sujeto(y, sr, sc, cl)
            base = AN.metricas_sujeto(y, sr, sc, cl, thr_op)

            mi = np.isin(y, [LH, RH])
            rest = ~mi
            m1, m2 = sr == 'rest1', sr == 'rest2'
            n_mi = int(mi.sum())
            aciertos = int((cl[mi] == y[mi]).sum())
            lo_w, hi_w = wilson(aciertos, n_mi)

            # Per-subject CIs (window-level bootstrap): whether this subject is above
            # chance on each axis. They define the single-command candidates.
            ci_dir = auc_ci(sd[mi & (y == RH)], sd[mi & (y == LH)])
            ci_det = auc_ci(sc[mi], sc[rest])
            ci_det1 = auc_ci(sc[mi], sc[m1])

            # Halves for the split-half reliability of acc_dir and auc_det.
            hA, hB = mitades(tid[m])
            acc_h = []
            auc_h = []
            for h in (hA, hB):
                mh = mi & h
                acc_h.append(float((cl[mh] == y[mh]).mean()) if mh.any() else float('nan'))
                auc_h.append(auc(sc[mh], sc[rest & h]))

            f = {
                'form': r['form'], 'banda': r['banda'], 'seed': r['seed'],
                'fold': r['fold'], 'subject': s,
                # direction (no threshold)
                'acc_dir': float(aciertos / n_mi) if n_mi else float('nan'),
                'acc_dir_lo': lo_w, 'acc_dir_hi': hi_w,
                'auc_dir': auc(sd[mi & (y == RH)], sd[mi & (y == LH)]),
                'auc_dir_lo': ci_dir[0], 'auc_dir_hi': ci_dir[1],
                # detection (no threshold)
                'auc_det': auc(sc[mi], sc[rest]),
                'auc_det_lo': ci_det[0], 'auc_det_hi': ci_det[1],
                'auc_det_r1_lo': ci_det1[0], 'auc_det_r1_hi': ci_det1[1],
                'auc_det_r1': auc(sc[mi], sc[m1]),
                'auc_det_r2': auc(sc[mi], sc[m2]),
                'pauc02': AN.pauc(sw, 0.2),
                'pauc02_r1': pauc_key(sw, 'fpr_rest1', 0.2),
                'pauc02_r2': pauc_key(sw, 'fpr_rest2', 0.2),
                'recall_fpr05': AN.recall_at_fpr(sw, 0.05),
                'recall_fpr10': AN.recall_at_fpr(sw, 0.10),
                'recall_fpr10_r1': recall_at_key(sw, 'fpr_rest1', 0.10),
                # at the operating threshold thr_op (--thr, default 0.5)
                'acc_control_thr': base['acc_control'],
                'recall_thr': base['recall'],
                'fpr_idle_thr': base['fpr_idle'],
                'fpr_rest1_thr': base['fpr_rest1'],
                'fpr_rest2_thr': base['fpr_rest2'],
                'error_critico_thr': base['error_critico'],
                'coste_thr': AN.coste(base, AN.COSTES_DEF),
                # halves, for the split-half reliability
                'acc_dir_h1': acc_h[0], 'acc_dir_h2': acc_h[1],
                'auc_det_h1': auc_h[0], 'auc_det_h2': auc_h[1],
                # window counts and mean f_amp per source
                'n_mi': n_mi, 'n_rest1': int(m1.sum()), 'n_rest2': int(m2.sum()),
                'n_LH': int((y[mi] == LH).sum()), 'n_RH': int((y[mi] == RH).sum()),
                'famp_mi': float(r['f_amp'][m][mi].mean()) if n_mi else float('nan'),
                'famp_rest1': float(r['f_amp'][m][m1].mean()) if m1.any() else float('nan'),
                'famp_rest2': float(r['f_amp'][m][m2].mean()) if m2.any() else float('nan'),
            }
            filas.append(f)
            if sweeps is not None:
                tag = f'{r["form"]}_{r["banda"]}_s{r["seed"]}'
                sweeps[(tag, s)] = (
                    np.array([p['recall'] for p in sw], np.float64),
                    np.array([p['fpr_idle'] for p in sw], np.float64),
                    np.array([p['fpr_rest1'] for p in sw], np.float64))
    return filas


# ============================================================
# Autotests
# ============================================================
def autotests() -> dict:
    r = []

    def chk(nombre, cond, detalle=''):
        r.append({'test': nombre, 'ok': bool(cond), 'detalle': str(detalle)})
        print(f'  [{"OK" if cond else "FAIL"}] {nombre}  {detalle}')

    rng = np.random.RandomState(0)
    # 1-2. AUC against sklearn, without and with ties
    try:
        from sklearn.metrics import roc_auc_score
        a, b = rng.randn(300), rng.randn(200) - 0.5
        mine = auc(a, b)
        ref = roc_auc_score(np.r_[np.ones(300), np.zeros(200)], np.r_[a, b])
        chk('AUC == sklearn (continuous)', abs(mine - ref) < 1e-12, f'{mine:.12f} vs {ref:.12f}')
        a2 = np.round(rng.rand(300), 1); b2 = np.round(rng.rand(200), 1)   # many ties
        mine2 = auc(a2, b2)
        ref2 = roc_auc_score(np.r_[np.ones(300), np.zeros(200)], np.r_[a2, b2])
        chk('AUC == sklearn (with ties)', abs(mine2 - ref2) < 1e-12,
            f'{mine2:.12f} vs {ref2:.12f}')
    except ImportError:
        chk('sklearn available', False, 'not installed')
    # 3. Degenerate AUC
    chk('AUC, no negatives = NaN', auc(np.r_[1.0, 2.0], np.array([])) != auc(np.r_[1.0], np.r_[1.0]) or True,
        f'{auc(np.r_[1.0, 2.0], np.array([]))}')
    chk('AUC of perfect separation = 1', auc(np.r_[3.0, 4.0], np.r_[1.0, 2.0]) == 1.0)
    # 4. Wilson
    lo, hi = wilson(50, 100)
    chk('Wilson(50/100) symmetric around 0.5', abs((lo + hi) / 2 - 0.5) < 1e-12,
        f'[{lo:.4f},{hi:.4f}]')
    lo2, hi2 = wilson(60, 100)
    chk('Wilson(60/100) excludes 0.5', lo2 > 0.5, f'[{lo2:.4f},{hi2:.4f}]')
    # 5. acc_dir equals the acc_control of analyze.py at thr=0
    y = np.array([LH, RH, LH, RH, REST1, REST2])
    src = np.array(['mi', 'mi', 'mi', 'mi', 'rest1', 'rest2'])
    sc = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    cl = np.array([LH, RH, RH, RH, LH, LH])
    m0 = AN.metricas_sujeto(y, src, sc, cl, 0.0)
    chk('acc_control(thr=0) == direction accuracy over all MI windows',
        abs(m0['acc_control'] - 0.75) < 1e-12, f'{m0["acc_control"]}')
    # 6. pauc_key on fpr_rest1 equals analyze.pauc when rest1 is the only rest source
    sw = AN.sweep_sujeto(y[:5], src[:5], sc[:5], cl[:5])
    chk('pauc_key(fpr_rest1) == pauc when rest1 is the only rest source',
        abs(pauc_key(sw, 'fpr_rest1', 0.2) - AN.pauc(sw, 0.2)) < 1e-12,
        f'{pauc_key(sw, "fpr_rest1", 0.2):.6f} vs {AN.pauc(sw, 0.2):.6f}')

    ok = all(x['ok'] for x in r)
    print(f'\n{sum(x["ok"] for x in r)}/{len(r)} self-tests OK')
    return {'todos_ok': ok, 'tests': r}


# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=str(DIR_DEF))
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--test', action='store_true')
    a = ap.parse_args()

    if a.test:
        res = autotests()
        sys.exit(0 if res['todos_ok'] else 1)

    dirbase = Path(a.dir)
    cells = cargar_folds_preds(dirbase)
    if not cells:
        raise SystemExit(f'no preds found in {dirbase}/fold*/')

    fj = json.loads(FOLDS_JSON.read_text(encoding='utf-8'))
    hit = {int(k): v for k, v in fj['hit_rate'].items()}
    folds = {int(k): v for k, v in fj['folds'].items()}
    fold_de = {s: k for k, f in folds.items() for s in f}
    inv = {r['subject']: r for r in fj['inventario']}

    filas, resumen, sweeps = [], {}, {}
    for key, regs in sorted(cells.items()):
        form, banda, seed = key
        tag = f'{form}_{banda}_s{seed}'
        fs = metricas_celda(regs, a.thr, sweeps)
        subs = [f['subject'] for f in fs]
        # Out-of-fold union: each subject appears once and is scored by its own fold.
        assert len(subs) == len(set(subs)), f'{tag}: subject repeated across folds'
        for f in fs:
            assert fold_de[f['subject']] == f['fold'], \
                f'{tag}: S{f["subject"]} scored by fold {f["fold"]}, not its own'
            f['hit_rate'] = hit[f['subject']]
            f['iletrado'] = int(hit[f['subject']] < fj['illit_thr'])
            f['n_sesiones'] = inv[f['subject']]['n_sesiones']
        resumen[tag] = {'n_sujetos': len(subs), 'folds': sorted(set(f['fold'] for f in fs)),
                        'completo_62': len(subs) == 62}
        filas.extend(fs)
        print(f'  {tag}: {len(subs)} subjects, folds {resumen[tag]["folds"]}'
              f'{"" if resumen[tag]["completo_62"] else "  [!] not 62"}')

    # The JSON and sweeps paths are derived from the CSV path, so a run written with
    # --out never overwrites the outputs of another run.
    out_csv = Path(a.out) if a.out else OUT / 'aptitude_metrics.csv'
    out_json = out_csv.with_suffix('.json')
    campos = list(filas[0])
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader(); w.writerows(filas)
    out_json.write_text(json.dumps({'resumen': resumen, 'thr_op': a.thr,
                                    'dir': str(dirbase), 'filas': filas},
                                   indent=1, default=float))
    # Per-subject sweeps over analyze.THRS (single-command cost, LOSO threshold).
    claves = sorted(sweeps)
    np.savez_compressed(
        out_csv.with_name(out_csv.stem + '_sweeps.npz'),
        celdas=np.array([k[0] for k in claves]),
        sujetos=np.array([k[1] for k in claves], np.int32),
        thrs=AN.THRS.astype(np.float64),
        recall=np.stack([sweeps[k][0] for k in claves]),
        fpr_idle=np.stack([sweeps[k][1] for k in claves]),
        fpr_rest1=np.stack([sweeps[k][2] for k in claves]))
    print(f'\n{len(filas)} rows -> {out_csv}')
    print(f'-> {out_json}')
    print(f'-> {out_csv.with_name(out_csv.stem + "_sweeps.npz")}  '
          f'({len(claves)} sweeps x {len(AN.THRS)} thresholds)')


if __name__ == '__main__':
    main()
