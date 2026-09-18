"""Aptitude study: online aptitude vs the direction and detection axes.

The online hit rate (`hit_rate`) is the aptitude measure. Reads the per-subject
out-of-fold metrics of aptitude_metrics.py (outputs/aptitude_metrics.csv and the
companion outputs/aptitude_metrics_sweeps.npz) and outputs/aptitude_folds.json, and
analyses every cell (formulation, band, seed) that has at least --min-n subjects
(default 62). Bootstrap CIs are over subjects (aptitude_common.py).

Per cell:
  R1   Spearman and Pearson correlation of hit_rate with each direction metric
       (MET_DIR) and detection metric (MET_DET), with permutation p-values, and the
       paired difference of two correlations computed on the same bootstrap resamples
       of subjects (PARES). Two overlapping CIs do not show that two correlations
       differ; the paired difference has its own CI.
  R2   correlations between the direction and detection metrics across subjects.
  R3   low-aptitude subjects (iletrado = 1: hit rate below illit_thr of
       aptitude_folds.json) vs the rest, with Holm-adjusted permutation p-values;
       group means restricted to the subjects of fold 0; and the per-subject values
       of the low-aptitude subjects.
  R4   candidates for a single-command interface: subjects whose detection is above
       chance and whose direction is not, judged by their own CIs, with a Fisher exact
       test of the 2x2 table (against all rest and against REST1 only).
  R4b  cost of a single-command interface with a leave-one-subject-out threshold,
       compared with never firing.
  R7   split-half reliability (Spearman-Brown) of acc_dir and auc_det. A null
       correlation is only informative if the metric is reliable.
Across cells, paired by subject (seed 42 cells):
  R5   band contrast bb - mb for each formulation.
  R6   formulation contrasts B - D, B - A and D - A for each band.

Writes outputs/aptitude_correlations.json (or --out).

Usage:
    python aptitude_correlations.py
    python aptitude_correlations.py --csv outputs/aptitude_metrics.csv --boot 10000
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
import aptitude_common as S                                          # noqa: E402

OUT = HERE / 'outputs'
CSV_DEF = OUT / 'aptitude_metrics.csv'
FOLDS_JSON = OUT / 'aptitude_folds.json'

CELDA_PRIMARIA = 'B_bb_s42'      # formulation B (LH/RH/IDLE), band 0.5-40 Hz, seed 42

# Metrics of each axis. The primary pair is (auc_dir, auc_det): both are threshold-free
# AUCs on the same scale, so the difference of their correlations is a fair contrast.
# acc_control_thr only counts MI windows that pass the gate at the operating threshold,
# so it also depends on detection; acc_dir and auc_dir do not.
MET_DIR = ['acc_dir', 'auc_dir', 'acc_control_thr']
MET_DET = ['auc_det', 'auc_det_r1', 'auc_det_r2', 'pauc02', 'pauc02_r1',
           'recall_fpr10']
PARES = [('auc_dir', 'auc_det'),            # primary
         ('acc_dir', 'pauc02'),
         ('acc_control_thr', 'pauc02'),     # thresholded direction metric
         ('auc_dir', 'auc_det_r1'),         # detection against REST1 (cued rest) only
         ('acc_dir', 'auc_det'),
         ('auc_dir', 'pauc02')]


# ============================================================
def cargar(csv_fp: Path) -> dict:
    filas = []
    with open(csv_fp, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            d = {}
            for k, v in r.items():
                if k in ('form', 'banda'):
                    d[k] = v
                elif v == '' or v is None:
                    d[k] = float('nan')
                else:
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
            filas.append(d)
    por_celda = defaultdict(list)
    for f in filas:
        por_celda[f'{f["form"]}_{f["banda"]}_s{int(f["seed"])}'].append(f)
    return {k: sorted(v, key=lambda r: r['subject']) for k, v in por_celda.items()}


def col(rows, k) -> np.ndarray:
    return np.array([r[k] for r in rows], float)


# ============================================================
# R1: correlations with the hit rate and paired differences of correlations
# ============================================================
def r1_correlaciones(rows, B, metodos=('spearman', 'pearson')) -> dict:
    hit = col(rows, 'hit_rate')
    out = {'n_sujetos': len(rows), 'correlaciones': {}, 'diferencias': {}}
    for met in metodos:
        for k in MET_DIR + MET_DET:
            c = S.corr_ci(hit, col(rows, k), met, B)
            c['p_permutacion'] = S.perm_p(hit, col(rows, k), met, B)
            c['eje'] = 'direccion' if k in MET_DIR else 'deteccion'
            out['correlaciones'][f'{met}|{k}'] = c
        for a, b in PARES:
            out['diferencias'][f'{met}|{a}-{b}'] = S.corr_dif_ci(
                hit, col(rows, a), col(rows, b), met, B)
    return out


# ============================================================
# R2: correlation between the direction and detection metrics
# ============================================================
def r2_desacople(rows, B) -> dict:
    out = {}
    for a in MET_DIR:
        for b in MET_DET:
            out[f'{a}~{b}'] = S.corr_ci(col(rows, a), col(rows, b), 'spearman', B)
    out['acc_dir~auc_dir'] = S.corr_ci(col(rows, 'acc_dir'), col(rows, 'auc_dir'),
                                       'spearman', B)
    out['auc_det~auc_det_r1'] = S.corr_ci(col(rows, 'auc_det'),
                                          col(rows, 'auc_det_r1'), 'spearman', B)
    return out


# ============================================================
# R3: low-aptitude subjects vs the rest
# ============================================================
def r3_iletrados(rows, B, fold0: list[int]) -> dict:
    il = [r for r in rows if r['iletrado'] == 1]
    no = [r for r in rows if r['iletrado'] == 0]
    out = {'n_iletrados': len(il), 'n_resto': len(no),
           'iletrados': sorted(int(r['subject']) for r in il), 'metricas': {}}
    for k in MET_DIR + MET_DET + ['recall_fpr05', 'fpr_idle_thr', 'recall_thr']:
        d = S.boot_dif_no_pareada(col(il, k), col(no, k), B)
        d['p_permutacion'] = S.perm_p_grupos(col(il, k), col(no, k), B)
        out['metricas'][k] = d
    # Holm correction over the family of all metrics compared in this section.
    hp = S.holm({k: v['p_permutacion'] for k, v in out['metricas'].items()})
    for k, v in out['metricas'].items():
        v['p_holm'] = hp[k]
    out['n_familia_holm'] = len(hp)

    # Group means restricted to the subjects of fold 0 (the canonical validation split).
    f0 = [r for r in rows if int(r['subject']) in fold0]
    il2 = [r for r in f0 if r['iletrado'] == 1]
    no2 = [r for r in f0 if r['iletrado'] == 0]
    out['n2_fold0'] = {
        'sujetos_iletrados': sorted(int(r['subject']) for r in il2),
        'n_il': len(il2), 'n_no': len(no2),
        'metricas': {k: {'iletrados': float(np.mean(col(il2, k))),
                         'resto': float(np.mean(col(no2, k))),
                         'dif': float(np.mean(col(il2, k)) - np.mean(col(no2, k)))}
                     for k in ['pauc02', 'acc_dir', 'acc_control_thr', 'auc_det',
                               'auc_dir']}}

    # Per-subject values of the low-aptitude subjects, by increasing hit rate.
    out['por_sujeto_iletrado'] = [
        {'subject': int(r['subject']), 'hit_rate': r['hit_rate'],
         **{k: r[k] for k in ['acc_dir', 'acc_dir_lo', 'acc_dir_hi', 'auc_dir',
                              'auc_det', 'auc_det_lo', 'auc_det_hi', 'auc_det_r1',
                              'pauc02', 'recall_fpr10', 'n_mi']}}
        for r in sorted(il, key=lambda x: x['hit_rate'])]
    return out


# ============================================================
# R4: candidates for a single-command interface
# ============================================================
def r4_candidatos(rows) -> dict:
    """Classify each subject with its own 95% CIs.

      direction usable  <=> lower bound of the Wilson CI of acc_dir > 0.5
      detection usable  <=> lower bound of the window-bootstrap CI of the
                            detection AUC > 0.5

    Subjects with usable detection and no usable direction are candidates for a
    single-command interface (fire / do not fire). The classification is done with
    the detection AUC against all rest (auc_det) and against REST1 only (auc_det_r1).
    """
    def clasif(r, det_key, det_lo):
        return (bool(r['acc_dir_lo'] > 0.5), bool(r[det_lo] > 0.5))

    out = {}
    for nm, det_key, det_lo in [('todo_reposo', 'auc_det', 'auc_det_lo'),
                                ('rest1_honesto', 'auc_det_r1', 'auc_det_r1_lo')]:
        tabla = {'dir_si_det_si': [], 'dir_si_det_no': [],
                 'dir_no_det_si': [], 'dir_no_det_no': []}
        for r in rows:
            dsi, tsi = clasif(r, det_key, det_lo)
            k = f'dir_{"si" if dsi else "no"}_det_{"si" if tsi else "no"}'
            tabla[k].append(int(r['subject']))
        n = {k: len(v) for k, v in tabla.items()}
        # Fisher exact test of independence of the two classifications across subjects.
        p = S.fisher_exacto(n['dir_si_det_si'], n['dir_si_det_no'],
                            n['dir_no_det_si'], n['dir_no_det_no'])
        cand = tabla['dir_no_det_si']
        detalle = [{'subject': int(r['subject']), 'hit_rate': r['hit_rate'],
                    'iletrado': int(r['iletrado']), 'acc_dir': r['acc_dir'],
                    'acc_dir_ic': [r['acc_dir_lo'], r['acc_dir_hi']],
                    det_key: r[det_key], 'det_ic': [r[det_lo], r[det_lo.replace('_lo', '_hi')]],
                    'recall_fpr10': r['recall_fpr10'], 'pauc02': r['pauc02']}
                   for r in rows if int(r['subject']) in cand]
        out[nm] = {'tabla': tabla, 'n': n, 'fisher_p': p,
                   'n_candidatos_un_comando': len(cand),
                   'candidatos': sorted(cand),
                   'detalle_candidatos': sorted(detalle, key=lambda x: -x[det_key])}
    return out


# ============================================================
# R4b: cost of a single-command interface
# ============================================================
def r4b_coste_un_comando(rows, sweeps_fp: Path, tag: str, B: int) -> dict:
    """Cost model without the left/right inversion term.

    With a single command there is no direction to invert, so the only errors are
    false positives at rest and missed commands:

        coste_1cmd = w_fp * fpr_idle + w_perdido * (1 - recall)     (w_perdido = 1)
        coste_silencio = w_perdido                                  (never fire)

    For each w_fp in (1, 2, 5, 10), the threshold applied to a subject minimizes the
    mean cost of the other subjects (leave one subject out), as in analyze.py. The
    per-subject oracle threshold is also reported as an upper bound.
    """
    if not sweeps_fp.exists():
        return {'error': f'falta {sweeps_fp.name}'}
    z = np.load(sweeps_fp, allow_pickle=True)
    m = z['celdas'] == tag
    subs = z['sujetos'][m]
    rec, fpr = z['recall'][m], z['fpr_idle'][m]
    thrs = z['thrs']
    orden = np.argsort(subs)
    subs, rec, fpr = subs[orden], rec[orden], fpr[orden]
    n = len(subs)

    out = {'tag': tag, 'n_sujetos': int(n), 'w_perdido': 1.0, 'por_peso': {}}
    cand = set(r4_cand_ids(rows))
    for w_fp in (1.0, 2.0, 5.0, 10.0):
        coste = w_fp * fpr + (1.0 - rec)                 # (n, n_thr)
        c_silencio = 1.0
        # per-subject oracle threshold (upper bound)
        j_or = np.nanargmin(coste, axis=1)
        c_or = coste[np.arange(n), j_or]
        # LOSO: threshold chosen on the other n - 1 subjects
        c_loso, j_loso = np.empty(n), np.empty(n, int)
        for i in range(n):
            otros = np.r_[0:i, i + 1:n]
            j = int(np.nanargmin(np.nanmean(coste[otros], axis=0)))
            j_loso[i], c_loso[i] = j, coste[i, j]
        gana = c_loso < c_silencio
        filas = [{'subject': int(subs[i]), 'coste_loso': float(c_loso[i]),
                  'coste_oraculo': float(c_or[i]), 'thr_loso': float(thrs[j_loso[i]]),
                  'recall': float(rec[i, j_loso[i]]), 'fpr': float(fpr[i, j_loso[i]]),
                  'gana_al_silencio': bool(gana[i]),
                  'candidato_un_comando': bool(int(subs[i]) in cand)}
                 for i in range(n)]
        out['por_peso'][f'w_fp={w_fp:g}'] = {
            'coste_silencio': c_silencio,
            'coste_loso_medio': S.boot_media(c_loso.tolist(), B),
            'coste_oraculo_medio': S.boot_media(c_or.tolist(), B),
            'n_gana_al_silencio': int(gana.sum()),
            'sujetos_que_ganan': sorted(int(subs[i]) for i in range(n) if gana[i]),
            'n_candidatos_que_ganan': int(sum(1 for f in filas
                                              if f['gana_al_silencio'] and
                                              f['candidato_un_comando'])),
            'n_candidatos': len(cand),
            'contraste_loso_menos_silencio': S.boot_dif_pareada(
                c_loso.tolist(), [c_silencio] * n, B),
            'por_sujeto': filas}
    return out


def r4_cand_ids(rows) -> list[int]:
    return r4_candidatos(rows)['todo_reposo']['candidatos']


# ============================================================
# R5/R6: band and formulation contrasts, paired by subject
# ============================================================
def contraste_pareado(rows_a, rows_b, B, claves) -> dict:
    sa = {int(r['subject']): r for r in rows_a}
    sb = {int(r['subject']): r for r in rows_b}
    com = sorted(set(sa) & set(sb))
    out = {'n_comunes': len(com)}
    for k in claves:
        out[k] = S.boot_dif_pareada([sa[s][k] for s in com],
                                    [sb[s][k] for s in com], B)
    return out


# ============================================================
# R7: split-half reliability and attenuation
# ============================================================
def r7_fiabilidad(rows, B, hr_json: dict | None) -> dict:
    """Split-half reliability of acc_dir and auc_det (Spearman-Brown corrected).

    The reliability of the hit rate is read from `hr_json` when given. main() passes
    None, so rel_hit is NaN and so are `techo` and `r_desatenuado`.
    """
    out = {}
    for k, h1, h2 in [('acc_dir', 'acc_dir_h1', 'acc_dir_h2'),
                      ('auc_det', 'auc_det_h1', 'auc_det_h2')]:
        c = S.corr_ci(col(rows, h1), col(rows, h2), 'pearson', B)
        out[f'rel_{k}'] = {'r_mitades': c['r'], 'ic95': [c['lo'], c['hi']],
                           'spearman_brown': S.sb(c['r']), 'n': c['n']}
    rel_hit = float('nan')
    if hr_json:
        rel_hit = hr_json['fiabilidad_runs_impar_par']['spearman_brown']
        out['rel_hit_rate'] = {
            'r_mitades': hr_json['fiabilidad_runs_impar_par']['pearson'],
            'spearman_brown': rel_hit,
            'sesion5_vs_6': hr_json['fiabilidad_sesion5_vs_6']['pearson']}
    hit = col(rows, 'hit_rate')
    for k in ['acc_dir', 'auc_det']:
        rel_m = out[f'rel_{k}']['spearman_brown']
        r_obs = S.pearson(hit, col(rows, k))
        out[f'atenuacion_{k}'] = {
            'r_observado_pearson': r_obs,
            'rel_metrica': rel_m, 'rel_hit': rel_hit,
            'techo': S.techo_atenuacion(rel_m, rel_hit),
            'r_desatenuado': S.desatenuar(r_obs, rel_m, rel_hit)}
    return out


# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', default=str(CSV_DEF))
    ap.add_argument('--boot', type=int, default=S.B_BOOT)
    ap.add_argument('--celda', default=CELDA_PRIMARIA)
    ap.add_argument('--out', default=None)
    ap.add_argument('--min-n', type=int, default=62,
                    help='minimum number of subjects for a cell to count as complete. '
                         'Lower it only to test the pipeline on unfinished folds; '
                         'the study uses all 62.')
    a = ap.parse_args()

    celdas = cargar(Path(a.csv))
    fj = json.loads(FOLDS_JSON.read_text(encoding='utf-8'))
    fold0 = fj['folds']['0']

    completas = {k: v for k, v in celdas.items() if len(v) >= a.min_n}
    print(f'cells found: {sorted(celdas)}')
    print(f'complete cells (>={a.min_n} subjects): {sorted(completas)}')
    if a.min_n != 62:
        print(f'[!!] --min-n={a.min_n}: pipeline test, not the full study')
    if not completas:
        raise SystemExit('no cell has the required number of subjects (--min-n): '
                         'training is incomplete')
    primaria = a.celda if a.celda in completas else sorted(completas)[0]
    if primaria != a.celda:
        print(f'[!] {a.celda} incomplete -> primary cell = {primaria}')

    res = {'celda_primaria': primaria, 'celdas_completas': sorted(completas),
           'celdas_incompletas': sorted(set(celdas) - set(completas)),
           'boot': a.boot, 'n_folds': fj['n_folds'],
           'metricas_direccion': MET_DIR, 'metricas_deteccion': MET_DET,
           'pares_contrastados': [list(p) for p in PARES],
           'por_celda': {}}

    for tag, rows in sorted(completas.items()):
        print(f'\n--- {tag} ---', flush=True)
        c = {'R1_correlaciones': r1_correlaciones(rows, a.boot),
             'R2_desacople': r2_desacople(rows, a.boot),
             'R3_iletrados': r3_iletrados(rows, a.boot, fold0),
             'R4_candidatos': r4_candidatos(rows),
             'R4b_coste_un_comando': r4b_coste_un_comando(
                 rows, Path(a.csv).with_name(Path(a.csv).stem + '_sweeps.npz'),
                 tag, a.boot),
             'R7_fiabilidad': r7_fiabilidad(rows, a.boot, None)}
        res['por_celda'][tag] = c
        d = c['R1_correlaciones']['diferencias']['spearman|auc_dir-auc_det']
        print(f'  rho(hit,auc_dir)={d["r1"]:+.3f}  rho(hit,auc_det)={d["r2"]:+.3f}  '
              f'dif={d["dif"]:+.3f} IC[{d["lo"]:+.3f},{d["hi"]:+.3f}] {d["veredicto"]}',
              flush=True)

    # R5: band (bb - mb) and R6: formulation, paired by subject
    claves = MET_DIR + ['auc_det', 'auc_det_r1', 'pauc02', 'recall_fpr10']
    res['R5_banda'] = {}
    for form in sorted(set(k.split('_')[0] for k in completas)):
        ab, am = f'{form}_bb_s42', f'{form}_mb_s42'
        if ab in completas and am in completas:
            res['R5_banda'][f'{form}: bb - mb'] = contraste_pareado(
                completas[ab], completas[am], a.boot, claves)
    res['R6_formulacion'] = {}
    formas = sorted(set(k.split('_')[0] for k in completas))
    for banda in ('bb', 'mb'):
        for fa, fb in [('B', 'D'), ('B', 'A'), ('D', 'A')]:
            ta, tb = f'{fa}_{banda}_s42', f'{fb}_{banda}_s42'
            if ta in completas and tb in completas:
                res['R6_formulacion'][f'{banda}: {fa} - {fb}'] = contraste_pareado(
                    completas[ta], completas[tb], a.boot, claves)

    out_fp = Path(a.out) if a.out else OUT / 'aptitude_correlations.json'
    out_fp.write_text(json.dumps(res, indent=1, default=float))
    print(f'\n-> {out_fp}')


if __name__ == '__main__':
    main()
