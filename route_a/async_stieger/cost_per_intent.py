"""Per-window vs per-intent evaluation of the asynchronous formulations.

Reads outputs/streams/win_<cell>.npz (written by infer_streams.py) and evaluates both
units with accumulate.py on exactly the same windows and probabilities:
  per window   every window in the active region is an independent decision;
  per intent   every trial is one event, decided by the real-time accumulator.

Sections:
  1. Window vs intent at two operating points: minimum cost, and a budget of false
     commands per minute of rest (PRESUPUESTOS_FP; main budget 1/min).
  2. Cost vs dwell time, the parameter that interpolates between the two units.
  3. Sensitivity to one design choice at a time (strategy, refractory period,
     attribution convention, trial set).
  4. Window vs intent at fixed thresholds, paired by subject.
  5. Sweep of the cost weights with the necessary condition a > 1 - 1/w_inv for firing
     to beat silence (a = direction accuracy), and the highest direction accuracy
     reached while more than 2% of the MI trials (MI windows, in the window unit)
     produce a command.
  6. Per-subject results of the main cell at the main budget (LOSO thresholds), and
     subject-level bootstrap CIs (B=--boot) of the per-intent metrics.
  7. Subject-paired contrasts between formulations (B-A, C-B, D-B) and bands (bb-mb),
     in both units.

Under the default weights the minimum cost is to stay silent, so comparing
configurations at their cost optimum compares silences. The primary operating point
therefore fixes a budget of false commands per minute and maximizes the fraction of
intents that produce the correct command; the cost is still reported, with a
`degenerado` flag.

Writes outputs/cost_per_intent.json.

Usage:
    python cost_per_intent.py --boot 10000
    python cost_per_intent.py --rapido        # coarse threshold grid, for debugging
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from accumulate import (WIN_DIR, OUT, COSTES_DEF, CFG_REF, THRS, PRESUPUESTOS_FP,
                        cargar, evaluar_intencion, evaluar_ventana, coste, agrega,
                        tabla_barrido, seleccionar, seleccionar_por_presupuesto,
                        boot_media, boot_pareado)

CELDAS = ['A_bb_W2_r1_s42', 'A_mb_W2_r1_s42', 'B_bb_W2_r1_s42', 'B_mb_W2_r1_s42',
          'C_bb_W2_r1_s42', 'C_mb_W2_r1_s42', 'D_bb_W2_r1_s42', 'D_mb_W2_r1_s42',
          'B_bb_W2_r1_s43', 'B_mb_W2_r1_s43']
PRINCIPAL = 'B_bb_W2_r1_s42'
PRESUP_PRIN = 1.0                     # 1 false command per minute of rest

CAMPOS = ['p_hit', 'p_inv', 'p_lost', 'acc_dir', 'fpr_episodio', 'fp_min',
          'latencia_ms', 'latencia_p90', 'prematuras_por_trial',
          'prem_precue_por_trial', 'prem_cue_por_trial',
          'repeticiones_por_emision', 'min_reposo', 'n_mi', 'n_rest']


def resumen(por_suj: dict, w=COSTES_DEF) -> dict:
    """Mean over subjects of every field in CAMPOS, plus the mean cost."""
    r = {c: agrega(por_suj, c) for c in CAMPOS}
    r['coste'] = float(np.nanmean([coste(por_suj[s], w) for s in por_suj])) \
        if por_suj else float('nan')
    r['n_sujetos'] = len(por_suj)
    return r


def punto_coste(b: dict, w=COSTES_DEF, thrs=THRS) -> dict:
    """Minimum-cost operating point (oracle and LOSO thresholds).

    degenerado_*     fewer than 2% of the MI trials produce a command (optimum = silence)
    en_borde         some LOSO threshold is the last value of the grid
    bate_silencio_*  cost below that of never firing (w['comando_perdido'])
    """
    ro, rh = resumen(b['oraculo'], w), resumen(b['honesto'], w)
    tl = [t for t in b['thr_loso'].values() if t is not None]
    return dict(thr_oraculo=b['t_oraculo'], oraculo=ro, honesto=rh,
                thr_loso=b['thr_loso'],
                degenerado_oraculo=bool(ro['p_hit'] + ro['p_inv'] < 0.02),
                degenerado_honesto=bool(rh['p_hit'] + rh['p_inv'] < 0.02),
                en_borde=bool(tl and max(tl) >= float(thrs[-1])),
                bate_silencio_oraculo=bool(ro['coste'] < w['comando_perdido']),
                bate_silencio_honesto=bool(rh['coste'] < w['comando_perdido']))


def punto_presupuesto(b: dict, w=COSTES_DEF) -> dict:
    """Operating point under a budget of false commands per minute of rest."""
    return dict(presupuesto=b['presupuesto'], factible=b['factible'],
                thr_oraculo=b['t_oraculo'], thr_loso=b['thr_loso'],
                oraculo=resumen(b['oraculo'], w), honesto=resumen(b['honesto'], w))


def _fmt(r: dict) -> str:
    return (f'hit={r["p_hit"]:.3f} inv={r["p_inv"]:.3f} accdir={r["acc_dir"]:.3f} '
            f'fp/min={r["fp_min"]:6.2f} lat={r["latencia_ms"]:6.0f}ms '
            f'cost={r["coste"]:6.3f}')


# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--rapido', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    thrs = THRS[::3] if args.rapido else THRS
    rng = np.random.RandomState(args.seed)
    res: dict = {'config': dict(n_thrs=len(thrs), boot=args.boot, cfg_ref=CFG_REF,
                                costes=COSTES_DEF, presupuesto_principal=PRESUP_PRIN,
                                celda_principal=PRINCIPAL)}
    t0 = time.time()

    disponibles = [c for c in CELDAS if (WIN_DIR / f'win_{c}.npz').exists()]
    print('=' * 92)
    print('COST PER INTENT vs PER WINDOW')
    print(f'  cells: {len(disponibles)}/{len(CELDAS)}   thresholds: {len(thrs)}   '
          f'bootstrap B={args.boot} (subject level)')
    print(f'  reference config: {CFG_REF}')
    print(f'  costs: {COSTES_DEF}   main budget: {PRESUP_PRIN} FP/min')
    print('=' * 92, flush=True)

    datos = {c: cargar(WIN_DIR / f'win_{c}.npz') for c in disponibles}
    d0 = datos[PRINCIPAL]

    # ---------------------------------------------------------
    # 1. Window vs intent
    # ---------------------------------------------------------
    print('\n### 1. Units: window vs intent ###', flush=True)
    res['unidades'] = {}
    tablas = {}
    for c in disponibles:
        d = datos[c]
        tv = tabla_barrido(d, 'ventana', CFG_REF, thrs=thrs)
        ti = tabla_barrido(d, 'intencion', CFG_REF, thrs=thrs)
        tablas[c] = dict(ventana=tv, intencion=ti)
        fila = {}
        for u, tab in (('ventana', tv), ('intencion', ti)):
            fila[u] = dict(
                coste=punto_coste(seleccionar(tab), thrs=thrs),
                presupuesto={str(b): punto_presupuesto(
                    seleccionar_por_presupuesto(tab, b)) for b in PRESUPUESTOS_FP})
        res['unidades'][c] = fila
        res.setdefault('honesto_por_sujeto', {})[c] = {
            u: {int(s): v for s, v in
                seleccionar_por_presupuesto(tablas[c][u], PRESUP_PRIN)['honesto'].items()}
            for u in ('ventana', 'intencion')}
        for u in ('ventana', 'intencion'):
            pc = fila[u]['coste']
            pb = fila[u]['presupuesto'][str(PRESUP_PRIN)]
            mudo = ' [mute]' if pc['degenerado_honesto'] else ''
            print(f'  {c:18s} {u:9s} cost  {_fmt(pc["honesto"])}{mudo}', flush=True)
            print(f'  {"":18s} {"":9s} FP<={PRESUP_PRIN:g}/min '
                  f'{_fmt(pb["honesto"]) if pb["factible"] else "not feasible"}',
                  flush=True)
        if c == PRINCIPAL:
            res['curvas'] = {u: {str(t): resumen(tab[t]) for t in tab}
                             for u, tab in (('ventana', tv), ('intencion', ti))}
    print(f'  -> {time.time()-t0:.0f}s', flush=True)

    # window unit with REST1 and REST2 as rest, the rest set of analyze.py
    tv2 = tabla_barrido(d0, 'ventana', CFG_REF, thrs=thrs, incluir_rest2=True)
    res['ventana_con_rest2'] = punto_coste(seleccionar(tv2), thrs=thrs)
    print(f'  [window, rest = REST1 + REST2 as in analyze.py] '
          f'{_fmt(res["ventana_con_rest2"]["honesto"])}', flush=True)

    # ---------------------------------------------------------
    # 2. Cost vs dwell time
    # ---------------------------------------------------------
    print('\n### 2. Cost vs dwell (main cell) ###', flush=True)
    dwells = [124, 248, 372, 496, 600, 744, 992, 1240, 1984]
    res['curva_dwell'] = {}
    for dm in dwells:
        tab = tabla_barrido(d0, 'intencion', dict(CFG_REF, dwell_ms=dm), thrs=thrs)
        pc = punto_coste(seleccionar(tab), thrs=thrs)
        pb = punto_presupuesto(seleccionar_por_presupuesto(tab, PRESUP_PRIN))
        res['curva_dwell'][dm] = dict(coste=pc, presupuesto=pb)
        if dm == 124:
            # Dwell of one window: the per-window decision rule, counted per trial.
            # Against dwell=600 it isolates the effect of accumulating; against the
            # window unit it isolates the effect of the counting unit (p_hit has a
            # different denominator in each unit: windows vs intents).
            res['curvas']['intencion_dwell1'] = {str(t): resumen(tab[t]) for t in tab}
        print(f'  dwell={dm:5d}ms  cost  {_fmt(pc["honesto"])}'
              f'{" [mute]" if pc["degenerado_honesto"] else ""}', flush=True)
        print(f'  {"":14s}  FP<=1/min {_fmt(pb["honesto"]) if pb["factible"] else "not feasible"}',
              flush=True)
    res['curva_dwell_ref_ventana'] = dict(
        coste=punto_coste(seleccionar(tablas[PRINCIPAL]['ventana']), thrs=thrs),
        presupuesto=punto_presupuesto(seleccionar_por_presupuesto(
            tablas[PRINCIPAL]['ventana'], PRESUP_PRIN)))

    # ---------------------------------------------------------
    # 3. Sensitivity
    # ---------------------------------------------------------
    print('\n### 3. Sensitivity (metric: p_hit at FP<=1/min, not the degenerate cost) ###',
          flush=True)
    # One change at a time with respect to CFG_REF. 'conv=estricta': a window counts
    # in a region only if it lies entirely inside it. 'todos_los_trials': also the LR
    # trials without a valid MI window (mi_ok False).
    variantes = {
        'ref': dict(cfg=CFG_REF, kw={}),
        'strategy=ema': dict(cfg=dict(CFG_REF, strategy='ema'), kw={}),
        'strategy=bayes': dict(cfg=dict(CFG_REF, strategy='bayes'), kw={}),
        'strategy=nofm': dict(cfg=dict(CFG_REF, strategy='nofm'), kw={}),
        'refractory=0': dict(cfg=dict(CFG_REF, refractory_ms=0), kw={}),
        'refractory=2000': dict(cfg=dict(CFG_REF, refractory_ms=2000), kw={}),
        'conv=estricta': dict(cfg=CFG_REF, kw=dict(conv='estricta')),
        'todos_los_trials': dict(cfg=CFG_REF, kw=dict(solo_mi_ok=False)),
    }
    res['sensibilidad'] = {}
    base = None
    for nom, v in variantes.items():
        tab = tabla_barrido(d0, 'intencion', v['cfg'], thrs=thrs, **v['kw'])
        pb = punto_presupuesto(seleccionar_por_presupuesto(tab, PRESUP_PRIN))
        pc = punto_coste(seleccionar(tab), thrs=thrs)
        res['sensibilidad'][nom] = dict(presupuesto=pb, coste=pc)
        h = pb['honesto']
        if nom == 'ref':
            base = h.get('p_hit', np.nan)
        print(f'  {nom:20s} p_hit={h.get("p_hit", float("nan")):.3f} '
              f'(d={h.get("p_hit", np.nan)-base:+.3f}) accdir={h.get("acc_dir", float("nan")):.3f} '
              f'fp/min={h.get("fp_min", float("nan")):5.2f} '
              f'lat={h.get("latencia_ms", float("nan")):6.0f}ms', flush=True)

    # ---------------------------------------------------------
    # 4. Window vs intent at fixed thresholds (d_* = intent - window)
    # ---------------------------------------------------------
    print('\n### 4. Window vs intent at fixed thresholds ###', flush=True)
    res['descomposicion'] = {}
    for t in [0.50, 0.70, 0.90, 0.95, 0.99, 0.999]:
        mv, mi = evaluar_ventana(d0, t), evaluar_intencion(d0, CFG_REF, t)
        rv, ri = resumen(mv), resumen(mi)
        res['descomposicion'][t] = dict(
            ventana=rv, intencion=ri,
            d_accdir=boot_pareado(mv, mi, 'acc_dir', args.boot, rng),
            d_fpmin=boot_pareado(mv, mi, 'fp_min', args.boot, rng),
            d_coste=boot_pareado({s: dict(x=coste(mv[s])) for s in mv},
                                 {s: dict(x=coste(mi[s])) for s in mi},
                                 'x', args.boot, rng))
        da = res['descomposicion'][t]['d_accdir']
        print(f'  thr={t:.3f}  WIN accdir={rv["acc_dir"]:.3f} fp/min={rv["fp_min"]:7.2f} '
              f'cost={rv["coste"]:6.3f}  |  INT accdir={ri["acc_dir"]:.3f} '
              f'fp/min={ri["fp_min"]:6.2f} cost={ri["coste"]:6.3f}  |  '
              f'daccdir={da["media"]:+.3f} CI[{da["ic"][0]:+.3f},{da["ic"][1]:+.3f}] '
              f'{da["veredicto"]}', flush=True)

    # ---------------------------------------------------------
    # 5. Cost weights and the analytic condition
    # ---------------------------------------------------------
    # cost = w_inv*p_inv + (1 - p_hit - p_inv) + w_fp*fpr; never firing costs 1
    #   => firing wins <=> p_hit*[a - (w_inv-1)(1-a)]/a > w_fp*fpr,  a = acc_dir
    #   The bracket is positive only if a > 1 - 1/w_inv (necessary condition).
    print('\n### 5. Cost weights (a_req = 1-1/w_inv, necessary condition) ###',
          flush=True)
    res['pesos'] = {}
    ti0, tv0 = tablas[PRINCIPAL]['intencion'], tablas[PRINCIPAL]['ventana']
    for w_inv in [1.0, 2.0, 3.0, 5.0, 10.0]:
        a_req = 1.0 - 1.0 / w_inv
        for w_fp in [1.0, 5.0, 10.0]:
            w = {'inversion': w_inv, 'comando_perdido': 1.0, 'falso_positivo': w_fp}
            pi = punto_coste(seleccionar(ti0, w), w, thrs)
            pv = punto_coste(seleccionar(tv0, w), w, thrs)
            res['pesos'][f'inv{w_inv:g}_fp{w_fp:g}'] = dict(
                intencion=pi, ventana=pv, a_requerida=a_req)
            print(f'  w_inv={w_inv:4.1f} w_fp={w_fp:4.1f} a_req={a_req:.2f}  '
                  f'INT cost={pi["honesto"]["coste"]:6.3f} '
                  f'{"fires" if pi["bate_silencio_honesto"] else "silence"}'
                  f'{" (mute)" if pi["degenerado_honesto"] else ""}  |  '
                  f'WIN cost={pv["honesto"]["coste"]:6.3f} '
                  f'{"fires" if pv["bate_silencio_honesto"] else "silence"}',
                  flush=True)

    # highest acc_dir in each unit among the thresholds where more than 2% of the MI
    # trials (windows) produce a command
    res['max_acc_dir'] = {}
    for u, tab in (('ventana', tv0), ('intencion', ti0)):
        filas = [(t, resumen(tab[t])) for t in sorted(tab)]
        val = [(r['acc_dir'], t, r['p_hit'], r['fp_min']) for t, r in filas
               if np.isfinite(r['acc_dir']) and (r['p_hit'] + r['p_inv']) > 0.02]
        if val:
            a, t, ph, fp = max(val)
            res['max_acc_dir'][u] = dict(acc_dir=a, thr=t, p_hit=ph, fp_min=fp)
            print(f'  max acc_dir {u:9s} = {a:.3f} at thr={t:.5f} '
                  f'(p_hit={ph:.3f}, fp/min={fp:.2f})   '
                  f'{"exceeds 0.90" if a > 0.90 else "does not exceed 0.90"}', flush=True)

    # ---------------------------------------------------------
    # 6. Per subject and CIs
    # ---------------------------------------------------------
    print(f'\n### 6. Per subject (main cell, FP<={PRESUP_PRIN:g}/min, LOSO) ###',
          flush=True)
    bp = seleccionar_por_presupuesto(ti0, PRESUP_PRIN)
    bpv = seleccionar_por_presupuesto(tv0, PRESUP_PRIN)
    hr = d0['hit_rate']
    res['por_sujeto'] = {}
    print(f'  {"subject":>7s} {"hit_rate":>9s} {"thr":>9s} {"hit":>6s} {"inv":>6s} '
          f'{"lost":>6s} {"accdir":>7s} {"fp/min":>7s} {"lat_ms":>7s} {"cost_int":>8s} '
          f'{"cost_win":>8s}')
    for s in sorted(bp['honesto']):
        mi_s = bp['honesto'][s]; mv_s = bpv['honesto'].get(s)
        res['por_sujeto'][s] = dict(
            hit_rate=hr.get(s), thr_loso=bp['thr_loso'].get(s),
            intencion=mi_s, ventana=mv_s, coste_intencion=coste(mi_s),
            coste_ventana=coste(mv_s) if mv_s else None)
        print(f'  S{s:<6d} {hr.get(s, float("nan")):9.3f} '
              f'{bp["thr_loso"].get(s, float("nan")):9.5f} '
              f'{mi_s["p_hit"]:6.3f} {mi_s["p_inv"]:6.3f} {mi_s["p_lost"]:6.3f} '
              f'{mi_s["acc_dir"]:7.3f} {mi_s["fp_min"]:7.2f} {mi_s["latencia_ms"]:7.0f} '
              f'{coste(mi_s):8.3f} '
              f'{coste(mv_s) if mv_s else float("nan"):8.3f}')

    res['ic_intencion'] = {}
    for campo in ['p_hit', 'p_inv', 'p_lost', 'acc_dir', 'fpr_episodio', 'fp_min',
                  'latencia_ms', 'prem_cue_por_trial']:
        v = np.array([bp['honesto'][s].get(campo, np.nan) for s in bp['honesto']], float)
        m, lo, hi = boot_media(v, args.boot, rng)
        res['ic_intencion'][campo] = dict(media=m, ic=(lo, hi))
        print(f'  CI95 {campo:20s} {m:8.3f}  [{lo:.3f}, {hi:.3f}]')
    v = np.array([coste(bp['honesto'][s]) for s in bp['honesto']], float)
    m, lo, hi = boot_media(v, args.boot, rng)
    res['ic_intencion']['coste'] = dict(media=m, ic=(lo, hi))
    print(f'  CI95 {"coste":20s} {m:8.3f}  [{lo:.3f}, {hi:.3f}]  (silence = 1.0)')

    # ---------------------------------------------------------
    # 7. Contrasts between formulations, in both units (d = treated - control)
    # ---------------------------------------------------------
    print('\n### 7. Contrasts (at the budget operating point) ###',
          flush=True)
    pares = [('B_bb_W2_r1_s42', 'A_bb_W2_r1_s42', 'B-A (bb)'),
             ('B_mb_W2_r1_s42', 'A_mb_W2_r1_s42', 'B-A (mb)'),
             ('C_bb_W2_r1_s42', 'B_bb_W2_r1_s42', 'C-B (bb)'),
             ('C_mb_W2_r1_s42', 'B_mb_W2_r1_s42', 'C-B (mb)'),
             ('D_bb_W2_r1_s42', 'B_bb_W2_r1_s42', 'D-B (bb)'),
             ('D_mb_W2_r1_s42', 'B_mb_W2_r1_s42', 'D-B (mb)'),
             ('B_bb_W2_r1_s42', 'B_mb_W2_r1_s42', 'bb-mb (B)'),
             ('B_bb_W2_r1_s43', 'B_mb_W2_r1_s43', 'bb-mb (B, s43)')]
    res['contrastes'] = {}
    hps = res['honesto_por_sujeto']
    for tratado, control, nom in pares:
        if tratado not in hps or control not in hps:
            continue
        fila = {}
        for u in ('ventana', 'intencion'):
            ctl, trt = hps[control][u], hps[tratado][u]
            fila[u] = dict(
                p_hit=boot_pareado(ctl, trt, 'p_hit', args.boot, rng),
                acc_dir=boot_pareado(ctl, trt, 'acc_dir', args.boot, rng),
                fp_min=boot_pareado(ctl, trt, 'fp_min', args.boot, rng))
        res['contrastes'][nom] = fila
        for u in ('ventana', 'intencion'):
            ph, ad = fila[u]['p_hit'], fila[u]['acc_dir']
            print(f'  {nom:16s} {u:9s} dp_hit={ph["media"]:+.3f} '
                  f'CI[{ph["ic"][0]:+.3f},{ph["ic"][1]:+.3f}] {ph["veredicto"]:16s} | '
                  f'daccdir={ad["media"]:+.3f} '
                  f'CI[{ad["ic"][0]:+.3f},{ad["ic"][1]:+.3f}] {ad["veredicto"]}',
                  flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cost_per_intent.json').write_text(json.dumps(res, indent=1, default=float))
    print(f'\n-> {OUT/"cost_per_intent.json"}   ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
