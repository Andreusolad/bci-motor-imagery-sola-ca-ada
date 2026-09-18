"""Stage 3 of 3 of the per-intent evaluation: evidence accumulation and per-intent cost.

Library module without a command-line entry point; cost_per_intent.py runs it. It reads
the window probabilities written by infer_streams.py (outputs/streams/win_{tag}.npz) and
scores them in two units, on exactly the same windows and probabilities, so that the only
difference between the two is the accumulation:

  window unit   every window is an independent event
  intent unit   every trial is one event, decided by the online evidence accumulator
                (rt_system/accumulator.py, unmodified) with the reference configuration
                of rt_system/config.py (dwell 600 ms, refractory period 1000 ms)

Trial time line (see prep_streams.py):
  [-2000, 0)                        pre-cue rest
  [0, 2000)                         cue visible, cursor not yet shown
  [2000, 2000 + triallength*1000)   motor imagery (LR trials) / sustained cued rest (REST1)
In LR trials, emissions before the MI onset are counted as premature (split into pre-cue
and cue periods), never as hits.

Two attribution conventions:
  'centro'    a window belongs to the region that contains its center
  'estricta'  the whole window must lie inside the region (no sample from before the
              MI onset)

False commands per minute are measured on REST1, not on REST2. The pre-cue region lasts
2.0 s, so only one 2 s window lies entirely inside it, while the 600 ms dwell needs 5
consecutive windows at the 125 ms stride: REST2 alone cannot produce a command. REST1
(cued rest, "clear your mind") is the only sustained rest in the dataset; it is not the
passive rest of deployment, so the false-command rate is a proxy.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from rt_system import config as C                                  # noqa: E402
from rt_system.accumulator import EvidenceAccumulator, ms_to_steps  # noqa: E402
from rt_system.gate import GateDecision                            # noqa: E402

WIN_DIR = HERE / 'outputs' / 'streams'
OUT = HERE / 'outputs'
IDLE3 = 2

COSTES_DEF = {'inversion': 10.0, 'comando_perdido': 1.0, 'falso_positivo': 10.0}
MIN_SUBS_VEREDICTO = 5              # minimum number of subjects for a bootstrap verdict
# Threshold grid: 15 linear points in [0.30, 0.94] plus 30 points log-spaced in 1 - thr
# from 0.06 down to 10^-4.5, to resolve the high-threshold tail.
THRS = np.round(np.unique(np.concatenate([
    np.linspace(0.30, 0.94, 15),
    1.0 - np.logspace(np.log10(0.06), -4.5, 30)])), 6)

# False-command budgets, in false commands per minute of rest.
PRESUPUESTOS_FP = [0.5, 1.0, 6.0]

# reference configuration = that of the online system (rt_system/config.py)
CFG_REF = dict(strategy='dwell', dwell_ms=C.DWELL_MS, refractory_ms=C.REFRACTORY_MS,
               ema_alpha=C.EMA_ALPHA, hmm_stay=C.HMM_STAY)


# ============================================================
# Loading
# ============================================================
N_SUBS_VAL = 12                     # subjects of the canonical validation split


def cargar(path: Path, exigir_completo: bool = True) -> dict:
    """Load the windows of one cell.

    With `exigir_completo`, stop if the file does not contain the 12 validation subjects:
    infer_streams.py names its output after the checkpoint tag only, so a run on smoke
    streams or smoke checkpoints writes the same file names as a full run.
    """
    z = np.load(path, allow_pickle=True)
    n_subs = len(set(z['subject'].tolist()))
    if exigir_completo and n_subs != N_SUBS_VAL:
        raise SystemExit(
            f'{path.name} has {n_subs} subjects, {N_SUBS_VAL} expected. '
            f'It may come from a --smoke run. Regenerate with:  python infer_streams.py --todos')
    d = {k: z[k] for k in ['probs3', 'ini_ms', 'centro_ms', 'fin_ms', 'win_trial',
                           'subject', 'session', 'trial_idx', 'kind', 'y4',
                           'triallen', 'result', 'mi_ok']}
    d['hit_rate'] = {int(s): float(v) for s, v in zip(z['hit_subjects'], z['hit_values'])}
    d['meta'] = [str(x) for x in z['meta']]
    d['stride_ms'] = float([m for m in d['meta'] if m.startswith('stride_ms')][0].split('=')[1])
    # per-trial window slices (a stable sort by trial keeps the time order)
    n_tr = len(d['kind'])
    orden = np.argsort(d['win_trial'], kind='stable')
    for k in ['probs3', 'ini_ms', 'centro_ms', 'fin_ms', 'win_trial']:
        d[k] = d[k][orden]
    bordes = np.searchsorted(d['win_trial'], np.arange(n_tr + 1))
    d['slice'] = list(zip(bordes[:-1], bordes[1:]))
    return d


def mascara_region(d: dict, i: int, lo: float, hi: float, conv: str) -> np.ndarray:
    """Windows of trial i that fall in [lo, hi) under the given convention."""
    a, b = d['slice'][i]
    if conv == 'estricta':
        return (d['ini_ms'][a:b] >= lo) & (d['fin_ms'][a:b] < hi)
    return (d['centro_ms'][a:b] >= lo) & (d['centro_ms'][a:b] < hi)


def region_activa(d: dict, i: int) -> tuple[float, float]:
    """Motor imagery period (LR trials) or sustained rest period (REST1), in ms."""
    return (float(C.MI_ONSET_MS),
            float(C.MI_ONSET_MS + d['triallen'][i] * 1000.0))


# ============================================================
# Intent unit: simulation with the online accumulator
# ============================================================
def simular_trial(d: dict, i: int, acc: EvidenceAccumulator, thr: float,
                  usa_gate: bool) -> list[tuple[float, float, int]]:
    """Run the accumulator over trial i. Return [(center_ms, end_ms, class), ...].

    `usa_gate=True`  (dwell/nofm): the threshold acts on score_comando = 1 - P(IDLE),
                     the same score that the per-window analysis thresholds.
    `usa_gate=False` (ema/bayes): the threshold is the accumulator's internal emit_thr
                     on the smoothed/filtered probability.
    """
    a, b = d['slice'][i]
    p3 = d['probs3'][a:b]; cen = d['centro_ms'][a:b]; fin = d['fin_ms'][a:b]
    acc.reset()
    out = []
    for k in range(len(p3)):
        p = p3[k]
        cls = int(np.argmax(p[:2]))
        ic = bool((1.0 - p[IDLE3]) >= thr) if usa_gate else True
        cmd = acc.update(GateDecision(label=cls if ic else -1, control_class=cls,
                                      probs=p, is_control=ic))
        if cmd is not None:
            out.append((float(cen[k]), float(fin[k]), int(cmd)))
    return out


def evaluar_intencion(d: dict, cfg: dict, thr: float, conv: str = 'centro',
                      solo_mi_ok: bool = True) -> dict:
    """Per-intent (per-trial) metrics, aggregated per subject. Return {sid: {...}}."""
    usa_gate = cfg['strategy'] in ('dwell', 'nofm')
    acc = EvidenceAccumulator(
        strategy=cfg['strategy'], dwell_ms=cfg['dwell_ms'],
        refractory_ms=cfg['refractory_ms'],
        emit_thr=(0.0 if usa_gate else thr),
        ema_alpha=cfg['ema_alpha'], hmm_stay=cfg['hmm_stay'],
        stride_ms=d['stride_ms'])

    por_suj: dict[int, dict] = {}
    for i in range(len(d['kind'])):
        if d['kind'][i] == 'mi' and solo_mi_ok and not d['mi_ok'][i]:
            continue
        lo, hi = region_activa(d, i)
        if hi <= lo:
            continue
        sid = int(d['subject'][i])
        s = por_suj.setdefault(sid, dict(
            n_mi=0, hit=0, inv=0, lost=0, prematuras=0, prem_precue=0, prem_cue=0,
            repeticiones=0, n_rest=0, rest_con_fp=0, fp=0, seg_rest=0.0, seg_mi=0.0,
            lat=[], n_win_mi=0, n_win_rest=0))

        a, _b = d['slice'][i]
        em = simular_trial(d, i, acc, thr, usa_gate)
        # seconds of the active period covered by windows under the convention
        m_act = mascara_region(d, i, lo, hi, conv)
        n_win_act = int(m_act.sum())
        segs = n_win_act * d['stride_ms'] / 1000.0

        if conv == 'estricta':
            # whole window inside [lo, hi): no sample from before the onset
            w_ms = float(d['fin_ms'][a] - d['ini_ms'][a])
            dentro = [(c, f, k) for c, f, k in em if (f - w_ms) >= lo and f < hi]
        else:
            dentro = [(c, f, k) for c, f, k in em if lo <= c < hi]
        antes = [(c, f, k) for c, f, k in em if c < lo]

        if d['kind'][i] == 'mi':
            s['n_mi'] += 1; s['seg_mi'] += segs; s['n_win_mi'] += n_win_act
            s['prematuras'] += len(antes)
            # premature emissions by region: < 0 = pre-cue rest;
            # [0, 2000) = cue visible without cursor
            s['prem_precue'] += sum(1 for c, _f, _k in antes if c < 0)
            s['prem_cue'] += sum(1 for c, _f, _k in antes if 0 <= c < C.MI_ONSET_MS)
            if dentro:
                c0, f0, k0 = dentro[0]
                if k0 == int(d['y4'][i]):
                    s['hit'] += 1
                else:
                    s['inv'] += 1
                s['repeticiones'] += len(dentro) - 1
                s['lat'].append(f0 - lo)          # latency from the MI onset
            else:
                s['lost'] += 1
        else:                                      # REST1
            s['n_rest'] += 1; s['seg_rest'] += segs; s['n_win_rest'] += n_win_act
            s['fp'] += len(dentro)
            s['rest_con_fp'] += int(len(dentro) > 0)

    return {sid: _cerrar_intencion(v) for sid, v in por_suj.items()}


def _cerrar_intencion(s: dict) -> dict:
    n = max(s['n_mi'], 1); nr = max(s['n_rest'], 1)
    mins = s['seg_rest'] / 60.0
    return dict(
        n_mi=s['n_mi'], n_rest=s['n_rest'],
        p_hit=s['hit'] / n, p_inv=s['inv'] / n, p_lost=s['lost'] / n,
        acc_dir=(s['hit'] / (s['hit'] + s['inv'])) if (s['hit'] + s['inv']) else np.nan,
        prematuras_por_trial=s['prematuras'] / n,
        prem_precue_por_trial=s['prem_precue'] / n,
        prem_cue_por_trial=s['prem_cue'] / n,
        repeticiones_por_emision=(s['repeticiones'] / (s['hit'] + s['inv']))
        if (s['hit'] + s['inv']) else np.nan,
        fpr_episodio=s['rest_con_fp'] / nr,
        fp_min=(s['fp'] / mins) if mins > 0 else np.nan,
        min_reposo=mins,
        latencia_ms=float(np.mean(s['lat'])) if s['lat'] else np.nan,
        latencia_p90=float(np.percentile(s['lat'], 90)) if s['lat'] else np.nan)


# ============================================================
# Window unit: every window is an event
# ============================================================
def evaluar_ventana(d: dict, thr: float, conv: str = 'centro',
                    solo_mi_ok: bool = True, incluir_rest2: bool = False) -> dict:
    """Per-window metrics, aggregated per subject. Return {sid: {...}}.

    `incluir_rest2` adds the pre-cue windows to the rest set. It is off by default so
    that the rest set matches the one of the intent unit (REST1 only); turning it on
    gives the rest set REST1 + REST2 of the per-window analysis.
    """
    por_suj: dict[int, dict] = {}
    for i in range(len(d['kind'])):
        es_mi = d['kind'][i] == 'mi'
        if es_mi and solo_mi_ok and not d['mi_ok'][i]:
            continue
        lo, hi = region_activa(d, i)
        if hi <= lo:
            continue
        a, b = d['slice'][i]
        m = mascara_region(d, i, lo, hi, conv)
        if not m.any():
            continue
        p3 = d['probs3'][a:b][m]
        emite = (1.0 - p3[:, IDLE3]) >= thr
        cls = np.argmax(p3[:, :2], axis=1)
        sid = int(d['subject'][i])
        s = por_suj.setdefault(sid, dict(n_mi=0, hit=0, inv=0, lost=0,
                                         n_rest=0, fp=0))
        if es_mi:
            ok = cls == int(d['y4'][i])
            s['n_mi'] += len(p3)
            s['hit'] += int((emite & ok).sum())
            s['inv'] += int((emite & ~ok).sum())
            s['lost'] += int((~emite).sum())
            if incluir_rest2:                      # pre-cue windows of the same trial
                m2 = mascara_region(d, i, float(C.T0_MS), 0.0, conv)
                if m2.any():
                    p2 = d['probs3'][a:b][m2]
                    s['n_rest'] += len(p2)
                    s['fp'] += int(((1.0 - p2[:, IDLE3]) >= thr).sum())
        else:
            s['n_rest'] += len(p3)
            s['fp'] += int(emite.sum())
    out = {}
    for sid, s in por_suj.items():
        n = max(s['n_mi'], 1); nr = max(s['n_rest'], 1)
        out[sid] = dict(n_mi=s['n_mi'], n_rest=s['n_rest'],
                        p_hit=s['hit'] / n, p_inv=s['inv'] / n, p_lost=s['lost'] / n,
                        acc_dir=(s['hit'] / (s['hit'] + s['inv']))
                        if (s['hit'] + s['inv']) else np.nan,
                        fpr_episodio=s['fp'] / nr,           # here = per-window fpr
                        fp_min=s['fp'] / nr * (60000.0 / d['stride_ms']),
                        min_reposo=np.nan, latencia_ms=np.nan, latencia_p90=np.nan,
                        prematuras_por_trial=np.nan,
                        repeticiones_por_emision=np.nan)
    return out


# ============================================================
# Cost: same formula in both units
# ============================================================
def coste(m: dict, w: dict = COSTES_DEF) -> float:
    """Weighted cost; p_hit + p_inv + p_lost = 1 by construction in both units.

    Never emitting costs exactly w['comando_perdido'] (1.0 by default), the reference
    against which emitting a command is judged.
    """
    if not np.isfinite(m['p_inv']) or not np.isfinite(m['fpr_episodio']):
        return float('nan')
    return (w['inversion'] * m['p_inv']
            + w['comando_perdido'] * m['p_lost']
            + w['falso_positivo'] * m['fpr_episodio'])


def agrega(por_suj: dict, campo: str) -> float:
    v = [por_suj[s][campo] for s in por_suj if np.isfinite(por_suj[s].get(campo, np.nan))]
    return float(np.mean(v)) if v else float('nan')


# ============================================================
# Threshold sweep, LOSO selection and subject-level bootstrap
# ============================================================
def tabla_barrido(d: dict, unidad: str, cfg: dict, conv='centro', solo_mi_ok=True,
                  thrs=THRS, incluir_rest2=False) -> dict:
    """Per-subject metrics for every threshold. Independent of the cost weights."""
    return {float(t): (evaluar_intencion(d, cfg, float(t), conv, solo_mi_ok)
                       if unidad == 'intencion'
                       else evaluar_ventana(d, float(t), conv, solo_mi_ok,
                                            incluir_rest2))
            for t in thrs}


def seleccionar(tabla: dict, w: dict = COSTES_DEF) -> dict:
    """Oracle threshold and leave-one-subject-out (LOSO) threshold from a computed table.

    Kept apart from `tabla_barrido` because the cost weights do not change the metrics,
    so several weight settings can reuse one table.
    """
    subs = sorted({s for m in tabla.values() for s in m})

    # mean cost per threshold (oracle) and LOSO selection
    coste_medio = {t: float(np.nanmean([coste(tabla[t][s], w) for s in subs
                                        if s in tabla[t]])) for t in tabla}
    t_orac = min(coste_medio, key=lambda t: coste_medio[t])

    loso = {}
    for s in subs:
        otros = [o for o in subs if o != s]
        cm = {t: float(np.nanmean([coste(tabla[t][o], w) for o in otros if o in tabla[t]]))
              for t in tabla}
        t_s = min(cm, key=lambda t: cm[t])
        loso[s] = (t_s, tabla[t_s].get(s))

    honesto = {s: loso[s][1] for s in subs if loso[s][1] is not None}
    return dict(tabla=tabla, coste_medio=coste_medio, t_oraculo=t_orac,
                oraculo={s: tabla[t_orac][s] for s in subs if s in tabla[t_orac]},
                honesto=honesto, thr_loso={s: loso[s][0] for s in subs})


def barrido(d: dict, unidad: str, cfg: dict, conv='centro', solo_mi_ok=True,
            w=COSTES_DEF, thrs=THRS, incluir_rest2=False) -> dict:
    """Table and selection in one call, for a single set of cost weights."""
    return seleccionar(tabla_barrido(d, unidad, cfg, conv, solo_mi_ok, thrs,
                                     incluir_rest2), w)


def seleccionar_por_presupuesto(tabla: dict, budget_fp_min: float) -> dict:
    """Operating point under a false-command budget instead of a weighted cost.

    Under the default weights the minimum-cost operating point can be to never emit, in
    which case comparing configurations at their minimum cost compares silent systems.
    This selection fixes the number of false commands per minute of rest and maximizes
    the fraction of intents that produce a correct command (p_hit), the per-intent
    analogue of recall at fpr <= 0.05.

    LOSO selection: the threshold of each subject is chosen on the other subjects.
    """
    subs = sorted({s for m in tabla.values() for s in m})
    ts = sorted(tabla)

    def elige(pool):
        mejor, best = None, -1.0
        for t in ts:
            fp = [tabla[t][o]['fp_min'] for o in pool if o in tabla[t]]
            hit = [tabla[t][o]['p_hit'] for o in pool if o in tabla[t]]
            fp = [x for x in fp if np.isfinite(x)]
            hit = [x for x in hit if np.isfinite(x)]
            if not fp or not hit or np.mean(fp) > budget_fp_min:
                continue
            if np.mean(hit) > best:
                best, mejor = float(np.mean(hit)), t
        return mejor

    t_orac = elige(subs)
    honesto, thr_loso = {}, {}
    for s in subs:
        t_s = elige([o for o in subs if o != s])
        thr_loso[s] = t_s
        if t_s is not None and s in tabla[t_s]:
            honesto[s] = tabla[t_s][s]
    return dict(tabla=tabla, presupuesto=budget_fp_min, t_oraculo=t_orac,
                oraculo=({s: tabla[t_orac][s] for s in subs if s in tabla[t_orac]}
                         if t_orac is not None else {}),
                honesto=honesto, thr_loso=thr_loso,
                factible=bool(t_orac is not None))


def boot_media(vals: np.ndarray, B: int, rng) -> tuple[float, float, float]:
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.randint(0, len(v), size=(B, len(v)))
    m = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_pareado(a: dict, b: dict, campo: str, B: int, rng) -> dict:
    """b - a, paired by subject, with a subject-level bootstrap 95% CI."""
    subs = sorted(set(a) & set(b))
    dif = np.array([b[s][campo] - a[s][campo] for s in subs
                    if np.isfinite(a[s].get(campo, np.nan))
                    and np.isfinite(b[s].get(campo, np.nan))], float)
    if len(dif) == 0:
        return dict(n=0, media=np.nan, ic=(np.nan, np.nan), veredicto='no data')
    idx = rng.randint(0, len(dif), size=(B, len(dif)))
    m = dif[idx].mean(axis=1)
    lo, hi = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    if len(dif) < MIN_SUBS_VEREDICTO:
        ver = f'no verdict (n={len(dif)}<{MIN_SUBS_VEREDICTO})'
    elif lo > 0 or hi < 0:
        ver = 'real'
    else:
        ver = 'inconclusive'
    return dict(n=len(dif), media=float(dif.mean()), ic=(lo, hi), veredicto=ver,
                n_mejora=int((dif > 0).sum()), n_empeora=int((dif < 0).sum()))
