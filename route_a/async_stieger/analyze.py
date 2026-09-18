"""Re-analysis of the predictions saved by run_experiments.py (no retraining).

Every formulation is projected to a common decision space {LH, RH, NO-COMMAND} plus a
continuous command score, and the threshold is swept on that score:

  A / E   score = max(p_LH, p_RH)          E is A with a threshold (A at 0 always fires)
  B       score = p_LH + p_RH              (= 1 - p_IDLE)
  C       score = p_LH + p_RH              (= 1 - p_REST1 - p_REST2)
  D       score = 1 - p_gate(rest)

Per validation subject, aggregated with a subject-level bootstrap (B=10000, 95%
percentile CI; a contrast whose CI crosses 0 is not conclusive):
  - recall, fpr_idle (also split into REST1 / REST2), direction accuracy of the windows
    that fire, critical error P(RH|LH) + P(LH|RH), missed commands P(rest|MI);
  - recall at fpr_idle <= 0.05 / 0.10 (maximum recall among the sweep points within the
    budget) and pAUC for fpr_idle <= 0.2 (normalized area under the recall-vs-fpr
    envelope);
  - weighted cost, a cost-based threshold chosen leave-one-subject-out, and the
    sensitivity of the ranking of formulations to the cost weights;
  - row-normalized confusion matrices (4 classes + NO-COMMAND, and collapsed to
    LH / RH / rest) with per-cell bootstrap CIs;
  - controls (windows per trial, per-trial score separation, task block of REST1,
    trial selection of REST2) and a breakdown by online hit rate.

Reads <dir>/preds_*.npz and the hit rates of cache/trials_mc4_*per_window*.npz.
Writes <dir>/ANALYSIS.md (or --out) and <dir>/analysis.json.

Usage:
    python analyze.py --dir outputs/main
    python analyze.py --dir outputs/smoke --boot 200      # quick code check
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from build_dataset import LH, RH, REST1, REST2             # noqa: E402

B_BOOT = 10000
ILLIT_THR = 0.50                             # online hit rate below which = illiterate
NOCMD = -1                                   # label for "no command"

# Default cost weights (the sensitivity analysis sweeps them). A reversal (LH<->RH) and
# a command fired during rest both move the drone when it should not, so they weigh the
# same; a missed command only costs a retry.
COSTES_DEF = {'inversion': 10.0, 'comando_perdido': 1.0, 'falso_positivo': 10.0}

# threshold grid of the sweep, denser near 1
THRS = np.concatenate([np.linspace(0.0, 0.95, 96), np.linspace(0.951, 0.9999, 40)])


# ============================================================
# Loading
# ============================================================
def cargar_preds(d: Path) -> list[dict]:
    out = []
    for fp in sorted(d.glob('preds_*.npz')):
        z = np.load(fp, allow_pickle=True)
        r = {k: z[k] for k in z.files if k != 'meta'}
        meta = {}
        for s in z['meta']:
            k, _, v = str(s).partition('=')
            meta[k] = v
        r['meta'] = meta
        r['tag'] = meta.get('tag', fp.stem.replace('preds_', ''))
        r['form'] = meta.get('form', r['tag'][0])
        r['banda'] = meta.get('banda', '?')
        r['seed'] = int(meta.get('seed', 0))
        r['ratio'] = float(meta.get('ratio', 1.0))
        out.append(r)
    return out


def score_y_pred(r: dict) -> tuple[np.ndarray, np.ndarray]:
    """Projection to the common space: (command score, command class in {LH, RH})."""
    f = r['form']
    if f == 'D':
        pg, pd = r['proba_gate'], r['proba_disc']
        return 1.0 - pg[:, 1], pd.argmax(1)
    p = r['proba']
    if f in ('A', 'E'):
        return p.max(1), p.argmax(1)
    if f == 'B':
        return p[:, LH] + p[:, RH], p[:, [LH, RH]].argmax(1)
    if f == 'C':
        return p[:, LH] + p[:, RH], p[:, [LH, RH]].argmax(1)
    raise ValueError(f)


# ============================================================
# Subject-level bootstrap
# ============================================================
def boot_mean(vals, seed=0, B=B_BOOT):
    """Mean and 95% percentile CI over subjects (NaNs dropped)."""
    v = np.asarray([x for x in vals if x == x], float)
    if len(v) == 0:
        return float('nan'), float('nan'), float('nan')
    bs = np.random.RandomState(seed).randint(0, len(v), (B, len(v)))
    m = v[bs].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_paired(a, b, seed=0, B=B_BOOT):
    """Paired by subject. Returns (mean of a - b, lo, hi, n, fraction of bootstrap
    means > 0)."""
    d = np.array([x - y for x, y in zip(a, b) if x == x and y == y], float)
    if len(d) == 0:
        return float('nan'), float('nan'), float('nan'), 0, float('nan')
    bs = d[np.random.RandomState(seed).randint(0, len(d), (B, len(d)))].mean(1)
    return (float(d.mean()), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)), len(d), float((bs > 0).mean()))


# With very few subjects the subject-level bootstrap degenerates (with n=1 every
# resample is the same value and the CI collapses to a point), so no verdict is given.
MIN_SUBS_VEREDICTO = 5


def veredicto(lo, hi, n: int | None = None) -> str:
    if lo != lo:
        return 'no data'
    if n is not None and n < MIN_SUBS_VEREDICTO:
        return f'too few subjects (n={n})'
    return 'real (CI excludes 0)' if (lo > 0 or hi < 0) else 'inconclusive'


# ============================================================
# Per-subject metrics
# ============================================================
def metricas_sujeto(y4, src, score, cls, thr) -> dict:
    """Metrics in the common space at threshold `thr` (windows of one subject)."""
    pasa = score >= thr
    es_mi = np.isin(y4, [LH, RH])
    es_rep = ~es_mi
    out = {}
    out['recall'] = float(pasa[es_mi].mean()) if es_mi.any() else np.nan
    out['fpr_idle'] = float(pasa[es_rep].mean()) if es_rep.any() else np.nan
    for nm, m in [('rest1', src == 'rest1'), ('rest2', src == 'rest2')]:
        out[f'fpr_{nm}'] = float(pasa[m].mean()) if m.any() else np.nan
    ok = es_mi & pasa
    out['acc_control'] = float((cls[ok] == y4[ok]).mean()) if ok.any() else np.nan
    # critical error: among the MI windows that pass, fraction assigned to the other
    # hand, summed over LH and RH (range 0..2)
    inv = 0.0
    for a, b in [(LH, RH), (RH, LH)]:
        m = (y4 == a) & pasa
        inv += float((cls[m] == b).mean()) if m.any() else 0.0
    out['error_critico'] = inv
    out['error_benigno'] = float((~pasa)[es_mi].mean()) if es_mi.any() else np.nan
    return out


def sweep_sujeto(y4, src, score, cls) -> list[dict]:
    return [dict(thr=float(t), **metricas_sujeto(y4, src, score, cls, t)) for t in THRS]


def recall_at_fpr(sweep, budget) -> float:
    """Maximum recall among the sweep points with fpr_idle <= budget (0 if none)."""
    ok = [p['recall'] for p in sweep
          if p['fpr_idle'] == p['fpr_idle'] and p['recall'] == p['recall']
          and p['fpr_idle'] <= budget]
    return float(max(ok)) if ok else 0.0


def pauc(sweep, fmax=0.2) -> float:
    """Area under the recall-vs-fpr_idle envelope up to fpr_idle = fmax, divided by fmax."""
    pts = sorted((p['fpr_idle'], p['recall']) for p in sweep
                 if p['fpr_idle'] == p['fpr_idle'] and p['recall'] == p['recall'])
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


def umbral_optimo(res_por_sujeto: dict, w: dict, sujetos) -> tuple[float, float]:
    """Threshold that minimizes the mean cost over `sujetos`. Returns (thr, cost)."""
    mejor_t, mejor_c = float('nan'), float('inf')
    n = len(THRS)
    for j in range(n):
        cs = [coste(res_por_sujeto[s]['sweep'][j], w) for s in sujetos]
        cs = [c for c in cs if c == c]
        if not cs:
            continue
        c = float(np.mean(cs))
        if c < mejor_c:
            mejor_c, mejor_t = c, float(THRS[j])
    return mejor_t, mejor_c


def barrido_umbral(per: dict, w: dict) -> dict:
    """Confidence threshold chosen by cost, leave-one-subject-out (LOSO).

    Without a threshold a binary model always fires (fpr_idle = 1.000), but choosing the
    best threshold on the same validation subjects that are reported inflates the result,
    and there is no separate test split. For each subject the threshold is chosen on the
    other subjects and applied to that subject, so no subject influences its own threshold.

    Returned figures:
      coste_honesto       cost at the LOSO thresholds (the figure to report)
      coste_oraculo       cost at the best single threshold chosen on all subjects
                          (optimistic reference)
      inflacion_por_tuneo difference between the two
    """
    subs = sorted(per)
    t_or, c_or = umbral_optimo(per, w, subs)
    if len(subs) < 2:
        # LOSO needs at least one other subject to choose the threshold
        return {'thr_oraculo': t_or, 'coste_oraculo': c_or,
                'coste_honesto': {'mean': float('nan'), 'lo': float('nan'),
                                  'hi': float('nan')},
                'inflacion_por_tuneo': float('nan'),
                'thr_mediano': float('nan'), 'thr_min': float('nan'),
                'thr_max': float('nan'), 'por_sujeto': [],
                'aviso': f'LOSO not possible with {len(subs)} subject(s)'}

    filas, costes_h, thrs = [], [], []
    for s in subs:
        otros = [x for x in subs if x != s]
        t_s, _ = umbral_optimo(per, w, otros)          # threshold chosen without s
        j = int(np.argmin(np.abs(THRS - t_s)))
        m = per[s]['sweep'][j]
        c_s = coste(m, w)
        filas.append({'sujeto': s, 'thr': t_s, 'coste': c_s,
                      'recall': m['recall'], 'fpr_idle': m['fpr_idle'],
                      'error_critico': m['error_critico'],
                      'fpr_rest1': m['fpr_rest1'], 'fpr_rest2': m['fpr_rest2']})
        costes_h.append(c_s); thrs.append(t_s)

    mu, lo, hi = boot_mean(costes_h)
    # When a missed command is cheap and a wrong command is expensive, the cost optimum
    # of a poor formulation is to never fire (cost = weight of comando_perdido). The
    # cost is correct but the system is useless, so this case is flagged.
    recall_medio = float(np.mean([f['recall'] for f in filas
                                  if f['recall'] == f['recall']] or [np.nan]))
    degenerado = recall_medio < 0.05
    return {'recall_en_thr': recall_medio, 'degenerado_no_dispara': bool(degenerado),
            'thr_oraculo': t_or, 'coste_oraculo': c_or,
            'coste_honesto': {'mean': mu, 'lo': lo, 'hi': hi},
            'inflacion_por_tuneo': float(mu - c_or),
            'thr_mediano': float(np.median(thrs)) if thrs else float('nan'),
            'thr_min': float(np.min(thrs)) if thrs else float('nan'),
            'thr_max': float(np.max(thrs)) if thrs else float('nan'),
            'por_sujeto': filas}


def coste(m: dict, w: dict) -> float:
    """Weighted cost per window.

    `error_critico` is P(RH|LH) + P(LH|RH), a sum of two rates conditioned on different
    classes (range 0..2). The cost uses half of it, the class-averaged reversal rate of
    the windows that fire, so that the three terms are on the same 0..1 scale.
    """
    if m['recall'] != m['recall']:
        return float('nan')
    return (w['inversion'] * (m['error_critico'] / 2.0)
            + w['comando_perdido'] * m['error_benigno']
            + w['falso_positivo'] * m['fpr_idle'])


# ============================================================
# Confusion matrices: 4 classes and collapsed
# ============================================================
def pred4(r: dict, thr: float) -> np.ndarray:
    """Prediction in the 4-class space, or NOCMD if the score is below `thr`.

    Only C has REST1/REST2 outputs: its windows above the threshold keep their 4-class
    argmax. For the other formulations a window is LH, RH or NOCMD, so the REST1/REST2
    columns are structurally 0; the collapsed matrix is the fair comparison.
    """
    f = r['form']
    if f == 'C':
        p = r['proba']
        out = p.argmax(1)
        sc = p[:, LH] + p[:, RH]
        out[sc < thr] = NOCMD
        return out
    sc, cls = score_y_pred(r)
    out = np.where(sc >= thr, cls, NOCMD)
    return out


def matriz_confusion(y4, pred, n=4) -> np.ndarray:
    """Rows = true class (4), columns = predicted class (4) + NO-COMMAND."""
    M = np.zeros((n, n + 1), float)
    for t in range(n):
        m = y4 == t
        if not m.any():
            continue
        pv = pred[m]
        for c in range(n):
            M[t, c] = float((pv == c).mean())
        M[t, n] = float((pv == NOCMD).mean())
    return M


def matriz_colapsada(y4, pred) -> np.ndarray:
    """3x3 matrix in the space shared by all formulations.

    Rows (truth):        LH, RH, REST (REST1 or REST2)
    Columns (decision):  LH, RH, NO-COMMAND

    Predicting a rest class and not passing the threshold have the same outcome (no
    command is sent), so C's REST1/REST2 predictions are merged into NO-COMMAND.
    """
    y = np.where(np.isin(y4, [REST1, REST2]), 2, y4)          # 2 = rest
    p = np.where(np.isin(pred, [REST1, REST2, NOCMD]), 2, pred)  # 2 = no command
    M = np.zeros((3, 3), float)
    for t in range(3):
        m = y == t
        if not m.any():
            continue
        for c in range(3):
            M[t, c] = float((p[m] == c).mean())
    return M


def confusiones(preds: list[dict], thr: float, boot: int) -> dict:
    """Row-normalized 4-class matrix and collapsed 3x3 matrix of every cell at `thr`,
    with a per-cell bootstrap CI over subjects.

    The collapsed matrix is the fair comparison between C (4 outputs) and B (3 outputs):
    a gain of C that only comes from its finer rest labels disappears after collapsing.
    """
    out = {}
    for r in preds:
        y4, subj = r['y_true4'], r['subject']
        pred = pred4(r, thr)
        subs = sorted(set(int(s) for s in subj))

        # one matrix per subject; the bootstrap resamples subjects for every cell
        Ms, Mc = [], []
        for s in subs:
            m = subj == s
            Ms.append(matriz_confusion(y4[m], pred[m], n=4))
            Mc.append(matriz_colapsada(y4[m], pred[m]))
        Ms, Mc = np.array(Ms), np.array(Mc)

        def boot_celdas(M):
            n = len(M)
            rng = np.random.RandomState(0)
            idx = rng.randint(0, n, (boot, n))
            bs = M[idx].mean(1)                       # (boot, rows, cols)
            return (M.mean(0), np.percentile(bs, 2.5, axis=0),
                    np.percentile(bs, 97.5, axis=0))

        m4, lo4, hi4 = boot_celdas(Ms)
        m3, lo3, hi3 = boot_celdas(Mc)
        out[r['tag']] = {'form': r['form'], 'n_sujetos': len(subs),
                         'conf4': {'mean': m4.tolist(), 'lo': lo4.tolist(),
                                   'hi': hi4.tolist()},
                         'conf3_colapsada': {'mean': m3.tolist(), 'lo': lo3.tolist(),
                                             'hi': hi3.tolist()},
                         'acc4': float(np.trace(m4[:, :4]) / 4),
                         'acc3_colapsada': float(np.trace(m3) / 3)}
    return out


# ============================================================
# Main analysis
# ============================================================
def analizar(preds: list[dict], boot: int, thr_op: float, costes: dict) -> dict:
    res = {'celdas': {}, 'por_sujeto': {}}
    for r in preds:
        tag = r['tag']
        y4, src, subj = r['y_true4'], r['source'], r['subject']
        score, cls = score_y_pred(r)
        subs = sorted(set(int(s) for s in subj))
        per = {}
        for s in subs:
            m = subj == s
            sw = sweep_sujeto(y4[m], src[m], score[m], cls[m])
            base = metricas_sujeto(y4[m], src[m], score[m], cls[m], thr_op)
            base['recall_fpr05'] = recall_at_fpr(sw, 0.05)
            base['recall_fpr10'] = recall_at_fpr(sw, 0.10)
            base['pauc02'] = pauc(sw, 0.2)
            base['coste'] = coste(base, costes)
            base['n'] = int(m.sum())
            per[s] = {'metrics': base, 'sweep': sw}
        res['por_sujeto'][tag] = {s: per[s]['metrics'] for s in subs}

        agg = {}
        for k in ['recall', 'fpr_idle', 'fpr_rest1', 'fpr_rest2', 'acc_control',
                  'error_critico', 'error_benigno', 'recall_fpr05', 'recall_fpr10',
                  'pauc02', 'coste']:
            mu, lo, hi = boot_mean([per[s]['metrics'][k] for s in subs], B=boot)
            agg[k] = {'mean': mu, 'lo': lo, 'hi': hi}
        res['celdas'][tag] = {'form': r['form'], 'banda': r['banda'],
                              'seed': r['seed'], 'ratio': r['ratio'],
                              'n_sujetos': len(subs), 'agg': agg}
        res['celdas'][tag]['_per'] = {s: per[s]['metrics'] for s in subs}
        res['celdas'][tag]['_sweeps'] = per        # with the sweeps, for barrido_umbral
    return res


def contrastes(res: dict, preds: list[dict], boot: int) -> list[dict]:
    """Subject-paired contrasts between formulations with the same band, seed and ratio."""
    by = defaultdict(dict)
    for tag, c in res['celdas'].items():
        by[(c['banda'], c['seed'], c['ratio'])][c['form']] = tag
    salida = []
    metricas = ['coste', 'fpr_idle', 'recall', 'error_critico', 'acc_control',
                'recall_fpr05', 'recall_fpr10', 'pauc02']
    for key, formas in sorted(by.items()):
        for fa, fb in [('C', 'B'), ('B', 'A'), ('D', 'B'), ('C', 'D')]:
            if fa not in formas or fb not in formas:
                continue
            ta, tb = formas[fa], formas[fb]
            comunes = sorted(set(res['celdas'][ta]['_per']) & set(res['celdas'][tb]['_per']))
            for met in metricas:
                a = [res['celdas'][ta]['_per'][s][met] for s in comunes]
                b = [res['celdas'][tb]['_per'][s][met] for s in comunes]
                mu, lo, hi, n, p = boot_paired(a, b, B=boot)
                salida.append({'banda': key[0], 'seed': key[1], 'ratio': key[2],
                               'contraste': f'{fa} - {fb}', 'metrica': met,
                               'delta': mu, 'lo': lo, 'hi': hi, 'n': n, 'P>0': p,
                               'veredicto': veredicto(lo, hi, n),
                               'up': int(sum(1 for x, y in zip(a, b) if x > y)),
                               'dn': int(sum(1 for x, y in zip(a, b) if x < y))})
    return salida


def sensibilidad(res: dict, preds: list[dict], boot: int) -> list[dict]:
    """Ranking of the formulations by mean cost over a grid of cost weights."""
    rejilla = []
    for inv in [2.0, 5.0, 10.0, 20.0]:
        for fp in [1.0, 2.0, 5.0, 10.0]:
            rejilla.append({'inversion': inv, 'comando_perdido': 1.0,
                            'falso_positivo': fp})
    by = defaultdict(dict)
    for tag, c in res['celdas'].items():
        by[(c['banda'], c['seed'], c['ratio'])][c['form']] = tag
    out = []
    for key, formas in sorted(by.items()):
        for w in rejilla:
            fila = {'banda': key[0], 'seed': key[1], 'ratio': key[2],
                    'w_inv': w['inversion'], 'w_fp': w['falso_positivo']}
            costes_f = {}
            for f, tag in formas.items():
                per = res['celdas'][tag]['_per']
                vals = [coste(per[s], w) for s in sorted(per)]
                mu, lo, hi = boot_mean(vals, B=boot)
                costes_f[f] = mu
            fila['costes'] = costes_f
            # NaN costs are left out of the sort, which they would silently corrupt
            validos = {f: v for f, v in costes_f.items() if v == v}
            fila['ranking'] = [f for f, _ in sorted(validos.items(), key=lambda kv: kv[1])]
            fila['sin_dato'] = sorted(set(costes_f) - set(validos))
            fila['mejor'] = fila['ranking'][0] if fila['ranking'] else None
            out.append(fila)
    return out


# ============================================================
# Controls
# ============================================================
def controles(preds: list[dict], res: dict, boot: int) -> dict:
    """Controls on the validation predictions.

      fuga_42                validation trials and windows per trial (train/val trial
                             disjointness is asserted in run_experiments.py)
      agrupado_por_trial_42  MI - rest score separation with one value per trial
      bloque_41              C only: REST1 accuracy in task 2 and task 3, REST2 accuracy
      seleccion_44           REST2 score of trials with a valid MI window minus the rest

    The cue-locking control is cue_free_control.py.
    """
    out = {}

    # --- windows per validation trial ---
    fuga = []
    for r in preds:
        ntr = len(set(r['trial_id'].tolist()))
        fuga.append({'tag': r['tag'], 'trials_val': ntr,
                     'ventanas_val': int(len(r['trial_id'])),
                     'ventanas_por_trial': round(len(r['trial_id']) / max(ntr, 1), 2)})
    out['fuga_42'] = fuga

    # --- score separation grouped by trial ---
    por_trial = []
    for r in preds:
        score, cls = score_y_pred(r)
        y4, tid, subj = r['y_true4'], r['trial_id'], r['subject']
        agg = defaultdict(list)
        for t, s, sc, c, yy in zip(tid, subj, score, cls, y4):
            agg[(int(s), int(t))].append((sc, c, yy))
        # one value per trial: mean score of its windows, label of its first window
        per_s = defaultdict(list)
        for (s, _t), v in agg.items():
            sc = float(np.mean([x[0] for x in v]))
            yy = v[0][2]
            per_s[s].append((sc, yy))
        vals = []
        for s, v in per_s.items():
            mi = [sc for sc, yy in v if yy in (LH, RH)]
            rp = [sc for sc, yy in v if yy in (REST1, REST2)]
            if mi and rp:
                vals.append(float(np.mean(mi) - np.mean(rp)))
        mu, lo, hi = boot_mean(vals, B=boot)
        por_trial.append({'tag': r['tag'], 'sep_score_MI_menos_reposo_por_trial':
                          {'mean': mu, 'lo': lo, 'hi': hi}})
    out['agrupado_por_trial_42'] = por_trial

    # --- task block: REST1 comes from tasks 2/3, MI and REST2 from task 1 ---
    bloque = []
    for r in preds:
        if r['form'] != 'C':
            continue
        p = r['proba']; src = r['source']; tsk = r['tasknumber']; subj = r['subject']
        fila = {'tag': r['tag']}
        for t in (2, 3):
            m = (src == 'rest1') & (tsk == t)
            if m.any():
                # REST1 class accuracy within this task block
                fila[f'acc_rest1_task{t}'] = float((p[m].argmax(1) == REST1).mean())
                fila[f'n_task{t}'] = int(m.sum())
        m2 = src == 'rest2'
        if m2.any():
            fila['acc_rest2'] = float((p[m2].argmax(1) == REST2).mean())
        bloque.append(fila)
    out['bloque_41'] = bloque

    # --- trial selection: REST2 of trials with vs without a valid MI window ---
    sel = []
    for r in preds:
        score, cls = score_y_pred(r)
        src, miv, subj = r['source'], r['mi_valido'], r['subject']
        a, b = [], []
        for s in sorted(set(subj.tolist())):
            m = (subj == s) & (src == 'rest2')
            if not m.any():
                continue
            m1 = m & miv; m0 = m & ~miv
            if m1.any() and m0.any():
                a.append(float(score[m1].mean())); b.append(float(score[m0].mean()))
        mu, lo, hi, n, p = boot_paired(a, b, B=boot)
        sel.append({'tag': r['tag'], 'delta_score_rest2_MIvalido_menos_no':
                    {'mean': mu, 'lo': lo, 'hi': hi, 'n': n},
                    'veredicto': veredicto(lo, hi, n)})
    out['seleccion_44'] = sel
    return out


# ============================================================
# Breakdown by online hit rate
# ============================================================
def por_habilidad(res: dict, hit_rate: dict, boot: int) -> list[dict]:
    """Metrics of subjects below / at or above ILLIT_THR online hit rate.

    Subjects without a hit rate count as not illiterate.
    """
    out = []
    for tag, c in res['celdas'].items():
        per = c['_per']
        il = [s for s in per if hit_rate.get(s, 1.0) < ILLIT_THR]
        no = [s for s in per if hit_rate.get(s, 1.0) >= ILLIT_THR]
        fila = {'tag': tag, 'n_illit': len(il), 'n_no_illit': len(no)}
        for grupo, ss in [('illit', il), ('no_illit', no)]:
            for met in ['recall', 'fpr_idle', 'acc_control', 'pauc02']:
                mu, lo, hi = boot_mean([per[s][met] for s in ss], B=boot)
                fila[f'{grupo}_{met}'] = {'mean': mu, 'lo': lo, 'hi': hi}
        out.append(fila)
    return out


# ============================================================
# Markdown report
# ============================================================
def _tabla_confusion(A, M, etiquetas_fila, etiquetas_col) -> None:
    A('| true \\ pred | ' + ' | '.join(etiquetas_col) + ' |')
    A('|' + '---|' * (len(etiquetas_col) + 1))
    for i, fila in enumerate(etiquetas_fila):
        cel = []
        for j in range(len(etiquetas_col)):
            cel.append(f'{M["mean"][i][j]:.3f} [{M["lo"][i][j]:.3f},{M["hi"][i][j]:.3f}]')
        A(f'| {fila} | ' + ' | '.join(cel) + ' |')
    A('')


def informe(res, contr, sens, ctrl, hab, conf, umbrales, thr_op, costes, boot) -> str:
    L = []
    A = L.append
    A('# Results - multiclass LH/RH/REST1/REST2 (3 vs 4 classes)\n')
    A(f'- Operating threshold for the point metrics: `score_comando >= {thr_op}`.')
    A(f'- Cost weights: {costes} (reversal / missed command / false positive).')
    A(f'- Subject-level bootstrap, B={boot}, 95% percentile CI.')
    A('- A contrast whose CI crosses 0 is reported as not conclusive.')
    n_subs = max((c['n_sujetos'] for c in res['celdas'].values()), default=0)
    if n_subs < MIN_SUBS_VEREDICTO:
        A(f'\n> Warning: only {n_subs} validation subject(s). The subject-level')
        A('> bootstrap degenerates with so few subjects (the CI collapses to a point).')
        A('> No verdicts are given: this run only checks the code. The full split')
        A('> has 12 validation subjects.')
    A('')

    A('> Formulation E has no row of its own: E is A with a threshold, so it comes')
    A('> from the sweep of A. The `recall@fpr<=0.05/0.10` and `pAUC` columns of row A')
    A('> are those of E; row A at threshold 0 is the plain binary classifier')
    A('> (fpr_idle = 1.000 by construction: it always emits a command).\n')

    A('## Metrics per cell\n')
    A('| cell | form | band | recall | fpr_idle | fpr_R1 | fpr_R2 | acc_ctrl | '
      'crit_err | pAUC | cost |')
    A('|---|---|---|---|---|---|---|---|---|---|---|')
    for tag, c in sorted(res['celdas'].items()):
        g = c['agg']
        f = lambda k: f'{g[k]["mean"]:.3f}'
        A(f'| {tag} | {c["form"]} | {c["banda"]} | {f("recall")} | {f("fpr_idle")} | '
          f'{f("fpr_rest1")} | {f("fpr_rest2")} | {f("acc_control")} | '
          f'{f("error_critico")} | {f("pauc02")} | {f("coste")} |')
    A('')

    A('## Confidence threshold chosen by cost\n')
    A(f'Costs: reversal={costes["inversion"]:g}, '
      f'false_positive={costes["falso_positivo"]:g}, '
      f'missed_command={costes["comando_perdido"]:g}.')
    A('The threshold is chosen leave-one-subject-out: for each subject, on the other')
    A('validation subjects, so no subject influences its own threshold. `oracle` = best')
    A('threshold chosen on all subjects at once; it is not attainable and shows how much')
    A('tuning on the evaluation set would inflate the result (there is no test split).\n')
    A('> Degenerate optimum: with these weights a missed command costs 1 and a wrong')
    A('> command costs 10, so the formal optimum of a poor formulation is to never')
    A('> fire (cost exactly 1.0, recall 0). The cost is correct but such a system is')
    A('> useless, so the `recall` column is shown next to it: a low cost with recall')
    A('> near 0 is a switched-off system.\n')
    A('| cell | median thr | thr range | recall | LOSO cost | 95% CI | oracle cost | inflation |')
    A('|---|---|---|---|---|---|---|---|')
    for tag, u in sorted(umbrales.items()):
        h = u['coste_honesto']
        deg = ' [never fires]' if u.get('degenerado_no_dispara') else ''
        A(f'| {tag} | {u["thr_mediano"]:.3f} | '
          f'[{u["thr_min"]:.2f},{u["thr_max"]:.2f}] | '
          f'{u.get("recall_en_thr", float("nan")):.3f}{deg} | {h["mean"]:.4f} | '
          f'[{h["lo"]:.4f},{h["hi"]:.4f}] | {u["coste_oraculo"]:.4f} | '
          f'{u["inflacion_por_tuneo"]:+.4f} |')
    A('\nPer-subject values at the LOSO threshold (lower cost is better):\n')
    for tag, u in sorted(umbrales.items()):
        A(f'<details><summary>{tag}</summary>\n')
        A('| subject | thr | cost | recall | fpr_idle | crit_err | fpr_R1 | fpr_R2 |')
        A('|---|---|---|---|---|---|---|---|')
        for f in u['por_sujeto']:
            A(f'| {f["sujeto"]} | {f["thr"]:.3f} | {f["coste"]:.4f} | '
              f'{f["recall"]:.3f} | {f["fpr_idle"]:.3f} | {f["error_critico"]:.3f} | '
              f'{f["fpr_rest1"]:.3f} | {f["fpr_rest2"]:.3f} |')
        A('\n</details>\n')

    A('## Confusion matrices: 4 classes and collapsed\n')
    A(f'Row-normalized, per-cell 95% bootstrap CI over subjects '
      f'(threshold {thr_op}).')
    A('`NO-CMD` = the window does not pass the threshold (no command is sent).\n')
    A('> Only C has its own REST1/REST2 outputs; in A/B/D those columns are')
    A('> structurally 0 and predicted rest appears as `NO-CMD`. This is expected,')
    A('> since only C distinguishes the rest sub-source, so formulations are compared')
    A('> on the collapsed matrix, not on the 4-class one.\n')
    for tag, c in sorted(conf.items()):
        A(f'### {tag}\n')
        A('4 classes:\n')
        _tabla_confusion(A, c['conf4'],
                         ['LH', 'RH', 'REST1', 'REST2'],
                         ['LH', 'RH', 'REST1', 'REST2', 'NO-CMD'])
        A('Collapsed (comparable across formulations). Predicting rest and not passing')
        A('the threshold are merged into NO-CMD because in both cases no command is')
        A('sent. The diagonal is the operational accuracy.\n')
        _tabla_confusion(A, c['conf3_colapsada'],
                         ['LH', 'RH', 'REST'],
                         ['LH', 'RH', 'NO-CMD'])
        A(f'collapsed accuracy (mean of the diagonal) = {c["acc3_colapsada"]:.4f}\n')
    A('')

    A('## Subject-paired contrasts\n')
    A('| band | contrast | metric | delta | 95% CI | up/dn | verdict |')
    A('|---|---|---|---|---|---|---|')
    for c in contr:
        A(f'| {c["banda"]} | {c["contraste"]} | {c["metrica"]} | {c["delta"]:+.4f} | '
          f'[{c["lo"]:+.4f},{c["hi"]:+.4f}] | {c["up"]}/{c["dn"]} | {c["veredicto"]} |')
    A('')

    A('## Sensitivity to the cost weights\n')
    A('Ranking of the formulations by mean cost for each pair of weights.\n')
    A('| band | w_reversal | w_fp | ranking (best->worst) | best |')
    A('|---|---|---|---|---|')
    for s in sens:
        nd = f' (no data: {",".join(s["sin_dato"])})' if s.get('sin_dato') else ''
        A(f'| {s["banda"]} | {s["w_inv"]:g} | {s["w_fp"]:g} | '
          f'{" < ".join(s["ranking"])}{nd} | {s["mejor"]} |')
    mejores = {s['mejor'] for s in sens if s['mejor']}
    A(f'\n-> formulations that are best at some point of the grid: {sorted(mejores)}')
    A(f'-> ranking {"stable" if len(mejores) == 1 else "unstable"} '
      f'across the weights.\n')

    A('## Controls\n')
    A('### Leakage through temporal adjacency of REST2 and MI')
    A('The split is by subject and `trial_id` is globally unique; run_experiments.py')
    A('asserts an empty train/val intersection. Windows per validation trial:\n')
    A('| cell | val trials | val windows | windows/trial |')
    A('|---|---|---|---|')
    for f in ctrl['fuga_42']:
        A(f'| {f["tag"]} | {f["trials_val"]:,} | {f["ventanas_val"]:,} | '
          f'{f["ventanas_por_trial"]} |')
    A('\nMetric grouped by trial (score separation MI - rest):\n')
    A('| cell | delta score | 95% CI |')
    A('|---|---|---|')
    for p in ctrl['agrupado_por_trial_42']:
        d = p['sep_score_MI_menos_reposo_por_trial']
        A(f'| {p["tag"]} | {d["mean"]:+.4f} | [{d["lo"]:+.4f},{d["hi"]:+.4f}] |')
    A('')

    A('### Task-block confound (REST1 comes from tasks 2/3, MI and REST2 from task 1)')
    A('If REST1 is detected much better than REST2, or differs between task 2 and')
    A('task 3, the distinction may reflect the recording block, not the brain state.\n')
    A('| cell | acc REST1 (task 2) | acc REST1 (task 3) | acc REST2 |')
    A('|---|---|---|---|')
    for b in ctrl['bloque_41']:
        A(f'| {b["tag"]} | {b.get("acc_rest1_task2", float("nan")):.3f} | '
          f'{b.get("acc_rest1_task3", float("nan")):.3f} | '
          f'{b.get("acc_rest2", float("nan")):.3f} |')
    A('\n> acc = fraction of the windows of that source whose 4-class argmax is the')
    A('> correct rest class. Only formulation C has these outputs.\n')

    A('### Trial selection confound')
    A('REST2 windows are taken from every left/right trial, MI windows only from trials')
    A('with `triallength >= W` and a non-NaN `result`. If the score differs between the')
    A('two groups, part of the MI/REST2 separation is trial selection, not state.\n')
    A('| cell | delta score (REST2 of MI-valid trials - others) | 95% CI | verdict |')
    A('|---|---|---|---|')
    for s in ctrl['seleccion_44']:
        d = s['delta_score_rest2_MIvalido_menos_no']
        A(f'| {s["tag"]} | {d["mean"]:+.4f} | [{d["lo"]:+.4f},{d["hi"]:+.4f}] | '
          f'{s["veredicto"]} |')
    A('')

    A('## Breakdown by online hit rate (illit = hit rate < 0.50)\n')
    A('| cell | n illit | recall illit | fpr_idle illit | recall non-illit | fpr_idle non-illit |')
    A('|---|---|---|---|---|---|')
    for h in hab:
        A(f'| {h["tag"]} | {h["n_illit"]} | {h["illit_recall"]["mean"]:.3f} | '
          f'{h["illit_fpr_idle"]["mean"]:.3f} | {h["no_illit_recall"]["mean"]:.3f} | '
          f'{h["no_illit_fpr_idle"]["mean"]:.3f} |')
    A('')
    return '\n'.join(L)


# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help='directory with preds_*.npz')
    ap.add_argument('--thr', type=float, default=0.5, help='operating threshold')
    ap.add_argument('--boot', type=int, default=B_BOOT)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    d = Path(a.dir)
    preds = cargar_preds(d)
    if not preds:
        raise SystemExit(f'no preds_*.npz in {d}')
    print(f'{len(preds)} cells: {[p["tag"] for p in preds]}')

    # Online hit rate per subject, for the breakdown. Smoke caches are excluded: sorted
    # by name, '_smoke.npz' comes after '.npz' and would be picked as the last file.
    cache = sorted(p for p in (HERE / 'cache').glob('trials_mc4_*per_window*.npz')
                   if 'smoke' not in p.name)
    hit_rate = {}
    if cache:
        z = np.load(cache[-1], allow_pickle=True)
        hit_rate = {int(s): float(v) for s, v in zip(z['hit_subjects'], z['hit_values'])}
        print(f'hit_rate from {cache[-1].name}: {len(hit_rate)} subjects')
    else:
        print('[!] no full cache: every subject counts as non-illiterate in the breakdown')

    res = analizar(preds, a.boot, a.thr, COSTES_DEF)
    contr = contrastes(res, preds, a.boot)
    sens = sensibilidad(res, preds, a.boot)
    ctrl = controles(preds, res, a.boot)
    hab = por_habilidad(res, hit_rate, a.boot)
    conf = confusiones(preds, a.thr, a.boot)
    umbrales = {tag: barrido_umbral(c['_sweeps'], COSTES_DEF)
                for tag, c in res['celdas'].items()}

    md = informe(res, contr, sens, ctrl, hab, conf, umbrales, a.thr, COSTES_DEF, a.boot)
    out_md = Path(a.out) if a.out else d / 'ANALYSIS.md'
    out_md.write_text(md, encoding='utf-8')

    limpio = {'celdas': {k: {kk: vv for kk, vv in v.items()
                             if kk not in ('_per', '_sweeps')}
                         for k, v in res['celdas'].items()},
              'por_sujeto': res['por_sujeto'], 'contrastes': contr,
              'sensibilidad': sens, 'controles': ctrl, 'habilidad': hab,
              'confusiones': conf, 'umbrales': umbrales, 'costes': COSTES_DEF}
    (d / 'analysis.json').write_text(json.dumps(limpio, indent=1, default=float))
    print(md[:3000])
    print(f'\n-> {out_md}\n-> {d / "analysis.json"}')


if __name__ == '__main__':
    main()
