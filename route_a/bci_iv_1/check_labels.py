"""Check the true labels of the evaluation recordings and reproduce the silence baseline.

The official target is 0 at rest and +-1 during motor imagery (MI), so the MSE of the
constant-zero output is exactly the fraction of evaluated time that is MI:

    MSE_zero(subject) = mean(target**2)  over the evaluated samples

The official baseline of this output is 0.509. Reproducing it checks at once the
parsing, the time alignment, the subject identity, the set of evaluated samples and
the definition of the metric. The script stops (assert) if the value of any real
subject (a, b, f, g) falls outside the official range 0.49-0.54.

Measured for each of the 7 subjects:
  1. the labels parsed from the MAT file and, on an independent path, from the TXT
     file; they must be identical.
  2. the length of `true_y` against the evaluation signal `cnt`: labels are at
     1000 Hz and the signal at 100 Hz, and the factor must be exactly 10 for the real
     subjects (extra label samples of the artificial subjects c, d, e are counted).
  3. the set of values and their histogram. NaN marks undefined mental states and is
     excluded from the MSE, never converted to 0.
  4. the structure of the NaN gaps: short gaps (<= 5 s) at state transitions are the
     reaction-time dead zone, long gaps (> 5 s) separate runs. The number of runs is
     compared with the 4 runs of the documentation.
  5. task onsets and offsets, and task and interval durations, also reconstructed by
     adding one dead zone and compared with the documented 1.5-8 s.
  6. the fraction of time in each state.
  7. the zero-output MSE with and without the official truncation of subject 'a',
     the quantization floor of working at 100 Hz, and the zero-output MSE for each of
     the 10 phases of decimating the labels to 100 Hz.

The zero-output MSE of a, b, f, g is then compared with the four zero-output values
published for the winning entry's subjects a, b, c, d; a one-to-one match
(max |diff| < 0.001) identifies their a, b, c, d as our a, b, f, g.

Reads the recordings and the MAT/TXT label files from $BCI_DATA/bci_iv_1/ (see
download.py).
Output: outputs/check_labels.json, outputs/check_labels.csv

Usage:
    python check_labels.py
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data_io import (SUBS, REALES, ARTIFICIALES, FS_IV1, FS_LAB,      # noqa: E402
                         TRUNCAR_LAB, MSE_SILENCIO_OFICIAL,
                         RANGO_SILENCIO_OFICIAL, cargar_mat, cargar_true_y,
                         cargar_true_y_txt, mse_cero, mse_oraculo_100hz,
                         segmentos_definidos, OUT, RAW)

# Zero-output MSE published for the winning entry's four subjects, named a, b, c, d.
# That they match ours in the order a, b, f, g is tested below, not assumed.
MSE0_PUBLICADO = [0.507, 0.515, 0.491, 0.524]
DUR_DOC = (1.5, 8.0)               # documented duration range of tasks and intervals (s)
UMBRAL_RUN_S = 5.0                 # a label gap longer than this (s) is a run boundary


# ============================================================
def tramos_nan(y: np.ndarray) -> list[dict]:
    """Maximal runs of NaN, with the valid value before and after each one."""
    nan = np.isnan(y)
    ch = np.diff(np.concatenate([[0], nan.view(np.int8), [0]]))
    ini = np.where(ch == 1)[0]
    fin = np.where(ch == -1)[0]
    out = []
    for a, b in zip(ini, fin):
        out.append(dict(ini=int(a), fin=int(b), n=int(b - a),
                        dur_s=float((b - a) / FS_LAB),
                        antes=(float(y[a - 1]) if a > 0 else None),
                        despues=(float(y[b]) if b < len(y) else None)))
    return out


def resumen(v: list[float]) -> dict:
    """Summary statistics of a list of values."""
    a = np.asarray(v, float)
    if len(a) == 0:
        return dict(n=0)
    return dict(n=int(len(a)), media=float(a.mean()), mediana=float(np.median(a)),
                min=float(a.min()), max=float(a.max()), sd=float(a.std(ddof=1))
                if len(a) > 1 else 0.0,
                p05=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)))


def histograma(v: list[float], paso: float = 0.5) -> dict:
    """Counts per bin of width `paso`."""
    a = np.asarray(v, float)
    if len(a) == 0:
        return {}
    b = np.floor(a / paso) * paso
    u, c = np.unique(b, return_counts=True)
    return {f'{x:.1f}-{x+paso:.1f}': int(n) for x, n in zip(u, c)}


# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--verificar-txt', action='store_true', default=True)
    a = ap.parse_args()

    t0 = time.time()
    print('=' * 92)
    print('True labels of the evaluation recordings and silence baseline')
    print('=' * 92, flush=True)

    res = dict(por_sujeto={}, parseo={}, gate_g1={}, offset_reaccion={},
               limites_run={}, truncacion_oficial={})
    res['truncacion_oficial'] = dict(
        regla=TRUNCAR_LAB,
        fuente='read_me.txt from true_labels.zip',
        texto=(RAW / 'true_labels__read_me.txt').read_text(encoding='utf-8').strip())

    for s in SUBS:
        ev = cargar_mat(s, 'eval')
        y_full = cargar_true_y(s, truncar=False)
        y = cargar_true_y(s, truncar=True)

        # ---- 1. parsing: MAT vs TXT (independent path) ----
        par = dict(claves_mat=['true_y'], n_mat=int(len(y_full)),
                   dtype_mat=str(y_full.dtype))
        if a.verificar_txt:
            ytxt = cargar_true_y_txt(s)
            igual = bool(len(ytxt) == len(y_full)
                         and np.array_equal(np.isnan(ytxt), np.isnan(y_full))
                         and np.array_equal(np.nan_to_num(ytxt, nan=9.0),
                                            np.nan_to_num(y_full, nan=9.0)))
            par.update(n_txt=int(len(ytxt)), mat_igual_a_txt=igual,
                       max_dif=float(np.nanmax(np.abs(ytxt - y_full))) if igual else
                       float(np.nanmax(np.abs(ytxt[:min(len(ytxt), len(y_full))]
                                              - y_full[:min(len(ytxt), len(y_full))]))))
            assert igual, f'ds1{s}: the MAT and TXT labels differ'
        res['parseo'][s] = par

        # ---- 2. length against the evaluation signal ----
        # The factor must be exactly 10 for the 4 real subjects. The artificial ones
        # have a few extra label samples at the end; they are counted and reported,
        # not trimmed.
        factor = len(y_full) / ev['n_muestras']
        n_extra = int(len(y_full) - (FS_LAB // FS_IV1) * ev['n_muestras'])
        if s in REALES:
            assert n_extra == 0, \
                f'ds1{s}: len(true_y) = 10*len(cnt) + {n_extra}, expected exactly 10*len(cnt)'

        # ---- 3. values ----
        fin = np.isfinite(y_full)
        u, c = np.unique(y_full[fin], return_counts=True)
        hist = {f'{x:g}': int(n) for x, n in zip(u, c)}
        hist['NaN'] = int((~fin).sum())
        solo_tres = bool(set(u.tolist()) <= {-1.0, 0.0, 1.0})

        # ---- 4. structure of the NaN gaps ----
        tn = tramos_nan(y_full)
        cortos = [t for t in tn if t['dur_s'] <= UMBRAL_RUN_S]
        largos = [t for t in tn if t['dur_s'] > UMBRAL_RUN_S]
        dur_cortos = [t['dur_s'] for t in cortos]
        # A run is a connected stretch of recording: close the short gaps (transition
        # dead zones) and count the components left. The gap before the start and the
        # tail after the end are therefore not counted as run boundaries.
        vale = np.isfinite(y_full)
        for t in cortos:
            vale[t['ini']:t['fin']] = True
        cam = np.diff(np.concatenate([[0], vale.view(np.int8), [0]]))
        run_ini = np.where(cam == 1)[0]
        run_fin = np.where(cam == -1)[0]
        runs = [dict(ini_s=float(i / FS_LAB), fin_s=float(f / FS_LAB),
                     dur_s=float((f - i) / FS_LAB)) for i, f in zip(run_ini, run_fin)]
        # run boundary (in 100 Hz samples) = start of every run except the first
        cortes_100 = [int(round(i / (FS_LAB / FS_IV1))) for i in run_ini[1:]]
        # long internal gaps (between two runs): the actual run boundaries
        largos_int = [t for t in largos if t['ini'] > 0 and t['fin'] < len(y_full)
                      and vale[:t['ini']].any() and vale[t['fin']:].any()]

        # ---- 5. events ----
        seg = segmentos_definidos(y_full)
        tareas = [x for x in seg if x['valor'] != 0]
        huecos = [x for x in seg if x['valor'] == 0]
        # reconstructed duration = labelled duration + one dead zone (half a dead zone
        # on each side adds up to one)
        zm = float(np.median(dur_cortos)) if dur_cortos else 0.0
        d_tar = [x['dur_s'] for x in tareas]
        d_hue = [x['dur_s'] for x in huecos if x['dur_s'] < UMBRAL_RUN_S]
        d_tar_rec = [x + zm for x in d_tar]
        d_hue_rec = [x + zm for x in d_hue]

        # ---- 6. fraction of time per state (over the evaluated samples) ----
        okf = np.isfinite(y)
        n_ev = int(okf.sum())
        frac = dict(
            clase1_mi=float((y[okf] == -1).sum() / n_ev),
            clase2_mi=float((y[okf] == +1).sum() / n_ev),
            no_control=float((y[okf] == 0).sum() / n_ev),
            indefinido_sobre_total=float((~okf).sum() / len(y)))

        # ---- 7. zero-output MSE ----
        m0 = mse_cero(y)
        m0_sin = mse_cero(y_full)
        suelo = mse_oraculo_100hz(y)
        fases = {str(k): float(np.mean(y[k::10][np.isfinite(y[k::10])] ** 2))
                 for k in range(10)}

        res['por_sujeto'][s] = dict(
            sujeto=s, artificial=(s in ARTIFICIALES),
            n_true_y=int(len(y_full)), n_cnt_eval=int(ev['n_muestras']),
            factor_longitud=float(factor), muestras_etiqueta_sobrantes=n_extra,
            factor_exacto=bool(n_extra == 0),
            n_evaluadas=n_ev, n_indefinidas=int(len(y) - n_ev),
            truncado=bool(s in TRUNCAR_LAB),
            n_true_y_tras_truncar=int(len(y)),
            histograma_valores=hist, solo_valores_menos1_0_mas1=solo_tres,
            nan_tramos=len(tn), nan_tramos_cortos=len(cortos),
            nan_tramos_largos=len(largos),
            zona_muerta_s=resumen(dur_cortos),
            zonas_muertas_anomalas=[dict(ini_s=t['ini'] / FS_LAB, dur_s=t['dur_s'])
                                    for t in cortos if t['dur_s'] > 1.1],
            huecos_largos=[dict(ini_s=t['ini'] / FS_LAB, dur_s=t['dur_s'])
                           for t in largos],
            huecos_largos_internos=[dict(ini_s=t['ini'] / FS_LAB, dur_s=t['dur_s'])
                                    for t in largos_int],
            n_runs=len(runs), runs=runs,
            cortes_run_muestras_100hz=cortes_100,
            # borders between defined segments, and how many of them have a dead zone
            n_bordes_entre_segmentos=int(max(len(seg) - 1, 0)),
            n_bordes_con_zona_muerta=int(sum(
                1 for x, z in zip(seg[:-1], seg[1:]) if z['ini'] > x['fin'])),
            n_tareas=len(tareas), n_huecos=len(huecos),
            dur_tarea_etiquetada_s=resumen(d_tar),
            dur_tarea_reconstruida_s=resumen(d_tar_rec),
            dur_hueco_etiquetado_s=resumen(d_hue),
            dur_hueco_reconstruido_s=resumen(d_hue_rec),
            hist_tarea_reconstruida=histograma(d_tar_rec),
            hist_hueco_reconstruido=histograma(d_hue_rec),
            frac_estados=frac,
            mse_cero=m0, mse_cero_sin_truncar=m0_sin,
            suelo_cuantizacion_100hz=suelo,
            mse_cero_por_fase_decimacion=fases,
            onsets_1000hz=[x['ini'] for x in tareas],
            offsets_1000hz=[x['fin'] for x in tareas],
            clases_tarea=[int(x['valor']) for x in tareas])

        d = res['por_sujeto'][s]
        print(f'  ds1{s} {"(artificial)" if s in ARTIFICIALES else "            "} '
              f'n={len(y_full):>8d} (=10*cnt{n_extra:+d})  NaN={hist["NaN"]:>7d}  '
              f'tasks={len(tareas):>3d}  dead_zone={zm*1000:.0f} ms  '
              f'runs={d["n_runs"]}  MSE0={m0:.4f}'
              + (f'  (untruncated {m0_sin:.4f})' if s in TRUNCAR_LAB else ''),
              flush=True)

    # ============================================================
    # Silence baseline
    # ============================================================
    print('\n--- silence baseline (constant-zero output) ---', flush=True)
    m0r = {s: res['por_sujeto'][s]['mse_cero'] for s in REALES}
    lo, hi = RANGO_SILENCIO_OFICIAL
    dentro = {s: bool(lo <= v <= hi) for s, v in m0r.items()}
    media = float(np.mean(list(m0r.values())))
    empar = [dict(nuestro=s, mse0=m0r[s], publicado=p, dif=m0r[s] - p)
             for s, p in zip(REALES, MSE0_PUBLICADO)]
    max_dif = float(max(abs(x['dif']) for x in empar))
    res['gate_g1'] = dict(
        mse_cero_por_sujeto=m0r, rango_oficial=list(RANGO_SILENCIO_OFICIAL),
        dentro_del_rango=dentro, n_dentro=int(sum(dentro.values())), n=len(REALES),
        media=media, media_oficial=MSE_SILENCIO_OFICIAL,
        dif_media=float(media - MSE_SILENCIO_OFICIAL),
        emparejamiento_con_publicado=empar, max_dif_emparejado=max_dif,
        emparejamiento_uno_a_uno_confirmado=bool(max_dif < 0.001),
        pasa=bool(all(dentro.values())))
    for s in REALES:
        d = res['por_sujeto'][s]
        marca = 'OK ' if dentro[s] else 'OUT'
        print(f'  [{marca}] ds1{s}: MSE_zero = {m0r[s]:.4f}  '
              f'(official range {lo}-{hi})   quantization floor at 100 Hz = '
              f'{d["suelo_cuantizacion_100hz"]:.4f}')
    print(f'  mean of the {len(REALES)} real subjects = {media:.4f}  '
          f'(official {MSE_SILENCIO_OFICIAL}; diff {media-MSE_SILENCIO_OFICIAL:+.4f})')
    print(f'  match with the published zero-output MSE of the winning entry subjects '
          f'{MSE0_PUBLICADO}: max|diff| = {max_dif:.4f} -> ' +
          ('their (a,b,c,d) are our (a,b,f,g)'
           if max_dif < 0.001 else 'no one-to-one match'))
    assert res['gate_g1']['pasa'], (
        f'silence baseline check failed: zero-output MSE outside {RANGO_SILENCIO_OFICIAL}: '
        f'{m0r}. The labels or the metric are misread.')

    # sensitivity of the baseline to the decimation phase
    fmax = {s: float(max(res['por_sujeto'][s]['mse_cero_por_fase_decimacion'].values())
                     - min(res['por_sujeto'][s]['mse_cero_por_fase_decimacion'].values()))
            for s in REALES}
    res['gate_g1']['rango_por_fase_decimacion'] = fmax
    print(f'  control: range of the zero-output MSE over the 10 decimation phases of the '
          f'labels = {max(fmax.values()):.5f} (max over the 4 subjects)')

    # ============================================================
    # Reaction offset and run boundaries
    # ============================================================
    print('\n--- reaction offset ---', flush=True)
    zm_all = [res['por_sujeto'][s]['zona_muerta_s']['mediana'] for s in REALES]
    zm_min = [res['por_sujeto'][s]['zona_muerta_s']['min'] for s in REALES]
    zm_max = [res['por_sujeto'][s]['zona_muerta_s']['max'] for s in REALES]
    n_bor = [res['por_sujeto'][s]['n_bordes_entre_segmentos'] for s in REALES]
    n_zm = [res['por_sujeto'][s]['n_bordes_con_zona_muerta'] for s in REALES]
    anom = sum(len(res['por_sujeto'][s]['zonas_muertas_anomalas']) for s in REALES)
    tar = [res['por_sujeto'][s]['dur_tarea_etiquetada_s'] for s in REALES]
    tarr = [res['por_sujeto'][s]['dur_tarea_reconstruida_s'] for s in REALES]
    huer = [res['por_sujeto'][s]['dur_hueco_reconstruido_s'] for s in REALES]
    d_tar_all, d_hue_all = [], []
    for s in REALES:
        segs = segmentos_definidos(cargar_true_y(s, truncar=False))
        z = res['por_sujeto'][s]['zona_muerta_s']['mediana']
        d_tar_all += [x['dur_s'] + z for x in segs if x['valor'] != 0]
        d_hue_all += [x['dur_s'] + z for x in segs
                      if x['valor'] == 0 and x['dur_s'] < UMBRAL_RUN_S]
    d_tar_all = np.asarray(d_tar_all); d_hue_all = np.asarray(d_hue_all)
    en_rango_t = float(((d_tar_all >= DUR_DOC[0] - 1e-9)
                        & (d_tar_all <= DUR_DOC[1] + 1e-9)).mean())
    en_rango_h = float(((d_hue_all >= DUR_DOC[0] - 1e-9)
                        & (d_hue_all <= DUR_DOC[1] + 1e-9)).mean())
    res['offset_reaccion'] = dict(
        offset_reaccion_ms=float(np.median(zm_all) * 1000),
        metodo='median duration of the NaN gaps that separate two consecutive defined '
               'segments (one per state border)',
        zona_muerta_mediana_s=float(np.median(zm_all)),
        zona_muerta_min_s=float(min(zm_min)), zona_muerta_max_s=float(max(zm_max)),
        n_bordes_entre_segmentos=n_bor, n_bordes_con_zona_muerta=n_zm,
        todos_los_bordes_llevan_zona_muerta=bool(
            all(a == b for a, b in zip(n_bor, n_zm))),
        n_zonas_muertas_anomalas=int(anom),
        dur_tarea_etiquetada=dict(min=float(min(x['min'] for x in tar)),
                                  max=float(max(x['max'] for x in tar)),
                                  mediana=float(np.median([x['mediana']
                                                           for x in tar]))),
        dur_tarea_reconstruida=dict(min=float(min(x['min'] for x in tarr)),
                                    max=float(max(x['max'] for x in tarr)),
                                    mediana=float(np.median([x['mediana']
                                                             for x in tarr]))),
        dur_hueco_reconstruido=dict(min=float(min(x['min'] for x in huer)),
                                    max=float(max(x['max'] for x in huer))),
        doc_dice=list(DUR_DOC),
        n_tareas_reales=int(len(d_tar_all)), n_huecos_reales=int(len(d_hue_all)),
        frac_tareas_en_rango_doc=en_rango_t, frac_huecos_en_rango_doc=en_rango_h,
        n_tareas_fuera_de_rango=int(round((1 - en_rango_t) * len(d_tar_all))),
        exceso_maximo_sobre_8s=float(max(0.0, d_tar_all.max() - DUR_DOC[1])),
        evidencia=('there is a NaN gap of ~999 ms at every border between two defined '
                   'segments; labelled task durations are systematically ~1 s '
                   'shorter than documented, and adding that dead zone puts the '
                   f'minimum exactly at {DUR_DOC[0]} s'))
    o = res['offset_reaccion']
    print(f'  dead zone: median {o["zona_muerta_mediana_s"]*1000:.0f} ms '
          f'(min {o["zona_muerta_min_s"]*1000:.0f}, max '
          f'{o["zona_muerta_max_s"]*1000:.0f}); borders between segments {n_bor}, '
          f'borders with a dead zone {n_zm} -> all: '
          f'{o["todos_los_bordes_llevan_zona_muerta"]} '
          f'({anom} anomalous dead zones > 1.1 s, counted, not dropped)')
    print(f'  labelled task duration      {o["dur_tarea_etiquetada"]["min"]:.3f} - '
          f'{o["dur_tarea_etiquetada"]["max"]:.3f} s  (median '
          f'{o["dur_tarea_etiquetada"]["mediana"]:.3f})   documented: '
          f'{DUR_DOC[0]}-{DUR_DOC[1]}')
    print(f'  reconstructed task duration {o["dur_tarea_reconstruida"]["min"]:.3f} - '
          f'{o["dur_tarea_reconstruida"]["max"]:.3f} s  (documented minimum '
          f'{DUR_DOC[0]} s); {100*en_rango_t:.1f} % of the {len(d_tar_all)} '
          f'tasks in [{DUR_DOC[0]},{DUR_DOC[1]}] s, '
          f'max excess {o["exceso_maximo_sobre_8s"]:.3f} s')
    print(f'  reconstructed interval duration {o["dur_hueco_reconstruido"]["min"]:.3f}'
          f' - {o["dur_hueco_reconstruido"]["max"]:.3f} s  ({100*en_rango_h:.1f} % in '
          f'range)')
    print(f'  -> reaction offset = {o["offset_reaccion_ms"]:.0f} ms')

    print('\n--- run boundaries ---', flush=True)
    for s in REALES:
        d = res['por_sujeto'][s]
        print(f'  ds1{s}: {d["n_runs"]} runs = ' +
              ' | '.join(f'{r["ini_s"]:.0f}-{r["fin_s"]:.0f}s' for r in d['runs']) +
              f'   ({len(d["huecos_largos_internos"])} long internal gaps)')
    res['limites_run'] = {s: dict(n_runs=res['por_sujeto'][s]['n_runs'],
                                  cortes_100hz=res['por_sujeto'][s]
                                  ['cortes_run_muestras_100hz'],
                                  huecos_largos=res['por_sujeto'][s]['huecos_largos'])
                          for s in SUBS}
    n_runs_doc = 4
    res['limites_run']['coherente_con_doc'] = bool(
        all(res['por_sujeto'][s]['n_runs'] == n_runs_doc for s in REALES))
    print(f'  the documentation states 4 evaluation runs; measured: ' +
          ', '.join(f'{s}={res["por_sujeto"][s]["n_runs"]}' for s in REALES))

    # effect of the official truncation on subject 'a'
    da = res['por_sujeto']['a']
    corte_a = TRUNCAR_LAB['a']
    runs_a = [h for h in da['huecos_largos'] if h['ini_s'] * FS_LAB > corte_a]
    res['truncacion_oficial']['efecto_en_a'] = dict(
        corte_muestras_1000hz=corte_a, corte_s=corte_a / FS_LAB,
        n_runs_totales=da['n_runs'],
        cae_dentro_de_hueco_largo=bool(any(
            h['ini_s'] * FS_LAB <= corte_a <= (h['ini_s'] + h['dur_s']) * FS_LAB
            for h in da['huecos_largos'])),
        mse_cero_con=da['mse_cero'], mse_cero_sin=da['mse_cero_sin_truncar'],
        muestras_descartadas=int(len(cargar_true_y('a', truncar=False)) - corte_a),
        huecos_largos_posteriores=len(runs_a))
    ta = res['truncacion_oficial']['efecto_en_a']
    print(f'\n  official truncation of ds1a: cut at {ta["corte_s"]:.1f} s; '
          f'inside a long gap between runs: {ta["cae_dentro_de_hueco_largo"]}; '
          f'zero-output MSE {ta["mse_cero_sin"]:.4f} (untruncated) -> '
          f'{ta["mse_cero_con"]:.4f} (truncated), published {MSE0_PUBLICADO[0]}')

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'check_labels.json').write_text(
        json.dumps(res, indent=1, default=float), encoding='utf-8')

    filas = []
    for s in SUBS:
        d = res['por_sujeto'][s]
        filas.append(dict(
            sujeto=s, artificial=d['artificial'], n_true_y=d['n_true_y'],
            n_cnt_eval=d['n_cnt_eval'], factor=d['factor_longitud'],
            n_evaluadas=d['n_evaluadas'], n_indefinidas=d['n_indefinidas'],
            n_tareas=d['n_tareas'], n_runs=d['n_runs'],
            zona_muerta_ms=round(d['zona_muerta_s']['mediana'] * 1000, 1),
            dur_tarea_med_s=round(d['dur_tarea_etiquetada_s']['mediana'], 3),
            dur_tarea_rec_med_s=round(d['dur_tarea_reconstruida_s']['mediana'], 3),
            frac_mi=round(d['frac_estados']['clase1_mi']
                          + d['frac_estados']['clase2_mi'], 5),
            mse_cero=round(d['mse_cero'], 6),
            suelo_100hz=round(d['suelo_cuantizacion_100hz'], 6)))
    with open(OUT / 'check_labels.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)

    print(f'\n[OK] silence baseline: {res["gate_g1"]["n_dentro"]}/{len(REALES)} subjects '
          f'with zero-output MSE in {RANGO_SILENCIO_OFICIAL}, mean {media:.4f} vs '
          f'{MSE_SILENCIO_OFICIAL} official')
    print(f'[OK] {time.time()-t0:.0f}s -> outputs/check_labels.json, outputs/check_labels.csv')


if __name__ == '__main__':
    main()
