"""Channel and band importance for detection (MI vs rest), measured by occluding the
inputs of the trained models (no retraining).

For every cell, the validation windows of cache/trials_mc4_W2_per_window.npz (MI, REST1
and REST2) are passed through outputs/main/ckpt_<cell>.pt. The command score is that of
infer_streams.py (1 - P(IDLE)) and the detection metrics are those of analyze.py. Before
any occlusion, the unoccluded inference must reproduce the probabilities saved in
outputs/main/preds_<cell>.npz (max abs difference < 1e-5); otherwise the script stops.

Occlusions:
  - channel c: after z-scoring, the channel is set to 0 (its training mean);
  - band [a, b]: the sub-band is subtracted from the signal already filtered to the
    model band (Xb - bandpass(Xb, a, b)), and the result is z-scored with the checkpoint
    mu/sd. Sub-bands: delta, theta, mu, beta, gamma.

Importance = base metric - occluded metric, per subject, with a subject-level bootstrap
(B=--boot, 95% percentile CI; a CI that crosses 0 is not conclusive). A positive value
means that the channel or band helps detection.

Writes outputs/occlusion.json and outputs/occlusion.csv.

Usage:
    python occlusion.py                 # default cells, B=10000
    python occlusion.py --smoke         # one cell, B=200 (code check)
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

from rt_system import config as C                                   # noqa: E402
from rt_system.preprocessing import bandpass                        # noqa: E402
from build_dataset import LH, RH                                    # noqa: E402
from run_experiments import (BANDAS, aplicar_banda, split_canonico,  # noqa: E402
                             predict_proba)
from infer_streams import cargar_modelos, probs_canonicas           # noqa: E402
from analyze import (sweep_sujeto, pauc, recall_at_fpr, metricas_sujeto,  # noqa: E402
                     boot_mean, veredicto)

CACHE = HERE / 'cache'
MAIN = HERE / 'outputs' / 'main'
OUT = HERE / 'outputs'
CANALES = list(C.CYTON_8)                 # ['FC3','FCZ','FC4','C3','CZ','C4','CP3','CP4']

# standard EEG sub-bands within the 0.5-40 Hz broadband; those outside the model band
# (e.g. delta and gamma for 'mb', 8-30 Hz) serve as a sanity check
SUBBANDAS = [('delta', 0.5, 4.0), ('theta', 4.0, 8.0), ('mu', 8.0, 13.0),
             ('beta', 13.0, 30.0), ('gamma', 30.0, 40.0)]

# detection metrics (as in analyze.py); pauc02 is the headline metric
METRICAS = ['pauc02', 'recall_fpr10', 'recall_fpr05', 'fpr_idle', 'recall_at_thr', 'acc_gate']


def metricas_deteccion_por_sujeto(y4, src, score, cls, subj, thr=0.5) -> dict:
    """Per-subject detection metrics in the common space (as in analyze.py).

    pauc02 and recall_fpr* sweep the threshold; fpr_idle and recall_at_thr are measured
    at `thr`; acc_gate is the MI-vs-rest accuracy of the pass / no-pass decision at `thr`.
    """
    out = {}
    for s in sorted(set(int(x) for x in subj)):
        m = subj == s
        sw = sweep_sujeto(y4[m], src[m], score[m], cls[m])
        base = metricas_sujeto(y4[m], src[m], score[m], cls[m], thr)
        es_mi = np.isin(y4[m], [LH, RH])
        pasa = score[m] >= thr
        acc_gate = float((pasa == es_mi).mean()) if len(es_mi) else np.nan
        out[s] = dict(pauc02=pauc(sw, 0.2),
                      recall_fpr10=recall_at_fpr(sw, 0.10),
                      recall_fpr05=recall_at_fpr(sw, 0.05),
                      fpr_idle=base['fpr_idle'],
                      recall_at_thr=base['recall'],
                      acc_gate=acc_gate)
    return out


def score_cls(models, form, Xz, batch=512):
    """(n, 8, w) z-scored windows -> (command score = 1 - P(IDLE), class LH/RH)."""
    p3 = probs_canonicas(models, form, Xz, batch)      # [P(LH),P(RH),P(IDLE)]
    score = (1.0 - p3[:, 2]).astype(np.float64)
    cls = p3[:, :2].argmax(1)
    return score, cls


def verificar_base(tag, form, models, Xz):
    """Check that the unoccluded inference reproduces preds_<tag>.npz (run_experiments.py)."""
    pr = np.load(MAIN / f'preds_{tag}.npz', allow_pickle=True)
    if form == 'D':
        pg = predict_proba(models[0], Xz); pdd = predict_proba(models[1], Xz)
        e1 = float(np.abs(pg - pr['proba_gate']).max())
        e2 = float(np.abs(pdd - pr['proba_disc']).max())
        err = max(e1, e2)
    else:
        p = predict_proba(models[0], Xz)
        err = float(np.abs(p - pr['proba']).max())
    ok = err < 1e-5
    print(f'  [base check] max|proba - preds| = {err:.2e}  {"OK" if ok else "FAIL"}',
          flush=True)
    assert ok, (f'unoccluded inference does not reproduce preds_{tag}.npz (err {err:.2e}); '
                f'the occlusions would not be reliable')


def delta_boot(base_ps: dict, occ_ps: dict, metrica: str, subs, boot: int) -> dict:
    """Importance = base - occluded per subject, with a subject-level bootstrap."""
    d = [base_ps[s][metrica] - occ_ps[s][metrica] for s in subs
         if base_ps[s][metrica] == base_ps[s][metrica]
         and occ_ps[s][metrica] == occ_ps[s][metrica]]
    mu, lo, hi = boot_mean(d, B=boot)
    return dict(delta=mu, lo=lo, hi=hi, n=len(d),
                veredicto=veredicto(lo, hi, len(d)),
                n_ayuda=int(sum(1 for x in d if x > 0)),
                n_estorba=int(sum(1 for x in d if x < 0)),
                por_sujeto={int(s): (base_ps[s][metrica] - occ_ps[s][metrica])
                            for s in subs
                            if base_ps[s][metrica] == base_ps[s][metrica]
                            and occ_ps[s][metrica] == occ_ps[s][metrica]})


def correr_celda(tag: str, Xraw, y4, src, subj, subs, boot: int) -> dict:
    ck = torch.load(MAIN / f'ckpt_{tag}.pt', weights_only=False, map_location=C.device)
    banda = ck['banda']; form = ck['form']; W = float(ck.get('W', 2.0))
    w_samp = int(round(W * C.FS_TGT))
    mu, sd = ck['mu'], ck['sd']
    lo_b, hi_b = BANDAS[banda]
    models, form = cargar_modelos(ck, w_samp)
    print(f'\n[{tag}] form={form} band={banda} W={W}s  n_val={len(Xraw)}', flush=True)

    # --- signal filtered to the model band (once) and z-scored base ---
    Xb = aplicar_banda(Xraw, banda)                    # (n, 8, w), model band
    Xz = ((Xb - mu) / sd).astype(np.float32)
    verificar_base(tag, form, models, Xz)

    sc0, cl0 = score_cls(models, form, Xz)
    base_ps = metricas_deteccion_por_sujeto(y4, src, sc0, cl0, subj)
    base_mean = {k: float(np.nanmean([base_ps[s][k] for s in subs])) for k in METRICAS}
    print('  base: ' + '  '.join(f'{k}={base_mean[k]:.3f}' for k in
                                  ['pauc02', 'recall_fpr10', 'fpr_idle', 'acc_gate']),
          flush=True)

    # --- channel occlusion: channel set to 0 after z-scoring ---
    canales = {}
    for c in range(8):
        Xo = Xz.copy(); Xo[:, c, :] = 0.0
        sc, cl = score_cls(models, form, Xo)
        occ_ps = metricas_deteccion_por_sujeto(y4, src, sc, cl, subj)
        canales[CANALES[c]] = {m: delta_boot(base_ps, occ_ps, m, subs, boot)
                               for m in METRICAS}
        d = canales[CANALES[c]]['pauc02']
        print(f'    channel {CANALES[c]:4s}  d_pauc={d["delta"]:+.4f} '
              f'[{d["lo"]:+.4f},{d["hi"]:+.4f}] {d["veredicto"]}', flush=True)

    # --- band occlusion: band-stop by subtraction within the model band ---
    bandas = {}
    for nb, a, b in SUBBANDAS:
        sub = bandpass(Xb, lo=a, hi=b, fs=C.FS_TGT)    # extracted sub-band
        Xba = (Xb - sub)                               # band-stop
        Xza = ((Xba - mu) / sd).astype(np.float32)
        sc, cl = score_cls(models, form, Xza)
        occ_ps = metricas_deteccion_por_sujeto(y4, src, sc, cl, subj)
        bandas[nb] = {m: delta_boot(base_ps, occ_ps, m, subs, boot) for m in METRICAS}
        bandas[nb]['rango_hz'] = [a, b]
        d = bandas[nb]['pauc02']
        print(f'    band {nb:6s} [{a:g}-{b:g}]  d_pauc={d["delta"]:+.4f} '
              f'[{d["lo"]:+.4f},{d["hi"]:+.4f}] {d["veredicto"]}', flush=True)

    del models
    if C.device.type == 'cuda':
        torch.cuda.empty_cache()
    return dict(tag=tag, form=form, banda=banda, W=W, n_val=int(len(Xraw)),
                n_sujetos=len(subs),
                base_mean=base_mean,
                base_por_sujeto={int(s): base_ps[s] for s in subs},
                canales=canales, bandas=bandas)


def escribir_csv(res: dict):
    """Long-format CSV: one row per cell / occlusion type / name / metric."""
    filas = []
    for tag, r in res.items():
        for tipo, dic in [('canal', r['canales']), ('banda', r['bandas'])]:
            for nombre, mets in dic.items():
                for met in METRICAS:
                    d = mets[met]
                    filas.append(dict(cell=tag, form=r['form'], banda_modelo=r['banda'],
                                      tipo=tipo, nombre=nombre, metrica=met,
                                      base=r['base_mean'][met], delta=d['delta'],
                                      lo=d['lo'], hi=d['hi'], veredicto=d['veredicto'],
                                      n_ayuda=d['n_ayuda'], n_estorba=d['n_estorba']))
    with open(OUT / 'occlusion.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)
    print(f'  -> {OUT / "occlusion.csv"}  ({len(filas)} rows)')


CELDAS_DEF = ['B_bb_W2_r1_s42', 'B_mb_W2_r1_s42', 'D_bb_W2_r1_s42',
              'C_bb_W2_r1_s42', 'A_bb_W2_r1_s42']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', default=CELDAS_DEF)
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        a.cells = ['B_bb_W2_r1_s42']; a.boot = 200

    t0 = time.time()
    print('=' * 80)
    print('Detection (MI vs rest): channel and band occlusion of the trained models')
    print(f'  device={C.device}  boot={a.boot}  cells={a.cells}')
    print('=' * 80, flush=True)

    z = np.load(CACHE / 'trials_mc4_W2_per_window.npz', allow_pickle=True)
    X, y4, subject, source = z['X'], z['y'], z['subject'], z['source']
    _, val_subs = split_canonico()
    m_va = np.isin(subject, val_subs)
    Xraw = X[m_va].astype(np.float32)
    y4v, srcv, subjv = y4[m_va], source[m_va].astype(str), subject[m_va].astype(int)
    subs = sorted(set(subjv.tolist()))
    print(f'val: {len(Xraw)} windows, {len(subs)} subjects {subs}', flush=True)
    print(f'  val sources: ' + '  '.join(
        f'{s}={int((srcv==s).sum())}' for s in ['mi', 'rest1', 'rest2']), flush=True)

    res = {}
    for tag in a.cells:
        if not (MAIN / f'ckpt_{tag}.pt').exists():
            print(f'[SKIP] ckpt_{tag}.pt not found', flush=True); continue
        res[tag] = correr_celda(tag, Xraw, y4v, srcv, subjv, subs, a.boot)
        (OUT / 'occlusion.json').write_text(
            json.dumps(res, indent=1, default=lambda o: (
                float(o) if isinstance(o, (np.floating, np.integer)) else str(o))))

    escribir_csv(res)
    print(f'\n[OK] {time.time()-t0:.0f}s')
    print(f'  -> {OUT / "occlusion.json"}')


if __name__ == '__main__':
    main()
