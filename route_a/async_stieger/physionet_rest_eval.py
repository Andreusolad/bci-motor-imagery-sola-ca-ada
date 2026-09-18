"""Score the Stieger-trained gates against passive rest from Physionet EEGMMIDB (zero-shot).

Measures how often the gate fires while the user sits without a task or a cue. All window
types come from the same Physionet subject, session and amplifier, so differences between
the rest types are not caused by the change of dataset, which affects all of them alike:

    positives    mi        motor imagery (runs 4/8/12, T1 = left / T2 = right)
    negative A   rest_t0   T0 rest between MI trials (task context)
    negative B   rest_eo   run R01, 60 s of continuous rest, eyes open
    negative C   rest_ec   run R02, 60 s of continuous rest, eyes closed

The models are the checkpoints trained on Stieger only by run_experiments.py
(outputs/main/ckpt_<cell>.pt by default); nothing is trained or adapted here.

Normalization (--norm; the z-score is where the change of amplifier enters):
  ckpt   fixed mu/sd stored in the checkpoint.
  gain   per-subject gain, then the checkpoint mu/sd. The gain scales each subject so that
         the median per-window std of its eyes-open rest (rest_eo) windows equals the median
         per-window std of the Stieger training windows in the same band (computed here and
         saved in the JSON). A robust statistic is matched because the checkpoint sd is
         dominated by a few high-amplitude REST2 windows; re-estimating mu/sd on the
         subject's own rest would put the inputs far outside the training distribution.
  ambos  both.
Within a subject, rest_t0 and rest_eo windows are scaled identically in either mode.

Metrics:
  - per window: AUC, pAUC (fpr <= 0.2), recall at fpr 0.10 and fpr at threshold 0.5 of MI
    against each rest type; recall at 0.5; direction accuracy acc_dir on MI windows, as a
    transfer check (if acc_dir is at chance the model does not transfer and the rest
    comparison is meaningless). MI and rest_t0 contribute the window whose start is nearest
    to 0.5 s after the event onset; rest_eo and rest_ec contribute every window.
  - per intent: false commands per minute with the real accumulator (dwell C.DWELL_MS,
    refractory C.REFRACTORY_MS, stride --stride-ms) over all windows of each rest type,
    cut into chunks with the same number of windows (see fp_por_minuto), at the
    thresholds THRS_FPM.
  - paired contrast rest_t0 vs rest_eo (AUC, pAUC, fpr at 0.5).
  Subject-level bootstrap with --boot resamples (default 10000); a CI that crosses 0 is
  reported as inconclusive.

Reads cache/pn_rest.npz (from physionet_rest_cache.py; pn_rest_smoke.npz with --smoke), the
checkpoints and, for --norm gain, cache/trials_mc4_W2_per_window.npz.
Writes outputs/physionet_rest_eval<suffix>.json and .csv, where <suffix> is '_smoke' with
--smoke followed by --sufijo.

Usage:
    python physionet_rest_eval.py --smoke
    python physionet_rest_eval.py --norm ambos
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

from rt_system import config as C                                    # noqa: E402
from rt_system.preprocessing import bandpass                         # noqa: E402
from rt_system.accumulator import EvidenceAccumulator                # noqa: E402
from rt_system.gate import GateDecision                              # noqa: E402
from run_experiments import BANDAS                                   # noqa: E402
from infer_streams import cargar_modelos, probs_canonicas            # noqa: E402
from analyze import THRS, pauc, recall_at_fpr, boot_mean, boot_paired, veredicto  # noqa: E402

CACHE = HERE / 'cache'
MAIN = HERE / 'outputs' / 'main'
OUT = HERE / 'outputs'
IDLE3 = 2

KINDS = ['mi', 'rest_t0', 'rest_eo', 'rest_ec']
NEGS = ['rest_t0', 'rest_eo', 'rest_ec']
CELDAS_DEF = ['B_bb_W2_r1_s42', 'D_bb_W2_r1_s42', 'B_mb_W2_r1_s42']
OFFSET_PRIM_MS = 500.0        # canonical window: starts 0.5 s after the event onset
N_WIN_CHUNK = 16              # windows per accumulator chunk; 16 x 125 ms = 2.0 s of
                              # decision time. Largest value that still gives a complete
                              # chunk in the shortest T0 segments (~4.2 s -> 18 positions)
THRS_FPM = [0.5, 0.8, 0.9, 0.95, 0.99]


def med_std_train_stieger(banda: str, paso: int = 7) -> float:
    """Median per-window std of the Stieger training windows, band-passed to `banda`.

    The median is used because the mean is dominated by the high-amplitude REST2 tail.
    One window in every `paso` is used, to save memory; the value is saved in the JSON.
    """
    from run_experiments import aplicar_banda, split_canonico
    z = np.load(CACHE / 'trials_mc4_W2_per_window.npz', allow_pickle=True)
    tr, _ = split_canonico()
    m = np.isin(z['subject'], tr)
    Xs = z['X'][m][::paso].astype(np.float32)
    Xb = aplicar_banda(Xs, banda)
    v = float(np.median(Xb.std(axis=(1, 2))))
    del Xs, Xb, z
    return v


def ventanas_segmento(x: np.ndarray, w_samp: int, stride: int):
    """(8, n) -> (k, 8, w) windows with their own mean removed, plus their start indices.

    Cast to float32 after centring, as in the Stieger cache. (None, None) if n < w.
    """
    idx = list(range(0, x.shape[1] - w_samp + 1, stride))
    if not idx:
        return None, None
    W = np.stack([x[:, s:s + w_samp] for s in idx])
    W = (W - W.mean(axis=2, keepdims=True)).astype(np.float32)
    return W, np.array(idx, np.int64)


def sweep_simple(sc_pos: np.ndarray, sc_neg: np.ndarray) -> list[dict]:
    """Recall/fpr curve over the THRS grid of analyze.py (same metric definition)."""
    return [dict(thr=float(t), recall=float((sc_pos >= t).mean()),
                 fpr_idle=float((sc_neg >= t).mean())) for t in THRS]


def auc(pos, neg) -> float:
    """Mann-Whitney AUC with average ranks for ties; NaN if either set is empty."""
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    v = np.concatenate([pos, neg]); order = np.argsort(v, kind='mergesort')
    r = np.empty(len(v), float); vs = v[order]; i = 0
    while i < len(vs):
        j = i
        while j + 1 < len(vs) and vs[j + 1] == vs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def fp_por_minuto(p3: np.ndarray, seg_id: np.ndarray, thr: float,
                  stride_ms: float, n_win_chunk: int = N_WIN_CHUNK) -> float:
    """False commands per minute with the real accumulator, on equal-length chunks.

    Each segment is cut into chunks of `n_win_chunk` consecutive windows (the same number
    for every rest type; leftover windows are dropped) and the accumulator is reset at the
    start of each chunk. Chunks are defined by window count, not by seconds, because a
    window needs 2 s of signal: a 4.2 s T0 segment yields only ~18 window positions while
    the same duration of continuous rest yields ~34, so normalizing by raw seconds would
    favour continuous rest. Equal chunks also give the dwell start-up (600 ms, ~5 windows)
    the same weight for every rest type.

    The denominator is decision time = number of windows x stride.
    """
    acc = EvidenceAccumulator(strategy='dwell', dwell_ms=C.DWELL_MS,
                              refractory_ms=C.REFRACTORY_MS, emit_thr=0.0,
                              ema_alpha=C.EMA_ALPHA, hmm_stay=C.HMM_STAY,
                              stride_ms=stride_ms)
    n_cmd, n_chunks = 0, 0
    for sg in np.unique(seg_id):
        idx = np.where(seg_id == sg)[0]            # consecutive windows of one segment
        for c0 in range(0, len(idx) - n_win_chunk + 1, n_win_chunk):
            acc.reset(); n_chunks += 1
            for p in p3[idx[c0:c0 + n_win_chunk]]:
                cls = int(np.argmax(p[:2]))
                ic = bool((1.0 - p[IDLE3]) >= thr)
                cmd = acc.update(GateDecision(label=cls if ic else -1, control_class=cls,
                                              probs=p, is_control=ic))
                if cmd is not None:
                    n_cmd += 1
    mins = n_chunks * n_win_chunk * stride_ms / 1000.0 / 60.0
    return float(n_cmd / mins) if mins > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cells', nargs='+', default=CELDAS_DEF)
    ap.add_argument('--norm', default='ambos', choices=['ckpt', 'gain', 'ambos'])
    ap.add_argument('--stride-ms', type=float, default=125.0)
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--ckpt-dir', default=None,
                    help='checkpoint directory (default: outputs/main)')
    ap.add_argument('--sufijo', default='',
                    help='suffix for the output file names, to keep the main run intact')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    ck_dir = Path(a.ckpt_dir) if a.ckpt_dir else MAIN
    if a.smoke:
        a.cells = ['B_bb_W2_r1_s42']; a.boot = 200

    cache = CACHE / ('pn_rest_smoke.npz' if a.smoke else 'pn_rest.npz')
    if not cache.exists():
        raise SystemExit(f'{cache} not found. Run first: python physionet_rest_cache.py')
    z = np.load(cache, allow_pickle=True)
    sig, offset = z['sig'], z['offset']
    subject, kind, yseg = z['subject'].astype(int), z['kind'].astype(str), z['y']
    subs = sorted(set(subject.tolist()))
    normas = ['ckpt', 'gain'] if a.norm == 'ambos' else [a.norm]

    t0 = time.time()
    print('=' * 80)
    print('Stieger-trained gates vs passive rest (Physionet EEGMMIDB, zero-shot)')
    print(f'  device={C.device}  subjects={len(subs)}  cells={a.cells}  norm={normas}')
    print(f'  segments: ' + '  '.join(f'{k}={int((kind==k).sum())}' for k in KINDS))
    print('=' * 80, flush=True)

    res = {}
    for tag in a.cells:
        cp = ck_dir / f'ckpt_{tag}.pt'
        if not cp.exists():
            print(f'[SKIP] {cp.name} not found', flush=True); continue
        ck = torch.load(cp, weights_only=False, map_location=C.device)
        W = float(ck.get('W', 2.0)); w_samp = int(round(W * C.FS_TGT))
        banda, form = ck['banda'], ck['form']
        lo_b, hi_b = BANDAS[banda]
        models, form = cargar_modelos(ck, w_samp)
        stride = int(round(a.stride_ms / 1000 * C.FS_TGT))
        print(f'\n[{tag}] form={form} band={banda} W={W}s stride={stride} samp', flush=True)

        med_train = float('nan')
        if 'gain' in normas:
            med_train = med_std_train_stieger(banda)
            print(f'  median per-window std of the Stieger training windows (band {banda}) = '
                  f'{med_train:.3f} uV  -> reference for the gain calibration',
                  flush=True)

        por_norma = {n: {s: {} for s in subs} for n in normas}
        gains = {n: {} for n in normas}
        for si, s in enumerate(subs, 1):
            idx_seg = np.where(subject == s)[0]
            # --- dense windows of every segment of the subject ---
            Xs, ks, ys, chunks, prim = [], [], [], [], []
            for j in idx_seg:
                x = sig[:, offset[j]:offset[j + 1]].astype(np.float64)
                Wn, ini = ventanas_segmento(x, w_samp, stride)
                if Wn is None:
                    continue
                Xs.append(Wn)
                ks.append(np.full(len(Wn), kind[j], '<U8'))
                ys.append(np.full(len(Wn), yseg[j], np.int64))
                chunks.append(np.full(len(Wn), j, np.int64))   # segment id
                # canonical window = the one starting nearest to 0.5 s after the onset
                pm = np.zeros(len(Wn), bool)
                pm[int(np.argmin(np.abs(ini - OFFSET_PRIM_MS / 1000 * C.FS_TGT)))] = True
                prim.append(pm)
            Xw = np.concatenate(Xs); kw = np.concatenate(ks); yw = np.concatenate(ys)
            cw = np.concatenate(chunks); pw = np.concatenate(prim)
            Xb = bandpass(Xw, lo=lo_b, hi=hi_b, fs=C.FS_TGT)

            for nrm in normas:
                mu, sd = ck['mu'], ck['sd']
                if nrm == 'ckpt':
                    g = 1.0
                else:
                    # gain: match the median per-window std of the subject's eyes-open
                    # rest to the Stieger training median (same band)
                    ref = Xb[kw == 'rest_eo']
                    g = float(med_train / max(np.median(ref.std(axis=(1, 2))), 1e-9))
                    gains[nrm][int(s)] = g
                Xz = ((Xb * g - mu) / sd).astype(np.float32)
                p3 = probs_canonicas(models, form, Xz)
                sc = (1.0 - p3[:, IDLE3]).astype(np.float64)
                cl = p3[:, :2].argmax(1)

                d = por_norma[nrm][s]
                m_mi_p = (kw == 'mi') & pw
                d['acc_dir'] = float((cl[m_mi_p] == yw[m_mi_p]).mean())
                d['n_mi'] = int(m_mi_p.sum())
                for neg in NEGS:
                    # rest_t0 uses its canonical window only, like MI; continuous rest
                    # uses every window
                    m_neg = (kw == neg) if neg != 'rest_t0' else ((kw == neg) & pw)
                    sw = sweep_simple(sc[m_mi_p], sc[m_neg])
                    d[neg] = dict(
                        auc=auc(sc[m_mi_p], sc[m_neg]),
                        pauc02=pauc(sw, 0.2), recall_fpr10=recall_at_fpr(sw, 0.10),
                        fpr_05=float((sc[m_neg] >= 0.5).mean()),
                        n=int(m_neg.sum()),
                        fp_min={f'{t:g}': fp_por_minuto(p3[kw == neg], cw[kw == neg],
                                                        t, a.stride_ms)
                                for t in THRS_FPM})
                d['recall_05'] = float((sc[m_mi_p] >= 0.5).mean())
            if si % 20 == 0 or a.smoke or si == len(subs):
                print(f'    subject {si}/{len(subs)} ({time.time()-t0:.0f}s)', flush=True)

        # ---- aggregation + subject-level bootstrap ----
        celda = dict(tag=tag, form=form, banda=banda, W=W, normas={},
                     med_std_train_stieger=med_train,
                     ganancia_por_sujeto=gains.get('gain', {}))
        for nrm in normas:
            P = por_norma[nrm]
            agg = {}
            for met in ['acc_dir', 'recall_05']:
                v = [P[s][met] for s in subs]
                mu_, lo_, hi_ = boot_mean(v, B=a.boot)
                agg[met] = dict(media=mu_, lo=lo_, hi=hi_)
            for neg in NEGS:
                agg[neg] = {}
                for met in ['auc', 'pauc02', 'recall_fpr10', 'fpr_05']:
                    v = [P[s][neg][met] for s in subs]
                    mu_, lo_, hi_ = boot_mean(v, B=a.boot)
                    agg[neg][met] = dict(media=mu_, lo=lo_, hi=hi_,
                                         por_sujeto={int(s): float(P[s][neg][met])
                                                     for s in subs})
                agg[neg]['fp_min'] = {}
                for t in THRS_FPM:
                    v = [P[s][neg]['fp_min'][f'{t:g}'] for s in subs]
                    mu_, lo_, hi_ = boot_mean(v, B=a.boot)
                    agg[neg]['fp_min'][f'{t:g}'] = dict(media=mu_, lo=lo_, hi=hi_)
            # paired contrast: within-task rest vs pure rest (same subjects)
            agg['contraste_t0_vs_eo'] = {}
            for met in ['auc', 'pauc02', 'fpr_05']:
                d_, lo_, hi_, n_, pg = boot_paired([P[s]['rest_t0'][met] for s in subs],
                                                   [P[s]['rest_eo'][met] for s in subs],
                                                   B=a.boot)
                agg['contraste_t0_vs_eo'][met] = dict(delta=d_, lo=lo_, hi=hi_, n=n_,
                                                      p_gt0=pg,
                                                      veredicto=veredicto(lo_, hi_, n_))
            celda['normas'][nrm] = agg
            print(f'  [{nrm}] acc_dir={agg["acc_dir"]["media"]:.3f} '
                  f'[{agg["acc_dir"]["lo"]:.3f},{agg["acc_dir"]["hi"]:.3f}]', flush=True)
            for neg in NEGS:
                g = agg[neg]
                print(f'      {neg:8s} AUC={g["auc"]["media"]:.3f} '
                      f'[{g["auc"]["lo"]:.3f},{g["auc"]["hi"]:.3f}]  '
                      f'pAUC={g["pauc02"]["media"]:.3f}  fpr@.5={g["fpr_05"]["media"]:.3f}  '
                      f'FP/min@.5={g["fp_min"]["0.5"]["media"]:.2f} '
                      f'@.95={g["fp_min"]["0.95"]["media"]:.2f}', flush=True)
            c = agg['contraste_t0_vs_eo']['auc']
            print(f'      contrast AUC(t0)-AUC(eo) = {c["delta"]:+.4f} '
                  f'[{c["lo"]:+.4f},{c["hi"]:+.4f}]  {c["veredicto"]}', flush=True)

        res[tag] = celda
        del models
        if C.device.type == 'cuda':
            torch.cuda.empty_cache()
        suf = ('_smoke' if a.smoke else '') + a.sufijo
        (OUT / f'physionet_rest_eval{suf}.json').write_text(
            json.dumps(res, indent=1, default=float), encoding='utf-8')

    # flat CSV
    filas = []
    for tag, c in res.items():
        for nrm, agg in c['normas'].items():
            for neg in NEGS:
                g = agg[neg]
                filas.append(dict(cell=tag, norm=nrm, negativo=neg,
                                  auc=g['auc']['media'], auc_lo=g['auc']['lo'],
                                  auc_hi=g['auc']['hi'], pauc02=g['pauc02']['media'],
                                  recall_fpr10=g['recall_fpr10']['media'],
                                  fpr_05=g['fpr_05']['media'],
                                  fp_min_05=g['fp_min']['0.5']['media'],
                                  fp_min_095=g['fp_min']['0.95']['media'],
                                  acc_dir=agg['acc_dir']['media']))
    suf = ('_smoke' if a.smoke else '') + a.sufijo
    if filas:
        with open(OUT / f'physionet_rest_eval{suf}.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader(); w.writerows(filas)
    print(f'\n[OK] {time.time()-t0:.0f}s -> {OUT / f"physionet_rest_eval{suf}.json"}')


if __name__ == '__main__':
    main()
