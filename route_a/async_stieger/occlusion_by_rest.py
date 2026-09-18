"""Importance of the delta band (0.5-4 Hz) for detection, split by rest source.

The two rest sources differ from MI in different ways:
  - REST2 = pre-cue window [-2000, 0) ms of the left/right trials. It differs from MI in
    its position relative to the cue, so a slow potential evoked by the cue could
    separate it from MI without any imagery.
  - REST1 = cued rest ('down' target of tasks 2 and 3), cropped at the MI onset
    (t >= 2000 ms) like MI. It differs from MI in mental state (and task block), not in
    its position relative to the cue.
If occluding delta lowers detection much more against REST2 than against REST1, the
delta contribution is largely the cue-evoked slow potential; if it lowers both alike,
delta carries state information.

For cells B_bb_W2_r1_s42 and D_bb_W2_r1_s42 and each rest source separately, computes
per subject the pAUC (fpr <= 0.2) and the recall at fpr <= 0.10 of MI against that
source, with and without delta (band-stop by subtraction, as in occlusion.py), and the
importance base - occluded with a subject-level bootstrap (B=10000). No retraining.

Reads cache/trials_mc4_W2_per_window.npz and outputs/main/ckpt_<cell>.pt.
Writes outputs/occlusion_by_rest.json.

Usage:
    python occlusion_by_rest.py
"""
from __future__ import annotations
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
from run_experiments import BANDAS, aplicar_banda, split_canonico   # noqa: E402
from infer_streams import cargar_modelos, probs_canonicas           # noqa: E402
from analyze import sweep_sujeto, pauc, recall_at_fpr, boot_mean, veredicto  # noqa: E402

CACHE = HERE / 'cache'
MAIN = HERE / 'outputs' / 'main'
OUT = HERE / 'outputs'
DELTA = (0.5, 4.0)


def score_cls(models, form, Xz, batch=512):
    """Command score (1 - P(IDLE)) and LH/RH class, as in occlusion.py."""
    p3 = probs_canonicas(models, form, Xz, batch)
    return (1.0 - p3[:, 2]).astype(np.float64), p3[:, :2].argmax(1)


def pauc_vs_fuente(y4, src, score, cls, subj, subs, neg: str):
    """Per-subject pAUC (fpr <= 0.2) and recall at fpr <= 0.10 of MI against the rest
    source `neg` only."""
    out = {}
    for s in subs:
        m = (subj == s) & (np.isin(y4, [LH, RH]) | (src == neg))
        if not m.any():
            out[s] = dict(pauc02=np.nan, recall_fpr10=np.nan); continue
        sw = sweep_sujeto(y4[m], src[m], score[m], cls[m])
        out[s] = dict(pauc02=pauc(sw, 0.2), recall_fpr10=recall_at_fpr(sw, 0.10))
    return out


def boot_imp(base, occ, met, subs, boot):
    """Importance = base - occluded per subject, with a subject-level bootstrap."""
    d = [base[s][met] - occ[s][met] for s in subs
         if base[s][met] == base[s][met] and occ[s][met] == occ[s][met]]
    mu, lo, hi = boot_mean(d, B=boot)
    return dict(delta=mu, lo=lo, hi=hi, n=len(d), veredicto=veredicto(lo, hi, len(d)))


def main():
    boot = 10000
    cells = ['B_bb_W2_r1_s42', 'D_bb_W2_r1_s42']
    z = np.load(CACHE / 'trials_mc4_W2_per_window.npz', allow_pickle=True)
    X, y4, subject, source = z['X'], z['y'], z['subject'], z['source']
    _, val = split_canonico()
    m = np.isin(subject, val)
    Xraw = X[m].astype(np.float32)
    y4v, srcv, subjv = y4[m], source[m].astype(str), subject[m].astype(int)
    subs = sorted(set(subjv.tolist()))

    print('=' * 78)
    print('Cue confound control: importance of delta (0.5-4 Hz) for detection')
    print('  MI vs REST1 (post-cue, differs in state) and MI vs REST2 (pre-cue, differs in timing)')
    print('=' * 78, flush=True)

    res = {}
    for tag in cells:
        ck = torch.load(MAIN / f'ckpt_{tag}.pt', weights_only=False, map_location=C.device)
        w_samp = int(round(float(ck['W']) * C.FS_TGT))
        mu, sd = ck['mu'], ck['sd']
        models, form = cargar_modelos(ck, w_samp)
        Xb = aplicar_banda(Xraw, ck['banda'])
        Xz = ((Xb - mu) / sd).astype(np.float32)
        sc0, cl0 = score_cls(models, form, Xz)                 # base
        sub = bandpass(Xb, lo=DELTA[0], hi=DELTA[1], fs=C.FS_TGT)
        Xzd = ((Xb - sub - mu) / sd).astype(np.float32)
        scd, cld = score_cls(models, form, Xzd)                # delta occluded

        fila = {'tag': tag}
        print(f'\n[{tag}]')
        for neg in ['rest1', 'rest2']:
            base = pauc_vs_fuente(y4v, srcv, sc0, cl0, subjv, subs, neg)
            occ = pauc_vs_fuente(y4v, srcv, scd, cld, subjv, subs, neg)
            b_mean = float(np.nanmean([base[s]['pauc02'] for s in subs]))
            o_mean = float(np.nanmean([occ[s]['pauc02'] for s in subs]))
            imp = boot_imp(base, occ, 'pauc02', subs, boot)
            imp_r = boot_imp(base, occ, 'recall_fpr10', subs, boot)
            fila[neg] = dict(pauc_base=b_mean, pauc_delta_ocluido=o_mean,
                             imp_delta_pauc=imp, imp_delta_recall_fpr10=imp_r)
            print(f'  MI vs {neg}: pAUC base={b_mean:.3f} -> without delta={o_mean:.3f}  '
                  f'imp_delta={imp["delta"]:+.4f} [{imp["lo"]:+.4f},{imp["hi"]:+.4f}] '
                  f'{imp["veredicto"]}')
        res[tag] = fila
        del models
        if C.device.type == 'cuda':
            torch.cuda.empty_cache()

    (OUT / 'occlusion_by_rest.json').write_text(json.dumps(res, indent=1, default=float))
    print(f'\n-> {OUT / "occlusion_by_rest.json"}')


if __name__ == '__main__':
    main()
