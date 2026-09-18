"""Train the formulations of the asynchronous four-class study (LH / RH / REST1 / REST2).

All formulations share preprocessing, subject split, seeds, architecture and
hyperparameters; they differ only in the training labels and the output head. All of
them predict the same validation windows, which contain the four classes, so a binary
model that never saw rest during training is still evaluated on rest windows.

  A  binary              LH/RH            trained on source == 'mi' only
  B  three classes       LH/RH/IDLE       IDLE = rest1 + rest2
  C  four classes        LH/RH/R1/R2
  D  hierarchical        gate + LH/RH     two models: MI vs rest, then LH vs RH
  E  binary + threshold  LH/RH + reject   not trained by default: analyze.py sweeps a
                                          rejection threshold over the probabilities
                                          of A

Reads cache/trials_mc4_W{W}_per_window.npz (build_dataset.py) and
../split_80_20_subjects.json (50 training / 12 validation subjects). Windows are
band-passed (mb = 8-30 Hz, bb = 0.5-40 Hz) and z-scored with training statistics, and
training rest windows are subsampled to at most --ratios times the number of MI windows.
For each cell (formulation, band, ratio, seed), with
tag = {form}_{band}_W{W}_r{ratio}_s{seed}, it writes to outputs/main (outputs/smoke with
--smoke, or --out):

  preds_{tag}.npz       raw class probabilities per validation window, with subject,
                        trial_id, source, tasknumber, mi_valido, y_true4 and f_amp, so
                        that analyze.py computes every metric without retraining
  ckpt_{tag}.pt         model weights and the training mu/sd (unless --no-ckpt)
  cells_summary.json    summary of the cells run so far

Usage:
    python run_experiments.py --dry-run        # print the plan, no training
    python run_experiments.py --smoke          # smoke cache (S1-S4), 3 epochs: code check
    python run_experiments.py                  # A B C D x mb bb x seeds 42 43
    python run_experiments.py --formulaciones C --bandas bb --seeds 42
"""
from __future__ import annotations
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

from rt_system import config as C                       # noqa: E402
from rt_system.models import build_model                # noqa: E402
from rt_system.train import train_loop                  # noqa: E402
from rt_system.preprocessing import bandpass            # noqa: E402

from build_dataset import LH, RH, REST1, REST2          # noqa: E402

CACHE = HERE / 'cache'
OUT = HERE / 'outputs'
SPLIT_JSON = HERE.parent / 'split_80_20_subjects.json'

BANDAS = {'mb': (8.0, 30.0), 'bb': (0.5, 40.0)}
FORMULACIONES = ['A', 'B', 'C', 'D', 'E']

# Training labels per formulation: etiquetas_train returns the mask of training windows.
# The evaluation set is always the same and contains the four classes.
IDLE_B = 2                                              # id of the merged IDLE class in B


def etiquetas_train(form: str, y4: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (mask of windows used, remapped labels, number of classes)."""
    if form in ('A', 'E'):                              # MI only, 2 classes
        m = np.isin(y4, [LH, RH])
        return m, y4[m], 2
    if form == 'B':                                     # LH/RH/IDLE
        lab = np.where(np.isin(y4, [REST1, REST2]), IDLE_B, y4)
        return np.ones(len(y4), bool), lab, 3
    if form == 'C':                                     # the 4 classes as they are
        return np.ones(len(y4), bool), y4.copy(), 4
    raise ValueError(form)


# ============================================================
# Loading, band-pass, split, balancing
# ============================================================
def cargar(cache_path: Path) -> dict:
    z = np.load(cache_path, allow_pickle=True)
    d = {k: z[k] for k in ['X', 'y', 'subject', 'session', 'source',
                           'tasknumber', 'trial_id', 'f_amp', 'mi_valido']}
    d['hit_rate'] = {int(s): float(v) for s, v in zip(z['hit_subjects'], z['hit_values'])}
    d['meta'] = list(z['meta'])
    return d


def aplicar_banda(X: np.ndarray, banda: str, causal: bool = False,
                  chunk: int = 4000) -> np.ndarray:
    """Band-pass in chunks of `chunk` windows to bound memory use."""
    lo, hi = BANDAS[banda]
    out = np.empty_like(X)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = bandpass(X[i:i + chunk], lo=lo, hi=hi,
                                    fs=C.FS_TGT, causal=causal)
    return out


def split_canonico() -> tuple[list[int], list[int]]:
    """Canonical subject split of the repository (50 train / 12 validation subjects).

    Used instead of the skill-stratified split of rt_system so that the numbers are
    comparable with the rest of the paper; that split is available with `--split rt`.
    """
    sp = json.loads(SPLIT_JSON.read_text(encoding='utf-8'))
    tr = sorted(int(s[1:]) if isinstance(s, str) else int(s) for s in sp['train_subjects'])
    va = sorted(int(s[1:]) if isinstance(s, str) else int(s) for s in sp['val_subjects'])
    return tr, va


def split_rt(subject, hit_rate) -> tuple[list[int], list[int]]:
    from rt_system.dataset import split_cross_subject
    s = split_cross_subject(subject, hit_rate, n_test=12)
    return s['train'], s['test']


def balancear_por_subfuente(y4: np.ndarray, source: np.ndarray, ratio: float,
                            rng: np.random.RandomState) -> np.ndarray:
    """Subsample rest to `ratio` = n_rest / n_MI, keeping the rest1/rest2 proportion.

    Returns a boolean mask of the windows kept.
    """
    keep = np.ones(len(y4), bool)
    n_mi = int(np.isin(y4, [LH, RH]).sum())
    idx_r = {s: np.where(source == s)[0] for s in ('rest1', 'rest2')}
    n_r = sum(len(v) for v in idx_r.values())
    if n_r == 0:
        return keep
    objetivo = int(round(ratio * n_mi))
    if objetivo >= n_r:
        return keep
    for s, idx in idx_r.items():
        cuota = int(round(objetivo * len(idx) / n_r))       # proportional quota
        cuota = min(cuota, len(idx))
        drop = rng.choice(idx, size=len(idx) - cuota, replace=False)
        keep[drop] = False
    return keep


_CACHE_BANDA: dict = {}


def _cache_banda(X, banda, causal, split_mode, m_tr):
    """Band-pass and z-score once per (band, causal, split).

    The z-score is applied in place to avoid temporary copies of the full array.
    """
    key = (banda, bool(causal), split_mode)
    if key in _CACHE_BANDA:
        return _CACHE_BANDA[key]
    Xb = aplicar_banda(X, banda, causal=causal)
    mu = Xb[m_tr].mean(axis=(0, 2), keepdims=True).astype(np.float32)
    sd = (Xb[m_tr].std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
    Xb -= mu
    Xb /= sd
    _CACHE_BANDA.clear()          # keep a single band in memory
    _CACHE_BANDA[key] = (Xb, mu, sd)
    return _CACHE_BANDA[key]


@torch.no_grad()
def predict_proba(model, X: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().unsqueeze(1).to(C.device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def pesos_clase(y: np.ndarray, n_clases: int) -> torch.Tensor:
    cnt = np.bincount(y, minlength=n_clases).astype(float)
    w = cnt.sum() / (n_clases * np.maximum(cnt, 1))
    return torch.tensor(w, dtype=torch.float32, device=C.device)


# ============================================================
# One cell of the experiment matrix
# ============================================================
def correr_celda(d: dict, form: str, banda: str, ratio: float, seed: int,
                 epochs: int, arch: str, split_mode: str, causal: bool,
                 out_dir: Path, W: float, guardar_ckpt: bool = True) -> dict:
    t0 = time.time()
    tag = f'{form}_{banda}_W{W:g}_r{ratio:g}_s{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'\n{"="*74}\n[{tag}] formulation {form} | band {banda} | ratio {ratio} | seed {seed}',
          flush=True)

    C.set_seed(seed)
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    subject, y4, source = d['subject'], d['y'], d['source']
    if split_mode == 'canon':
        tr_s, va_s = split_canonico()
    else:
        tr_s, va_s = split_rt(subject, d['hit_rate'])
    tr_s_set, va_s_set = set(tr_s), set(va_s)

    # cache subjects that are not in the split (e.g. with --smoke) are ignored
    m_tr = np.array([int(s) in tr_s_set for s in subject])
    m_va = np.array([int(s) in va_s_set for s in subject])
    if not m_tr.any() or not m_va.any():
        raise SystemExit(f'empty split: train={m_tr.sum()} val={m_va.sum()} '
                         f'(subjects in cache: {sorted(set(subject.tolist()))})')

    # ---- leakage check: no trial and no subject in both partitions ----
    t_tr = set(d['trial_id'][m_tr].tolist())
    t_va = set(d['trial_id'][m_va].tolist())
    inter = t_tr & t_va
    assert not inter, f'leakage: {len(inter)} trial_id in both train and val'
    s_tr = set(subject[m_tr].tolist()); s_va = set(subject[m_va].tolist())
    assert not (s_tr & s_va), f'leakage: shared subjects {s_tr & s_va}'
    print(f'  [OK] no leakage: {len(t_tr):,d} train trials / {len(t_va):,d} val, '
          f'intersection 0 | subjects {len(s_tr)}/{len(s_va)}')

    # ---- band-pass + z-score with training statistics only ----
    # Filtering depends only on (band, causal) and the z-score statistics on the split,
    # which is fixed for the whole run, so the result is cached across cells.
    Xb, mu, sd = _cache_banda(d['X'], banda, causal, split_mode, m_tr)

    # ---- rest balancing in train only (validation keeps its natural proportions) ----
    idx_tr = np.where(m_tr)[0]
    keep = balancear_por_subfuente(y4[idx_tr], source[idx_tr], ratio, rng)
    idx_tr = idx_tr[keep]

    Xva, yva4 = Xb[m_va], y4[m_va]
    res = {'tag': tag, 'form': form, 'banda': banda, 'ratio': ratio, 'seed': seed,
           'W': W, 'arch': arch, 'split': split_mode, 'epochs': epochs,
           'n_train': int(len(idx_tr)), 'n_val': int(m_va.sum()),
           'subjects_train': sorted(int(s) for s in s_tr),
           'subjects_val': sorted(int(s) for s in s_va)}

    guardar = dict(subject=subject[m_va], trial_id=d['trial_id'][m_va],
                   source=source[m_va], tasknumber=d['tasknumber'][m_va],
                   mi_valido=d['mi_valido'][m_va], y_true4=yva4,
                   f_amp=d['f_amp'][m_va])

    if form == 'D':
        # --- hierarchical: gate (MI vs rest) + discriminator (LH vs RH) ---
        y_gate = np.isin(y4[idx_tr], [REST1, REST2]).astype(np.int64)   # 1 = rest
        mg = build_model(arch, n_channels=8, n_samples=Xb.shape[2], n_classes=2).to(C.device)
        train_loop(mg, Xb[idx_tr], y_gate, epochs=epochs, lr=C.LR_PRETRAIN,
                   batch=C.BATCH, tag=f'{tag}/gate', weights=pesos_clase(y_gate, 2))
        p_gate = predict_proba(mg, Xva)

        sel = idx_tr[np.isin(y4[idx_tr], [LH, RH])]
        y_disc = y4[sel].astype(np.int64)
        md = build_model(arch, n_channels=8, n_samples=Xb.shape[2], n_classes=2).to(C.device)
        train_loop(md, Xb[sel], y_disc, epochs=epochs, lr=C.LR_PRETRAIN,
                   batch=C.BATCH, tag=f'{tag}/disc', weights=pesos_clase(y_disc, 2))
        p_disc = predict_proba(md, Xva)

        guardar.update(proba_gate=p_gate, proba_disc=p_disc)
        res['n_clases'] = 2
        res['cabezas'] = ['gate(MI=0,rest=1)', 'disc(LH=0,RH=1)']
        if guardar_ckpt:
            torch.save({'gate': mg.state_dict(), 'disc': md.state_dict(),
                        'mu': mu, 'sd': sd, 'arch': arch, 'n_clases': 2,
                        'form': form, 'banda': banda, 'W': W},
                       out_dir / f'ckpt_{tag}.pt')
        del mg, md
    else:
        m_use, y_use, n_cls = etiquetas_train(form, y4[idx_tr])
        sel = idx_tr[m_use]
        model = build_model(arch, n_channels=8, n_samples=Xb.shape[2],
                            n_classes=n_cls).to(C.device)
        train_loop(model, Xb[sel], y_use.astype(np.int64), epochs=epochs,
                   lr=C.LR_PRETRAIN, batch=C.BATCH, tag=tag,
                   weights=pesos_clase(y_use.astype(np.int64), n_cls))
        guardar['proba'] = predict_proba(model, Xva)
        res['n_clases'] = n_cls
        res['n_train_usado'] = int(len(sel))
        if guardar_ckpt:
            # mu/sd are saved with the checkpoint so that new windows (infer_streams.py,
            # cue_free_control.py) are normalized with the training statistics
            torch.save({'model': model.state_dict(), 'mu': mu, 'sd': sd,
                        'arch': arch, 'n_clases': n_cls, 'form': form,
                        'banda': banda, 'W': W}, out_dir / f'ckpt_{tag}.pt')
        del model

    if C.device.type == 'cuda':
        torch.cuda.empty_cache()

    np.savez_compressed(out_dir / f'preds_{tag}.npz', **guardar,
                        meta=np.array([f'{k}={v}' for k, v in res.items()]))
    res['segundos'] = round(time.time() - t0, 1)
    print(f'  saved preds_{tag}.npz  ({res["segundos"]}s)', flush=True)
    return res


# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', default=None, help='npz written by build_dataset.py')
    ap.add_argument('--W', type=float, default=2.0)
    ap.add_argument('--formulaciones', nargs='+', default=['A', 'B', 'C', 'D'],
                    choices=FORMULACIONES,
                    help='E is not trained by default: analyze.py derives it from A')
    ap.add_argument('--bandas', nargs='+', default=['mb', 'bb'], choices=list(BANDAS))
    ap.add_argument('--ratios', type=float, nargs='+', default=[1.0])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    ap.add_argument('--arch', default='conformer')
    ap.add_argument('--epochs', type=int, default=C.EPOCHS_PRETRAIN)
    ap.add_argument('--split', default='canon', choices=['canon', 'rt'])
    ap.add_argument('--causal', action='store_true', help='causal band-pass (online use)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--smoke', action='store_true',
                    help='smoke cache + 3 epochs: checks the code, results are not meaningful')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and exit')
    ap.add_argument('--no-ckpt', action='store_true',
                    help='do not save model weights (infer_streams.py and '
                         'cue_free_control.py need them)')
    a = ap.parse_args()

    if a.smoke:
        a.epochs = 3; a.seeds = [42]; a.bandas = ['mb']; a.ratios = [1.0]
        cache = CACHE / f'trials_mc4_W{a.W:g}_per_window_smoke.npz'
    else:
        cache = Path(a.cache) if a.cache else CACHE / f'trials_mc4_W{a.W:g}_per_window.npz'
    out_dir = Path(a.out) if a.out else (OUT / ('smoke' if a.smoke else 'main'))

    # Band is the outer loop: the filtered array is cached for one band at a time, so
    # alternating bands would filter the whole cache again for every cell.
    celdas = [(f, b, r, s) for b, f, r, s in
              itertools.product(a.bandas, a.formulaciones, a.ratios, a.seeds)]
    print(f'cache : {cache}')
    print(f'output: {out_dir}')
    print(f'cells: {len(celdas)}  (formulations={a.formulaciones} bands={a.bandas} '
          f'ratios={a.ratios} seeds={a.seeds})')
    print(f'arch={a.arch} epochs={a.epochs} split={a.split} causal={a.causal} '
          f'device={C.device}')
    for f, b, r, s in celdas:
        print(f'   - {f}_{b}_W{a.W:g}_r{r:g}_s{s}')
    if a.dry_run:
        print('\n[dry-run] nothing is trained.')
        return
    if not cache.exists():
        raise SystemExit(f'missing cache {cache}. Run first:\n'
                         f'  python build_dataset.py --W {a.W:g}'
                         f'{" --smoke" if a.smoke else ""}')

    d = cargar(cache)
    print(f'\ncache loaded: X{d["X"].shape}  meta={d["meta"]}', flush=True)

    resultados, t0 = [], time.time()
    for f, b, r, s in celdas:
        resultados.append(correr_celda(d, f, b, r, s, a.epochs, a.arch,
                                       a.split, a.causal, out_dir, a.W,
                                       guardar_ckpt=not a.no_ckpt))
        (out_dir / 'cells_summary.json').write_text(json.dumps(resultados, indent=1))
    print(f'\n{len(resultados)} cells in {time.time()-t0:.0f}s -> {out_dir}')
    print('next step:  python analyze.py --dir', out_dir)


if __name__ == '__main__':
    main()
