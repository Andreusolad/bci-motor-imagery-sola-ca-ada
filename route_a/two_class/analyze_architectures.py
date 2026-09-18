#!/usr/bin/env python
r"""Aggregate the compare_architectures.py runs: EEGSym vs dual-branch, transfer to BCI-IV-2a.

Reads outputs/results_arch_seed*.csv, averages the seeds per subject and reports, with a
subject-level bootstrap (B = 10000, 95% percentile intervals):
  - the level of each arm: zero-shot, calibrated (after fine-tuning) and lift;
  - the architecture contrast at each crop length, EEGSym - dual-branch, paired by subject;
  - the crop-length contrast per architecture, W = 3.0 s - W = 0.5 s (zero-shot), paired.
A difference is called real only if its interval excludes 0.

Output: outputs/architectures.md (the printed report).

Usage:
  python analyze_architectures.py
"""
import sys, glob
from pathlib import Path
from collections import defaultdict
import csv as csvmod
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / 'outputs'
ARMS = [('dualbranch', 0.5), ('eegsym', 0.5), ('dualbranch', 3.0), ('eegsym', 3.0)]
LAB = {('dualbranch', 0.5): 'dual-branch W=0.5', ('eegsym', 0.5): 'EEGSym W=0.5',
       ('dualbranch', 3.0): 'dual-branch W=3.0', ('eegsym', 3.0): 'EEGSym W=3.0'}
B = 10000
lines = []
def emit(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); lines.append(s)


def load():
    files = sorted(glob.glob(str(OUTDIR / 'results_arch_seed*.csv')))
    agg = defaultdict(lambda: defaultdict(list))
    for f in files:
        for r in csvmod.DictReader(open(f)):
            key = (r['arch'], float(r['window']), int(r['subject']))
            for m in ('acc_zeroshot', 'acc_calib', 'lift'):
                agg[key][m].append(float(r[m]))
    data = {}
    for (arch, w, subj), md in agg.items():
        data.setdefault((arch, w), {})[subj] = {m: float(np.mean(v)) for m, v in md.items()}
    seeds = [Path(f).stem.split('seed')[-1] for f in files]
    emit(f"# seeds: {seeds} ({len(files)} files)")
    return data


def arr(data, arm, metric):
    d = data[arm]; return np.array([d[s][metric] for s in sorted(d)])


def boot_mean(x, seed=0):
    rng = np.random.RandomState(seed); n = len(x)
    bs = np.array([x[rng.randint(0, n, n)].mean() for _ in range(B)])
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def boot_paired(xa, xb, seed=7):
    d = xa - xb; rng = np.random.RandomState(seed); n = len(d)
    bs = np.array([d[rng.randint(0, n, n)].mean() for _ in range(B)])
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), float((bs > 0).mean()), d


def main():
    data = load()
    if not data:
        emit("no results yet"); return
    emit("\n## Level per arm (mean [95% CI], subject bootstrap, 9 subjects of 2a)")
    emit(f"{'arm':<22} {'zero-shot':<22} {'calibrated':<22} {'lift pts':<12}")
    for arm in ARMS:
        if arm not in data: continue
        zs, ca, lf = arr(data, arm, 'acc_zeroshot'), arr(data, arm, 'acc_calib'), arr(data, arm, 'lift')
        mz, lz, hz = boot_mean(zs, 1); mc, lc, hc = boot_mean(ca, 2); ml, ll, hl = boot_mean(lf, 3)
        emit(f"{LAB[arm]:<22} {mz:.3f} [{lz:.3f},{hz:.3f}]   {mc:.3f} [{lc:.3f},{hc:.3f}]   "
             f"{ml*100:+.1f} [{ll*100:+.1f},{hl*100:+.1f}]")

    def contrast(name, a, b, metric):
        if a not in data or b not in data: return
        m, lo, hi, pgt, d = boot_paired(arr(data, a, metric), arr(data, b, metric))
        sig = "real (CI excludes 0)" if (lo > 0 or hi < 0) else "inconclusive"
        emit(f"  {name:<40} diff={m*100:+.2f} pts  95% CI [{lo*100:+.2f},{hi*100:+.2f}]  "
             f"P(>0)={pgt:.2f}  {int((d>0).sum())} up/{int((d<0).sum())} down  -> {sig}")

    emit("\n## Architecture contrast (EEGSym - dual-branch), paired by subject")
    emit(" [W=3.0 s, EEGSym native crop length]")
    contrast("EEGSym-dualbranch W3 (zero-shot)", ('eegsym', 3.0), ('dualbranch', 3.0), 'acc_zeroshot')
    contrast("EEGSym-dualbranch W3 (calibrated)", ('eegsym', 3.0), ('dualbranch', 3.0), 'acc_calib')
    emit(" [W=0.5 s, low-latency crop length]")
    contrast("EEGSym-dualbranch W0.5 (zero-shot)", ('eegsym', 0.5), ('dualbranch', 0.5), 'acc_zeroshot')
    contrast("EEGSym-dualbranch W0.5 (calibrated)", ('eegsym', 0.5), ('dualbranch', 0.5), 'acc_calib')

    emit("\n## Crop-length contrast (W=3.0 - W=0.5), paired by subject")
    contrast("EEGSym: W3-W0.5 (zero-shot)", ('eegsym', 3.0), ('eegsym', 0.5), 'acc_zeroshot')
    contrast("dual-branch: W3-W0.5 (zero-shot)", ('dualbranch', 3.0), ('dualbranch', 0.5), 'acc_zeroshot')

    (HERE / 'outputs' / 'architectures.md').write_text('\n'.join(lines), encoding='utf-8')
    emit(f"# written -> {HERE/'outputs' / 'architectures.md'}")


if __name__ == '__main__':
    main()
