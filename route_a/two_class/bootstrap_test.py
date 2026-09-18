#!/usr/bin/env python
"""Subject-level bootstrap interval of the held-out test accuracy.

Reads the per-subject test accuracies written by `confirm_test.py` and resamples the
ten test subjects (B = 10000, seed 0). Two estimators are reported: the trial-weighted
mean (each subject weighted by its number of trials, the figure quoted in the paper)
and the plain mean over subjects.

Usage:
    python bootstrap_test.py
"""
import json

import numpy as np

from run_grid import NPZ, OUTDIR


def boot_ci(acc, n, B=10000, seed=0):
    rng = np.random.default_rng(seed); k = len(acc)
    sw = np.empty(B); ss = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, k, k)
        sw[b] = np.average(acc[idx], weights=n[idx])   # weighted by trials
        ss[b] = acc[idx].mean()                        # plain mean over subjects
    return sw, ss


def main():
    ct = json.load(open(OUTDIR / 'confirm_test.json', encoding='utf-8'))
    tps = ct['test_per_subject']
    subj = np.load(NPZ, allow_pickle=True)['subject'].astype(int)
    test_subs = sorted(int(k) for k in tps)
    acc_test = np.array([tps[str(s)] for s in test_subs], float)
    n_test = np.array([int((subj == s).sum()) for s in test_subs], float)
    print('test accuracy:', dict(zip([f'S{s}' for s in test_subs], acc_test)))
    print('test trials  :', dict(zip([f'S{s}' for s in test_subs], n_test.astype(int))))

    sw, ss = boot_ci(acc_test, n_test, B=10000)
    w_point = float(np.average(acc_test, weights=n_test)); s_point = float(acc_test.mean())
    wlo, whi = np.percentile(sw, [2.5, 97.5]); slo, shi = np.percentile(ss, [2.5, 97.5])
    print(f'\nweighted  : {w_point:.3f}  95% CI [{wlo:.3f}, {whi:.3f}]')
    print(f'unweighted: {s_point:.3f}  95% CI [{slo:.3f}, {shi:.3f}]')
    print(f'P(acc > 0.5) under the weighted bootstrap: {float((sw > 0.5).mean()):.4f}')


if __name__ == '__main__':
    main()
