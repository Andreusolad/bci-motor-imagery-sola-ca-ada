"""Statistics shared by the aptitude study scripts.

Conventions: subject-level bootstrap with B = 10000 resamples, 95% percentile CI, seed 42.
A CI that excludes 0 is labelled 'real (CI excludes 0)', otherwise 'inconclusive'; with
fewer than 5 subjects no verdict is given (same guard as analyze.py).

  corr_ci              correlation with a subject-level bootstrap CI.
  corr_dif_ci          difference of two correlations that share x, computed on the same
                       resamples, so the contrast is paired. Showing that x relates more to
                       y1 than to y2 requires this difference to exclude 0; one high and one
                       low correlation are not enough.
  perm_p               two-sided permutation p-value of a correlation.
  parcial              partial correlation, to rule out covariates (artefacts, number of
                       trials).
  boot_media, boot_dif_pareada, boot_dif_no_pareada, perm_p_grupos
                       means and differences of means.
  sb                   Spearman-Brown correction for split-half reliabilities;
                       techo_atenuacion = sqrt(rel_x * rel_y), the upper bound that
                       reliability places on any observed correlation.
  holm                 Holm-Bonferroni adjusted p-values.
  coef_bimodalidad     bimodality coefficient.
  fisher_exacto        two-sided Fisher exact test for a 2x2 table.

Usage:  python aptitude_common.py --test
"""
from __future__ import annotations
import argparse
import math
import sys

import numpy as np

B_BOOT = 10000
SEED = 42
MIN_SUBS_VEREDICTO = 5          # same guard as analyze.py


# ============================================================
# Correlations
# ============================================================
def rangos(v: np.ndarray) -> np.ndarray:
    """Ranks 1..n, with average ranks for ties."""
    v = np.asarray(v, float)
    o = np.argsort(v, kind='mergesort')
    r = np.empty(len(v), float)
    r[o] = np.arange(1, len(v) + 1, dtype=float)
    vs = v[o]
    i = 0
    while i < len(vs):
        j = i
        while j + 1 < len(vs) and vs[j + 1] == vs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float('nan')
    x, y = x[m], y[m]
    sx, sy = x.std(), y.std()
    if sx <= 0 or sy <= 0:
        return float('nan')
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float('nan')
    return pearson(rangos(x[m]), rangos(y[m]))


CORR = {'pearson': pearson, 'spearman': spearman}


# ============================================================
# Subject-level bootstrap
# ============================================================
def idx_boot(n: int, B: int = B_BOOT, seed: int = SEED) -> np.ndarray:
    return np.random.RandomState(seed).randint(0, n, (B, n))


def _ic(vals: np.ndarray) -> tuple[float, float, int]:
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        return float('nan'), float('nan'), 0
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)


def veredicto(lo: float, hi: float, n: int | None = None) -> str:
    if lo != lo:
        return 'no data'
    if n is not None and n < MIN_SUBS_VEREDICTO:
        return f'too few subjects (n={n})'
    return 'real (CI excludes 0)' if (lo > 0 or hi < 0) else 'inconclusive'


def veredicto_corr(lo: float, hi: float, n: int | None = None) -> str:
    """For a correlation, "crossing 0" means that 0 lies inside the CI."""
    return veredicto(lo, hi, n)


def corr_ci(x, y, metodo: str = 'spearman', B: int = B_BOOT,
            seed: int = SEED) -> dict:
    fn = CORR[metodo]
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        return {'r': float('nan'), 'lo': float('nan'), 'hi': float('nan'),
                'n': n, 'metodo': metodo, 'veredicto': 'no data'}
    bi = idx_boot(n, B, seed)
    vals = np.array([fn(x[i], y[i]) for i in bi])
    lo, hi, nb = _ic(vals)
    return {'r': fn(x, y), 'lo': lo, 'hi': hi, 'n': n, 'metodo': metodo,
            'n_boot_validos': nb, 'veredicto': veredicto_corr(lo, hi, n)}


def corr_dif_ci(x, y1, y2, metodo: str = 'spearman', B: int = B_BOOT,
                seed: int = SEED) -> dict:
    """corr(x, y1) - corr(x, y2) on the same subject resamples.

    Only subjects with all three values finite are used, so that both correlations are
    computed on exactly the same sample and the difference is a paired contrast.
    """
    fn = CORR[metodo]
    x, y1, y2 = (np.asarray(v, float) for v in (x, y1, y2))
    m = np.isfinite(x) & np.isfinite(y1) & np.isfinite(y2)
    x, y1, y2 = x[m], y1[m], y2[m]
    n = len(x)
    if n < 3:
        return {'dif': float('nan'), 'lo': float('nan'), 'hi': float('nan'),
                'n': n, 'veredicto': 'no data'}
    bi = idx_boot(n, B, seed)
    d = np.array([fn(x[i], y1[i]) - fn(x[i], y2[i]) for i in bi])
    lo, hi, nb = _ic(d)
    return {'r1': fn(x, y1), 'r2': fn(x, y2), 'dif': fn(x, y1) - fn(x, y2),
            'lo': lo, 'hi': hi, 'n': n, 'n_boot_validos': nb, 'metodo': metodo,
            'r_entre_y1_y2': fn(y1, y2),
            'P>0': float((d[np.isfinite(d)] > 0).mean()),
            'veredicto': veredicto(lo, hi, n)}


def perm_p(x, y, metodo: str = 'spearman', B: int = B_BOOT,
           seed: int = SEED) -> float:
    """Two-sided permutation p-value: P(|r_permuted| >= |r_observed|)."""
    fn = CORR[metodo]
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float('nan')
    r0 = abs(fn(x, y))
    rng = np.random.RandomState(seed)
    cnt = 0
    for _ in range(B):
        if abs(fn(x, rng.permutation(y))) >= r0 - 1e-15:
            cnt += 1
    return float((cnt + 1) / (B + 1))


def parcial(x, y, z, metodo: str = 'pearson') -> float:
    """Partial correlation of x and y controlling for z (residuals of a linear fit on z).

    If z explains x (or y) almost exactly, the residual is floating-point noise and its
    correlation is an arbitrary number. NaN is returned when a residual std is <= 1e-6
    times the std of the original variable.
    """
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    if len(x) < 4:
        return float('nan')
    A = np.column_stack([np.ones(len(z)), z])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    for res, ori in ((rx, x), (ry, y)):
        if res.std() <= 1e-6 * max(ori.std(), 1e-300):
            return float('nan')
    return CORR[metodo](rx, ry)


# ============================================================
# Differences of means
# ============================================================
def boot_media(v, B: int = B_BOOT, seed: int = SEED) -> dict:
    v = np.asarray([x for x in v if x == x], float)
    if len(v) == 0:
        return {'mean': float('nan'), 'lo': float('nan'), 'hi': float('nan'), 'n': 0}
    bs = v[idx_boot(len(v), B, seed)].mean(1)
    return {'mean': float(v.mean()), 'lo': float(np.percentile(bs, 2.5)),
            'hi': float(np.percentile(bs, 97.5)), 'n': len(v)}


def boot_dif_no_pareada(a, b, B: int = B_BOOT, seed: int = SEED) -> dict:
    """mean(a) - mean(b), resampling each group independently."""
    a = np.asarray([x for x in a if x == x], float)
    b = np.asarray([x for x in b if x == x], float)
    if len(a) == 0 or len(b) == 0:
        return {'dif': float('nan'), 'lo': float('nan'), 'hi': float('nan'),
                'na': len(a), 'nb': len(b), 'veredicto': 'no data'}
    rng = np.random.RandomState(seed)
    da = a[rng.randint(0, len(a), (B, len(a)))].mean(1)
    db = b[rng.randint(0, len(b), (B, len(b)))].mean(1)
    d = da - db
    return {'dif': float(a.mean() - b.mean()), 'media_a': float(a.mean()),
            'media_b': float(b.mean()),
            'lo': float(np.percentile(d, 2.5)), 'hi': float(np.percentile(d, 97.5)),
            'na': len(a), 'nb': len(b), 'P>0': float((d > 0).mean()),
            'veredicto': veredicto(float(np.percentile(d, 2.5)),
                                   float(np.percentile(d, 97.5)), min(len(a), len(b)))}


def boot_dif_pareada(a, b, B: int = B_BOOT, seed: int = SEED) -> dict:
    """mean(a - b) over the same subjects (a[i] and b[i] belong to the same subject)."""
    d = np.array([x - y for x, y in zip(a, b) if x == x and y == y], float)
    if len(d) == 0:
        return {'dif': float('nan'), 'lo': float('nan'), 'hi': float('nan'),
                'n': 0, 'veredicto': 'no data'}
    bs = d[idx_boot(len(d), B, seed)].mean(1)
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return {'dif': float(d.mean()), 'lo': lo, 'hi': hi, 'n': len(d),
            'up': int((d > 0).sum()), 'dn': int((d < 0).sum()),
            'P>0': float((bs > 0).mean()), 'veredicto': veredicto(lo, hi, len(d))}


def perm_p_grupos(a, b, B: int = B_BOOT, seed: int = SEED) -> float:
    """Two-sided permutation p-value for mean(a) - mean(b) (group labels shuffled)."""
    a = np.asarray([x for x in a if x == x], float)
    b = np.asarray([x for x in b if x == x], float)
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    todo = np.concatenate([a, b]); na = len(a)
    d0 = abs(a.mean() - b.mean())
    rng = np.random.RandomState(seed)
    cnt = 0
    for _ in range(B):
        p = rng.permutation(todo)
        if abs(p[:na].mean() - p[na:].mean()) >= d0 - 1e-15:
            cnt += 1
    return float((cnt + 1) / (B + 1))


# ============================================================
# Reliability
# ============================================================
def sb(r: float) -> float:
    """Spearman-Brown: reliability of the full test from the split-half correlation."""
    if r != r or (1 + r) == 0:
        return float('nan')
    return float(2 * r / (1 + r))


def techo_atenuacion(rel_x: float, rel_y: float) -> float:
    """Upper bound of |corr(x, y)| given the reliability of each variable."""
    if rel_x != rel_x or rel_y != rel_y:
        return float('nan')
    return float(math.sqrt(max(rel_x, 0) * max(rel_y, 0)))


def desatenuar(r: float, rel_x: float, rel_y: float) -> float:
    t = techo_atenuacion(rel_x, rel_y)
    return float(r / t) if t and t == t and t > 0 else float('nan')


# ============================================================
# Multiple comparisons
# ============================================================
def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values over a family of tests.

    NaN p-values are left out of the family and returned as NaN. Holm does not assume
    independence; when several metrics in the family are nearly the same measure,
    treating them as k separate tests is conservative.
    """
    items = [(k, v) for k, v in pvals.items() if v == v]
    items.sort(key=lambda kv: kv[1])
    k = len(items)
    out, prev = {}, 0.0
    for i, (nm, p) in enumerate(items):
        adj = min(1.0, (k - i) * p)
        adj = max(adj, prev)          # adjusted p-values must be monotone
        prev = adj
        out[nm] = float(adj)
    for nm, v in pvals.items():
        out.setdefault(nm, float('nan'))
    return out


# ============================================================
# Bimodality
# ============================================================
def coef_bimodalidad(v) -> float:
    """SAS bimodality coefficient: BC = (g1^2 + 1) / (g2 + 3(n-1)^2/((n-2)(n-3))).

    Reference: BC of a uniform distribution = 5/9 = 0.5556; higher values suggest
    bimodality, lower values unimodality. It is a descriptor, not a hypothesis test.
    """
    v = np.asarray([x for x in v if x == x], float)
    n = len(v)
    if n < 4:
        return float('nan')
    sd = v.std(ddof=1)
    if sd <= 0:
        return float('nan')
    z = (v - v.mean()) / sd
    g1 = n * np.sum(z ** 3) / ((n - 1) * (n - 2))                     # sample skewness
    g2 = ((n * (n + 1) * np.sum(z ** 4)) / ((n - 1) * (n - 2) * (n - 3))
          - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))                   # sample excess kurtosis
    return float((g1 ** 2 + 1) / (g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


BC_UNIFORME = 5.0 / 9.0


# ============================================================
# Fisher exact test (2x2)
# ============================================================
def fisher_exacto(a: int, b: int, c: int, d: int) -> float:
    """Two-sided p-value of the table [[a, b], [c, d]].

    Sums the probabilities of all tables with the same margins that are no more likely
    than the observed one.
    """
    n = a + b + c + d
    f1, f2 = a + b, c + d
    c1 = a + c

    def pr(x):
        return (math.comb(f1, x) * math.comb(f2, c1 - x) / math.comb(n, c1))
    lo = max(0, c1 - f2); hi = min(f1, c1)
    p0 = pr(a)
    return float(min(1.0, sum(pr(x) for x in range(lo, hi + 1)
                              if pr(x) <= p0 * (1 + 1e-12))))


# ============================================================
# Self-tests
# ============================================================
def autotests() -> dict:
    r = []

    def chk(nombre, cond, detalle=''):
        r.append({'test': nombre, 'ok': bool(cond), 'detalle': str(detalle)})
        print(f'  [{"OK" if cond else "FAIL"}] {nombre}  {detalle}')

    rng = np.random.RandomState(0)
    x = rng.randn(60); y = 0.6 * x + rng.randn(60) * 0.8
    try:
        from scipy import stats as st
        chk('pearson == scipy', abs(pearson(x, y) - st.pearsonr(x, y)[0]) < 1e-12,
            f'{pearson(x, y):.12f}')
        chk('spearman == scipy', abs(spearman(x, y) - st.spearmanr(x, y)[0]) < 1e-12,
            f'{spearman(x, y):.12f}')
        xe = np.round(x, 1)
        chk('spearman with ties == scipy',
            abs(spearman(xe, y) - st.spearmanr(xe, y)[0]) < 1e-12, f'{spearman(xe, y):.12f}')
        chk('fisher exact == scipy', abs(fisher_exacto(3, 9, 30, 20) -
                                         st.fisher_exact([[3, 9], [30, 20]])[1]) < 1e-9,
            f'{fisher_exacto(3, 9, 30, 20):.9f}')
    except ImportError:
        chk('scipy available', False, 'not installed')

    # null correlation: the CI must contain 0
    z = rng.randn(60)
    c0 = corr_ci(x, z, 'spearman', B=2000)
    chk('null correlation -> CI crosses 0', c0['lo'] < 0 < c0['hi'],
        f'r={c0["r"]:+.3f} CI[{c0["lo"]:+.3f},{c0["hi"]:+.3f}]')
    c1 = corr_ci(x, y, 'spearman', B=2000)
    chk('true correlation -> CI excludes 0', c1['lo'] > 0,
        f'r={c1["r"]:+.3f} CI[{c1["lo"]:+.3f},{c1["hi"]:+.3f}]')

    # difference of correlations: y correlates with x, z does not -> dif > 0
    dd = corr_dif_ci(x, y, z, 'spearman', B=2000)
    chk('difference of correlations detects the contrast', dd['lo'] > 0,
        f'dif={dd["dif"]:+.3f} CI[{dd["lo"]:+.3f},{dd["hi"]:+.3f}]')
    # a correlation minus itself must be exactly 0
    dd0 = corr_dif_ci(x, y, y, 'spearman', B=500)
    chk('difference of a correlation with itself is exactly 0',
        dd0['dif'] == 0 and dd0['lo'] == 0 and dd0['hi'] == 0, f'{dd0["dif"]}')

    # permutation p-values
    p_nula = perm_p(x, z, 'spearman', B=2000)
    p_real = perm_p(x, y, 'spearman', B=2000)
    chk('permutation p: high under the null, low for a true correlation',
        p_nula > 0.05 and p_real < 0.01, f'null={p_nula:.4f} true={p_real:.4f}')

    # partial correlation: yy = x + w; controlling for x, the yy-w relation remains
    w = rng.randn(60); yy = x + w
    chk('parcial(y,w|x) high', parcial(yy, w, x) > 0.8, f'{parcial(yy, w, x):.4f}')
    chk('parcial(y,x|x) = NaN (perfect collinearity)',
        parcial(yy, x, x) != parcial(yy, x, x), f'{parcial(yy, x, x)}')

    # Spearman-Brown and attenuation ceiling
    chk('Spearman-Brown(0.5) = 0.667', abs(sb(0.5) - 2 / 3) < 1e-12, f'{sb(0.5):.6f}')
    chk('techo_atenuacion(0.81,0.64) = 0.72',
        abs(techo_atenuacion(0.81, 0.64) - 0.72) < 1e-12,
        f'{techo_atenuacion(0.81, 0.64):.6f}')

    # differences of means
    a = rng.randn(30) + 1.0; b = rng.randn(30)
    dnp = boot_dif_no_pareada(a, b, B=2000)
    chk('unpaired difference detects +1', dnp['lo'] > 0, f'{dnp["dif"]:+.3f}')
    dp = boot_dif_pareada(list(a), list(a), B=500)
    chk('paired difference of a with itself = 0', dp['dif'] == 0 and dp['lo'] == 0)

    # Holm
    h = holm({'a': 0.01, 'b': 0.04, 'c': 0.20})
    chk('Holm: smallest p multiplied by k', abs(h['a'] - 0.03) < 1e-12, f'{h["a"]:.4f}')
    chk('Holm: second smallest multiplied by k-1', abs(h['b'] - 0.08) < 1e-12,
        f'{h["b"]:.4f}')
    chk('Holm is monotone and <= 1',
        h['a'] <= h['b'] <= h['c'] and h['c'] <= 1.0, f'{h}')
    chk('Holm with a single test leaves p unchanged',
        abs(holm({'x': 0.03})['x'] - 0.03) < 1e-12)

    # bimodality coefficient
    uni = np.random.RandomState(1).rand(4000)
    bim = np.r_[np.random.RandomState(2).randn(2000) - 3,
                np.random.RandomState(3).randn(2000) + 3]
    nor = np.random.RandomState(4).randn(4000)
    chk('BC of a uniform sample ~ 5/9', abs(coef_bimodalidad(uni) - 5 / 9) < 0.03,
        f'{coef_bimodalidad(uni):.4f}')
    chk('BC of a clearly bimodal sample > 0.555', coef_bimodalidad(bim) > 0.555,
        f'{coef_bimodalidad(bim):.4f}')
    chk('BC of a normal sample < 0.555', coef_bimodalidad(nor) < 0.555,
        f'{coef_bimodalidad(nor):.4f}')

    # verdict
    chk('no verdict with n<5', 'too few' in veredicto(0.1, 0.2, 4))
    chk('verdict REAL if the CI excludes 0', veredicto(0.1, 0.2, 10).startswith('real'))
    chk('verdict "inconclusive" if the CI crosses 0',
        veredicto(-0.1, 0.2, 10) == 'inconclusive')

    ok = all(x['ok'] for x in r)
    print(f'\n{sum(x["ok"] for x in r)}/{len(r)} self-tests OK')
    return {'todos_ok': ok, 'tests': r}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true')
    a = ap.parse_args()
    if a.test:
        res = autotests()
        sys.exit(0 if res['todos_ok'] else 1)
    print(__doc__)
