"""BCI Competition IV Data Set 1: data loading, causal output geometry and the metric.

Reads the competition files from $BCI_DATA/bci_iv_1/ (written by download.py) and
provides:
  - loaders for the recordings (`cargar_mat`, `senal_uv`) and for the true evaluation
    labels (`cargar_true_y`, and `cargar_true_y_txt` as an independent parsing path);
  - `expandir_causal`, which turns one prediction per window into a per-sample output;
  - the official metric (`mse_oficial`), the MSE of the constant output 0 (`mse_cero`)
    and the quantization floor of a 100 Hz output (`mse_oraculo_100hz`);
  - constant-label segments (`segmentos_definidos`) and the block bootstrap
    (`boot_bloques`) used for within-subject confidence intervals.

Causality. The target is defined per sample, so the output at sample t may only use
data up to t. A window is assigned to its last sample, as in the winning entry ("the
window endpoint representing the current sample whose class label needed
prediction"), and each prediction is held until the next one.

Sampling rates. The distributed signal `cnt` is at 100 Hz (low-pass filtered at 49 Hz
before downsampling, according to the dataset description). The true labels `true_y`
are at 1000 Hz, exactly 10 times the length of `cnt`. A 100 Hz output is repeated 10
times and compared with the 1000 Hz labels; `mse_oraculo_100hz` measures the error
floor this imposes.

Evaluated samples. `true_y` is NaN in undefined mental states; those samples are
excluded from the MSE, not set to 0. The read_me.txt of the label archive states that,
because of a transmission error, only the first 1,759,140 samples of subject 'a' were
used in the evaluation; `cargar_true_y` applies that truncation by default because it
is part of the official metric.

Subjects. Of the seven subjects a..g, c, d and e are artificially generated according
to the dataset description; only a, b, f and g are used.

Usage:
    import data_io as A
    d = A.cargar_mat('a', 'calib')
    x = A.senal_uv(d)             # (59, n_samples) in uV
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RAW = Path(os.environ.get('BCI_DATA', HERE.parents[1] / 'data')) / 'bci_iv_1'
OUT = HERE / 'outputs'

# ============================================================
# Dataset constants
# ============================================================
SUBS = list('abcdefg')
REALES = ['a', 'b', 'f', 'g']            # c, d, e are artificial
ARTIFICIALES = ['c', 'd', 'e']

FS_IV1 = 100                              # nfo.fs of the distributed files
FS_LAB = 1000                             # sampling rate of the true labels
FS_TGT = 250                              # rate of the route A models; default fs_in
                                          # of `expandir_causal`
ESCALA_UV = 0.1                           # cnt is int16; uV = 0.1 * double(cnt)
# read_me.txt of true_labels.zip: because of a transmission error only the first
# 1,759,140 samples (at 1000 Hz) of subject 'a' were used in the evaluation.
TRUNCAR_LAB = {'a': 1759140}

# Published values (BCI Competition IV results for data set 1, and Zhang et al. 2012),
# used for comparison only: MSE of the constant output 0, and the range expected for
# each subject.
MSE_SILENCIO_OFICIAL = 0.509
RANGO_SILENCIO_OFICIAL = (0.49, 0.54)
SEED = 42
B_BOOT = 10000                            # bootstrap resamples


# ============================================================
# Dataset I/O
# ============================================================
def _txt(x) -> str:
    """A Matlab field that should be text, returned as a stripped str."""
    if isinstance(x, np.ndarray):
        x = x.item() if x.size == 1 else x
    return str(x).strip()


def cargar_mat(sujeto: str, tipo: str) -> dict:
    """Read BCICIV_{tipo}_ds1{sujeto}.mat and return its parsed contents as a dict.

    `struct_as_record=False, squeeze_me=True` are required: without them `nfo` is
    returned as a nested object array and `clab`/`classes` are not read correctly.
    The raw int16 signal is returned as `cnt_int16` ([time x channels]); `senal_uv`
    converts it to microvolts. `mrk_pos`/`mrk_y` (cue onsets and classes) are only
    present in the calibration files.
    """
    assert tipo in ('calib', 'eval'), tipo
    p = RAW / f'BCICIV_{tipo}_ds1{sujeto}.mat'
    m = loadmat(p, struct_as_record=False, squeeze_me=True)
    nfo = m['nfo']
    cnt = m['cnt']
    clab = [_txt(c) for c in np.atleast_1d(nfo.clab)]
    # `cnt` is [time x channels]; checked against clab rather than assumed.
    assert cnt.ndim == 2 and cnt.shape[1] == len(clab), \
        f'{p.name}: cnt {cnt.shape} is not [time x {len(clab)} channels]'
    assert cnt.shape[0] > cnt.shape[1], f'{p.name}: cnt looks transposed {cnt.shape}'
    d = dict(
        sujeto=sujeto, tipo=tipo, fichero=p.name,
        cnt_int16=cnt, n_muestras=int(cnt.shape[0]), n_canales=int(cnt.shape[1]),
        dtype=str(cnt.dtype), fs=int(nfo.fs), clab=clab,
        clab_up=[c.upper() for c in clab],
        classes=[_txt(c) for c in np.atleast_1d(nfo.classes)],
        xpos=np.asarray(nfo.xpos, np.float64), ypos=np.asarray(nfo.ypos, np.float64),
        tiene_mrk='mrk' in m)
    if d['tiene_mrk']:
        d['mrk_pos'] = np.asarray(m['mrk'].pos, np.int64)
        d['mrk_y'] = np.asarray(m['mrk'].y, np.int64)
    assert d['fs'] == FS_IV1, f'{p.name}: fs={d["fs"]} != {FS_IV1}'
    assert len(set(d['clab_up'])) == len(d['clab_up']), \
        f'{p.name}: duplicate channel names after upper-casing'
    return d


def senal_uv(d: dict) -> np.ndarray:
    """(n_channels, n_samples) in microvolts: uV = 0.1 * double(cnt).

    Omitting the 0.1 factor would raise no error and would only shift absolute
    log-powers by a constant, so the scaling is an explicit step.
    """
    return (ESCALA_UV * d['cnt_int16'].astype(np.float64)).T


def cargar_true_y(sujeto: str, truncar: bool = True) -> np.ndarray:
    """True labels at 1000 Hz. NaN marks an undefined state, excluded from the MSE.

    With `truncar` the truncation stated in the official read_me is applied (subject
    'a' only).
    """
    p = RAW / f'BCICIV_eval_ds1{sujeto}_1000Hz_true_y.mat'
    y = np.asarray(loadmat(p, struct_as_record=False, squeeze_me=True)['true_y'],
                   np.float64).ravel()
    if truncar and sujeto in TRUNCAR_LAB:
        y = y[:TRUNCAR_LAB[sujeto]]
    return y


def cargar_true_y_txt(sujeto: str) -> np.ndarray:
    """The same labels read from the TXT release (independent parsing path, not
    truncated)."""
    p = RAW / f'BCICIV_eval_ds1{sujeto}_1000Hz_true_y.txt'
    return np.loadtxt(p, dtype=np.float64)


# ============================================================
# Causal window geometry
# ============================================================
def expandir_causal(valores: np.ndarray, fin_idx: np.ndarray, n_out: int,
                    fs_in: int = FS_TGT, fs_out: int = FS_IV1,
                    relleno: float = 0.0) -> np.ndarray:
    """One prediction per window -> one value per sample, causal (zero-order hold).

    Output sample j (time j/fs_out) takes the value of the last window whose end time
    (fin_idx/fs_in) is <= j/fs_out; `fin_idx` must be in increasing order. Before the
    first complete window the output is `relleno` (default 0, no decision), and those
    samples are still scored by the MSE.
    """
    out = np.full(int(n_out), float(relleno), np.float64)
    if len(fin_idx) == 0:
        return out
    t_fin = np.asarray(fin_idx, np.float64) / float(fs_in)
    t_out = np.arange(n_out, dtype=np.float64) / float(fs_out)
    k = np.searchsorted(t_fin, t_out, side='right') - 1
    ok = k >= 0
    out[ok] = np.asarray(valores, np.float64)[k[ok]]
    return out


# ============================================================
# Official metric
# ============================================================
def mse_oficial(out_100: np.ndarray, true_y: np.ndarray) -> dict:
    """MSE of a 100 Hz output against the 1000 Hz labels.

    Each output sample is held for 10 label samples (sample j -> [10j, 10j+10) at
    1000 Hz), which is what an entry working on the 100 Hz data had to submit. Only
    samples with a finite label are scored.
    """
    o = np.repeat(np.asarray(out_100, np.float64), FS_LAB // FS_IV1)
    t = np.asarray(true_y, np.float64)
    n = min(len(o), len(t))
    o, t = o[:n], t[:n]
    ok = np.isfinite(t)
    assert ok.any(), 'no sample left to evaluate'
    e = o[ok] - t[ok]
    return dict(mse=float(np.mean(e ** 2)), n_evaluadas=int(ok.sum()),
                n_total=int(len(t)), n_indefinidas=int((~ok).sum()))


def mse_cero(true_y: np.ndarray) -> float:
    """MSE of the constant output 0, which equals the fraction of evaluated time in MI.

    This is an identity: with o == 0, (o - t)^2 == t^2, and for t in {-1, 0, +1} t^2 is
    the MI indicator.
    """
    t = np.asarray(true_y, np.float64)
    ok = np.isfinite(t)
    return float(np.mean(t[ok] ** 2))


def mse_oraculo_100hz(true_y: np.ndarray) -> float:
    """Quantization floor: the lowest MSE reachable by an output that is constant
    within each block of 10 label samples, i.e. by any system that decides at 100 Hz.

    The optimum in each block is the mean of its finite labels, which lies in [-1, 1]
    and is therefore a valid output. Labels after the last complete block are scored
    against 0.
    """
    t = np.asarray(true_y, np.float64)
    k = FS_LAB // FS_IV1
    nb = len(t) // k
    T = t[:nb * k].reshape(nb, k)
    ok = np.isfinite(T)
    cnt = ok.sum(1)
    sm = np.where(ok, T, 0.0).sum(1)
    mu = np.divide(sm, np.maximum(cnt, 1), where=cnt > 0)
    res = np.where(ok, T - mu[:, None], 0.0)
    resto = t[nb * k:]
    extra = resto[np.isfinite(resto)]
    num = float((res ** 2).sum() + (extra ** 2).sum() if len(extra) else (res ** 2).sum())
    den = float(cnt.sum() + len(extra))
    return num / den


def segmentos_definidos(y: np.ndarray) -> list[dict]:
    """Maximal runs of constant value among the defined (finite) samples.

    NaNs split runs and are never filled, so a task period is exactly what the label
    file marks as one. These runs are the resampling units of the block bootstrap.
    `ini`/`fin` are sample indices (`fin` exclusive); `dur_s` assumes the 1000 Hz
    label rate.
    """
    y = np.asarray(y, np.float64)
    ok = np.isfinite(y)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return []
    v = y[idx]
    corta = (np.diff(idx) != 1) | (np.diff(v) != 0)
    bordes = np.concatenate([[0], np.where(corta)[0] + 1, [len(idx)]])
    out = []
    for a, b in zip(bordes[:-1], bordes[1:]):
        out.append(dict(ini=int(idx[a]), fin=int(idx[b - 1]) + 1,
                        n=int(idx[b - 1] + 1 - idx[a]),
                        dur_s=float((idx[b - 1] + 1 - idx[a]) / FS_LAB),
                        valor=float(v[a])))
    return out


# ============================================================
# Block bootstrap (within subject)
# ============================================================
def boot_bloques(sse: np.ndarray, n: np.ndarray, B: int = B_BOOT, seed: int = SEED,
                 tam_bloque: int = 1) -> tuple[float, float, float]:
    """95% percentile interval of the MSE, resampling blocks rather than samples.

    `sse[i]` and `n[i]` are the sum of squared errors and the number of evaluated
    samples of block i (one task or rest period). Blocks are resampled with
    replacement and the statistic is the ratio of sums, which is the MSE. Adjacent
    samples are strongly correlated, so an i.i.d. bootstrap over samples would give
    far too narrow intervals. `tam_bloque > 1` merges consecutive blocks (sensitivity
    check). Returns (mse, lo, hi); lo and hi are NaN with fewer than 2 blocks.
    """
    sse = np.asarray(sse, np.float64); n = np.asarray(n, np.float64)
    if tam_bloque > 1:
        m = (len(sse) // tam_bloque) * tam_bloque
        resto_s, resto_n = sse[m:].sum(), n[m:].sum()
        sse = sse[:m].reshape(-1, tam_bloque).sum(1)
        n = n[:m].reshape(-1, tam_bloque).sum(1)
        if resto_n > 0:
            sse = np.append(sse, resto_s); n = np.append(n, resto_n)
    k = len(sse)
    punto = float(sse.sum() / max(n.sum(), 1e-12))
    if k < 2:
        return punto, float('nan'), float('nan')
    bs = np.random.RandomState(seed).randint(0, k, (B, k))
    m = sse[bs].sum(1) / np.maximum(n[bs].sum(1), 1e-12)
    return punto, float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
