"""Building blocks of the classical system for BCI Competition IV Data Set 1.

The system follows the recipe of the winning entry (Zhang, Guan, Ang, Wang, Chin,
Front. Neurosci. 6:7, 2012), as described in that paper:
  - the 100 Hz version of the data, all 59 channels;
  - a bank of 8 zero-phase Chebyshev type II filters, order 4, centered at
    8, 9.75, 11.89, 14.49, 17.67, 21.53, 26.25 and 32 Hz, with Q = 0.33;
  - FBCSP: for each band and each class pair (MI1-MI2, MI1-NC, MI2-NC) the filters of
    the 2 largest and the 2 smallest eigenvalues;
  - the non-control (NC, rest) class split into sub-states by PCA + k-means, with a
    number of clusters per subject;
  - feature selection by mutual information (estimated there with a Gaussian KDE);
  - GRNN regression (Nadaraya-Watson with a Gaussian kernel) on features scaled to
    [-1, 1], with spread equal to the SD of the features;
  - one output every 0.1 s from a 2.5 s window that ends at the predicted sample;
  - post-processing by a second GRNN on a 4 s buffer of predictions;
  - the final output clipped to [-1, 1].
The winner's subjects a, b, c, d are subjects a, b, f, g of this dataset.

This module provides the filter banks, sliding power, CSP, the calibration label
stream, NC sub-states, the regression and posterior heads, unsupervised feature
alignment, mutual-information ranking, the temporal post-processors and JSON helpers.
pipeline.py assembles them. Configurations were selected on calibration data only; the
selected one is stored in config/dev_best.json. Every parameter that affects the
result is explicit and is serialized with the results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import cheby2, butter, sosfilt, sosfiltfilt
from scipy.linalg import eigh

import data_io as A

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
CACHE = HERE / 'cache'
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

FS = A.FS_IV1                 # 100 Hz, used natively as in the winning entry
SEED = 42

# --- calibration geometry, from the official dataset description -----------------
# "Cues were displayed for a period of 4s"; the break is "2s of blank screen and 2s
# with a fixation cross". "Transient periods (1s after each cue) were excluded from
# averaging."
CAL_MI_S = 4.0
CAL_BREAK_S = 4.0
TRANSITORIO_S = 1.0

# --- the winner's filter bank -----------------------------------------------------
FCS_GANADOR = (8.0, 9.75, 11.89, 14.49, 17.67, 21.53, 26.25, 32.0)
Q_GANADOR = 0.33
ORDEN_GANADOR = 4
RS_CHEBY2 = 30.0              # stopband attenuation (dB); not given in the paper

# --- classical FBCSP bank --------------------------------------------------------
BANDAS_FBCSP = tuple((lo, lo + 4.0) for lo in range(4, 40, 4))    # 4-8 ... 36-40

# --- extra bands defined by their edges ------------------------------------------
# The winner's bank covers 6.68-37.3 Hz (8 * (1 - Q/2) to 32 * (1 + Q/2)) and has no
# feature below 6.68 Hz. These bands use Butterworth filters defined by their edges
# instead of center + Q: at 0.5-4 Hz the equivalent Q would be 3.5/2.25 = 1.56, far
# from the Q = 0.33 the winner's bank was designed for, and an order-4 Chebyshev II
# designed there has a -3 dB band of about 0.7-2.9 Hz instead of 0.5-4.0 Hz.
BANDAS_EXTRA = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'mu': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'alta': (30.0, 45.0),
    'bb': (0.5, 40.0),
}
ORDEN_BUTTER = 4

# Composite banks: the winner's 8 filters plus the listed extra bands. A comparison
# against 'ganador' is then strictly additive (same bank plus a band), so a difference
# can be attributed to the added band.
BANCOS_COMPUESTOS = {
    'ganador_delta': ('delta',),
    'ganador_theta': ('theta',),
    'ganador_bajo': ('delta', 'theta'),
    'ganador_bb': ('bb',),
    'ganador_alta': ('alta',),
    'ganador_bajo_alta': ('delta', 'theta', 'alta'),
}


# ============================================================
# Filter banks
# ============================================================
def banco_ganador(q: float = Q_GANADOR, orden: int = ORDEN_GANADOR,
                  fcs=FCS_GANADOR, rs: float = RS_CHEBY2) -> list[np.ndarray]:
    """The winner's 8 Chebyshev II band-pass filters (SOS), with edges fc * (1 -+ q/2),
    i.e. a width of q * fc. `cheby2` treats these edges as the frequencies where the
    attenuation first reaches `rs`, so the -3 dB band is narrower."""
    sos = []
    for fc in fcs:
        lo, hi = fc * (1.0 - q / 2.0), fc * (1.0 + q / 2.0)
        assert 0 < lo < hi < FS / 2, (fc, lo, hi)
        sos.append(cheby2(orden, rs, [lo, hi], btype='bandpass', fs=FS, output='sos'))
    return sos


def banco_fbcsp(bandas=BANDAS_FBCSP, orden: int = 4) -> list[np.ndarray]:
    """Classical FBCSP bank: Butterworth filters on 4 Hz bands."""
    return [butter(orden, [lo, hi], btype='bandpass', fs=FS, output='sos')
            for lo, hi in bandas]


def banco_butter(bandas, orden: int = ORDEN_BUTTER) -> list[np.ndarray]:
    """Butterworth band-pass filters with explicit edges. In SOS form they are stable at
    100 Hz even with a 0.5 Hz lower edge (largest pole modulus about 0.99)."""
    return [butter(orden, [lo, hi], btype='bandpass', fs=FS, output='sos')
            for lo, hi in bandas]


def construir_banco(nombre: str) -> tuple[list[np.ndarray], list[str]]:
    """Filters (SOS) and band names of the bank called `nombre`."""
    if nombre == 'ganador':
        return banco_ganador(), [f'{fc:g}Hz' for fc in FCS_GANADOR]
    if nombre == 'fbcsp':
        return banco_fbcsp(), [f'{lo:g}-{hi:g}' for lo, hi in BANDAS_FBCSP]
    if nombre == 'mb':                       # control: a single 8-30 Hz band
        return ([butter(4, [8.0, 30.0], btype='bandpass', fs=FS, output='sos')],
                ['8-30'])
    if nombre in BANDAS_EXTRA:               # a single extra band
        lo, hi = BANDAS_EXTRA[nombre]
        return banco_butter([(lo, hi)]), [f'{lo:g}-{hi:g}']
    if nombre in BANCOS_COMPUESTOS:          # the winner's 8 filters + extra bands
        claves = BANCOS_COMPUESTOS[nombre]
        sos = banco_ganador() + banco_butter([BANDAS_EXTRA[k] for k in claves])
        nom = ([f'{fc:g}Hz' for fc in FCS_GANADOR]
               + [f'{BANDAS_EXTRA[k][0]:g}-{BANDAS_EXTRA[k][1]:g}' for k in claves])
        return sos, nom
    raise ValueError(nombre)


def filtrar(x: np.ndarray, sos: np.ndarray, causal: bool) -> np.ndarray:
    """(n_ch, T) -> (n_ch, T). causal=True uses sosfilt (usable online);
    causal=False uses sosfiltfilt (zero-phase), as in the winning entry."""
    f = sosfilt if causal else sosfiltfilt
    return f(sos, np.asarray(x, np.float64), axis=-1)


# ============================================================
# Sliding power over a causal window (cumulative sums, O(T))
# ============================================================
def potencia_deslizante(z: np.ndarray, w: int, fin: np.ndarray) -> np.ndarray:
    """(n_sig, T) -> (n_win, n_sig): mean of z^2 over the window of length w that ends
    (inclusive) at each index in `fin`.

    The window [fin-w+1, fin] only uses the past. Cumulative sums make the cost
    independent of w.
    """
    z = np.asarray(z, np.float64)
    c = np.concatenate([np.zeros((z.shape[0], 1)), np.cumsum(z ** 2, axis=1)], axis=1)
    fin = np.asarray(fin, np.int64)
    assert fin.min() >= w - 1 and fin.max() < z.shape[1], 'window out of range'
    p = (c[:, fin + 1] - c[:, fin + 1 - w]) / float(w)
    return p.T


# ============================================================
# CSP
# ============================================================
def regularizar(C: np.ndarray, lam: float) -> np.ndarray:
    """Shrinkage toward the scaled identity. lam=0 leaves C unchanged."""
    if lam <= 0:
        return C
    n = C.shape[0]
    return (1.0 - lam) * C + lam * (np.trace(C) / n) * np.eye(n)


def csp(C1: np.ndarray, C2: np.ndarray, m: int = 2, lam: float = 0.05) -> np.ndarray:
    """CSP filters: the eigenvectors of the m smallest and the m largest eigenvalues of
    the generalized problem C1 w = lambda (C1 + C2) w. Returns (2m, n_ch)."""
    C1, C2 = regularizar(C1, lam), regularizar(C2, lam)
    vals, vecs = eigh(C1, C1 + C2)
    orden = np.argsort(vals)
    sel = np.concatenate([orden[:m], orden[-m:]])
    return vecs[:, sel].T.copy()


# ============================================================
# Calibration label stream and signals
# ============================================================
def stream_calibracion(s: str) -> dict:
    """Per-sample labels (100 Hz) of the calibration recording: +-1 during the 4 s of
    each cue, 0 elsewhere, NaN during the 1 s transient after the start and after the
    end of each cue.

    This turns the cued calibration recording into a continuous stream scored like the
    evaluation. It is the development set on which configurations are selected.
    """
    d = A.cargar_mat(s, 'calib')
    T = d['n_muestras']
    y = np.zeros(T, np.float64)
    nan = np.zeros(T, bool)
    n_mi = int(round(CAL_MI_S * FS))
    n_tr = int(round(TRANSITORIO_S * FS))
    for p, yy in zip(d['mrk_pos'], d['mrk_y']):
        p = int(p)
        y[p:p + n_mi] = float(yy)
        nan[p:min(p + n_tr, T)] = True
        nan[p + n_mi:min(p + n_mi + n_tr, T)] = True
    y[nan] = np.nan
    return dict(y=y, T=T, mrk_pos=np.asarray(d['mrk_pos'], np.int64),
                mrk_y=np.asarray(d['mrk_y'], np.int64), clases=d['classes'],
                n_mi_samp=n_mi, n_transitorio_samp=n_tr)


def senal_calibracion(s: str, car: bool) -> np.ndarray:
    """(59, T) in uV at 100 Hz, not resampled. car=True subtracts the common average."""
    d = A.cargar_mat(s, 'calib')
    x = A.senal_uv(d)
    return x - x.mean(axis=0, keepdims=True) if car else x


def senal_evaluacion(s: str, car: bool) -> np.ndarray:
    """Same as `senal_calibracion` for the evaluation recording."""
    d = A.cargar_mat(s, 'eval')
    x = A.senal_uv(d)
    return x - x.mean(axis=0, keepdims=True) if car else x


# ============================================================
# NC sub-states (PCA + k-means), as in the winning entry
# ============================================================
def subestados_nc(F_nc: np.ndarray, nk: int, seed: int = SEED) -> np.ndarray:
    """Cluster label of each NC window. nk <= 1 puts all windows in one state."""
    if nk <= 1:
        return np.zeros(len(F_nc), np.int64)
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    n_comp = int(min(10, F_nc.shape[1], max(1, len(F_nc) - 1)))
    Z = PCA(n_components=n_comp, random_state=seed).fit_transform(F_nc)
    return KMeans(n_clusters=nk, n_init=10, random_state=seed).fit_predict(Z)


# ============================================================
# Regressors
# ============================================================
class GRNN:
    """Generalized regression neural network, i.e. Nadaraya-Watson regression with a
    Gaussian kernel, the regressor of the winning entry. Its only parameter is the
    spread; by default it is the SD of all training feature values."""

    def __init__(self, spread: float | None = None, max_pts: int = 6000,
                 seed: int = SEED):
        self.spread, self.max_pts, self.seed = spread, max_pts, seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, np.float64)
        y = np.asarray(y, np.float64)
        if len(X) > self.max_pts:                     # stratified subsample
            rng = np.random.default_rng(self.seed)
            keep = []
            for v in np.unique(y):
                i = np.where(y == v)[0]
                k = max(1, int(round(self.max_pts * len(i) / len(y))))
                keep.append(rng.choice(i, size=min(k, len(i)), replace=False))
            sel = np.sort(np.concatenate(keep))
            X, y = X[sel], y[sel]
        self.X_, self.y_ = X, y
        self.s_ = float(self.spread) if self.spread else float(np.std(X))
        self.s_ = max(self.s_, 1e-6)
        self.xx_ = (X ** 2).sum(1)
        return self

    def predict(self, X: np.ndarray, chunk: int = 4096) -> np.ndarray:
        X = np.asarray(X, np.float64)
        out = np.empty(len(X), np.float64)
        d = 2.0 * self.s_ ** 2
        for a in range(0, len(X), chunk):
            Q = X[a:a + chunk]
            D2 = (Q ** 2).sum(1)[:, None] + self.xx_[None, :] - 2.0 * (Q @ self.X_.T)
            np.maximum(D2, 0.0, out=D2)
            D2 -= D2.min(axis=1, keepdims=True)       # numerical stability
            K = np.exp(-D2 / d)
            out[a:a + chunk] = (K @ self.y_) / np.maximum(K.sum(1), 1e-300)
        return out


class Ridge:
    """Ridge regression with an unpenalized intercept, closed form."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, np.float64)
        y = np.asarray(y, np.float64)
        self.mu_, self.ym_ = X.mean(0), y.mean()
        Xc, yc = X - self.mu_, y - self.ym_
        G = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        self.w_ = np.linalg.solve(G, Xc.T @ yc)
        return self

    def predict(self, X):
        return (np.asarray(X, np.float64) - self.mu_) @ self.w_ + self.ym_


# ============================================================
# Cost-sensitive heads
# ============================================================
# The head always has exactly three classes {-1, 0, +1}: the NC sub-states are only
# used to build CSP filter pairs, and `Posterior3` predicts 3 classes. A cross-entropy
# over 3 nominal classes ignores that confusing -1 with +1 costs four times as much as
# confusing -1 with 0.
#
# Cost matrix of the official metric: C[true, pred] = (c_pred - c_true)^2 with
# c = (-1, 0, +1). The standard reduction of a cost matrix to per-sample weights (Abe,
# Zadrozny and Langford) gives a sample of true class k the sum of row k: (5, 2, 5).
# Motor-imagery samples therefore weigh 5/2 = 2.5 times rest samples; this value
# follows from the metric and is not tuned.
def pesos_por_coste(y: np.ndarray, costo_signo: float) -> np.ndarray | None:
    """Per-sample multiplicative factor that makes a cross-entropy head cost-sensitive.
    `costo_signo` is the weight of a motor-imagery sample (+-1) relative to a rest
    sample (0).

    Returns None when `costo_signo == 1.0`, so that the code path is exactly the
    unweighted one (an array of ones might not give bit-identical results).

    No normalization here: `normalizar_al_total` does it at the single place where the
    weights are combined, because it must be relative to the total of the reference
    set, not to the mean of each factor.
    """
    if float(costo_signo) == 1.0:
        return None
    y = np.asarray(y, np.float64)
    return np.where(y != 0.0, float(costo_signo), 1.0)


def normalizar_al_total(w: np.ndarray, total: float) -> np.ndarray:
    """Rescale the weights so that they sum exactly to `total`.

    The objective of sklearn (and of `OrdinalLogit3` and `SoftmaxMSE3`) is

        0.5*||w||^2 + C * sum_i s_i * loss_i

    so the sum of the weights multiplies the likelihood against the penalty. If it
    changed with `costo_signo` or with the training mode, a comparison would also
    measure a change of effective regularization. With a fixed total, `C` means the
    same in every configuration, and the 'mixto' and 'blando' training modes are
    matched exactly, which makes 'mixto' a capacity control for 'blando'.
    """
    w = np.asarray(w, np.float64)
    return w * (float(total) / float(np.sum(w)))


class OrdinalLogit3:
    """Proportional-odds model (McCullagh 1980) over the ordered classes -1 < 0 < +1:

        P(y <= c_k | x) = sigmoid(theta_k - x.w)      k = 0, 1

    A single `w` serves both thresholds. This is the ordinal assumption: one latent
    axis on which rest lies between the two motor-imagery classes, so -1 cannot be
    confused with +1 without passing through 0. `Posterior3` treats rest as a state
    that is not intermediate between the two classes, so this model encodes an
    assumption that may not hold. The competition's runner-up used ordinal regression.

    Objective, with the same convention as `sklearn.LogisticRegression` so that `C`
    means the same in both heads:

        0.5 * ||w||^2 + C * sum_i s_i * (-log p_i(y_i))

    The intercepts (the two thresholds) are not penalized, as in sklearn. The order
    theta_0 <= theta_1 holds by construction with theta_1 = theta_0 + exp(d). The
    gradient is analytic.
    """

    classes_esperadas = (-1.0, 0.0, 1.0)

    # tol = 1e-12: with looser tolerances, the fits in the full feature space and in the
    # reduced row space (`pipeline._reducir_exacto`) differ by optimizer slack, although
    # the reduction itself is exact.
    def __init__(self, C: float = 1.0, tol: float = 1e-12, max_iter: int = 500):
        self.C, self.tol, self.max_iter = C, tol, max_iter

    # --- loss and gradient ----------------------------------------------------------
    def _perdida_grad(self, par, X, iy, s):
        from scipy.special import expit
        p_dim = X.shape[1]
        w, t0, d = par[:p_dim], par[p_dim], par[p_dim + 1]
        ed = np.exp(d)
        z = X @ w
        q0 = expit(t0 - z)
        q1 = expit(t0 + ed - z)
        g0, g1 = q0 * (1.0 - q0), q1 * (1.0 - q1)
        # probability of the observed class, per sample
        pr = np.where(iy == 0, q0, np.where(iy == 1, q1 - q0, 1.0 - q1))
        pr = np.maximum(pr, 1e-12)
        perdida = 0.5 * float(w @ w) + self.C * float(np.sum(s * -np.log(pr)))
        # d(nll)/dz and d(nll)/dtheta_k, per sample
        dz = np.where(iy == 0, g0 / pr,
                      np.where(iy == 1, (g1 - g0) / pr, -g1 / pr))
        dt0 = np.where(iy == 0, -g0 / pr, np.where(iy == 1, g0 / pr, 0.0))
        dt1 = np.where(iy == 1, -g1 / pr, np.where(iy == 2, g1 / pr, 0.0))
        sw = s * self.C
        gw = w + X.T @ (sw * dz)
        gt0 = float(np.sum(sw * (dt0 + dt1)))          # theta_1 also depends on t0
        gd = float(np.sum(sw * dt1)) * ed
        return perdida, np.concatenate([gw, [gt0, gd]])

    def fit(self, X, y, sample_weight=None):
        from scipy.optimize import minimize
        X = np.asarray(X, np.float64)
        y = np.asarray(y, np.float64)
        self.classes_ = np.unique(y)
        assert tuple(self.classes_) == self.classes_esperadas, self.classes_
        iy = np.searchsorted(self.classes_, y).astype(np.int64)
        s = (np.ones(len(y)) if sample_weight is None
             else np.asarray(sample_weight, np.float64))
        # Start at w = 0 with the thresholds at the logits of the cumulative prior,
        # which is the exact solution of the model without features.
        pi = np.array([np.sum(s[iy == k]) for k in range(3)]) / np.sum(s)
        cum = np.clip(np.cumsum(pi)[:2], 1e-6, 1 - 1e-6)
        t0 = float(np.log(cum[0] / (1 - cum[0])))
        t1 = float(np.log(cum[1] / (1 - cum[1])))
        par0 = np.concatenate([np.zeros(X.shape[1]),
                               [t0, np.log(max(t1 - t0, 1e-3))]])
        self.perdida_inicial_ = float(self._perdida_grad(par0, X, iy, s)[0])
        r = minimize(self._perdida_grad, par0, args=(X, iy, s), jac=True,
                     method='L-BFGS-B',
                     options=dict(maxiter=self.max_iter, ftol=self.tol,
                                  gtol=self.tol))
        par = r.x
        p_dim = X.shape[1]
        self.w_ = par[:p_dim]
        self.theta_ = np.array([par[p_dim], par[p_dim] + np.exp(par[p_dim + 1])])
        self.perdida_final_ = float(r.fun)
        self.n_iter_ = int(r.nit)
        self.convergio_ = bool(r.success)
        return self

    def predict_proba(self, X) -> np.ndarray:
        from scipy.special import expit
        z = np.asarray(X, np.float64) @ self.w_
        q0 = expit(self.theta_[0] - z)
        q1 = expit(self.theta_[1] - z)
        P = np.stack([q0, np.maximum(q1 - q0, 0.0), 1.0 - q1], axis=1)
        return P / np.maximum(P.sum(1, keepdims=True), 1e-300)


class SoftmaxMSE3:
    """Linear 3-class softmax trained on the loss of the metric itself:

        o_i = sum_k p_ik * c_k   with c = (-1, 0, +1)
        0.5 * ||W||_F^2 + C * sum_i s_i * (o_i - y_i)^2

    The parameterization is that of the multinomial head (same number of parameters,
    same family of functions); only the training objective differs. The output is
    still a 3-class posterior, so `p_mi` and the factorization o = p_mi * margin remain
    available, unlike with the ridge regressor.

    The optimization starts at the multinomial solution, so the training loss can only
    decrease from that of the multinomial head, and `perdida_final_ ==
    perdida_inicial_` shows that the optimizer did not move.

    Squared loss and cross-entropy have different scales, so the same `C` does not
    regularize both heads equally; `C` is selected separately for this head.
    """

    def __init__(self, C: float = 1.0, seed: int = SEED, tol: float = 1e-10,
                 max_iter: int = 500):
        self.C, self.seed, self.tol, self.max_iter = C, seed, tol, max_iter

    def _perdida_grad(self, par, X, yv, s):
        n, p = X.shape
        W = par[:3 * p].reshape(3, p)
        b = par[3 * p:]
        Z = X @ W.T + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        o = P @ self.c_
        r = o - yv
        perdida = 0.5 * float(np.sum(W * W)) + self.C * float(np.sum(s * r * r))
        # d o_i / d z_ik = p_ik * (c_k - o_i)
        G = (2.0 * self.C * s * r)[:, None] * P * (self.c_[None, :] - o[:, None])
        gW = W + G.T @ X
        gb = G.sum(0)
        return perdida, np.concatenate([gW.ravel(), gb])

    def fit(self, X, y, sample_weight=None):
        from scipy.optimize import minimize
        from sklearn.linear_model import LogisticRegression
        X = np.asarray(X, np.float64)
        y = np.asarray(y, np.float64)
        self.classes_ = np.unique(y)
        assert tuple(self.classes_) == (-1.0, 0.0, 1.0), self.classes_
        self.c_ = np.asarray(self.classes_, np.float64)
        s = (np.ones(len(y)) if sample_weight is None
             else np.asarray(sample_weight, np.float64))
        base = LogisticRegression(C=self.C, max_iter=2000,
                                 random_state=self.seed)
        base.fit(X, y, sample_weight=sample_weight)
        par0 = np.concatenate([np.asarray(base.coef_, np.float64).ravel(),
                               np.asarray(base.intercept_, np.float64)])
        assert len(par0) == 3 * X.shape[1] + 3, (base.coef_.shape, X.shape)
        self.perdida_inicial_ = float(self._perdida_grad(par0, X, y, s)[0])
        r = minimize(self._perdida_grad, par0, args=(X, y, s), jac=True,
                     method='L-BFGS-B',
                     options=dict(maxiter=self.max_iter, ftol=self.tol,
                                  gtol=self.tol))
        p = X.shape[1]
        self.W_ = r.x[:3 * p].reshape(3, p)
        self.b_ = r.x[3 * p:]
        self.perdida_final_ = float(r.fun)
        self.n_iter_ = int(r.nit)
        self.convergio_ = bool(r.success)
        return self

    def predict_proba(self, X) -> np.ndarray:
        Z = np.asarray(X, np.float64) @ self.W_.T + self.b_
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        return P / P.sum(1, keepdims=True)


class Posterior3:
    """Probabilistic 3-class model over {-1, 0, +1} whose output is the posterior mean
    E[y|x] = P(+1|x) - P(-1|x).

    The MSE is minimized exactly by the posterior mean. A ridge regression to
    {-1, 0, +1} only approximates it with a linear function of the features, and it
    treats rest (0) as the midpoint between the two motor-imagery classes.
    """

    def __init__(self, tipo: str = 'multinomial', C: float = 1.0,
                 shrink: float = 0.1, seed: int = SEED,
                 prior_objetivo: np.ndarray | None = None):
        self.tipo, self.C, self.shrink, self.seed = tipo, C, shrink, seed
        self.prior_objetivo = prior_objetivo

    def fit(self, X, y, sample_weight=None):
        """`sample_weight` carries the cost weights and the weights of soft targets on
        transition windows. With `sample_weight=None` the code path is exactly the
        unweighted one (not an array of ones), so unweighted results stay
        bit-identical."""
        y = np.asarray(y, np.float64)
        if self.tipo == 'multinomial':
            from sklearn.linear_model import LogisticRegression
            self.m_ = LogisticRegression(C=self.C, max_iter=2000,
                                         random_state=self.seed)
        elif self.tipo == 'lda':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            self.m_ = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=self.shrink)
        elif self.tipo == 'ordinal':
            self.m_ = OrdinalLogit3(C=self.C)
        elif self.tipo == 'mse':
            self.m_ = SoftmaxMSE3(C=self.C, seed=self.seed)
        else:
            raise ValueError(self.tipo)
        if sample_weight is None:
            self.m_.fit(np.asarray(X, np.float64), y)
        else:
            self.m_.fit(np.asarray(X, np.float64), y,
                        sample_weight=np.asarray(sample_weight, np.float64))
        self.cls_ = np.asarray(self.m_.classes_, np.float64)
        # Empirical prior of the training set, in the order of `classes_`. With sample
        # weights it is weighted too, so the prior correction starts from the class
        # composition the model was trained on.
        if sample_weight is None:
            self.pi_tr_ = np.array([float(np.mean(y == c)) for c in self.cls_])
        else:
            sw = np.asarray(sample_weight, np.float64)
            self.pi_tr_ = np.array([float(np.sum(sw[y == c]) / np.sum(sw))
                                    for c in self.cls_])
        self.razon_ = None
        if self.prior_objetivo is not None:
            pi = np.asarray(self.prior_objetivo, np.float64)
            assert len(pi) == 3, pi
            # the target prior is given in the order (-1, 0, +1); reorder to `classes_`
            mapa = {-1.0: 0, 0.0: 1, 1.0: 2}
            obj = np.array([pi[mapa[float(c)]] for c in self.cls_])
            self.razon_ = obj / np.maximum(self.pi_tr_, 1e-12)
        return self

    def predict_proba_ajustada(self, X) -> np.ndarray:
        """Class probabilities after the prior correction, in the order of `classes_`.
        `predict` is their posterior mean; this method exposes the components of the
        output without retraining."""
        P = self.m_.predict_proba(np.asarray(X, np.float64))
        if self.razon_ is not None:
            P = P * self.razon_
            P = P / np.maximum(P.sum(1, keepdims=True), 1e-300)
        return P

    def partes(self, X) -> tuple[np.ndarray, np.ndarray]:
        """(output, p_mi), with p_mi = P(+1) + P(-1) the probability of motor imagery.
        Since the output is P(+1) - P(-1), exactly

            output = p_mi * (2*p_pos - 1)   with p_pos = P(+1) / p_mi

        so the 3-class model factorizes into "is there motor imagery" and "which
        class". The direction margin is |2*p_pos - 1| = |output| / p_mi.
        """
        P = self.predict_proba_ajustada(X)
        idx = {float(c): i for i, c in enumerate(self.cls_)}
        p_mi = P[:, idx[1.0]] + P[:, idx[-1.0]]
        return P @ self.cls_, p_mi

    def predict(self, X):
        """Posterior mean. With a target prior, the Bayes correction

            p'(c|x)  proportional to  p(c|x) * pi_target(c) / pi_train(c)

        is applied first. It is exact when only the class proportions differ between
        training and deployment (prior shift); here rest has a different share in the
        calibration and in the evaluation recordings.
        """
        P = self.m_.predict_proba(np.asarray(X, np.float64))
        if self.razon_ is not None:
            P = P * self.razon_
            P = P / np.maximum(P.sum(1, keepdims=True), 1e-300)
        return P @ self.cls_


# Regressors that produce a 3-class posterior and therefore keep `p_mi` and the
# factorization o = p_mi * margin. 'grnn' and 'ridge' do not.
REGRESORES_POSTERIOR = ('multinomial', 'lda', 'ordinal', 'mse')
# Regressors that are linear with an L2 penalty: for them the projection onto the row
# space of the training features (`pipeline._reducir_exacto`) is exact by the
# representer theorem. Shrinkage LDA is not in this group.
REGRESORES_L2_LINEALES = ('multinomial', 'ridge', 'ordinal', 'mse')


def hacer_regresor(nombre: str, **kw):
    """Regressor called `nombre`, built with the keyword arguments `kw`."""
    if nombre == 'grnn':
        return GRNN(**kw)
    if nombre == 'ridge':
        return Ridge(**kw)
    if nombre in REGRESORES_POSTERIOR:
        return Posterior3(tipo=nombre, **kw)
    raise ValueError(nombre)


# ============================================================
# Unsupervised alignment of the target recording
# ============================================================
MODOS_ALINEAMIENTO = ('ninguno', 'centrado', 'mediana', 'zscore', 'cuantil', 'coral')
# Shrinkage of both `coral` covariances toward the scaled identity. The covariance is
# over hundreds of features and is estimated from thousands of strongly correlated
# windows; its small directions are noise that an inverse square root would amplify.
# Same reason as the ridge term in `pipeline._inv_sqrt`.
LAM_CORAL = 0.05
# Reference rows needed to use a local statistic: a block or window needs more than
# this many. Otherwise a fallback is used, described where it applies.
MIN_FILAS_ALINEAMIENTO = 10


def periodo_de_fila(fines: np.ndarray) -> int:
    """Number of samples between consecutive rows of the output grid, read from the
    grid itself. It converts the seconds of `causal_taps` and of the alignment
    segments into rows, and is defined once here so that every caller uses the same
    unit.
    """
    d = np.diff(np.asarray(fines, np.int64))
    u = np.unique(d)
    assert len(u) == 1, f'the output grid is not uniform: {u[:5]}'
    return int(u[0])


def alinear_features(F_tr: np.ndarray, F_ap: np.ndarray, modo: str,
                     filas_ref: np.ndarray | None = None,
                     n_q: int = 256,
                     bloques: np.ndarray | None = None,
                     causal_taps: int | None = None,
                     coral_lam: float | None = None) -> np.ndarray:
    """Map the features of the target recording onto the training feature space using
    only statistics of the target recording itself.

    No target label is used, so this can be done at deployment by recording a few
    minutes of unlabeled signal.

    The features are log-powers: a change of signal scale shifts them (the log of a
    factor) and a change of variability stretches them.
      - `centrado`  corrects the shift only, with the mean
      - `mediana`   the same with the median, less sensitive to heavy-tailed powers
      - `zscore`    corrects shift and scale
      - `cuantil`   also corrects the shape, feature by feature
      - `coral`     corrects the shift and the covariance between features (second
                    order). It is the only mode that is not monotone per column: it
                    mixes features.

    `filas_ref` restricts the target rows used to estimate the statistics.

    Where the statistic is estimated:
      - by default, once for the whole recording;
      - `bloques` gives a group id per row, and one statistic is estimated per group
        (per run or per time segment);
      - `causal_taps` uses a causal running mean over the last `causal_taps` rows,
        the only variant that can run online (`centrado` only).
    With `bloques=None` and `causal_taps=None` a single statistic is used.
    `coral_lam` overrides `LAM_CORAL`.
    """
    if modo in (None, 'ninguno'):
        return F_ap
    assert modo in MODOS_ALINEAMIENTO, modo
    lam = LAM_CORAL if coral_lam is None else float(coral_lam)
    if causal_taps is not None:
        assert bloques is None, 'use either bloques or causal_taps, not both'
        assert modo == 'centrado', ('the causal running mean is only defined for '
                                    f'`centrado`, not for `{modo}`')
        return _centrado_causal(F_tr, F_ap, filas_ref, int(causal_taps))
    if bloques is not None:
        return _alinear_por_bloques(F_tr, F_ap, modo, filas_ref, n_q, bloques, lam)
    return _alinear_un_bloque(F_tr, F_ap, modo, filas_ref, n_q, lam)


def _alinear_por_bloques(F_tr, F_ap, modo, filas_ref, n_q, bloques,
                         lam: float = LAM_CORAL):
    """One statistic per group of rows.

    Fallback: a group with at most `MIN_FILAS_ALINEAMIENTO` reference rows gets the
    global statistic, which amounts to no local correction for that group and keeps
    the result from depending on a handful of windows.
    """
    b = np.asarray(bloques)
    assert len(b) == len(F_ap), (len(b), len(F_ap))
    global_ = _alinear_un_bloque(F_tr, F_ap, modo, filas_ref, n_q, lam)
    use = (np.ones(len(F_ap), bool) if filas_ref is None
           else np.asarray(filas_ref, bool))
    out = np.array(global_, dtype=np.float64, copy=True)
    for g in np.unique(b):
        sel = (b == g)
        ref = sel & use
        if int(ref.sum()) <= MIN_FILAS_ALINEAMIENTO:
            continue                       # keep the global statistic
        out[sel] = _alinear_un_bloque(F_tr, F_ap[sel], modo, ref[sel], n_q, lam)
    return out


def _centrado_causal(F_tr, F_ap, filas_ref, taps: int) -> np.ndarray:
    """Centering with the mean of the reference rows among the last `taps` rows (the
    current row included), without using any later row.

    A row is corrected only when more than `MIN_FILAS_ALINEAMIENTO` reference rows
    fall in its window; otherwise its shift is zero. At start-up this is what happens
    when the system is switched on, and a global statistic would use the future.
    """
    assert taps > 0, taps
    F_ap = np.asarray(F_ap, np.float64)
    use = (np.ones(len(F_ap), bool) if filas_ref is None
           else np.asarray(filas_ref, bool))
    X = np.where(use[:, None], F_ap, 0.0)
    cs = np.vstack([np.zeros((1, F_ap.shape[1])), np.cumsum(X, axis=0)])
    cn = np.concatenate([[0.0], np.cumsum(use.astype(np.float64))])
    i = np.arange(1, len(F_ap) + 1)
    lo = np.maximum(i - taps, 0)
    S, N = cs[i] - cs[lo], cn[i] - cn[lo]
    mu_tr = F_tr.mean(0)
    listo = N > MIN_FILAS_ALINEAMIENTO
    # Build the shift rather than the mean, so that uncorrected rows get an exact zero
    # and not the rounding residue of `- mu + mu`.
    desp = np.where(listo[:, None], mu_tr - S / np.maximum(N, 1.0)[:, None], 0.0)
    return F_ap + desp


def _potencia_psd(C: np.ndarray, p: float, lam: float) -> np.ndarray:
    """C^p for a symmetric positive semidefinite C, after shrinkage toward the identity
    scaled by the mean variance. Without shrinkage the small directions, which are
    noise, would be amplified by 1/sqrt(eps).
    """
    # `np.cov` returns a scalar, not a 1x1 matrix, when there is a single feature.
    C = np.atleast_2d(np.asarray(C, np.float64))
    d = C.shape[0]
    C = 0.5 * (C + C.T)
    if lam > 0:
        C = (1.0 - lam) * C + lam * (float(np.trace(C)) / d) * np.eye(d)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, max(float(w.max()) * 1e-12, 1e-30))
    return (V * (w ** p)) @ V.T


def _alinear_un_bloque(F_tr: np.ndarray, F_ap: np.ndarray, modo: str,
                       filas_ref: np.ndarray | None, n_q: int,
                       lam: float = LAM_CORAL) -> np.ndarray:
    """The alignment formulas, with a single statistic for all rows."""
    S = F_ap if filas_ref is None else F_ap[filas_ref]
    assert len(S) > 10, f'only {len(S)} rows left to estimate the alignment'
    if modo == 'centrado':
        return F_ap - S.mean(0) + F_tr.mean(0)
    if modo == 'mediana':
        return F_ap - np.median(S, 0) + np.median(F_tr, 0)
    if modo == 'zscore':
        sd = np.maximum(S.std(0), 1e-12)
        return (F_ap - S.mean(0)) / sd * F_tr.std(0) + F_tr.mean(0)
    if modo == 'coral':
        # CORAL: whiten the target with its own covariance and recolor it with the
        # training covariance. `zscore` is the special case with diagonal matrices;
        # CORAL adds the off-diagonal terms, i.e. how the features covary.
        mu_s, mu_t = S.mean(0), F_tr.mean(0)
        A = (_potencia_psd(np.cov(S, rowvar=False), -0.5, lam)
             @ _potencia_psd(np.cov(F_tr, rowvar=False), 0.5, lam))
        return (F_ap - mu_s) @ A + mu_t
    # 'cuantil': each target feature goes through its own empirical CDF and is mapped
    # onto the training quantiles. The map is monotone per feature, but the classifier
    # combines features linearly, so its output is not a monotone function of the
    # unaligned output and the AUC can change.
    qs = np.linspace(0.0, 1.0, n_q)
    Qt = np.quantile(F_tr, qs, axis=0)
    out = np.empty_like(F_ap)
    for j in range(F_ap.shape[1]):
        r = np.searchsorted(np.sort(S[:, j]), F_ap[:, j], side='left')
        out[:, j] = np.interp(np.clip((r + 0.5) / len(S), 0.0, 1.0), qs, Qt[:, j])
    return out


# ============================================================
# Feature ranking by mutual information
# ============================================================
def mi_features(F: np.ndarray, y: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Mutual information between each feature (discretized into `n_bins` quantile
    bins) and the class label. The winner used the same criterion estimated with a
    KDE; a quantile histogram has no smoothing parameter."""
    F = np.asarray(F, np.float64)
    y = np.asarray(y)
    clases = np.unique(y)
    py = np.array([(y == c).mean() for c in clases])
    Hy = -np.sum(py * np.log(np.maximum(py, 1e-300)))
    mi = np.zeros(F.shape[1])
    for j in range(F.shape[1]):
        q = np.quantile(F[:, j], np.linspace(0, 1, n_bins + 1)[1:-1])
        b = np.searchsorted(q, F[:, j])
        Hyx = 0.0
        for k in range(n_bins):
            m = b == k
            if not m.any():
                continue
            pk = m.mean()
            pyk = np.array([(y[m] == c).mean() for c in clases])
            Hyx += pk * (-np.sum(pyk * np.log(np.maximum(pyk, 1e-300))))
        mi[j] = Hy - Hyx
    return mi


# ============================================================
# Temporal post-processing
# ============================================================
class PostLineal:
    """Ridge regression from a causal buffer of the last `taps` values of the input
    sequence to the current label. A closed-form counterpart of the winner's second
    GRNN. `taps` counts samples of the input sequence."""

    def __init__(self, taps: int = 40, alpha: float = 1.0):
        self.taps, self.alpha = taps, alpha

    @staticmethod
    def _buffer(r: np.ndarray, taps: int) -> np.ndarray:
        """(n,) -> (n, taps): row i = [r_i, r_{i-1}, ..., r_{i-taps+1}], padded with the
        first value. Strictly causal."""
        r = np.asarray(r, np.float64)
        idx = np.arange(len(r))[:, None] - np.arange(taps)[None, :]
        return r[np.maximum(idx, 0)]

    def fit(self, r: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None):
        B = self._buffer(r, self.taps)
        m = np.isfinite(y) if mask is None else (mask & np.isfinite(y))
        self.lin_ = Ridge(self.alpha).fit(B[m], y[m])
        return self

    def predict(self, r: np.ndarray) -> np.ndarray:
        return self.lin_.predict(self._buffer(r, self.taps))


class PostGRNN(PostLineal):
    """The same buffer with a GRNN, as described by the winner. It is fitted on every
    `submuestreo`-th usable sample."""

    def __init__(self, taps: int = 40, spread: float | None = None,
                 max_pts: int = 4000, submuestreo: int = 5):
        self.taps, self.spread, self.max_pts, self.sub = taps, spread, max_pts, submuestreo

    def fit(self, r, y, mask=None):
        B = self._buffer(r, self.taps)
        m = np.isfinite(y) if mask is None else (mask & np.isfinite(y))
        i = np.where(m)[0][::self.sub]
        self.g_ = GRNN(self.spread, self.max_pts).fit(B[i], y[i])
        return self

    def predict(self, r):
        return self.g_.predict(self._buffer(r, self.taps))


class PostHMM:
    """Causal Bayesian filter over the 3 states {-1, 0, +1}, applied to a sequence of
    outputs spaced `paso_s` seconds apart.

    The duration prior comes from the official dataset description, which was public
    to every competitor: in the evaluation data, MI periods and the intervals between
    them last "between 1.5 and 8 seconds". The mean of that uniform range is 4.75 s,
    so the probability of leaving a state in one step is paso_s/4.75. No evaluation
    label is used.

    The output is the posterior mean P(+1) - P(-1), which minimizes the MSE under the
    model. `p_mi` is the fraction of time in MI (also public: the official MSE of the
    constant output 0, 0.509, equals that fraction).
    """

    def __init__(self, dur_media_s: float = 4.75, paso_s: float = 0.1,
                 p_mi: float = 0.509, n_bins: int = 40):
        self.dur, self.paso, self.p_mi, self.n_bins = dur_media_s, paso_s, p_mi, n_bins

    def fit(self, r: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None):
        """Learn p(r | state) as a histogram over quantile bins of the calibration
        predictions (out-of-fold), with Laplace smoothing."""
        m = np.isfinite(y) if mask is None else (mask & np.isfinite(y))
        r, y = np.asarray(r, np.float64)[m], np.asarray(y, np.float64)[m]
        self.bordes_ = np.quantile(r, np.linspace(0, 1, self.n_bins + 1))
        self.bordes_[0], self.bordes_[-1] = -np.inf, np.inf
        b = np.clip(np.searchsorted(self.bordes_, r, side='right') - 1,
                    0, self.n_bins - 1)
        self.estados_ = np.array([-1.0, 0.0, 1.0])
        L = np.zeros((3, self.n_bins))
        for k, e in enumerate(self.estados_):
            c = np.bincount(b[y == e], minlength=self.n_bins).astype(float)
            L[k] = (c + 1.0) / (c.sum() + self.n_bins)          # Laplace
        self.L_ = L
        return self

    def _transicion(self) -> np.ndarray:
        """3x3 transition matrix. From MI the chain can only move to NC; from NC it
        moves to either MI class with equal probability. The rates follow from the
        duration prior and the MI fraction."""
        q = self.paso / self.dur                       # prob. of leaving an MI state
        # mean NC duration such that the MI fraction equals p_mi
        d_nc = self.dur * (1.0 - self.p_mi) / max(self.p_mi, 1e-9)
        q_nc = min(self.paso / max(d_nc, 1e-9), 0.99)
        T = np.zeros((3, 3))
        T[0] = [1 - q, q, 0.0]                         # -1 -> NC
        T[2] = [0.0, q, 1 - q]                         # +1 -> NC
        T[1] = [q_nc / 2, 1 - q_nc, q_nc / 2]          # NC -> +-1
        return T / T.sum(1, keepdims=True)

    def predict(self, r: np.ndarray) -> np.ndarray:
        T = self._transicion()
        b = np.clip(np.searchsorted(self.bordes_, np.asarray(r, np.float64),
                                    side='right') - 1, 0, self.n_bins - 1)
        E = self.L_[:, b]                              # (3, n)
        pi = np.array([self.p_mi / 2, 1 - self.p_mi, self.p_mi / 2])
        out = np.empty(len(b))
        p = pi
        for i in range(len(b)):
            p = (p @ T) * E[:, i]
            ssum = p.sum()
            p = p / ssum if ssum > 0 else pi
            out[i] = p[2] - p[0]                       # posterior mean
        return out


def hacer_post(nombre: str, **kw):
    """Post-processor called `nombre`; None for 'ninguno'."""
    if nombre == 'ninguno':
        return None
    if nombre == 'lineal':
        return PostLineal(**kw)
    if nombre == 'grnn':
        return PostGRNN(**kw)
    if nombre == 'hmm':
        return PostHMM(**kw)
    raise ValueError(nombre)


# ============================================================
# Utilities
# ============================================================
def hash_cfg(cfg: dict) -> str:
    """Short (16 hex digits) SHA-256 of a configuration."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True,
                                     default=str).encode()).hexdigest()[:16]


def guardar_json(p: Path, obj) -> None:
    """Write `obj` as JSON through a temporary file and a rename; numpy values are
    converted by `_conv`."""
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_conv),
                   encoding='utf-8')
    tmp.replace(p)


def _conv(o):
    """JSON conversion of numpy scalars and arrays; non-finite floats become null."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def mse_stream(out: np.ndarray, y: np.ndarray,
               mask: np.ndarray | None = None) -> dict:
    """MSE of a 100 Hz output against 100 Hz labels (calibration streams), together
    with the MSE of the constant output 0 on the same samples."""
    o, t = np.asarray(out, np.float64), np.asarray(y, np.float64)
    n = min(len(o), len(t))
    o, t = o[:n], t[:n]
    ok = np.isfinite(t)
    if mask is not None:
        ok &= mask[:n]
    e = o[ok] - t[ok]
    return dict(mse=float(np.mean(e ** 2)), mse_cero=float(np.mean(t[ok] ** 2)),
                n=int(ok.sum()))
