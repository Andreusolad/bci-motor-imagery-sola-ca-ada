"""Classical asynchronous system for BCI Competition IV Data Set 1: the single code path.

One code path serves both uses:
  - `correr_cv`: 5-fold temporal cross-validation inside the calibration recording
    (the development bench, the only data on which configurations are chosen).
  - `correr_eval` / `correr_eval_ensamblado`: train on the whole calibration recording
    and predict the continuous evaluation recording, with one configuration or with the
    average of several.

Geometry (the same in both cases, and the winner's): a causal window of `w_s` seconds
that ends at the predicted sample, one output every 0.1 s, held until the next one
(causal zero-order hold) to give a 100 Hz output stream.

Per configuration: band-pass filter bank -> CSP filters per band for MI1 vs MI2 and for
each MI class vs each rest sub-state (optionally after Euclidean recentering) ->
log-power of the CSP projections -> optional unsupervised alignment of the target
features -> optional feature selection -> min-max normalization -> regression or
classification head -> optional temporal post-processing -> clipping to
[-recorte, recorte]. Every option is in `CFG_BASE`; `cfg_con(**changes)` builds a
configuration.

Used by run_shot.py (evaluation shots), causality_test.py (perturbation hook of the
evaluation) and check_regression.py (development-bench regression check).

Usage:
    import pipeline as P
    cfg = P.cfg_con(w_s=1.0, post_taps=320)
    r_dev = P.correr_cv('a', cfg)      # r_dev['mse']: development-bench MSE
    r_ev = P.correr_eval('a', cfg)     # r_ev['mse']: official MSE on the evaluation
"""
from __future__ import annotations

import numpy as np
from scipy.signal import sosfiltfilt

import data_io as A
import components as K
from components import FS, potencia_deslizante, csp, construir_banco, filtrar, subestados_nc, mi_features, hacer_regresor, hacer_post, stream_calibracion, senal_calibracion, senal_evaluacion, mse_stream

STRIDE_SALIDA = 10            # 0.1 s at 100 Hz: the output step (the winner's)
STRIDE_COV = 50               # 0.5 s: calibration grid of the CSP covariances
STRIDE_COV_EV = 200           # 2 s: coarser grid on the target, for recentering statistics
MIN_POR_CLASE = 10            # minimum grid windows per class and per rest sub-state
# Cap on the number of features. A cost limit, not a conceptual one: each feature costs
# a spatial projection of the whole signal. The cost of the head does not grow with it
# (see `_reducir_exacto`), so the cap can be high.
MAX_FEATURES = 12000


class ConfigInviable(Exception):
    """The configuration cannot be evaluated: too few windows of some class, a singular
    CSP problem, too many features or incompatible options. Not a code error."""


# Every option of the system with its default. `cfg_con` builds a configuration from it;
# config/dev_best.json holds the one chosen on the development bench.
CFG_BASE = dict(
    # Filter bank (see `components.construir_banco`): 'ganador' (the winner's 8
    # Chebyshev-II bands), 'fbcsp' (4 Hz Butterworth bands, 4-40 Hz), 'mb' (one 8-30 Hz
    # band), one band of `components.BANDAS_EXTRA`, or the winner's bank plus extra
    # bands (`components.BANCOS_COMPUESTOS`).
    banco='ganador',
    # How the band-pass is applied. It decides whether the system can run live:
    #   'continuo_zp'     filtfilt over the whole recording: uses future samples
    #   'continuo_causal' one forward sosfilt pass over the recording: causal, but with
    #                     phase delay and a start-up transient
    #   'ventana_zp'      filtfilt inside each window: causal (the window ends at the
    #                     predicted sample, so it holds no future sample) and without
    #                     phase delay
    # The winning entry describes its filtering as zero-phase without saying whether it
    # was applied per epoch or to the whole recording.
    filtro='continuo_zp',
    # Not read by the pipeline (the band-pass is set by `filtro`); kept so that stored
    # configurations such as config/dev_best.json still load.
    causal=False,
    car=False,                # common average reference over the 59 channels
    w_s=2.5,                  # window (s) of the CSP covariances, training masks and power
    w_pot=(),                 # extra power windows (s), multi-scale; () = only w_s
    # Grid windows that train the model:
    #   'todo'         every window whose end sample has a defined label
    #   'puro'         only windows entirely inside one state
    #   'epoca_unica'  one window per constant-state segment, the last one that fits in
    #                  it (the winner's one epoch per trial)
    #   'mixto'        'todo' plus the transition windows (end label undefined) with the
    #                  hard label of their majority state. It is the capacity control of
    #                  'blando': same rows, and only the target changes.
    #   'blando'       'todo' plus the transition windows with the fraction of the window
    #                  in each state as target. The MSE-optimal output is the conditional
    #                  mean, so on those windows the model can learn to output ~0 instead
    #                  of a confident value.
    # 'puro' and 'epoca_unica' also narrow the set used for the CSP filters; 'mixto' and
    # 'blando' change only the training set of the head.
    entrenar='todo',          # 'todo' | 'puro' | 'epoca_unica' | 'mixto' | 'blando'
    # Weight of each transition window added by 'mixto' and 'blando'. 1.0 = the same
    # weight as a hard row.
    blando_peso=1.0,
    m_csp=2,                  # CSP filters per end: 2 largest + 2 smallest eigenvalues
    lam_cov=0.05,             # shrinkage of the CSP covariances toward scaled identity
    # Source of the rest (NC) windows for CSP: 'todo' all of them, 'pausas' only those
    # inside the long pauses between trials, 'entretrial' only the others.
    nc_fuente='todo',         # 'todo' | 'pausas' | 'entretrial'
    nk=1,                     # rest sub-states (PCA + k-means), each with its CSP pairs
    nk_modo='kmeans',         # 'kmeans' | 'aleatorio' (control, see `_grupos_nc`)
    # Unsupervised Euclidean recentering: the training set is whitened with its mean
    # covariance and the target recording with its own, without labels.
    recentrar=False,
    # Alignment of the target-recording features with its own statistics, without
    # labels (see `components.alinear_features`).
    alineamiento='ninguno',   # 'ninguno'|'centrado'|'mediana'|'zscore'|'cuantil'|'coral'
    # Where the alignment is estimated. 0.0 = once for the whole target recording. > 0 =
    # once per block of that many seconds, cut by sample index only (no labels, no task
    # structure). Not causal: a row early in a block uses later rows of the block.
    alineamiento_bloques_s=0.0,
    # Causal alternative, for 'centrado' only: a running mean over the last
    # `alineamiento_causal_s` seconds of target rows, never a later row. Seconds are
    # converted to rows with `components.periodo_de_fila`, which reads the actual step
    # of the output grid. Excludes `alineamiento_bloques_s`. 0.0 = off.
    alineamiento_causal_s=0.0,
    # Causal version of `recentrar` (acts only with recentrar=True). With 0.0 the target
    # is whitened with its mean covariance over the whole recording, future included.
    # With > 0 the target is cut into blocks of that many seconds and block k is
    # whitened with the covariances of the earlier blocks; block 0, which has no past,
    # keeps the training whitening, as a live system would when switched on.
    recentrar_causal_s=0.0,
    # Head. 'ridge' regresses to {-1, 0, +1} and 'grnn' is the winner's GRNN.
    # 'multinomial', 'lda', 'ordinal' (proportional odds on the ordered classes) and
    # 'mse' (3-class softmax trained on the squared error of the metric) output the
    # posterior mean P(+1) - P(-1) and also return `p_mi` (see `components.Posterior3`).
    regresor='ridge',         # 'ridge'|'grnn'|'multinomial'|'lda'|'ordinal'|'mse'
    alpha=1.0,                # ridge
    C=1.0,                    # multinomial / ordinal / mse
    shrink=0.1,               # lda
    # Training weight of an MI sample (+-1) relative to a rest sample; 1.0 = no weights.
    # The value derived from the cost matrix of the metric is 2.5 (row sums 5, 2, 5; see
    # `components.pesos_por_coste`). Values other than 1.0 need a posterior head.
    costo_signo=1.0,
    # 'objetivo': Bayes prior correction of the posterior heads to the class prior
    # (p/2, 1 - p, p/2), with p the fraction of time in MI (from the calibration labels
    # in cross-validation, the published 0.509 in evaluation). No effect on ridge/grnn.
    prior='ninguno',          # 'ninguno' | 'objetivo'
    n_feats=0,                # 0 = all features; k > 0 = top k by mutual information
    post='lineal',            # temporal post-processing: 'ninguno'|'lineal'|'grnn'|'hmm'
    # The post-processing is fitted and applied on the 100 Hz stream produced by
    # `data_io.expandir_causal`, not on the window grid, so `post_taps` counts 10 ms
    # samples: 40 taps = 0.4 s (4 distinct window outputs), and the winner's 4 s buffer
    # is 400 taps.
    post_taps=40,             # causal buffer of the post-processing, in 100 Hz samples
    post_alpha=10.0,          # ridge penalty of the 'lineal' post-processing
    recorte=1.0,              # the output is clipped to [-recorte, recorte]
)

# Temporal prior of the HMM post-processing and class prior. On the development bench
# they are measured on the calibration stream (training data). In evaluation they come
# from the public description of the dataset, available to every competitor: periods
# last "between 1.5 and 8 seconds" (mean 4.75 s), and the constant-zero output scores
# 0.509, which equals the fraction of time in MI. No evaluation label is read.
HMM_DUR_PUBLICADA = (1.5 + 8.0) / 2.0
HMM_PMI_PUBLICADA = 0.509


def cfg_con(**kw) -> dict:
    """CFG_BASE with the given changes. Unknown keys are rejected."""
    c = dict(CFG_BASE)
    desconocidas = set(kw) - set(c)
    assert not desconocidas, f'unknown parameters: {desconocidas}'
    c.update(kw)
    c['w_pot'] = tuple(c['w_pot'])
    return c


def _ventanas_potencia(cfg: dict) -> tuple[float, ...]:
    """Power windows in seconds: `w_s` plus `w_pot`, sorted, without repeats."""
    return tuple(sorted({float(cfg['w_s'])} | {float(v) for v in cfg['w_pot']}))


def _kw_post(cfg: dict, dur: float, pmi: float) -> dict:
    """Keyword arguments of the post-processing `cfg['post']`."""
    if cfg['post'] == 'lineal':
        return dict(taps=cfg['post_taps'], alpha=cfg['post_alpha'])
    if cfg['post'] == 'grnn':
        return dict(taps=cfg['post_taps'])
    if cfg['post'] == 'hmm':
        return dict(dur_media_s=dur, p_mi=pmi)
    return {}


def _prior_de(y: np.ndarray) -> tuple[float, float]:
    """Mean segment duration (s) and fraction of MI of a label stream."""
    e = np.sign(np.nan_to_num(y, nan=0.0))
    cam = int(np.sum(e[1:] != e[:-1])) + 1
    ok = np.isfinite(y)
    return float(len(e) / max(cam, 1) / FS), float(np.mean(np.abs(y[ok])))


def _senal_perturbada(sujeto: str, car: bool, perturbacion: dict | None):
    """Evaluation signal, with an impulse added if `perturbacion` is given.

    With `antes_del_car=True` (the default) and CAR on, the signal is loaded without
    CAR, perturbed, and re-referenced here with the same formula as the loader
    (`x - x.mean(0)`), so the impulse also goes through the CAR. causality_test.py
    checks that with zero amplitude this route gives the loader's signal bit for bit.

    The impulse must go to a subset of channels: added to all 59 and followed by the
    common average reference it would cancel exactly, and the test would pass without
    testing anything. `canales=None` (all channels) is therefore rejected.
    """
    if perturbacion is None:
        return senal_evaluacion(sujeto, car)
    t = int(perturbacion['muestra'])
    amp = float(perturbacion['amplitud'])
    antes = bool(perturbacion.get('antes_del_car', True))
    if antes and car:
        x = np.array(senal_evaluacion(sujeto, False), np.float64, copy=True)
    else:
        x = np.array(senal_evaluacion(sujeto, car), np.float64, copy=True)
    ch = perturbacion.get('canales')
    ch = np.arange(x.shape[0]) if ch is None else np.asarray(ch, np.int64)
    assert 0 <= t < x.shape[1], (t, x.shape)
    assert len(ch) < x.shape[0], ('the impulse cannot go to all channels: the CAR '
                                  'would cancel it and the test would prove nothing')
    x[ch, t] += amp
    if antes and car:
        x = x - x.mean(axis=0, keepdims=True)
    return x


def _inv_sqrt(R: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Symmetric, well-conditioned R^{-1/2}.

    The relative ridge `eps` matters. With CAR the 59-channel covariance has rank 58,
    so R has (near) null directions; flooring the eigenvalues alone would amplify them
    by 1/sqrt(floor) and turn noise into the dominant component, and the CSP computed
    afterwards then fails ("leading minor ... is not positive definite"). Adding
    eps*trace/n before inverting keeps the problem well conditioned; eps=0 gives the
    exact R^{-1/2}.
    """
    R = np.asarray(R, np.float64)
    n = R.shape[0]
    if eps > 0:
        R = R + eps * (np.trace(R) / n) * np.eye(n)
    v, U = np.linalg.eigh(R)
    v = np.maximum(v, max(float(v.max()) * 1e-12, 1e-30))
    return (U / np.sqrt(v)) @ U.T


# ============================================================
# Expensive preparation, shared between configurations
# ============================================================
class Sesion:
    """Band-filtered signals and trace-normalized covariances on a fixed grid.

    Everything expensive depends only on the subject, `banco`, `filtro`, `car`, `w_s`,
    `w_pot` and the target recording (plus, for the evaluation, whether
    `recentrar_causal_s > 0`), so configurations that differ only in the other options
    can share one session (argument `ses` of `correr_cv` and `correr_eval`).

    Main attributes: `g_fin` window end samples of the calibration grid (every 0.5 s),
    `g_y` the label at each end, `g_pura`/`g_epoca_unica`/`g_pausa` masks and `g_frac`
    state fractions on that grid, `xb` per-band signals, `Cg`/`trg` per-band covariances
    and traces; `xb_ev`, `R_ev`, `C_ev`, `g_ev` hold the same for the target recording.
    """

    def __init__(self, sujeto: str, cfg: dict, con_eval: bool = False,
                 destino_calib: str | None = None,
                 perturbacion: dict | None = None):
        """Prepare the session of `sujeto` from its calibration recording.

        `con_eval=True` also prepares the subject's evaluation recording as the target.
        `destino_calib` instead uses the calibration recording of another subject as the
        target. That target comes with its labels (`y_dst`, `mrk_pos_dst`), so transfer
        between recordings can be measured without touching the evaluation data; its
        features are computed exactly as for the evaluation.

        `perturbacion` adds an impulse to the evaluation signal (only with
        `con_eval=True`). It is the hook of the causality test (causality_test.py): if
        the system is causal, an impulse at sample `t` cannot change any output before
        `t`. It lives here so that the test goes through the complete system on the same
        code path as the evaluation.
        Format: dict(muestra=int, amplitud=float in uV, canales=list[int],
                     antes_del_car=bool). With None the code path is unchanged.
        """
        assert not (con_eval and destino_calib), 'ambiguous target: eval and calib'
        self.perturbacion = None if perturbacion is None else dict(perturbacion)
        self.s, self.cfg = sujeto, cfg
        self.destino = ('eval' if con_eval else
                        (f'calib:{destino_calib}' if destino_calib else None))
        self.w = int(round(cfg['w_s'] * FS))
        self.w_pot = [int(round(v * FS)) for v in _ventanas_potencia(cfg)]
        self.w_max = max([self.w] + self.w_pot)
        sos, self.nombres_banda = construir_banco(cfg['banco'])
        self.n_bandas = len(sos)

        st = stream_calibracion(sujeto)
        self.y, self.T = st['y'], st['T']
        self.mrk_pos, self.clases = st['mrk_pos'], st['clases']
        x = senal_calibracion(sujeto, cfg['car'])
        self.n_ch = x.shape[0]

        self.g_fin = np.arange(self.w_max - 1, self.T, STRIDE_COV, dtype=np.int64)
        self.g_y = self.y[self.g_fin]
        self.g_pura = self._mascara_pura(st)
        self.g_epoca_unica = self._mascara_epoca_unica(st)
        self.g_frac = self._composicion(st)
        # Cross-check: `g_pura` and `g_frac` are derived separately from the same state
        # vector, so a window is pure if and only if one of its three fractions is 1.
        assert np.array_equal(self.g_pura, (self.g_frac == 1.0).any(axis=1)), \
            'g_pura and g_frac disagree'
        self.g_pausa = self._mascara_pausa(st)

        self.sos = sos
        self.por_ventana = cfg['filtro'] == 'ventana_zp'
        cau = cfg['filtro'] == 'continuo_causal'
        self.xb, self.Cg, self.trg = [], [], []
        for S in sos:
            # with 'ventana_zp' the signal is stored unfiltered and the band-pass is
            # applied inside each window; spatial projection and temporal filtering
            # are both linear and commute, so the order does not change the result
            xf = x if self.por_ventana else filtrar(x, S, cau)
            C, tr = self._cov_grid(xf, self.g_fin, sos=S if self.por_ventana else None)
            self.xb.append(np.asarray(xf, np.float32))
            self.Cg.append(C)
            self.trg.append(tr)
        # log-power per channel and band on the grid, from the covariance diagonals
        # (the covariances are trace-normalized, hence the product with the trace)
        self.pot_grid = np.concatenate(
            [np.log(np.maximum(np.einsum('nii->ni', C) * tr[:, None], 1e-30))
             for C, tr in zip(self.Cg, self.trg)], axis=1)

        self.xb_ev = self.R_ev = None
        self.y_dst = self.mrk_pos_dst = None
        if con_eval or destino_calib:
            if con_eval:
                xe = _senal_perturbada(sujeto, cfg['car'], perturbacion)
            else:
                # a calibration target brings its labels, so transfer can be measured
                # without the evaluation recording
                st_d = stream_calibracion(destino_calib)
                self.y_dst = st_d['y']
                self.mrk_pos_dst = st_d['mrk_pos']
                xe = senal_calibracion(destino_calib, cfg['car'])
            self.T_ev = xe.shape[1]
            g_ev = np.arange(self.w_max - 1, self.T_ev, STRIDE_COV_EV, dtype=np.int64)
            self.g_ev = g_ev
            # The per-grid-point covariances of the target are kept only when the causal
            # recentering needs them (`recentrar_causal_s > 0`): they take about 130 MB
            # per session.
            self.C_ev = [] if float(cfg.get('recentrar_causal_s', 0.0)) > 0.0 else None
            self.xb_ev, self.R_ev = [], []
            for S in sos:
                xf = xe if self.por_ventana else filtrar(xe, S, cau)
                Ce, _ = self._cov_grid(
                    xf, g_ev, sos=S if self.por_ventana else None)
                self.xb_ev.append(np.asarray(xf, np.float32))
                self.R_ev.append(Ce.mean(0).astype(np.float64))
                if self.C_ev is not None:
                    self.C_ev.append(Ce)
            self.n_grid_ev = int(len(g_ev))

    def _mascara_pura(self, st: dict) -> np.ndarray:
        """Windows that lie entirely inside a single state (no state change inside).

        A window that crosses a transition is part rest and part MI; training it with a
        hard +-1 target teaches the model to be confident where it should not be.
        """
        estado = np.zeros(self.T, np.int64)
        for p, yy in zip(st['mrk_pos'], st['mrk_y']):
            estado[int(p):int(p) + st['n_mi_samp']] = int(yy)
        cs = np.concatenate([[0], np.cumsum(estado)])
        ca = np.concatenate([[0], np.cumsum(np.abs(estado))])
        f, w = self.g_fin, self.w
        suma = cs[f + 1] - cs[f + 1 - w]
        abso = ca[f + 1] - ca[f + 1 - w]
        # the state is constant (v) over the whole window <=>
        #   v = 0  : abso == 0
        #   v = +-1: abso == w (all non-zero) and |suma| == w (all of the same sign)
        return (abso == 0) | ((abso == w) & (np.abs(suma) == w))

    def _mascara_epoca_unica(self, st: dict) -> np.ndarray:
        """One clean window per constant-state segment: the last one that fits in it.

        This follows the winner's recipe of one training epoch per trial: their epoch,
        1.5-4.0 s after the cue, is the 2.5 s window that ends at the end of the 4 s cue,
        the last one that fits in the MI segment. Here the rule is applied on the grid
        and also to rest segments, giving one epoch per trial and one per rest interval
        (their "200 balanced trials"). Unlike 'puro', which keeps every window that does
        not cross a state change, this keeps far fewer windows.
        """
        estado = np.zeros(self.T, np.int64)
        for p, yy in zip(st['mrk_pos'], st['mrk_y']):
            estado[int(p):int(p) + st['n_mi_samp']] = int(yy)
        # maximal segments of constant state
        cortes = np.flatnonzero(np.diff(estado)) + 1
        bordes = np.concatenate([[0], cortes, [self.T]])
        m = np.zeros(len(self.g_fin), bool)
        f, w = self.g_fin, self.w
        for a, b in zip(bordes[:-1], bordes[1:]):
            # windows [f-w+1, f] entirely inside [a, b)
            ok = (f - w + 1 >= a) & (f < b)
            idx = np.flatnonzero(ok)
            if len(idx):
                m[idx[-1]] = True            # the last one (latest end)
        # an epoch exists only in a segment it fits in, so every epoch is pure by
        # construction and the mask must be a subset of `g_pura`
        assert np.all(self.g_pura[m]), 'a single epoch is not pure: check the segments'
        return m

    def _composicion(self, st: dict) -> np.ndarray:
        """(n_grid, 3): exact fraction of each window in the states (-1, 0, +1).

        It gives a target to the transition windows, which have none under the official
        convention (their end label is NaN): the soft target of entrenar='blando' and
        the majority label of 'mixto'. Computed with cumulative sums, exactly and in
        O(T). The three columns sum to 1 by construction, and `__init__` cross-checks
        the result against `g_pura`.
        """
        estado = np.zeros(self.T, np.int64)
        for p, yy in zip(st['mrk_pos'], st['mrk_y']):
            estado[int(p):int(p) + st['n_mi_samp']] = int(yy)
        cs = np.concatenate([[0], np.cumsum(estado)])
        ca = np.concatenate([[0], np.cumsum(np.abs(estado))])
        f, w = self.g_fin, self.w
        assert int(f[0]) + 1 - w >= 0, 'the first window starts before sample 0'
        suma = cs[f + 1] - cs[f + 1 - w]
        abso = ca[f + 1] - ca[f + 1 - w]
        n_pos = (abso + suma) // 2
        n_neg = (abso - suma) // 2
        assert np.array_equal(n_pos + n_neg, abso), 'inconsistent sign counts'
        return np.stack([n_neg, w - abso, n_pos], axis=1).astype(np.float64) / float(w)

    def _mascara_pausa(self, st: dict, hueco_min: int = 1000) -> np.ndarray:
        """Windows that lie entirely inside a long pause between trials.

        When two consecutive cues are more than `hueco_min` samples apart, the pause runs
        from 8 s after the first cue (4 s of MI plus the 4 s break) to the next cue; the
        tail after the last cue counts too. Besides the regular 4 s rest between trials,
        the calibration recordings contain a few pauses of about 30 s. Rest in those
        pauses does not follow imagery closely, unlike the regular 4 s rest, which comes
        right after the imagery and may carry its tail. Used by `nc_fuente`.
        """
        pos = np.asarray(st['mrk_pos'], np.int64)
        fin_trial = st['n_mi_samp'] + int(round(K.CAL_BREAK_S * FS))
        es = np.zeros(self.T, np.int64)
        for k in range(len(pos) - 1):
            if pos[k + 1] - pos[k] > hueco_min:
                es[pos[k] + fin_trial:pos[k + 1]] = 1
        if self.T - pos[-1] > hueco_min:          # the tail after the last cue
            es[pos[-1] + fin_trial:] = 1
        c = np.concatenate([[0], np.cumsum(es)])
        f, w = self.g_fin, self.w
        return (c[f + 1] - c[f + 1 - w]) == w

    def _cov_grid(self, xf: np.ndarray, g_fin: np.ndarray, chunk: int = 512,
                  sos=None):
        """Trace-normalized covariance of each grid window, and its trace.

        Each window is demeaned. If `sos` is given, the signal arrives unfiltered and the
        band-pass is applied inside each window with filtfilt (filtro='ventana_zp').
        Returns (C (n, n_ch, n_ch) float32, trace (n,)).
        """
        n = len(g_fin)
        C = np.empty((n, self.n_ch, self.n_ch), np.float32)
        tr = np.empty(n, np.float64)
        ar = np.arange(self.w)
        for a in range(0, n, chunk):
            f = g_fin[a:a + chunk]
            idx = f[:, None] - (self.w - 1) + ar[None, :]
            W = xf[:, idx].transpose(1, 0, 2)
            W = W - W.mean(axis=2, keepdims=True)
            if sos is not None:
                W = sosfiltfilt(sos, W, axis=-1)
            # batched matmul (BLAS) instead of einsum: same result, much faster
            Cc = np.matmul(W, W.transpose(0, 2, 1)) / float(self.w)
            t = np.trace(Cc, axis1=1, axis2=2)
            C[a:a + chunk] = (Cc / np.maximum(t, 1e-30)[:, None, None]).astype(np.float32)
            tr[a:a + chunk] = t
        return C, tr


# ============================================================
# CSP: class pairs and filters
# ============================================================
def _grupos_nc(ses: Sesion, m_nc: np.ndarray, cfg: dict) -> list[np.ndarray]:
    """Split of the rest (NC) grid windows into `nk` sub-states.

    `nk_modo='kmeans'` is the winner's recipe (PCA + k-means on the per-channel,
    per-band log-powers of the rest windows). `nk_modo='aleatorio'` splits the same
    windows into nk random groups: a control that separates modelling rest sub-states
    from simply adding CSP filters. Groups of MIN_POR_CLASE windows or fewer are dropped.
    """
    idx = np.where(m_nc)[0]
    nk = int(cfg['nk'])
    if nk <= 1:
        return [idx]
    if cfg['nk_modo'] == 'aleatorio':
        rng = np.random.default_rng(K.SEED + nk)
        cl = rng.integers(0, nk, size=len(idx))
    elif cfg['nk_modo'] == 'kmeans':
        cl = subestados_nc(ses.pot_grid[idx], nk)
    else:
        raise ValueError(cfg['nk_modo'])
    g = [idx[cl == k] for k in range(nk)]
    g = [x for x in g if len(x) > MIN_POR_CLASE]
    return g or [idx]


def _pares_y_filtros(ses: Sesion, m_tr: np.ndarray, cfg: dict):
    """CSP filters per band, in the channel space of the source recording.

    Class pairs per band: MI1 vs MI2, and MI1 vs NC_k and MI2 vs NC_k for each rest
    sub-state k, each pair giving 2*m_csp filters. With `recentrar`, the covariances
    are first whitened with the mean training covariance, and that whitening is
    returned in `blanq`. Returns (filters per band, whitening per band, feature names,
    size of each rest sub-state).

    Raises ConfigInviable if a class has MIN_POR_CLASE windows or fewer, or if the CSP
    problem is singular.
    """
    y = ses.g_y
    m1, m2 = m_tr & (y == -1.0), m_tr & (y == +1.0)
    mnc = m_tr & (y == 0.0)
    if cfg['nc_fuente'] == 'pausas':
        mnc = mnc & ses.g_pausa
    elif cfg['nc_fuente'] == 'entretrial':
        mnc = mnc & ~ses.g_pausa
    elif cfg['nc_fuente'] != 'todo':
        raise ValueError(cfg['nc_fuente'])
    if min(m1.sum(), m2.sum(), mnc.sum()) <= MIN_POR_CLASE:
        raise ConfigInviable(
            f'too few windows per class: MI1={int(m1.sum())} MI2={int(m2.sum())} '
            f'NC={int(mnc.sum())}')
    grupos = _grupos_nc(ses, mnc, cfg)

    lam, m = cfg['lam_cov'], cfg['m_csp']
    blanq = []
    filtros, etiquetas = [], []
    for b in range(ses.n_bandas):
        C = ses.Cg[b]
        if cfg['recentrar']:
            Rm = _inv_sqrt(C[m_tr].mean(0).astype(np.float64))
            trans = lambda M: Rm @ M @ Rm
        else:
            Rm = np.eye(ses.n_ch)
            trans = lambda M: M
        blanq.append(Rm)
        c1 = trans(C[m1].mean(0).astype(np.float64))
        c2 = trans(C[m2].mean(0).astype(np.float64))
        # With CAR the 59-channel covariance has rank 58, so with lam_cov = 0 the
        # generalized problem C1 w = lam (C1+C2) w has no solution (C1+C2 is singular).
        # Not a code error: that configuration cannot be evaluated.
        try:
            Wl, et = [csp(c1, c2, m, lam)], [f'b{b}_mi1mi2']
            for k, g in enumerate(grupos):
                cn = trans(C[g].mean(0).astype(np.float64))
                Wl.append(csp(c1, cn, m, lam)); et.append(f'b{b}_mi1nc{k}')
                Wl.append(csp(c2, cn, m, lam)); et.append(f'b{b}_mi2nc{k}')
        except np.linalg.LinAlgError as e:
            raise ConfigInviable(
                f'singular CSP in band {b} with lam_cov={lam} and car={cfg["car"]}: '
                f'{e}') from e
        filtros.append(np.concatenate(Wl, axis=0))
        etiquetas += [f'{e}_f{i}' for e in et for i in range(2 * m)]
    return filtros, blanq, etiquetas, [int(len(g)) for g in grupos]


def _potencia_por_ventana(Z: np.ndarray, sos, w: int, fines: np.ndarray,
                          chunk: int = 400) -> np.ndarray:
    """Power with the band-pass applied inside each window (filtfilt).

    This is the causal variant without phase delay: window [fin-w+1, fin] holds only
    past samples, so filtering it forward and backward uses no sample after the one
    being predicted. Each window is filtered separately, so cumulative sums cannot be
    used; windows are processed in chunks. Returns (len(fines), n_signals).
    """
    ar = np.arange(w)
    out = np.empty((len(fines), Z.shape[0]), np.float64)
    for a in range(0, len(fines), chunk):
        f = fines[a:a + chunk]
        idx = f[:, None] - (w - 1) + ar[None, :]
        V = Z[:, idx]                                   # (n_sig, m, w)
        V = V - V.mean(axis=2, keepdims=True)
        V = sosfiltfilt(sos, V, axis=-1)
        out[a:a + chunk] = (V ** 2).mean(axis=2).T
    return out


def _features(ses: Sesion, filtros, blanq, xb_list, fines: np.ndarray) -> np.ndarray:
    """Log-power of the CSP projections, for every band and every power window.

    Returns (len(fines), n_features), one block of columns per band and window.
    """
    bloques = []
    for b in range(ses.n_bandas):
        Z = (filtros[b] @ blanq[b]) @ xb_list[b].astype(np.float64)
        for w in ses.w_pot:
            P = (_potencia_por_ventana(Z, ses.sos[b], w, fines) if ses.por_ventana
                 else potencia_deslizante(Z, w, fines))
            bloques.append(np.log(np.maximum(P, 1e-30)))
    return np.concatenate(bloques, axis=1)


def _normalizar(F_tr, F_ap):
    """Linear scaling to [-1, 1] with the training minimum and maximum (as the winner
    did). Applied rows are clipped to [-1.5, 1.5]."""
    lo, hi = F_tr.min(0), F_tr.max(0)
    rng = np.maximum(hi - lo, 1e-12)
    g = lambda F: np.clip(2.0 * (F - lo) / rng - 1.0, -1.5, 1.5)
    return g(F_tr), g(F_ap)


def _blanqueo_destino(ses: Sesion, cfg: dict, blanq_tr, es_eval: bool,
                      m_destino: np.ndarray | None = None):
    """Whitening of the target recording (or of the cross-validation test fold).

    With `recentrar`, the target is whitened with its own mean covariance, estimated
    without labels, to correct the drift between recordings. In cross-validation the
    test fold also uses its own covariance, not the training one: otherwise the
    recentering would do nothing on the development bench and could not be evaluated
    there. Not causal: it uses the whole target (see `recentrar_causal_s`).
    """
    if not cfg['recentrar']:
        return blanq_tr
    if es_eval:
        return [_inv_sqrt(R) for R in ses.R_ev]
    assert m_destino is not None and m_destino.sum() > 10, 'target has too few grid points'
    return [_inv_sqrt(ses.Cg[b][m_destino].mean(0).astype(np.float64))
            for b in range(ses.n_bandas)]


def _covarianzas_del_destino(ses: Sesion, es_eval: bool,
                             m_destino: np.ndarray | None):
    """(per-grid-point covariances of the target, their positions in samples).

    Needed by the causal recentering, which only looks back. In evaluation the target is
    the evaluation recording (`C_ev`/`g_ev`), which requires the session to be built
    with `recentrar_causal_s > 0`; in cross-validation it is the test fold (`Cg`/`g_fin`
    masked).
    """
    if es_eval:
        assert ses.C_ev is not None, (
            'the session did not keep the per-point covariances of the evaluation: '
            'build it with `recentrar_causal_s > 0` in the cfg')
        return ses.C_ev, ses.g_ev
    assert m_destino is not None and m_destino.sum() > 10, 'target has too few grid points'
    return ([ses.Cg[b][m_destino] for b in range(ses.n_bandas)],
            ses.g_fin[m_destino])


# ============================================================
# Causal Euclidean recentering
# ============================================================
def bloques_causales(fines: np.ndarray, bloque_s: float) -> np.ndarray:
    """Block index of each output row, from its sample index only (no labels), as for
    `alineamiento_bloques_s`."""
    return (np.asarray(fines, np.int64)
            // int(round(float(bloque_s) * FS))).astype(np.int64)


def blanqueo_causal_por_bloques(C_lista, g, fines, bloque_s: float, blanq_inicial):
    """Strictly causal Euclidean whitening of the target, block by block.

    `C_lista[b]` holds the per-grid-point covariances of the target in band `b` and `g`
    their positions in samples (window ends). Block k covers samples [k*B, (k+1)*B) and
    is whitened with the mean covariance of the grid points at positions strictly below
    k*B. Every row of block k lies in [k*B, (k+1)*B), so no sample after the predicted
    one is used.

    Start-up: block 0 has no past and keeps `blanq_inicial`, the training whitening,
    which is what a live system would use when switched on (a global mean would look
    into the future). The same holds for any block with MIN_FILAS_ALINEAMIENTO past grid
    points or fewer.

    Returns ({block: (whitening per band, number of past grid points)}, block of each
    row).
    """
    g = np.asarray(g, np.int64)
    bl = bloques_causales(fines, bloque_s)
    ancho = int(round(float(bloque_s) * FS))
    n_bandas = len(C_lista)
    salida = {}
    for k in np.unique(bl):
        ini = int(k) * ancho
        pasado = g < ini
        if int(pasado.sum()) <= K.MIN_FILAS_ALINEAMIENTO:
            salida[int(k)] = (blanq_inicial, 0)
            continue
        salida[int(k)] = ([_inv_sqrt(C_lista[b][pasado].mean(0).astype(np.float64))
                           for b in range(n_bandas)], int(pasado.sum()))
    return salida, bl


def features_por_bloques(ses: 'Sesion', filtros, blanq_por_bloque, xb_list,
                         fines: np.ndarray, bloques: np.ndarray) -> np.ndarray:
    """`_features` with a different whitening per block.

    Rows of block k only need the signal between `min(fin) - w_max + 1` and `max(fin)`,
    so the signal is sliced per block instead of being projected whole once per block;
    the extra cost is one window of overlap per block. With a single block this is
    `_features` on a slice of the signal.
    """
    fines = np.asarray(fines, np.int64)
    bloques = np.asarray(bloques, np.int64)
    out = None
    for k in np.unique(bloques):
        sel = bloques == k
        f = fines[sel]
        ini = int(f.min()) - (ses.w_max - 1)
        assert ini >= 0, ('the first window of the block starts before sample 0: '
                          f'{ini}')
        fin = int(f.max()) + 1
        trozo = [np.asarray(x[:, ini:fin]) for x in xb_list]
        blanq = blanq_por_bloque[int(k)][0]
        Fk = _features(ses, filtros, blanq, trozo, f - ini)
        if out is None:
            out = np.empty((len(fines), Fk.shape[1]), np.float64)
        out[sel] = Fk
    return out


def _reducir_exacto(F_tr: np.ndarray, F_ap: np.ndarray):
    """Projection onto the row space of the (centered) training features. Exact.

    With p > n the rank of F_tr is at most n, so there are directions with no training
    variance. For a linear model with an L2 penalty the solution lies in the row space
    (representer theorem) and the penalty is invariant to orthogonal rotations, so
    training on F_tr or on F_tr @ V gives the same model and the same predictions, also
    out of sample. It keeps the cost of the head from growing with the number of
    features. Returns (reduced F_tr, reduced F_ap, rank).
    """
    mu = F_tr.mean(0)
    Xc = F_tr - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = int((S > S.max() * 1e-10).sum()) if S.size and S.max() > 0 else 1
    V = Vt[:k].T
    return Xc @ V, (F_ap - mu) @ V, k


MODOS_ENTRENAR = ('todo', 'puro', 'epoca_unica', 'mixto', 'blando')
CLASES3 = (-1.0, 0.0, 1.0)
# Modes that narrow the hard mask, and therefore also change the CSP filters: they
# reproduce a whole training recipe. 'mixto' and 'blando' instead add rows to the head
# and leave the filters unchanged.
MODOS_QUE_ESTRECHAN = ('puro', 'epoca_unica')


def mascara_estrecha(ses: 'Sesion', modo: str) -> np.ndarray | None:
    """Extra mask imposed by the training mode, or None if it imposes none.

    The single implementation, read by `_ajustar_predecir` and `_conjunto_cabeza`, so
    that the CSP filters and the head always use the same set.
    """
    if modo == 'puro':
        return ses.g_pura
    if modo == 'epoca_unica':
        return ses.g_epoca_unica
    return None


def _conjunto_cabeza(ses: Sesion, m_grid: np.ndarray, cfg: dict):
    """Training set of the head, from a grid mask.

    Returns (m_feat, filas, y, peso, y_fila):
      - `m_feat`  grid points to turn into features
      - `filas`   row of F_tr of each training sample, or None for the identity (one
                  sample per grid point)
      - `y`       target of each sample
      - `peso`    weight of each sample, or None if all weights are 1
      - `y_fila`  one hard label per row of F_tr, for computations made per row rather
                  than per sample: mutual-information selection needs a vector of the
                  length of F_tr, not of the expanded set.

    `filas=None` and `peso=None` together mean the plain path: one sample per grid point
    and no sample weights.

    The CSP filters are not affected: they are always estimated with the hard mask
    (defined labels, narrowed by 'puro' or 'epoca_unica'), so 'mixto' and 'blando'
    change only the training distribution of the head and not the feature space.
    """
    modo = cfg['entrenar']
    if modo not in MODOS_ENTRENAR:
        raise ValueError(modo)
    m_dura = m_grid & np.isfinite(ses.g_y)
    estrecha = mascara_estrecha(ses, modo)
    if estrecha is not None:
        m_dura = m_dura & estrecha
    if modo in ('todo',) + MODOS_QUE_ESTRECHAN:
        y = ses.g_y[m_dura]
        fac = K.pesos_por_coste(y, cfg['costo_signo'])
        peso = (None if fac is None
                else K.normalizar_al_total(fac, float(len(y))))
        return m_dura, None, y, peso, y

    # --- 'mixto' and 'blando': the transition windows enter as well --------------
    # They are the grid windows whose end label is undefined: NaN in the 1 s after each
    # cue onset and offset, the official convention.
    m_feat = m_grid.copy()
    idx = np.where(m_feat)[0]
    es_dura = m_dura[idx]
    j_pura = np.where(es_dura)[0]
    j_extra = np.where(~es_dura)[0]
    fr = ses.g_frac[idx[j_extra]]
    pe = float(cfg['blando_peso'])
    # Hard label of the majority state of each transition window. Ties go to rest,
    # deterministically (the columns are reordered to (0, -1, +1) before the argmax). It
    # is the target of 'mixto' and the per-row label needed by feature selection.
    k_may = np.array([1, 0, 2])[np.argmax(fr[:, [1, 0, 2]], axis=1)]
    y_may = np.asarray(CLASES3, np.float64)[k_may]
    y_fila = np.empty(len(idx), np.float64)
    y_fila[j_pura] = ses.g_y[idx[j_pura]]
    y_fila[j_extra] = y_may

    if modo == 'blando':
        # one row per (window, state present), with the fraction as weight: a soft
        # target for a classifier that only accepts hard labels and weights. The
        # fractions sum to 1, so each transition window contributes the same total
        # weight `blando_peso` as a hard row in 'mixto'.
        jj, kk = np.nonzero(fr > 0.0)
        f_extra = j_extra[jj]
        y_extra = np.asarray(CLASES3, np.float64)[kk]
        w_extra = fr[jj, kk] * pe
    else:
        f_extra = j_extra
        y_extra = y_may
        w_extra = np.full(len(j_extra), pe, np.float64)

    filas = np.concatenate([j_pura, f_extra]).astype(np.int64)
    y = np.concatenate([ses.g_y[idx[j_pura]], y_extra])
    peso = np.concatenate([np.ones(len(j_pura)), w_extra])
    assert np.isfinite(y).all(), 'undefined target in the soft training set'
    # The total weight is fixed before the cost factor is applied and kept afterwards,
    # so `C` regularizes the same way in every mode and 'mixto' and 'blando' stay
    # matched in weight.
    total = float(peso.sum())
    fac = K.pesos_por_coste(y, cfg['costo_signo'])
    if fac is not None:
        peso = K.normalizar_al_total(peso * fac, total)
    return m_feat, filas, y, peso, y_fila


def _ajustar_predecir(ses: Sesion, m_tr: np.ndarray, cfg: dict,
                      xb_ap_list, fines_ap: np.ndarray, es_eval: bool = False,
                      prior_objetivo: np.ndarray | None = None,
                      m_destino: np.ndarray | None = None):
    """Train on the masked calibration grid and predict at the window ends `fines_ap`.

    `xb_ap_list` holds the per-band signals of the target, `es_eval` says whether the
    target is the evaluation recording, `prior_objetivo` is the class prior
    (-1, 0, +1) of the prior correction, and `m_destino` is the grid mask of the test
    fold in cross-validation (used by the recentering).

    Grid points whose label is NaN (the 1 s after each cue onset and offset, which the
    official metric excludes) have no target and cannot train the CSP filters; they are
    dropped here, in one place, for both the development and the evaluation path.

    Returns (prediction per row of `fines_ap`, metadata dict).
    """
    m_dura = m_tr & np.isfinite(ses.g_y)
    estrecha = mascara_estrecha(ses, cfg['entrenar'])
    if estrecha is not None:
        m_dura = m_dura & estrecha
    # `m_feat` can be larger than `m_dura` ('mixto' and 'blando'), but the CSP filters
    # are still estimated with the hard mask: those modes act on the head, not on the
    # feature space.
    m_feat, filas_tr, y_tr, peso_tr, y_fila = _conjunto_cabeza(ses, m_tr, cfg)
    m_tr = m_dura

    filtros, blanq_tr, etiquetas, tam_nc = _pares_y_filtros(ses, m_tr, cfg)
    n_feat_prev = len(etiquetas) * len(ses.w_pot)
    if n_feat_prev > MAX_FEATURES:
        raise ConfigInviable(f'{n_feat_prev} features > cap {MAX_FEATURES}')
    F_tr = _features(ses, filtros, blanq_tr, ses.xb, ses.g_fin[m_feat])

    # --- Euclidean whitening of the target -----------------------------------------
    # With `recentrar_causal_s = 0` the target is whitened with its mean covariance over
    # the whole recording, which uses the future. With > 0 it is split into blocks and
    # each block uses only the earlier ones.
    causal_rec = float(cfg['recentrar_causal_s'])
    diag_rec = None
    if cfg['recentrar'] and causal_rec > 0.0:
        C_dst, g_dst = _covarianzas_del_destino(ses, es_eval, m_destino)
        blanq_por_bloque, bl_rec = blanqueo_causal_por_bloques(
            C_dst, g_dst, fines_ap, causal_rec, blanq_tr)
        F_ap = features_por_bloques(ses, filtros, blanq_por_bloque, xb_ap_list,
                                    fines_ap, bl_rec)
        diag_rec = dict(
            n_tramos_recentrado=int(len(blanq_por_bloque)),
            n_tramos_con_arranque=int(sum(1 for v in blanq_por_bloque.values()
                                          if v[1] == 0)),
            puntos_de_pasado_por_tramo=[int(v[1]) for _, v in
                                        sorted(blanq_por_bloque.items())])
    else:
        blanq_ap = _blanqueo_destino(ses, cfg, blanq_tr, es_eval, m_destino)
        F_ap = _features(ses, filtros, blanq_ap, xb_ap_list, fines_ap)

    # The alignment comes before selection and normalization: it acts on the
    # log-powers, where a mismatch between recordings is a shift. Blocks, if any, come
    # from the sample index of each row, never from labels; with
    # `alineamiento_bloques_s=0` there are none.
    #
    # `alineamiento_causal_s > 0` replaces the blocks with a causal running mean. The
    # two are mutually exclusive (`K.alinear_features` asserts it); the conflict is
    # rejected here first, with an explicit message.
    bl = None
    taps = None
    if float(cfg['alineamiento_bloques_s']) > 0.0:
        bl = (np.asarray(fines_ap, np.int64)
              // int(round(float(cfg['alineamiento_bloques_s']) * FS)))
    if float(cfg['alineamiento_causal_s']) > 0.0:
        if bl is not None:
            raise ConfigInviable(
                'alineamiento_bloques_s and alineamiento_causal_s are mutually '
                'exclusive: either clock blocks or a causal running mean, not both')
        # seconds -> rows, with the actual step of the output grid
        per = K.periodo_de_fila(fines_ap)
        taps = int(round(float(cfg['alineamiento_causal_s']) * FS / per))
        assert taps > 0, (f'alineamiento_causal_s={cfg["alineamiento_causal_s"]} with '
                          f'a row period of {per} gives {taps} rows')
    F_ap = K.alinear_features(F_tr, F_ap, cfg['alineamiento'], bloques=bl,
                              causal_taps=taps)

    sel = None
    if cfg['n_feats'] and cfg['n_feats'] < F_tr.shape[1]:
        # `y_fila`, not `y_tr`: mutual information is computed per column of F_tr and
        # needs one label per row. In the modes without expansion both are the same
        # array.
        mi = mi_features(F_tr, y_fila)
        sel = np.sort(np.argsort(mi)[::-1][:int(cfg['n_feats'])])
        F_tr, F_ap = F_tr[:, sel], F_ap[:, sel]

    # Normalization and reduction on the unique rows, expansion afterwards:
    #  - the reduction is equally exact (repeated rows do not change the row space),
    #    and an SVD of the expanded set would cost more for the same result;
    #  - 'mixto' and 'blando' get the same feature matrix, so the only difference
    #    between them is the target, which makes 'mixto' a true capacity control.
    F_tr, F_ap = _normalizar(F_tr, F_ap)
    # Only for linear models with an L2 penalty, where the reduction is exact. The GRNN
    # and the shrinkage LDA are not invariant to dropping directions.
    k_red = None
    if (cfg['regresor'] in K.REGRESORES_L2_LINEALES
            and F_tr.shape[1] > F_tr.shape[0]):
        F_tr, F_ap, k_red = _reducir_exacto(F_tr, F_ap)
    n_filas_unicas = int(F_tr.shape[0])
    if filas_tr is not None:
        F_tr = F_tr[filas_tr]
    kw = {'ridge': dict(alpha=cfg['alpha']),
          'multinomial': dict(C=cfg['C']),
          'ordinal': dict(C=cfg['C']),
          'mse': dict(C=cfg['C']),
          'lda': dict(shrink=cfg['shrink'])}.get(cfg['regresor'], {})
    if cfg['prior'] == 'objetivo' and cfg['regresor'] in K.REGRESORES_POSTERIOR:
        kw['prior_objetivo'] = prior_objetivo
    elif cfg['prior'] not in ('ninguno', 'objetivo'):
        raise ValueError(cfg['prior'])
    reg = hacer_regresor(cfg['regresor'], **kw)
    if peso_tr is None:
        reg = reg.fit(F_tr, y_tr)
    else:
        # A regressor that does not accept sample weights must not receive them
        # silently: it would run something other than what the configuration says.
        if cfg['regresor'] not in K.REGRESORES_POSTERIOR:
            raise ConfigInviable(
                f'{cfg["regresor"]} does not accept sample weights; '
                f'entrenar={cfg["entrenar"]} costo_signo={cfg["costo_signo"]}')
        reg = reg.fit(F_tr, y_tr, sample_weight=peso_tr)
    # `p_mi` exists only if the head is a 3-class posterior. It is returned as extra
    # information; `partes` gives the same prediction as `predict`.
    if hasattr(reg, 'partes'):
        pred, p_mi = reg.partes(F_ap)
    else:
        pred, p_mi = reg.predict(F_ap), None
    return pred, dict(
        p_mi=p_mi,
        n_features=int(n_feat_prev), n_dim_modelo=int(F_tr.shape[1]),
        n_train=int(m_tr.sum()), rango_reduccion=k_red,
        tam_subestados_nc=tam_nc, n_pares=1 + 2 * len(tam_nc),
        n_seleccionadas=(None if sel is None else int(len(sel))),
        # recorded so the output shows whether the causal alignment was applied
        alineamiento_causal_taps=taps,
        **(diag_rec or {}),
        **diag_cabeza(reg, n_filas_unicas, y_tr, peso_tr))


def diag_cabeza(reg, n_filas_unicas: int, y_tr, peso_tr) -> dict:
    """Head diagnostics recorded in the output.

    `n_filas_cabeza` > `n_filas_unicas` means that the soft training set was actually
    expanded. For the heads that expose them, the initial and final loss of the
    optimizer, its iterations and its convergence flag are added; equal initial and
    final losses would mean that the head never moved from its starting point.
    """
    d = dict(n_filas_unicas=int(n_filas_unicas), n_filas_cabeza=int(len(y_tr)),
             peso_total=(None if peso_tr is None else float(np.sum(peso_tr))))
    m = getattr(reg, 'm_', None)
    for k in ('perdida_inicial_', 'perdida_final_', 'n_iter_', 'convergio_'):
        if not hasattr(m, k):
            continue
        v = getattr(m, k)
        if isinstance(v, (bool, np.bool_)):
            v = bool(v)
        else:
            # `LogisticRegression.n_iter_` is an array while the other heads store
            # scalars; the maximum keeps the field a plain number.
            v = float(np.max(np.asarray(v, np.float64)))
        d['cabeza_' + k.rstrip('_')] = v
    return d


def _prior_objetivo(p_mi: float) -> np.ndarray:
    """Class prior (-1, 0, +1) from the fraction of time in MI."""
    return np.array([p_mi / 2.0, 1.0 - p_mi, p_mi / 2.0])


# ============================================================
# Cross-validation inside the calibration recording (development bench)
# ============================================================
def _limites(ses: Sesion, n_folds: int) -> list[int]:
    """Fold boundaries in samples: `n_folds` contiguous blocks cut at cue onsets."""
    b = [int(ses.mrk_pos[i]) for i in
         np.linspace(0, len(ses.mrk_pos), n_folds + 1)[1:-1].astype(int)]
    return [0] + b + [ses.T]


def correr_cv(sujeto: str, cfg: dict, n_folds: int = 5,
              ses: Sesion | None = None, devolver: bool = False) -> dict:
    """Temporal cross-validation inside the calibration recording of `sujeto`.

    The recording is cut at cue onsets into `n_folds` contiguous blocks. For each block
    the model is trained on the grid windows that end outside it and predicts every
    0.1 s inside it, starting `w_max - 1` samples after the block start so that no test
    window reaches into the previous block. The post-processing is fitted on the
    out-of-fold predictions of the other blocks. The MSE is computed on the predicted
    samples with a defined label of the calibration stream
    (`components.stream_calibracion`).

    `ses` reuses a prepared session. With `devolver=True` the result also holds the
    streams (`_salida`, `_crudo`, `_mascara`, `_y`, `_p_mi`, `_limites`).
    """
    ses = ses or Sesion(sujeto, cfg)
    T, wmax = ses.T, ses.w_max
    limites = _limites(ses, n_folds)
    pri = _prior_objetivo(_prior_de(ses.y)[1])

    salida = np.zeros(T, np.float64)
    p_mi_st = np.zeros(T, np.float64)
    mascara = np.zeros(T, bool)
    info = []
    for f in range(n_folds):
        a, b = limites[f], limites[f + 1]
        m_tr = (ses.g_fin < a) | (ses.g_fin >= b)
        fin_ap = np.arange(a + wmax - 1, b, STRIDE_SALIDA, dtype=np.int64)
        if len(fin_ap) < 10:
            continue
        r, meta = _ajustar_predecir(ses, m_tr, cfg, ses.xb, fin_ap,
                                    prior_objetivo=pri, m_destino=~m_tr)
        # `p_mi` is a per-window array: it is kept out of `info`, which must stay
        # JSON-serializable, and expanded separately.
        pm = meta.pop('p_mi', None)
        ext = A.expandir_causal(r, fin_ap, T, fs_in=FS, fs_out=FS, relleno=0.0)
        seg = slice(a + wmax - 1, b)
        salida[seg], mascara[seg] = ext[seg], True
        if pm is not None:
            p_mi_st[seg] = A.expandir_causal(pm, fin_ap, T, fs_in=FS, fs_out=FS,
                                             relleno=0.0)[seg]
        info.append(dict(fold=f, ini=a, fin=b, n_ventanas=int(len(fin_ap)), **meta))

    post_out = (_post_cv(ses, salida, mascara, cfg, limites, n_folds)
                if cfg['post'] != 'ninguno' else None)

    bruto = mse_stream(np.clip(salida, -cfg['recorte'], cfg['recorte']),
                       ses.y, mascara)
    res = dict(sujeto=sujeto, cfg=cfg, n_folds=n_folds, folds=info,
               mse=bruto['mse'], mse_cero=bruto['mse_cero'], n=bruto['n'],
               mse_sin_post=bruto['mse'])
    if post_out is not None:
        rp = mse_stream(np.clip(post_out, -cfg['recorte'], cfg['recorte']),
                        ses.y, mascara)
        res.update(mse=rp['mse'], mse_post=rp['mse'])
    res['reduccion_relativa'] = float(1.0 - res['mse'] / res['mse_cero'])
    o_final = np.clip(post_out if post_out is not None else salida,
                      -cfg['recorte'], cfg['recorte'])
    # The diagnostic is always computed: it is a pure function of the final output,
    # does not change the MSE, and tells whether a change acts on detection or elsewhere.
    res['diag'] = diagnostico(o_final, ses.y, mascara)
    if devolver:
        res.update(_salida=o_final, _crudo=salida, _mascara=mascara, _y=ses.y,
                   _p_mi=p_mi_st, _limites=limites)
    return res


def _post_cv(ses: Sesion, crudo: np.ndarray, mascara: np.ndarray, cfg: dict,
             limites: list[int], n_folds: int) -> np.ndarray:
    """Fit the post-processing on the training folds and apply it to the test fold.

    `crudo` already holds out-of-fold predictions, so the post-processing never sees
    in-sample (optimistic) predictions of the model.
    """
    out = np.zeros(len(crudo), np.float64)
    kw = _kw_post(cfg, *_prior_de(ses.y))
    for f in range(n_folds):
        a, b = limites[f], limites[f + 1]
        m_tr = mascara.copy()
        m_tr[a:b] = False
        if m_tr.sum() < 1000:
            continue
        out[a:b] = hacer_post(cfg['post'], **kw).fit(
            crudo, ses.y, m_tr).predict(crudo)[a:b]
    return out


# ============================================================
# Evaluation
# ============================================================
def correr_eval(sujeto: str, cfg: dict, ses: Sesion | None = None,
                n_folds_post: int = 5, perturbacion: dict | None = None) -> dict:
    """Train on the whole calibration recording and predict the continuous evaluation.

    Outputs every 0.1 s from a causal window ending at the predicted sample, held until
    the next output. The post-processing is fitted on out-of-fold predictions of the
    calibration recording (`n_folds_post` folds), never on in-sample predictions, which
    would be overfitted and make the post-processing overconfident. Class and HMM priors
    come from the public description of the dataset (`HMM_PMI_PUBLICADA`,
    `HMM_DUR_PUBLICADA`).

    `perturbacion` adds an impulse to the evaluation signal (causality test, see
    `Sesion`). With a perturbation the labels are not read and no MSE is computed
    (`mse=None`): the MSE of a system fed a modified signal means nothing.
    """
    ses = ses or Sesion(sujeto, cfg, con_eval=True, perturbacion=perturbacion)
    assert ses.xb_ev is not None, 'the session has no evaluation signal'
    fin_ap = np.arange(ses.w_max - 1, ses.T_ev, STRIDE_SALIDA, dtype=np.int64)
    r_ev, meta = _ajustar_predecir(
        ses, np.ones(len(ses.g_fin), bool), cfg, ses.xb_ev, fin_ap, es_eval=True,
        prior_objetivo=_prior_objetivo(HMM_PMI_PUBLICADA))
    # `p_mi` is a per-window array and `meta` goes into a JSON record: it is taken out
    # here and expanded separately, as in cross-validation.
    pm = meta.pop('p_mi', None)
    salida = A.expandir_causal(r_ev, fin_ap, ses.T_ev, fs_in=FS, fs_out=FS,
                               relleno=0.0)
    p_mi_st = (A.expandir_causal(pm, fin_ap, ses.T_ev, fs_in=FS, fs_out=FS,
                                 relleno=0.0) if pm is not None else None)

    if cfg['post'] != 'ninguno':
        crudo, masc = _crudo_fuera_de_fold(ses, cfg, n_folds_post)
        salida = hacer_post(
            cfg['post'], **_kw_post(cfg, HMM_DUR_PUBLICADA, HMM_PMI_PUBLICADA)
        ).fit(crudo, ses.y, masc).predict(salida)

    salida = np.clip(salida, -cfg['recorte'], cfg['recorte'])
    if perturbacion is not None:
        return dict(sujeto=sujeto, cfg=cfg, salida=salida, p_mi=p_mi_st,
                    perturbacion=dict(perturbacion), mse=None,
                    nota='perturbed signal: labels not read, MSE not computed',
                    **meta)
    true_y = A.cargar_true_y(sujeto)
    r = A.mse_oficial(salida, true_y)
    return dict(sujeto=sujeto, cfg=cfg, salida=salida, p_mi=p_mi_st,
                true_y_len=int(len(true_y)), mse=r['mse'],
                n_evaluadas=r['n_evaluadas'], mse_cero=A.mse_cero(true_y), **meta)


def correr_eval_ensamblado(sujeto: str, cfgs: list, idx_post: int = 0,
                           n_folds_post: int = 5,
                           perturbacion: dict | None = None) -> dict:
    """Evaluation with several configurations averaged on their raw outputs.

    The raw outputs (before post-processing) of all members are averaged, and a single
    post-processing, that of member `idx_post`, is then applied. Averaging after the
    post-processing would be a different system. With one member this reduces to
    `correr_eval`.
    """
    assert len(cfgs) >= 1, 'an ensemble needs at least one member'
    crudas, metas = [], []
    ses_post = None
    for i, cfg in enumerate(cfgs):
        ses = Sesion(sujeto, cfg, con_eval=True, perturbacion=perturbacion)
        fin_ap = np.arange(ses.w_max - 1, ses.T_ev, STRIDE_SALIDA, dtype=np.int64)
        r_ev, meta = _ajustar_predecir(
            ses, np.ones(len(ses.g_fin), bool), cfg, ses.xb_ev, fin_ap, es_eval=True,
            prior_objetivo=_prior_objetivo(HMM_PMI_PUBLICADA))
        meta.pop('p_mi', None)
        crudas.append(A.expandir_causal(r_ev, fin_ap, ses.T_ev, fs_in=FS, fs_out=FS,
                                        relleno=0.0))
        metas.append(meta)
        if i == idx_post:
            ses_post, cfg_post = ses, cfg
        else:
            del ses                       # sessions take several GB: do not keep them
    n = min(len(c) for c in crudas)
    assert all(len(c) == n for c in crudas), 'the members differ in output length'
    salida = np.mean(np.stack(crudas, 0), axis=0)

    if cfg_post['post'] != 'ninguno':
        crudo, masc = _crudo_fuera_de_fold(ses_post, cfg_post, n_folds_post)
        salida = hacer_post(
            cfg_post['post'],
            **_kw_post(cfg_post, HMM_DUR_PUBLICADA, HMM_PMI_PUBLICADA)
        ).fit(crudo, ses_post.y, masc).predict(salida)
    salida = np.clip(salida, -cfg_post['recorte'], cfg_post['recorte'])
    m0 = dict(metas[idx_post])
    m0['n_miembros'] = len(cfgs)
    m0['hash_miembros'] = [K.hash_cfg(c) for c in cfgs]
    if perturbacion is not None:
        # as in `correr_eval`: with a perturbed signal `true_y` is not read
        return dict(sujeto=sujeto, cfg=cfg_post, salida=salida, p_mi=None,
                    perturbacion=dict(perturbacion), mse=None,
                    nota='perturbed signal: labels not read, MSE not computed',
                    **m0)
    true_y = A.cargar_true_y(sujeto)
    r = A.mse_oficial(salida, true_y)
    return dict(sujeto=sujeto, cfg=cfg_post, salida=salida, p_mi=None,
                true_y_len=int(len(true_y)), mse=r['mse'],
                n_evaluadas=r['n_evaluadas'], mse_cero=A.mse_cero(true_y), **m0)


def _crudo_fuera_de_fold(ses: Sesion, cfg: dict, n_folds: int):
    """Out-of-fold raw predictions on the calibration recording (100 Hz stream and its
    mask), used to fit the post-processing of the evaluation."""
    limites = _limites(ses, n_folds)
    crudo = np.zeros(ses.T, np.float64)
    masc = np.zeros(ses.T, bool)
    pri = _prior_objetivo(_prior_de(ses.y)[1])
    for f in range(n_folds):
        a, b = limites[f], limites[f + 1]
        m_tr = (ses.g_fin < a) | (ses.g_fin >= b)
        fin_ap = np.arange(a + ses.w_max - 1, b, STRIDE_SALIDA, dtype=np.int64)
        if len(fin_ap) < 10:
            continue
        r, _ = _ajustar_predecir(ses, m_tr, cfg, ses.xb, fin_ap, prior_objetivo=pri,
                                 m_destino=~m_tr)
        ext = A.expandir_causal(r, fin_ap, ses.T, fs_in=FS, fs_out=FS, relleno=0.0)
        seg = slice(a + ses.w_max - 1, b)
        crudo[seg], masc[seg] = ext[seg], True
    return crudo, masc


# ============================================================
# Diagnostics
# ============================================================
def diagnostico(o: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    """Decompose the MSE to see where it is lost: detection, direction or amplitude.

    Identity used: MSE = E[y^2] - E[2*o*y - o^2], so the gain over the silent output is
    exactly E[2*o*y - o^2] and can be split by state. `auc_deteccion` is the AUC of |o|
    for MI vs rest, `acierto_direccion` the sign agreement on MI samples, and
    `escala_optima` the scalar gain of the output that minimizes the MSE, with the gain
    over silence it would reach.
    """
    from scipy.stats import rankdata
    m = mask & np.isfinite(y)
    o, y = np.asarray(o, np.float64)[m], np.asarray(y, np.float64)[m]
    es_mi = y != 0.0
    g = 2.0 * o * y - o ** 2
    oo, oy = float(np.mean(o ** 2)), float(np.mean(o * y))
    r = rankdata(np.abs(o))
    n1, n0 = int(es_mi.sum()), int((~es_mi).sum())
    auc = float((r[es_mi].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan
    return dict(
        mse=float(np.mean((o - y) ** 2)), mse_cero=float(np.mean(y ** 2)),
        ganancia=float(np.mean(g)), ganancia_en_mi=float(g[es_mi].mean()),
        ganancia_en_reposo=float(g[~es_mi].mean()), frac_mi=float(es_mi.mean()),
        rms_salida=float(np.sqrt(oo)),
        abs_medio_en_mi=float(np.abs(o[es_mi]).mean()),
        abs_medio_en_reposo=float(np.abs(o[~es_mi]).mean()),
        escala_optima=float(oy / oo) if oo > 0 else 0.0,
        ganancia_con_escala_optima=float(oy ** 2 / oo) if oo > 0 else 0.0,
        auc_deteccion=auc,
        acierto_direccion=float((np.sign(o[es_mi]) == np.sign(y[es_mi])).mean()),
        n=int(m.sum()))
