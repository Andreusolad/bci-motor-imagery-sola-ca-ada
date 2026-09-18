"""Temporal evidence accumulation.

Single windows are noisy, so no command is issued from one window: evidence is accumulated
over consecutive windows and a command is emitted only when it is sustained, at the cost
of added latency.

Four strategies:
  - 'dwell' : the same accepted control class must be held for dwell_ms.
  - 'nofm'  : at least N of the last M windows are accepted with the same control class.
  - 'ema'   : exponential moving average of the probabilities; a command is emitted when
              the smoothed control probability is >= emit_thr and above P(IDLE) for the
              same class during dwell_ms.
  - 'bayes' : recursive Bayesian filter over {LH, RH, IDLE} with a sticky transition
              matrix (staying in the same state is favoured); same emission rule as 'ema'
              on the posterior.

'dwell' and 'nofm' use the gate decision (is_control, control_class); 'ema' and 'bayes'
use only the calibrated probabilities. All strategies apply a refractory period after
each command. Durations are converted to window steps of stride_ms.
Input: one GateDecision per window. Output: a command class (0/1) or None.
"""
from __future__ import annotations
from collections import deque

import numpy as np

from . import config as C
from .gate import GateDecision


# ============================================================
# Time utilities (ms -> window steps)
# ============================================================
def ms_to_steps(ms: float, stride_ms: float = C.STRIDE_MS) -> int:
    return max(1, int(round(ms / stride_ms)))


# ============================================================
# Accumulator
# ============================================================
class EvidenceAccumulator:
    """Turns a stream of GateDecision into discrete, stable commands."""

    def __init__(self, strategy: str = 'bayes',
                 dwell_ms: float = C.DWELL_MS, refractory_ms: float = C.REFRACTORY_MS,
                 emit_thr: float = 0.60, ema_alpha: float = C.EMA_ALPHA,
                 nofm: tuple[int, int] = (4, 6), hmm_stay: float = C.HMM_STAY,
                 stride_ms: float = C.STRIDE_MS):
        self.strategy = strategy
        self.dwell = ms_to_steps(dwell_ms, stride_ms)
        self.refractory = ms_to_steps(refractory_ms, stride_ms)
        self.emit_thr = emit_thr
        self.ema_alpha = ema_alpha
        self.n_need, self.m_win = nofm
        self.stride_ms = stride_ms

        # sticky 3x3 transition matrix (rows = previous state)
        off = (1.0 - hmm_stay) / (C.N_CLASES - 1)
        self.trans = np.full((C.N_CLASES, C.N_CLASES), off)
        np.fill_diagonal(self.trans, hmm_stay)

        self.reset()

    def reset(self):
        self._streak_cls = -1
        self._streak_len = 0
        self._recent = deque(maxlen=self.m_win)
        self._ema = np.ones(C.N_CLASES) / C.N_CLASES
        self._post = np.ones(C.N_CLASES) / C.N_CLASES
        self._cooldown = 0
        self._ema_streak = 0
        self._bayes_streak = 0

    # ---- main API ----
    def update(self, dec: GateDecision) -> int | None:
        """Process one window. Returns the emitted command class (0/1) or None."""
        if self._cooldown > 0:
            self._cooldown -= 1
            # keep the EMA and posterior up to date during the refractory period
            self._advance_state(dec)
            return None
        cmd = getattr(self, f'_step_{self.strategy}')(dec)
        if cmd is not None:
            self._cooldown = self.refractory
            self._reset_streaks()
        return cmd

    # ---- state update (EMA / Bayes) during the refractory period ----
    def _advance_state(self, dec: GateDecision):
        self._ema = self.ema_alpha * dec.probs + (1 - self.ema_alpha) * self._ema
        prior = self.trans.T @ self._post
        post = prior * dec.probs
        self._post = post / (post.sum() + 1e-12)

    def _reset_streaks(self):
        self._streak_cls = -1; self._streak_len = 0
        self._ema_streak = 0; self._bayes_streak = 0
        self._recent.clear()

    # ---- dwell strategy ----
    def _step_dwell(self, dec: GateDecision) -> int | None:
        if dec.is_control:
            if dec.control_class == self._streak_cls:
                self._streak_len += 1
            else:
                self._streak_cls = dec.control_class; self._streak_len = 1
        else:
            self._streak_cls = -1; self._streak_len = 0
        if self._streak_len >= self.dwell:
            return self._streak_cls
        return None

    # ---- N-of-M strategy ----
    def _step_nofm(self, dec: GateDecision) -> int | None:
        self._recent.append(dec.control_class if dec.is_control else -1)
        if len(self._recent) < self.m_win:
            return None
        arr = np.array(self._recent)
        for cls in (0, 1):
            if int((arr == cls).sum()) >= self.n_need:
                return cls
        return None

    # ---- EMA strategy ----
    def _step_ema(self, dec: GateDecision) -> int | None:
        self._ema = self.ema_alpha * dec.probs + (1 - self.ema_alpha) * self._ema
        ctrl = np.delete(self._ema, C.IDLE_ID)
        cls = int(np.argmax(ctrl))
        if self._ema[C.IDLE_ID] < ctrl.max() and ctrl.max() >= self.emit_thr:
            if cls == getattr(self, '_ema_cls', -1):
                self._ema_streak += 1
            else:
                self._ema_cls = cls; self._ema_streak = 1
            if self._ema_streak >= self.dwell:
                return cls
        else:
            self._ema_streak = 0
        return None

    # ---- sticky Bayesian strategy ----
    def _step_bayes(self, dec: GateDecision) -> int | None:
        prior = self.trans.T @ self._post              # predict
        post = prior * dec.probs                        # update with the likelihood
        self._post = post / (post.sum() + 1e-12)
        ctrl = np.delete(self._post, C.IDLE_ID)
        cls = int(np.argmax(ctrl))
        if self._post[C.IDLE_ID] < ctrl.max() and ctrl.max() >= self.emit_thr:
            if cls == getattr(self, '_bayes_cls', -1):
                self._bayes_streak += 1
            else:
                self._bayes_cls = cls; self._bayes_streak = 1
            if self._bayes_streak >= self.dwell:
                return cls
        else:
            self._bayes_streak = 0
        return None
