"""Two-stage gate: intentional control (IC) vs no control (NC).

Combines the idle class, the confidence threshold and the OOD detectors into one decision
per window:

  stage 1 (state detection): decide whether there is an intention to control now
     - P(IDLE) above idle_thr                        -> NC
     - OOD score above the detector threshold        -> NC
     - command confidence below conf_thr             -> NC
  stage 2 (classification): for IC windows, choose LH or RH
     - argmax over the control classes of the calibrated softmax.

Each check can be switched on or off. The rule is conservative, since a wrong command is
worse than no command: a single check reporting NC rejects the window.

Returns a GateDecision (the input of the temporal accumulator) with the final label
(LH/RH/NC), the proposed control class, the calibrated probabilities and the reasons for
a rejection.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .calibration import TemperatureScaler, get_logits


# ============================================================
# Result of a window decision
# ============================================================
@dataclass
class GateDecision:
    label: int                       # 0=LH, 1=RH, -1=NC (no control)
    control_class: int               # proposed argmax(LH, RH), even if rejected
    probs: np.ndarray                # calibrated softmax [P(LH), P(RH), P(IDLE)]
    is_control: bool                 # True if the window passed the gate (IC)
    reasons: dict = field(default_factory=dict)   # why NC (idle/low_conf/ood_*)

    @property
    def conf_control(self) -> float:
        p = np.delete(self.probs, C.IDLE_ID)
        return float(p.max())


# ============================================================
# Gate
# ============================================================
class ControlStateGate:
    """Combines the idle class, confidence and OOD checks into an IC/NC decision.

    Switches: use_idle / use_conf / use_ood_riemann / use_ood_mahal. An OOD check is active
    only if its detector is given and has a fitted threshold_.
    Thresholds: idle_thr and conf_thr default to config (IDLE_PROB_THR, CONF_THR); the OOD
    thresholds are read from the detectors.
    """

    def __init__(self, model, scaler: TemperatureScaler | None = None,
                 riemann_ood=None, mahal_ood=None,
                 use_idle=True, use_conf=True, use_ood_riemann=True, use_ood_mahal=False,
                 idle_thr=None, conf_thr=None):
        self.model = model
        self.scaler = scaler
        self.riemann_ood = riemann_ood
        self.mahal_ood = mahal_ood
        self.use_idle = use_idle
        self.use_conf = use_conf
        self.use_ood_riemann = use_ood_riemann and riemann_ood is not None
        self.use_ood_mahal = use_ood_mahal and mahal_ood is not None
        self.idle_thr = C.IDLE_PROB_THR if idle_thr is None else idle_thr
        self.conf_thr = C.CONF_THR if conf_thr is None else conf_thr

    # ---- calibrated probabilities ----
    def _probs(self, X: np.ndarray) -> np.ndarray:
        logits = get_logits(self.model, X)
        if self.scaler is not None:
            return self.scaler.probs(logits)
        # no scaler: plain softmax
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    # ---- decision for a batch of windows ----
    def decide_batch(self, X: np.ndarray) -> list[GateDecision]:
        X = X if X.ndim == 3 else X[None]
        probs = self._probs(X)
        s_rie = self.riemann_ood.score(X) if self.use_ood_riemann else None
        s_mah = self.mahal_ood.score(X) if self.use_ood_mahal else None
        out = []
        for i in range(len(X)):
            p = probs[i]
            ctrl_p = np.delete(p, C.IDLE_ID)
            control_class = int(np.argmax(ctrl_p))          # 0 or 1 (LH/RH)
            reasons = {}
            ic = True
            if self.use_idle and p[C.IDLE_ID] > self.idle_thr:
                ic = False; reasons['idle'] = float(p[C.IDLE_ID])
            if self.use_conf and float(ctrl_p.max()) < self.conf_thr:
                ic = False; reasons['low_conf'] = float(ctrl_p.max())
            if self.use_ood_riemann and self.riemann_ood.threshold_ is not None \
                    and s_rie[i] > self.riemann_ood.threshold_:
                ic = False; reasons['ood_riemann'] = float(s_rie[i])
            if self.use_ood_mahal and self.mahal_ood.threshold_ is not None \
                    and s_mah[i] > self.mahal_ood.threshold_:
                ic = False; reasons['ood_mahal'] = float(s_mah[i])
            out.append(GateDecision(
                label=control_class if ic else -1,
                control_class=control_class, probs=p, is_control=ic, reasons=reasons))
        return out

    def decide(self, x: np.ndarray) -> GateDecision:
        """Decision for a single window (n_ch, w)."""
        return self.decide_batch(x[None])[0]


# ============================================================
# Gate evaluation (3-state confusion)
# ============================================================
def evaluar_gate(gate: ControlStateGate, X: np.ndarray, y: np.ndarray) -> dict:
    """Asynchronous-use metrics on labelled windows.

    - acc_control: LH/RH accuracy on the control windows accepted by the gate.
    - recall_control: fraction of true MI windows that the gate lets through.
    - fpr_idle: fraction of IDLE windows that the gate lets through as a command
      (false positives: a command issued while the user is at rest).
    """
    decs = gate.decide_batch(X)
    is_ctrl_true = y != C.IDLE_ID
    passed = np.array([d.is_control for d in decs])
    pred = np.array([d.control_class for d in decs])

    ctrl = is_ctrl_true & passed
    acc_control = float((pred[ctrl] == y[ctrl]).mean()) if ctrl.any() else float('nan')
    recall_control = float(passed[is_ctrl_true].mean()) if is_ctrl_true.any() else float('nan')
    idle = ~is_ctrl_true
    fpr_idle = float(passed[idle].mean()) if idle.any() else float('nan')
    return dict(acc_control=acc_control, recall_control=recall_control,
                fpr_idle=fpr_idle, n=len(y))
