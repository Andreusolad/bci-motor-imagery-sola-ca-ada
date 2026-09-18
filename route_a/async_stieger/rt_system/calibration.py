"""Confidence rejection with a temperature-calibrated softmax.

The raw softmax of a neural network tends to be overconfident (an artifact can get a high
LH probability with no motor imagery present), so it is calibrated before thresholding.
Temperature scaling (Guo et al., 2017) divides the logits by a single scalar T fitted on
validation data by minimizing the NLL. It is post hoc and does not change the predictions
(argmax is invariant to T), only the confidence.

select_conf_threshold() then picks the threshold tau on the calibrated probabilities that
gives a target false-positive rate on IDLE windows.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import device


# ============================================================
# Temperature scaling
# ============================================================
class TemperatureScaler(nn.Module):
    """Learns a single scalar T and returns calibrated probabilities softmax(z / T)."""

    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))   # T = exp(log_T) > 0

    @property
    def T(self) -> float:
        return float(self.log_T.exp().item())

    def forward_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_T.exp()

    def fit(self, logits: np.ndarray, y: np.ndarray, max_iter: int = 200) -> 'TemperatureScaler':
        """Fit T on validation (logits, y) with LBFGS."""
        z = torch.tensor(np.asarray(logits), dtype=torch.float32, device=device)
        t = torch.tensor(np.asarray(y), dtype=torch.long, device=device)
        self.to(device)
        opt = torch.optim.LBFGS([self.log_T], lr=0.05, max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            loss = nll(self.forward_logits(z), t)
            loss.backward()
            return loss
        opt.step(closure)
        return self

    def probs(self, logits: np.ndarray) -> np.ndarray:
        """Calibrated probabilities from logits (numpy in, numpy out)."""
        z = torch.tensor(np.asarray(logits), dtype=torch.float32, device=device)
        with torch.no_grad():
            p = F.softmax(self.forward_logits(z), dim=-1)
        return p.cpu().numpy()


# ============================================================
# Logit extraction (for calibration and thresholding)
# ============================================================
@torch.no_grad()
def get_logits(model: nn.Module, X: np.ndarray, batch: int = 256) -> np.ndarray:
    """Model logits for X (n, n_ch, w) -> (n, n_classes)."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().unsqueeze(1).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,))


# ============================================================
# Calibration metric and threshold selection
# ============================================================
def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """ECE: |confidence - accuracy| per confidence bin, weighted by bin size. 0 = perfect."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def select_conf_threshold(probs_idle: np.ndarray, idle_id: int,
                          target_fpr: float = 0.05) -> float:
    """Pick tau so that only a fraction `target_fpr` of IDLE windows exceed it.

    probs_idle: calibrated softmax of windows that are truly IDLE. The command confidence
    of a window is its maximum probability over the control classes (LH, RH). Returns the
    (1 - target_fpr) quantile of that confidence, clipped to [0.34, 0.99].
    """
    control = np.delete(probs_idle, idle_id, axis=1)   # LH, RH columns
    conf_control = control.max(axis=1)                 # command confidence at rest
    tau = float(np.quantile(conf_control, 1.0 - target_fpr))
    return min(max(tau, 0.34), 0.99)
