"""Out-of-distribution (open-set) detection: windows that resemble no known MI class.

The IDLE class covers the rest seen in training and the confidence threshold catches
uncertain outputs, but neither detects inputs unlike the training data (a muscle artifact,
the user speaking, a moving electrode). Those are detected by distance to the training
data rather than by the classifier output.

Two detectors:

  RiemannOOD: minimum Riemannian (geodesic) distance from the window covariance to the
      Riemannian means of the LH and RH classes.

  MahalanobisOOD: squared Mahalanobis distance in the feature space of the network
      (Lee et al., 2018). One Gaussian per class with a shared covariance is fitted on the
      training features, and the score is the minimum distance over classes.

Both provide .fit() on control (LH/RH) training windows, .score() (higher = more unusual)
and .select_threshold(), which sets the threshold at the keep_frac quantile of
in-distribution scores (the default 0.95 lets about 95% of real MI windows through).
"""
from __future__ import annotations

import numpy as np

from .config import IDLE_ID


# ============================================================
# Riemannian OOD (distance to class means)
# ============================================================
class RiemannOOD:
    """Minimum geodesic distance to the LH and RH covariance means."""

    def __init__(self, estimator: str = 'oas'):
        self.estimator = estimator
        self.means_: list[np.ndarray] = []
        self.threshold_: float | None = None

    def _covs(self, X: np.ndarray) -> np.ndarray:
        from pyriemann.estimation import Covariances
        return Covariances(self.estimator).fit_transform(X.astype(np.float64))

    def fit(self, X_ctrl: np.ndarray, y_ctrl: np.ndarray) -> 'RiemannOOD':
        """X_ctrl: control windows (n, 8, w). y_ctrl: 0/1 (LH/RH), no IDLE."""
        from pyriemann.utils.mean import mean_riemann
        covs = self._covs(X_ctrl)
        self.means_ = []
        for cls in sorted(set(int(c) for c in y_ctrl)):
            self.means_.append(mean_riemann(covs[y_ctrl == cls]))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Per-window OOD score: geodesic distance to the nearest class mean."""
        from pyriemann.utils.distance import distance_riemann
        covs = self._covs(X) if X.ndim == 3 else self._covs(X[None])
        out = np.empty(len(covs))
        for i, cov in enumerate(covs):
            out[i] = min(distance_riemann(cov, M) for M in self.means_)
        return out

    def select_threshold(self, X_in: np.ndarray, keep_frac: float = 0.95) -> float:
        """Threshold = keep_frac quantile of the scores of in-distribution MI windows."""
        s = self.score(X_in)
        self.threshold_ = float(np.quantile(s, keep_frac))
        return self.threshold_

    def is_ood(self, X: np.ndarray) -> np.ndarray:
        assert self.threshold_ is not None, 'call select_threshold first'
        return self.score(X) > self.threshold_


# ============================================================
# Mahalanobis OOD (in the feature space of the network)
# ============================================================
class MahalanobisOOD:
    """Per-class Gaussian with a shared covariance on the network features."""

    def __init__(self, model, batch: int = 256):
        self.model = model
        self.batch = batch
        self.means_: np.ndarray | None = None       # (n_classes, d)
        self.prec_: np.ndarray | None = None         # (d, d) inverse of the shared covariance
        self.threshold_: float | None = None

    def _features(self, X: np.ndarray) -> np.ndarray:
        import torch
        from .config import device
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch):
                xb = torch.from_numpy(X[i:i + self.batch]).float().unsqueeze(1).to(device)
                out.append(self.model.features(xb).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def fit(self, X_ctrl: np.ndarray, y_ctrl: np.ndarray, reg: float = 1e-3) -> 'MahalanobisOOD':
        f = self._features(X_ctrl)
        classes = sorted(set(int(c) for c in y_ctrl))
        means, centered = [], []
        for cls in classes:
            fc = f[y_ctrl == cls]
            mu = fc.mean(axis=0)
            means.append(mu); centered.append(fc - mu)
        self.means_ = np.stack(means)
        centered = np.concatenate(centered)          # shared (tied) covariance
        cov = np.cov(centered, rowvar=False)
        cov += reg * np.eye(cov.shape[0])            # ridge term keeps cov invertible
        self.prec_ = np.linalg.inv(cov)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self.means_ is not None
        f = self._features(X if X.ndim == 3 else X[None])
        out = np.empty(len(f))
        for i, fi in enumerate(f):
            d = fi[None, :] - self.means_            # (n_classes, d)
            m = np.einsum('cd,de,ce->c', d, self.prec_, d)   # squared Mahalanobis per class
            out[i] = m.min()
        return out

    def select_threshold(self, X_in: np.ndarray, keep_frac: float = 0.95) -> float:
        s = self.score(X_in)
        self.threshold_ = float(np.quantile(s, keep_frac))
        return self.threshold_

    def is_ood(self, X: np.ndarray) -> np.ndarray:
        assert self.threshold_ is not None, 'call select_threshold first'
        return self.score(X) > self.threshold_


# ============================================================
# Helper: keep the control windows (LH/RH), drop IDLE
# ============================================================
def solo_control(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = y != IDLE_ID
    return X[m], y[m]
