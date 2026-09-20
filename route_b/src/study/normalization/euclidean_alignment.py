"""Euclidean Alignment (EA), implemented from scratch with NumPy/SciPy only.

Euclidean Alignment (He & Wu, 2020, *"Transfer Learning for Brain-Computer
Interfaces: A Euclidean Space Data Alignment Approach"*) whitens the spatial
covariance of EEG trials towards the identity matrix. A single reference
covariance matrix is estimated by averaging the (uncentered) per-trial spatial
covariance matrices, and every trial is then left-multiplied by the inverse
matrix square root of that reference.

**Why this reduces cross-subject domain shift**: differences in skull
geometry, electrode impedance and individual anatomy make each subject's EEG
spatial covariance structure systematically different, even for the same
mental task -- this is a major source of the distribution shift that makes
cross-subject BCI classifiers generalize poorly. Whitening every trial with a
common reference re-centers the *second-order statistics* (the spatial
covariance) of all trials around the same target (approximately the identity
matrix scaled by the reference), so a downstream classifier sees inputs whose
channel-covariance structure is far more comparable across subjects, even
though the class-discriminative information in the signal is preserved (EA is
a linear, invertible transform and uses no label information).

**Leakage control**: the reference covariance matrix is fit **exclusively**
from training trials (see :meth:`EuclideanAlignment.fit`) -- validation and
test trials are only ever *transformed* with the already-fitted reference,
never used to compute it. This is a single, global reference (pooled across
all training subjects/sessions), which is a conservative choice: it avoids
any use of validation/test statistics, at the cost of not doing the (more
common but leakier-looking) per-subject on-the-fly recalibration that would
compute a separate reference for each test subject from that subject's own
unlabeled trials.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .. import config as study_config


@dataclass
class EuclideanAlignment:
    """Whitening transform with a single reference fit on training data only.

    Parameters
    ----------
    eps:
        Eigenvalue floor applied before inverting, for numerical stability.
    """

    eps: float = study_config.EUCLIDEAN_ALIGNMENT.eps
    _reference_inv_sqrt: Optional[np.ndarray] = field(default=None, repr=False)
    _reference_matrix: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def is_fitted(self) -> bool:
        return self._reference_inv_sqrt is not None

    @staticmethod
    def _trial_covariance(trial: np.ndarray) -> np.ndarray:
        """Uncentered spatial covariance ``(X X^T) / T`` of one trial."""
        n_samples = trial.shape[1]
        return (trial @ trial.T) / float(n_samples)

    def fit(self, train_trials: List[np.ndarray]) -> "EuclideanAlignment":
        """Estimate the reference covariance matrix from training trials only.

        ``train_trials`` must contain *only* trials belonging to the training
        split; validation/test trials must never be passed here.
        """
        if not train_trials:
            raise ValueError("Cannot fit EuclideanAlignment on zero trials.")

        n_channels = train_trials[0].shape[0]
        cov_sum = np.zeros((n_channels, n_channels), dtype=np.float64)
        for trial in train_trials:
            cov_sum += self._trial_covariance(trial)
        reference = cov_sum / len(train_trials)

        eigvals, eigvecs = np.linalg.eigh(reference)
        eigvals = np.maximum(eigvals, self.eps)
        inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
        reference_inv_sqrt = (eigvecs * inv_sqrt_eigvals) @ eigvecs.T

        self._reference_matrix = reference.astype(np.float32)
        self._reference_inv_sqrt = reference_inv_sqrt.astype(np.float32)
        return self

    def transform(self, trial: np.ndarray) -> np.ndarray:
        """Align a ``(n_channels, n_samples)`` trial with the fitted reference."""
        if not self.is_fitted:
            raise RuntimeError("EuclideanAlignment.fit() must be called before transform().")
        return (self._reference_inv_sqrt @ trial).astype(np.float32, copy=False)

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot of the fitted reference (for auditing)."""
        if not self.is_fitted:
            return {"fitted": False}
        return {
            "fitted": True,
            "eps": self.eps,
            "reference_covariance": self._reference_matrix.tolist(),
            "reference_inv_sqrt": self._reference_inv_sqrt.tolist(),
        }
