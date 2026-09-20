"""Classic per-channel Z-score normalization -- the study's controlled baseline.

Wraps :class:`first_ml.src.preprocessing.ZScoreNormalizer`, the exact
implementation the project's very first experiment (Approach A) already
uses, so there is a single implementation of per-channel z-scoring in the
whole codebase -- this module adds no new math, only an adapter exposing the
same ``fit``/``transform``/``is_fitted`` interface that
:class:`~first_ml.src.study.normalization.euclidean_alignment.EuclideanAlignment`
exposes, so ``run_experiment.py`` can treat Z-score, Running Exponential
Standardization and Euclidean Alignment uniformly.

**Leakage control**: per-channel mean/std are fit **exclusively** on
training trials (see :meth:`ZScoreNormalizer.fit`) -- validation and test
trials are only ever *transformed* with the already-fitted statistics, never
used to compute them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ...preprocessing import ZScoreNormalizer as _BaseZScoreNormalizer
from .. import config as study_config


@dataclass
class ZScoreNormalizer:
    """Per-channel Z-score, fit on training trials only.

    Parameters
    ----------
    eps:
        Numerical floor added to the standard deviation before dividing.
    """

    eps: float = study_config.Z_SCORE.eps
    _inner: Optional[_BaseZScoreNormalizer] = field(default=None, repr=False)

    @property
    def is_fitted(self) -> bool:
        return self._inner is not None

    def fit(self, train_trials: List[np.ndarray]) -> "ZScoreNormalizer":
        """Estimate per-channel mean/std from training trials only.

        ``train_trials`` must contain *only* trials belonging to the
        training split; validation/test trials must never be passed here.
        """
        if not train_trials:
            raise ValueError("Cannot fit ZScoreNormalizer on zero trials.")
        self._inner = _BaseZScoreNormalizer.fit(list(train_trials), eps=self.eps)
        return self

    def transform(self, trial: np.ndarray) -> np.ndarray:
        """Standardize a ``(n_channels, n_samples)`` trial with the fitted stats."""
        if not self.is_fitted:
            raise RuntimeError("ZScoreNormalizer.fit() must be called before transform().")
        return self._inner.transform(trial)

    def to_dict(self) -> dict:
        """Fitted per-channel mean/std, saved for exact reproduction at inference."""
        if not self.is_fitted:
            return {"fitted": False}
        payload = dict(self._inner.to_dict())
        payload["fitted"] = True
        payload["eps"] = self.eps
        return payload
