"""Running Exponential Standardization (causal, real-time-friendly).

Implements, from scratch with NumPy only, the exponential-moving-average
standardization commonly used in Braindecode-style EEG pipelines
(Schirrmeister et al., 2017): at every timestep the per-channel mean and
variance are updated with an exponential decay factor and the sample is
standardized with the statistics known *up to and including that timestep*.
Because no future sample is ever used, the transform is strictly causal and
can be applied online, one sample at a time, exactly as it would run in a
real-time BCI loop.

An initial "warm-start" block (``init_block_size`` samples) is standardized
with its own block mean/std, since the exponential statistics have not
accumulated enough evidence yet; the running recursion then continues from
the end of that block. This is the standard fix for the otherwise very noisy
early samples and is itself still causal at inference time (the block is
observed before any of its own samples are reported as standardized).

No external EEGNet/Braindecode code is imported or copied -- only the
mathematical formulation is reused.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config as study_config


@dataclass
class RunningExponentialStandardizer:
    """Causal, per-channel exponential moving standardization.

    Parameters
    ----------
    factor_new:
        Exponential decay factor ``alpha`` in ``(0, 1]``; larger values track
        non-stationarity faster but are noisier.
    init_block_size:
        Number of leading samples standardized with a plain block mean/std
        instead of the (still unstable) running statistics.
    eps:
        Numerical floor added to the variance before taking the square root.
    """

    factor_new: float = study_config.RUNNING_EXPONENTIAL.factor_new
    init_block_size: int = study_config.RUNNING_EXPONENTIAL.init_block_size
    eps: float = study_config.RUNNING_EXPONENTIAL.eps

    def transform(self, trial: np.ndarray) -> np.ndarray:
        """Standardize a ``(n_channels, n_samples)`` trial causally in time.

        This method is stateless across calls (it never uses information
        from any other trial), so it can be applied independently and
        identically to train, validation and test trials without any risk
        of leakage.
        """
        n_channels, n_samples = trial.shape
        out = np.empty_like(trial, dtype=np.float32)

        block = min(self.init_block_size, n_samples)
        init_mean = trial[:, :block].mean(axis=1)
        init_std = trial[:, :block].std(axis=1)
        init_std = np.maximum(init_std, self.eps)
        out[:, :block] = (trial[:, :block] - init_mean[:, None]) / init_std[:, None]

        running_mean = init_mean.astype(np.float64)
        running_var = np.square(init_std.astype(np.float64))
        alpha = self.factor_new

        for t in range(block, n_samples):
            x_t = trial[:, t].astype(np.float64)
            running_mean = alpha * x_t + (1.0 - alpha) * running_mean
            running_var = alpha * np.square(x_t - running_mean) + (1.0 - alpha) * running_var
            denom = np.sqrt(np.maximum(running_var, self.eps))
            out[:, t] = (x_t - running_mean) / denom

        return out
