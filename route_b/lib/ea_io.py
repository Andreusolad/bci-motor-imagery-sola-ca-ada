"""Save / load the Euclidean Alignment reference so Part 2 can reuse the
*exact* whitening matrix fitted during Part 1 training (never recomputed on
evaluation data -- the brief's hard requirement).

EA is a fixed linear spatial transform ``x -> W @ x`` with ``W`` the inverse
matrix square root of the training reference covariance. Applying it per-crop
or to a whole continuous stream is identical (it acts independently on every
time sample), so continuous inference just left-multiplies the stream by ``W``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import lib  # noqa: F401
from src.study.normalization.euclidean_alignment import EuclideanAlignment  # noqa: E402


def fit_ea(train_signals) -> EuclideanAlignment:
    """Fit EA on training signals only (single global reference)."""
    ea = EuclideanAlignment()
    ea.fit(list(train_signals))
    return ea


def save_ea(ea: EuclideanAlignment, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ea.to_dict(), indent=2), encoding="utf-8")


def load_ea_matrix(path: Path) -> np.ndarray:
    """Return the fitted ``reference_inv_sqrt`` (W) as a float32 (C, C) matrix."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("fitted", False):
        raise RuntimeError(f"EA reference at {path} is not fitted.")
    return np.asarray(payload["reference_inv_sqrt"], dtype=np.float32)


def apply_ea(matrix: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Left-multiply a ``(C, T)`` signal by the EA matrix ``W``."""
    return (matrix @ signal).astype(np.float32, copy=False)
