"""Normalization factory: pick Euclidean Alignment (EA) or Running Exponential
Standardization (REST) with one interface, for both training and continuous
inference.

* **euclidean_alignment** -- a linear spatial whitening ``x -> W @ x`` with ``W``
  fit on the training set only; ``W`` is persisted (``ea_reference.json``) and
  reloaded for inference, never recomputed on evaluation data.
* **running_exponential** -- a causal per-channel exponential-moving
  standardization. It is *stateless across calls* and needs no fitting, so
  there is nothing to persist; the same transform is applied at inference. On
  the continuous stream it runs exactly as it would online (one warm-up at the
  start), which is the realistic real-time behaviour of this normalization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import lib  # noqa: F401
from . import ea_io

from src.study.normalization.euclidean_alignment import EuclideanAlignment  # noqa: E402
from src.study.normalization.running_exponential import RunningExponentialStandardizer  # noqa: E402

NormalizeFn = Callable[[np.ndarray], np.ndarray]


def build_normalizer(method: str, train_signals: List[np.ndarray]
                     ) -> Tuple[NormalizeFn, Optional[object], Dict[str, object]]:
    """Return ``(normalize_fn, saveable, params)`` for training.

    ``saveable`` is an object with ``to_dict()`` to persist (EA) or ``None``
    (running exponential needs no persisted state).
    """
    if method == "euclidean_alignment":
        ea = EuclideanAlignment()
        ea.fit(list(train_signals))
        params = {"method": "euclidean_alignment", "eps": ea.eps,
                  "fit_on": "training segments only (single global reference)",
                  "n_signals_used_for_reference": len(train_signals)}
        return ea.transform, ea, params
    if method == "running_exponential":
        rex = RunningExponentialStandardizer()
        params = {"method": "running_exponential", "factor_new": rex.factor_new,
                  "init_block_size": rex.init_block_size, "eps": rex.eps,
                  "causal": True, "fit_on": "none (stateless causal per-signal transform)"}
        return rex.transform, None, params
    raise ValueError(f"Unknown normalization method: {method!r}")


def save_normalizer(method: str, saveable: Optional[object], model_dir: Path) -> None:
    if method == "euclidean_alignment":
        ea_io.save_ea(saveable, model_dir / "ea_reference.json")  # type: ignore[arg-type]
    # running_exponential: nothing to persist.


def load_normalizer(method: str, model_dir: Path) -> NormalizeFn:
    """Return the normalize function for inference (Part 2)."""
    if method == "euclidean_alignment":
        W = ea_io.load_ea_matrix(model_dir / "ea_reference.json")
        return lambda sig, _W=W: ea_io.apply_ea(_W, sig)
    if method == "running_exponential":
        return RunningExponentialStandardizer().transform
    raise ValueError(f"Unknown normalization method: {method!r}")
