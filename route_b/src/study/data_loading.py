"""Direct-from-``.mat`` trial loading for the study -- deliberately no NPZ cache.

Reuses ``first_ml.src.dataset`` for raw session discovery and trial parsing
(``discover_sessions`` / ``extract_session_trials``, which already applies
band-pass -> downsample -> CAR via ``first_ml.src.preprocessing``) and simply
never calls the cache-writing side of that module. Every call to
:func:`load_trials_for_sessions` performs a fresh pass over the requested raw
``.mat`` files; nothing is persisted to disk between experiment runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import config as base_config
from .. import dataset as base_dataset
from ..utils import get_logger

logger = get_logger()


@dataclass
class LoadedTrial:
    """One preprocessed trial, read directly from the raw dataset."""

    trial_id: str           # f"{session_key}#trial{original_mat_trial_index}"
    session_key: str
    subject: str
    signal: np.ndarray       # (n_channels, n_samples) float32; band-pass+downsample+CAR applied
    label: int


def _session_refs_by_key() -> Dict[str, base_dataset.SessionRef]:
    """Map every discovered session's key to its :class:`SessionRef`."""
    return {ref.key: ref for ref in base_dataset.discover_sessions(base_config.DATA_DIR)}


def load_trials_for_sessions(
    session_keys: Sequence[str],
    window_fn: Optional[base_dataset.WindowFn] = None,
) -> List[LoadedTrial]:
    """Parse the requested sessions directly from raw ``.mat``, in memory.

    Parameters
    ----------
    session_keys:
        Session keys (e.g. ``"S1_Session_5"``) belonging to a single split,
        as recorded in the reused ``dataset_split.json``.
    window_fn:
        Optional per-trial window selector forwarded to
        :func:`first_ml.src.dataset.extract_session_trials`. ``None`` keeps the
        legacy ``[0, triallength]`` window used by the published experiments.

    Returns
    -------
    list of LoadedTrial
        Every valid Left/Right trial across the requested sessions, each
        tagged with a globally unique ``trial_id``. No file is written.
    """
    refs_by_key = _session_refs_by_key()
    missing = [k for k in session_keys if k not in refs_by_key]
    if missing:
        raise RuntimeError(f"Session(s) not found under {base_config.DATA_DIR}: {missing}")

    trials: List[LoadedTrial] = []
    total = len(session_keys)
    for pos, key in enumerate(session_keys, start=1):
        ref = refs_by_key[key]
        logger.info("[%3d/%d] parsing %s directly from .mat (no cache)...", pos, total, key)
        result = base_dataset.extract_session_trials(ref, window_fn=window_fn)
        if result is None:
            logger.warning("Session %s skipped: missing motor channel(s).", key)
            continue
        session_trials, labels, meta, counters = result
        if counters.kept == 0:
            logger.warning("Session %s has no valid Left/Right trials.", key)
            continue

        trial_indices = meta["trial_indices"]
        for signal, label, original_idx in zip(session_trials, labels, trial_indices):
            trials.append(
                LoadedTrial(
                    trial_id=f"{key}#trial{int(original_idx)}",
                    session_key=key,
                    subject=ref.subject,
                    signal=signal,
                    label=int(label),
                )
            )

    logger.info(
        "Loaded %d trials from %d sessions (direct .mat read, no NPZ cache).",
        len(trials), total,
    )
    return trials
