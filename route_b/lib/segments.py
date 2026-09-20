"""Extract real MI (imagery) and REST (baseline) segments from the raw ``.mat``.

Reuses the *exact* first_ml preprocessing chain (band-pass 0.5-40 Hz @1000 Hz
-> downsample to 250 Hz -> CAR) via ``src.preprocessing.preprocess_trial`` and
the raw-parsing helpers of ``src.dataset``. For every valid Left/Right trial we
produce two segments from the same trial:

* an **MI** segment  = the corrected imagery window ``[2000, 2000+triallength]``
  (LEFT or RIGHT), identical to what the winning 2-class model was trained on;
* a **REST** segment = the pre-cue baseline ``[-2000, 0)`` (idle, no imagery),
  exactly 2 s -> 500 samples @250 Hz after preprocessing.

Nothing here writes into first_ml; the raw ``.mat`` files are only ever read.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.io import loadmat

import lib  # noqa: F401
from . import config3 as C

from src import config as base_config  # noqa: E402
from src.dataset import (  # noqa: E402
    SessionRef,
    _channel_labels,
    _maybe_float,
    _maybe_int,
    _select_channel_indices,
    discover_sessions,
    imagery_window_from_feedback,
)
from src.preprocessing import preprocess_trial  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger()


@dataclass
class Segment:
    """One preprocessed segment (250 Hz, CAR'd, NOT yet EA-aligned)."""

    trial_id: str      # unique: f"{session_key}#trial{idx}" (MI) or f"...#rest{idx}"
    session_key: str
    subject: str
    kind: str          # "mi" | "rest"
    label: int         # 0=REST, 1=LEFT, 2=RIGHT
    signal: np.ndarray  # (n_channels, T) float32 @250 Hz


def _rest_window(time_ms: np.ndarray) -> tuple[int, int]:
    """Raw sample bounds of the pre-cue baseline ``[-2000, 0)`` ms."""
    lo, hi = C.REST_WINDOW_MS
    start = int(np.searchsorted(time_ms, lo, side="left"))
    stop = int(np.searchsorted(time_ms, hi, side="left"))
    return start, stop


def _session_refs_by_key() -> Dict[str, SessionRef]:
    return {ref.key: ref for ref in discover_sessions(base_config.DATA_DIR)}


def extract_session_segments(ref: SessionRef) -> List[Segment]:
    """Return MI + REST segments for every valid Left/Right trial of a session."""
    mat = loadmat(str(ref.path))
    bci = mat["BCI"][0, 0]

    chan_idx = _select_channel_indices(_channel_labels(bci))
    if chan_idx is None:
        logger.warning("[%s] missing motor channel(s) -> skipped.", ref.key)
        return []
    chan_idx = np.asarray(chan_idx)

    data = bci["data"]
    time_field = bci["time"]
    td = bci["TrialData"]
    n_trials = data.shape[1]

    min_raw = base_config.MIN_TRIAL_SAMPLES * base_config.DOWNSAMPLE_FACTOR
    segments: List[Segment] = []

    for i in range(n_trials):
        task = _maybe_int(td["tasknumber"][0, i])
        target = _maybe_int(td["targetnumber"][0, i])
        result = _maybe_int(td["result"][0, i])
        triallength = _maybe_float(td["triallength"][0, i])

        if task not in base_config.TASK_FILTER:
            continue
        if target not in C.TARGET_TO_ID3:
            continue
        if result != base_config.VALID_RESULT:
            continue

        raw = np.asarray(data[0, i], dtype=np.float32)
        time_ms = np.asarray(time_field[0, i]).squeeze()

        # --- MI (corrected imagery window) --- #
        m0, m1 = imagery_window_from_feedback(time_ms, triallength, raw.shape[1])
        mi_raw = raw[chan_idx, m0:m1]
        if mi_raw.shape[1] < min_raw:
            continue  # too short to form one crop (same rule as first_ml)
        mi_sig = preprocess_trial(mi_raw)
        if mi_sig.shape[1] < base_config.MIN_TRIAL_SAMPLES:
            continue

        # --- REST (pre-cue baseline) --- #
        r0, r1 = _rest_window(time_ms)
        rest_raw = raw[chan_idx, r0:r1]
        if rest_raw.shape[1] < min_raw:
            # Extremely rare; skip REST but keep MI would break pairing symmetry.
            # We require both to exist to keep counting clean, so skip the trial.
            continue
        rest_sig = preprocess_trial(rest_raw)
        if rest_sig.shape[1] < base_config.MIN_TRIAL_SAMPLES:
            continue

        segments.append(Segment(
            trial_id=f"{ref.key}#trial{i}", session_key=ref.key, subject=ref.subject,
            kind="mi", label=C.TARGET_TO_ID3[target], signal=mi_sig,
        ))
        segments.append(Segment(
            trial_id=f"{ref.key}#rest{i}", session_key=ref.key, subject=ref.subject,
            kind="rest", label=C.REST_ID, signal=rest_sig,
        ))

    return segments


def extract_segments_for_sessions(session_keys: Sequence[str]) -> List[Segment]:
    """Extract segments for the given session keys (one split), directly from .mat."""
    refs = _session_refs_by_key()
    missing = [k for k in session_keys if k not in refs]
    if missing:
        raise RuntimeError(f"Sessions not found under {base_config.DATA_DIR}: {missing}")
    out: List[Segment] = []
    total = len(session_keys)
    for pos, key in enumerate(sorted(session_keys), start=1):
        logger.info("[%3d/%d] extracting MI+REST from %s ...", pos, total, key)
        out.extend(extract_session_segments(refs[key]))
    n_mi = sum(1 for s in out if s.kind == "mi")
    n_rest = sum(1 for s in out if s.kind == "rest")
    logger.info("Extracted %d segments (%d MI, %d REST) from %d sessions.",
                len(out), n_mi, n_rest, total)
    return out


# --------------------------------------------------------------------------- #
# Per-subject pools (for the continuous-EEG generator), cached to disk so the
# 5 robustness seeds and Part 1 / Part 2 do not re-read the raw .mat repeatedly.
# --------------------------------------------------------------------------- #
@dataclass
class SubjectPool:
    """Real preprocessed segments for one subject, grouped by class."""

    subject: str
    mi_left: List[np.ndarray]
    mi_right: List[np.ndarray]
    rest: List[np.ndarray]

    def summary(self) -> Dict[str, int]:
        return {
            "subject": self.subject,
            "n_mi_left": len(self.mi_left),
            "n_mi_right": len(self.mi_right),
            "n_rest": len(self.rest),
        }


def _subject_session_keys(subject: str, split_payload: dict) -> List[str]:
    for bucket in split_payload["splits"].values():
        if subject in bucket["subjects"]:
            return [k for k in bucket["session_keys"] if k.startswith(f"{subject}_")]
    raise RuntimeError(f"Subject {subject} not found in split.")


def load_or_build_subject_pool(subject: str, split_payload: dict) -> SubjectPool:
    """Return (building + caching if needed) one subject's real-segment pool."""
    C.ensure_dirs()
    cache = C.SEGMENT_CACHE / f"{subject}_pool.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=True) as npz:
            return SubjectPool(
                subject=str(npz["subject"]),
                mi_left=[np.asarray(a, dtype=np.float32) for a in npz["mi_left"]],
                mi_right=[np.asarray(a, dtype=np.float32) for a in npz["mi_right"]],
                rest=[np.asarray(a, dtype=np.float32) for a in npz["rest"]],
            )

    keys = _subject_session_keys(subject, split_payload)
    segs = extract_segments_for_sessions(keys)
    pool = SubjectPool(
        subject=subject,
        mi_left=[s.signal for s in segs if s.label == C.LEFT_ID],
        mi_right=[s.signal for s in segs if s.label == C.RIGHT_ID],
        rest=[s.signal for s in segs if s.label == C.REST_ID],
    )

    def _obj(arrs: List[np.ndarray]) -> np.ndarray:
        o = np.empty(len(arrs), dtype=object)
        for i, a in enumerate(arrs):
            o[i] = a
        return o

    np.savez_compressed(
        cache, subject=subject,
        mi_left=_obj(pool.mi_left), mi_right=_obj(pool.mi_right), rest=_obj(pool.rest),
    )
    logger.info("Cached pool for %s: %s", subject, pool.summary())
    return pool
