"""Dataset discovery, ``.mat`` parsing and the lightweight per-session cache.

The raw dataset (207 GB across 350 ``.mat`` files at 1000 Hz) is far too large to
re-read every epoch. This module performs a single pass that, per session, keeps
only the eight motor channels of the valid Left/Right trials, restricted to the
motor-imagery window ``[0, triallength]``, band-passed / downsampled / CAR-ed,
and stores the result as a small ``.npz`` (a few MB) under ``data/cache``.

The cache is a disposable derivative of the read-only originals: delete it and it
regenerates. It is *not* a rebuilt dataset and holds no crops.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat

from . import config
from .preprocessing import preprocess_trial
from .utils import get_logger

logger = get_logger()

# TrialData fields we read per trial.
_TD_FIELDS = ("tasknumber", "targetnumber", "result", "triallength")

# (time_ms, triallength_s, n_samples) -> (start, stop) sample indices.
WindowFn = Callable[[np.ndarray, float, int], Tuple[int, int]]


# --------------------------------------------------------------------------- #
# Session discovery
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SessionRef:
    """A pointer to one raw session file plus its identity."""

    subject: str          # e.g. "S1"
    session: int          # e.g. 5
    path: Path

    @property
    def key(self) -> str:
        """Stable unique identifier, e.g. ``S1_Session_5``."""
        return f"{self.subject}_Session_{self.session}"

    @property
    def cache_path(self) -> Path:
        return config.CACHE_DIR / f"{self.key}.npz"


def _subject_number(name: str) -> int:
    return int(name.lstrip("S"))


def _session_number(path: Path) -> int:
    return int(path.stem.split("_Session_")[-1])


def discover_sessions(data_dir: Path = config.DATA_DIR) -> List[SessionRef]:
    """List every session file, sorted by (subject number, session number)."""
    sessions: List[SessionRef] = []
    for subject_dir in sorted(
        (p for p in data_dir.iterdir() if p.is_dir()),
        key=lambda p: _subject_number(p.name),
    ):
        for mat_path in sorted(
            subject_dir.glob(f"{subject_dir.name}_Session_*.mat"),
            key=_session_number,
        ):
            sessions.append(
                SessionRef(subject_dir.name, _session_number(mat_path), mat_path)
            )
    return sessions


# --------------------------------------------------------------------------- #
# Low-level .mat helpers (self-contained; the raw files are only ever read)
# --------------------------------------------------------------------------- #
def _scalar(value: object) -> object:
    arr = np.asarray(value).squeeze()
    return arr.item() if arr.shape == () else arr


def _maybe_int(value: object) -> Optional[int]:
    arr = np.asarray(value).squeeze()
    try:
        f = float(arr)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else int(f)


def _maybe_float(value: object) -> float:
    try:
        return float(np.asarray(value).squeeze())
    except (TypeError, ValueError):
        return float("nan")


def _channel_labels(bci: np.ndarray) -> List[str]:
    labels_raw = np.asarray(bci["chaninfo"][0, 0]["label"]).squeeze()
    labels: List[str] = []
    for label in labels_raw:
        label = np.asarray(label).squeeze()
        labels.append(str(label.item() if label.shape == () else label).strip())
    return labels


def _select_channel_indices(labels: Sequence[str]) -> Optional[List[int]]:
    """Map the configured motor channels to indices (case-insensitive).

    Returns ``None`` if any requested channel is missing, so the caller can skip
    the session as required.
    """
    upper = [str(name).strip().upper() for name in labels]
    indices: List[int] = []
    for wanted in config.MOTOR_CHANNELS:
        try:
            indices.append(upper.index(wanted.upper()))
        except ValueError:
            return None
    return indices


# --------------------------------------------------------------------------- #
# Trial extraction
# --------------------------------------------------------------------------- #
@dataclass
class SessionCounters:
    """Per-session bookkeeping of why trials were kept or dropped."""

    n_trials_total: int = 0
    kept: int = 0
    dropped_wrong_task: int = 0
    dropped_wrong_target: int = 0
    dropped_not_success: int = 0
    dropped_too_short: int = 0
    kept_left: int = 0
    kept_right: int = 0


def _imagery_window(time_ms: np.ndarray, triallength_s: float, n_samples: int) -> Tuple[int, int]:
    """Return ``(start, stop)`` sample indices for the ``[0, triallength]`` window.

    ``time_ms`` is the per-trial time axis, which starts at -2000 ms.

    .. warning::
       **This window is not the motor-imagery window.** Verified empirically
       (295 trials / 3 subjects): the cursor (``positionx``) is NaN until
       *exactly* +2000 ms and the axis always ends at
       ``2000 + triallength*1000 + 1000``. The real trial structure is
       ``[-2000, 0)`` baseline -> ``[0, 2000)`` cue only, no cursor ->
       ``[2000, 2000 + triallength*1000]`` imagery + feedback -> 1000 ms tail.
       So ``triallength`` is measured from +2000, and this window both starts
       and ends 2 s early: it is ~28 % imagery on average, and 0 % imagery for
       the ~28 % of kept trials with ``triallength <= 2``. Because the cue is a
       lateralised on-screen target perfectly correlated with the label, models
       trained on it partly decode the visual evoked response.

       Kept as the default only so the already-published experiments and the
       NPZ cache stay reproducible. New work should pass
       :func:`imagery_window_from_feedback` as ``window_fn``.
    """
    start = int(np.searchsorted(time_ms, 0, side="left"))
    stop = start + int(round(triallength_s * config.FS_ORIGINAL))
    return start, min(stop, n_samples)


def imagery_window_from_feedback(
    time_ms: np.ndarray, triallength_s: float, n_samples: int
) -> Tuple[int, int]:
    """Return ``(start, stop)`` for the true imagery window ``[2000, 2000+triallength]``.

    Starts at feedback onset (+2000 ms, when the cursor appears), so the cue
    period is excluded entirely. Same duration as :func:`_imagery_window`
    (``triallength`` seconds), hence the same trials survive the
    ``MIN_TRIAL_SAMPLES`` filter and the same number of crops is produced --
    the correction costs no data.
    """
    start = int(np.searchsorted(time_ms, config.FEEDBACK_ONSET_MS, side="left"))
    stop = start + int(round(triallength_s * config.FS_ORIGINAL))
    return start, min(stop, n_samples)


def extract_session_trials(
    ref: SessionRef,
    window_fn: Optional[WindowFn] = None,
) -> Optional[Tuple[List[np.ndarray], np.ndarray, Dict[str, np.ndarray], SessionCounters]]:
    """Parse one session and return preprocessed valid Left/Right trials.

    Returns ``None`` if the session is missing any motor channel (it is skipped
    and reported by the caller). Otherwise returns
    ``(trials, labels, meta, counters)`` where ``trials`` is a list of
    ``(n_channels, T)`` float32 arrays at ``FS_TARGET``.

    Parameters
    ----------
    window_fn:
        ``(time_ms, triallength_s, n_samples) -> (start, stop)`` selecting the
        per-trial window. Defaults to :func:`_imagery_window` (the legacy
        ``[0, triallength]`` window) so existing callers and the NPZ cache are
        unaffected. Pass :func:`imagery_window_from_feedback` for the true
        imagery window.
    """
    select_window = window_fn if window_fn is not None else _imagery_window
    mat = loadmat(str(ref.path))
    bci = mat["BCI"][0, 0]

    labels_all = _channel_labels(bci)
    chan_idx = _select_channel_indices(labels_all)
    if chan_idx is None:
        missing = [c for c in config.MOTOR_CHANNELS if c.upper() not in
                   {l.upper() for l in labels_all}]
        logger.warning("[%s] missing channels %s -> skipping session.", ref.key, missing)
        return None

    data = bci["data"]
    time_field = bci["time"]
    td = bci["TrialData"]
    n_trials = data.shape[1]

    counters = SessionCounters(n_trials_total=n_trials)
    trials: List[np.ndarray] = []
    labels: List[int] = []
    target_numbers: List[int] = []
    trial_indices: List[int] = []
    triallengths: List[float] = []

    for i in range(n_trials):
        task = _maybe_int(td["tasknumber"][0, i])
        target = _maybe_int(td["targetnumber"][0, i])
        result = _maybe_int(td["result"][0, i])
        triallength = _maybe_float(td["triallength"][0, i])

        if task not in config.TASK_FILTER:
            counters.dropped_wrong_task += 1
            continue
        if target not in config.TARGET_TO_LABEL:
            counters.dropped_wrong_target += 1
            continue
        if result != config.VALID_RESULT:
            counters.dropped_not_success += 1
            continue

        raw = np.asarray(data[0, i], dtype=np.float32)
        time_ms = np.asarray(time_field[0, i]).squeeze()
        start, stop = select_window(time_ms, triallength, raw.shape[1])
        window = raw[np.asarray(chan_idx), start:stop]

        # Need at least one full crop after downsampling.
        if window.shape[1] < config.MIN_TRIAL_SAMPLES * config.DOWNSAMPLE_FACTOR:
            counters.dropped_too_short += 1
            continue

        processed = preprocess_trial(window)
        if processed.shape[1] < config.MIN_TRIAL_SAMPLES:
            counters.dropped_too_short += 1
            continue

        label_name = config.TARGET_TO_LABEL[target]
        label_id = config.LABEL_TO_ID[label_name]
        trials.append(processed)
        labels.append(label_id)
        target_numbers.append(target)
        trial_indices.append(i)
        triallengths.append(triallength)
        counters.kept += 1
        if label_name == "left":
            counters.kept_left += 1
        else:
            counters.kept_right += 1

    meta = {
        "target_numbers": np.asarray(target_numbers, dtype=np.int16),
        "trial_indices": np.asarray(trial_indices, dtype=np.int32),
        "triallengths_s": np.asarray(triallengths, dtype=np.float32),
        "n_samples": np.asarray([t.shape[1] for t in trials], dtype=np.int32),
    }
    return trials, np.asarray(labels, dtype=np.int8), meta, counters


# --------------------------------------------------------------------------- #
# Cache persistence
# --------------------------------------------------------------------------- #
def save_session_cache(
    ref: SessionRef,
    trials: List[np.ndarray],
    labels: np.ndarray,
    meta: Dict[str, np.ndarray],
) -> None:
    """Persist one session's preprocessed trials as a compressed ``.npz``."""
    trial_object = np.empty(len(trials), dtype=object)
    for idx, trial in enumerate(trials):
        trial_object[idx] = trial
    np.savez_compressed(
        ref.cache_path,
        trials=trial_object,
        labels=labels,
        subject=ref.subject,
        session=ref.session,
        channels=np.asarray(config.MOTOR_CHANNELS),
        fs=config.FS_TARGET,
        **meta,
    )


@dataclass
class CachedSession:
    """A session loaded back from the cache."""

    subject: str
    session: int
    trials: List[np.ndarray]
    labels: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.subject}_Session_{self.session}"


def load_session_cache(cache_path: Path) -> CachedSession:
    """Load a cached session ``.npz`` back into memory."""
    with np.load(cache_path, allow_pickle=True) as npz:
        trials = [np.asarray(t, dtype=np.float32) for t in npz["trials"]]
        return CachedSession(
            subject=str(npz["subject"]),
            session=int(npz["session"]),
            trials=trials,
            labels=npz["labels"].astype(np.int64),
        )


def cache_exists() -> bool:
    """True if at least one cached session file is present."""
    return any(config.CACHE_DIR.glob("*.npz"))


# --------------------------------------------------------------------------- #
# Cache building (single pass over the raw dataset)
# --------------------------------------------------------------------------- #
def build_cache(
    sessions: Optional[List[SessionRef]] = None,
    force: bool = False,
) -> List[Dict[str, object]]:
    """Build (or refresh) the per-session cache and return summary records.

    Parameters
    ----------
    sessions:
        Sessions to process; defaults to every discovered session.
    force:
        If ``False`` (default), sessions whose cache file already exists are
        skipped, making cache building resumable.

    Returns
    -------
    list of dict
        One summary row per successfully cached session (subject, session,
        trial counts) for ``dataset_statistics.json``.
    """
    config.ensure_output_dirs()
    sessions = sessions if sessions is not None else discover_sessions()
    summaries: List[Dict[str, object]] = []
    total = len(sessions)

    for pos, ref in enumerate(sessions, start=1):
        if not force and ref.cache_path.exists():
            logger.info("[%3d/%d] %s cached, skipping.", pos, total, ref.key)
            cached = load_session_cache(ref.cache_path)
            summaries.append(_summary_from_labels(ref, cached.labels))
            continue

        logger.info("[%3d/%d] %s parsing...", pos, total, ref.key)
        result = extract_session_trials(ref)
        if result is None:
            continue
        trials, labels, meta, counters = result
        if counters.kept == 0:
            logger.warning("[%s] no valid Left/Right trials; not caching.", ref.key)
            continue
        save_session_cache(ref, trials, labels, meta)
        summaries.append(
            {
                "subject": ref.subject,
                "session": ref.session,
                "key": ref.key,
                "n_trials_total": counters.n_trials_total,
                "n_kept": counters.kept,
                "n_left": counters.kept_left,
                "n_right": counters.kept_right,
                "dropped_wrong_task": counters.dropped_wrong_task,
                "dropped_wrong_target": counters.dropped_wrong_target,
                "dropped_not_success": counters.dropped_not_success,
                "dropped_too_short": counters.dropped_too_short,
            }
        )
    return summaries


def _summary_from_labels(ref: SessionRef, labels: np.ndarray) -> Dict[str, object]:
    """Reconstruct a minimal summary from an already-cached session."""
    left = int(np.sum(labels == config.LABEL_TO_ID["left"]))
    right = int(np.sum(labels == config.LABEL_TO_ID["right"]))
    return {
        "subject": ref.subject,
        "session": ref.session,
        "key": ref.key,
        "n_trials_total": None,
        "n_kept": int(labels.size),
        "n_left": left,
        "n_right": right,
    }
