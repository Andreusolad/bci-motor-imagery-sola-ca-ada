"""``second_ml`` -- 3-class (REST/LEFT/RIGHT) extension of the winning BCI model.

This package deliberately *reuses* the validated ``first_ml`` pipeline
(preprocessing, EEGSym architecture, Euclidean Alignment, the canonical
subject-level ``dataset_split.json``) instead of re-implementing it, so the
only things that change here are:

1. a third class (REST / idle), sourced from the real pre-cue baseline
   ``[-2000, 0)`` ms of every trial, and
2. a continuous-EEG evaluation scenario built by concatenating real segments.

Importing ``lib`` puts the route_b root on ``sys.path`` so both ``import
src.<...>`` (the shared core) and ``import lib.<...>`` resolve. The shared core
under ``src`` is only ever read, never modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Appended (not inserted at 0) so a script's own directory keeps priority;
    # the shared core package is ``src`` and never collides with ``lib``.
    sys.path.append(str(_REPO_ROOT))
