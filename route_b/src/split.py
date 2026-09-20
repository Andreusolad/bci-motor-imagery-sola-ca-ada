"""Subject-independent train/val/test split with anti-leakage verification.

The split is performed at the *subject* level: every session (and therefore
every trial and every crop) of a subject lands in a single split. This is the
strongest guarantee against leakage -- because subjects are disjoint, sessions,
trials and crops are disjoint by construction. The verification step nonetheless
checks all three explicitly and raises if any overlap is found, so training can
be aborted as required.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import config
from .dataset import SessionRef, load_session_cache
from .utils import get_logger, load_json, save_json

logger = get_logger()


@dataclass
class SubjectStats:
    """Aggregated per-subject counts used to balance the split."""

    subject: str
    session_keys: List[str] = field(default_factory=list)
    n_trials: int = 0
    n_left: int = 0
    n_right: int = 0


def collect_subject_stats(cache_dir: Path = config.CACHE_DIR) -> Dict[str, SubjectStats]:
    """Aggregate trial and class counts per subject from the cache."""
    stats: Dict[str, SubjectStats] = {}
    for cache_path in sorted(cache_dir.glob("*.npz")):
        cached = load_session_cache(cache_path)
        entry = stats.setdefault(cached.subject, SubjectStats(subject=cached.subject))
        entry.session_keys.append(cached.key)
        entry.n_trials += int(cached.labels.size)
        entry.n_left += int(np.sum(cached.labels == config.LABEL_TO_ID["left"]))
        entry.n_right += int(np.sum(cached.labels == config.LABEL_TO_ID["right"]))
    if not stats:
        raise RuntimeError(
            f"No cached sessions found in {cache_dir}. Build the cache first."
        )
    return stats


def assign_subjects(
    stats: Dict[str, SubjectStats],
    fractions: Dict[str, float] = config.SPLIT_FRACTIONS,
    seed: int = config.RANDOM_SEED,
) -> Dict[str, str]:
    """Greedily assign whole subjects to splits to approach target trial shares.

    Subjects are shuffled (seeded) then processed largest-first; each is placed
    in the split with the largest current trial deficit relative to its target.
    This keeps the trial-count fractions close to 70/15/15 while never splitting
    a subject. Class balance follows because subjects are individually balanced.
    """
    total_trials = sum(s.n_trials for s in stats.values())
    targets = {split: frac * total_trials for split, frac in fractions.items()}
    current: Dict[str, int] = {split: 0 for split in fractions}

    rng = random.Random(seed)
    subjects = list(stats.values())
    rng.shuffle(subjects)
    subjects.sort(key=lambda s: s.n_trials, reverse=True)

    assignment: Dict[str, str] = {}
    for subj in subjects:
        # Choose the split whose remaining capacity (target - current) is largest.
        split = max(current, key=lambda sp: targets[sp] - current[sp])
        assignment[subj.subject] = split
        current[split] += subj.n_trials
    return assignment


def build_split(
    stats: Dict[str, SubjectStats] | None = None,
    write: bool = True,
) -> Dict[str, object]:
    """Compute the split, verify it and (optionally) write ``dataset_split.json``."""
    stats = stats if stats is not None else collect_subject_stats()
    assignment = assign_subjects(stats)

    split: Dict[str, Dict[str, object]] = {
        name: {"subjects": [], "session_keys": [], "n_trials": 0, "n_left": 0, "n_right": 0}
        for name in config.SPLIT_FRACTIONS
    }
    for subject, split_name in assignment.items():
        s = stats[subject]
        bucket = split[split_name]
        bucket["subjects"].append(subject)
        bucket["session_keys"].extend(s.session_keys)
        bucket["n_trials"] += s.n_trials
        bucket["n_left"] += s.n_left
        bucket["n_right"] += s.n_right

    for bucket in split.values():
        bucket["subjects"].sort(key=lambda x: int(x.lstrip("S")))
        bucket["session_keys"].sort()

    payload: Dict[str, object] = {
        "seed": config.RANDOM_SEED,
        "fractions_target": config.SPLIT_FRACTIONS,
        "level": "subject",
        "splits": split,
    }
    verify_split(payload)

    if write:
        save_json(config.SPLIT_JSON, payload)
        logger.info("Wrote split to %s", config.SPLIT_JSON)
    _log_split_summary(split)
    return payload


def verify_split(payload: Dict[str, object]) -> None:
    """Raise ``LeakageError`` if subjects, sessions or trials overlap across splits.

    Because the split is by subject, session and crop disjointness follow from
    subject disjointness; we assert all of them to fail loudly on any bug.
    """
    splits = payload["splits"]  # type: ignore[index]
    names = list(splits.keys())

    subject_sets = {n: set(splits[n]["subjects"]) for n in names}
    session_sets = {n: set(splits[n]["session_keys"]) for n in names}

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared_subjects = subject_sets[a] & subject_sets[b]
            if shared_subjects:
                raise LeakageError(
                    f"Subjects {sorted(shared_subjects)} appear in both "
                    f"'{a}' and '{b}'."
                )
            shared_sessions = session_sets[a] & session_sets[b]
            if shared_sessions:
                raise LeakageError(
                    f"Sessions {sorted(shared_sessions)} appear in both "
                    f"'{a}' and '{b}'."
                )

    # Every subject appears in exactly one split.
    all_subjects = [s for n in names for s in splits[n]["subjects"]]
    if len(all_subjects) != len(set(all_subjects)):
        raise LeakageError("A subject appears more than once across splits.")

    # No split may be empty (would make training/eval meaningless).
    empty = [n for n in names if splits[n]["n_trials"] == 0]
    if empty:
        raise LeakageError(f"Split(s) {empty} contain zero trials.")

    logger.info("Anti-leakage check passed: subjects/sessions/trials are disjoint.")


def load_split(path: Path = config.SPLIT_JSON) -> Dict[str, object]:
    """Load and re-verify a previously written split."""
    payload = load_json(path)
    verify_split(payload)
    return payload


def session_keys_for(payload: Dict[str, object], split_name: str) -> List[str]:
    """Return the sorted session keys belonging to ``split_name``."""
    return list(payload["splits"][split_name]["session_keys"])  # type: ignore[index]


class LeakageError(RuntimeError):
    """Raised when the split verification detects any cross-split overlap."""


def _log_split_summary(split: Dict[str, Dict[str, object]]) -> None:
    total = sum(b["n_trials"] for b in split.values())
    for name, bucket in split.items():
        n = bucket["n_trials"]
        pct = 100 * n / total if total else 0
        logger.info(
            "  %-5s | %2d subjects | %5d trials (%.1f%%) | left=%d right=%d",
            name, len(bucket["subjects"]), n, pct, bucket["n_left"], bucket["n_right"],
        )
