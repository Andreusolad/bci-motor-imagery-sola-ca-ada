"""Build artificial *continuous* EEG by concatenating REAL Stieger segments.

No signal is synthesized and no generative model is used: every sample comes
from a real preprocessed segment of the chosen subject (MI imagery windows and
pre-cue REST baselines). One 300 s trial has the structure

    R  M  R  M  R  M  R  M  R  M  R  M  R          (6 MI, 7 REST periods)

with MI durations taken verbatim from real trials, REST durations drawn at
random so the whole trial is *exactly* 300 s (75000 samples @250 Hz), and no
overlap between events. Everything is deterministic given the RNG seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from . import config3 as C
from .segments import SubjectPool


@dataclass
class ContinuousTrial:
    subject: str
    trial_index: int
    seed: int
    signal: np.ndarray        # (C, 75000) float32, preprocessed @250 Hz, PRE-EA
    ground_truth: np.ndarray  # (75000,) int8  (0=REST, 1=LEFT, 2=RIGHT)
    events: List[Dict]        # per block: start/end sample+seconds, class


def select_subjects(split_payload: dict, n: int = C.CONT_N_SUBJECTS,
                    seed: int = C.CONT_SUBJECT_SELECTION_SEED) -> List[str]:
    """Deterministically pick ``n`` TEST subjects (unseen in training)."""
    test_subjects = sorted(split_payload["splits"]["test"]["subjects"],
                           key=lambda s: int(s[1:]))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(test_subjects), size=n, replace=False)
    return sorted(chosen.tolist(), key=lambda s: int(s[1:]))


def _mi_class_sequence(rng: np.random.Generator) -> List[int]:
    """6 MI periods, exactly balanced 3 LEFT / 3 RIGHT, order shuffled."""
    half = C.CONT_N_MI_PER_TRIAL // 2
    seq = np.array([C.LEFT_ID] * half + [C.RIGHT_ID] * (C.CONT_N_MI_PER_TRIAL - half))
    rng.shuffle(seq)
    return seq.tolist()


def _partition(total: int, n_parts: int, min_each: int,
               rng: np.random.Generator) -> np.ndarray:
    """Random integer partition of ``total`` into ``n_parts`` parts >= min_each."""
    if total < n_parts * min_each:
        raise ValueError(f"Cannot fit {n_parts} REST periods of >= {min_each} "
                         f"samples into {total} samples.")
    extra = total - n_parts * min_each
    counts = rng.multinomial(extra, [1.0 / n_parts] * n_parts)
    return min_each + counts


def _make_rest_period(pool_rest: List[np.ndarray], n_samples: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Concatenate random 2 s REST segments, trimmed to exactly ``n_samples``."""
    chunks: List[np.ndarray] = []
    got = 0
    while got < n_samples:
        seg = pool_rest[int(rng.integers(len(pool_rest)))]
        chunks.append(seg)
        got += seg.shape[1]
    return np.concatenate(chunks, axis=1)[:, :n_samples].astype(np.float32)


def build_continuous_trial(pool: SubjectPool, trial_index: int,
                           rng: np.random.Generator, seed: int = 0) -> ContinuousTrial:
    if not pool.mi_left or not pool.mi_right or not pool.rest:
        raise ValueError(f"Subject {pool.subject} lacks a required class pool.")

    # --- choose 6 real MI segments (real durations) --- #
    classes = _mi_class_sequence(rng)
    mi_segments: List[np.ndarray] = []
    for cls in classes:
        src = pool.mi_left if cls == C.LEFT_ID else pool.mi_right
        mi_segments.append(src[int(rng.integers(len(src)))])
    mi_total = sum(seg.shape[1] for seg in mi_segments)

    remaining = C.CONT_TRIAL_SAMPLES - mi_total
    rest_sizes = _partition(remaining, C.CONT_N_REST_PERIODS, C.CONT_MIN_REST_SAMPLES, rng)

    # --- assemble  R M R M ... M R --- #
    blocks: List[np.ndarray] = []
    labels: List[int] = []
    for k in range(C.CONT_N_MI_PER_TRIAL):
        blocks.append(_make_rest_period(pool.rest, int(rest_sizes[k]), rng))
        labels.append(C.REST_ID)
        blocks.append(mi_segments[k])
        labels.append(classes[k])
    blocks.append(_make_rest_period(pool.rest, int(rest_sizes[-1]), rng))
    labels.append(C.REST_ID)

    signal = np.concatenate(blocks, axis=1).astype(np.float32)
    assert signal.shape[1] == C.CONT_TRIAL_SAMPLES, signal.shape

    gt = np.empty(C.CONT_TRIAL_SAMPLES, dtype=np.int8)
    events: List[Dict] = []
    cursor = 0
    for blk, lab in zip(blocks, labels):
        L = blk.shape[1]
        gt[cursor:cursor + L] = lab
        events.append({
            "start_sample": int(cursor), "end_sample": int(cursor + L),
            "start_s": cursor / C.FS_TARGET, "end_s": (cursor + L) / C.FS_TARGET,
            "class_id": int(lab), "class_name": C.CLASS_NAMES[lab],
        })
        cursor += L

    return ContinuousTrial(subject=pool.subject, trial_index=trial_index, seed=seed,
                           signal=signal, ground_truth=gt, events=events)
