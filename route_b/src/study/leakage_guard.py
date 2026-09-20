"""Full anti-leakage verification: subject / session / trial / crop levels.

Layers on top of :func:`first_ml.src.split.verify_split` (subject/session
disjointness, already validated against the reused ``dataset_split.json``)
with the two extra levels this study introduces: trial identity and crop
identity. If any check fails, a :class:`LeakageError` is raised and the
caller must abort training immediately -- no experiment may start on data
that failed this guard.
"""
from __future__ import annotations

from typing import Dict, List

from ..split import LeakageError, verify_split
from ..utils import get_logger
from .crops import StudyCropArrays
from .data_loading import LoadedTrial

logger = get_logger()


def _verify_disjoint(named_id_lists: Dict[str, List[str]], level: str) -> None:
    """Raise if any id is duplicated within a split or shared across splits."""
    names = list(named_id_lists.keys())
    id_sets: Dict[str, set] = {}
    for name in names:
        ids = named_id_lists[name]
        ids_set = set(ids)
        if len(ids_set) != len(ids):
            raise LeakageError(f"Duplicate {level} id(s) found within split '{name}'.")
        id_sets[name] = ids_set

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = id_sets[a] & id_sets[b]
            if shared:
                example = sorted(shared)[:3]
                raise LeakageError(
                    f"{len(shared)} {level}(s) shared between '{a}' and '{b}', "
                    f"e.g. {example}. Aborting to prevent data leakage."
                )
    logger.info("%s-level anti-leakage check passed across %s.", level.capitalize(), names)


def verify_full_leakage(
    split_payload: Dict[str, object],
    trials_by_split: Dict[str, List[LoadedTrial]],
    crops_by_split: Dict[str, StudyCropArrays],
) -> None:
    """Run every anti-leakage check (subject/session/trial/crop) before training.

    Raises :class:`LeakageError` immediately on the first violation found;
    the caller (``run_experiment.py``) must not proceed to training if this
    raises.
    """
    verify_split(split_payload)  # subject + session disjointness

    trial_ids = {name: [t.trial_id for t in trials] for name, trials in trials_by_split.items()}
    _verify_disjoint(trial_ids, "trial")

    crop_ids = {name: crops.crop_ids for name, crops in crops_by_split.items()}
    _verify_disjoint(crop_ids, "crop")

    # Every crop's trial_id must belong to a trial actually present in the
    # same split (catches any accidental cross-split mixing during cropping).
    for name, crops in crops_by_split.items():
        valid_trial_ids = set(trial_ids[name])
        stray = set(crops.trial_ids) - valid_trial_ids
        if stray:
            raise LeakageError(
                f"Split '{name}' contains crops referencing trial id(s) not in "
                f"that split's trial set: {sorted(stray)[:3]}."
            )

    logger.info("Full anti-leakage guard passed: subject/session/trial/crop are all disjoint.")
