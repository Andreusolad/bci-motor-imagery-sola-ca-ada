"""Pre-flight scientific-comparability check.

Before any experiment is allowed to train, this asserts that every
experiment configuration passed in is identical in every dimension *except*
the normalization method: EEGNet architecture, hyper-parameters, optimizer,
callbacks, seed, max epochs, split and evaluation procedure. If anything else
differs, a :class:`ComparabilityError` is raised and training must not start.

Works for any number of experiments (two today, three once Z-score is
added), so adding a new normalization method never requires touching this
check's logic -- only calling it with one more config.
"""
from __future__ import annotations

from typing import Dict

from ..utils import get_logger

logger = get_logger()

_SHARED_KEYS = ("eegnet", "training", "dataset")


class ComparabilityError(RuntimeError):
    """Raised when experiment configs differ outside the allowed field."""


def build_experiment_config(method: str, normalization_params: Dict[str, object]) -> Dict[str, object]:
    """Assemble the full, JSON-serialisable configuration for one experiment."""
    from . import config as study_config

    return {
        "normalization_method": method,
        "normalization_params": normalization_params,
        "eegnet": study_config.eegnet_as_dict(),
        "training": study_config.training_as_dict(),
        "dataset": study_config.dataset_as_dict(),
    }


def assert_comparable(*configs: Dict[str, object]) -> None:
    """Raise :class:`ComparabilityError` unless every config only differs
    in ``normalization_method`` / ``normalization_params``.

    Accepts any number of experiment configs (>= 2).
    """
    if len(configs) < 2:
        raise ValueError("Need at least two experiment configs to compare.")

    reference = configs[0]
    for key in _SHARED_KEYS:
        for other in configs[1:]:
            if other[key] != reference[key]:
                raise ComparabilityError(
                    f"Experiment configs differ in shared field '{key}', which must be "
                    f"identical: {reference[key]!r} != {other[key]!r}"
                )

    methods = [c["normalization_method"] for c in configs]
    if len(set(methods)) != len(methods):
        raise ComparabilityError(
            "Two or more experiment configs use the same normalization method; "
            "the study requires exactly one method per experiment."
        )
    logger.info(
        "Comparability check passed: %d experiments differ only in normalization "
        "method (%s).", len(configs), ", ".join(methods),
    )
