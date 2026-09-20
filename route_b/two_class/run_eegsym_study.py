"""Orchestrator for the EEGSym normalization-comparison study.

Sibling of ``run_study.py`` (EEGNet): same three normalization methods
(Z-score, Running Exponential Standardization, Euclidean Alignment), same
training protocol and split, but built exclusively from ``src/eegsym_study/``,
its own package that never imports from ``src/study/``.

Usage (run from route_b/two_class/, with BCI_DATA set)::

    python run_eegsym_study.py                       # all experiments + comparison
    python run_eegsym_study.py --only z_score
    python run_eegsym_study.py --only running_exponential
    python run_eegsym_study.py --only euclidean_alignment
    python run_eegsym_study.py --compare-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eegsym_study import config as study_config
from src.eegsym_study.comparability import assert_comparable, build_experiment_config
from src.eegsym_study.compare import build_comparison
from src.eegsym_study.run_experiment import run_experiment
from src.study.artifacts import verify_experiment_outputs
from src.utils import get_logger

logger = get_logger()


def preflight() -> None:
    """Assert every EEGSym experiment is methodologically comparable before training.

    Only the shared (data-independent) parts of the configuration (EEGSym
    architecture, training hyper-parameters, dataset/crop settings) need to
    be checked here; they never depend on which normalization method will be
    used, so this can run before any data is touched.
    """
    configs = [
        build_experiment_config(method, {}) for method in study_config.NORMALIZATION_METHODS
    ]
    assert_comparable(*configs)
    logger.info("Pre-flight scientific-comparability check passed (EEGSym).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EEGSym normalization-comparison study.")
    parser.add_argument(
        "--only", choices=study_config.NORMALIZATION_METHODS, default=None,
        help="Run only one experiment instead of all three.",
    )
    parser.add_argument(
        "--compare-only", action="store_true",
        help="Skip training; only rebuild the comparison from existing metrics.json files.",
    )
    args = parser.parse_args()

    if args.compare_only:
        build_comparison()
        return

    preflight()

    methods = [args.only] if args.only else list(study_config.NORMALIZATION_METHODS)
    for method in methods:
        logger.info("##### Starting EEGSym experiment: %s #####", method)
        run_experiment(method)
        verify_experiment_outputs(study_config.experiment_dir(method))
        logger.info("##### EEGSym experiment %s verified complete. #####", method)

    if not args.only:
        build_comparison()

    logger.info("EEGSym study complete.")


if __name__ == "__main__":
    main()
