"""Orchestrate the five remaining corrected-window ([2000, 2000+triallength] ms)
normalization experiments, end to end, in one sequence.

EEGSym + Running Exponential Standardization on the corrected window is assumed
already trained and is skipped here. This script trains the other five
combinations:

    1. EEGNet + Z-score                     (corrected)
    2. EEGNet + Running Exponential Std.     (corrected)
    3. EEGNet + Euclidean Alignment          (corrected)
    4. EEGSym + Z-score                      (corrected)
    5. EEGSym + Euclidean Alignment          (corrected)

Then builds both comparison reports (EEGNet-only, EEGSym-only) for the
corrected window. Runs are sequential (TensorFlow runs on CPU, so parallelising
would only cause contention). A machine-readable
``corrected_window_study_progress.json`` is written at every phase transition.

Usage (run from route_b/two_class/, with BCI_DATA set)::

    python run_corrected_window_study.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eegsym_study.compare import build_comparison as build_eegsym_comparison
from src.eegsym_study.run_experiment import run_experiment as run_eegsym
from src.study.compare import build_comparison as build_eegnet_comparison
from src.study.run_experiment import run_experiment as run_eegnet
from src.utils import get_logger

logger = get_logger()

ROOT = Path(__file__).resolve().parent
PROGRESS_JSON = ROOT / "corrected_window_study_progress.json"

_PHASES = [
    "eegnet_z_score", "eegnet_running_exponential", "eegnet_euclidean_alignment",
    "eegsym_z_score", "eegsym_euclidean_alignment",
    "eegnet_comparison", "eegsym_comparison",
]


def _write_progress(phase: str, status: str, extra: dict | None = None) -> None:
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "phase_index": _PHASES.index(phase) if phase in _PHASES else -1,
        "total_phases": len(_PHASES),
        "status": status,
    }
    if extra:
        payload.update(extra)
    PROGRESS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    try:
        _write_progress("eegnet_z_score", "running")
        run_eegnet("z_score", window="corrected")
        _write_progress("eegnet_z_score", "done")

        _write_progress("eegnet_running_exponential", "running")
        run_eegnet("running_exponential", window="corrected")
        _write_progress("eegnet_running_exponential", "done")

        _write_progress("eegnet_euclidean_alignment", "running")
        run_eegnet("euclidean_alignment", window="corrected")
        _write_progress("eegnet_euclidean_alignment", "done")

        _write_progress("eegsym_z_score", "running")
        run_eegsym("z_score", window="corrected")
        _write_progress("eegsym_z_score", "done")

        _write_progress("eegsym_euclidean_alignment", "running")
        run_eegsym("euclidean_alignment", window="corrected")
        _write_progress("eegsym_euclidean_alignment", "done")

        _write_progress("eegnet_comparison", "running")
        build_eegnet_comparison(window="corrected")
        _write_progress("eegnet_comparison", "done")

        _write_progress("eegsym_comparison", "running")
        build_eegsym_comparison(window="corrected")
        _write_progress("eegsym_comparison", "done")

        elapsed = time.perf_counter() - t0
        logger.info("=== CORRECTED-WINDOW STUDY COMPLETE in %.1fs (%.2f h) ===", elapsed, elapsed / 3600)
        _write_progress("all_done", "all_done", {"elapsed_s": elapsed})
    except Exception:  # noqa: BLE001 - persist the failure for the outside monitor
        logger.error("CORRECTED-WINDOW STUDY FAILED:\n%s", traceback.format_exc())
        _write_progress("FAILED", "error", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
