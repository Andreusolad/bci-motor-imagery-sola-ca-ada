"""Cross-cutting helpers: logging, reproducibility, GPU probing and artefact I/O."""
from __future__ import annotations

import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_LOGGER_NAME = "approach_a"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a module-wide logger with a single stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow (if imported) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # TensorFlow is optional at import time for the data-only stages.
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:  # pragma: no cover - TF always present at train time.
        pass


def probe_gpu() -> Dict[str, Any]:
    """Report whether TensorFlow sees a usable GPU.

    Fulfils the "GPU encontrada / Entrenamiento en CPU" requirement. On native
    Windows, TensorFlow >= 2.11 is CPU-only (GPU support needs WSL2), so this
    will typically report the CPU path here.
    """
    logger = get_logger()
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    info: Dict[str, Any] = {
        "tensorflow_version": tf.__version__,
        "gpu_available": bool(gpus),
        "gpu_devices": [g.name for g in gpus],
    }
    if gpus:
        # Grow memory on demand rather than grabbing the whole card up front.
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:  # already initialised
                logger.warning("Could not set memory growth: %s", exc)
        logger.info("GPU encontrada: %s", ", ".join(info["gpu_devices"]))
    else:
        logger.info("Entrenamiento en CPU (no se ha detectado GPU para TensorFlow).")
    return info


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write ``payload`` as pretty-printed, UTF-8 JSON, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_pickle(path: Path, obj: Any) -> None:
    """Pickle ``obj`` to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh)


def to_native(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
