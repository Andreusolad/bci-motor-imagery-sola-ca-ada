"""Build the cross-experiment comparison once all three EEGSym runs are complete.

Reads ``metrics.json`` from every registered EEGSym experiment directory
(``eegsym_z_score/``, ``eegsym_running_exponential/``,
``eegsym_euclidean_alignment/``) and writes ``comparison_metrics.csv``,
``comparison_metrics.json`` and ``comparison_report.md`` into the EEGSym
comparison directory, including an automatic verdict on which normalization
method generalizes best cross-subject for this architecture.

Same table shape as the EEGNet comparison (src/study/compare.py), but reads
from and writes to entirely separate files/directories.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from ..utils import get_logger, save_json
from . import config as study_config

logger = get_logger()

_DISPLAY_NAMES: Dict[str, str] = {
    "z_score": "EEGSym + Z-score",
    "running_exponential": "EEGSym + Running Exponential Standardization",
    "euclidean_alignment": "EEGSym + Euclidean Alignment",
}

_COLUMNS: Tuple[str, ...] = (
    "Model",
    "Normalization",
    "Accuracy Crop",
    "Accuracy Trial",
    "Precision Trial",
    "Recall Trial",
    "F1 Trial",
    "ROC AUC Trial",
    "Num Parameters",
    "Model Size",
    "Training Time",
)

_VERDICT_METRICS: List[Tuple[str, str, float]] = [
    ("trial", "accuracy", 0.4),
    ("trial", "f1_macro", 0.3),
    ("trial", "roc_auc", 0.3),
]


def _load_metrics(method: str, window: str) -> Dict[str, object]:
    path = study_config.experiment_dir(method, window) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run experiment '{method}' (window={window}) first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _model_row(method: str, metrics: Dict[str, object]) -> Dict[str, object]:
    """One row of the comparison table for a single experiment."""
    crop, trial, model, timing = metrics["crop"], metrics["trial"], metrics["model"], metrics["timing"]
    return {
        "Model": _DISPLAY_NAMES.get(method, method),
        "Normalization": method,
        "Accuracy Crop": float(crop["accuracy"]),
        "Accuracy Trial": float(trial["accuracy"]),
        "Precision Trial": float(trial["precision_macro"]),
        "Recall Trial": float(trial["recall_macro"]),
        "F1 Trial": float(trial["f1_macro"]),
        "ROC AUC Trial": float(trial["roc_auc"]),
        "Num Parameters": int(model["n_params"]),
        "Model Size": int(model["model_size_bytes"]),
        "Training Time": float(timing["total_train_time_s"]),
    }


def _verdict(metrics_by_method: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """Weighted trial-level score per method; picks the best (or reports a tie)."""
    scores = {
        method: sum(w * float(metrics[sect][key]) for sect, key, w in _VERDICT_METRICS)
        for method, metrics in metrics_by_method.items()
    }
    best = max(scores.values())
    winners = [m for m, s in scores.items() if s == best]
    winner = winners[0] if len(winners) == 1 else "tie (" + ", ".join(sorted(winners)) + ")"
    return {
        "winner": winner,
        "scores": scores,
        "criterion": (
            "0.4 * trial_accuracy + 0.3 * trial_f1_macro + 0.3 * trial_roc_auc "
            "(trial-level = primary metrics)"
        ),
    }


def build_comparison(window: str = "legacy") -> Dict[str, object]:
    """Load every EEGSym experiment's metrics and write the comparison artifacts."""
    methods = list(study_config.NORMALIZATION_METHODS)
    metrics = {method: _load_metrics(method, window) for method in methods}
    rows = [_model_row(method, metrics[method]) for method in methods]
    verdict = _verdict(metrics)

    comparison_payload: Dict[str, object] = {
        "window": window,
        "methods": methods,
        "columns": list(_COLUMNS),
        "rows": rows,
        "verdict": verdict,
    }

    out_dir = study_config.comparison_dir(window)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "comparison_metrics.json", comparison_payload)
    _write_csv(out_dir / "comparison_metrics.csv", rows)
    _write_report(out_dir / "comparison_report.md", comparison_payload)
    logger.info("EEGSym comparison written to %s", out_dir)
    return comparison_payload


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %s", path)


def _fmt(column: str, value: object) -> str:
    if column in ("Accuracy Crop", "Accuracy Trial", "Precision Trial", "Recall Trial", "F1 Trial"):
        return f"{float(value):.4f}"
    if column == "ROC AUC Trial":
        return f"{float(value):.4f}"
    if column == "Num Parameters":
        return f"{int(value):,}"
    if column == "Model Size":
        return f"{int(value) / 1024:.1f} KB"
    if column == "Training Time":
        return f"{float(value):.1f} s"
    return str(value)


def _write_report(path: Path, comparison_payload: Dict[str, object]) -> None:
    rows = comparison_payload["rows"]
    verdict = comparison_payload["verdict"]
    methods = comparison_payload["methods"]

    lines = [
        "# EEGSym comparison: Z-score vs. Running Exponential Standardization vs. Euclidean Alignment",
        "",
        f"Same EEGSym architecture, hyper-parameters, optimizer (AdamW + weight decay), "
        f"label smoothing, callbacks, seed (42) and train/val/test split for all "
        f"{len(methods)} experiments. The only experimental variable is the "
        f"normalization method.",
        "",
        "## Metrics",
        "",
        "| " + " | ".join(_COLUMNS) + " |",
        "|" + "---|" * len(_COLUMNS),
    ]
    for row in rows:
        cells = [_fmt(col, row[col]) for col in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Verdict",
        "",
        f"**Best cross-subject generalization: `{verdict['winner']}`**",
        "",
        f"Weighted score ({verdict['criterion']}):",
    ]
    for method in methods:
        lines.append(f"- `{method}` = {verdict['scores'][method]:.4f}")
    lines += [
        "",
        "Trial-level metrics (mean of crop softmax probabilities per trial, then "
        "argmax) are the primary metrics for this study; crop-level accuracy is "
        "reported for reference only.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved %s", path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build the EEGSym normalization comparison.")
    parser.add_argument("--window", choices=study_config.WINDOWS, default="legacy")
    args = parser.parse_args()
    build_comparison(args.window)


if __name__ == "__main__":
    main()
