"""3-class metrics + BCI-specific metrics (FPR/FNR of the REST vs MI decision).

All metrics come from real predictions; nothing is estimated. ``y`` uses the
3-class ids 0=REST, 1=LEFT, 2=RIGHT.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config3 as C


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: Optional[np.ndarray] = None) -> Dict[str, object]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(C.N_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Per-class accuracy == recall == diagonal / row-sum.
    row_sums = cm.sum(axis=1)
    per_class_acc = np.divide(np.diag(cm), row_sums,
                              out=np.zeros(C.N_CLASSES, dtype=float),
                              where=row_sums > 0)

    out: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_per_class": precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "f1_per_class": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "per_class_accuracy": {C.CLASS_NAMES[i]: float(per_class_acc[i]) for i in labels},
        "confusion_matrix": cm.tolist(),
        "support_per_class": {C.CLASS_NAMES[i]: int(row_sums[i]) for i in labels},
        "class_names": list(C.CLASS_NAMES),
        "n_samples": int(len(y_true)),
    }

    if y_prob is not None and len(np.unique(y_true)) == C.N_CLASSES:
        try:
            out["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
            )
        except ValueError:
            out["roc_auc_ovr_macro"] = None
    return out


def bci_continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """REST-vs-MI decision quality for a continuous BCI.

    * FPR = P(predict MI | truly REST)  -> false activations while idle.
    * FNR = P(predict REST | truly MI)  -> missed intentions.
    Also the collapsed 2-class (REST vs MI) accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_rest_true = y_true == C.REST_ID
    is_rest_pred = y_pred == C.REST_ID
    is_mi_true = ~is_rest_true
    is_mi_pred = ~is_rest_pred

    n_rest = int(is_rest_true.sum())
    n_mi = int(is_mi_true.sum())
    fp = int((is_rest_true & is_mi_pred).sum())   # REST predicted as MI
    fn = int((is_mi_true & is_rest_pred).sum())   # MI predicted as REST

    binary_correct = int((is_rest_true == is_rest_pred).sum())
    return {
        "false_positive_rate_rest_as_mi": (fp / n_rest) if n_rest else float("nan"),
        "false_negative_rate_mi_as_rest": (fn / n_mi) if n_mi else float("nan"),
        "n_rest_windows": n_rest,
        "n_mi_windows": n_mi,
        "rest_vs_mi_accuracy": (binary_correct / len(y_true)) if len(y_true) else float("nan"),
    }
