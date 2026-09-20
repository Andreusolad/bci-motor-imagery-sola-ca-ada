r"""Part 1 --- train a 3-class REST/LEFT/RIGHT classifier from scratch.

Generalized over ``(architecture, normalization)`` so the same protocol can be
run for the winner (EEGSym + Euclidean Alignment) *and* for EEGNet + Running
Exponential Standardization. Everything else is identical to the winning model:
same subject split (70/15/15, seed 42), same optimizer / scheduler / callbacks,
same 0.1 label smoothing; only the output width (3) and the 3-class loss change.

REST is the real pre-cue baseline [-2000, 0) ms (idle, no imagery), balanced
against LEFT/RIGHT. The normalization is fit on the training set only (EA) or is
stateless/causal (running exponential); it is saved so Part 2 reuses it exactly.

Usage (run from route_b/continuous/, BCI_DATA set):
    python train_3class.py --arch eegnet --method running_exponential
    python train_3class.py --arch eegsym  --method euclidean_alignment
"""
from __future__ import annotations
import sys

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import config3 as C
from lib import crops3, metrics3, model3, normalize3, viz3
from lib.segments import Segment, extract_segments_for_sessions

from src.split import load_split, session_keys_for  # noqa: E402
from src.utils import get_logger, probe_gpu, save_json, save_pickle, set_global_seed  # noqa: E402

logger = get_logger()


def _attach_log(out: Path) -> logging.FileHandler:
    out.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(out / "training.log", mode="w", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    return h


def _assert_disjoint(name: str, sets: Dict[str, set]) -> None:
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sets[a] & sets[b]
            if shared:
                raise RuntimeError(f"LEAKAGE: {len(shared)} {name}(s) shared between "
                                   f"{a} and {b}, e.g. {sorted(shared)[:3]}")
    logger.info("Anti-leakage OK at %s level across %s.", name, names)


def _crop_counts(y: np.ndarray) -> Dict[str, int]:
    return {C.CLASS_NAMES[i]: int((y == i).sum()) for i in range(C.N_CLASSES)}


def run(arch: str, method: str) -> None:
    out = C.model_dir(arch, method)
    C.ensure_dirs()
    handler = _attach_log(out)
    t0 = time.perf_counter()
    try:
        logger.info("=== PART 1: 3-class %s + %s -> %s ===", arch, method, out)
        set_global_seed(C.RANDOM_SEED)
        gpu = probe_gpu()
        split = load_split(C.SPLIT_JSON)

        # 1) Extract MI + REST segments per split. ---------------------------- #
        raw_segs: Dict[str, List[Segment]] = {}
        for name in ("train", "val", "test"):
            logger.info("=== Extracting %s segments ===", name)
            raw_segs[name] = extract_segments_for_sessions(session_keys_for(split, name))

        # 2) Balance REST against LEFT/RIGHT (seeded, whole-segment). ---------- #
        balanced: Dict[str, List[Segment]] = {}
        balance_info: Dict[str, Dict] = {}
        for name, segs in raw_segs.items():
            balanced[name], balance_info[name] = crops3.balance_rest(segs, seed=C.RANDOM_SEED)
            logger.info("[%s] balance: %s", name, balance_info[name])

        # 3) Build the normalizer (EA fit on train / running-exp stateless). -- #
        logger.info("Building normalizer '%s' on %d training segments...",
                    method, len(balanced["train"]))
        normalize_fn, saveable, norm_params = normalize3.build_normalizer(
            method, [s.signal for s in balanced["train"]])
        normalize3.save_normalizer(method, saveable, out)

        # 4) Build crops (normalize per whole segment, then slice). ----------- #
        crops = {name: crops3.build_crops(segs, normalize_fn) for name, segs in balanced.items()}
        for name, c in crops.items():
            logger.info("[%s] crops: %d  %s", name, len(c), _crop_counts(c.y))

        # 5) Full anti-leakage guard (trial + crop, across splits). ----------- #
        _assert_disjoint("trial", {n: set(c.trial_ids) for n, c in crops.items()})
        _assert_disjoint("crop", {n: set(c.crop_ids) for n, c in crops.items()})

        # 6) Artefacts. ------------------------------------------------------- #
        save_json(out / "dataset_statistics.json", {
            "config": {
                "architecture": arch, "normalization": method,
                "normalization_params": norm_params,
                "classes": {name: i for i, name in enumerate(C.CLASS_NAMES)},
                "rest_source": "pre-cue baseline [-2000, 0) ms (idle, no imagery)",
                "mi_window": "corrected [2000, 2000+triallength] ms (feedback onset)",
                "motor_channels": list(C.MOTOR_CHANNELS), "fs_target": C.FS_TARGET,
                "crop_samples": C.CROP_SAMPLES, "crop_stride_samples": C.CROP_STRIDE_SAMPLES,
                "crop_overlap": C.CROP_OVERLAP, "split_json": str(C.SPLIT_JSON),
                "split_seed": C.RANDOM_SEED,
            },
            "splits": {name: {
                "subjects": sorted({s.subject for s in balanced[name]}, key=lambda s: int(s[1:])),
                "n_segments": len(balanced[name]), "n_crops": len(crops[name]),
                "crops_per_class": _crop_counts(crops[name].y), "balance": balance_info[name],
            } for name in ("train", "val", "test")},
        })
        save_json(out / "experiment_config.json", {
            "study": "3class_rest_left_right", "architecture": arch, "normalization": method,
            "trained_from": "scratch", "training": model3.training_hparams(arch), "gpu": gpu,
        })

        # 7) Train from scratch (winner protocol). ---------------------------- #
        set_global_seed(C.RANDOM_SEED)
        net = model3.compile_model3(model3.build_model3(arch))
        n_params = int(sum(w.numpy().size for w in net.trainable_weights))
        net.summary(print_fn=logger.info)
        logger.info("3-class %s trainable params: %d", arch, n_params)

        from src.eegsym_study import config as tcfg
        epoch_times: List[float] = []
        train_ds = crops3.make_tf_dataset(crops["train"], tcfg.TRAIN.batch_size, True, C.RANDOM_SEED)
        val_ds = crops3.make_tf_dataset(crops["val"], tcfg.TRAIN.batch_size, False, C.RANDOM_SEED)

        t_train = time.perf_counter()
        history = net.fit(train_ds, validation_data=val_ds, epochs=tcfg.TRAIN.epochs,
                          callbacks=model3.build_callbacks(out, epoch_times), shuffle=False, verbose=2)
        train_time = time.perf_counter() - t_train

        net.save_weights(out / "weights.weights.h5")
        save_pickle(out / "history.pkl", history.history)
        viz3.plot_training_curves(history.history, out / "training_curves.png")

        # 8) Evaluate on the held-out test split (crop + trial). -------------- #
        logger.info("=== Evaluating on test split ===")
        test_ds = crops3.make_tf_dataset(crops["test"], 512, False, 0)
        crop_probs = net.predict(test_ds, verbose=0)
        crop_true = crops["test"].y
        crop_metrics = metrics3.compute_metrics(crop_true, crop_probs.argmax(1), crop_probs)

        tr_true, tr_pred, tr_prob, tr_counts = crops3.aggregate_by_trial(
            crop_probs, crops["test"].trial_ids, crop_true)
        trial_metrics = metrics3.compute_metrics(tr_true, tr_pred, tr_prob)
        trial_metrics["n_trials"] = int(len(tr_true))
        trial_metrics["mean_crops_per_trial"] = float(np.mean(tr_counts))

        cm = np.asarray(trial_metrics["confusion_matrix"])
        viz3.plot_confusion_matrix(cm, out / "confusion_matrix.png",
                                   title=f"3-class confusion -- {arch}+{C.NORM_SHORT[method]} (test trials)")
        viz3.plot_confusion_matrix(cm, out / "confusion_matrix_normalized.png",
                                   title=f"3-class confusion -- {arch}+{C.NORM_SHORT[method]} (normalized)",
                                   normalize=True)
        viz3.plot_prob_histograms(tr_true, tr_prob, out / "prob_histograms.png")
        _write_report(out / "classification_report.txt", crop_metrics, trial_metrics)

        metrics = {
            "architecture": arch, "normalization": method, "n_params": n_params, "gpu": gpu,
            "timing": {"total_train_time_s": train_time, "n_epochs_run": len(epoch_times),
                       "mean_epoch_time_s": float(np.mean(epoch_times)) if epoch_times else 0.0},
            "crop": crop_metrics, "trial": trial_metrics,
            "comparison_to_2class": _compare_to_2class(arch, method),
        }
        save_json(out / "metrics.json", metrics)

        logger.info("TRIAL: acc=%.4f bal_acc=%.4f f1=%.4f | per-class acc=%s",
                    trial_metrics["accuracy"], trial_metrics["balanced_accuracy"],
                    trial_metrics["f1_macro"], trial_metrics["per_class_accuracy"])
        logger.info("=== PART 1 COMPLETE in %.1fs -> %s ===", time.perf_counter() - t0, out)
    finally:
        logger.removeHandler(handler)
        handler.close()


def _write_report(path: Path, crop_m: Dict, trial_m: Dict) -> None:
    def block(title, m):
        lines = [f"{'=' * 60}", title, "=" * 60,
                 f"accuracy           : {m['accuracy']:.4f}",
                 f"balanced accuracy  : {m['balanced_accuracy']:.4f}",
                 f"precision (macro)  : {m['precision_macro']:.4f}",
                 f"recall (macro)     : {m['recall_macro']:.4f}",
                 f"f1 (macro)         : {m['f1_macro']:.4f}",
                 "per-class accuracy :"]
        for name, v in m["per_class_accuracy"].items():
            lines.append(f"    {name:5s}: {v:.4f}  (support={m['support_per_class'][name]})")
        lines.append("confusion matrix (rows=true REST/LEFT/RIGHT, cols=pred):")
        for row in m["confusion_matrix"]:
            lines.append("    " + " ".join(f"{v:6d}" for v in row))
        return "\n".join(lines)
    path.write_text(block("TRIAL-LEVEL (primary)", trial_m) + "\n\n" +
                    block("CROP-LEVEL (secondary)", crop_m) + "\n", encoding="utf-8")


def _compare_2class_path(arch: str, method: str) -> Path:
    return (Path(__file__).resolve().parents[1] / "experiments" / "corrected_window"
            / "normalization" / arch / method / "metrics.json")


def _compare_to_2class(arch: str, method: str) -> Dict[str, object]:
    out: Dict[str, object] = {"note": (
        "The matching 2-class model only decides LEFT vs RIGHT and never sees "
        "REST; the 3-class model additionally rejects idle, so numbers are not "
        "directly comparable class-for-class.")}
    try:
        prev = json.loads(_compare_2class_path(arch, method).read_text(encoding="utf-8"))
        out["winner_2class_trial"] = {"architecture": arch, "normalization": method,
                                      "accuracy": prev["trial"]["accuracy"],
                                      "f1_macro": prev["trial"]["f1_macro"],
                                      "roc_auc": prev["trial"].get("roc_auc")}
    except Exception as exc:  # noqa: BLE001
        out["winner_2class_trial"] = f"unavailable: {exc}"
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train a 3-class REST/LEFT/RIGHT model.")
    p.add_argument("--arch", choices=model3.ARCHITECTURES, default="eegsym")
    p.add_argument("--method", choices=("euclidean_alignment", "running_exponential"),
                   default="euclidean_alignment")
    args = p.parse_args()
    run(args.arch, args.method)


if __name__ == "__main__":
    main()
