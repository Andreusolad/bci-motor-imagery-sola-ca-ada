r"""Orchestrator: train + evaluate the EEGNet-regression decoder on BCI-IV-1.

    python run.py --mode pretrain   # Stieger pretrain + per-subject fine-tune (best)
    python run.py --mode scratch    # per-subject from scratch (calib only)

Reports per-subject and mean (real subjects a,b,f,g) official MSE, next to the
reference numbers: constant-zero 0.509, champion 0.382, and prior runs of this
decoder (scratch 0.4332, pretrain+fine-tune 0.4131).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from preprocessing import REAL_SUBJECTS
from evaluate import evaluate, zero_output_baseline
import train as T

OUT = Path(__file__).with_name("results")
REFS = {"zero_baseline": 0.509, "champion": 0.382, "csp_baseline": 0.375,
        "scratch": 0.4332, "pretrain_ft": 0.4131}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["scratch", "pretrain"], default="pretrain")
    mode = ap.parse_args().mode
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    from tensorflow import keras  # noqa: F401
    # sanity gate: constant-zero should reproduce ~0.509 on the real subjects
    gate = float(np.mean([zero_output_baseline(s) for s in REAL_SUBJECTS]))
    print(f"[gate] constant-zero MSE (real) = {gate:.4f}  (official 0.509)")

    if mode == "pretrain":
        print("[run] pretraining on Stieger ...")
        T.pretrain_stieger()

    results = []
    for s in REAL_SUBJECTS:
        model, epochs = (T.finetune(s) if mode == "pretrain" else T.train_from_scratch(s))
        m = evaluate(model, s)
        keras.backend.clear_session()
        m.update({"subject": s, "epochs": epochs})
        results.append(m)
        print(f"ds1{s}: ep={epochs} MSE raw={m['mse_raw']:.4f} smooth={m['mse_smooth']:.4f}")

    mean_raw = float(np.mean([r["mse_raw"] for r in results]))
    mean_smooth = float(np.mean([r["mse_smooth"] for r in results]))
    summary = {"mode": mode, "gate_zero_real": round(gate, 4),
               "real_mean_mse_raw": round(mean_raw, 4),
               "real_mean_mse_smooth": round(mean_smooth, 4),
               "references": REFS, "per_subject": results,
               "runtime_s": round(time.perf_counter() - t0, 1)}
    (OUT / f"summary_{mode}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"=== {mode}: real-mean MSE smooth = {mean_smooth:.4f} "
          f"(scratch 0.4332, pretrain+ft 0.4131, champion 0.382) ===")


if __name__ == "__main__":
    main()
