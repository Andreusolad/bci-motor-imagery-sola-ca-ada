"""End-to-end inference pipeline: sliding window + gate + evidence accumulator.

    continuous stream (8 ch at 250 Hz)
        -> SlidingWindowStream        (sliding window + causal preprocessing)
        -> ControlStateGate           (idle class + confidence + OOD)
        -> EvidenceAccumulator        (dwell / EMA / N-of-M / Bayes + refractory period)
        -> discrete command           (or None: no action)

Provides:
  - RealtimeBCI.process_chunk(raw) -> list of commands emitted in that chunk.
  - simulate_stream(...) to evaluate gate + accumulator on a sequence of labelled windows
    (counting correct commands and false positives) without hardware.
  - make_pseudo_stream(...) to arrange labelled windows in alternating rest / MI blocks.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .streaming import SlidingWindowStream, CalibrationProfile
from .gate import ControlStateGate, GateDecision
from .accumulator import EvidenceAccumulator


# ============================================================
# Emitted command
# ============================================================
@dataclass
class Command:
    label: int                       # 0=LH, 1=RH
    name: str                        # 'left_hand' / 'right_hand'
    sample: int                      # stream samples seen after the chunk that emitted it


# ============================================================
# Real-time system
# ============================================================
class RealtimeBCI:
    def __init__(self, profile: CalibrationProfile, gate: ControlStateGate,
                 accumulator: EvidenceAccumulator | None = None):
        self.profile = profile
        self.stream = SlidingWindowStream(profile)
        self.gate = gate
        self.acc = accumulator or EvidenceAccumulator()
        self.history: list[GateDecision] = []

    def process_chunk(self, chunk_raw: np.ndarray) -> list[Command]:
        """chunk_raw: (8, n) new samples at 250 Hz. Returns the commands emitted."""
        cmds = []
        for win in self.stream.push(chunk_raw):
            dec = self.gate.decide(win)
            self.history.append(dec)
            out = self.acc.update(dec)
            if out is not None:
                cmds.append(Command(out, C.LABELS[out], self.stream.n_seen))
        return cmds

    def reset(self):
        self.stream.reset(); self.acc.reset(); self.history.clear()


# ============================================================
# Simulator (window-level evaluation without hardware)
# ============================================================
@dataclass
class SimResult:
    n_windows: int
    commands: list = field(default_factory=list)      # (window index, emitted class, truth)
    true_labels: np.ndarray = None

    def resumen(self) -> dict:
        """Command counts (correct, wrong, false positives on IDLE) and precision."""
        if not self.commands:
            return dict(n_commands=0, correct=0, wrong=0, false_pos_idle=0,
                        precision=float('nan'))
        correct = sum(1 for _, c, t in self.commands if t != C.IDLE_ID and c == t)
        wrong_ctrl = sum(1 for _, c, t in self.commands if t != C.IDLE_ID and c != t)
        fp_idle = sum(1 for _, c, t in self.commands if t == C.IDLE_ID)
        n = len(self.commands)
        return dict(n_commands=n, correct=correct, wrong=wrong_ctrl,
                    false_pos_idle=fp_idle, precision=correct / n)


def simulate_stream(gate: ControlStateGate, accumulator: EvidenceAccumulator,
                    X_windows: np.ndarray, y_windows: np.ndarray) -> SimResult:
    """Run gate + accumulator over a sequence of already preprocessed windows.

    Each window is one tick of the stream (as if the user alternated MI and rest blocks).
    Records which commands the accumulator emits and whether they are correct.

    X_windows: (T, 8, w), preprocessed (EA + z-score applied). y_windows: (T,) true labels.
    """
    accumulator.reset()
    decs = gate.decide_batch(X_windows)
    commands = []
    for i, dec in enumerate(decs):
        out = accumulator.update(dec)
        if out is not None:
            commands.append((i, out, int(y_windows[i])))
    return SimResult(n_windows=len(X_windows), commands=commands, true_labels=y_windows)


def make_pseudo_stream(X: np.ndarray, y: np.ndarray, block: int = 12,
                       seed: int = C.SEED) -> tuple[np.ndarray, np.ndarray]:
    """Arrange windows in alternating blocks of `block` windows (idle, LH, idle, RH, ...)
    to mimic use in which the user rests, imagines for a while, rests again, and so on.
    """
    rng = np.random.RandomState(seed)
    idle = np.where(y == C.IDLE_ID)[0]
    lh = np.where(y == 0)[0]
    rh = np.where(y == 1)[0]
    rng.shuffle(idle); rng.shuffle(lh); rng.shuffle(rh)
    pools = {C.IDLE_ID: list(idle), 0: list(lh), 1: list(rh)}
    order = []
    seq = [C.IDLE_ID, 0, C.IDLE_ID, 1]                 # rest, LH, rest, RH, ...
    k = 0
    while all(pools[c] for c in seq):
        cls = seq[k % len(seq)]
        for _ in range(block):
            if pools[cls]:
                order.append(pools[cls].pop())
        k += 1
        if k > 400:
            break
    order = np.array(order)
    return X[order], y[order]
