"""rt_system: asynchronous (self-paced) motor-imagery decoding package.

Turns a cue-based left/right-hand classifier into a system that runs on continuous
8-channel EEG (OpenBCI Cyton montage, 250 Hz), where most of the time there is no
intentional control.

Modules:
  config          constants shared by all modules (paths, montage, windows, thresholds)
  preprocessing   CAR, channel selection, downsampling, baseline, band-pass, EA, z-score
  models          EEG-Conformer and EEGNet (3 output classes by default)
  dataset         LH / RH / IDLE window cache from Stieger et al. (2021)
  calibration     temperature scaling and confidence threshold
  ood             out-of-distribution detectors (Riemannian and Mahalanobis)
  gate            control / no-control gate (idle class + confidence + OOD)
  streaming       sliding window with causal preprocessing over a continuous stream
  accumulator     temporal evidence accumulation (dwell, EMA, N-of-M, Bayesian filter)
  realtime        end-to-end pipeline and a window-level simulator
  train           training utilities (library functions, not a script)
"""
from . import config
from .config import CLASES, LABELS, N_CLASES, IDLE_ID, device, log

# Package-level shortcuts
from .dataset import (construir_cache_3clase, load_cache, split_cross_subject,
                      preprocesar_sesion)
from .models import build_model, EEGConformer, EEGNet, load_pretrained_backbone
from .calibration import TemperatureScaler, get_logits, select_conf_threshold
from .ood import RiemannOOD, MahalanobisOOD, solo_control
from .gate import ControlStateGate, GateDecision, evaluar_gate
from .streaming import SlidingWindowStream, CalibrationProfile, build_profile_from_calib
from .accumulator import EvidenceAccumulator
from .realtime import RealtimeBCI, simulate_stream, make_pseudo_stream
from .train import train_loop, make_val_split, class_weights, predict_batched

__all__ = [
    'config', 'CLASES', 'LABELS', 'N_CLASES', 'IDLE_ID', 'device', 'log',
    'construir_cache_3clase', 'load_cache', 'split_cross_subject', 'preprocesar_sesion',
    'build_model', 'EEGConformer', 'EEGNet', 'load_pretrained_backbone',
    'TemperatureScaler', 'get_logits', 'select_conf_threshold',
    'RiemannOOD', 'MahalanobisOOD', 'solo_control',
    'ControlStateGate', 'GateDecision', 'evaluar_gate',
    'SlidingWindowStream', 'CalibrationProfile', 'build_profile_from_calib',
    'EvidenceAccumulator', 'RealtimeBCI', 'simulate_stream', 'make_pseudo_stream',
    'train_loop', 'make_val_split', 'class_weights', 'predict_batched',
]
