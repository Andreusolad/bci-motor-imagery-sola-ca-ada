"""Approach A: raw-EEG end-to-end 1D-CNN with cropped training.

Package modules:
    config         - all constants, paths and hyper-parameters
    utils          - logging, seeding, GPU probing, artefact I/O
    preprocessing  - band-pass, downsample, CAR, z-score
    dataset        - .mat discovery/parsing and the per-session cache
    split          - subject-independent split + anti-leakage checks
    crops          - dynamic cropped-training windows and tf.data pipelines
    model          - the 1D-CNN
    train          - training loop and callbacks
    evaluate       - metrics and figures
"""

__all__ = [
    "config",
    "utils",
    "preprocessing",
    "dataset",
    "split",
    "crops",
    "model",
    "train",
    "evaluate",
]
