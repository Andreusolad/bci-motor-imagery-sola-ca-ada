"""EEGSym normalization-comparison study.

Fully self-contained sibling of :mod:`first_ml.src.study` (the EEGNet
study). Nothing in this package writes to or imports anything that mutates
``first_ml/src/study/`` or any other file used to build the EEGNet /
CNN 1D models -- it only *reads* (imports) the generic, architecture-agnostic
pieces of that study (data loading, cropping, normalization, leakage guard,
trial aggregation, evaluation, artifact writing) and the base project
config, so the two studies stay methodologically aligned without ever
touching each other's files.
"""
