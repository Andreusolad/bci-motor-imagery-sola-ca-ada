"""Cross-subject normalization comparison study (EEGNet only).

This sub-package evolves the existing ``first_ml`` project with a second,
independent pipeline that compares two normalization/alignment strategies --
Running Exponential Standardization vs. Euclidean Alignment -- while keeping
everything else (architecture, hyper-parameters, optimizer, callbacks, seed,
split, evaluation procedure) identical between the two experiments.

It reuses ``first_ml.src.dataset`` (raw ``.mat`` parsing + trial extraction),
``first_ml.src.preprocessing`` (band-pass/downsample/CAR) and
``first_ml.src.split`` (subject-level split + anti-leakage checks) as-is. It
never writes an NPZ cache: every experiment run re-parses the raw ``.mat``
files directly.
"""
