# Detection Is Not Discrimination

Code for the paper *Detection Is Not Discrimination: Deep Learning and the Ceiling of
Asynchronous Motor-Imagery Brain–Computer Interfaces on Eight EEG Channels*,
by Aniol Cañada and Andreu Solà.

The study has two routes that share no code, data split or preprocessing, so that the
places where they agree can be read as replications. This repository holds the code of
both: **route A** (Andreu Solà) in `route_a/` and **route B** (Aniol Cañada) in `route_b/`.

Only the scripts behind the results reported in the paper are included; exploratory
analyses were left out.

## Contents

| Folder | What it computes | Paper |
|---|---|---|
| `route_a/two_class/` | Dual-branch raw + wavelet CNN on Stieger2021: 36-cell input grid, held-out test, EEGSym port, zero-shot transfer to BCI Competition IV 2a | Sec. 5.1, 5.4.1 |
| `route_a/async_stieger/` | Asynchronous decoding on Stieger2021 with an explicit rest class: rest-class formulations, detector occlusion, visual-cue controls, passive rest from Physionet, cost per intent, user aptitude | Sec. 5.2, 5.3, 5.5.1 |
| `route_a/bci_iv_1/` | BCI Competition IV Data Set 1 (asynchronous, official MSE): silence baseline, the D12 and D13 systems, causality test by perturbation | Sec. 5.4.2, 5.4.3 |
| `route_a/split_80_20_subjects.json` | Subject split used throughout (50 training, 12 validation subjects) | |
| `route_b/two_class/` | EEGNet and EEGSym on Stieger2021 under three normalizations, legacy vs corrected window, subject-level bootstrap CIs, per-subject calibration, temporal and band importance | Sec. 5.1 |
| `route_b/continuous/` | Three-class REST/LEFT/RIGHT decoder and continuous five-minute streams: hierarchical gate, threshold sweep, calibration, temporal HMM filter, continuous training, per-command evaluation | Sec. 5.2, 5.3 |
| `route_b/detection/` | Detection diagnostics: delta-band confound (retrain behind a 4 Hz high-pass), band/channel importance, dedicated binary REST-vs-MI gate | Sec. 5.2 |
| `route_b/transfer/` | Zero-shot transfer of the Stieger decoder to BCI-IV-2a (EEGSym and EEGNet) and Physionet, with a contralateral-ERD label check | Sec. 5.4.1 |
| `route_b/bci_iv_1/` | BCI Competition IV Data Set 1, official MSE: zero-shot three-class, soft vs hard output, EMA + hedging, per-subject calibration, and a per-subject EEGNet regressor (`regression/`) | Sec. 5.4.2 |
| `route_b/channels/` | Electrode-count sweep (8/16/32 channels, EEGNet + EA) on Stieger and on BCI-IV-1 | Sec. 5.5.2 |
| `route_b/dataset_split.json` | Subject split used throughout route B (36 training, 13 validation, 13 test subjects) | |

## Setup

```
pip install -r requirements.txt
```

Tested with Python 3.13, numpy 2.2, scipy 1.16, scikit-learn 1.7, torch 2.6 (CUDA 12.4),
mne 1.12 and pyriemann 0.11. The networks were trained on an 8 GB RTX 4060; the
BCI-IV-1 system runs on CPU.

## Data

All four datasets are public. The scripts read them from the folder given by the
environment variable `BCI_DATA` (default: `data/` at the repository root):

```
$BCI_DATA/
  stieger2021/                S1_Session_5.mat ... S62_Session_11.mat
  bci_iv_2a/                  A01T.npz ... A09E.npz
  bci_iv_2a/true_labels/      A01T.mat ... A09E.mat
  physionet/eegmmidb-1.0.0/   S001/ ... S109/
  bci_iv_1/                   BCICIV_calib_ds1*.mat, BCICIV_eval_ds1*.mat, true labels
```

- **Stieger2021**: Stieger, Engel and He, *Scientific Data* 8:98 (2021),
  [figshare](https://doi.org/10.6084/m9.figshare.13123148). The two-class cache uses
  sessions 5-11 and the asynchronous cache sessions 5-6.
- **BCI Competition IV 2a**: the `.npz` version at
  [bregydoc/bcidatasetIV2a](https://github.com/bregydoc/bcidatasetIV2a) and the true labels
  (`true_labels.zip`) from the [competition results page](https://www.bbci.de/competition/iv/results/).
- **Physionet EEG Motor Movement/Imagery**:
  [physionet.org/content/eegmmidb/1.0.0](https://physionet.org/content/eegmmidb/1.0.0/).
- **BCI Competition IV Data Set 1**: `python route_a/bci_iv_1/download.py` downloads the
  100 Hz recordings and the true labels from bbci.de.

**Route B** reads the same four datasets from the same `$BCI_DATA` root, with two format
differences: it uses the **`.mat`** version of BCI Competition IV 2a
(`bci_iv_2a/A01T.mat ... A09T.mat`) instead of the `.npz`, and the **1000 Hz** BCI
Competition IV Data Set 1 (`bci_iv_1/BCICIV_1calib_1000Hz_mat/`,
`bci_iv_1/BCICIV_1eval_1000Hz_mat/`, `bci_iv_1/true_labels_official/mat/`) from the
[competition page](https://www.bbci.de/competition/iv/). Stieger2021
(`stieger2021/S{n}_Session_{m}.mat`) and Physionet use the same layout as above.

Caches and results are written to `cache/` and `outputs/` inside each folder; route B
writes trained models and metrics under `route_b/experiments/`.

## Reproducing the results

Run the scripts from their own folder. Times are for the hardware above.

### Two-class decoding (`route_a/two_class`)

```
python build_cache.py                          # Stieger trials, 8 channels, 250 Hz, [+2, +5] s  (~25 min)
python run_grid.py                             # 36-cell grid -> outputs/results.csv  (~3 h, GPU)
python confirm_test.py                         # retrain on 40 subjects, test on 10 unseen ones
python bootstrap_test.py                       # subject bootstrap of the test accuracy
python transfer_2a.py --build-cache
python transfer_2a.py --protocol P2            # leave-one-subject-out within BCI-IV-2a
python transfer_2a.py --protocol P3            # Stieger model applied to BCI-IV-2a, no adaptation
python eegsym.py                               # checks the EEGSym port (36 output features, N=24)
python compare_architectures.py --seed 42      # EEGSym vs dual-branch, transfer to 2a (also --seed 43)
python analyze_architectures.py
```

| Result | Value | Where |
|---|---|---|
| Best grid cell (W = 0.5 s, 16 frequencies, downsampling on), validation | 0.6991, κ = 0.398 | `outputs/results.csv` |
| Held-out test, 10 subjects never used for selection | 0.628, 95% CI [0.579, 0.667] | `confirm_test.py`, `bootstrap_test.py` |
| BCI-IV-2a, leave-one-subject-out | 0.694 | `outputs/results_2a_P2_W0.5.csv` |
| BCI-IV-2a, Stieger model applied without retraining (per-subject z-score only) | 0.656 | `outputs/results_2a_P3_W0.5.csv` |

`transfer_2a.py --protocol P3` uses `checkpoints/winner_W0.5_F16_dsON.pt`, the best grid
cell trained on the 50 training subjects. `train_winner.py` rebuilds it (~3 min, GPU).

### Asynchronous decoding on Stieger (`route_a/async_stieger`)

`rt_system/` is the decoding package (EEG-Conformer, rejection gate, evidence
accumulator); the scripts in the folder use it as a library.

```
python build_dataset.py --W 2.0                # 2 s windows: LH, RH, cued rest (REST1), pre-cue rest (REST2)
python run_experiments.py                      # formulations A/B/C/D x bands x seeds -> outputs/main  (~1.5 h, GPU)
python analyze.py --dir outputs/main           # operating curves, pAUC, confusion, cost weights
python occlusion.py                            # band and channel importance of the trained gates
python occlusion_by_rest.py                    # delta importance, pre-cue vs post-cue rest
python cue_free_control.py --ckpt outputs/main/ckpt_B_bb_W2_r1_s42.pt
python physionet_rest_cache.py                 # passive rest from Physionet
python physionet_rest_eval.py                  # Stieger gates scored against passive rest
python prep_streams.py                         # continuous streams of the 12 validation subjects
python infer_streams.py --todos                # dense sliding-window inference over the streams
python cost_per_intent.py                      # per-window vs per-intent cost
python aptitude_folds.py                       # five subject folds for the aptitude study
python aptitude_train.py                       # 30 trainings (~3 h, GPU)
python aptitude_metrics.py
python aptitude_correlations.py
```

| Result | Value | Where |
|---|---|---|
| Explicit rest class vs binary decoder, pAUC | +0.110 to +0.242 | `outputs/main/analysis.json` |
| Delta occlusion of the broadband gates, pAUC | -0.11 to -0.14 | `outputs/occlusion.json` |
| Share of the rest-detection gain that precedes imagery onset | about 60% | `outputs/cue_free_control_*.json` |
| Gates scored against passive rest (Physionet), AUC | 0.56-0.58 | `outputs/physionet_rest_eval.json` |
| False commands per minute of rest, per window vs per intent (threshold 0.5) | 367 vs 18 | `outputs/cost_per_intent.json` |
| Best per-intent direction accuracy of the three-class broadband gate, over thresholds that fire on at least 2% of intents (firing pays only above 0.90) | 0.658 | `outputs/cost_per_intent.json` |
| Aptitude vs direction / vs detection, Spearman, 62 subjects | +0.398 / -0.019 | `outputs/aptitude_correlations.json` |
| Subjects detectable but not decodable | 12 of 62 | `outputs/aptitude_correlations.json` |

### BCI Competition IV Data Set 1 (`route_a/bci_iv_1`)

The system follows the winning entry's recipe (Chebyshev II filter bank, FBCSP with
pairs against rest, rest sub-states, a three-class posterior head, linear
post-processing). Its configuration was selected on the calibration recordings only and
is stored in `config/dev_best.json`.

```
python download.py
python check_labels.py                         # label checks and the silence baseline  (seconds)
python check_regression.py                     # the development-bench result must reproduce exactly  (~3 min)
python run_shot.py --shot D12                  # offline system  (~9 min)
python run_shot.py --shot D13                  # strictly causal system  (~40 min)
python causality_test.py                       # perturbation test  (~1 h)
```

| Result | Value | Where |
|---|---|---|
| Silence (constant 0), mean of a, b, f, g | 0.5094 | `outputs/check_labels.json` |
| D12, offline | 0.3754 | `outputs/shot_D12.json` |
| D13, strictly causal | 0.3930 | `outputs/shot_D13.json` |
| Causality test: causal system passes, offline fails, each piece caught | 7/7 as expected | `outputs/causality_test.json` |

The 2012 winning entry scored 0.382. This pipeline has no randomness left unseeded:
with the library versions above, `run_shot.py` reproduces the per-subject values of the
paper exactly.

## Reproducing the results (Route B)

Route B uses TensorFlow/Keras (CPU is enough; no GPU required) and is organized as a
shared core (`route_b/src`, `route_b/lib`) plus one folder per result group. Run each
script from its own folder with `BCI_DATA` set; outputs are written under
`route_b/experiments/`. Several evaluations load the two-class and three-class models
trained first, so run the folders in the order below. Seeds are fixed (42); training the
networks is not bit-reproducible across machines, so small deviations are expected.

The sixth corrected-window two-class model (EEGSym + running-exponential) is assumed
already trained; `run_corrected_window_study.py` trains the other five.

### Two-class decoding (`route_b/two_class`)

```
python run_study.py                              # EEGNet x {z-score, running-exp, EA}, legacy window
python run_eegsym_study.py                       # EEGSym x {z-score, running-exp, EA}, legacy window
python run_corrected_window_study.py             # both architectures, corrected [+2, +2+len] window
python phase1_subject_ci.py                      # subject-bootstrap CIs and paired contrasts
python analyze_temporal_decay_corrected.py       # softmax mass across the trial, corrected window
python analyze_frequency_importance_corrected.py # band-permutation importance
python finetune_subject_corrected.py             # per-subject calibration, 5 subjects
python main.py --stage all                       # plain 1-D CNN baseline
```

| Result | Value | Where |
|---|---|---|
| 1-D CNN baseline, per crop | 0.623, AUC 0.671 | `experiments/legacy_window/approach_a_baseline/metrics.json` |
| Six corrected-window models, subject-mean accuracy | 0.693 to 0.752 | `experiments/subject_ci/summary.json` |
| EEGSym - EEGNet (paired, 13 subjects) | +0.025, 95% CI [+0.012, +0.039], 11/2 | `experiments/subject_ci/summary.json` |
| Legacy vs corrected window | normalization ranking flips; accuracy rises | `experiments/*/normalization/*/comparison/` |
| Per-subject calibration on Stieger (5 subjects) | +4.0 points, 4/5 improve | `experiments/corrected_window/subject_finetuning/summary.json` |

### Three-class and continuous decoding (`route_b/continuous`)

Needs the corrected-window EEGSym + EA two-class winner (above).

```
python train_3class.py --arch eegsym --method euclidean_alignment    # 3-class REST/LEFT/RIGHT
python continuous_eval.py --arch eegsym --method euclidean_alignment  # 5-min streams
python hierarchical_eval.py                      # REST gate + separate L/R head
python threshold_sweep.py
python calibration_eval.py                       # per-subject calibration
python combined_eval.py                          # calibration + gate + smoothing ladder
python temporal_filter_eval.py                   # majority vote and causal HMM
python train_continuous.py                       # detector trained on continuous windows
python continuous_plus_filter.py                 # continuous detector + HMM
python command_eval.py                           # per-command (drone) evaluation
python calibration_command_eval.py               # calibration -> directional error
```

| Result | Value | Where |
|---|---|---|
| Flat 3-class recall: REST / RIGHT / LEFT (LEFT collapse) | 0.865 / 0.600 / 0.168 | `experiments/rest_3class/summary.json` |
| LEFT recall, flat vs hierarchical (continuous) | 0.171 -> 0.468 | `experiments/continuous_eval_hierarchical/summary.json` |
| MI detection on continuous streams, AUC | ~0.77 (0.865 on clean crops) | `experiments/continuous_eval/summary.json` |
| Per-command directional error when acting | 34-38%, near-invariant across models | `experiments/command_eval/summary.json` |
| Per-subject calibration -> directional error | 38% -> 28%, 9/10 improve | `experiments/calibration_command/summary.json` |
| Continuous + HMM, best MI-detection F1 | 0.262 | `experiments/continuous_trained/with_temporal_filter/summary.json` |

### Detection diagnostics (`route_b/detection`)

```
python delta_confound_control.py                 # retrain behind a 4 Hz high-pass (double dissociation)
python feature_importance.py                     # channel and band importance of the 3-class model
python binary_rest_mi.py                         # dedicated binary REST-vs-MI gate
python binary_on_continuous.py
python train_binary_continuous.py
```

| Result | Value | Where |
|---|---|---|
| Detection AUC without delta (retrain) | 0.887 -> 0.713 (-0.175) | `experiments/delta_confound/summary.json` |
| Direction AUC without delta (control) | 0.797 -> 0.795 (-0.001) | `experiments/delta_confound/summary.json` |
| Dedicated binary gate vs 3-class gate, AUC | 0.759 vs 0.732 | `experiments/binary_on_continuous/summary.json` |

### Transfer (`route_b/transfer`)

Zero-shot from the Stieger EEGSym + EA winner; target labels are checked by contralateral
ERD before scoring.

```
python bciciv2a_eval.py                          # EEGSym + EA -> BCI-IV-2a
python bciciv2a_eegnet_eval.py                   # EEGNet + EA -> BCI-IV-2a
python physionet_eval.py                         # 106 subjects, bimodal distribution
```

| Result | Value | Where |
|---|---|---|
| BCI-IV-2a, EEGSym + EA, cold | 0.738, 95% CI [0.680, 0.806], AUC 0.817 | `experiments/bciciv2a_eval/summary.json` |
| BCI-IV-2a, EEGNet + EA, cold | 0.736, 95% CI [0.667, 0.813] | `experiments/bciciv2a_eegnet_eval/summary.json` |
| Physionet, EEGSym + EA, 106 subjects | 0.637, 95% CI [0.611, 0.665], AUC 0.700 | `experiments/physionet_eval/summary.json` |

### BCI Competition IV Data Set 1 (`route_b/bci_iv_1`)

```
python bciciv1_eval.py                           # 3-class zero-shot, asynchronous metrics
python bciciv1_mse_eval.py                       # official MSE: soft and hard output
python bciciv1_async_postproc.py                 # EMA smoothing + hedging (leave-one-subject-out)
python bciciv1_calibration.py                    # per-subject calibration on calib
python bciciv1_regressor.py                      # per-subject regressor trained on the dataset
python bciciv1_regressor_pretrained.py           # Stieger-pretrained regressor, fine-tuned per subject
python regression/run.py --mode pretrain         # self-contained modular regressor
python regression/analyze_bciciv1.py             # per-subject CIs and detection/direction split
```

| Result | Value | Where |
|---|---|---|
| Silence baseline (real subjects), MSE | 0.5098 pooled (0.5094 subject mean) | `experiments/bciciv1_mse/summary.json` |
| Soft vs hard output, MSE | 0.4795 vs 0.7922 | `experiments/bciciv1_mse/summary.json` |
| + EMA smoothing and hedging | 0.4726 | `experiments/bciciv1_postproc/summary.json` |
| Per-subject calibration | 0.4830 (worse); balanced acc 0.403 -> 0.423 | `experiments/bciciv1_calibration/summary.json` |
| Regressor from scratch / Stieger pretrain + fine-tune | 0.4332 / 0.4131 | `experiments/bciciv1_regressor_pretrained/summary.json` |
| Best regressor: detection vs direction | detection AUC 0.580, sign accuracy 0.762 | `bci_iv_1/regression/results/` |

### Electrode-count sweep (`route_b/channels`)

```
python train_channels.py --set 8
python train_channels.py --set 16
python train_channels.py --set 32
python bciciv1_mse_channels.py --set 8
python bciciv1_mse_channels.py --set 16
```

| Result | Value | Where |
|---|---|---|
| Stieger accuracy at 8 / 16 / 32 channels | 0.7068 / 0.7260 / 0.7458 | `experiments/eegnet_ea_{8,16,32}ch/summary.json` |
| 8 -> 32 channels, paired | +0.034, 95% CI [+0.009, +0.057], 11/2 | `experiments/eegnet_ea_32ch/summary.json` |
| Channel models on BCI-IV-1 (binary, no idle), MSE | 0.527 (8) / 0.548 (16), both above 0.509 | `experiments/eegnet_ea_{8,16}ch/bciciv1_mse.json` |

## Notes

- Training the networks on a GPU is not bit-reproducible across machines. On the machine
  used for the paper, `confirm_test.py` and `train_winner.py` reproduce the reported
  numbers exactly.
- Variable names and the keys of the output files are in Spanish; comments and
  documentation are in English.

## Citation

```
@misc{canada_sola_2026,
  title  = {Detection Is Not Discrimination: Deep Learning and the Ceiling of Asynchronous
            Motor-Imagery Brain--Computer Interfaces on Eight EEG Channels},
  author = {Ca{\~n}ada, Aniol and Sol{\`a}, Andreu},
  year   = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).

## Contact

Aniol Cañada (aniolcanada@gmail.com), Andreu Solà (andreusolad@gmail.com).

