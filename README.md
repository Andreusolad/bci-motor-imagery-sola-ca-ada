# Detection Is Not Discrimination

Code for the paper *Detection Is Not Discrimination: Deep Learning and the Ceiling of
Asynchronous Motor-Imagery Brain–Computer Interfaces on Eight EEG Channels*,
by Aniol Cañada and Andreu Solà.

The study has two routes that share no code, data split or preprocessing, so that the
places where they agree can be read as replications. This repository holds the code of
**route A** (Andreu Solà) in `route_a/`. Route B (Aniol Cañada) will be added in `route_b/`.

Only the scripts behind the results reported in the paper are included; exploratory
analyses were left out.

## Contents

| Folder | What it computes | Paper |
|---|---|---|
| `route_a/two_class/` | Dual-branch raw + wavelet CNN on Stieger2021: 36-cell input grid, held-out test, EEGSym port, zero-shot transfer to BCI Competition IV 2a | Sec. 5.1, 5.4.1 |
| `route_a/async_stieger/` | Asynchronous decoding on Stieger2021 with an explicit rest class: rest-class formulations, detector occlusion, visual-cue controls, passive rest from Physionet, cost per intent, user aptitude | Sec. 5.2, 5.3, 5.5.1 |
| `route_a/bci_iv_1/` | BCI Competition IV Data Set 1 (asynchronous, official MSE): silence baseline, the D12 and D13 systems, causality test by perturbation | Sec. 5.4.2, 5.4.3 |
| `route_a/split_80_20_subjects.json` | Subject split used throughout (50 training, 12 validation subjects) | |

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

Caches and results are written to `cache/` and `outputs/` inside each folder.

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
