# AF Detection from 12-Lead ECG Across Hospitals

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24052057.svg)](https://doi.org/10.5281/zenodo.24052057)

Deep learning detects atrial fibrillation (AF) on the ECG very accurately, but it is not clear *what* the models rely on: the atrial signs of AF (missing P-waves, fibrillatory f-waves) or the simpler cue of irregular beat timing (RR intervals).

This repository trains a 1D ResNet (and, as a check, an InceptionTime network) on 12-lead ECGs from **PTB-XL**, tests them without retraining on **Chapman-Shaoxing** and **Georgia** from two other hospitals, and compares them with a model that only sees RR intervals. Six attribution methods measure where the evidence falls (atrial vs ventricular parts of each beat), and controlled occlusion and counterfactual edits test what the model actually needs. AF with a regular ventricular response and paced rhythms, where the RR cue is weak, are analysed as subgroups.

**Keywords:** atrial fibrillation · ECG · electrocardiogram · 12-lead ECG · deep learning · 1D CNN · ResNet · explainable AI · XAI · Grad-CAM · integrated gradients · occlusion · counterfactual explanations · external validation · domain shift · generalisation · PTB-XL · Chapman-Shaoxing · Georgia · InceptionTime · DeepLIFT · SHAP · faithfulness · PyTorch · medical AI · cardiology

---

## Hardware and software

Developed and run on:

| Component | Specification |
|---|---|
| Computer | Apple MacBook Pro, M5 chip |
| Memory | 16 GB unified memory |
| Storage | 512 GB SSD (about 10 GB free is enough for this project) |
| GPU | Apple GPU through PyTorch MPS (used automatically) |
| OS | macOS |
| Python | 3.11 or newer (developed on 3.14) |

Also runs on Linux or Windows: an NVIDIA GPU is used automatically through CUDA, otherwise the CPU (training is then much slower).

**Measured run times on the machine above:**

| Step | Time |
|---|---|
| Download PTB-XL (~1.7 GB zip) | 10–40 min, depends on PhysioNet |
| Download Chapman-Shaoxing (parallel script) | 30–60 min |
| Label tables | seconds |
| Preprocessing | ~10 s per dataset |
| R-peaks and RR features | ~5 s per dataset |
| RR baseline | seconds |
| ResNet training | ~2 min per epoch, ~25 min per seed (early stopping) |

Large arrays are memory-mapped from disk, so 16 GB RAM is enough. Processed signals take about 2.6 GB (PTB-XL), 1.2 GB (Chapman) and about 1.2 GB (Georgia) as float16.

---

## Setup

```bash
git clone https://github.com/sshemanth/af-ecg-xai.git
cd af-ecg-xai
bash setup.sh
```

`setup.sh` creates a virtual environment in `.venv`, installs `requirements.txt` and runs `scripts/check_setup.py` (packages, GPU, data). Downloads need `wget` (macOS: `brew install wget`).

In every new terminal:

```bash
cd af-ecg-xai
source .venv/bin/activate
```

If the prompt shows Conda's `(base)`, run `conda deactivate` first. In zsh, paste commands without trailing `#` comments, or enable them once with `echo 'setopt interactivecomments' >> ~/.zshrc`.

---

## Data

| | PTB-XL | Chapman-Shaoxing | Georgia |
|---|---|---|---|
| Role | Training, validation, internal test | External test only | External test only |
| Source | PhysioNet, v1.0.3 | PhysioNet/CinC Challenge 2021 copy | PhysioNet/CinC Challenge 2021 copy |
| Format | 12-lead, 500 Hz, 10 s | 12-lead, 500 Hz, 10 s | 12-lead, 500 Hz, mostly 10 s |
| Labels | SCP-ECG statements | SNOMED-CT codes | SNOMED-CT codes |
| ECGs kept after preprocessing | 21,742 (7.0% AF) | 9,794 (18.2% AF) | see `results/02_preprocess_report_georgia.md` |

External sets are never used for training or tuning. The Challenge copies carry no patient IDs: Chapman-Shaoxing has one ECG per patient, and each Georgia record is treated as its own patient in the bootstrap.

```bash
bash scripts/download_data.sh                      # PTB-XL, then Chapman one file at a time
bash scripts/download_challenge.sh chapman         # faster Chapman download (6 folders in parallel)
bash scripts/download_challenge.sh georgia         # Georgia, same way
```

All scripts can be re-run; they resume and skip finished files. Data goes to `data/raw/` and is not tracked by git.

**Label definitions**

| Variable | PTB-XL | Chapman, Georgia | Meaning |
|---|---|---|---|
| `af` | `AFIB` | `164889003` | Atrial fibrillation |
| `aflutter` | `AFLT` | `164890007` | Atrial flutter |
| `paced` | `PACE` | `10370003` | Paced rhythm |
| `sinus` | `SR` | `426783006` | Sinus rhythm (reference for the regular-rate rule only) |

Target: 1 = AF (with or without flutter), 0 = neither AF nor flutter. Flutter without AF is excluded (`FLUTTER_POLICY` in `src/config.py`).

---

## Pipeline

Run the scripts in order from the project folder. Each writes a Markdown report to `results/`.

### 1. Label tables

```bash
python scripts/01_count_labels.py
```

Builds one row per ECG with AF / flutter / paced / sinus flags and the train/val/test split. Report: `results/01_label_counts.md`.

### 2. Preprocessing

```bash
python scripts/02_preprocess.py --dataset ptbxl
python scripts/02_preprocess.py --dataset chapman
python scripts/02_preprocess.py --dataset georgia
```

Run without `--dataset` to process every dataset that has a label table. A single dataset writes its own report (`02_preprocess_report_<dataset>.md`), so earlier reports are kept.

Loads every ECG, puts the leads in standard order, excludes bad recordings (missing or flat lead, unreadable or NaN, shorter than 10 s, flutter-only, no diagnosis code), applies a 0.5–100 Hz zero-phase band-pass and saves float16 arrays. Per-lead normalisation statistics come from the PTB-XL training split only (`data/processed/norm_stats.json`). The report includes exclusions by reason, AF per split, an amplitude check and an edge check that compares AF with non-AF. Add `--limit 200` for a quick test, `--workers N` to set CPU processes.

### 3. R-peaks and RR features

```bash
python scripts/03_rpeaks.py --dataset ptbxl
python scripts/03_rpeaks.py --dataset chapman
python scripts/03_rpeaks.py --dataset georgia
```

Detects R-peaks (NeuroKit2, lead II, then V5, then I) and computes heart rate, rr_cv, RMSSD, nRMSSD and pNN50. Flags **AF with low RR variability** (rr_cv at or below the 90th percentile of sinus-rhythm PTB-XL training ECGs) and **near-regular AF** (75th percentile). PTB-XL must run first, since the cut-offs come from its training split.

Visual checks:

```bash
python scripts/plot_examples.py
python scripts/plot_examples.py --group sinus
python scripts/plot_raw_vs_filtered.py --record 17360
```

### 4. RR-only baseline

```bash
python scripts/04_rr_baseline.py
```

Logistic regression on the RR features only. Trained on the training split; threshold chosen on validation. Saved to `results/models/rr_baseline/`.

### 5. Train the networks

```bash
python scripts/05_train.py --limit 2000 --epochs 2
python scripts/05_train.py --seed 1
python scripts/05_train.py --seed all
python scripts/05_train.py --seed 2,3,4,5
python scripts/05_train.py --arch inception --seed all
```

The first command is a quick check (it writes to `resnet_seed1`, so run it before the real training, not after). `--seed all` trains seeds 1–5. `--arch inception` trains the second network: an InceptionTime-style model (parallel convolutions of three widths, about 312, 152 and 72 ms at 125 Hz, residual links every three modules, 0.5M parameters) with the same data, loss, augmentation, schedule and threshold rule. It checks that the findings are not specific to the ResNet. Runs are saved as `results/models/<arch>_seed<N>/`. Training uses the training split only, early stopping on validation AUPRC, a class-weighted loss, and augmentation that keeps atrial morphology (random 9 s crop, amplitude scaling, slow baseline wander). The decision threshold is chosen on validation (Youden's J) and saved with each run. The test split and the external sets are never loaded during training.

### 6. Evaluate on held-out data

```bash
python scripts/06_evaluate.py --dry-run
python scripts/06_evaluate.py
```

`--dry-run` checks readiness without touching test data. The evaluation scores the PTB-XL test split and every external set with the locked models and thresholds of every trained network (each with its own primary seed): AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1 and confusion counts with patient-level bootstrap 95% CIs; calibration (plot, intercept, slope, Brier score, ECE); paired comparisons on the same ECGs (each network vs the RR baseline, InceptionTime vs ResNet); internal vs each external AUROC; and detection rates in the low-RR-variability and near-regular AF subgroups. The primary model is the seed with the best validation AUPRC; all seeds are reported.

Held-out data is meant to be used once per model. Each evaluation is recorded in `results/eval/LOCK_<dataset>.json` with a SHA-256 fingerprint of every model file. Later runs may add new models or datasets and may re-score identical models (that reveals nothing new), but the script refuses to score a model whose file changed after it was tested, unless you add `--force`. It also stops while a network has only some of its seeds trained (override with `--partial`).

Calibration plots show each bin's observed AF rate with a 95% CI and the number of ECGs in the bin, plus a histogram of predicted probabilities, so points resting on a few ECGs are easy to spot. To redraw them from the saved predictions without running any model (and without touching the lock):

```bash
python scripts/plot_calibration.py
python scripts/plot_calibration.py --strategy quantile
```

The second uses bins with equal numbers of ECGs. Bin tables: `results/eval/calibration_bins_<dataset>.csv`.

### 7. Attribution

```bash
python scripts/07_attributions.py --n-per-class 20
python scripts/07_attributions.py
```

The first command is a quick check. The script explains the primary model on held-out ECGs with Integrated Gradients (lead × time) and Grad-CAM (time). Each beat is split, from its R-peaks and its measured T-wave end, into atrial (P/f-waves), QRS and ST-T windows; the report gives the share of evidence for AF in each window and its enrichment (share ÷ time), by group (AF, non-AF, low-RR-variability AF, near-regular AF, paced AF), by lead, for caught vs missed AF, and PTB-XL vs each external set. It also reports agreement between the two methods and stability under 0.01 mV noise. Example plots: `results/xai/examples_<dataset>.png`. The first and last 0.5 s of each ECG are excluded.

### 8. Occlusion and counterfactual tests

```bash
python scripts/08_occlusion.py --n-per-class 20
python scripts/08_occlusion.py
python scripts/08_occlusion.py --seed all
```

The first command is a quick check. On the same held-out ECGs as step 7, the primary model scores each ECG and a set of controlled changes; the outcome is the change in AF probability:

| Change | What it does |
|---|---|
| `atrial_removed` | Atrial windows replaced by a straight line (P/f-waves gone; QRS, ST-T and timing kept) |
| `qrs_removed`, `stt_removed` | The same for QRS or ST-T windows |
| `matched_control` | The same total length removed at random places (may include QRS or ST-T) |
| `rr_regularised` | Beats moved to even spacing; every QRS and ST-T copied exactly, only atrial windows resampled |
| `atrial_swapped` | Atrial windows filled from an opposite-class ECG (sinus P-waves into AF, f-waves into non-AF), timing unchanged |
| `atrial_removed_<lead>` | Atrial windows removed in one lead only |

If removing atrial activity lowers the AF probability more than regularising the rhythm (and more than the matched control), the model depends on atrial morphology; the reverse means it depends on timing. Every changed ECG is checked automatically (finite values, plausible amplitude, no jumps) and failures are counted. Statistics are paired: Wilcoxon signed-rank tests, rank-biserial effect sizes, bootstrap 95% CIs and Benjamini-Hochberg correction across comparisons and across leads. Report: `results/08_occlusion_report.md`; example plot: `results/occlusion/example_<dataset>.png`.

`--seed all` (or e.g. `--seed 1,2`) runs the same changed ECGs through every trained seed, to check the result is not specific to the primary model. Each ECG is changed once and scored by all seeds, so it takes only a little longer than one seed. The primary model's report is written as usual; `results/08_occlusion_seeds_report.md` puts the seeds side by side (mean change per seed, and the paired comparisons with how many seeds reach adjusted p < 0.05). Per-seed tables: `results/occlusion/occlusion_<dataset>_resnet_seed<N>.csv`.

### 9. False alarms by rhythm (external sets)

```bash
python scripts/09_false_alarms.py
python scripts/09_false_alarms.py --model resnet_seed1
```

Chapman-Shaoxing and Georgia headers list every diagnosis per ECG (SNOMED-CT codes), so the false alarms can be split by rhythm: sinus rhythm, sinus brady/tachycardia, sinus arrhythmia, premature atrial or ventricular beats, supraventricular tachycardia (SVT, atrial tachycardia, AVNRT, AVRT), wandering pacemaker, paced. For each rhythm the report gives the false-alarm rate of the ResNet and the RR baseline (95% Wilson CI), a table for every diagnosis code, the beat timing of false alarms vs correct non-AF calls, and SVT vs sinus tachycardia within the same heart-rate bands (Mantel-Haenszel odds ratio), to tell hidden P-waves apart from a fast rate. Only saved predictions and the raw headers are read; no model is run. Code names come from the PhysioNet/CinC Challenge 2021 mapping (`src/snomed_codes.csv`). Use `--dataset georgia` for Georgia and `--arch inception` for the second network. Outputs: `results/09_false_alarms_report*.md`, `results/eval/false_alarms_<dataset>*.csv`.

### 10. Paired tests for the swap and P-wave removal

```bash
python scripts/10_occlusion_tests.py
```

Run after `08_occlusion.py --seed all`. In AF ECGs, flattening the atrial windows changes little, since there were no normal P-waves to begin with. This step tests where the atrial evidence shows: removing P-waves from (or putting f-waves into) non-AF ECGs, and putting sinus P-waves into AF ECGs, each against no change, the matched control and a perfectly regular rhythm. It gives mean differences with bootstrap 95% CIs, rank-biserial effect sizes and Benjamini-Hochberg adjusted p-values for every seed, plus the share of ECGs whose call changes at each seed's locked threshold. These tests were added after the step 8 results and are labelled as such. Only the saved step 8 tables are read; seed tables from an older step 8 run are detected and left out. Report: `results/10_occlusion_tests_report.md`.

### 11. Attribution methods compared

```bash
python scripts/11_attribution_methods.py --n-per-class 20
python scripts/11_attribution_methods.py
python scripts/11_attribution_methods.py --arch inception
```

Run after step 8. Six methods explain the same model on the same ECGs (100 AF and 100 non-AF per dataset by default): Integrated Gradients, DeepLIFT, GradientSHAP (baselines drawn from 32 PTB-XL training ECGs), Input × Gradient, Grad-CAM and sliding-window Occlusion (100 ms windows across all leads). For each it reports the share and enrichment of evidence for AF in atrial, QRS and ST-T windows in AF ECGs, the share of evidence *against* AF in atrial windows in non-AF ECGs, a deletion test (100 ms segments removed most-relevant-first and least-relevant-first; area between the curves, compared with random order), the correlation of each map with the step 8 counterfactual effects, and the agreement between every pair of methods. A removed segment is a flat line with no P-wave, which the model may itself read as AF evidence, so the deletion test is reported with that caveat (`src/faithfulness.py`). Report: `results/11_attribution_methods_report.md`; deletion curves: `results/xai/deletion_<dataset>.png`.

### 12. Doubtful AF labels

```bash
python scripts/12_label_check.py
```

An ECG labelled both AF and a sinus-type rhythm (sinus rhythm, sinus brady- or tachycardia, sinus arrhythmia) most likely had AF at another time, not in this 10 s strip. The check finds these from the labels alone, never from model output, and reports how many there are (overall and in the low-RR-variability and near-regular AF groups), how often each model detects them compared with clean AF labels, and performance with them left out as a sensitivity analysis. The main results always keep every label as given. Report: `results/12_label_check_report.md`.

**Blinded clinician review.** For a direct check, a clinician can review the doubtful ECGs without seeing labels or model output:

```bash
python scripts/review_set.py make --dataset georgia
python scripts/review_set.py score --dataset georgia
```

`make` takes every near-regular AF ECG (strict rule; `--group main` for the wider rule) and mixes in controls (30 irregular AF, 30 sinus rhythm, 15 other non-AF), shuffled under anonymous IDs. It writes `results/review/<dataset>/review_ecgs.pdf` (one 12-lead ECG per page at 25 mm/s and 10 mm/mV), `review_sheet.csv`, `INSTRUCTIONS.txt`, and the answer key in `key/`, which stays with you. The reviewer marks each ECG AF, not AF or unsure. `score` reports agreement with the official labels in each group (the controls show how reliable the review is) and how the models did on AF labels the reviewer confirmed vs rejected. Report: `results/review_report_<dataset>.md`.

### Second network and second external set

Every step from 6 on takes `--arch inception` (except 9, which takes `--arch` or `--model`) and works on Georgia as well as Chapman. Other architectures add `_<arch>` to their output file names; a single `--dataset` adds `_<dataset>` to report names, so nothing overwrites the main ResNet reports. Order after downloading Georgia and training InceptionTime:

```bash
python scripts/01_count_labels.py
python scripts/02_preprocess.py --dataset georgia
python scripts/03_rpeaks.py --dataset georgia
python scripts/06_evaluate.py
python scripts/07_attributions.py --dataset georgia
python scripts/07_attributions.py --arch inception
python scripts/08_occlusion.py --dataset georgia --seed all
python scripts/08_occlusion.py --arch inception --seed all
python scripts/09_false_alarms.py --dataset georgia
python scripts/09_false_alarms.py --arch inception
python scripts/10_occlusion_tests.py
python scripts/10_occlusion_tests.py --arch inception
python scripts/11_attribution_methods.py
python scripts/11_attribution_methods.py --arch inception
python scripts/plot_calibration.py
```

### Check the setup at any time

```bash
python scripts/check_setup.py
```

---

## Key design decisions

All settings are in `src/config.py`.

| Setting | Value | Reason |
|---|---|---|
| Sampling rate | 500 Hz, 10 s, both datasets | Keeps atrial detail; the same rate for both rules out a sampling-rate effect |
| Split | Patient-level, stratified by AF and paced, ~80/10/10, seed 42 | No patient in two splits; same AF rate in each split |
| External set | Chapman-Shaoxing, test only | A different hospital and recording setup |
| Filtering | 0.5–100 Hz Butterworth, order 2, zero-phase; edges padded at a median level | Removes wander and noise without shifting or blurring P/f-waves; stops start-up artefacts spreading |
| Normalisation | Per lead, PTB-XL training statistics only | No information from validation, test or external sets |
| Class imbalance | Class-weighted loss | ~7% AF in PTB-XL |
| Threshold | Youden's J on validation, then locked | Chosen without test data |
| Primary model | Seed with best validation AUPRC | Chosen without test data |
| Regular-rate AF | rr_cv ≤ 90th percentile of sinus rhythm (strict: 75th) | "As regular as a normal heartbeat" |
| Beat windows | QRS R−60 to R+100 ms; ST-T to the later of the measured T-wave end + 20 ms and QRS onset + 0.44·√RR; atrial to next R−60 ms | Conservative against labelling ventricular activity (including T-wave tails) as atrial |
| Edge margin | First and last 0.5 s left out of attribution and never changed by occlusion | Edges of a 10 s strip are uncertain |
| Occlusion | Straight-line replacement; RR regularisation keeps QRS and ST-T exact | Isolates atrial content from timing without distorting ventricular shapes |

---

## Project structure

```
af-ecg-xai/
├── setup.sh                       environment setup
├── requirements.txt
├── scripts/
│   ├── download_data.sh           PTB-XL + Chapman download
│   ├── download_challenge.sh      parallel Chapman or Georgia download
│   ├── download_chapman_fast.sh   shortcut for the Chapman download
│   ├── check_setup.py             packages, GPU, data
│   ├── 01_count_labels.py         label tables, split, counts
│   ├── 02_preprocess.py           load, check, filter, save signals
│   ├── 03_rpeaks.py               R-peaks, RR features, regular-rate AF
│   ├── 04_rr_baseline.py          RR-feature logistic regression
│   ├── 05_train.py                train the 1D ResNet
│   ├── 06_evaluate.py             held-out metrics, CIs, calibration, subgroups
│   ├── 07_attributions.py         Integrated Gradients, Grad-CAM, window shares
│   ├── 08_occlusion.py            occlusion and counterfactual tests, one or all seeds
│   ├── 09_false_alarms.py         false alarms by rhythm (external sets)
│   ├── 10_occlusion_tests.py      paired tests for swap and P-wave removal, every seed
│   ├── 11_attribution_methods.py  six attribution methods, deletion test, agreement
│   ├── 12_label_check.py          AF labels that conflict with a sinus-rhythm label
│   ├── review_set.py              blinded clinician review of doubtful labels
│   ├── plot_calibration.py        calibration plots from saved predictions
│   ├── plot_examples.py           ECGs with R-peaks, for visual checks
│   └── plot_raw_vs_filtered.py    one ECG before and after filtering
├── src/
│   ├── config.py                  all settings
│   ├── labels.py                  label loaders for both datasets
│   ├── splits.py                  patient-level stratified split
│   ├── signals.py                 read, check and filter one ECG
│   ├── rr.py                      R-peak detection and RR features
│   ├── data.py                    PyTorch dataset, normalisation, augmentation
│   ├── model.py                   1D ResNet and InceptionTime
│   ├── metrics.py                 metrics and threshold rule
│   ├── evaluation.py              bootstrap CIs and calibration
│   ├── plots.py                   calibration plot
│   ├── snomed_codes.csv           SNOMED-CT code names (Challenge 2021)
│   ├── runs.py                    trained runs and primary model per network
│   ├── selection.py               which held-out ECGs are analysed
│   ├── windows.py                 atrial / QRS / ST-T windows, T-wave ends
│   ├── xai.py                     attribution methods
│   ├── faithfulness.py            deletion test
│   ├── dx.py                      diagnosis codes and rhythm groups (Challenge sets)
│   ├── stats.py                   paired tests, bootstrap, multiple-testing correction
│   └── perturb.py                 occlusion and counterfactual changes
├── data/raw/                      downloaded data (not tracked)
├── data/processed/                label tables, signal arrays (not tracked)
├── results/                       reports, models, plots (not tracked)
└── notebooks/
```

---

## Datasets and licences

The code in this repository is released under the MIT License (see `LICENSE`).

Both datasets are de-identified and public on PhysioNet. Cite them if you use this code:

- Wagner P, Strodthoff N, Bousseljot R-D, et al. PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data* 7, 154 (2020). https://doi.org/10.1038/s41597-020-0495-6
- Zheng J, Zhang J, Danioko S, et al. A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients. *Scientific Data* 7, 48 (2020). https://doi.org/10.1038/s41597-020-0386-x
- Reyna MA, Sadr N, Alday EAP, et al. Will two do? Varying dimensions in electrocardiography: the PhysioNet/Computing in Cardiology Challenge 2021. *Computing in Cardiology* 48 (2021). https://doi.org/10.23919/CinC53138.2021.9662687
- Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220 (2000).

See each dataset's PhysioNet page for its licence terms.
