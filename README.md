# AF Detection from 12-Lead ECG Across Hospitals

Deep learning detects atrial fibrillation (AF) on the ECG very accurately, but it is not clear *what* the models rely on: the atrial signs of AF (missing P-waves, fibrillatory f-waves) or the simpler cue of irregular beat timing (RR intervals).

This repository trains a 1D ResNet on 12-lead ECGs from **PTB-XL**, tests it without retraining on **Chapman-Shaoxing** from a different hospital, compares it with a model that only sees RR intervals, and uses Integrated Gradients and Grad-CAM to measure how much of the model's evidence falls on atrial versus ventricular parts of each beat. AF with a regular ventricular response and paced rhythms, where the RR cue is weak, are analysed as subgroups.

**Keywords:** atrial fibrillation · ECG · electrocardiogram · 12-lead ECG · deep learning · 1D CNN · ResNet · explainable AI · XAI · Grad-CAM · integrated gradients · occlusion · counterfactual explanations · external validation · domain shift · generalisation · PTB-XL · Chapman-Shaoxing · PyTorch · medical AI · cardiology

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

Large arrays are memory-mapped from disk, so 16 GB RAM is enough. Processed signals take about 2.6 GB (PTB-XL) and 1.2 GB (Chapman) as float16.

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

| | PTB-XL | Chapman-Shaoxing |
|---|---|---|
| Role | Training, validation, internal test | External test only (never used for training or tuning) |
| Source | PhysioNet, v1.0.3 | PhysioNet/CinC Challenge 2021 copy |
| Format | 12-lead, 500 Hz, 10 s | 12-lead, 500 Hz, 10 s |
| Labels | SCP-ECG statements | SNOMED-CT codes |
| ECGs kept after preprocessing | 21,742 (7.0% AF) | 9,794 (18.2% AF) |

```bash
bash scripts/download_data.sh            # PTB-XL, then Chapman one file at a time
bash scripts/download_chapman_fast.sh    # faster Chapman download (6 folders in parallel)
```

Both scripts can be re-run; they resume and skip finished files. Data goes to `data/raw/` and is not tracked by git.

**Label definitions**

| Variable | PTB-XL | Chapman | Meaning |
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
```

Loads every ECG, puts the leads in standard order, excludes bad recordings (missing or flat lead, unreadable or NaN, shorter than 10 s, flutter-only, no diagnosis code), applies a 0.5–100 Hz zero-phase band-pass and saves float16 arrays. Per-lead normalisation statistics come from the PTB-XL training split only (`data/processed/norm_stats.json`). The report includes exclusions by reason, AF per split, an amplitude check and an edge check that compares AF with non-AF. Add `--limit 200` for a quick test, `--workers N` to set CPU processes.

### 3. R-peaks and RR features

```bash
python scripts/03_rpeaks.py --dataset ptbxl
python scripts/03_rpeaks.py --dataset chapman
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

### 5. Train the 1D ResNet

```bash
python scripts/05_train.py --limit 2000 --epochs 2
python scripts/05_train.py --seed 1
python scripts/05_train.py --seed all
python scripts/05_train.py --seed 2,3,4,5
```

The first command is a quick check; `--seed all` trains seeds 1–5. Training uses the training split only, early stopping on validation AUPRC, a class-weighted loss, and augmentation that keeps atrial morphology (random 9 s crop, amplitude scaling, slow baseline wander). The decision threshold is chosen on validation (Youden's J) and saved with each run in `results/models/resnet_seed<N>/`. The test split and Chapman are never loaded during training.

### 6. Evaluate on held-out data

```bash
python scripts/06_evaluate.py --dry-run
python scripts/06_evaluate.py
```

`--dry-run` checks readiness without touching test data. The evaluation scores the PTB-XL test split and Chapman with the locked models and thresholds: AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1 and confusion counts with patient-level bootstrap 95% CIs; calibration (plot, intercept, slope, Brier score, ECE); a paired comparison of the ResNet with the RR baseline on the same ECGs; internal vs external AUROC; and detection rates in the low-RR-variability and near-regular AF subgroups. The primary model is the seed with the best validation AUPRC; all seeds are reported.

Held-out data is meant to be used once: each dataset is locked after evaluation (`results/eval/LOCK_<dataset>.json`) and the script refuses to evaluate it again without `--force`. It also stops until all seeds are trained (override with `--partial`).

### 7. Attribution

```bash
python scripts/07_attributions.py --n-per-class 20
python scripts/07_attributions.py
```

The first command is a quick check. The script explains the primary model on held-out ECGs with Integrated Gradients (lead × time) and Grad-CAM (time). Each beat is split, from its R-peaks, into atrial (P/f-waves), QRS and ST-T windows; the report gives the share of evidence for AF in each window and its enrichment (share ÷ time), by group (AF, non-AF, low-RR-variability AF, near-regular AF, paced AF), by lead, for caught vs missed AF, and PTB-XL vs Chapman. It also reports agreement between the two methods and stability under 0.01 mV noise. Example plots: `results/xai/examples_<dataset>.png`. The first and last 0.5 s of each ECG are excluded.

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
| Normalisation | Per lead, PTB-XL training statistics only | No information from validation, test or Chapman |
| Class imbalance | Class-weighted loss | ~7% AF in PTB-XL |
| Threshold | Youden's J on validation, then locked | Chosen without test data |
| Primary model | Seed with best validation AUPRC | Chosen without test data |
| Regular-rate AF | rr_cv ≤ 90th percentile of sinus rhythm (strict: 75th) | "As regular as a normal heartbeat" |
| Beat windows | QRS R−60 to R+100 ms; ST-T to QRS onset + 0.44·√RR; atrial to next R−60 ms | Conservative against labelling ventricular activity as atrial |
| Edge margin | First and last 0.5 s left out of attribution | Edges of a 10 s strip are uncertain |

---

## Project structure

```
af-ecg-xai/
├── setup.sh                       environment setup
├── requirements.txt
├── scripts/
│   ├── download_data.sh           PTB-XL + Chapman download
│   ├── download_chapman_fast.sh   parallel Chapman download
│   ├── check_setup.py             packages, GPU, data
│   ├── 01_count_labels.py         label tables, split, counts
│   ├── 02_preprocess.py           load, check, filter, save signals
│   ├── 03_rpeaks.py               R-peaks, RR features, regular-rate AF
│   ├── 04_rr_baseline.py          RR-feature logistic regression
│   ├── 05_train.py                train the 1D ResNet
│   ├── 06_evaluate.py             held-out metrics, CIs, calibration, subgroups
│   ├── 07_attributions.py         Integrated Gradients, Grad-CAM, window shares
│   ├── plot_examples.py           ECGs with R-peaks, for visual checks
│   └── plot_raw_vs_filtered.py    one ECG before and after filtering
├── src/
│   ├── config.py                  all settings
│   ├── labels.py                  label loaders for both datasets
│   ├── splits.py                  patient-level stratified split
│   ├── signals.py                 read, check and filter one ECG
│   ├── rr.py                      R-peak detection and RR features
│   ├── data.py                    PyTorch dataset, normalisation, augmentation
│   ├── model.py                   1D ResNet
│   ├── metrics.py                 metrics and threshold rule
│   ├── evaluation.py              bootstrap CIs and calibration
│   ├── runs.py                    trained runs and primary model
│   ├── windows.py                 atrial / QRS / ST-T windows
│   └── xai.py                     attribution methods
├── data/raw/                      downloaded data (not tracked)
├── data/processed/                label tables, signal arrays (not tracked)
├── results/                       reports, models, plots (not tracked)
└── notebooks/
```

---

## Datasets and licences

Both datasets are de-identified and public on PhysioNet. Cite them if you use this code:

- Wagner P, Strodthoff N, Bousseljot R-D, et al. PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data* 7, 154 (2020). https://doi.org/10.1038/s41597-020-0495-6
- Zheng J, Zhang J, Danioko S, et al. A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients. *Scientific Data* 7, 48 (2020). https://doi.org/10.1038/s41597-020-0386-x
- Reyna MA, Sadr N, Alday EAP, et al. Will two do? Varying dimensions in electrocardiography: the PhysioNet/Computing in Cardiology Challenge 2021. *Computing in Cardiology* 48 (2021). https://doi.org/10.23919/CinC53138.2021.9662687
- Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220 (2000).

See each dataset's PhysioNet page for its licence terms.
