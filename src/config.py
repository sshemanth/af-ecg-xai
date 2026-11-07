"""Project-wide settings. Change things here, not inside scripts."""
from pathlib import Path

# ---- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

PTBXL_DIR = RAW_DIR / "ptbxl"
CHAPMAN_DIR = RAW_DIR / "chapman"

# ---- Signal settings -------------------------------------------------------
# Both datasets are recorded at 500 Hz for 10 s. Keeping 500 Hz preserves
# atrial detail (P-waves, f-waves), and both datasets use the same rate so a
# sampling-rate difference can't explain any cross-dataset effect.
TARGET_FS = 500          # Hz
SIGNAL_SECONDS = 10      # seconds per recording
N_SAMPLES = TARGET_FS * SIGNAL_SECONDS
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]

# ---- Preprocessing ---------------------------------------------------------
# Minimal band-pass that leaves P-waves and f-waves (about 4-9 Hz) untouched.
FILTER_LOW_HZ = 0.5      # removes baseline wander
FILTER_HIGH_HZ = 100.0   # removes high-frequency noise
FILTER_ORDER = 2         # Butterworth, applied forward + backward (zero-phase)

# Edge handling. Before filtering, each end is extended with a flat line at
# the median level of its nearest 0.2 s, then the extension is cut off.
# Anchoring on a median (not the first sample) stops a recorder start-up
# artefact from being spread into a half-second swoop by the filter.
FILTER_PAD_S = 2.0
FILTER_PAD_REF_S = 0.2

# Edge check, measured after filtering: the median level of each lead in the
# first and last 0.25 s (a median ignores a heartbeat in the window). Only
# measured, not used to exclude ECGs; the preprocessing report compares AF
# with non-AF and with the middle of the recording, to rule out an edge
# difference the model could use as a shortcut.
EDGE_DEV_WINDOW_S = 0.25
EDGE_DEV_FLAG_MV = 0.2

# The first and last 0.5 s of every ECG are left out of the attribution
# and occlusion analyses, since a 10 s strip's edges are uncertain whatever
# the filter. The model still receives the full 10 s.
EDGE_MARGIN_S = 0.5

# A lead with less variation than this (in mV) counts as flat / missing
FLAT_LEAD_STD_MV = 0.001

# Signals are saved as float16 to halve disk space (PTB-XL ~2.6 GB).
# Rounding error is about 1-4 microvolts at normal ECG amplitudes,
# far below the size of P-waves (~100 uV) or f-waves (~50 uV and up).
SIGNAL_DTYPE = "float16"

# ---- R-peaks and RR features -----------------------------------------------
RPEAK_METHOD = "neurokit"          # NeuroKit2's default detector
RPEAK_LEADS = ["II", "V5", "I"]    # tried in this order
MIN_BEATS = 4                      # fewer than this -> RR features left empty
HR_RANGE = (25, 250)               # bpm; outside this the detection is wrong

# AF with regular ventricular response. An AF ECG counts as "regular" if its
# RR coefficient of variation is no higher than the given percentile of
# rr_cv among sinus-rhythm, non-paced ECGs in the PTB-XL training split,
# i.e. as regular as a normal heartbeat. Training data only; the cut-off is
# computed by scripts/03_rpeaks.py and saved before any model is trained.
REGULAR_RR_METRIC = "rr_cv"
REGULAR_RR_PERCENTILE = 90
# Stricter rule for a sensitivity analysis. Sinus rhythm has a long rr_cv
# tail (ectopic beats, occasional detection slips), so the 90th percentile
# can admit AF that is not truly regular.
REGULAR_RR_STRICT_PERCENTILE = 75
REGULAR_RR_REFERENCE = "sinus rhythm, not paced, PTB-XL train split"

# ---- PTB-XL label codes (keys in the scp_codes column) ---------------------
PTBXL_AF = "AFIB"     # atrial fibrillation
PTBXL_AFL = "AFLT"    # atrial flutter
PTBXL_PACE = "PACE"   # pacemaker
PTBXL_SR = "SR"       # sinus rhythm

# ---- Data split ------------------------------------------------------------
# Patient-level split, stratified by AF status and paced rhythm, with a
# fixed seed. 10 folds -> 1 test, 1 val, 8 train (about 80/10/10).
# See src/splits.py.
SPLIT_N_FOLDS = 10
SPLIT_TEST_FOLD = 0
SPLIT_VAL_FOLD = 1

# ---- Chapman-Shaoxing label codes (SNOMED-CT, Challenge 2021 headers) ------
SNOMED_AF = "164889003"     # atrial fibrillation
SNOMED_AFL = "164890007"    # atrial flutter
SNOMED_PACE = "10370003"    # pacing rhythm
SNOMED_SR = "426783006"     # sinus rhythm

# ---- Label policy ----------------------------------------------------------
# What to do with recordings that have atrial flutter but NOT AF:
#   "exclude"  -> drop them (keeps the AF vs non-AF task clean)  [default]
#   "negative" -> count them as non-AF
#   "positive" -> count them as AF
FLUTTER_POLICY = "exclude"

SEED = 42

# ---- Models ----------------------------------------------------------------
MODELS_DIR = RESULTS_DIR / "models"

# RR-feature logistic regression: a comparator that only sees rhythm. If
# the ResNet beats it, the ResNet is using more than RR irregularity.
RR_FEATURES = ["hr_mean", "rr_mean", "rr_sd", "rr_cv", "rmssd", "nrmssd", "pnn50", "n_beats"]

# 1D ResNet
RESNET_WIDTHS = [64, 128, 192, 256]   # channels in the 4 stages
RESNET_BLOCKS = 2                      # residual blocks per stage
RESNET_KERNEL = 9
DROPOUT = 0.3

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 40
PATIENCE = 6                  # stop if val AUPRC hasn't improved for this many epochs
# Class imbalance (~7% AF): the AF class is weighted in the loss. This pushes
# predicted probabilities upward, which the evaluation's calibration
# analysis reports.
POS_WEIGHT = "balanced"       # "balanced" = n_non_af / n_af, or a number, or None

# Augmentation: only changes that keep atrial morphology. Training uses a
# random 9 s crop (timing shift); val/test use the full 10 s, which works
# because the model ends in global average pooling.
TRAIN_CROP_S = 9.0
AUG_SCALE = (0.9, 1.1)        # amplitude scaling, whole ECG
AUG_BASELINE_MV = 0.05        # slow sine (0.05-0.4 Hz) added, like breathing wander

# Decision threshold, chosen on PTB-XL validation data only, then locked
# before any test or external evaluation. Rule: maximise Youden's J
# (sensitivity + specificity - 1), which weighs both errors equally.
THRESHOLD_RULE = "youden"

# Repeated runs with different seeds, to measure sensitivity to random
# initialisation.
RUN_SEEDS = [1, 2, 3, 4, 5]

# ---- Evaluation ------------------------------------------------------------
EVAL_DIR = RESULTS_DIR / "eval"
N_BOOTSTRAP = 2000            # patient-level bootstrap draws for 95% CIs
CALIBRATION_BINS = 10

# The "primary" ResNet run: the seed with the best VALIDATION AUPRC, so it
# is chosen without looking at test results. All seeds are also reported
# (mean +/- SD).
PRIMARY_MODEL_RULE = "best val AUPRC"

# ---- Attribution -----------------------------------------------------------
XAI_DIR = RESULTS_DIR / "xai"

# Beat windows, built from the R-peaks of each ECG (milliseconds):
#   QRS     R - 60 .. R + 100                         ventricular depolarisation
#   ST-T    R + 100 .. QRS onset + QT                 ventricular repolarisation
#   atrial  end of ST-T .. next R - 60                P-wave (sinus) / f-waves (AF)
# QT is estimated from each beat's RR interval with Bazett's formula,
# QT = QTC_S * sqrt(RR), so the windows stretch and shrink with heart rate.
# Samples before the first or after the last R-peak, and the 0.5 s edge
# margins, belong to no window and are left out.
#
# Boundaries are deliberately conservative toward NOT calling things
# atrial: the QRS runs 100 ms past the R-peak so wide S-waves (e.g. bundle
# branch block) count as ventricular, and QTc 0.44 s (upper normal) makes
# the atrial window start late so T-waves don't spill into it.
QRS_BEFORE_MS = 60
QRS_AFTER_MS = 100
QRS_ONSET_MS = 40          # QRS onset taken as this long before the R-peak
QTC_S = 0.44

# Which ECGs to explain (held-out sets only: PTB-XL test, Chapman): a
# class-balanced random sample, plus every low-RR-variability, near-regular
# and paced AF ECG.
XAI_N_PER_CLASS = 200      # random AF and random non-AF per dataset
XAI_IG_STEPS = 32          # Integrated Gradients steps
XAI_NOISE_MV = 0.01        # small noise for the stability check
XAI_N_STABILITY = 100      # ECGs per dataset re-explained with noise
