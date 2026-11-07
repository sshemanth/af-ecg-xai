"""R-peak detection and RR-interval features.

Beat locations are used ONLY to (1) derive RR-related variables (for the
RR baseline model and the regular-rate AF subgroup) and (2) define the
atrial / ventricular windows for attribution and occlusion.
They never replace the raw waveform as input to the main model.
"""
import warnings

import neurokit2 as nk
import numpy as np

from src import config

# Leads to try in order. Lead II usually has the clearest R-peaks; fall back
# to V5 and I if II gives too few beats.
LEAD_ORDER = [config.LEADS.index(name) for name in config.RPEAK_LEADS]


def detect_rpeaks(sig: np.ndarray, fs: int = config.TARGET_FS) -> tuple[np.ndarray, str]:
    """Return (R-peak sample indices, lead used) for one ECG [12, samples].

    Uses NeuroKit2's default detector, which has been benchmarked on
    public annotated ECG databases. Returns an empty array if no lead
    gives a plausible set of beats.
    """
    for li in LEAD_ORDER:
        x = sig[li].astype(np.float64)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clean = nk.ecg_clean(x, sampling_rate=fs, method=config.RPEAK_METHOD)
                _, info = nk.ecg_peaks(clean, sampling_rate=fs,
                                       method=config.RPEAK_METHOD, correct_artifacts=False)
        except Exception:
            continue
        peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
        if _plausible(peaks, fs):
            return peaks, config.LEADS[li]
    return np.array([], dtype=int), ""


def _plausible(peaks: np.ndarray, fs: int) -> bool:
    """Enough beats, and a heart rate a human can have."""
    if len(peaks) < config.MIN_BEATS:
        return False
    hr = 60 * fs / np.diff(peaks).mean()
    return config.HR_RANGE[0] <= hr <= config.HR_RANGE[1]


def rr_features(peaks: np.ndarray, fs: int = config.TARGET_FS) -> dict:
    """RR-interval features for one ECG. All NaN if too few beats.

    rr_cv   = SD / mean of RR intervals (overall irregularity)
    nrmssd  = RMSSD / mean RR (beat-to-beat irregularity, a common AF marker)
    pnn50   = share of successive RR differences > 50 ms
    """
    names = ["n_beats", "hr_mean", "rr_mean", "rr_sd", "rr_cv",
             "rmssd", "nrmssd", "pnn50"]
    if len(peaks) < config.MIN_BEATS:
        out = dict.fromkeys(names, np.nan)
        out["n_beats"] = len(peaks)
        return out
    rr = np.diff(peaks) / fs                    # seconds
    d = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(d ** 2)))
    return {
        "n_beats": len(peaks),
        "hr_mean": float(60 / rr.mean()),
        "rr_mean": float(rr.mean()),
        "rr_sd": float(rr.std(ddof=1)),
        "rr_cv": float(rr.std(ddof=1) / rr.mean()),
        "rmssd": rmssd,
        "nrmssd": float(rmssd / rr.mean()),
        "pnn50": float(np.mean(np.abs(d) > 0.050)),
    }
