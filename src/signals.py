"""Load, check and filter one ECG recording.

- same pipeline for both datasets
- resample to a common rate, fixed 10 s length
- minimal filtering, so atrial morphology (P-waves, f-waves) is kept
- consistent exclusion of missing/flat leads, corrupted or short signals

Normalisation is NOT done here. Its mean and std must come from the
PTB-XL training set only, so it's computed in scripts/02_preprocess.py
and applied when the data is loaded for training.
"""
from pathlib import Path

import numpy as np
import scipy.io
import wfdb
from scipy.signal import butter, resample_poly, sosfiltfilt

from src import config

# Standard lead names in upper case, to match both datasets' spelling
# (PTB-XL writes AVR, Chapman writes aVR).
LEADS_UPPER = [lead.upper() for lead in config.LEADS]


class Excluded(Exception):
    """Raised when a recording fails a check. The message is the reason."""


def _read_raw(path: str):
    """Return (signal [samples, leads] in mV, fs, lead names).

    Tries wfdb first. The Challenge 2021 copy of Chapman stores signals in
    .mat files; if wfdb can't read those, fall back to scipy and apply the
    gain and baseline from the header ourselves.
    """
    try:
        sig, fields = wfdb.rdsamp(path)
        return sig, float(fields["fs"]), list(fields["sig_name"])
    except Exception:
        mat = Path(path + ".mat")
        if not mat.exists():
            raise Excluded("unreadable file")
        try:
            hdr = wfdb.rdheader(path)
            raw = scipy.io.loadmat(mat)["val"].astype(np.float64)   # [leads, samples]
            gain = np.asarray(hdr.adc_gain, dtype=float)[:, None]
            base = np.asarray(hdr.baseline, dtype=float)[:, None]
            sig = ((raw - base) / gain).T
            return sig, float(hdr.fs), list(hdr.sig_name)
        except Exception:
            raise Excluded("unreadable file")


def _reorder_leads(sig: np.ndarray, names: list) -> np.ndarray:
    """Put the 12 leads in the standard order. Missing lead -> excluded."""
    upper = [n.upper() for n in names]
    idx = []
    for lead in LEADS_UPPER:
        if lead not in upper:
            raise Excluded("missing lead")
        idx.append(upper.index(lead))
    return sig[:, idx]


def _filter(sig: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase band-pass, settings in config (default 0.5-100 Hz).

    0.5 Hz removes baseline wander; 100 Hz removes high-frequency noise.
    Both are well outside the f-wave band (roughly 4-9 Hz) and P-wave
    content, so atrial morphology is kept. Zero-phase filtering
    (forward + backward) doesn't shift any waves in time.

    Edges: each end is extended with a flat line at the median of its
    nearest 0.2 s before filtering, and the extension is cut off after.
    scipy's default instead mirrors the signal around its very first
    sample, so a start-up jolt in the recording (seen in some PTB-XL
    ECGs) became a big half-second swoop. Real samples are not changed.
    """
    sos = butter(config.FILTER_ORDER,
                 [config.FILTER_LOW_HZ, config.FILTER_HIGH_HZ],
                 btype="bandpass", fs=fs, output="sos")
    pad = int(config.FILTER_PAD_S * fs)
    ref = int(config.FILTER_PAD_REF_S * fs)
    left = np.repeat(np.median(sig[:ref], axis=0, keepdims=True), pad, axis=0)
    right = np.repeat(np.median(sig[-ref:], axis=0, keepdims=True), pad, axis=0)
    out = sosfiltfilt(sos, np.concatenate([left, sig, right]), axis=0)
    return out[pad:pad + sig.shape[0]]


def edge_deviation(filtered: np.ndarray, fs: float) -> float:
    """Largest leftover baseline offset (mV) at either edge, over all leads.

    Uses the median of the first and last EDGE_DEV_WINDOW_S of each
    filtered lead. A clean edge sits near 0; a filter swoop or a
    start-up artefact pushes the median away from 0.
    """
    w = int(config.EDGE_DEV_WINDOW_S * fs)
    start = np.abs(np.median(filtered[:w], axis=0))
    end = np.abs(np.median(filtered[-w:], axis=0))
    return float(np.maximum(start, end).max())


def middle_deviation(filtered: np.ndarray, fs: float) -> float:
    """The same measure as edge_deviation, but on two windows in the middle
    of the recording (starting at 3 s and 6.5 s), where there are no edge
    effects. Comparing the two tells an edge artefact apart from a
    difference that's just how the rhythm looks everywhere."""
    w = int(config.EDGE_DEV_WINDOW_S * fs)
    a, b = int(3.0 * fs), int(6.5 * fs)
    m1 = np.abs(np.median(filtered[a:a + w], axis=0))
    m2 = np.abs(np.median(filtered[b:b + w], axis=0))
    return float(np.maximum(m1, m2).max())


def load_and_clean(path: str) -> tuple[np.ndarray, float, float]:
    """Return (recording as float32 [12 leads, N_SAMPLES], edge deviation
    in mV, middle deviation in mV), or raise Excluded."""
    sig, fs, names = _read_raw(path)

    if sig.ndim != 2 or sig.size == 0:
        raise Excluded("corrupted waveform")
    if not np.isfinite(sig).all():
        raise Excluded("corrupted waveform")

    sig = _reorder_leads(sig, names)

    # Resample to the common rate if needed (both datasets are 500 Hz,
    # but this keeps the pipeline correct if that ever changes)
    if round(fs) != config.TARGET_FS:
        sig = resample_poly(sig, config.TARGET_FS, int(round(fs)), axis=0)

    # Fixed length: too short is excluded; longer is cut to the first 10 s
    if sig.shape[0] < config.N_SAMPLES:
        raise Excluded("shorter than 10 s")
    sig = sig[: config.N_SAMPLES]

    # Flat lead = (almost) no signal, e.g. a disconnected electrode
    if (sig.std(axis=0) < config.FLAT_LEAD_STD_MV).any():
        raise Excluded("flat lead")

    sig = _filter(sig, config.TARGET_FS)
    edge = edge_deviation(sig, config.TARGET_FS)
    middle = middle_deviation(sig, config.TARGET_FS)
    return sig.T.astype(np.float32), edge, middle   # [leads, samples]
