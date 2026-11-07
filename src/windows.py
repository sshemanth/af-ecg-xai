"""Atrial and ventricular windows for one ECG, from its R-peaks.

Windows are defined in advance relative to the detected QRS complexes:
the atrial window covers where P-waves (sinus) or fibrillatory waves (AF)
are expected; the QRS window, which also carries the RR timing, stands for
ventricular evidence. Boundaries are set in src/config.py.

Each sample of the 10 s ECG gets one code:
    0  outside  (before first / after last R-peak, or in the 0.5 s edge margins)
    1  atrial   (from end of ST-T to the next QRS: P-waves or f-waves)
    2  QRS      (R - 60 .. R + 100 ms: ventricular depolarisation, also carries RR timing)
    3  ST-T     (after QRS until the estimated end of the T-wave)
"""
import numpy as np

from src import config

OUTSIDE, ATRIAL, QRS, STT = 0, 1, 2, 3
NAMES = {ATRIAL: "atrial", QRS: "qrs", STT: "stt"}


def beat_windows(rpeaks: np.ndarray, n: int = config.N_SAMPLES,
                 fs: int = config.TARGET_FS) -> np.ndarray:
    codes = np.zeros(n, dtype=np.int8)
    r = np.sort(np.asarray(rpeaks, dtype=int))
    if len(r) < 2:
        return codes
    before = int(config.QRS_BEFORE_MS * fs / 1000)
    after = int(config.QRS_AFTER_MS * fs / 1000)
    onset = int(config.QRS_ONSET_MS * fs / 1000)

    for a, b in zip(r[:-1], r[1:]):                      # one cycle: R_a .. R_b
        rr = (b - a) / fs
        stt_end = a - onset + int(config.QTC_S * np.sqrt(rr) * fs)
        stt_end = min(stt_end, b - before)               # can't run into the next QRS
        codes[a + after:stt_end] = STT
        codes[max(stt_end, a + after):b - before] = ATRIAL
    for p in r:                                          # QRS last, so it wins overlaps
        codes[max(0, p - before):min(n, p + after + 1)] = QRS

    codes[:max(0, r[0] - before)] = OUTSIDE                    # before the first QRS
    codes[min(n, r[-1] + after + 1):] = OUTSIDE                # after the last QRS
    margin = int(config.EDGE_MARGIN_S * fs)
    codes[:margin] = OUTSIDE
    codes[n - margin:] = OUTSIDE
    return codes


def shares(attr_time: np.ndarray, codes: np.ndarray) -> dict:
    """Share of positive attribution (evidence FOR AF) in each window, the
    share of time each window takes, and their ratio (enrichment).

    attr_time: attribution per sample, already summed over leads [n].
    Enrichment > 1 means the window gets more attribution than its length
    alone would give it. Comparing enrichment is fairer than raw shares,
    because atrial windows are much longer than QRS windows.
    """
    pos = np.clip(attr_time, 0, None)
    inside = codes != OUTSIDE
    total = pos[inside].sum()
    n_in = inside.sum()
    out = {}
    for code, name in NAMES.items():
        m = codes == code
        mass = pos[m].sum() / total if total > 0 else np.nan
        time = m.sum() / n_in if n_in else np.nan
        out[f"{name}_share"] = float(mass)
        out[f"{name}_time"] = float(time)
        out[f"{name}_enrich"] = float(mass / time) if time and time > 0 else np.nan
    return out
