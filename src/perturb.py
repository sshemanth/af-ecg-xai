"""Controlled changes to an ECG, to test what the model NEEDS.

All functions take a normalised ECG x [12 leads, samples] and the beat
window codes from src/windows.py, and return a modified copy. The edge
margins (code OUTSIDE) are never changed by the window-based edits.

Occlusion
  remove_windows     replace every window of one type with a straight line
                     between its end points (removes P/f-waves, QRS or ST-T
                     shape; beat timing is untouched for atrial / ST-T)
  matched_control    the same total length removed as the atrial windows,
                     but shifted to random places (which may include QRS or
                     ST-T): separates "atrial content mattered" from
                     "removing any signal of that length mattered"

Counterfactuals
  regularise_rr      beats moved to perfectly even spacing; each QRS and ST-T
                     keeps its exact shape and only the atrial window is
                     stretched or squeezed to fit, so atrial content stays
  swap_atrial        atrial windows filled with atrial activity from another
                     ECG (sinus P-waves into AF, or AF f-waves into sinus),
                     beat timing unchanged
"""
import numpy as np

from src import config
from src.windows import ATRIAL, OUTSIDE

BEFORE = int(config.QRS_BEFORE_MS * config.TARGET_FS / 1000)
AFTER = int(config.QRS_AFTER_MS * config.TARGET_FS / 1000)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/end (end exclusive) of each contiguous True stretch."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def _bridge(x: np.ndarray, a: int, b: int, leads) -> None:
    """In place: straight line across x[leads, a:b] joining the samples
    just outside the stretch, so there's no jump at either end."""
    left, right = max(a - 1, 0), min(b, x.shape[1] - 1)
    t = np.linspace(0, 1, b - a + 2)[1:-1]
    x[leads, a:b] = x[leads, left, None] + (x[leads, right, None] - x[leads, left, None]) * t


def remove_windows(x: np.ndarray, codes: np.ndarray, code: int, leads=None) -> np.ndarray:
    leads = list(range(x.shape[0])) if leads is None else leads
    y = x.copy()
    for a, b in runs(codes == code):
        _bridge(y, a, b, leads)
    return y


def matched_control(x: np.ndarray, codes: np.ndarray, rng: np.random.Generator,
                    code: int = ATRIAL) -> np.ndarray:
    """Same stretches as `code`, rotated by a random offset inside the
    analysed span, then removed the same way."""
    inside = np.flatnonzero(codes != OUTSIDE)
    if len(inside) == 0:
        return x.copy()
    lo, hi = inside[0], inside[-1] + 1
    span = hi - lo
    shift = int(rng.integers(int(0.2 * span), int(0.8 * span) + 1))
    m = np.zeros_like(codes, dtype=bool)
    m[lo:hi] = np.roll(codes[lo:hi] == code, shift)
    y = x.copy()
    for a, b in runs(m):
        _bridge(y, a, b, list(range(x.shape[0])))
    return y


def regularise_rr(x: np.ndarray, peaks: np.ndarray, codes: np.ndarray) -> np.ndarray | None:
    """Even beat spacing between the first and last R-peak.

    Each beat's QRS and ST-T (R-peak to the start of its atrial window) and
    the next QRS's leading part are copied unchanged; only the atrial
    window is resampled to fill the new spacing. Ventricular shapes are
    therefore identical, and pieces join at the same samples as in the
    original, so there are no jumps. Returns None if a new beat gap is too
    short to hold its unchanged QRS and ST-T.
    """
    r = np.sort(np.asarray(peaks, dtype=int))
    if len(r) < 3:
        return None
    target = np.round(np.linspace(r[0], r[-1], len(r))).astype(int)
    y = x.copy()
    pos = r[0]
    for i in range(len(r) - 1):
        a, b = r[i], r[i + 1]
        head = b - BEFORE                                  # start of next QRS
        atr = np.flatnonzero(codes[a:head] == ATRIAL)
        start = a + atr[0] if len(atr) else head           # end of this beat's ST-T
        fixed_tail = x[:, a:start]                         # QRS (after R) + ST-T, unchanged
        new_atrial = (target[i + 1] - target[i]) - (start - a) - BEFORE
        if new_atrial < 0:
            return None
        y[:, pos:pos + fixed_tail.shape[1]] = fixed_tail
        pos += fixed_tail.shape[1]
        if new_atrial:
            src = x[:, start - 1:head + 1]                 # atrial window plus its two neighbours
            t_src = np.linspace(0, 1, src.shape[1])
            t_new = np.linspace(0, 1, new_atrial + 2)[1:-1]
            y[:, pos:pos + new_atrial] = np.stack([np.interp(t_new, t_src, lead) for lead in src])
            pos += new_atrial
        y[:, pos:pos + BEFORE] = x[:, head:b]              # next QRS's leading part
        pos += BEFORE
    return y if pos == r[-1] else None


def _detrend(seg: np.ndarray) -> np.ndarray:
    """Subtract the straight line between a segment's end points, so it
    starts and ends at zero."""
    t = np.linspace(0, 1, seg.shape[-1])
    return seg - (seg[..., :1] + (seg[..., -1:] - seg[..., :1]) * t)


def swap_atrial(x: np.ndarray, codes: np.ndarray, donor: np.ndarray,
                donor_codes: np.ndarray) -> np.ndarray | None:
    """Fill each atrial window of x with the donor's atrial activity.

    The donor's atrial stretches are joined into one stream per lead (each
    detrended so they join without jumps). Each target window takes the
    next piece of that stream, detrended again and laid over the straight
    line between the window's own end points, so it joins smoothly.
    """
    pieces = [_detrend(donor[:, a:b]) for a, b in runs(donor_codes == ATRIAL) if b - a > 4]
    if not pieces:
        return None
    stream = np.concatenate(pieces, axis=1)
    y = x.copy()
    pos = 0
    for a, b in runs(codes == ATRIAL):
        L = b - a
        idx = (pos + np.arange(L)) % stream.shape[1]
        pos += L
        base = y[:, [max(a - 1, 0)]] + (y[:, [min(b, y.shape[1] - 1)]] - y[:, [max(a - 1, 0)]]) \
            * np.linspace(0, 1, L + 2)[1:-1]
        y[:, a:b] = base + _detrend(stream[:, idx])
    return y


def plausible(original: np.ndarray, changed: np.ndarray) -> bool:
    """Automatic check of a changed ECG: same shape, finite, no amplitude
    far outside the original's range, no jump steeper than the original's
    steepest one."""
    if changed is None or changed.shape != original.shape or not np.isfinite(changed).all():
        return False
    amp_ok = np.abs(changed).max() <= 1.5 * np.abs(original).max() + 1.0
    jump_ok = np.abs(np.diff(changed, axis=1)).max() <= 1.5 * np.abs(np.diff(original, axis=1)).max() + 0.1
    return bool(amp_ok and jump_ok)
