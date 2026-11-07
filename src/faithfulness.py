"""Deletion test: does removing what a map calls important change the output?

The 10 s ECG is cut into 100 ms segments (the 0.5 s edges are never
touched). Segments are ranked by the map's evidence for AF (attribution
summed over leads and over the segment), then removed in that order
(most relevant first, MoRF) and in the reverse order (least relevant
first, LeRF), in steps of 5% up to half the segments. Removed segments are
set to the training mean in every lead (zero after normalisation, the same
baseline as Integrated Gradients).

Score: area between the two curves (ABPC) = mean over steps of
P(AF | LeRF) - P(AF | MoRF). For ECGs the model calls AF, a faithful map
gives a large positive ABPC; a random ranking gives about zero.

Caveat specific to this task: a flattened segment has no P-wave, which the
model may itself read as AF evidence. Deletion therefore partly measures
the model's reaction to the flat line, not only the map. The counterfactual
tests in step 8 avoid this, which is why the comparison step also checks
each map against those results.
"""
import numpy as np
import torch

from src import config

SEGMENT_MS = 100
FRACTIONS = np.round(np.arange(0.0, 0.501, 0.05), 2)


def segments(n_samples: int = config.N_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    """(segment start indices, eligible mask); edge segments are not eligible."""
    w = int(SEGMENT_MS * config.TARGET_FS / 1000)
    starts = np.arange(0, n_samples - w + 1, w)
    margin = int(config.EDGE_MARGIN_S * config.TARGET_FS)
    ok = (starts >= margin) & (starts + w <= n_samples - margin)
    return starts, ok


def segment_scores(attr_time: np.ndarray) -> np.ndarray:
    """Sum of a time profile [samples] within each segment."""
    starts, _ = segments(len(attr_time))
    w = int(SEGMENT_MS * config.TARGET_FS / 1000)
    return np.array([attr_time[s:s + w].sum() for s in starts])


def _deleted(x: np.ndarray, order: np.ndarray, frac: float) -> np.ndarray:
    starts, ok = segments(x.shape[-1])
    w = int(SEGMENT_MS * config.TARGET_FS / 1000)
    n = int(round(frac * ok.sum()))
    out = x.copy()
    for i in order[:n]:
        out[:, starts[i]:starts[i] + w] = 0.0
    return out


@torch.no_grad()
def curves(model, x: np.ndarray, rankings: dict, device, batch: int = 64) -> dict:
    """x: one normalised ECG [12, samples]. rankings: {name: segment scores}.
    Returns {name: (morf probs, lerf probs)} over FRACTIONS."""
    _, ok = segments(x.shape[-1])
    elig = np.flatnonzero(ok)
    jobs = []
    for name, sc in rankings.items():
        order = elig[np.argsort(-sc[elig], kind="stable")]       # most evidence for AF first
        for o in (order, order[::-1]):                           # MoRF, then LeRF
            for f in FRACTIONS:
                jobs.append(_deleted(x, o, f))
    probs = []
    for i in range(0, len(jobs), batch):
        xb = torch.from_numpy(np.stack(jobs[i:i + batch]).astype(np.float32)).to(device)
        probs.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    probs = np.concatenate(probs)
    out, k = {}, 0
    for name in rankings:
        morf = probs[k:k + len(FRACTIONS)]
        lerf = probs[k + len(FRACTIONS):k + 2 * len(FRACTIONS)]
        out[name] = (morf, lerf)
        k += 2 * len(FRACTIONS)
    return out


def abpc(morf: np.ndarray, lerf: np.ndarray) -> float:
    return float(np.mean(lerf - morf))
