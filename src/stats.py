"""Paired statistics for the occlusion tests."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src import config


def boot_mean(v: np.ndarray, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(config.SEED)
    m = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon signed-rank on a - b, with the matched-pairs rank-biserial
    effect size (from -1 to 1; positive means a > b)."""
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    d = d[d != 0]
    if len(d) < 6:
        return {"n": int(ok.sum()), "median_diff": float("nan"), "r": float("nan"), "p": float("nan")}
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    r = (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()
    p = wilcoxon(d).pvalue
    return {"n": int(ok.sum()), "median_diff": float(np.median(a[ok] - b[ok])), "r": float(r), "p": float(p)}


def bh(p: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (NaNs kept as NaN)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        pv = p[ok]
        order = np.argsort(pv)
        ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
        adj = np.minimum.accumulate(ranked[::-1])[::-1]
        res = np.empty_like(pv)
        res[order] = np.minimum(adj, 1.0)
        out[ok] = res
    return out.tolist()


def fmt_p(p: float) -> str:
    return "–" if p != p else ("<0.001" if p < 0.001 else f"{p:.3f}")
