"""Bootstrap confidence intervals and calibration."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config
from src.metrics import summarise

EPS = 1e-6


# ---------------------------------------------------------------------------
# Patient-level bootstrap
# ---------------------------------------------------------------------------
def _patient_groups(patients: np.ndarray) -> list[np.ndarray]:
    """Row indices for each patient, so we resample whole patients."""
    _, inverse = np.unique(patients, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.flatnonzero(np.diff(inverse[order])) + 1
    return np.split(order, bounds)


def bootstrap_rows(patients: np.ndarray, n_boot: int, seed: int):
    """Yield row indices for each bootstrap sample (patients drawn with replacement)."""
    groups = _patient_groups(patients)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        yield np.concatenate([groups[g] for g in pick])


def _metric_fns(threshold: float) -> dict:
    def at_thr(name):
        return lambda y, p: summarise(y, p, threshold)[name]
    return {"auroc": lambda y, p: roc_auc_score(y, p),
            "auprc": lambda y, p: average_precision_score(y, p),
            "sensitivity": at_thr("sensitivity"), "specificity": at_thr("specificity"),
            "ppv": at_thr("ppv"), "npv": at_thr("npv"), "f1": at_thr("f1"),
            "brier": lambda y, p: float(np.mean((p - y) ** 2))}


def bootstrap_ci(y, p, patients, threshold, n_boot=config.N_BOOTSTRAP, seed=config.SEED):
    """95% percentile CIs for every metric, resampling patients."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    fns = _metric_fns(threshold)
    draws = {k: [] for k in fns}
    for rows in bootstrap_rows(np.asarray(patients), n_boot, seed):
        yb, pb = y[rows], p[rows]
        if yb.min() == yb.max():           # need both classes
            continue
        for k, f in fns.items():
            draws[k].append(f(yb, pb))
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
            for k, v in draws.items() if v}


def paired_auroc_diff(y, p_a, p_b, patients, n_boot=config.N_BOOTSTRAP, seed=config.SEED):
    """AUROC(a) - AUROC(b) on the SAME ECGs, with a paired bootstrap 95% CI
    and the share of bootstrap draws where the difference is <= 0."""
    y = np.asarray(y).astype(int)
    diffs = []
    for rows in bootstrap_rows(np.asarray(patients), n_boot, seed):
        yb = y[rows]
        if yb.min() == yb.max():
            continue
        diffs.append(roc_auc_score(yb, p_a[rows]) - roc_auc_score(yb, p_b[rows]))
    diffs = np.asarray(diffs)
    point = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    return {"diff": float(point), "ci": (float(np.percentile(diffs, 2.5)),
                                         float(np.percentile(diffs, 97.5))),
            "p_le_0": float((diffs <= 0).mean())}


def independent_diff_ci(draws_a: np.ndarray, draws_b: np.ndarray) -> tuple[float, float]:
    """CI for a difference between two independent datasets (e.g. internal
    vs external AUROC) from their separate bootstrap draws."""
    n = min(len(draws_a), len(draws_b))
    d = np.asarray(draws_a[:n]) - np.asarray(draws_b[:n])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def auroc_draws(y, p, patients, n_boot=config.N_BOOTSTRAP, seed=config.SEED) -> np.ndarray:
    y = np.asarray(y).astype(int)
    out = []
    for rows in bootstrap_rows(np.asarray(patients), n_boot, seed):
        yb = y[rows]
        if yb.min() != yb.max():
            out.append(roc_auc_score(yb, p[rows]))
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibration(y, p, n_bins: int = config.CALIBRATION_BINS) -> dict:
    """Calibration intercept and slope, Brier score, ECE and the binned curve.

    slope:      fit logit P(y=1) = a + b * logit(p); b = 1 is ideal,
                b < 1 means predictions are too extreme.
    intercept:  calibration-in-the-large, fit with the slope fixed at 1;
                0 is ideal, < 0 means probabilities are too high overall.
    ECE:        average |observed AF rate - mean predicted| over equal-width
                bins, weighted by bin size.
    """
    y = np.asarray(y).astype(float)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    lp = logit(p)

    def nll(a, b):
        q = np.clip(expit(a + b * lp), EPS, 1 - EPS)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    intercept = minimize_scalar(lambda a: nll(a, 1.0), bounds=(-20, 20), method="bounded").x
    # slope (with its own intercept): 2-parameter logistic fit by bounded
    # quasi-Newton. Plain Newton steps can overshoot and run away when many
    # probabilities sit at 0 or 1, so the search is kept to a sane range.
    fit = minimize(lambda ab: nll(ab[0], ab[1]), x0=[0.0, 1.0], method="L-BFGS-B",
                   bounds=[(-20, 20), (0.01, 50)])
    b = fit.x[1]

    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    curve, ece = [], 0.0
    for k in range(n_bins):
        m = idx == k
        if m.any():
            obs, pred = y[m].mean(), p[m].mean()
            ece += m.mean() * abs(obs - pred)
            curve.append((float(pred), float(obs), int(m.sum())))
    return {"intercept": float(intercept), "slope": float(b),
            "brier": float(np.mean((p - y) ** 2)), "ece": float(ece), "curve": curve}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI for a proportion (e.g. sensitivity in a small subgroup)."""
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)
