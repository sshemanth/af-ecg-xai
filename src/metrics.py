"""Metrics and the decision threshold, shared by every model so they are
all judged the same way."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src import config


def choose_threshold(y: np.ndarray, p: np.ndarray, rule: str = config.THRESHOLD_RULE) -> float:
    """Pick the decision threshold on VALIDATION data only.

    youden: the threshold that maximises sensitivity + specificity - 1.
    """
    if rule != "youden":
        raise ValueError(f"Unknown threshold rule: {rule}")
    fpr, tpr, thr = roc_curve(y, p)
    best = np.argmax(tpr - fpr)
    # roc_curve's first threshold is +inf (predict nothing positive); skip it
    return float(thr[best]) if np.isfinite(thr[best]) else float(thr[1])


def summarise(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Threshold-free and threshold-based metrics for one set of predictions."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def div(a, b):
        return a / b if b else float("nan")

    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    ppv, npv = div(tp, tp + fp), div(tn, tn + fn)
    return {
        "n": int(len(y)), "n_af": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "auprc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "threshold": float(threshold),
        "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
        "f1": div(2 * ppv * sens, ppv + sens) if ppv == ppv and sens == sens else float("nan"),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
