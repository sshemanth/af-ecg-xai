"""Calibration plot shared by the evaluation step and scripts/plot_calibration.py."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.evaluation import calibration_bins  # noqa: E402

COLOURS = ["#1f6fb4", "#d9822b", "#3a9a5b", "#8e5ab5"]


def plot_calibration(models: dict, title: str, path: Path, n_bins: int = 10,
                     strategy: str = "uniform") -> None:
    """models: {label: (y, p, slope)}. Top: observed vs predicted AF rate per
    bin, with a 95% CI and the number of ECGs in the bin, so points that rest
    on a handful of ECGs are easy to spot. Bottom: how many ECGs get each
    predicted probability (log scale)."""
    fig, (ax, hx) = plt.subplots(2, 1, figsize=(5.6, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.2]})
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect")
    for i, (name, (y, p, slope)) in enumerate(models.items()):
        c = COLOURS[i % len(COLOURS)]
        b = calibration_bins(y, p, n_bins, strategy)
        lab = f"{name} (slope {slope:.2f})" if slope is not None else name
        ax.errorbar(b["mean_pred"], b["obs"], yerr=[b["obs"] - b["obs_lo"], b["obs_hi"] - b["obs"]],
                    marker="o", ms=4, lw=1.4, capsize=2, color=c, label=lab)
        for r in b.itertuples():
            right = r.mean_pred > 0.85              # keep labels inside the frame
            ax.annotate(str(r.n), (r.mean_pred, r.obs), textcoords="offset points",
                        xytext=(-5 if right else 4, -10 if i else 5),
                        ha="right" if right else "left", fontsize=6, color=c)
        hx.hist(p, bins=np.linspace(0, 1, 41), histtype="step", lw=1.3, color=c, log=True)
    ax.set_ylabel("Observed AF rate")
    kind = "equal-width" if strategy == "uniform" else "equal-count"
    ax.set_title(f"{title}\n{n_bins} {kind} bins; bars = 95% CI, number = ECGs in bin",
                 fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, loc="upper left")
    hx.set_xlabel("Predicted AF probability")
    hx.set_ylabel("ECGs (log)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
