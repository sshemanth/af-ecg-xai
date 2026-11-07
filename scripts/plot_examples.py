"""Plot a few ECGs with their detected R-peaks, for checking by eye.

Run from the project folder, after 03_rpeaks.py:
    python scripts/plot_examples.py                       # regular-rate AF (main rule)
    python scripts/plot_examples.py --group regular_strict
    python scripts/plot_examples.py --group af
    python scripts/plot_examples.py --group sinus
    python scripts/plot_examples.py --dataset chapman --n 8

Saves results/examples_<dataset>_<group>.png, showing lead II and V1
(V1 is where f-waves are usually easiest to see).

What to look for in regular-rate AF:
  - red dots sit on the R-peaks (detection is right)
  - no clear, repeating P-wave before each QRS
  - a wavy or flat baseline between beats, often clearest in V1
If many look like normal sinus rhythm with P-waves, the AF labels or the
regular-rate rule for those ECGs need a closer look.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402

GROUPS = {
    "regular": lambda r: r["regular_af"].astype(bool),
    "regular_strict": lambda r: r["regular_af_strict"].astype(bool),
    "af": lambda r: (r["label"] == 1) & ~r["regular_af"].astype(bool),
    "sinus": lambda r: (r["label"] == 0) & r["sinus"].fillna(False).astype(bool),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=config.ALL_DATASETS, default="ptbxl")
    ap.add_argument("--group", choices=list(GROUPS), default="regular")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    rr = pd.read_csv(config.PROCESSED_DIR / f"{args.dataset}_rr.csv",
                     dtype={"record_id": str})
    pick = rr[GROUPS[args.group](rr)]
    if pick.empty:
        sys.exit(f"No ECGs in group '{args.group}'.")
    pick = pick.sample(min(args.n, len(pick)), random_state=args.seed)

    sig = np.load(config.PROCESSED_DIR / f"{args.dataset}_signals.npy", mmap_mode="r")
    t = np.arange(config.N_SAMPLES) / config.TARGET_FS
    leads = [config.LEADS.index("II"), config.LEADS.index("V1")]

    fig, axes = plt.subplots(len(pick), 1, figsize=(14, 2.6 * len(pick)), squeeze=False)
    for ax, (_, r) in zip(axes[:, 0], pick.iterrows()):
        x = np.asarray(sig[int(r["row"])], dtype=np.float32)
        peaks = np.array(str(r["rpeaks"]).split(), dtype=int) if isinstance(r["rpeaks"], str) else []
        # gap between the two leads, scaled to this ECG so they never overlap
        offset = float(np.ptp(x[leads[0]]) / 2 + np.ptp(x[leads[1]]) / 2) + 0.3
        for k, li in enumerate(leads):
            y = x[li] - k * offset
            ax.plot(t, y, lw=0.7, color="black")
            ax.text(-0.25, y[:250].mean(), config.LEADS[li], ha="right", va="center")
        if len(peaks):
            ax.plot(t[peaks], x[leads[0]][peaks], "o", ms=4, color="red")
        ax.set_title(f"{args.dataset} {r['record_id']}  |  split {r['split']}  |  "
                     f"HR {r['hr_mean']:.0f} bpm  |  rr_cv {r['rr_cv']:.3f}  |  "
                     f"label {'AF' if r['label'] == 1 else 'non-AF'}", fontsize=9, loc="left")
        ax.set_xlim(0, config.SIGNAL_SECONDS)
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.3)
    axes[-1, 0].set_xlabel("seconds")
    fig.tight_layout()

    out = config.RESULTS_DIR / f"examples_{args.dataset}_{args.group}.png"
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
