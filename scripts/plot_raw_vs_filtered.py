"""Plot one ECG before and after filtering, to see where an artefact comes from.

Run from the project folder, after 02_preprocess.py:
    python scripts/plot_raw_vs_filtered.py --record 17360
    python scripts/plot_raw_vs_filtered.py --record JS00001 --dataset chapman

Saves results/raw_vs_filtered_<dataset>_<record>.png with leads II and V1:
the original recording (grey, re-centred so it lines up) and what we saved
after filtering (black). Zoomed on the first 2 seconds, plus the full 10 s.

If the swoop at the start is in BOTH, it's in the recording itself.
If it's only in black, the filter made it.
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
from src.signals import _read_raw, _reorder_leads  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="record_id, e.g. 17360 or JS00001")
    ap.add_argument("--dataset", choices=["ptbxl", "chapman"], default="ptbxl")
    args = ap.parse_args()

    index = pd.read_csv(config.PROCESSED_DIR / f"{args.dataset}_index.csv",
                        dtype={"record_id": str})
    hit = index[index["record_id"] == str(args.record)]
    if hit.empty:
        sys.exit(f"Record {args.record} not found in {args.dataset}_index.csv")
    r = hit.iloc[0]

    raw, fs, names = _read_raw(r["path"])
    raw = _reorder_leads(raw, names)[: config.N_SAMPLES].T          # [leads, samples]
    saved = np.load(config.PROCESSED_DIR / f"{args.dataset}_signals.npy",
                    mmap_mode="r")[int(r["row"])].astype(np.float32)

    t = np.arange(config.N_SAMPLES) / config.TARGET_FS
    leads = ["II", "V1"]
    fig, axes = plt.subplots(len(leads), 2, figsize=(15, 5),
                             gridspec_kw={"width_ratios": [1, 3]})
    for row_i, lead in enumerate(leads):
        li = config.LEADS.index(lead)
        # re-centre the raw lead on its median so both traces line up
        raw_c = raw[li] - np.median(raw[li])
        for col, xmax in enumerate([2, config.SIGNAL_SECONDS]):
            ax = axes[row_i, col]
            ax.plot(t, raw_c, color="0.6", lw=1.2, label="original (re-centred)")
            ax.plot(t, saved[li], color="black", lw=0.7, label="saved (filtered)")
            ax.set_xlim(0, xmax)
            ax.set_ylabel(f"{lead} (mV)" if col == 0 else "")
            ax.grid(alpha=0.3)
            if row_i == 0:
                ax.set_title("first 2 s" if col == 0 else "full 10 s", fontsize=9)
    axes[0, 1].legend(loc="upper right", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("seconds")
    fig.suptitle(f"{args.dataset} {r['record_id']}  |  original vs filtered", fontsize=10)
    fig.tight_layout()

    out = config.RESULTS_DIR / f"raw_vs_filtered_{args.dataset}_{r['record_id']}.png"
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
