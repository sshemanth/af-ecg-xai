"""Redraw the calibration plots from saved predictions (no model is run).

Each point shows the observed AF rate in a bin of predicted probability,
with a 95% CI and the number of ECGs in the bin, plus a histogram of
predicted probabilities. Also writes the bin table, so you can see how
many ECGs sit behind each point.

Run from the project folder, after 06_evaluate.py:
    python scripts/plot_calibration.py
    python scripts/plot_calibration.py --strategy quantile   # equal-count bins

Outputs: results/eval/calibration_<dataset>.png and
results/eval/calibration_bins_<dataset>.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.evaluation import calibration, calibration_bins  # noqa: E402
from src.plots import plot_calibration  # noqa: E402
from src.runs import arch_label, find_runs, primary_run  # noqa: E402

TITLES = {k: config.DATASET_NAMES[k] + (" test (internal)" if k == "ptbxl" else " (external)")
          for k in config.ALL_DATASETS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=config.CALIBRATION_BINS)
    ap.add_argument("--strategy", choices=["uniform", "quantile"], default="uniform")
    args = ap.parse_args()

    primaries = {a: primary_run(r) for a in config.ARCHS if (r := find_runs(a, verbose=False))}
    for key, title in TITLES.items():
        f = config.EVAL_DIR / f"predictions_{key}.csv"
        if not f.exists():
            print(f"[skip] {key}: {f.name} not found")
            continue
        pred = pd.read_csv(f)
        y = pred["label"].to_numpy()
        models = {}
        cols = [(arch_label(a), f"prob_{p['name']}") for a, p in primaries.items()]
        for name, col in cols + [("RR baseline", "prob_rr_baseline")]:
            if col not in pred:                      # network not evaluated on this set yet
                continue
            ok = pred[col].notna().to_numpy()
            models[name] = (y[ok], pred.loc[ok, col].to_numpy(), calibration(y[ok], pred.loc[ok, col])["slope"])
        out = config.EVAL_DIR / f"calibration_{key}.png"
        plot_calibration(models, title, out, args.bins, args.strategy)
        table = pd.concat([calibration_bins(yy, pp, args.bins, args.strategy).assign(model=n)
                           for n, (yy, pp, _) in models.items()])
        table.to_csv(config.EVAL_DIR / f"calibration_bins_{key}.csv", index=False)
        print(f"{key}: {out.relative_to(config.ROOT)}")
        print(table[["model", "n", "n_af", "mean_pred", "obs", "obs_lo", "obs_hi"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
