"""Step 3: R-peaks, RR features and the regular-rate AF subgroup.

R-peaks are used only for RR variables and for the attribution / occlusion
windows. Regularity is defined by an RR variability threshold computed
from the PTB-XL training split (see src/config.py).

Run from the project folder, after 02_preprocess.py:
    python scripts/03_rpeaks.py                  # every dataset already preprocessed
    python scripts/03_rpeaks.py --dataset ptbxl

Outputs:
    data/processed/<dataset>_rr.csv    R-peaks + RR features per ECG
    data/processed/rr_threshold.json   the regular-rate cut-off (from PTB-XL train)
    results/03_rpeaks_report.md
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src import config  # noqa: E402
from src.rr import detect_rpeaks, rr_features  # noqa: E402

_SIG = None  # each worker opens the signal file once


def _init(path: str) -> None:
    global _SIG
    _SIG = np.load(path, mmap_mode="r")


def _work(row: int) -> dict:
    peaks, lead = detect_rpeaks(np.asarray(_SIG[row]))
    feats = rr_features(peaks)
    feats["rpeak_lead"] = lead
    feats["rpeaks"] = " ".join(map(str, peaks))     # sample indices, space-separated
    return feats


def run(name: str, workers: int) -> pd.DataFrame:
    index_csv = config.PROCESSED_DIR / f"{name}_index.csv"
    sig_path = config.PROCESSED_DIR / f"{name}_signals.npy"
    if not index_csv.exists():
        sys.exit(f"{index_csv} not found. Run scripts/02_preprocess.py first.")
    index = pd.read_csv(index_csv, dtype={"record_id": str, "patient_id": str})

    # Check the label table first, so we don't run the slow part for nothing
    labels = pd.read_csv(config.PROCESSED_DIR / f"{name}_labels.csv",
                         dtype={"record_id": str})
    if "sinus" not in labels:
        sys.exit("Label table has no 'sinus' column. Run scripts/01_count_labels.py again.")

    with Pool(workers, initializer=_init, initargs=(str(sig_path),)) as pool:
        feats = list(tqdm(pool.imap(_work, index["row"].tolist(), chunksize=64),
                          total=len(index), desc=name))
    keep = ["row", "record_id", "patient_id", "dataset", "label", "split", "paced"]
    out = pd.concat([index[keep].reset_index(drop=True), pd.DataFrame(feats)], axis=1)

    # Sinus-rhythm flag comes from the label table, so preprocessing
    # doesn't need re-running when label columns are added.
    out = out.merge(labels[["record_id", "sinus"]], on="record_id", how="left")
    return out


def threshold_from_ptbxl(rr: pd.DataFrame) -> dict:
    ref_mask = ((rr["split"] == "train") & (rr["label"] == 0)
                & rr["sinus"].astype(bool) & ~rr["paced"].astype(bool))
    ref = rr.loc[ref_mask, config.REGULAR_RR_METRIC].dropna()
    cut = float(np.percentile(ref, config.REGULAR_RR_PERCENTILE))
    strict = float(np.percentile(ref, config.REGULAR_RR_STRICT_PERCENTILE))
    return {"metric": config.REGULAR_RR_METRIC,
            "percentile": config.REGULAR_RR_PERCENTILE,
            "threshold": cut,
            "strict_percentile": config.REGULAR_RR_STRICT_PERCENTILE,
            "strict_threshold": strict,
            "reference": config.REGULAR_RR_REFERENCE,
            "n_reference": int(len(ref))}


def report(title: str, rr: pd.DataFrame, thr: dict) -> list:
    ok = rr["rr_cv"].notna()
    af, non = rr["label"] == 1, rr["label"] == 0
    lines = [f"## {title}\n",
             f"- ECGs: {len(rr)}",
             f"- R-peaks found (≥{config.MIN_BEATS} plausible beats): "
             f"**{ok.sum()}** ({100 * ok.mean():.1f}%)",
             f"- Lead used: " + ", ".join(f"{k or 'none'} {v}" for k, v in
                                           rr["rpeak_lead"].value_counts().items()),
             "",
             "| Group | Median HR (bpm) | Median rr_cv | Median nrmssd | Median pNN50 |",
             "|---|---|---|---|---|"]
    sinus = non & rr["sinus"].fillna(False).astype(bool) & ~rr["paced"].astype(bool)
    for lab, mask in [("AF", af), ("Non-AF", non), ("Sinus rhythm (not paced)", sinus)]:
        g = rr[mask & ok]
        lines.append(f"| {lab} | {g['hr_mean'].median():.0f} | {g['rr_cv'].median():.3f} | "
                     f"{g['nrmssd'].median():.3f} | {g['pnn50'].median():.2f} |")
    lines.append("\n(AF should show clearly higher rr_cv, nrmssd and pNN50 than non-AF.)\n")

    lines += [f"### AF with regular ventricular response "
              f"({thr['metric']} ≤ {thr['threshold']:.4f})\n",
              f"Stricter sensitivity rule: {thr['metric']} ≤ {thr['strict_threshold']:.4f} "
              f"({thr['strict_percentile']}th percentile).\n",
              "| Split | AF ECGs | Regular-rate AF (main) | Regular-rate AF (strict) | Paced AF |",
              "|---|---|---|---|---|"]
    for split, s in rr.groupby("split", sort=False):
        a = s["label"] == 1
        lines.append(f"| {split} | {int(a.sum())} | {int(s['regular_af'].sum())} | "
                     f"{int(s['regular_af_strict'].sum())} | "
                     f"{int((a & s['paced'].astype(bool)).sum())} |")
    n = int(rr["regular_af"].sum())
    verdict = ("enough for a main analysis" if n >= 50 else
               "borderline — report as exploratory" if n >= 30 else
               "too few — exploratory only")
    n_strict = int(rr["regular_af_strict"].sum())
    lines += ["", f"Total regular-rate AF: **{n}** (main) → {verdict}; "
              f"**{n_strict}** (strict)", ""]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ptbxl", "chapman", "all"], default="all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    names = ["ptbxl", "chapman"] if args.dataset == "all" else [args.dataset]
    names = [n for n in names if (config.PROCESSED_DIR / f"{n}_index.csv").exists()
             or args.dataset != "all"]

    thr_path = config.PROCESSED_DIR / "rr_threshold.json"
    results, thr = {}, None
    for name in names:
        results[name] = run(name, args.workers)
        if name == "ptbxl":
            thr = threshold_from_ptbxl(results[name])
            thr_path.write_text(json.dumps(thr, indent=2))

    if thr is None:
        if not thr_path.exists():
            sys.exit("Run PTB-XL first: the regular-rate cut-off comes from its training split.")
        thr = json.loads(thr_path.read_text())
        if "strict_threshold" not in thr:
            sys.exit("rr_threshold.json is from an older version. Run with --dataset ptbxl first.")

    lines = ["# R-peaks and RR features\n",
             f"Detector: NeuroKit2 `{config.RPEAK_METHOD}`, leads tried "
             f"{', '.join(config.RPEAK_LEADS)}; plausible HR {config.HR_RANGE[0]}–"
             f"{config.HR_RANGE[1]} bpm.\n",
             f"Regular-rate rule (fixed before training): AF with {thr['metric']} at or "
             f"below the {thr['percentile']}th percentile in {thr['reference']} "
             f"= **{thr['threshold']:.4f}** (n = {thr['n_reference']}).\n"]
    for name, rr in results.items():
        rr["regular_af"] = (rr["label"] == 1) & (rr[thr["metric"]] <= thr["threshold"])
        rr["regular_af_strict"] = ((rr["label"] == 1)
                                   & (rr[thr["metric"]] <= thr["strict_threshold"]))
        rr.to_csv(config.PROCESSED_DIR / f"{name}_rr.csv", index=False)
        lines += report({"ptbxl": "PTB-XL", "chapman": "Chapman-Shaoxing"}[name], rr, thr)

    text = "\n".join(lines)
    (config.RESULTS_DIR / "03_rpeaks_report.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
