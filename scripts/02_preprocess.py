"""Step 2: load, check, filter and save all ECG signals.

Run from the project folder, after 01_count_labels.py:
    python scripts/02_preprocess.py                    # every dataset with a label table
    python scripts/02_preprocess.py --dataset georgia  # one dataset (own report file)
    python scripts/02_preprocess.py --limit 200        # quick test run

Run an external set only once its download has finished; files still
missing would otherwise be counted as "unreadable".

Outputs (data/processed/):
    <dataset>_signals.npy   float16 [n, 12, 5000], filtered, NOT normalised
    <dataset>_index.csv     one row per kept ECG: row number, label, split...
    norm_stats.json         per-lead mean/std from PTB-XL TRAIN only
and results/02_preprocess_report.md
"""
import argparse
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src import config  # noqa: E402
from src.signals import Excluded, load_and_clean  # noqa: E402


def _work(path: str):
    """Runs in a worker process.
    Returns (signal or None, (edge, middle) deviation in mV, reason or None)."""
    try:
        sig, edge, middle = load_and_clean(path)
        return sig, (edge, middle), None
    except Excluded as e:
        return None, (np.nan, np.nan), str(e)
    except Exception:            # anything unexpected still gets counted
        return None, (np.nan, np.nan), "corrupted waveform"


def label_exclusions(df: pd.DataFrame) -> pd.Series:
    """Reason a row is excluded before its signal is even read (or None)."""
    reason = pd.Series(None, index=df.index, dtype=object)
    reason[df["label"] == -1] = "flutter without AF"
    if "n_dx" in df:
        reason[df["n_dx"] == 0] = "ambiguous label (no diagnosis codes)"
    return reason


def process(name: str, workers: int, limit: int | None) -> dict:
    labels_csv = config.PROCESSED_DIR / f"{name}_labels.csv"
    if not labels_csv.exists():
        sys.exit(f"{labels_csv} not found. Run scripts/01_count_labels.py first.")
    df = pd.read_csv(labels_csv, dtype={"record_id": str, "patient_id": str})
    if limit and limit < len(df):
        # keep the AF / non-AF mix roughly the same in the test sample
        df = df.groupby("label").sample(frac=limit / len(df), random_state=config.SEED)
    df = df.reset_index(drop=True)

    reasons = label_exclusions(df)
    todo = df.index[reasons.isna()].tolist()

    out_path = config.PROCESSED_DIR / f"{name}_signals.npy"
    signals = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=config.SIGNAL_DTYPE,
        shape=(len(todo), len(config.LEADS), config.N_SAMPLES))

    kept_rows, kept_df_idx, edges = [], [], []
    paths = df.loc[todo, "path"].tolist()
    with Pool(workers) as pool:
        results = pool.imap(_work, paths, chunksize=32)
        for slot, (df_i, (sig, edge, why)) in enumerate(
                tqdm(zip(todo, results), total=len(todo), desc=name)):
            if sig is None:
                reasons[df_i] = why
            else:
                signals[slot] = sig
                kept_rows.append(slot)
                kept_df_idx.append(df_i)
                edges.append(edge)
    signals.flush()

    index = df.loc[kept_df_idx].copy()
    index.insert(0, "row", kept_rows)        # row in <name>_signals.npy
    index["edge_dev_mv"] = [e for e, _ in edges]    # baseline offset at the edges
    index["mid_dev_mv"] = [m for _, m in edges]     # same measure, middle of the ECG
    index.to_csv(config.PROCESSED_DIR / f"{name}_index.csv", index=False)

    return {"df": df, "index": index, "reasons": reasons, "signals": out_path}


def norm_stats(info: dict) -> dict:
    """Per-lead mean and std over PTB-XL TRAINING ECGs only."""
    sig = np.load(info["signals"], mmap_mode="r")
    rows = info["index"].loc[info["index"]["split"] == "train", "row"].to_numpy()
    n_leads = sig.shape[1]
    total = np.zeros(n_leads)
    total_sq = np.zeros(n_leads)
    count = 0
    for start in range(0, len(rows), 500):          # chunks, to spare RAM
        chunk = sig[np.sort(rows[start:start + 500])].astype(np.float64)
        total += chunk.sum(axis=(0, 2))
        total_sq += (chunk ** 2).sum(axis=(0, 2))
        count += chunk.shape[0] * chunk.shape[2]
    mean = total / count
    std = np.sqrt(total_sq / count - mean ** 2)
    return {"leads": config.LEADS, "mean": mean.tolist(), "std": std.tolist(),
            "computed_on": "ptbxl train split only", "n_ecgs": int(len(rows))}


def report_section(title: str, info: dict) -> list:
    df, index, reasons = info["df"], info["index"], info["reasons"]
    lines = [f"## {title}\n",
             f"- Candidates: {len(df)}",
             f"- Kept: **{len(index)}**",
             f"- Excluded: {int(reasons.notna().sum())}\n",
             "| Exclusion reason | Count |", "|---|---|"]
    for why, n in Counter(reasons.dropna()).most_common():
        lines.append(f"| {why} | {n} |")
    lines += ["", "| Split | Kept ECGs | AF | Non-AF | AF % |", "|---|---|---|---|---|"]
    for split, s in index.groupby("split", sort=False):
        n_af = int((s["label"] == 1).sum())
        lines.append(f"| {split} | {len(s)} | {n_af} | {len(s) - n_af} | "
                     f"{100 * n_af / max(len(s), 1):.1f} |")
    # Amplitude sanity check: a file read with the wrong format shows up as
    # huge values. Normal 12-lead ECG peaks are a few mV.
    sig = np.load(info["signals"], mmap_mode="r")
    rows = index["row"].to_numpy()
    sample = np.sort(np.random.default_rng(config.SEED).choice(
        rows, size=min(500, len(rows)), replace=False)) if len(rows) else rows
    if len(sample):
        peaks = np.abs(sig[sample].astype(np.float32)).max(axis=(1, 2))
        med, p99 = np.median(peaks), np.percentile(peaks, 99)
        lines.append(f"\nPeak amplitude per ECG (sample of {len(sample)}): "
                     f"median {med:.2f} mV, 99th percentile {p99:.2f} mV")
        if med > 10:
            lines.append("> **Warning:** amplitudes look far too large. The files "
                         "may have been read in the wrong format; check the raw files.")

    # Edge check: leftover baseline offset at the start/end after filtering.
    # Fine if it looks the same in AF and non-AF. If it differs, the model
    # could use it as a shortcut, so the report says so.
    if "edge_dev_mv" in index:
        lines += ["", f"Edge check (median baseline in first/last "
                  f"{config.EDGE_DEV_WINDOW_S} s after filtering, worst lead):", "",
                  f"| Group | Where | Median (mV) | 95th percentile (mV) | Share > {config.EDGE_DEV_FLAG_MV} mV |",
                  "|---|---|---|---|---|"]
        shares = {}
        for lab, g in [("AF", index[index["label"] == 1]), ("Non-AF", index[index["label"] == 0])]:
            for where, col in [("edges", "edge_dev_mv"), ("middle", "mid_dev_mv")]:
                e = g[col].dropna()
                if len(e):
                    shares[(lab, where)] = 100 * (e > config.EDGE_DEV_FLAG_MV).mean()
                    lines.append(f"| {lab} | {where} | {e.median():.3f} | "
                                 f"{np.percentile(e, 95):.3f} | {shares[(lab, where)]:.1f}% |")
        if len(shares) == 4:
            gap_edge = shares[("AF", "edges")] - shares[("Non-AF", "edges")]
            gap_mid = shares[("AF", "middle")] - shares[("Non-AF", "middle")]
            extra = gap_edge - gap_mid
            lines.append(f"\nAF minus non-AF: {gap_edge:+.1f} points at the edges, "
                         f"{gap_mid:+.1f} in the middle → edge-only part {extra:+.1f} points.")
            if abs(extra) > 5:
                lines.append("> **Note:** the AF/non-AF gap is bigger at the edges than in "
                             "the middle, so it's partly an edge artefact; review edge handling.")
            else:
                lines.append("The gap is about the same in the middle, so it reflects how "
                             "the rhythms look, not an edge artefact.")

    unreadable = int((reasons == "unreadable file").sum())
    if unreadable > 0.05 * len(df):
        lines.append("\n> **Warning:** many files were unreadable. If the "
                     "download isn't finished, finish it and run this again.")
    return lines + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=config.ALL_DATASETS + ["all"], default="all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None,
                    help="process only about this many ECGs (for a quick test)")
    args = ap.parse_args()

    names = config.ALL_DATASETS if args.dataset == "all" else [args.dataset]
    if args.dataset == "all":                       # skip datasets not downloaded yet
        names = [n for n in names if (config.PROCESSED_DIR / f"{n}_labels.csv").exists()]
    lines = ["# Preprocessing report\n",
             f"Common rate {config.TARGET_FS} Hz, {config.SIGNAL_SECONDS} s; "
             f"band-pass {config.FILTER_LOW_HZ}-{config.FILTER_HIGH_HZ} Hz "
             f"(Butterworth order {config.FILTER_ORDER}, zero-phase); "
             f"flat lead if std < {config.FLAT_LEAD_STD_MV} mV.\n"]
    if args.limit:
        lines.append(f"**Test run: limited to about {args.limit} ECGs per dataset.**\n")

    for name in names:
        info = process(name, args.workers, args.limit)
        lines += report_section(config.DATASET_NAMES[name], info)
        if name == "ptbxl":
            stats = norm_stats(info)
            path = config.PROCESSED_DIR / "norm_stats.json"
            path.write_text(json.dumps(stats, indent=2))
            lines += ["### Normalisation (from PTB-XL train only)\n",
                      "| Lead | Mean (mV) | Std (mV) |", "|---|---|---|"]
            for lead, m, s in zip(stats["leads"], stats["mean"], stats["std"]):
                lines.append(f"| {lead} | {m:.4f} | {s:.4f} |")
            lines.append("")

    report = "\n".join(lines)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # one dataset on its own gets its own report, so earlier reports are kept
    suffix = "" if args.dataset == "all" else f"_{args.dataset}"
    (config.RESULTS_DIR / f"02_preprocess_report{suffix}.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
