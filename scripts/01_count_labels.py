"""Step 1: build label tables and count AF, flutter and paced cases.

Run from the project folder:
    python scripts/01_count_labels.py

Outputs:
    data/processed/ptbxl_labels.csv
    data/processed/chapman_labels.csv
    results/01_label_counts.md     (summary report)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.labels import load_chapman, load_ptbxl  # noqa: E402


def counts(df: pd.DataFrame) -> dict:
    af, afl, paced = df["af"], df["aflutter"], df["paced"]
    return {
        "Recordings": len(df),
        "Patients": df["patient_id"].nunique(),
        "AF": int(af.sum()),
        "Atrial flutter": int(afl.sum()),
        "AF + flutter (both)": int((af & afl).sum()),
        "Flutter only (no AF)": int((afl & ~af).sum()),
        "Paced": int(paced.sum()),
        "AF + paced": int((af & paced).sum()),
        "Excluded by flutter policy": int((df["label"] == -1).sum()),
        "Final AF (label=1)": int((df["label"] == 1).sum()),
        "Final non-AF (label=0)": int((df["label"] == 0).sum()),
    }


def check_patient_leakage(df: pd.DataFrame) -> int:
    """Number of patients that appear in more than one split (should be 0)."""
    splits_per_patient = df.groupby("patient_id")["split"].nunique()
    return int((splits_per_patient > 1).sum())


def main() -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tables, lines = {}, []
    lines.append("# Label counts\n")
    lines.append(f"Flutter policy: **{config.FLUTTER_POLICY}**\n")

    for name, loader in [("PTB-XL", load_ptbxl), ("Chapman-Shaoxing", load_chapman)]:
        try:
            df = loader()
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
            lines.append(f"## {name}\nNot found — {e}\n")
            continue
        tables[name] = df
        out = config.PROCESSED_DIR / f"{df['dataset'].iloc[0]}_labels.csv"
        df.to_csv(out, index=False)
        print(f"Saved {out}")

        lines.append(f"## {name}\n")
        lines.append("| Count | Value |\n|---|---|")
        for k, v in counts(df).items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

        # Sanity check: every Chapman header should carry diagnosis codes.
        # If many don't, the header format wasn't read properly.
        if "n_dx" in df:
            no_dx = int((df["n_dx"] == 0).sum())
            lines.append(f"Recordings with no diagnosis codes read: **{no_dx}**")
            if no_dx > 0.05 * len(df):
                lines.append("> **Warning:** many headers had no diagnosis codes. "
                             "The labels above are not reliable. Check a header file "
                             "with: `head -20 $(find data/raw/chapman -name '*.hea' | head -1)`")
            lines.append("")

        # Recording format (sampling rate, length, leads)
        lines.append("### Recording format\n")
        for col, label in [("fs", "Sampling rate (Hz)"),
                           ("n_samples", "Samples per recording"),
                           ("n_leads", "Leads")]:
            if col in df:
                vc = df[col].value_counts(dropna=False).head(5)
                shown = ", ".join(f"{k}: {v}" for k, v in vc.items())
                lines.append(f"- {label}: {shown}")
        lines.append("")

        if name == "PTB-XL":
            lines.append("### By split\n")
            lines.append(f"Patient-level, stratified by AF and paced, seed {config.SEED}\n")
            lines.append("| Split | Recordings | Patients | AF | Non-AF | AF % | Excluded | AF + paced |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for split in ["train", "val", "test"]:
                s = df[df["split"] == split]
                n_af, n_non = (s["label"] == 1).sum(), (s["label"] == 0).sum()
                pct = 100 * n_af / max(n_af + n_non, 1)
                lines.append(
                    f"| {split} | {len(s)} | {s['patient_id'].nunique()} | {n_af} | "
                    f"{n_non} | {pct:.1f} | {(s['label'] == -1).sum()} | "
                    f"{(s['af'] & s['paced']).sum()} |")
            leaks = check_patient_leakage(df)
            lines.append(f"\nPatients in more than one split: **{leaks}** (must be 0)\n")

    # Is the paced-AF subgroup big enough to analyse?
    if tables:
        lines.append("## Paced AF subgroup size (AF + paced)\n")
        for name, df in tables.items():
            n = int((df["af"] & df["paced"]).sum())
            verdict = ("enough for a main analysis" if n >= 50 else
                       "borderline — report as exploratory" if n >= 30 else
                       "too few — exploratory only, or drop")
            lines.append(f"- {name}: {n} recordings → {verdict}")
        lines.append("")

    summary = "\n".join(lines)
    (config.RESULTS_DIR / "01_label_counts.md").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
