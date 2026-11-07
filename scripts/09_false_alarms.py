"""Step 9: which non-AF rhythms does the model mistake for AF on Chapman-Shaoxing?

Chapman headers list every diagnosis per ECG (SNOMED-CT codes), so the
false alarms (non-AF ECGs scored above the locked threshold) can be split
by rhythm. Two views:

  main rhythm   each non-AF ECG gets one rhythm group, the first that matches
                in the order of RHYTHM_GROUPS below (e.g. an ECG coded both
                sinus tachycardia and PACs counts as "premature atrial beats")
  every code    for each diagnosis code, how many non-AF ECGs carry it and how
                often each model raises a false alarm on them (an ECG can
                carry several codes, so these rows overlap)

The ResNet and the RR-only baseline are compared on the same ECGs. Rhythms
where atrial activity is hard to see (fast SVT, atrial tachycardia,
wandering pacemaker) are the ones to watch if the model relies on atrial
morphology; frequent premature beats are the ones to watch if it relies on
RR irregularity.

Only saved predictions and headers are read; no model is run, so this does
not touch the test data again.

Run from the project folder, after 06_evaluate.py:
    python scripts/09_false_alarms.py
    python scripts/09_false_alarms.py --model resnet_seed1   # another seed

Outputs: results/eval/false_alarms_chapman.csv and
results/09_false_alarms_report.md
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.evaluation import wilson_ci  # noqa: E402
from src.labels import _parse_header  # noqa: E402
from src.runs import find_resnet_runs, primary_run  # noqa: E402

CODES = pd.read_csv(Path(__file__).resolve().parents[1] / "src" / "snomed_codes.csv",
                    dtype={"code": str})
NAMES = dict(zip(CODES["code"], CODES["name"]))
ABBR = dict(zip(CODES["code"], CODES["abbr"]))

# first match wins; codes from the PhysioNet/CinC Challenge 2021 mapping
RHYTHM_GROUPS = [
    ("Paced", ["10370003", "251268003", "251266004"]),
    ("Supraventricular tachycardia (SVT, AT, AVNRT, AVRT, junctional)",
     ["426761007", "67198005", "713422000", "251166008", "233897008", "426648003"]),
    ("Wandering pacemaker / atrial rhythm",
     ["17366009", "195101003", "106068003", "233892002"]),
    ("Premature atrial beats", ["284470004", "63593006", "251173003", "251170000", "251168009"]),
    ("Premature ventricular beats", ["427172004", "17338001", "251182009"]),
    ("Sinus arrhythmia", ["427393009"]),
    ("Sinus tachycardia", ["427084000"]),
    ("Sinus bradycardia", ["426177001"]),
    ("Sinus rhythm", ["426783006"]),
]


def main_rhythm(dx: set) -> str:
    for name, codes in RHYTHM_GROUPS:
        if dx & set(codes):
            return name
    return "Other / no rhythm code"


def read_dx(labels: pd.DataFrame) -> pd.Series:
    """Diagnosis codes per record, from each ECG's .hea header."""
    out = {}
    for rid, path in zip(labels["record_id"], labels["path"]):
        hea = Path(str(path) + ".hea")
        if not hea.exists():                     # data folder moved: search for it
            hits = list(config.CHAPMAN_DIR.rglob(f"{rid}.hea"))
            hea = hits[0] if hits else None
        out[rid] = set(_parse_header(hea)["dx"]) if hea else set()
    return pd.Series(out, name="dx")


def rate(k: int, n: int) -> str:
    if n == 0:
        return "–"
    lo, hi = wilson_ci(k, n)
    return f"{k}/{n} = {k / n:.2f} ({lo:.2f}–{hi:.2f})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="primary",
                    help="ResNet run name (e.g. resnet_seed1); default: the primary model")
    ap.add_argument("--top", type=int, default=20, help="rows in the every-code table")
    args = ap.parse_args()

    runs = find_resnet_runs(verbose=False)
    if not runs:
        sys.exit("No trained ResNet runs found.")
    by_name = {r["name"]: r for r in runs}
    run = primary_run(runs) if args.model == "primary" else by_name.get(args.model)
    if run is None:
        sys.exit(f"{args.model} not found. Trained runs: {', '.join(by_name)}")

    pred_f = config.EVAL_DIR / "predictions_chapman.csv"
    if not pred_f.exists():
        sys.exit(f"{pred_f} not found. Run scripts/06_evaluate.py first.")
    pred = pd.read_csv(pred_f, dtype={"record_id": str})
    labels = pd.read_csv(config.PROCESSED_DIR / "chapman_labels.csv", dtype={"record_id": str})
    rr = pd.read_csv(config.PROCESSED_DIR / "chapman_rr.csv", dtype={"record_id": str})
    thr_base = json.loads((config.MODELS_DIR / "rr_baseline" / "threshold.json").read_text())["threshold"]

    df = pred[pred["label"] == 0].merge(labels[["record_id", "path"]], on="record_id", how="left")
    df = df.merge(rr[["record_id", "rr_cv", "hr_mean"]], on="record_id", how="left")
    print(f"Reading headers for {len(df)} non-AF ECGs...")
    df["dx"] = df["record_id"].map(read_dx(df))
    n_no_dx = int((df["dx"].apply(len) == 0).sum())
    df["rhythm"] = df["dx"].apply(main_rhythm)
    df["fa_resnet"] = df[f"prob_{run['name']}"] >= run["threshold"]
    df["fa_rr"] = df["prob_rr_baseline"] >= thr_base
    rr_ok = df["prob_rr_baseline"].notna()

    out = df.drop(columns=["path"]).copy()
    out["dx"] = out["dx"].apply(lambda s: " ".join(sorted(s)))
    out.to_csv(config.EVAL_DIR / "false_alarms_chapman.csv", index=False)

    n_fa = int(df["fa_resnet"].sum())
    L = ["# False alarms on Chapman-Shaoxing by rhythm\n",
         f"Model: **{run['name']}** (threshold {run['threshold']:.3f}); RR baseline threshold "
         f"{thr_base:.3f}. {len(df)} non-AF ECGs; ResNet false alarms: **{n_fa}** "
         f"({100 * n_fa / len(df):.1f}%), RR baseline: {int(df.loc[rr_ok, 'fa_rr'].sum())} "
         f"(of {int(rr_ok.sum())} it could score).\n"]
    if n_no_dx:
        L.append(f"{n_no_dx} ECGs had no readable diagnosis codes (counted as Other).\n")

    # main rhythm
    L += ["## By main rhythm\n",
          "Each ECG counted once (first matching group, in the order listed). "
          "False-alarm rate with 95% Wilson CI.\n",
          "| Main rhythm | Non-AF ECGs | Share of ResNet false alarms | ResNet false-alarm rate | "
          "RR baseline false-alarm rate | Median RR CV |",
          "|---|---|---|---|---|---|"]
    order = [g for g, _ in RHYTHM_GROUPS] + ["Other / no rhythm code"]
    for g in order:
        m = df["rhythm"] == g
        if not m.any():
            continue
        k = int(df.loc[m, "fa_resnet"].sum())
        mb = m & rr_ok
        L.append(f"| {g} | {int(m.sum())} | {100 * k / max(n_fa, 1):.1f}% | {rate(k, int(m.sum()))} | "
                 f"{rate(int(df.loc[mb, 'fa_rr'].sum()), int(mb.sum()))} | "
                 f"{df.loc[m, 'rr_cv'].median():.3f} |")
    L.append("")

    # every code
    rows = []
    for code in sorted(set().union(*df["dx"])):
        m = df["dx"].apply(lambda s, c=code: c in s)
        k = int(df.loc[m, "fa_resnet"].sum())
        mb = m & rr_ok
        rows.append({"code": code, "abbr": ABBR.get(code, "?"), "name": NAMES.get(code, "unknown code"),
                     "n": int(m.sum()), "fa": k, "fa_rr": int(df.loc[mb, "fa_rr"].sum()), "n_rr": int(mb.sum())})
    codes = pd.DataFrame(rows).sort_values(["fa", "n"], ascending=False)
    codes.to_csv(config.EVAL_DIR / "false_alarms_chapman_by_code.csv", index=False)
    L += [f"## Every diagnosis code (top {args.top} by number of ResNet false alarms)\n",
          "ECGs can carry several codes, so rows overlap. Code names from the "
          "PhysioNet/CinC Challenge 2021 mapping.\n",
          "| Code | Diagnosis | Non-AF ECGs | ResNet false alarms | ResNet rate | RR baseline rate |",
          "|---|---|---|---|---|---|"]
    for r in codes.head(args.top).itertuples():
        L.append(f"| {r.code} ({r.abbr}) | {r.name} | {r.n} | {r.fa} | {rate(r.fa, r.n)} | "
                 f"{rate(r.fa_rr, r.n_rr)} |")
    L.append("\nFull table: `results/eval/false_alarms_chapman_by_code.csv`\n")

    # timing of false alarms vs correct non-AF calls
    tn = df[~df["fa_resnet"]]
    fa = df[df["fa_resnet"]]
    L += ["## Beat timing: false alarms vs correct non-AF calls\n",
          "| ECGs | n | Median RR CV | Median heart rate |", "|---|---|---|---|",
          f"| ResNet false alarms | {len(fa)} | {fa['rr_cv'].median():.3f} | {fa['hr_mean'].median():.0f} |",
          f"| Correct non-AF | {len(tn)} | {tn['rr_cv'].median():.3f} | {tn['hr_mean'].median():.0f} |",
          f"\nBoth models wrong: {int((df['fa_resnet'] & df['fa_rr']).sum())}; "
          f"ResNet only: {int((df['fa_resnet'] & ~df['fa_rr']).sum())}; "
          f"RR baseline only: {int((~df['fa_resnet'] & df['fa_rr'] & rr_ok).sum())}.\n"]

    text = "\n".join(L)
    (config.RESULTS_DIR / "09_false_alarms_report.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
