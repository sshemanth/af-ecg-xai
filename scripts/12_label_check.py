"""Step 12: AF labels that conflict with a sinus-rhythm label on the same ECG.

Each ECG can carry several diagnosis codes. An ECG labelled AF that is
also labelled sinus rhythm (or sinus brady-/tachycardia or sinus
arrhythmia) most likely had AF at another time, not in this 10 s strip:
AF and sinus rhythm cannot both be the rhythm of one short recording.
Those AF labels are doubtful.

This check uses the labels only, never a model's output, so the rule could
not have been tuned on the results. It reports:

  - how many AF ECGs carry a sinus-type code, overall and within the
    low-RR-variability and near-regular AF groups
  - how often each model detects AF in conflicting vs clean AF labels
  - performance with the conflicting AF ECGs left out, as a sensitivity
    analysis next to the main result (which keeps every label as given)

Datasets: PTB-XL test (SCP codes SR, SBRAD, STACH, SARRH), Chapman-Shaoxing
and Georgia (SNOMED-CT codes). Only saved predictions and labels are read.

Run from the project folder, after 06_evaluate.py:
    python scripts/12_label_check.py

Outputs: results/eval/label_conflicts_<dataset>.csv and
results/12_label_check_report.md
"""
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.dx import SINUS_CODES, rate, read_dx  # noqa: E402
from src.evaluation import bootstrap_ci  # noqa: E402
from src.metrics import summarise  # noqa: E402
from src.runs import arch_label, find_runs, primary_run  # noqa: E402

PTBXL_SINUS = {"SR": "sinus rhythm", "SBRAD": "sinus bradycardia",
               "STACH": "sinus tachycardia", "SARRH": "sinus arrhythmia"}
N_BOOT = 1000


def sinus_codes(key: str, labels: pd.DataFrame) -> pd.Series:
    """Sinus-type rhythm names found on each record."""
    if key == "ptbxl":
        db = pd.read_csv(config.PTBXL_DIR / "ptbxl_database.csv", usecols=["ecg_id", "scp_codes"])
        codes = dict(zip(db["ecg_id"].astype(str), db["scp_codes"].apply(ast.literal_eval)))
        return pd.Series({rid: sorted(PTBXL_SINUS[c] for c in codes.get(rid, {}) if c in PTBXL_SINUS)
                          for rid in labels["record_id"]})
    dx = read_dx(labels, key)
    return dx.apply(lambda s: sorted(SINUS_CODES[c] for c in s if c in SINUS_CODES))


def fmt(m: dict, ci: dict, k: str) -> str:
    return f"{m[k]:.3f} ({ci[k][0]:.3f}–{ci[k][1]:.3f})"


def main() -> None:
    primaries = {a: primary_run(r) for a in config.ARCHS if (r := find_runs(a, verbose=False))}
    if not primaries:
        sys.exit("No trained runs found.")
    thr_base = json.loads((config.MODELS_DIR / "rr_baseline" / "threshold.json").read_text())["threshold"]

    L = ["# AF labels that conflict with a sinus-rhythm label\n",
         "An ECG labelled both AF and a sinus-type rhythm most likely had AF at another time, "
         "not in this 10 s strip. Found from the labels alone, never from model output. The "
         "main results keep every label as given; leaving these out is a sensitivity analysis.\n"]
    for key in config.ALL_DATASETS:
        pred_f = config.EVAL_DIR / f"predictions_{key}.csv"
        lab_f = config.PROCESSED_DIR / f"{key}_labels.csv"
        if not pred_f.exists() or not lab_f.exists():
            print(f"[skip] {key}: run steps 01-06 first")
            continue
        if key == "ptbxl" and not (config.PTBXL_DIR / "ptbxl_database.csv").exists():
            print("[skip] ptbxl: ptbxl_database.csv not found")
            continue
        pred = pd.read_csv(pred_f, dtype={"record_id": str})
        labels = pd.read_csv(lab_f, dtype={"record_id": str})
        af = pred[pred["label"] == 1].merge(labels[["record_id", "path"]], on="record_id", how="left")
        print(f"Reading codes for {len(af)} AF ECGs in {key}...")
        af["sinus_codes"] = af["record_id"].map(sinus_codes(key, af))
        af["conflict"] = af["sinus_codes"].apply(len) > 0
        out = af.drop(columns=["path"]).copy()
        out["sinus_codes"] = out["sinus_codes"].apply(" + ".join)
        out.to_csv(config.EVAL_DIR / f"label_conflicts_{key}.csv", index=False)

        n_c = int(af["conflict"].sum())
        name = config.DATASET_NAMES[key] + (" test" if key == "ptbxl" else "")
        L += [f"## {name}\n",
              f"AF ECGs: {len(af)}; also labelled a sinus-type rhythm: **{n_c}** "
              f"({100 * n_c / max(len(af), 1):.1f}%).\n"]
        if n_c:
            counts = pd.Series([c for cs in af["sinus_codes"] for c in cs]).value_counts()
            L.append("Codes found: " + ", ".join(f"{c} {n}" for c, n in counts.items()) + "\n")

        # by RR-regularity group
        L += ["| AF group | AF ECGs | With a sinus-type code |", "|---|---|---|"]
        for gname, m in [("All AF", np.ones(len(af), bool)),
                         ("Low RR variability (main rule)", af["regular_af"].fillna(False).astype(bool)),
                         ("Near-regular (strict rule)", af["regular_af_strict"].fillna(False).astype(bool))]:
            m = np.asarray(m)
            L.append(f"| {gname} | {int(m.sum())} | {rate(int(af.loc[m, 'conflict'].sum()), int(m.sum()))} |")
        L.append("")
        if n_c == 0:
            continue

        # detection: conflicting vs clean AF labels
        models = [(f"{arch_label(a)} ({p['name']})", f"prob_{p['name']}", p["threshold"])
                  for a, p in primaries.items() if f"prob_{p['name']}" in pred]
        models.append(("RR baseline", "prob_rr_baseline", thr_base))
        L += ["### AF detected: conflicting vs clean labels\n",
              "| Model | AF also labelled sinus | AF only |", "|---|---|---|"]
        for mname, col, thr in models:
            cells = []
            for m in [af["conflict"], ~af["conflict"]]:
                g = af[m & af[col].notna()]
                cells.append(rate(int((g[col] >= thr).sum()), len(g)))
            L.append(f"| {mname} | {cells[0]} | {cells[1]} |")
        L.append("\nIf models rarely 'detect' AF in ECGs also labelled sinus rhythm, those "
                 "labels, not the models, explain part of the missed AF.\n")

        # sensitivity analysis: leave the conflicting AF ECGs out
        drop = set(af.loc[af["conflict"], "record_id"])
        keep = ~pred["record_id"].isin(drop)
        L += ["### Sensitivity analysis: conflicting AF labels left out\n",
              f"{len(drop)} ECGs left out; {int(keep.sum())} remain "
              f"({int(pred.loc[keep, 'label'].sum())} AF). Locked thresholds; 95% CI from "
              f"{N_BOOT} patient-level bootstrap draws.\n",
              "| Model | Labels | AUROC | AUPRC | Sensitivity | Specificity |", "|---|---|---|---|---|---|"]
        for mname, col, thr in models:
            for lname, mask in [("all, as given", np.ones(len(pred), bool)), ("conflicts left out", keep.to_numpy())]:
                d = pred[mask & pred[col].notna().to_numpy()]
                y, p, pats = d["label"].to_numpy().astype(int), d[col].to_numpy(), d["patient_id"].to_numpy()
                m = summarise(y, p, thr)
                ci = bootstrap_ci(y, p, pats, thr, N_BOOT)
                L.append(f"| {mname} | {lname} | {fmt(m, ci, 'auroc')} | {fmt(m, ci, 'auprc')} | "
                         f"{fmt(m, ci, 'sensitivity')} | {fmt(m, ci, 'specificity')} |")
        L.append("")

    text = "\n".join(L)
    (config.RESULTS_DIR / "12_label_check_report.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
