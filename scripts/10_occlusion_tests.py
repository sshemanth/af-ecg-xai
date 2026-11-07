"""Step 10: paired tests for the swap and P-wave-removal results, every seed.

In AF ECGs, flattening the atrial windows changes little: there were no
normal P-waves to begin with, so the ECG still looks like AF. The evidence
for the atrial question therefore sits in two places that step 8 does not
test directly:

  non-AF ECGs   does removing P-waves (or putting in f-waves) raise the AF
                probability, more than the matched control or a perfectly
                regular rhythm does?
  AF ECGs       does putting in sinus P-waves lower the AF probability, more
                than a perfectly regular rhythm does?

These tests were added after seeing the step 8 results; report them as
such. Each seed is tested on the same ECGs; p-values are Benjamini-Hochberg
adjusted across all tests within a dataset and seed. "Flipped" uses each
seed's own locked threshold.

Only the saved per-ECG tables from step 8 are read; no model is run.

Run from the project folder, after 08_occlusion.py --seed all:
    python scripts/10_occlusion_tests.py

Output: results/10_occlusion_tests_report.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.runs import find_resnet_runs, primary_run  # noqa: E402
from src.selection import DATASETS  # noqa: E402
from src.stats import bh, boot_mean, fmt_p, paired_test  # noqa: E402

OCC_DIR = config.RESULTS_DIR / "occlusion"

GROUPS = {
    "non_af": ("Non-AF (random sample)", lambda r: r["grp_non_random"]),
    "af": ("Detected AF (random sample)", lambda r: r["grp_af_random"] & r["pred_af"] & (r["label"] == 1)),
    "low_rr": ("Detected low-RR-variability AF",
               lambda r: r["regular_af"] & r["pred_af"] & (r["label"] == 1)),
}
# (group, first change, second change or None = compared with no change)
TESTS = [
    ("non_af", "atrial_removed", None),
    ("non_af", "atrial_removed", "matched_control"),
    ("non_af", "atrial_removed", "rr_regularised"),
    ("non_af", "atrial_swapped", None),
    ("non_af", "atrial_swapped", "rr_regularised"),
    ("af", "atrial_swapped", None),
    ("af", "atrial_swapped", "matched_control"),
    ("af", "atrial_swapped", "rr_regularised"),
    ("low_rr", "atrial_swapped", None),
    ("low_rr", "atrial_swapped", "rr_regularised"),
]
# share of ECGs whose call changes: (group, change, wording)
FLIPS = [("non_af", "atrial_removed", "Non-AF called AF after P-waves removed"),
         ("non_af", "atrial_swapped", "Non-AF called AF after f-waves put in"),
         ("af", "atrial_swapped", "AF called non-AF after sinus P-waves put in"),
         ("af", "rr_regularised", "AF called non-AF after rhythm made regular")]


def label(test: tuple) -> str:
    _, a, b = test
    return f"{a} vs no change" if b is None else f"{a} vs {b}"


def run_tests(res: pd.DataFrame) -> list[dict]:
    out = []
    for test in TESTS:
        g = res[GROUPS[test[0]][1](res)]
        a = g[f"delta_{test[1]}"].to_numpy()
        b = np.zeros(len(g)) if test[2] is None else g[f"delta_{test[2]}"].to_numpy()
        t = paired_test(a, b)
        d = (a - b)[~(np.isnan(a) | np.isnan(b))]
        t["mean_diff"] = float(d.mean()) if len(d) else float("nan")
        t["ci"] = boot_mean(d) if len(d) >= 3 else (float("nan"), float("nan"))
        out.append(t)
    for t, q in zip(out, bh([t["p"] for t in out])):
        t["q"] = q
    return out


def flip_rate(res: pd.DataFrame, group: str, change: str, thr: float) -> float:
    g = res[GROUPS[group][1](res)]
    v = g[f"delta_{change}"].dropna()
    if v.empty:
        return float("nan")
    before = g.loc[v.index, "prob_original"] >= thr
    after = (g.loc[v.index, "prob_original"] + v) >= thr
    return float((before != after).mean())


def main() -> None:
    runs = find_resnet_runs(verbose=False)
    if not runs:
        sys.exit("No trained ResNet runs found.")
    primary = primary_run(runs)
    runs = sorted(runs, key=lambda r: r["seed"])

    L = ["# Paired tests for the swap and P-wave-removal results\n",
         "Added after seeing the step 8 results. Mean difference = mean of (first Δ − second Δ), "
         "where Δ is the change in AF probability; positive means the first change pushes "
         "towards AF more. r = rank-biserial effect size (−1 to 1). * = adjusted p < 0.05 "
         "(Benjamini-Hochberg within each dataset and seed).\n"]
    for key, (title, _) in DATASETS.items():
        tables = {}
        for run in runs:
            suffix = "" if run["name"] == primary["name"] else f"_{run['name']}"
            f = OCC_DIR / f"occlusion_{key}{suffix}.csv"
            if f.exists():
                tables[run["name"]] = (run, pd.read_csv(f, dtype={"record_id": str}))
        if not tables:
            print(f"[skip] {key}: no occlusion tables (run 08_occlusion.py --seed all)")
            continue
        if primary["name"] not in tables:
            print(f"[skip] {key}: primary model's table missing")
            continue
        # a seed table left over from an older step 8 run covers other ECGs: leave it out
        ids = set(tables[primary["name"]][1]["record_id"])
        stale = [n for n, (_, r) in tables.items() if set(r["record_id"]) != ids]
        for n in stale:
            print(f"[skip] {key}: {n} table is from a different step 8 run "
                  f"(re-run 08_occlusion.py --seed all)")
            del tables[n]
        results = {n: run_tests(res) for n, (_, res) in tables.items()}
        names = list(tables)
        head = " | ".join(f"Seed {tables[n][0]['seed']}" for n in names)
        sep = "|".join(["---"] * (len(names) + 3))

        L += [f"## {title}\n", "### Primary model "
              f"({primary['name']}): mean difference with 95% bootstrap CI\n",
              "| Group | Test | n | Mean difference (95% CI) | r | p (adjusted) |",
              "|---|---|---|---|---|---|"]
        for test, t in zip(TESTS, results[primary["name"]]):
            ci = "" if t["ci"][0] != t["ci"][0] else f" ({t['ci'][0]:+.3f} to {t['ci'][1]:+.3f})"
            md = "–" if t["mean_diff"] != t["mean_diff"] else f"{t['mean_diff']:+.3f}{ci}"
            L.append(f"| {GROUPS[test[0]][0]} | {label(test)} | {t['n']} | {md} | "
                     f"{'–' if t['r'] != t['r'] else format(t['r'], '+.2f')} | {fmt_p(t['q'])} |")

        L += ["\n### Every seed: mean difference\n",
              f"| Group | Test | {head} | Significant in |", f"|{sep}|"]
        for i, test in enumerate(TESTS):
            cells, k, tested = [], 0, 0
            for n in names:
                t = results[n][i]
                sig = t["q"] == t["q"] and t["q"] < 0.05
                k += sig
                tested += t["q"] == t["q"]
                cells.append("–" if t["mean_diff"] != t["mean_diff"] else f"{t['mean_diff']:+.3f}{'*' if sig else ''}")
            L.append(f"| {GROUPS[test[0]][0]} | {label(test)} | {' | '.join(cells)} | {k}/{tested} seeds |")

        L += ["\n### Every seed: share of ECGs whose call changes (each seed's locked threshold)\n",
              f"| Change | {head} |", f"|{'|'.join(['---'] * (len(names) + 1))}|"]
        for group, change, words in FLIPS:
            cells = [flip_rate(tables[n][1], group, change, tables[n][0]["threshold"]) for n in names]
            L.append(f"| {words} | " + " | ".join("–" if c != c else f"{100 * c:.0f}%" for c in cells) + " |")
        L.append("")

    text = "\n".join(L)
    (config.RESULTS_DIR / "10_occlusion_tests_report.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
