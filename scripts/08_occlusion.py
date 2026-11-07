"""Step 8: occlusion and counterfactual tests. Does the model NEED atrial features?

For each held-out ECG (same selection as the attribution step) the primary
model scores the original and a set of controlled changes; the outcome is
the change in predicted AF probability (changed minus original).

  atrial_removed    atrial windows replaced by a straight line (P/f-waves gone,
                    QRS, ST-T and beat timing kept)
  qrs_removed       QRS windows replaced the same way
  stt_removed       ST-T windows replaced the same way
  matched_control   the same total length as the atrial windows removed at
                    random places, which may include QRS or ST-T (does removing
                    ANY signal of that length matter?)
  rr_regularised    beats moved to perfectly even spacing; every QRS and ST-T
                    kept exactly, only the atrial windows resampled to fit
  atrial_swapped    atrial windows filled from another held-out ECG of the
                    opposite class: sinus P-waves into AF, AF f-waves into
                    non-AF; beat timing unchanged
  atrial_removed_<lead>   atrial windows removed in one lead only

If removing atrial activity lowers the AF probability more than making the
rhythm regular does (and more than the matched control), the model depends
on atrial morphology; the opposite pattern means it depends on timing.

Every changed ECG is checked automatically (shape, finite values, no
amplitude or jump far outside the original); failures are left out and
counted. Windows use the measured T-wave end of each beat, so T-wave
tails are not counted as atrial. The 0.5 s edge margins are never changed.

Statistics are paired (each ECG is its own control): Wilcoxon signed-rank
tests, rank-biserial effect sizes, bootstrap 95% CIs, and Benjamini-Hochberg
correction across comparisons and across leads.

By default only the primary model is tested. --seed all runs the same
changed ECGs through every trained seed, to check the result is not
specific to one model; each ECG is changed once and scored by all seeds.

Run from the project folder, after 06_evaluate.py:
    python scripts/08_occlusion.py --n-per-class 20   # quick check
    python scripts/08_occlusion.py                    # primary model
    python scripts/08_occlusion.py --seed all         # every seed
    python scripts/08_occlusion.py --seed 1,2         # chosen seeds (primary always included)
    python scripts/08_occlusion.py --arch inception --seed all   # second architecture
    python scripts/08_occlusion.py --dataset georgia --seed all  # one dataset (own report files)

Outputs: results/occlusion/ (per-ECG tables, example plot),
results/08_occlusion_report.md (primary model) and, with more than one
seed, results/08_occlusion_seeds_report.md (results side by side). Other
architectures add _<arch> to every file name; a single --dataset adds
_<dataset> to the report names.
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
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src import config  # noqa: E402
from src.data import ECGDataset, load_norm_stats  # noqa: E402
from src.model import pick_device  # noqa: E402
from src.perturb import (matched_control, plausible, regularise_rr,  # noqa: E402
                         remove_windows, swap_atrial)
from src.runs import arch_tag, find_runs, load_model, primary_run  # noqa: E402
from src.selection import DATASETS, GROUPS, held_out_table, select  # noqa: E402
from src.stats import bh, boot_mean, fmt_p, paired_test  # noqa: E402
from src.windows import ATRIAL, QRS, STT, beat_windows, t_wave_ends  # noqa: E402

OUT_DIR = config.RESULTS_DIR / "occlusion"
LEAD_II = config.LEADS.index("II")
MAIN = ["atrial_removed", "qrs_removed", "stt_removed", "matched_control",
        "rr_regularised", "atrial_swapped"]
LEAD_TESTS = [f"atrial_removed_{l}" for l in config.LEADS]
COMPARISONS = [("atrial_removed", "matched_control"),
               ("atrial_removed", "rr_regularised"),
               ("atrial_removed", "qrs_removed")]


def windows_for(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    return beat_windows(peaks, t_ends=t_wave_ends(x[LEAD_II], peaks))


def variants(x, codes, peaks, donor, rng) -> dict:
    out = {
        "atrial_removed": remove_windows(x, codes, ATRIAL),
        "qrs_removed": remove_windows(x, codes, QRS),
        "stt_removed": remove_windows(x, codes, STT),
        "matched_control": matched_control(x, codes, rng),
        "rr_regularised": regularise_rr(x, peaks, codes),
        "atrial_swapped": swap_atrial(x, codes, *donor) if donor is not None else None,
    }
    for li, lead in enumerate(config.LEADS):
        out[f"atrial_removed_{lead}"] = remove_windows(x, codes, ATRIAL, leads=[li])
    return out


@torch.no_grad()
def score(model, batch: list[np.ndarray], device) -> np.ndarray:
    xb = torch.from_numpy(np.stack(batch).astype(np.float32)).to(device)
    return torch.sigmoid(model(xb)).float().cpu().numpy()


def with_predictions(sel: pd.DataFrame, key: str, run: dict) -> pd.DataFrame:
    """The selected ECGs with this run's prediction (same ECGs for every seed)."""
    pred = held_out_table(key, run)[["record_id", "prob", "pred_af", "correct"]]
    return sel.drop(columns=["prob", "pred_af", "correct"]).merge(pred, on="record_id", how="left")


def run_dataset(key: str, models: dict, primary: dict, device, n_per_class: int):
    """models: {run name: (run, model)}. Returns per-run result tables, the
    count of changed ECGs that failed the checks, and an example for the plot."""
    sel = select(key, n_per_class, primary)
    pool = held_out_table(key, primary)
    ds = ECGDataset(key, DATASETS[key][1], train=False)
    pos = {rid: i for i, rid in enumerate(ds.index["record_id"])}
    rng = np.random.default_rng(config.SEED)

    # donors for the atrial swap: sinus rhythm (not paced) for AF targets, AF for the rest
    donors = {1: pool[(pool["label"] == 0) & pool["sinus"] & ~pool["paced"]]["record_id"].tolist(),
              0: pool[pool["label"] == 1]["record_id"].tolist()}
    donor_cache = {}

    def donor_for(label: int, own_id: str):
        ids = [d for d in donors[label] if d != own_id]
        if not ids:
            return None
        did = ids[int(rng.integers(len(ids)))]
        if did not in donor_cache:
            dx = ds[pos[did]][0].numpy()
            dp = np.array(str(pool.loc[pool["record_id"] == did, "rpeaks"].iloc[0]).split(), dtype=int)
            donor_cache[did] = (dx, windows_for(dx, dp))
        return donor_cache[did]

    rows = {name: [] for name in models}
    failed, example = {k: 0 for k in MAIN + LEAD_TESTS}, None
    for _, r in tqdm(sel.iterrows(), total=len(sel), desc=f"{key} occlusion"):
        x = ds[pos[r["record_id"]]][0].numpy()
        peaks = np.array(str(r["rpeaks"]).split(), dtype=int)
        codes = windows_for(x, peaks)
        var = variants(x, codes, peaks, donor_for(int(r["label"]), r["record_id"]), rng)
        names, batch = ["original"], [x]
        for name, v in var.items():
            if plausible(x, v):
                names.append(name)
                batch.append(v)
            else:
                failed[name] += 1
        for run_name, (_, model) in models.items():
            probs = dict(zip(names, score(model, batch, device)))
            row = {"record_id": r["record_id"], "prob_original": float(probs["original"])}
            for name in var:
                row[f"delta_{name}"] = float(probs[name] - probs["original"]) if name in probs else np.nan
            rows[run_name].append(row)
            if (run_name == primary["name"] and example is None and r["label"] == 1
                    and r["correct"] and all(n in probs for n in MAIN)):
                example = {"id": r["record_id"], "x": x, "codes": codes, "var": var, "probs": probs}
    res = {name: (sel if name == primary["name"] else with_predictions(sel, key, run))
           .merge(pd.DataFrame(rows[name]), on="record_id")
           for name, (run, _) in models.items()}
    return res, failed, example


def comparison_tests(res: pd.DataFrame) -> list[tuple]:
    """Paired comparisons on detected AF ECGs, BH-adjusted across all rows:
    [(group, first, second, test result, adjusted p), ...]"""
    tests, labels = [], []
    for gname, col in [("AF (random sample)", "grp_af_random"), ("Low-RR-variability AF", "regular_af")]:
        g = res[res[col] & res["pred_af"] & (res["label"] == 1)]
        for a, b in COMPARISONS:
            tests.append(paired_test(g[f"delta_{a}"].to_numpy(), g[f"delta_{b}"].to_numpy()))
            labels.append((gname, a, b))
    return [(*lab, t, q) for lab, t, q in zip(labels, tests, bh([t["p"] for t in tests]))]


def report_dataset(key: str, res: pd.DataFrame, failed: dict, threshold: float) -> list[str]:
    L = [f"## {DATASETS[key][0]}\n",
         f"{len(res)} ECGs. Δ = changed minus original AF probability "
         f"(negative: AF evidence lost). Flipped = crosses the locked threshold "
         f"({threshold:.3f}).\n",
         "### Change in AF probability\n",
         "| Group | Change | n | Mean Δ (95% CI) | Median Δ | Flipped |",
         "|---|---|---|---|---|---|"]
    for gname, col in GROUPS:
        g = res[res[col]]
        if g.empty:
            continue
        for name in MAIN:
            v = g[f"delta_{name}"].dropna()
            if v.empty:
                continue
            lo, hi = boot_mean(v.to_numpy()) if len(v) >= 3 else (np.nan, np.nan)
            before = g.loc[v.index, "prob_original"] >= threshold
            after = (g.loc[v.index, "prob_original"] + v) >= threshold
            flipped = (before != after).mean()
            ci = f" ({lo:+.3f} to {hi:+.3f})" if lo == lo else ""
            L.append(f"| {gname} | {name} | {len(v)} | {v.mean():+.3f}{ci} | "
                     f"{v.median():+.3f} | {100 * flipped:.1f}% |")
    L.append("")

    # paired comparisons, AF ECGs that the model detected
    L += ["### Paired comparisons (AF ECGs the model detected)\n",
          "Each row asks whether the first change lowers the AF probability more than "
          "the second, on the same ECGs. r is the rank-biserial effect size "
          "(−1 to 1; negative = first change lowers AF probability more). "
          "p-values Benjamini-Hochberg adjusted across all rows.\n",
          "| Group | First | Second | n | Median difference in Δ | r | p (adjusted) |",
          "|---|---|---|---|---|---|---|"]
    for gname, a, b, t, q in comparison_tests(res):
        L.append(f"| {gname} | {a} | {b} | {t['n']} | {t['median_diff']:+.3f} | "
                 f"{t['r']:+.2f} | {fmt_p(q)} |")
    L.append("")

    # by lead
    af = res[res["grp_af_random"] & res["pred_af"] & (res["label"] == 1)]
    lead_tests = [paired_test(af[f"delta_{n}"].to_numpy(), np.zeros(len(af))) for n in LEAD_TESTS]
    qs = bh([t["p"] for t in lead_tests])
    L += ["### Atrial windows removed in one lead only (detected AF, random sample)\n",
          "| Lead | n | Mean Δ | r | p (adjusted across leads) |", "|---|---|---|---|---|"]
    for lead, name, t, q in zip(config.LEADS, LEAD_TESTS, lead_tests, qs):
        L.append(f"| {lead} | {t['n']} | {af[f'delta_{name}'].mean():+.4f} | {t['r']:+.2f} | {fmt_p(q)} |")
    L.append("")

    bad = {k: v for k, v in failed.items() if v}
    L.append("Automatic checks: " + (", ".join(f"{k} {v}" for k, v in bad.items())
                                     + " changed ECGs failed and were left out." if bad
                                     else "every changed ECG passed.") + "\n")
    return L


def seeds_report(key: str, res: dict, runs: dict) -> list[str]:
    """Each seed's results side by side. Each seed judges "detected" with its
    own locked threshold, so the detected-AF groups can differ slightly."""
    names = sorted(res, key=lambda n: runs[n]["seed"])
    head = " | ".join(f"Seed {runs[n]['seed']}" for n in names)
    sep = "|".join(["---"] * (len(names) + 2))
    L = [f"## {DATASETS[key][0]}\n"]
    groups = [("Detected AF (random sample)", lambda r: r["grp_af_random"] & r["pred_af"] & (r["label"] == 1)),
              ("Non-AF (random sample)", lambda r: r["grp_non_random"]),
              ("Detected low-RR-variability AF", lambda r: r["regular_af"] & r["pred_af"] & (r["label"] == 1))]
    for gname, pick in groups:
        L += [f"### {gname}: mean Δ by seed\n",
              f"| Change | {head} | Across seeds: mean (min to max) |", f"|{sep}|"]
        for change in MAIN:
            means = [res[n][pick(res[n])][f"delta_{change}"].mean() for n in names]
            m = np.array(means, float)
            if np.isnan(m).all():
                continue
            cells = " | ".join("–" if v != v else f"{v:+.3f}" for v in m)
            L.append(f"| {change} | {cells} | {np.nanmean(m):+.3f} ({np.nanmin(m):+.3f} to {np.nanmax(m):+.3f}) |")
        ns = [int(pick(res[n]).sum()) for n in names]
        L.append(f"\nECGs per seed: {', '.join(map(str, ns))}\n")

    L += ["### Paired comparisons by seed (rank-biserial r; * = adjusted p < 0.05)\n",
          "Negative r: the first change lowers the AF probability more than the second.\n",
          f"| Group | First | Second | {head} | Significant in |", f"|---|---|{sep}|"]
    per_seed = {n: comparison_tests(res[n]) for n in names}
    for i, (gname, a, b, _, _) in enumerate(per_seed[names[0]]):
        cells, k, tested = [], 0, 0
        for n in names:
            t, q = per_seed[n][i][3], per_seed[n][i][4]
            star = "*" if q == q and q < 0.05 else ""
            k += bool(star)
            tested += q == q
            cells.append("–" if t["r"] != t["r"] else f"{t['r']:+.2f}{star}")
        L.append(f"| {gname} | {a} | {b} | {' | '.join(cells)} | {k}/{tested} seeds |")
    L.append("")
    return L


def plot_example(key: str, ex: dict, mean: np.ndarray, std: np.ndarray, tag: str = "") -> Path:
    t = np.arange(config.N_SAMPLES) / config.TARGET_FS
    li = config.LEADS.index("V1")
    show = ["original"] + MAIN
    fig, axes = plt.subplots(len(show), 1, figsize=(14, 1.6 * len(show)), sharex=True)
    for ax, name in zip(axes, show):
        sig = ex["x"] if name == "original" else ex["var"][name]
        ax.plot(t, sig[li] * std[li, 0] + mean[li, 0], lw=0.6, color="black")
        m = ex["codes"] == ATRIAL
        for a, b in zip(*[np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))[i::2] for i in (0, 1)]):
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color="#4c9be8", alpha=0.12, lw=0)
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
        ax.text(1.0, 0.85, f"AF prob {ex['probs'][name]:.2f}", transform=ax.transAxes,
                ha="right", fontsize=8)
    axes[0].set_title(f"{key} {ex['id']} | lead V1 | blue = atrial windows", fontsize=9, loc="left")
    axes[-1].set_xlabel("seconds")
    fig.tight_layout()
    out = OUT_DIR / f"example_{key}{tag}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def compare_with_ptbxl(results: dict, tag: str) -> list[str]:
    """PTB-XL vs each external set, detected AF (random sample). Tables not
    run this time are read from earlier runs of this step."""
    def table(k):
        if k in results:
            return results[k]
        f = OUT_DIR / f"occlusion_{k}{tag}.csv"
        return pd.read_csv(f, dtype={"record_id": str}) if f.exists() else None

    base = table("ptbxl")
    others = [(k, table(k)) for k in config.EXTERNAL_DATASETS]
    others = [(k, t) for k, t in others if t is not None]
    if base is None or not others:
        return []
    pick = lambda r, name: r[r["grp_af_random"] & r["pred_af"] & (r["label"] == 1)][f"delta_{name}"].dropna().to_numpy()
    L = []
    for k, other in others:
        L += [f"## PTB-XL vs {config.DATASET_NAMES[k]} (detected AF, random sample)\n",
              f"| Change | PTB-XL mean Δ | {config.DATASET_NAMES[k]} mean Δ | Difference (95% CI) |",
              "|---|---|---|---|"]
        rng = np.random.default_rng(config.SEED)
        for name in MAIN:
            a, b = pick(base, name), pick(other, name)
            if len(a) < 3 or len(b) < 3:
                continue
            d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(2000)]
            L.append(f"| {name} | {a.mean():+.3f} | {b.mean():+.3f} | {a.mean() - b.mean():+.3f} "
                     f"({np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f}) |")
        L.append("")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=config.ALL_DATASETS + ["all"], default="all")
    ap.add_argument("--arch", choices=config.ARCHS, default="resnet")
    ap.add_argument("--n-per-class", type=int, default=config.XAI_N_PER_CLASS)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", default="primary",
                    help="'primary' (default), 'all', or seeds such as 1,2")
    args = ap.parse_args()

    runs = find_runs(args.arch, verbose=False)
    if not runs:
        sys.exit(f"No trained {args.arch} runs found.")
    primary = primary_run(runs)
    tag = arch_tag(args.arch)
    if args.seed == "primary":
        chosen = [primary]
    elif args.seed == "all":
        chosen = runs
    else:
        want = {int(s) for s in args.seed.split(",")}
        chosen = [r for r in runs if r["seed"] in want or r is primary]
        missing = want - {r["seed"] for r in runs}
        if missing:
            sys.exit(f"Seed(s) {sorted(missing)} not trained. Trained: {sorted(r['seed'] for r in runs)}")
    device = pick_device(args.device)
    models = {r["name"]: (r, load_model(r, device)) for r in chosen}
    run_info = {r["name"]: r for r in chosen}
    print(f"Models: {', '.join(models)} (primary: {primary['name']})")
    mean, std = load_norm_stats()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    keys = config.ALL_DATASETS if args.dataset == "all" else [args.dataset]
    keys = [k for k in keys if (config.EVAL_DIR / f"predictions_{k}.csv").exists() or args.dataset != "all"]
    lines = ["# Occlusion and counterfactual tests\n",
             f"Model: **{primary['name']}** (primary). Does the model need atrial activity, "
             f"or does it rely on beat timing? First/last {config.EDGE_MARGIN_S} s never changed.\n"]
    results, seed_lines = {}, ["# Occlusion and counterfactual tests: every seed\n",
                               f"Same ECGs and the same changes for each seed; primary model "
                               f"{primary['name']}. Does the pattern hold for every trained model?\n"]
    for key in keys:
        by_run, failed, ex = run_dataset(key, models, primary, device, args.n_per_class)
        for name, r in by_run.items():
            suffix = tag if name == primary["name"] else f"_{name}"
            r.to_csv(OUT_DIR / f"occlusion_{key}{suffix}.csv", index=False)
        res = by_run[primary["name"]]
        results[key] = res
        if len(models) > 1:
            seed_lines += seeds_report(key, by_run, run_info)
        lines += report_dataset(key, res, failed, primary["threshold"])
        if ex is not None:
            p = plot_example(key, ex, mean, std, tag)
            lines.append(f"Example: `{p.relative_to(config.ROOT)}`\n")

    lines += compare_with_ptbxl(results, tag)

    text = "\n".join(lines)
    suffix = tag + ("" if args.dataset == "all" else f"_{args.dataset}")
    (config.RESULTS_DIR / f"08_occlusion_report{suffix}.md").write_text(text)
    print("\n" + text)
    if len(models) > 1:
        text = "\n".join(seed_lines)
        (config.RESULTS_DIR / f"08_occlusion_seeds_report{suffix}.md").write_text(text)
        print("\n" + text)


if __name__ == "__main__":
    main()
