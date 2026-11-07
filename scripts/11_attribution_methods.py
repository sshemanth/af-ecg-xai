"""Step 11: do other attribution methods also miss the model's use of atrial activity?

Six methods explain the same model on the same held-out ECGs:
Integrated Gradients, DeepLIFT, GradientSHAP, Input x Gradient (lead x
time), Grad-CAM and sliding-window Occlusion (time only). For each:

  where it looks   share of evidence for AF in atrial, QRS and ST-T windows
                   in AF ECGs, and share of evidence AGAINST AF in atrial
                   windows in non-AF ECGs (step 8 showed normal P-waves are
                   what pulls the model away from AF)
  deletion test    removes 100 ms segments in the order the map ranks them
                   and scores the area between the most-relevant-first and
                   least-relevant-first curves (ABPC), against random order
                   (see src/faithfulness.py, including its caveat)
  agreement with   does the map's atrial share track how much each ECG's AF
  step 8           probability falls when sinus P-waves are swapped in?
  agreement        between every pair of methods (time profiles)

A class-balanced random subset of the step 7-8 sample is used
(config.XAI_METHODS_N_PER_CLASS per class, without the extra subgroups), so
every ECG also has its step 8 counterfactual results. Run after 08_occlusion.py, so the step 8 comparison can be made.

Run from the project folder:
    python scripts/11_attribution_methods.py --n-per-class 20   # quick check
    python scripts/11_attribution_methods.py
    python scripts/11_attribution_methods.py --arch inception

Outputs: results/xai/methods_<dataset>.csv, deletion curve plots, and
results/11_attribution_methods_report.md (_<arch> for other architectures)
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
from scipy.stats import spearmanr, wilcoxon  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src import config  # noqa: E402
from src.data import ECGDataset  # noqa: E402
from src.faithfulness import FRACTIONS, abpc, curves, segment_scores  # noqa: E402
from src.model import pick_device  # noqa: E402
from src.runs import arch_label, arch_tag, find_runs, load_model, primary_run  # noqa: E402
from src.selection import DATASETS, select  # noqa: E402
from src.stats import bh, boot_mean, fmt_p  # noqa: E402
from src.windows import OUTSIDE, beat_windows, shares, t_wave_ends  # noqa: E402
from src.xai import MultiExplainer, spearman  # noqa: E402

LEAD_II = config.LEADS.index("II")
METHODS = MultiExplainer.METHODS
SIGNED = [m for m in METHODS if m != "Grad-CAM"]       # Grad-CAM keeps positive evidence only
N_RANDOM = 3


def background(n: int = 32) -> torch.Tensor:
    """Random PTB-XL training ECGs, the baseline set for GradientSHAP."""
    ds = ECGDataset("ptbxl", ["train"], train=False)
    pick = np.random.default_rng(config.SEED).choice(len(ds), size=min(n, len(ds)), replace=False)
    return torch.stack([ds[int(i)][0] for i in pick])


def run_dataset(key: str, sel: pd.DataFrame, expl: MultiExplainer, model, device, batch: int):
    ds = ECGDataset(key, DATASETS[key][1], train=False)
    pos = {rid: i for i, rid in enumerate(ds.index["record_id"])}
    rng = np.random.default_rng(config.SEED)
    rows, del_curves, pair_rho = [], {m: [] for m in METHODS + ["Random"]}, []
    for start in tqdm(range(0, len(sel), batch), desc=f"{key} methods"):
        part = sel.iloc[start:start + batch]
        x = torch.stack([ds[pos[r]][0] for r in part["record_id"]])
        maps = {m: expl.attribute(m, x) for m in METHODS}
        for j, (_, r) in enumerate(part.iterrows()):
            xj = x[j].numpy()
            peaks = np.array(str(r["rpeaks"]).split(), dtype=int)
            codes = beat_windows(peaks, t_ends=t_wave_ends(xj[LEAD_II], peaks))
            inside = codes != OUTSIDE
            prof = {m: (a[j].sum(axis=0) if a.ndim == 3 else a[j]) for m, a in maps.items()}
            row = {"record_id": r["record_id"]}
            for m, t in prof.items():
                sh = shares(t, codes)
                row.update({f"{m}_{k}": sh[k] for k in
                            ["atrial_share", "qrs_share", "stt_share", "atrial_enrich", "qrs_enrich"]})
                if m in SIGNED:
                    row[f"{m}_atrial_share_against"] = shares(-t, codes)["atrial_share"]
            # pairwise agreement of the time profiles (evidence for AF)
            rho = {}
            for a in range(len(METHODS)):
                for b in range(a + 1, len(METHODS)):
                    ma, mb = METHODS[a], METHODS[b]
                    rho[(ma, mb)] = spearman(np.clip(prof[ma], 0, None)[inside],
                                             np.clip(prof[mb], 0, None)[inside])
            pair_rho.append(rho)
            # deletion test on AF ECGs the model detected
            if r["label"] == 1 and r["pred_af"]:
                ranks = {m: segment_scores(t) for m, t in prof.items()}
                n_seg = len(next(iter(ranks.values())))
                for i in range(N_RANDOM):
                    ranks[f"Random{i}"] = rng.random(n_seg)
                cv = curves(model, xj, ranks, device)
                for m in METHODS:
                    row[f"{m}_abpc"] = abpc(*cv[m])
                    del_curves[m].append(cv[m][0])
                rand = [abpc(*cv[f"Random{i}"]) for i in range(N_RANDOM)]
                row["Random_abpc"] = float(np.mean(rand))
                del_curves["Random"].append(np.mean([cv[f"Random{i}"][0] for i in range(N_RANDOM)], axis=0))
            rows.append(row)
    res = sel.merge(pd.DataFrame(rows), on="record_id")
    agree = {k: float(np.nanmedian([d[k] for d in pair_rho])) for k in pair_rho[0]} if pair_rho else {}
    return res, {m: np.mean(v, axis=0) for m, v in del_curves.items() if v}, agree


def plot_deletion(key: str, mean_curves: dict, tag: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for m, c in mean_curves.items():
        ax.plot(100 * FRACTIONS, c, marker="o", ms=3, lw=1.4 if m != "Random" else 2,
                ls="--" if m == "Random" else "-", color="grey" if m == "Random" else None, label=m)
    ax.set_xlabel("Segments removed, most relevant first (%)")
    ax.set_ylabel("Mean AF probability")
    ax.set_title(f"{config.DATASET_NAMES[key]}: deletion curves (detected AF)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = config.XAI_DIR / f"deletion_{key}{tag}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def mean_ci(v) -> str:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return "–" if len(v) == 0 else f"{v.mean():.3f}"
    lo, hi = boot_mean(v)
    return f"{v.mean():.3f} ({lo:.3f}–{hi:.3f})"


def report(key: str, res: pd.DataFrame, agree: dict, occ: pd.DataFrame | None) -> list[str]:
    af = res[res["grp_af_random"] & (res["label"] == 1)]
    det = af[af["pred_af"]]
    non = res[res["grp_non_random"]]
    L = [f"## {DATASETS[key][0]}\n",
         f"{len(af)} AF ECGs ({len(det)} detected) and {len(non)} non-AF ECGs (random sample).\n",
         "### Where each method looks\n",
         "Shares are of positive attribution (evidence for AF) unless marked *against*. "
         "Enrichment = share ÷ share of time; above 1 = more than its length alone would give.\n",
         "| Method | AF: atrial share | AF: atrial enrichment | AF: QRS enrichment | "
         "Non-AF: atrial share of evidence *against* AF |",
         "|---|---|---|---|---|"]
    for m in METHODS:
        against = mean_ci(non[f"{m}_atrial_share_against"]) if m in SIGNED else "– (positive only)"
        L.append(f"| {m} | {mean_ci(af[f'{m}_atrial_share'])} | {mean_ci(af[f'{m}_atrial_enrich'])} | "
                 f"{mean_ci(af[f'{m}_qrs_enrich'])} | {against} |")

    # deletion test
    L += ["\n### Deletion test (detected AF)\n",
          "ABPC = mean of P(AF) with least-relevant-first removal minus most-relevant-first "
          "removal; higher = removing what the map ranks highest matters more. 'vs random' = "
          "paired Wilcoxon test on each ECG's ABPC minus its random-order ABPC, "
          "Benjamini-Hochberg adjusted across methods.\n",
          "| Method | ABPC (95% CI) | Difference from random | p (adjusted) |", "|---|---|---|---|"]
    tests = []
    for m in METHODS:
        d = (det[f"{m}_abpc"] - det["Random_abpc"]).dropna().to_numpy()
        p = wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else float("nan")
        tests.append((m, d, p))
    for (m, d, _), q in zip(tests, bh([t[2] for t in tests])):
        L.append(f"| {m} | {mean_ci(det[f'{m}_abpc'])} | {np.mean(d):+.3f} | {fmt_p(q)} |" if len(d)
                 else f"| {m} | – | – | – |")
    L.append(f"| Random order | {mean_ci(det['Random_abpc'])} | – | – |")
    L.append("\nCaveat: a removed segment is a flat line, which has no P-wave; the model may "
             "read that as AF evidence, so deletion partly measures its reaction to the flat "
             "line (see src/faithfulness.py).\n")

    # agreement with step 8
    if occ is not None:
        merged = det.merge(occ[["record_id", "delta_atrial_swapped", "delta_qrs_removed"]], on="record_id")
        L += ["### Agreement with the counterfactual tests of step 8 (detected AF)\n",
              "Spearman correlation across ECGs between the map's share and the drop in AF "
              "probability when that part is changed in step 8. A map that reflects what the "
              "model uses should correlate positively.\n",
              "| Method | Atrial share vs drop after sinus P-wave swap | QRS share vs drop after QRS removal |",
              "|---|---|---|"]
        for m in METHODS:
            cells = []
            for share, delta in [(f"{m}_atrial_share", "delta_atrial_swapped"),
                                 (f"{m}_qrs_share", "delta_qrs_removed")]:
                d = merged[[share, delta]].dropna()
                if len(d) < 10:
                    cells.append("–")
                    continue
                rho, p = spearmanr(d[share], -d[delta])
                cells.append(f"ρ = {rho:+.2f} (p {fmt_p(p)}, n {len(d)})")
            L.append(f"| {m} | {cells[0]} | {cells[1]} |")
        L.append("")
    else:
        L.append("(No step 8 table for this model and dataset, so the comparison with the "
                 "counterfactual tests is left out.)\n")

    # pairwise agreement
    if agree:
        L += ["### Agreement between methods (median per-ECG Spearman of time profiles)\n",
              "| | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
        for a in METHODS:
            cells = []
            for b in METHODS:
                v = 1.0 if a == b else agree.get((a, b), agree.get((b, a), np.nan))
                cells.append(f"{v:.2f}")
            L.append(f"| {a} | " + " | ".join(cells) + " |")
        L.append("")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=config.ALL_DATASETS + ["all"], default="all")
    ap.add_argument("--arch", choices=config.ARCHS, default="resnet")
    ap.add_argument("--n-per-class", type=int, default=config.XAI_METHODS_N_PER_CLASS)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    runs = find_runs(args.arch, verbose=False)
    if not runs:
        sys.exit(f"No trained {args.arch} runs found.")
    primary = primary_run(runs)
    tag = arch_tag(args.arch)
    device = pick_device(args.device)
    model = load_model(primary, device)
    expl = MultiExplainer(model, device, background())
    config.XAI_DIR.mkdir(parents=True, exist_ok=True)

    keys = config.ALL_DATASETS if args.dataset == "all" else [args.dataset]
    keys = [k for k in keys if (config.EVAL_DIR / f"predictions_{k}.csv").exists() or args.dataset != "all"]
    lines = [f"# Attribution methods compared ({arch_label(args.arch)})\n",
             f"Model: **{primary['name']}** (primary). Six methods on the same ECGs and the same "
             f"output (the AF logit). First/last {config.EDGE_MARGIN_S} s left out.\n"]
    for key in keys:
        # draw from the same sample as steps 7-8, so every ECG here also has
        # its counterfactual results from step 8; select() returns a random
        # order, so the first N of each class are a fair subset
        sel = select(key, config.XAI_N_PER_CLASS, primary)
        sel = pd.concat([sel[sel["grp_af_random"]].head(args.n_per_class),
                         sel[sel["grp_non_random"] & ~sel["grp_af_random"]].head(args.n_per_class)])
        sel = sel.reset_index(drop=True)
        res, mean_curves, agree = run_dataset(key, sel, expl, model, device, args.batch)
        res.to_csv(config.XAI_DIR / f"methods_{key}{tag}.csv", index=False)
        occ_f = config.RESULTS_DIR / "occlusion" / f"occlusion_{key}{tag}.csv"
        occ = pd.read_csv(occ_f, dtype={"record_id": str}) if occ_f.exists() else None
        lines += report(key, res, agree, occ)
        if mean_curves:
            p = plot_deletion(key, mean_curves, tag)
            lines.append(f"Deletion curves: `{p.relative_to(config.ROOT)}`\n")

    text = "\n".join(lines)
    suffix = tag + ("" if args.dataset == "all" else f"_{args.dataset}")
    (config.RESULTS_DIR / f"11_attribution_methods_report{suffix}.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
