"""Step 7: where does the model look? Attribution maps and window shares.

  - two complementary attribution methods: Integrated Gradients
    (lead x time) and Grad-CAM (time only)
  - windows defined from the R-peaks: atrial (P/f-waves), QRS, ST-T
  - share of positive attribution (evidence for AF) in each window,
    compared with how much time each window takes (enrichment)
  - AF vs non-AF, correct vs missed, PTB-XL vs each external set
  - a class-balanced random sample plus all low-RR-variability,
    near-regular and paced AF ECGs
  - agreement between the two methods, and stability under small noise
The first and last 0.5 s are left out (config.EDGE_MARGIN_S).

These maps show where the model looks, not what it NEEDS. That causal
question is tested by occluding the windows.

Uses the primary model and the held-out sets only, after 06_evaluate.py
(it needs that step's predictions to know which AF ECGs were missed).

Run from the project folder:
    python scripts/07_attributions.py
    python scripts/07_attributions.py --dataset chapman
    python scripts/07_attributions.py --n-per-class 20      # quick check
    python scripts/07_attributions.py --arch inception      # second architecture

Outputs: results/xai/ (per-ECG table, example plots) and
results/07_attribution_report.md (07_attribution_report_<arch>.md for
other architectures)
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
from src.runs import arch_tag, find_runs, load_model, primary_run  # noqa: E402
from src.selection import DATASETS, GROUPS, select  # noqa: E402
from src.windows import ATRIAL, OUTSIDE, QRS, STT, beat_windows, shares, t_wave_ends  # noqa: E402
from src.xai import Explainer, spearman  # noqa: E402

LEAD_II = config.LEADS.index("II")


def explain(key: str, sel: pd.DataFrame, explainer: Explainer, noise_std: np.ndarray,
            batch: int) -> tuple[pd.DataFrame, dict]:
    ds = ECGDataset(key, DATASETS[key][1], train=False)
    pos = {rid: i for i, rid in enumerate(ds.index["record_id"])}
    rows, examples = [], {}
    n_stab = min(config.XAI_N_STABILITY, len(sel))
    for start in tqdm(range(0, len(sel), batch), desc=f"{key} attributions"):
        part = sel.iloc[start:start + batch]
        x = torch.stack([ds[pos[r]][0] for r in part["record_id"]])
        ig = explainer.integrated_gradients(x)                   # [b, 12, T]
        cam = explainer.grad_cam(x)                              # [b, T]

        # stability: same ECGs with a little noise (first n_stab only)
        stab_mask = (np.arange(start, start + len(part)) < n_stab)
        ig_noisy = None
        if stab_mask.any():
            noise = torch.randn_like(x) * torch.from_numpy(noise_std)
            ig_noisy = explainer.integrated_gradients(x + noise)

        for j, (_, r) in enumerate(part.iterrows()):
            peaks = np.array(str(r["rpeaks"]).split(), dtype=int)
            codes = beat_windows(peaks, t_ends=t_wave_ends(x[j, LEAD_II].numpy(), peaks))
            inside = codes != OUTSIDE
            ig_t = ig[j].sum(axis=0)                             # signed, summed over leads
            row = {"record_id": r["record_id"]}
            row.update({f"ig_{k}": v for k, v in shares(ig_t, codes).items()})
            row.update({f"gc_{k}": v for k, v in shares(cam[j], codes).items()})
            # per lead: share of that lead's positive IG that falls in atrial windows
            for li, lead in enumerate(config.LEADS):
                p = np.clip(ig[j, li], 0, None)
                tot = p[inside].sum()
                row[f"lead_{lead}_atrial_share"] = float(p[codes == ATRIAL].sum() / tot) if tot > 0 else np.nan
            row["agreement_spearman"] = spearman(np.clip(ig_t, 0, None)[inside], cam[j][inside])
            if stab_mask[j]:
                ig_n = ig_noisy[j].sum(axis=0)
                row["stability_spearman"] = spearman(ig_t[inside], ig_n[inside])
                row["stability_atrial_diff"] = abs(shares(ig_n, codes)["atrial_share"]
                                                   - row["ig_atrial_share"])
            rows.append(row)
            # keep a few AF examples for plotting
            want = (r["label"] == 1) and (len(examples) < 4 or (r["regular_af"] and
                                          sum(e["regular"] for e in examples.values()) < 2))
            if want and len(examples) < 6:
                examples[r["record_id"]] = {"x": x[j].numpy(), "ig": ig[j], "cam": cam[j],
                                            "codes": codes, "peaks": peaks, "prob": r["prob"],
                                            "regular": bool(r["regular_af"])}
    return sel.merge(pd.DataFrame(rows), on="record_id"), examples


def plot_examples(key: str, examples: dict, mean: np.ndarray, std: np.ndarray, tag: str = "") -> Path:
    leads = [config.LEADS.index("II"), config.LEADS.index("V1")]
    t = np.arange(config.N_SAMPLES) / config.TARGET_FS
    colours = {ATRIAL: "#4c9be8", QRS: "#e8804c", STT: "#9ad17a"}
    fig, axes = plt.subplots(len(examples), 1, figsize=(14, 3.0 * len(examples)), squeeze=False)
    for ax, (rid, e) in zip(axes[:, 0], examples.items()):
        mv = e["x"] * std + mean                                  # back to mV for display
        # shade windows
        for code, col in colours.items():
            m = e["codes"] == code
            edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                ax.axvspan(t[a], t[min(b, len(t) - 1)], color=col, alpha=0.15, lw=0)
        gap = float(np.ptp(mv[leads[0]]) / 2 + np.ptp(mv[leads[1]]) / 2) + 0.3
        ig_pos = np.clip(e["ig"], 0, None)
        scale = ig_pos[leads].max() or 1.0
        for k, li in enumerate(leads):
            y = mv[li] - k * gap
            ax.plot(t, y, color="black", lw=0.6)
            # attribution as red dots sized by IG on that lead
            s = ig_pos[li] / scale
            hot = s > 0.15
            ax.scatter(t[hot], y[hot], s=12 * s[hot], color="red", alpha=0.6, lw=0)
            ax.text(-0.3, y[:250].mean(), config.LEADS[li], ha="right", va="center")
        cam = e["cam"] / (e["cam"].max() or 1.0)
        ax.plot(t, mv[leads[1]].min() - gap - 0.3 + 0.6 * cam, color="purple", lw=1)
        ax.text(-0.3, mv[leads[1]].min() - gap, "Grad-CAM", ha="right", va="center", fontsize=8)
        ax.set_title(f"{key} {rid} | AF prob {e['prob']:.2f}"
                     f"{' | low RR variability' if e['regular'] else ''} | "
                     f"red = Integrated Gradients (evidence for AF); shading: blue atrial, "
                     f"orange QRS, green ST-T", fontsize=8, loc="left")
        ax.set_xlim(0, config.SIGNAL_SECONDS)
        ax.set_yticks([])
        ax.axvspan(0, config.EDGE_MARGIN_S, color="grey", alpha=0.25, lw=0)
        ax.axvspan(config.SIGNAL_SECONDS - config.EDGE_MARGIN_S, config.SIGNAL_SECONDS,
                   color="grey", alpha=0.25, lw=0)
    axes[-1, 0].set_xlabel("seconds (grey = edge margins, left out)")
    fig.tight_layout()
    out = config.XAI_DIR / f"examples_{key}{tag}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def mean_ci(v: pd.Series, n_boot: int = 1000) -> str:
    v = v.dropna().to_numpy()
    if len(v) == 0:
        return "–"
    if len(v) < 3:
        return f"{v.mean():.3f}"
    rng = np.random.default_rng(config.SEED)
    boots = [rng.choice(v, len(v)).mean() for _ in range(n_boot)]
    return f"{v.mean():.3f} ({np.percentile(boots, 2.5):.3f}–{np.percentile(boots, 97.5):.3f})"


def report_dataset(key: str, res: pd.DataFrame) -> list[str]:
    L = [f"## {DATASETS[key][0]}\n", f"{len(res)} ECGs explained.\n",
         "### Share of evidence for AF in each window (mean, 95% CI)\n",
         "Enrichment = share ÷ share of time. Above 1: the window gets more "
         "attention than its length alone would give it.\n",
         "| Group | Method | n | Atrial share | QRS share | ST-T share | Atrial enrichment | QRS enrichment |",
         "|---|---|---|---|---|---|---|---|"]
    for gname, col in GROUPS:
        g = res[res[col]]
        if g.empty:
            continue
        for meth, p in [("IG", "ig"), ("Grad-CAM", "gc")]:
            L.append(f"| {gname} | {meth} | {len(g)} | {mean_ci(g[f'{p}_atrial_share'])} | "
                     f"{mean_ci(g[f'{p}_qrs_share'])} | {mean_ci(g[f'{p}_stt_share'])} | "
                     f"{mean_ci(g[f'{p}_atrial_enrich'])} | {mean_ci(g[f'{p}_qrs_enrich'])} |")
    t = res[res["grp_af_random"]]
    if not t.empty:
        L.append(f"\nTime taken by each window in AF ECGs (mean): atrial "
                 f"{t['ig_atrial_time'].mean():.2f}, QRS {t['ig_qrs_time'].mean():.2f}, "
                 f"ST-T {t['ig_stt_time'].mean():.2f}.\n")

    af = res[res["label"] == 1]
    L += ["### By lead (Integrated Gradients, all AF explained)\n",
          "Share of each lead's evidence for AF that falls in atrial windows.\n",
          "| Lead | " + " | ".join(config.LEADS) + " |",
          "|---|" + "---|" * len(config.LEADS),
          "| Atrial share | " + " | ".join(f"{af[f'lead_{l}_atrial_share'].mean():.2f}"
                                       for l in config.LEADS) + " |", ""]

    L += ["### Correct vs missed AF (Integrated Gradients)\n",
          "| AF ECGs | n | Atrial share | QRS share |", "|---|---|---|---|"]
    for name, m in [("Caught", af["correct"]), ("Missed", ~af["correct"])]:
        g = af[m]
        L.append(f"| {name} | {len(g)} | {mean_ci(g['ig_atrial_share'])} | {mean_ci(g['ig_qrs_share'])} |")

    L += ["", "### Checks\n",
          f"- Agreement IG vs Grad-CAM (Spearman, per ECG, time profile): median "
          f"{res['agreement_spearman'].median():.2f}",
          f"- Agreement on atrial share across ECGs (Pearson): "
          f"{res[['ig_atrial_share', 'gc_atrial_share']].corr().iloc[0, 1]:.2f}"]
    if "stability_spearman" in res:
        s = res.dropna(subset=["stability_spearman"])
        L += [f"- Stability of IG under {config.XAI_NOISE_MV} mV noise ({len(s)} ECGs): "
              f"median Spearman {s['stability_spearman'].median():.2f}; atrial share "
              f"changes by a median {s['stability_atrial_diff'].median():.3f}"]
    return L + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=config.ALL_DATASETS + ["all"], default="all")
    ap.add_argument("--arch", choices=config.ARCHS, default="resnet")
    ap.add_argument("--n-per-class", type=int, default=config.XAI_N_PER_CLASS)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    runs = find_runs(args.arch, verbose=False)
    if not runs:
        sys.exit(f"No trained {args.arch} runs found.")
    primary = primary_run(runs)
    tag = arch_tag(args.arch)
    device = pick_device(args.device)
    explainer = Explainer(load_model(primary, device), device)
    mean, std = load_norm_stats()
    noise_std = (config.XAI_NOISE_MV / std).astype(np.float32)   # same noise in mV for every lead
    config.XAI_DIR.mkdir(parents=True, exist_ok=True)

    keys = config.ALL_DATASETS if args.dataset == "all" else [args.dataset]
    keys = [k for k in keys if (config.EVAL_DIR / f"predictions_{k}.csv").exists() or args.dataset != "all"]
    lines = ["# Attribution: where does the model look?\n",
             f"Model: **{primary['name']}** (primary). Integrated Gradients ({config.XAI_IG_STEPS} steps, "
             f"zero baseline) and Grad-CAM (last conv stage). First/last {config.EDGE_MARGIN_S} s "
             f"left out. These show where the model looks, not what it needs; occlusion "
             f"tests that.\n"]
    results = {}
    for key in keys:
        sel = select(key, args.n_per_class, primary)
        res, examples = explain(key, sel, explainer, noise_std, args.batch)
        res.to_csv(config.XAI_DIR / f"attributions_{key}{tag}.csv", index=False)
        results[key] = res
        lines += report_dataset(key, res)
        if examples:
            p = plot_examples(key, examples, mean, std, tag)
            lines.append(f"Example plots: `{p.relative_to(config.ROOT)}`\n")

    externals = [k for k in results if k != "ptbxl"]
    if "ptbxl" in results and externals:
        lines.append("## PTB-XL vs external sets (does the attention replicate across hospitals?)\n")
        a = results["ptbxl"][results["ptbxl"]["grp_af_random"]]["ig_atrial_share"].dropna()
        for k in externals:
            b = results[k][results[k]["grp_af_random"]]["ig_atrial_share"].dropna()
            rng = np.random.default_rng(config.SEED)
            d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(2000)]
            lines.append(f"- IG atrial share in random AF ECGs: PTB-XL {a.mean():.3f}, "
                         f"{config.DATASET_NAMES[k]} {b.mean():.3f}; difference {a.mean() - b.mean():+.3f} "
                         f"(95% CI {np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f})")
        lines.append("")

    text = "\n".join(lines)
    suffix = tag + ("" if args.dataset == "all" else f"_{args.dataset}")
    (config.RESULTS_DIR / f"07_attribution_report{suffix}.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
