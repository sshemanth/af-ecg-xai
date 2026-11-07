"""Step 6: evaluate on the PTB-XL test split and the external sets.

  - locked models and thresholds (chosen on PTB-XL validation only)
  - AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1, confusion matrix
  - 95% CIs from a patient-level bootstrap
  - calibration: plot, intercept, slope, Brier score, ECE
  - internal (PTB-XL test) and each external set reported separately
  - paired comparisons on the same ECGs: each network vs the RR-only
    baseline, and the second architecture vs the ResNet
  - subgroups: low-RR-variability AF and near-regular AF
Every trained architecture is evaluated (ResNet, and InceptionTime once
trained); each has its own primary seed (best validation AUPRC).

The held-out data is meant to be used ONCE. Each evaluation is recorded in
results/eval/LOCK_<dataset>.json with a fingerprint (SHA-256) of every
model file. A later run may add new models or datasets, and may re-score
identical models (that reveals nothing new), but it refuses to score a
model whose file changed after it was tested, unless you add --force.

Run from the project folder, after training all seeds:
    python scripts/06_evaluate.py                     # every dataset that is ready
    python scripts/06_evaluate.py --dataset georgia   # one dataset only
    python scripts/06_evaluate.py --dry-run           # check readiness, touches no test data
It stops if an architecture has some but not all seeds trained (add
--partial to override).

Outputs: results/eval/ (predictions, calibration plots, lock files) and
results/06_evaluation_report.md
"""
import argparse
import datetime as dt
import hashlib
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src import config  # noqa: E402
from src.data import ECGDataset  # noqa: E402
from src.evaluation import (auroc_draws, bootstrap_ci, calibration,  # noqa: E402
                            independent_diff_ci, paired_auroc_diff, wilson_ci)
from src.metrics import summarise  # noqa: E402
from src.model import pick_device  # noqa: E402
from src.plots import plot_calibration  # noqa: E402
from src.runs import arch_label, find_runs, load_model, primary_run  # noqa: E402

DATASETS = {k: {"name": config.DATASET_NAMES[k] + (" test (internal)" if k == "ptbxl" else " (external)"),
                "splits": config.HELD_OUT_SPLITS[k]} for k in config.ALL_DATASETS}
BASELINE_DIR = config.MODELS_DIR / "rr_baseline"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@torch.no_grad()
def net_predict(run: dict, ds: ECGDataset, device) -> np.ndarray:
    model = load_model(run, device)
    out = []
    for x, _ in DataLoader(ds, batch_size=128, shuffle=False):
        out.append(torch.sigmoid(model(x.to(device))).float().cpu().numpy())
    return np.concatenate(out)


def baseline_predict(rr: pd.DataFrame) -> tuple[np.ndarray, float]:
    with open(BASELINE_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    thr = json.loads((BASELINE_DIR / "threshold.json").read_text())["threshold"]
    p = np.full(len(rr), np.nan)
    ok = rr[config.RR_FEATURES].notna().all(axis=1).to_numpy()
    if ok.any():
        p[ok] = model.predict_proba(rr.loc[ok, config.RR_FEATURES].to_numpy())[:, 1]
    return p, thr


# ---------------------------------------------------------------------------
# Lock: the held-out data is used once per model
# ---------------------------------------------------------------------------
def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def model_hashes(runs: list[dict]) -> dict:
    out = {r["name"]: fingerprint(r["dir"] / "model.pt") for r in runs}
    out["rr_baseline"] = fingerprint(BASELINE_DIR / "model.pkl")
    return out


def check_lock(key: str, hashes: dict, force: bool) -> tuple[list, list]:
    """Returns (new models, re-scored models). Stops if a tested model changed."""
    lock = config.EVAL_DIR / f"LOCK_{key}.json"
    if not lock.exists():
        return list(hashes), []
    info = json.loads(lock.read_text())
    seen = info.get("hashes")
    if seen is None:
        # lock written before fingerprints were kept: the models listed there
        # are taken to be the current files
        seen = {m: hashes.get(m) for m in info.get("models", [])}
    changed = [m for m, h in hashes.items() if m in seen and seen[m] not in (None, h)]
    if changed and not force:
        sys.exit(f"{DATASETS[key]['name']}: these models changed after they were tested: "
                 f"{', '.join(changed)}.\nThe held-out data is meant to be used once per model. "
                 f"If you really need this (e.g. a bug fix, not tuning), add --force and write down why.")
    new = [m for m in hashes if m not in seen]
    return new, [m for m in hashes if m in seen]


def write_lock(key: str, hashes: dict, new: list, force: bool) -> None:
    lock = config.EVAL_DIR / f"LOCK_{key}.json"
    old = json.loads(lock.read_text()) if lock.exists() else {}
    known = {**old.get("hashes", {}), **hashes}
    entry = {"when": dt.datetime.now().isoformat(timespec="seconds"), "models": sorted(hashes),
             "new_models": sorted(new), "forced": force}
    lock.write_text(json.dumps({**entry, "hashes": known,
                                "history": old.get("history", []) + [entry]}, indent=2))


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------
def fmt_ci(v, ci):
    return f"{v:.3f} ({ci[0]:.3f}–{ci[1]:.3f})" if ci else f"{v:.3f}"


def metrics_table(rows: list[tuple[str, dict, dict]]) -> list[str]:
    out = ["| Model | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | F1 |",
           "|---|---|---|---|---|---|---|---|"]
    for name, m, ci in rows:
        out.append(f"| {name} | " + " | ".join(
            fmt_ci(m[k], ci.get(k)) for k in
            ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1"]) + " |")
    return out


def paired_line(label: str, y, pa, pb, pats, n_boot) -> str:
    c = paired_auroc_diff(y, pa, pb, pats, n_boot)
    return (f"| {label} | **{c['diff']:+.4f}** | {c['ci'][0]:+.4f} to {c['ci'][1]:+.4f} | "
            f"{c['p_le_0']:.3f} |")


# ---------------------------------------------------------------------------
def evaluate_dataset(key: str, runs_by_arch: dict, primaries: dict, device, args) -> dict:
    spec = DATASETS[key]
    ds = ECGDataset(key, spec["splits"], train=False)
    idx = ds.index[["record_id", "patient_id", "label"]].copy()
    rr = pd.read_csv(config.PROCESSED_DIR / f"{key}_rr.csv", dtype={"record_id": str})
    idx = idx.merge(rr[["record_id", "regular_af", "regular_af_strict"] + config.RR_FEATURES],
                    on="record_id", how="left")
    y, pats = idx["label"].to_numpy().astype(int), idx["patient_id"].to_numpy()

    preds = {}
    for runs in runs_by_arch.values():
        for run in runs:
            print(f"  predicting {run['name']} on {key} ({len(ds)} ECGs)...")
            preds[run["name"]] = net_predict(run, ds, device)
    p_base, thr_base = baseline_predict(idx)

    out = idx[["record_id", "patient_id", "label", "regular_af", "regular_af_strict"]].copy()
    for name, p in preds.items():
        out[f"prob_{name}"] = p
    out["prob_rr_baseline"] = p_base
    out.to_csv(config.EVAL_DIR / f"predictions_{key}.csv", index=False)

    res = {"key": key, "n": len(y), "n_af": int(y.sum()), "lines": [], "auroc": {}, "draws": {}}
    L = res["lines"]
    L += [f"## {spec['name']}\n", f"{len(y)} ECGs, {int(y.sum())} AF "
          f"({100 * y.mean():.1f}%), {len(np.unique(pats))} patients.\n"]

    # main metrics with CIs: each architecture's primary seed and the RR baseline
    ok = ~np.isnan(p_base)
    rows, conf = [], []
    for arch, prim in primaries.items():
        p = preds[prim["name"]]
        m = summarise(y, p, prim["threshold"])
        rows.append((f"{arch_label(arch)} ({prim['name']}, primary)", m,
                     bootstrap_ci(y, p, pats, prim["threshold"], args.n_boot)))
        conf.append(f"Confusion, {arch_label(arch)} primary: TP {m['tp']}, FN {m['fn']}, "
                    f"FP {m['fp']}, TN {m['tn']} (threshold {prim['threshold']:.3f})")
        res["auroc"][arch] = m["auroc"]
        res["draws"][arch] = auroc_draws(y, p, pats, args.n_boot)
    m_base = summarise(y[ok], p_base[ok], thr_base)
    rows.append(("RR-only baseline", m_base, bootstrap_ci(y[ok], p_base[ok], pats[ok], thr_base, args.n_boot)))
    L += ["### Performance (95% CI, patient-level bootstrap)\n"] + metrics_table(rows)
    if (~ok).any():
        L.append(f"\nRR baseline could not score {int((~ok).sum())} ECGs (no R-peaks); "
                 f"its row uses the other {int(ok.sum())}.")
    L += [""] + conf + [f"Confusion, RR baseline: TP {m_base['tp']}, FN {m_base['fn']}, "
                        f"FP {m_base['fp']}, TN {m_base['tn']} (threshold {thr_base:.3f})", ""]

    # all seeds
    L += ["### All seeds (reliability)\n", "| Network | Seed | AUROC | AUPRC | Sens | Spec |",
          "|---|---|---|---|---|---|"]
    summary = []
    for arch, runs in runs_by_arch.items():
        sd = pd.DataFrame([{"seed": r["seed"], **summarise(y, preds[r["name"]], r["threshold"])}
                           for r in runs])
        L += [f"| {arch_label(arch)} | {r.seed} | {r.auroc:.4f} | {r.auprc:.4f} | {r.sensitivity:.3f} | "
              f"{r.specificity:.3f} |" for r in sd.itertuples()]
        if len(sd) > 1:
            summary.append(f"{arch_label(arch)}, mean ± SD over {len(sd)} seeds: AUROC "
                           f"{sd.auroc.mean():.4f} ± {sd.auroc.std():.4f}, AUPRC "
                           f"{sd.auprc.mean():.4f} ± {sd.auprc.std():.4f}")
    L += [""] + summary + [""]

    # paired comparisons (same ECGs)
    L += ["### Paired comparisons (same ECGs, paired bootstrap)\n",
          "| Comparison | AUROC difference | 95% CI | Share of draws ≤ 0 |", "|---|---|---|---|"]
    for arch, prim in primaries.items():
        L.append(paired_line(f"{arch_label(arch)} − RR baseline", y[ok], preds[prim["name"]][ok],
                             p_base[ok], pats[ok], args.n_boot))
    if "resnet" in primaries:
        for arch, prim in primaries.items():
            if arch != "resnet":
                L.append(paired_line(f"{arch_label(arch)} − ResNet", y, preds[prim["name"]],
                                     preds[primaries["resnet"]["name"]], pats, args.n_boot))
    L.append("")

    # calibration
    cal = {arch_label(a): (y, preds[p["name"]]) for a, p in primaries.items()}
    cal["RR baseline"] = (y[ok], p_base[ok])
    stats = {n: calibration(yy, pp) for n, (yy, pp) in cal.items()}
    L += ["### Calibration\n", "| Model | Intercept (0 = ideal) | Slope (1 = ideal) | Brier | ECE |",
          "|---|---|---|---|---|"]
    L += [f"| {n} | {c['intercept']:+.3f} | {c['slope']:.3f} | {c['brier']:.4f} | {c['ece']:.4f} |"
          for n, c in stats.items()]
    L.append("")
    for n, c in stats.items():
        level = ("too high overall" if c["intercept"] < -0.1 else
                 "too low overall" if c["intercept"] > 0.1 else "about right overall")
        spread = ("too extreme (overconfident)" if c["slope"] < 0.8 else
                  "too cautious (bunched together)" if c["slope"] > 1.25 else "about right")
        L.append(f"- {n}: probabilities are {level}; their spread is {spread}.")
    L.append("\nThe networks use a class-weighted loss, which tends to push probabilities up. "
             "Calibration only affects how far the probabilities can be taken at face value; "
             "ranking (AUROC) and the locked-threshold results are unaffected.\n")
    plot_calibration({n: (yy, pp, stats[n]["slope"]) for n, (yy, pp) in cal.items()},
                     spec["name"], config.EVAL_DIR / f"calibration_{key}.png")
    L.append(f"Plot: `results/eval/calibration_{key}.png`\n")

    # subgroups: how many of these AF ECGs each model catches
    names = [arch_label(a) for a in primaries]
    L += ["### Subgroups — AF detected (sensitivity, 95% Wilson CI)\n",
          "| Group | AF ECGs | " + " | ".join(names) + " | RR baseline |",
          "|---|---|" + "---|" * (len(names) + 1)]
    groups = [("All AF", y == 1),
              ("Low RR variability AF (main rule)", (y == 1) & idx["regular_af"].fillna(False).astype(bool).to_numpy()),
              ("Near-regular AF (strict rule)", (y == 1) & idx["regular_af_strict"].fillna(False).astype(bool).to_numpy())]
    for gname, mask in groups:
        n = int(mask.sum())
        if n == 0:
            L.append(f"| {gname} | 0 | " + " | ".join("–" for _ in names) + " | – |")
            continue
        cells = []
        for prim in primaries.values():
            k = int((preds[prim["name"]][mask] >= prim["threshold"]).sum())
            lo, hi = wilson_ci(k, n)
            cells.append(f"{k}/{n} = {k / n:.3f} ({lo:.3f}–{hi:.3f})")
        mb = mask & ok
        k_b = int((p_base[mb] >= thr_base).sum())
        lo_b, hi_b = wilson_ci(k_b, int(mb.sum()))
        cells.append(f"{k_b}/{int(mb.sum())} = {k_b / max(mb.sum(), 1):.3f} ({lo_b:.3f}–{hi_b:.3f})")
        L.append(f"| {gname} | {n} | " + " | ".join(cells) + " |")
    L.append("\nIf the networks keep catching AF where RR irregularity is weak, while the RR "
             "baseline drops, that points to atrial morphology; the attribution and "
             "occlusion steps test this directly.\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=config.ALL_DATASETS + ["all"], default="all")
    ap.add_argument("--n-boot", type=int, default=config.N_BOOTSTRAP)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true", help="score models that changed after testing")
    ap.add_argument("--dry-run", action="store_true", help="check readiness only")
    ap.add_argument("--partial", action="store_true",
                    help="evaluate even if an architecture has only some of its seeds trained")
    args = ap.parse_args()

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    runs_by_arch = {a: r for a in config.ARCHS if (r := find_runs(a))}
    if "resnet" not in runs_by_arch:
        sys.exit("No trained ResNet runs found. Run scripts/05_train.py first.")
    if not (BASELINE_DIR / "model.pkl").exists():
        sys.exit("RR baseline not found. Run scripts/04_rr_baseline.py first.")
    primaries = {a: primary_run(r) for a, r in runs_by_arch.items()}
    all_runs = [r for runs in runs_by_arch.values() for r in runs]

    keys = config.ALL_DATASETS if args.dataset == "all" else [args.dataset]
    ready = []
    for k in keys:
        need = [config.PROCESSED_DIR / f"{k}_index.csv", config.PROCESSED_DIR / f"{k}_rr.csv"]
        if all(p.exists() for p in need):
            ready.append(k)
        else:
            print(f"[skip] {k}: not preprocessed yet (needs steps 01–03)")

    incomplete = []
    for a, runs in runs_by_arch.items():
        print(f"{arch_label(a)} runs: {', '.join(r['name'] for r in runs)}; primary "
              f"({config.PRIMARY_MODEL_RULE}): {primaries[a]['name']} (val AUPRC {primaries[a]['val_auprc']:.4f})")
        if len(runs) < len(config.RUN_SEEDS):
            incomplete.append(f"{arch_label(a)} ({len(runs)} of {len(config.RUN_SEEDS)} seeds)")
    print(f"Datasets ready: {', '.join(ready) or 'none'}")
    if incomplete:
        print(f"Not all seeds trained: {', '.join(incomplete)}")
    if args.dry_run or not ready:
        if incomplete:
            print("Train them all before evaluating, since the held-out data is used once per model.")
        return
    if incomplete and not args.partial:
        sys.exit("Stopping: train all seeds first (python scripts/05_train.py --arch <arch> --seed all). "
                 "To evaluate anyway, add --partial.")

    hashes = model_hashes(all_runs)
    plans = {k: check_lock(k, hashes, args.force) for k in ready}
    for k, (new, again) in plans.items():
        what = f"new: {', '.join(new) or 'none'}"
        if again:
            what += f"; re-scored, unchanged: {len(again)} models"
        print(f"{k}: {what}")

    device = pick_device(args.device)
    lines = ["# Evaluation on held-out data\n",
             "Models and thresholds locked on PTB-XL validation. Primary model of each network = "
             f"{config.PRIMARY_MODEL_RULE}, fixed before testing: "
             + ", ".join(f"**{p['name']}**" for p in primaries.values())
             + f". Bootstrap draws: {args.n_boot}.\n"]
    results = {}
    for k in ready:
        results[k] = evaluate_dataset(k, runs_by_arch, primaries, device, args)
        lines += results[k]["lines"]
        write_lock(k, hashes, plans[k][0], args.force)

    if "ptbxl" in results and len(results) > 1:
        lines += ["## Internal vs external\n",
                  "| Network | External set | AUROC internal | AUROC external | Drop (95% CI) |",
                  "|---|---|---|---|---|"]
        a = results["ptbxl"]
        for k in [k for k in results if k != "ptbxl"]:
            b = results[k]
            for arch in primaries:
                lo, hi = independent_diff_ci(a["draws"][arch], b["draws"][arch])
                lines.append(f"| {arch_label(arch)} | {config.DATASET_NAMES[k]} | {a['auroc'][arch]:.4f} | "
                             f"{b['auroc'][arch]:.4f} | **{a['auroc'][arch] - b['auroc'][arch]:+.4f}** "
                             f"({lo:+.4f} to {hi:+.4f}) |")
        lines += ["", "Differences in calibration are in the tables above; a shift there can mean "
                  "a different AF rate, rhythm mix or recording setup, not only a weaker model.\n"]

    text = "\n".join(lines)
    suffix = "" if args.dataset == "all" else f"_{args.dataset}"
    (config.RESULTS_DIR / f"06_evaluation_report{suffix}.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
