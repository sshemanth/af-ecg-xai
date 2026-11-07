"""Step 6: evaluate on the PTB-XL test split and Chapman-Shaoxing.

  - locked models and thresholds (chosen on PTB-XL validation only)
  - AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1, confusion matrix
  - 95% CIs from a patient-level bootstrap
  - calibration: plot, intercept, slope, Brier score, ECE
  - internal (PTB-XL test) and external (Chapman) reported separately
  - paired comparison with the RR-only baseline on the same ECGs
  - subgroups: low-RR-variability AF and near-regular AF

The test split is meant to be used ONCE. This script records each
evaluation in results/eval/LOCK_<dataset>.json and refuses to run on that
dataset again unless you add --force. Chapman is treated the same way.

Run from the project folder, after training all seeds:
    python scripts/06_evaluate.py                     # PTB-XL test + Chapman (if ready)
    python scripts/06_evaluate.py --dataset chapman   # one dataset only
    python scripts/06_evaluate.py --dry-run           # check everything is ready, touches no test data
It stops if not all seeds are trained (add --partial to override).

Outputs: results/eval/ (predictions, calibration plots, lock files) and
results/06_evaluation_report.md
"""
import argparse
import datetime as dt
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
from src.runs import find_resnet_runs, load_model, primary_run  # noqa: E402

DATASETS = {"ptbxl": {"name": "PTB-XL test (internal)", "splits": ["test"]},
            "chapman": {"name": "Chapman-Shaoxing (external)", "splits": ["external"]}}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@torch.no_grad()
def resnet_predict(run: dict, ds: ECGDataset, device) -> np.ndarray:
    model = load_model(run, device)
    out = []
    for x, _ in DataLoader(ds, batch_size=128, shuffle=False):
        out.append(torch.sigmoid(model(x.to(device))).float().cpu().numpy())
    return np.concatenate(out)


def baseline_predict(rr: pd.DataFrame) -> tuple[np.ndarray, float]:
    d = config.MODELS_DIR / "rr_baseline"
    with open(d / "model.pkl", "rb") as f:
        model = pickle.load(f)
    thr = json.loads((d / "threshold.json").read_text())["threshold"]
    p = np.full(len(rr), np.nan)
    ok = rr[config.RR_FEATURES].notna().all(axis=1).to_numpy()
    if ok.any():
        p[ok] = model.predict_proba(rr.loc[ok, config.RR_FEATURES].to_numpy())[:, 1]
    return p, thr


# ---------------------------------------------------------------------------
# Lock: the test data is used once
# ---------------------------------------------------------------------------
def check_lock(key: str, force: bool) -> None:
    lock = config.EVAL_DIR / f"LOCK_{key}.json"
    if lock.exists() and not force:
        info = json.loads(lock.read_text())
        sys.exit(f"{DATASETS[key]['name']} was already evaluated on {info['when']} "
                 f"(models: {', '.join(info['models'])}).\nThe test data is meant to be used once. "
                 f"If you really need to re-run (e.g. a bug in this script, not a model change), "
                 f"add --force and write down why.")


def write_lock(key: str, models: list[str], force: bool) -> None:
    lock = config.EVAL_DIR / f"LOCK_{key}.json"
    history = json.loads(lock.read_text()).get("history", []) if lock.exists() else []
    entry = {"when": dt.datetime.now().isoformat(timespec="seconds"), "models": models,
             "forced": force}
    lock.write_text(json.dumps({**entry, "history": history + [entry]}, indent=2))


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


# ---------------------------------------------------------------------------
def evaluate_dataset(key: str, runs: list[dict], primary: dict, device, args) -> dict:
    spec = DATASETS[key]
    ds = ECGDataset(key, spec["splits"], train=False)
    idx = ds.index[["record_id", "patient_id", "label"]].copy()
    rr = pd.read_csv(config.PROCESSED_DIR / f"{key}_rr.csv", dtype={"record_id": str})
    idx = idx.merge(rr[["record_id", "regular_af", "regular_af_strict"] + config.RR_FEATURES],
                    on="record_id", how="left")
    y, pats = idx["label"].to_numpy().astype(int), idx["patient_id"].to_numpy()

    preds = {}
    for run in runs:
        print(f"  predicting {run['name']} on {key} ({len(ds)} ECGs)...")
        preds[run["name"]] = resnet_predict(run, ds, device)
    p_base, thr_base = baseline_predict(idx)

    out = idx[["record_id", "patient_id", "label", "regular_af", "regular_af_strict"]].copy()
    for name, p in preds.items():
        out[f"prob_{name}"] = p
    out["prob_rr_baseline"] = p_base
    out.to_csv(config.EVAL_DIR / f"predictions_{key}.csv", index=False)

    res = {"key": key, "n": len(y), "n_af": int(y.sum()), "lines": []}
    L = res["lines"]
    L += [f"## {spec['name']}\n", f"{len(y)} ECGs, {int(y.sum())} AF "
          f"({100 * y.mean():.1f}%), {len(np.unique(pats))} patients.\n"]

    # main metrics with CIs: primary ResNet and RR baseline
    p_prim = preds[primary["name"]]
    m_prim = summarise(y, p_prim, primary["threshold"])
    ci_prim = bootstrap_ci(y, p_prim, pats, primary["threshold"], args.n_boot)
    ok = ~np.isnan(p_base)
    m_base = summarise(y[ok], p_base[ok], thr_base)
    ci_base = bootstrap_ci(y[ok], p_base[ok], pats[ok], thr_base, args.n_boot)
    L += ["### Performance (95% CI, patient-level bootstrap)\n"]
    L += metrics_table([(f"ResNet ({primary['name']}, primary)", m_prim, ci_prim),
                        ("RR-only baseline", m_base, ci_base)])
    if (~ok).any():
        L.append(f"\nRR baseline could not score {int((~ok).sum())} ECGs (no R-peaks); "
                 f"its row uses the other {int(ok.sum())}.")
    L.append(f"\nConfusion, ResNet primary: TP {m_prim['tp']}, FN {m_prim['fn']}, "
             f"FP {m_prim['fp']}, TN {m_prim['tn']} (threshold {primary['threshold']:.3f})")
    L.append(f"Confusion, RR baseline: TP {m_base['tp']}, FN {m_base['fn']}, "
             f"FP {m_base['fp']}, TN {m_base['tn']} (threshold {thr_base:.3f})\n")

    # all seeds
    rows = []
    for run in runs:
        m = summarise(y, preds[run["name"]], run["threshold"])
        rows.append({"seed": run["seed"], **m})
    sd = pd.DataFrame(rows)
    L += ["### All seeds (reliability)\n", "| Seed | AUROC | AUPRC | Sens | Spec |", "|---|---|---|---|---|"]
    L += [f"| {r.seed} | {r.auroc:.4f} | {r.auprc:.4f} | {r.sensitivity:.3f} | {r.specificity:.3f} |"
          for r in sd.itertuples()]
    if len(sd) > 1:
        L.append(f"\nMean ± SD over {len(sd)} seeds: AUROC {sd.auroc.mean():.4f} ± {sd.auroc.std():.4f}, "
                 f"AUPRC {sd.auprc.mean():.4f} ± {sd.auprc.std():.4f}\n")

    # paired comparison with the baseline (same ECGs)
    cmp_ = paired_auroc_diff(y[ok], p_prim[ok], p_base[ok], pats[ok], args.n_boot)
    L += ["### ResNet vs RR-only baseline (same ECGs, paired bootstrap)\n",
          f"AUROC difference: **{cmp_['diff']:+.4f}** (95% CI {cmp_['ci'][0]:+.4f} to "
          f"{cmp_['ci'][1]:+.4f}); share of bootstrap draws ≤ 0: {cmp_['p_le_0']:.3f}\n"]

    # calibration
    cal = {"ResNet": calibration(y, p_prim), "RR baseline": calibration(y[ok], p_base[ok])}
    L += ["### Calibration\n", "| Model | Intercept (0 = ideal) | Slope (1 = ideal) | Brier | ECE |",
          "|---|---|---|---|---|"]
    L += [f"| {n} | {c['intercept']:+.3f} | {c['slope']:.3f} | {c['brier']:.4f} | {c['ece']:.4f} |"
          for n, c in cal.items()]
    L.append("")
    for n, c in cal.items():
        level = ("too high overall" if c["intercept"] < -0.1 else
                 "too low overall" if c["intercept"] > 0.1 else "about right overall")
        spread = ("too extreme (overconfident)" if c["slope"] < 0.8 else
                  "too cautious (bunched together)" if c["slope"] > 1.25 else "about right")
        L.append(f"- {n}: probabilities are {level}; their spread is {spread}.")
    L.append("\nThe ResNet uses a class-weighted loss, which tends to push probabilities up. "
             "Calibration only affects how far the probabilities can be taken at face value; "
             "ranking (AUROC) and the locked-threshold results are unaffected.\n")
    plot_calibration({"ResNet": (y, p_prim, cal["ResNet"]["slope"]),
                      "RR baseline": (y[ok], p_base[ok], cal["RR baseline"]["slope"])},
                     spec["name"], config.EVAL_DIR / f"calibration_{key}.png")
    L.append(f"Plot: `results/eval/calibration_{key}.png`\n")

    # subgroups: how many of these AF ECGs each model catches
    L += ["### Subgroups — AF detected (sensitivity, 95% Wilson CI)\n",
          "| Group | AF ECGs | ResNet | RR baseline |", "|---|---|---|---|"]
    groups = [("All AF", y == 1),
              ("Low RR variability AF (main rule)", (y == 1) & idx["regular_af"].fillna(False).astype(bool).to_numpy()),
              ("Near-regular AF (strict rule)", (y == 1) & idx["regular_af_strict"].fillna(False).astype(bool).to_numpy())]
    for gname, mask in groups:
        n = int(mask.sum())
        if n == 0:
            L.append(f"| {gname} | 0 | – | – |")
            continue
        k_r = int((p_prim[mask] >= primary["threshold"]).sum())
        mb = mask & ok
        k_b = int((p_base[mb] >= thr_base).sum())
        lo_r, hi_r = wilson_ci(k_r, n)
        lo_b, hi_b = wilson_ci(k_b, int(mb.sum()))
        L.append(f"| {gname} | {n} | {k_r}/{n} = {k_r / n:.3f} ({lo_r:.3f}–{hi_r:.3f}) | "
                 f"{k_b}/{int(mb.sum())} = {k_b / max(mb.sum(), 1):.3f} ({lo_b:.3f}–{hi_b:.3f}) |")
    L.append("\nIf the ResNet keeps catching AF where RR irregularity is weak, while the RR "
             "baseline drops, that points to atrial morphology; the attribution and "
             "occlusion steps test this directly.\n")

    res["auroc_draws"] = auroc_draws(y, p_prim, pats, args.n_boot)
    res["auroc"] = m_prim["auroc"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ptbxl", "chapman", "all"], default="all")
    ap.add_argument("--n-boot", type=int, default=config.N_BOOTSTRAP)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true", help="re-run on already-used test data")
    ap.add_argument("--dry-run", action="store_true", help="check readiness only")
    ap.add_argument("--partial", action="store_true",
                    help="evaluate even if not every seed in config.RUN_SEEDS is trained")
    args = ap.parse_args()

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    runs = find_resnet_runs()
    if not runs:
        sys.exit("No trained ResNet runs found. Run scripts/05_train.py first.")
    if not (config.MODELS_DIR / "rr_baseline" / "model.pkl").exists():
        sys.exit("RR baseline not found. Run scripts/04_rr_baseline.py first.")
    primary = primary_run(runs)

    keys = ["ptbxl", "chapman"] if args.dataset == "all" else [args.dataset]
    ready = []
    for k in keys:
        need = [config.PROCESSED_DIR / f"{k}_index.csv", config.PROCESSED_DIR / f"{k}_rr.csv"]
        if all(p.exists() for p in need):
            ready.append(k)
        else:
            print(f"[skip] {k}: not preprocessed yet (needs steps 01–03)")

    print(f"ResNet runs: {', '.join(r['name'] for r in runs)}")
    print(f"Primary model ({config.PRIMARY_MODEL_RULE}): {primary['name']} "
          f"(val AUPRC {primary['val_auprc']:.4f})")
    print(f"Datasets ready: {', '.join(ready) or 'none'}")
    missing = len(config.RUN_SEEDS) - len(runs)
    if missing > 0:
        print(f"Only {len(runs)} of {len(config.RUN_SEEDS)} seeds trained.")
    if args.dry_run or not ready:
        if missing > 0:
            print("Train them all before evaluating, since the test data is used once.")
        return
    if missing > 0 and not args.partial:
        sys.exit("Stopping: the test data is used once, so train all seeds first "
                 "(python scripts/05_train.py --seed all). To evaluate with fewer seeds "
                 "anyway, add --partial.")
    for k in ready:
        check_lock(k, args.force)

    device = pick_device(args.device)
    lines = ["# Evaluation on held-out data\n",
             f"Models and thresholds locked on PTB-XL validation. Primary ResNet = "
             f"**{primary['name']}** ({config.PRIMARY_MODEL_RULE}, fixed before testing). "
             f"Seeds evaluated: {len(runs)}. Bootstrap draws: {args.n_boot}.\n"]
    results = {}
    for k in ready:
        results[k] = evaluate_dataset(k, runs, primary, device, args)
        lines += results[k]["lines"]
        write_lock(k, [r["name"] for r in runs] + ["rr_baseline"], args.force)

    if "ptbxl" in results and "chapman" in results:
        a, b = results["ptbxl"], results["chapman"]
        lo, hi = independent_diff_ci(a["auroc_draws"], b["auroc_draws"])
        lines += ["## Internal vs external\n",
                  f"Primary ResNet AUROC: PTB-XL test {a['auroc']:.4f}, Chapman {b['auroc']:.4f}. "
                  f"Drop: **{a['auroc'] - b['auroc']:+.4f}** (95% CI {lo:+.4f} to {hi:+.4f}).\n",
                  "Differences in calibration between the two are in the tables above; "
                  "a shift there can mean a different AF rate or recording setup, not only "
                  "a weaker model.\n"]

    text = "\n".join(lines)
    (config.RESULTS_DIR / "06_evaluation_report.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
