"""Step 4: RR-feature logistic regression baseline.

A simple model that only sees rhythm (RR features), used as a comparator. If the deep model clearly beats it,
the deep model is using more than RR irregularity.

Trains on the PTB-XL training split, chooses the threshold on the
validation split, and stops there. The test split and Chapman are NOT
touched here; they are scored once by the evaluation script.

Run from the project folder, after 03_rpeaks.py:
    python scripts/04_rr_baseline.py

Outputs (results/models/rr_baseline/):
    model.pkl            fitted scaler + logistic regression
    threshold.json       locked threshold, chosen on validation
    val_predictions.csv  validation probabilities
    coefficients.csv     which RR features matter most
and results/04_rr_baseline_report.md
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import config  # noqa: E402
from src.metrics import choose_threshold, summarise  # noqa: E402


def main() -> None:
    rr_csv = config.PROCESSED_DIR / "ptbxl_rr.csv"
    if not rr_csv.exists():
        sys.exit("ptbxl_rr.csv not found. Run scripts/03_rpeaks.py --dataset ptbxl first.")
    rr = pd.read_csv(rr_csv, dtype={"record_id": str})
    rr = rr[rr["label"].isin([0, 1])].dropna(subset=config.RR_FEATURES)

    train, val = rr[rr["split"] == "train"], rr[rr["split"] == "val"]
    X_tr, y_tr = train[config.RR_FEATURES].to_numpy(), train["label"].to_numpy()
    X_va, y_va = val[config.RR_FEATURES].to_numpy(), val["label"].to_numpy()

    # class_weight="balanced" matches the class-weighted loss used for the ResNet
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=2000,
                                             random_state=config.SEED))
    model.fit(X_tr, y_tr)

    p_va = model.predict_proba(X_va)[:, 1]
    thr = choose_threshold(y_va, p_va)
    val_metrics = summarise(y_va, p_va, thr)

    out = config.MODELS_DIR / "rr_baseline"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    (out / "threshold.json").write_text(json.dumps(
        {"threshold": thr, "rule": config.THRESHOLD_RULE, "chosen_on": "ptbxl val"}, indent=2))
    pd.DataFrame({"record_id": val["record_id"], "label": y_va, "prob": p_va}).to_csv(
        out / "val_predictions.csv", index=False)

    coefs = model[-1].coef_[0]          # per standard deviation of each feature
    coef_df = pd.DataFrame({"feature": config.RR_FEATURES, "coef_per_sd": coefs})
    coef_df = coef_df.reindex(coef_df["coef_per_sd"].abs().sort_values(ascending=False).index)
    coef_df.to_csv(out / "coefficients.csv", index=False)

    m = val_metrics
    lines = ["# RR-feature baseline (logistic regression)\n",
             f"Trained on PTB-XL train ({len(train)} ECGs, {int(y_tr.sum())} AF). "
             f"Threshold chosen on PTB-XL val by {config.THRESHOLD_RULE} and locked.",
             "Test split and Chapman not used here.\n",
             "## Validation results\n",
             "| AUROC | AUPRC | Threshold | Sensitivity | Specificity | PPV | NPV | F1 |",
             "|---|---|---|---|---|---|---|---|",
             f"| {m['auroc']:.3f} | {m['auprc']:.3f} | {m['threshold']:.3f} | "
             f"{m['sensitivity']:.3f} | {m['specificity']:.3f} | {m['ppv']:.3f} | "
             f"{m['npv']:.3f} | {m['f1']:.3f} |",
             f"\nConfusion (val): TP {m['tp']}, FN {m['fn']}, FP {m['fp']}, TN {m['tn']}\n",
             "## Feature weights (per standard deviation)\n",
             "| Feature | Weight |", "|---|---|"]
    lines += [f"| {r.feature} | {r.coef_per_sd:+.3f} |" for r in coef_df.itertuples()]
    if "regular_af" in val:
        reg = val["regular_af"].astype(bool)
        if reg.any():
            pred = p_va >= thr
            lines.append(f"\nLow-RR-variability AF in val: {int(reg.sum())} ECGs, "
                         f"baseline catches {int(pred[reg.to_numpy()].sum())} of them "
                         f"(rhythm-only models should struggle here).")
    text = "\n".join(lines) + "\n"
    (config.RESULTS_DIR / "04_rr_baseline_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
