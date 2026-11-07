"""Which held-out ECGs the attribution and occlusion steps analyse.

A class-balanced random sample (AF and non-AF), plus every
low-RR-variability, near-regular and paced AF ECG, all from the held-out
sets (PTB-XL test, Chapman). Both steps use the same selection, so their
results can be compared ECG by ECG.
"""
import numpy as np
import pandas as pd

from src import config

DATASETS = {"ptbxl": ("PTB-XL test", ["test"]), "chapman": ("Chapman-Shaoxing", ["external"])}
GROUPS = [("AF (random sample)", "grp_af_random"),
          ("Non-AF (random sample)", "grp_non_random"),
          ("Low-RR-variability AF", "regular_af"),
          ("Near-regular AF", "regular_af_strict"),
          ("Paced AF", "paced_af")]


def held_out_table(key: str, primary: dict) -> pd.DataFrame:
    """Every held-out ECG with labels, subgroup flags, R-peaks and the
    primary model's prediction (from the evaluation step)."""
    pred_f = config.EVAL_DIR / f"predictions_{key}.csv"
    if not pred_f.exists():
        raise SystemExit(f"{pred_f} not found. Run scripts/06_evaluate.py first.")
    pred = pd.read_csv(pred_f, dtype={"record_id": str, "patient_id": str})
    pcol = f"prob_{primary['name']}"
    if pcol not in pred:
        raise SystemExit(f"{pcol} not in {pred_f.name}; was the evaluation run with this model?")
    idx = pd.read_csv(config.PROCESSED_DIR / f"{key}_index.csv", dtype={"record_id": str})
    rr = pd.read_csv(config.PROCESSED_DIR / f"{key}_rr.csv", dtype={"record_id": str})
    labels = pd.read_csv(config.PROCESSED_DIR / f"{key}_labels.csv", dtype={"record_id": str})
    df = (pred[["record_id", "label", "regular_af", "regular_af_strict", pcol]]
          .rename(columns={pcol: "prob"})
          .merge(idx[["record_id", "paced"]], on="record_id", how="left")
          .merge(labels[["record_id", "sinus"]], on="record_id", how="left")
          .merge(rr[["record_id", "rpeaks", "rr_cv"]], on="record_id", how="left"))
    for c in ["regular_af", "regular_af_strict", "paced", "sinus"]:
        df[c] = df[c].fillna(False).astype(bool)
    df["paced_af"] = df["paced"] & (df["label"] == 1)
    df["pred_af"] = df["prob"] >= primary["threshold"]
    df["correct"] = df["pred_af"] == (df["label"] == 1)
    return df[df["rpeaks"].notna()].reset_index(drop=True)


def select(key: str, n_per_class: int, primary: dict) -> pd.DataFrame:
    """The ECGs to analyse, with group flags, in a fixed random order."""
    df = held_out_table(key, primary)
    rng = np.random.default_rng(config.SEED)
    df["grp_af_random"] = False
    df["grp_non_random"] = False
    for lab, col in [(1, "grp_af_random"), (0, "grp_non_random")]:
        pool = df.index[df["label"] == lab].to_numpy()
        pick = rng.choice(pool, size=min(n_per_class, len(pool)), replace=False)
        df.loc[pick, col] = True
    keep = df[[c for _, c in GROUPS]].any(axis=1)
    df = df[keep].reset_index(drop=True)
    # random order, so any "first N" subset is a fair mix
    return df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
