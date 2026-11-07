"""Load label tables for PTB-XL and Chapman-Shaoxing.

Both loaders return one row per recording with the same columns, so later
steps can treat the two datasets the same way:

    record_id, patient_id, path, dataset, age, sex,
    af, aflutter, paced, label, split
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.splits import assign_splits


def _apply_flutter_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary 'label' column (1 = AF) using FLUTTER_POLICY."""
    df = df.copy()
    flutter_only = df["aflutter"] & ~df["af"]
    df["label"] = df["af"].astype(int)
    policy = config.FLUTTER_POLICY
    if policy == "exclude":
        df.loc[flutter_only, "label"] = -1          # -1 = excluded
    elif policy == "positive":
        df.loc[flutter_only, "label"] = 1
    elif policy == "negative":
        pass                                         # already 0
    else:
        raise ValueError(f"Unknown FLUTTER_POLICY: {policy}")
    return df


# ---------------------------------------------------------------------------
# PTB-XL
# ---------------------------------------------------------------------------
def load_ptbxl(ptbxl_dir: Path = config.PTBXL_DIR) -> pd.DataFrame:
    db_path = ptbxl_dir / "ptbxl_database.csv"
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} not found. Run: bash scripts/download_data.sh")

    db = pd.read_csv(db_path)
    codes = db["scp_codes"].apply(ast.literal_eval)   # string -> dict

    df = pd.DataFrame({
        "record_id": db["ecg_id"].astype(str),
        "patient_id": db["patient_id"].astype("Int64").astype(str),
        "path": db["filename_hr"].apply(lambda p: str(ptbxl_dir / p)),  # 500 Hz
        "dataset": "ptbxl",
        "age": db["age"],
        "sex": db["sex"].map({0: "M", 1: "F"}),       # PTB-XL: 0 = male, 1 = female
        "af": codes.apply(lambda c: config.PTBXL_AF in c),
        "aflutter": codes.apply(lambda c: config.PTBXL_AFL in c),
        "paced": codes.apply(lambda c: config.PTBXL_PACE in c),
        "sinus": codes.apply(lambda c: config.PTBXL_SR in c),
        # PTB-XL's documented recording format (checked again when the
        # signals are loaded in preprocessing)
        "fs": 500,
        "n_samples": 5000,
        # Kept only for reference; the split comes from src/splits.py
        "ptbxl_strat_fold": db["strat_fold"],
    })

    df = _apply_flutter_policy(df)
    df["split"] = assign_splits(df)
    return df


# ---------------------------------------------------------------------------
# External sets: Chapman-Shaoxing and Georgia (PhysioNet/CinC Challenge 2021)
# ---------------------------------------------------------------------------
def _parse_header(hea_path: Path) -> dict:
    """Read sampling rate, length, lead count, age, sex and diagnosis codes
    from a Challenge-style .hea file."""
    info = {"fs": np.nan, "n_samples": np.nan, "n_leads": np.nan,
            "age": np.nan, "sex": None, "dx": []}
    lines = hea_path.read_text().splitlines()
    # First line: record_name  n_leads  fs  n_samples ...
    first = lines[0].split() if lines else []
    if len(first) >= 4:
        info["n_leads"] = int(first[1])
        info["fs"] = float(first[2].split("/")[0])
        info["n_samples"] = int(first[3])
    for line in lines:
        # Comment lines can be written "#Dx: ..." or "# Dx: ...", so strip
        # the '#' and any spaces before looking at the key.
        if not line.startswith("#") or ":" not in line:
            continue
        key, val = line.lstrip("#").split(":", 1)
        key, val = key.strip().lower(), val.strip()
        if key == "age":
            try:
                info["age"] = float(val)
            except ValueError:
                pass                                   # "NaN" or blank
        elif key == "sex":
            info["sex"] = {"male": "M", "female": "F"}.get(val.lower())
        elif key == "dx":
            info["dx"] = [c.strip() for c in val.split(",") if c.strip()]
    return info


def load_challenge(key: str, folder: Path | None = None) -> pd.DataFrame:
    """Label table for one Challenge 2021 dataset ('chapman' or 'georgia')."""
    folder = folder or config.CHALLENGE_DIRS[key]
    headers = sorted(folder.rglob("*.hea"))
    if not headers:
        raise FileNotFoundError(
            f"No .hea files in {folder}. Run: bash scripts/download_challenge.sh {key}")

    rows = []
    for hea in headers:
        info = _parse_header(hea)
        dx = set(info["dx"])
        rows.append({
            "record_id": hea.stem,
            # The Challenge copies carry no patient IDs. Chapman-Shaoxing has
            # one ECG per patient; for Georgia each record is treated as its
            # own patient (a limitation for the patient-level bootstrap).
            "patient_id": hea.stem,
            "path": str(hea.with_suffix("")),
            "dataset": key,
            "age": info["age"],
            "sex": info["sex"],
            "af": config.SNOMED_AF in dx,
            "aflutter": config.SNOMED_AFL in dx,
            "paced": config.SNOMED_PACE in dx,
            "sinus": config.SNOMED_SR in dx,
            "n_dx": len(dx),
            "fs": info["fs"],
            "n_samples": info["n_samples"],
            "n_leads": info["n_leads"],
        })

    df = pd.DataFrame(rows)
    df["split"] = "external"        # never used for training or tuning
    return _apply_flutter_policy(df)


def load_chapman(folder: Path | None = None) -> pd.DataFrame:
    return load_challenge("chapman", folder)


def load_georgia(folder: Path | None = None) -> pd.DataFrame:
    return load_challenge("georgia", folder)
