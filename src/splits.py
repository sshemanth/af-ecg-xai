"""Patient-level, stratified train/val/test split for PTB-XL.

- stratified by AF status and, where metadata permit, paced rhythm
- performed at the patient level, so all of a patient's ECGs land in
  one partition only
- fixed, reproducible random seed

Chapman-Shaoxing is never split. It stays as the external test set.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src import config


def stratum(df: pd.DataFrame, n_folds: int = config.SPLIT_N_FOLDS) -> pd.Series:
    """One stratum per combination of AF label and paced flag.

    The label is 1 (AF), 0 (non-AF) or -1 (flutter-only, excluded later).
    Keeping the excluded rows in the split means the same patient's
    other ECGs still go to the right partition.

    A combination with fewer ECGs than folds (e.g. AF + paced in PTB-XL,
    only 4 cases) can't be spread across folds, so it joins the
    non-paced group with the same AF label.
    """
    label = df["label"].astype(str)
    combo = label + "_" + df["paced"].astype(int).astype(str)
    sizes = combo.map(combo.value_counts())
    return combo.where(sizes >= n_folds, label + "_0")


def assign_splits(df: pd.DataFrame,
                  n_folds: int = config.SPLIT_N_FOLDS,
                  seed: int = config.SEED) -> pd.Series:
    """Return a 'split' column: train / val / test.

    The data is cut into n_folds patient-grouped, stratified folds.
    One fold becomes the test set, one the validation set, and the
    rest are training. With 10 folds this gives roughly 80/10/10.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1)
    for i, (_, idx) in enumerate(sgkf.split(df, y=stratum(df), groups=df["patient_id"])):
        fold[idx] = i

    split = pd.Series("train", index=df.index)
    split[fold == config.SPLIT_TEST_FOLD] = "test"
    split[fold == config.SPLIT_VAL_FOLD] = "val"
    return split
