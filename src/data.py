"""PyTorch dataset over the preprocessed ECG files.

- Normalises each lead with the mean/std from the PTB-XL TRAINING split
  only (norm_stats.json), for every split and for Chapman.
- Training-only augmentation that keeps atrial morphology: a random 9 s
  crop (timing shift), whole-ECG amplitude scaling, and slow baseline
  wander. Nothing that would blur or remove P-waves or f-waves.
"""
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config


def load_norm_stats() -> tuple[np.ndarray, np.ndarray]:
    path = config.PROCESSED_DIR / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError("norm_stats.json not found. Run scripts/02_preprocess.py --dataset ptbxl")
    stats = json.loads(path.read_text())
    mean = np.asarray(stats["mean"], dtype=np.float32)[:, None]
    std = np.asarray(stats["std"], dtype=np.float32)[:, None]
    return mean, std


class ECGDataset(Dataset):
    def __init__(self, dataset: str, splits: list[str], train: bool = False,
                 limit: int | None = None, seed: int = config.SEED):
        index = pd.read_csv(config.PROCESSED_DIR / f"{dataset}_index.csv",
                            dtype={"record_id": str, "patient_id": str})
        index = index[index["split"].isin(splits) & index["label"].isin([0, 1])]
        if limit and limit < len(index):
            index = index.groupby("label").sample(frac=limit / len(index), random_state=seed)
        self.index = index.reset_index(drop=True)
        self.path = str(config.PROCESSED_DIR / f"{dataset}_signals.npy")
        self.mean, self.std = load_norm_stats()
        self.train = train
        self._sig = None           # opened lazily, once per worker process

    def __len__(self) -> int:
        return len(self.index)

    @property
    def labels(self) -> np.ndarray:
        return self.index["label"].to_numpy()

    def _signals(self):
        if self._sig is None:
            self._sig = np.load(self.path, mmap_mode="r")
        return self._sig

    def _augment(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        fs = config.TARGET_FS
        # random crop = small shift in timing
        crop = int(config.TRAIN_CROP_S * fs)
        start = rng.integers(0, x.shape[1] - crop + 1)
        x = x[:, start:start + crop]
        # whole-ECG amplitude scaling (same for all leads, keeps lead ratios)
        x = x * rng.uniform(*config.AUG_SCALE)
        # slow baseline wander, like breathing (well below P/f-wave frequencies)
        t = np.arange(x.shape[1]) / fs
        f = rng.uniform(0.05, 0.4)
        wander = config.AUG_BASELINE_MV * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        return x + wander[None, :].astype(np.float32)

    def __getitem__(self, i: int):
        r = self.index.iloc[i]
        x = np.asarray(self._signals()[int(r["row"])], dtype=np.float32)   # [12, samples], mV
        if self.train:
            x = self._augment(x, np.random.default_rng())
        x = (x - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(float(r["label"]))
