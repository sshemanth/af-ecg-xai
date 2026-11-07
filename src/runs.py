"""Find trained ResNet runs and the primary model (shared by evaluation and XAI)."""
import json

import pandas as pd
import torch

from src import config
from src.metrics import summarise
from src.model import ResNet1d


def find_resnet_runs(verbose: bool = True) -> list[dict]:
    """Every finished full run (quick-test runs made with --limit are skipped)."""
    runs = []
    for d in sorted(config.MODELS_DIR.glob("resnet_seed*")):
        files = [d / "model.pt", d / "threshold.json", d / "run_config.json", d / "val_predictions.csv"]
        if not all(f.exists() for f in files):
            continue
        cfg = json.loads((d / "run_config.json").read_text())
        if cfg.get("limit"):
            if verbose:
                print(f"[skip] {d.name}: quick-test run (--limit), not a real model")
            continue
        val = pd.read_csv(d / "val_predictions.csv")
        val_auprc = summarise(val["label"].to_numpy(), val["prob"].to_numpy(), 0.5)["auprc"]
        runs.append({"name": d.name, "dir": d, "seed": cfg["seed"], "val_auprc": val_auprc,
                     "threshold": json.loads((d / "threshold.json").read_text())["threshold"]})
    return runs


def primary_run(runs: list[dict]) -> dict:
    """The seed with the best VALIDATION AUPRC (config.PRIMARY_MODEL_RULE)."""
    return max(runs, key=lambda r: r["val_auprc"])


def load_model(run: dict, device) -> torch.nn.Module:
    model = ResNet1d().to(device)
    model.load_state_dict(torch.load(run["dir"] / "model.pt", map_location=device))
    return model.eval()
