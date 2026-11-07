"""Find trained runs and the primary model of each architecture (shared by
evaluation, attribution and occlusion)."""
import json

import pandas as pd
import torch

from src import config
from src.metrics import summarise
from src.model import build_model


def find_runs(arch: str = "resnet", verbose: bool = True) -> list[dict]:
    """Every finished full run of one architecture (quick-test runs made
    with --limit are skipped)."""
    runs = []
    for d in sorted(config.MODELS_DIR.glob(f"{arch}_seed*")):
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
        runs.append({"name": d.name, "dir": d, "arch": cfg.get("arch", "resnet"), "seed": cfg["seed"],
                     "val_auprc": val_auprc,
                     "threshold": json.loads((d / "threshold.json").read_text())["threshold"]})
    return sorted(runs, key=lambda r: r["seed"])


def find_resnet_runs(verbose: bool = True) -> list[dict]:
    return find_runs("resnet", verbose)


def primary_run(runs: list[dict]) -> dict:
    """The seed with the best VALIDATION AUPRC (config.PRIMARY_MODEL_RULE)."""
    return max(runs, key=lambda r: r["val_auprc"])


def load_model(run: dict, device) -> torch.nn.Module:
    model = build_model(run.get("arch", "resnet")).to(device)
    state = torch.load(run["dir"] / "model.pt", map_location=device)
    # models saved by earlier versions also stored the last stage a second
    # time under "last_conv."; those weights are identical, so drop them
    state = {k: v for k, v in state.items() if not k.startswith("last_conv.")}
    model.load_state_dict(state)
    return model.eval()


def arch_tag(arch: str) -> str:
    """Suffix for output files: none for the main ResNet, '_<arch>' otherwise."""
    return "" if arch == "resnet" else f"_{arch}"


def arch_label(arch: str) -> str:
    return {"resnet": "ResNet", "inception": "InceptionTime"}.get(arch, arch)
