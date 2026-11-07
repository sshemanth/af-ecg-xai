"""Step 5: train the 1D ResNet on PTB-XL.

  - trained on the PTB-XL training split only
  - validation split for early stopping, model choice and the threshold
  - class-weighted loss for the ~7% AF rate
  - augmentation that keeps atrial morphology (see src/data.py)
  - repeated runs with different seeds
The test split and Chapman are NOT touched here; the evaluation script
scores them once with the locked model and threshold.

Run from the project folder, after 02_preprocess.py:
    python scripts/05_train.py --limit 2000 --epochs 2    # quick check
    python scripts/05_train.py --seed 1                   # one full run
    python scripts/05_train.py --seed all                 # all seeds in config.RUN_SEEDS
    python scripts/05_train.py --seed 2,3,4,5             # several specific seeds

Outputs per run (results/models/resnet_seed<N>/):
    model.pt, threshold.json, val_predictions.csv, history.csv, run_config.json
and results/05_train_report.md (summary of every run so far)
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src import config  # noqa: E402
from src.data import ECGDataset  # noqa: E402
from src.metrics import choose_threshold, summarise  # noqa: E402
from src.model import ResNet1d, pick_device  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        out.append(torch.sigmoid(model(x.to(device))).float().cpu().numpy())
    return np.concatenate(out)


def train_one(seed: int, args) -> dict:
    set_seed(seed)
    device = pick_device(args.device)
    run_dir = config.MODELS_DIR / f"resnet_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ECGDataset("ptbxl", ["train"], train=True, limit=args.limit, seed=seed)
    val_ds = ECGDataset("ptbxl", ["val"], train=False,
                        limit=args.limit // 8 if args.limit else None, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    loader_kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    if len(train_ds) < args.batch_size:
        sys.exit(f"Only {len(train_ds)} training ECGs, fewer than one batch "
                 f"({args.batch_size}). Use a smaller --batch-size or a bigger --limit.")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          generator=gen, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, **loader_kw)

    n_pos = float(train_ds.labels.sum())
    n_neg = float(len(train_ds) - n_pos)
    if config.POS_WEIGHT == "balanced":
        pos_weight = n_neg / max(n_pos, 1.0)
    else:
        pos_weight = float(config.POS_WEIGHT) if config.POS_WEIGHT else 1.0

    model = ResNet1d(dropout=args.dropout).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"\n=== seed {seed} | device {device} | train {len(train_ds)} ({int(n_pos)} AF) "
          f"| val {len(val_ds)} | pos_weight {pos_weight:.1f} | "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params ===")

    best, best_epoch, history, waited = -1.0, -1, [], 0
    y_val = val_ds.labels
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()

        p_val = predict(model, val_dl, device)
        m = summarise(y_val, p_val, 0.5)
        history.append({"epoch": epoch, "train_loss": total / max(n, 1),
                        "val_auroc": m["auroc"], "val_auprc": m["auprc"],
                        "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0})
        improved = m["auprc"] > best
        print(f"epoch {epoch:2d} | loss {total / max(n, 1):.4f} | val AUROC {m['auroc']:.4f} "
              f"| val AUPRC {m['auprc']:.4f} | {time.time() - t0:.0f}s"
              + ("  <- best" if improved else ""))
        if improved:
            best, best_epoch, waited = m["auprc"], epoch, 0
            torch.save(model.state_dict(), run_dir / "model.pt")
        else:
            waited += 1
            if waited >= config.PATIENCE:
                print(f"early stop: no val AUPRC gain for {config.PATIENCE} epochs")
                break

    # Best model -> validation predictions -> locked threshold
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    p_val = predict(model, val_dl, device)
    thr = choose_threshold(y_val, p_val)
    val_metrics = summarise(y_val, p_val, thr)

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame({"record_id": val_ds.index["record_id"], "label": y_val, "prob": p_val}).to_csv(
        run_dir / "val_predictions.csv", index=False)
    (run_dir / "threshold.json").write_text(json.dumps(
        {"threshold": thr, "rule": config.THRESHOLD_RULE, "chosen_on": "ptbxl val"}, indent=2))
    (run_dir / "run_config.json").write_text(json.dumps({
        "seed": seed, "lr": args.lr, "dropout": args.dropout, "batch_size": args.batch_size,
        "max_epochs": args.epochs, "best_epoch": best_epoch, "pos_weight": pos_weight,
        "limit": args.limit, "device": str(device), "widths": config.RESNET_WIDTHS,
        "blocks": config.RESNET_BLOCKS, "kernel": config.RESNET_KERNEL,
        "train_crop_s": config.TRAIN_CROP_S, "weight_decay": config.WEIGHT_DECAY,
    }, indent=2))
    return {"seed": seed, "best_epoch": best_epoch, "quick_test": bool(args.limit), **val_metrics}


def write_report() -> None:
    rows = []
    for d in sorted(config.MODELS_DIR.glob("resnet_seed*")):
        cfg_f, pred_f, thr_f = d / "run_config.json", d / "val_predictions.csv", d / "threshold.json"
        if not (cfg_f.exists() and pred_f.exists() and thr_f.exists()):
            continue
        cfg, pred = json.loads(cfg_f.read_text()), pd.read_csv(pred_f)
        thr = json.loads(thr_f.read_text())["threshold"]
        m = summarise(pred["label"].to_numpy(), pred["prob"].to_numpy(), thr)
        rows.append({"seed": cfg["seed"], "best_epoch": cfg["best_epoch"],
                     "quick_test": bool(cfg.get("limit")), **m})
    if not rows:
        return
    df = pd.DataFrame(rows)
    lines = ["# 1D ResNet training (validation only)\n",
             "Threshold per run chosen on PTB-XL val (Youden) and locked. "
             "Test split and Chapman not used.\n",
             "| Seed | Best epoch | AUROC | AUPRC | Threshold | Sens | Spec | PPV | F1 | Note |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.seed} | {r.best_epoch} | {r.auroc:.4f} | {r.auprc:.4f} | "
                     f"{r.threshold:.3f} | {r.sensitivity:.3f} | {r.specificity:.3f} | "
                     f"{r.ppv:.3f} | {r.f1:.3f} | {'quick test' if r.quick_test else ''} |")
    full = df[~df["quick_test"]]
    if len(full) > 1:
        lines.append(f"\nAcross {len(full)} full runs: AUROC {full['auroc'].mean():.4f} ± "
                     f"{full['auroc'].std():.4f}, AUPRC {full['auprc'].mean():.4f} ± "
                     f"{full['auprc'].std():.4f} (mean ± SD)")
    base = config.RESULTS_DIR / "04_rr_baseline_report.md"
    if base.exists():
        lines.append("\nCompare with the RR-only baseline in `results/04_rr_baseline_report.md`.")
    text = "\n".join(lines) + "\n"
    (config.RESULTS_DIR / "05_train_report.md").write_text(text)
    print("\n" + text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="1",
                    help="a number, a comma list like 2,3,4,5, or 'all' for config.RUN_SEEDS")
    ap.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--dropout", type=float, default=config.DROPOUT)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only about this many training ECGs (quick check)")
    args = ap.parse_args()

    seeds = (config.RUN_SEEDS if args.seed == "all"
             else [int(s) for s in args.seed.split(",") if s.strip()])
    for seed in seeds:
        train_one(seed, args)
    write_report()


if __name__ == "__main__":
    main()
