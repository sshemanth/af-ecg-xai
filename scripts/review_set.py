"""Blinded review of doubtful AF labels by a clinician.

make   Picks the AF ECGs with a near-regular rhythm (strict rule; --group
       regular for the wider rule) and mixes in control ECGs: clear AF
       (irregular), sinus rhythm and other non-AF. Order is shuffled and
       every ECG gets an anonymous ID, so the reviewer sees no label, no
       group and no model output. Writes:
         results/review/<dataset>/review_ecgs.pdf    one 12-lead ECG per page
         results/review/<dataset>/review_sheet.csv   to fill in
         results/review/<dataset>/INSTRUCTIONS.txt
         results/review/<dataset>/key/review_key.csv do NOT give to the reviewer
score  Reads the filled sheet and the key and reports how often the
       reviewer agreed with the official label in each group (the controls
       show how reliable the review is), and how the models did on AF
       labels the reviewer confirmed vs rejected.

The ECGs are the preprocessed signals (0.5-100 Hz band-pass), drawn at the
usual 25 mm/s and 10 mm/mV on a 1 mm / 5 mm grid.

Run from the project folder, after 03_rpeaks.py (score after 06_evaluate.py):
    python scripts/review_set.py make --dataset georgia
    python scripts/review_set.py score --dataset georgia
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from src import config  # noqa: E402
from src.dx import rate  # noqa: E402
from src.runs import arch_label, find_runs, primary_run  # noqa: E402

ANSWERS = {"af": "AF", "not af": "not AF", "non-af": "not AF", "non af": "not AF", "no": "not AF",
           "unsure": "unsure", "?": "unsure", "": ""}


def out_dir(key: str) -> Path:
    return config.RESULTS_DIR / "review" / key


def pick(key: str, group: str, n_af: int, n_sinus: int, n_other: int) -> pd.DataFrame:
    rr = pd.read_csv(config.PROCESSED_DIR / f"{key}_rr.csv", dtype={"record_id": str})
    rr = rr[rr["rpeaks"].notna()]
    rng = np.random.default_rng(config.SEED)
    col = "regular_af_strict" if group == "strict" else "regular_af"
    target = rr[(rr["label"] == 1) & rr[col].fillna(False).astype(bool)]
    af_irr = rr[(rr["label"] == 1) & ~rr["regular_af"].fillna(False).astype(bool)]
    sinus = rr[(rr["label"] == 0) & rr["sinus"].fillna(False).astype(bool)]
    other = rr[(rr["label"] == 0) & ~rr["sinus"].fillna(False).astype(bool)]

    def take(df, n, g):
        idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
        return df.iloc[idx].assign(group=g)

    sel = pd.concat([target.assign(group=f"near-regular AF ({group} rule)"),
                     take(af_irr, n_af, "control: AF, irregular"),
                     take(sinus, n_sinus, "control: sinus rhythm"),
                     take(other, n_other, "control: other non-AF")])
    sel = sel.iloc[rng.permutation(len(sel))].reset_index(drop=True)
    sel["review_id"] = [f"R{i + 1:03d}" for i in range(len(sel))]
    return sel


def draw(ax, x: np.ndarray, title: str) -> None:
    """12 leads stacked, on true ECG paper scale: 25 mm/s and 10 mm/mV, with
    1 mm and 5 mm grid lines; each lead gets a 25 mm band."""
    t = np.arange(x.shape[1]) / config.TARGET_FS
    gap, clip = 2.5, 1.2                                  # mV per band, mV clip
    top, bottom = gap / 2, -gap * 11 - gap / 2
    ax.set_xticks(np.arange(0, config.SIGNAL_SECONDS + 0.001, 0.2))
    ax.set_xticks(np.arange(0, config.SIGNAL_SECONDS + 0.001, 0.04), minor=True)
    ax.set_yticks(np.arange(bottom, top + 0.001, 0.5))
    ax.set_yticks(np.arange(bottom, top + 0.001, 0.1), minor=True)
    ax.grid(which="major", color="#e8a39b", lw=0.6)
    ax.grid(which="minor", color="#f6d5d1", lw=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", length=0, labelleft=False, labelbottom=False)
    for i, lead in enumerate(config.LEADS):
        base = -gap * i
        ax.plot(t, np.clip(x[i], -clip, clip) + base, color="black", lw=0.6)
        ax.text(0.03, base + 0.75, lead, fontsize=9, fontweight="bold")
    for s in range(1, int(config.SIGNAL_SECONDS)):        # second marks along the bottom
        ax.text(s, bottom + 0.1, f"{s} s", fontsize=6, ha="center", color="#a0524a")
    ax.set_xlim(0, config.SIGNAL_SECONDS)
    ax.set_ylim(bottom, top)
    ax.set_aspect(0.4)                                    # 1 mV = 10 mm, 1 s = 25 mm
    ax.set_title(f"{title}      25 mm/s · 10 mm/mV · amplitudes over {clip} mV clipped",
                 fontsize=11, loc="left")


def make(args) -> None:
    sel = pick(args.dataset, args.group, args.n_af, args.n_sinus, args.n_other)
    d = out_dir(args.dataset)
    (d / "key").mkdir(parents=True, exist_ok=True)
    sig = np.load(config.PROCESSED_DIR / f"{args.dataset}_signals.npy", mmap_mode="r")
    pdf_path = d / "review_ecgs.pdf"
    with PdfPages(pdf_path) as pdf:
        for _, r in sel.iterrows():
            fig, ax = plt.subplots(figsize=(10.26, 12.69))         # trace area 250 x 300 mm at 100% print
            draw(ax, np.asarray(sig[int(r["row"])], dtype=np.float32), f"ECG {r['review_id']}")
            fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02)
            pdf.savefig(fig)
            plt.close(fig)
    pd.DataFrame({"review_id": sel["review_id"], "rhythm": "", "notes": ""}).to_csv(
        d / "review_sheet.csv", index=False)
    sel[["review_id", "record_id", "group", "label", "hr_mean", "rr_cv"]].to_csv(
        d / "key" / "review_key.csv", index=False)
    (d / "INSTRUCTIONS.txt").write_text(
        f"Blinded ECG review ({len(sel)} ECGs)\n\n"
        "Each page of review_ecgs.pdf is one 10-second 12-lead ECG (filtered 0.5-100 Hz),\n"
        "drawn at 25 mm/s and 10 mm/mV (true scale if printed at 100% on A3).\n"
        "For each ECG, write in the 'rhythm' column of review_sheet.csv one of:\n"
        "    AF        atrial fibrillation present in this strip\n"
        "    not AF    no atrial fibrillation in this strip (any other rhythm)\n"
        "    unsure    cannot decide\n"
        "The 'notes' column is optional (e.g. 'sinus rhythm', 'flutter', 'noisy').\n\n"
        "Please judge only what is visible in each strip. The ECGs come from a public\n"
        "dataset and are shown in random order; no labels or computer predictions are given.\n")
    counts = sel["group"].value_counts()
    print(f"Saved {pdf_path} ({len(sel)} pages), review_sheet.csv and INSTRUCTIONS.txt in {d}")
    print("Groups (kept in key/review_key.csv, not shown to the reviewer):")
    for g, n in counts.items():
        print(f"  {g}: {n}")
    print("Give the reviewer review_ecgs.pdf, review_sheet.csv and INSTRUCTIONS.txt, not the key folder.")


def score(args) -> None:
    d = out_dir(args.dataset)
    sheet_f = Path(args.sheet) if args.sheet else d / "review_sheet.csv"
    sheet = pd.read_csv(sheet_f, dtype=str).fillna("")
    key = pd.read_csv(d / "key" / "review_key.csv", dtype={"record_id": str})
    sheet["answer"] = sheet["rhythm"].str.strip().str.lower().map(ANSWERS)
    bad = sheet[sheet["answer"].isna()]
    if len(bad):
        sys.exit(f"Unrecognised answers (use AF / not AF / unsure): "
                 f"{', '.join(bad['review_id'] + '=' + bad['rhythm'])}")
    df = key.merge(sheet[["review_id", "answer", "notes"]], on="review_id")
    done = df[df["answer"] != ""]
    L = [f"# Blinded review: {config.DATASET_NAMES[args.dataset]}\n",
         f"{len(done)} of {len(df)} ECGs answered.\n",
         "| Group | ECGs answered | Reviewer: AF | Reviewer: not AF | Reviewer: unsure |",
         "|---|---|---|---|---|"]
    for g, gdf in done.groupby("group"):
        n = len(gdf)
        L.append(f"| {g} | {n} | " + " | ".join(rate(int((gdf['answer'] == a).sum()), n)
                                                for a in ["AF", "not AF", "unsure"]) + " |")
    ctrl = done[done["group"].str.startswith("control")]
    agree = ((ctrl["label"] == 1) & (ctrl["answer"] == "AF")) | ((ctrl["label"] == 0) & (ctrl["answer"] == "not AF"))
    L.append(f"\nAgreement with the official label on control ECGs: {rate(int(agree.sum()), len(ctrl))} "
             "(how reliable the review is; 'unsure' counts as disagreement).\n")

    # models on the doubtful AF labels, split by the reviewer's call
    pred_f = config.EVAL_DIR / f"predictions_{args.dataset}.csv"
    target = done[done["group"].str.startswith("near-regular")]
    if pred_f.exists() and len(target):
        pred = pd.read_csv(pred_f, dtype={"record_id": str})
        t = target.merge(pred, on="record_id")
        thr_base = json.loads((config.MODELS_DIR / "rr_baseline" / "threshold.json").read_text())["threshold"]
        models = [(arch_label(a), f"prob_{p['name']}", p["threshold"])
                  for a in config.ARCHS if (r := find_runs(a, verbose=False)) for p in [primary_run(r)]]
        models.append(("RR baseline", "prob_rr_baseline", thr_base))
        L += ["## Near-regular AF labels: models vs reviewer\n",
              "Share called AF by each model, among labels the reviewer confirmed or rejected.\n",
              "| Model | Reviewer: AF | Reviewer: not AF | Reviewer: unsure |", "|---|---|---|---|"]
        for name, col, thr in models:
            if col not in t:
                continue
            cells = []
            for a in ["AF", "not AF", "unsure"]:
                g = t[(t["answer"] == a) & t[col].notna()]
                cells.append(rate(int((g[col] >= thr).sum()), len(g)))
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("\nIf the models mostly 'miss' the AF labels the reviewer rejected, those misses "
                 "reflect the labels rather than the models.\n")

    text = "\n".join(L)
    out = config.RESULTS_DIR / f"review_report_{args.dataset}.md"
    out.write_text(text)
    df.to_csv(d / "key" / "review_scored.csv", index=False)
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--dataset", choices=config.ALL_DATASETS, default="georgia")
    m.add_argument("--group", choices=["strict", "main"], default="strict",
                   help="near-regular AF by the strict (default) or main rule")
    m.add_argument("--n-af", type=int, default=30, help="irregular AF controls")
    m.add_argument("--n-sinus", type=int, default=30, help="sinus rhythm controls")
    m.add_argument("--n-other", type=int, default=15, help="other non-AF controls")
    s = sub.add_parser("score")
    s.add_argument("--dataset", choices=config.ALL_DATASETS, default="georgia")
    s.add_argument("--sheet", help="filled sheet, if not saved over review_sheet.csv")
    args = ap.parse_args()
    make(args) if args.cmd == "make" else score(args)


if __name__ == "__main__":
    main()
