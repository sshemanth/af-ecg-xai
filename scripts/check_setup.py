"""Quick check that packages, the Apple GPU (MPS) and the data are ready.

Run from the project folder:
    python scripts/check_setup.py
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

OK, BAD = "OK ", "-- "


def check_packages() -> None:
    print("Packages")
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "tqdm",
                "wfdb", "neurokit2", "torch", "captum", "statsmodels"]:
        try:
            mod = importlib.import_module(pkg)
            print(f"  {OK}{pkg} {getattr(mod, '__version__', '')}")
        except ImportError:
            print(f"  {BAD}{pkg} missing  (pip install -r requirements.txt)")


def check_device() -> None:
    print("Device")
    try:
        import torch
    except ImportError:
        print(f"  {BAD}torch not installed")
        return
    if torch.backends.mps.is_available():
        x = torch.ones(2, 2, device="mps")
        print(f"  {OK}Apple GPU (MPS) works: {float((x @ x).sum()):.0f} == 8")
    elif torch.cuda.is_available():
        print(f"  {OK}CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  {BAD}No GPU found, will use CPU (slower but works)")


def check_data() -> None:
    print("Data")
    ptb = config.PTBXL_DIR / "ptbxl_database.csv"
    print(f"  {OK if ptb.exists() else BAD}PTB-XL: {ptb}")
    n_chap = len(list(config.CHAPMAN_DIR.rglob("*.hea"))) if config.CHAPMAN_DIR.exists() else 0
    print(f"  {OK if n_chap else BAD}Chapman-Shaoxing: {n_chap} header files in {config.CHAPMAN_DIR}")
    if not ptb.exists() or not n_chap:
        print("  Download with:  bash scripts/download_data.sh")


if __name__ == "__main__":
    check_packages()
    check_device()
    check_data()
