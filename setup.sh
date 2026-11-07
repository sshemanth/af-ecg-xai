#!/usr/bin/env bash
# One-time setup on macOS: creates a virtual environment and installs packages.
# Run from the project folder:  bash setup.sh
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install it first (e.g. 'brew install python')."
  exit 1
fi

echo "==> Creating virtual environment in .venv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing packages"
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Checking wget (needed for downloads)"
if ! command -v wget >/dev/null 2>&1; then
  echo "    wget not found. Install it with:  brew install wget"
fi

echo "==> Checking the setup"
python scripts/check_setup.py

echo
echo "Done. Next time, activate the environment with:  source .venv/bin/activate"
