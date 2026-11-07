#!/usr/bin/env bash
# Faster Chapman-Shaoxing download (folders in parallel). Kept for
# convenience; same as:  bash scripts/download_challenge.sh chapman
# Run from the project folder:  bash scripts/download_chapman_fast.sh [jobs]
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/download_challenge.sh chapman "${1:-6}"
