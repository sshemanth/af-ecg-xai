#!/usr/bin/env bash
# Faster Chapman-Shaoxing download: fetches the g1..g11 folders in parallel
# instead of one file at a time. Safe to run after the normal script was
# stopped; files already downloaded are skipped.
#
# Run from the project folder:  bash scripts/download_chapman_fast.sh
# Optional: number of folders at once (default 6), e.g.
#           bash scripts/download_chapman_fast.sh 4
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="data/raw/chapman"
BASE="https://physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing"
JOBS="${1:-6}"   # kept modest so we don't hammer PhysioNet

if ! command -v wget >/dev/null 2>&1; then
  echo "wget not found. Install it with:  brew install wget"
  exit 1
fi
mkdir -p "$OUT" logs

# If the slow download was stopped with Ctrl+C, the last file may be cut
# off. Remove the 2 most recent files so they get downloaded again.
if [ -n "$(find "$OUT" -type f -name '*.*' -print -quit)" ]; then
  find "$OUT" -type f \( -name '*.mat' -o -name '*.hea' \) -print0 \
    | xargs -0 ls -t | head -2 | while read -r f; do rm -f "$f"; done
fi

echo "==> Downloading g1..g11 with $JOBS folders at a time"
echo "    Progress per folder goes to logs/chapman_gN.log"

fetch_group() {
  g="$1"
  # -nc skips files that already exist, -q keeps the terminal quiet
  wget -r -nc -np -nH --cut-dirs=5 -R "index.html*" -q \
    -P "$OUT" "$BASE/$g/" > "logs/chapman_$g.log" 2>&1 \
    && echo "    done: $g" || echo "    problem with $g (see logs/chapman_$g.log, just re-run)"
}
export -f fetch_group
export OUT BASE

printf "g%s\n" $(seq 1 11) | xargs -P "$JOBS" -I{} bash -c 'fetch_group {}'

n_hea=$(find "$OUT" -name "*.hea" | wc -l | tr -d ' ')
n_mat=$(find "$OUT" -name "*.mat" | wc -l | tr -d ' ')
echo
echo "Chapman files: $n_hea headers, $n_mat signal files"
echo "If these two numbers differ, run this script again to fill the gaps."
