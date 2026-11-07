#!/usr/bin/env bash
# Download one PhysioNet/CinC Challenge 2021 dataset, fetching its g1, g2, ...
# folders in parallel. Safe to re-run: files already downloaded are skipped.
#
# Run from the project folder:
#   bash scripts/download_challenge.sh georgia
#   bash scripts/download_challenge.sh chapman
# Optional second argument: folders at once (default 6), e.g.
#   bash scripts/download_challenge.sh georgia 4
set -euo pipefail

cd "$(dirname "$0")/.."
KEY="${1:-}"
case "$KEY" in
  chapman) REMOTE="chapman_shaoxing" ;;
  georgia) REMOTE="georgia" ;;
  *) echo "Usage: bash scripts/download_challenge.sh chapman|georgia [jobs]"; exit 1 ;;
esac
OUT="data/raw/$KEY"
BASE="https://physionet.org/files/challenge-2021/1.0.3/training/$REMOTE"
JOBS="${2:-6}"   # kept modest so we don't hammer PhysioNet

if ! command -v wget >/dev/null 2>&1; then
  echo "wget not found. Install it with:  brew install wget"
  exit 1
fi
mkdir -p "$OUT" logs

# If an earlier download was stopped with Ctrl+C, the last file may be cut
# off. Remove the 2 most recent files so they get downloaded again.
if [ -n "$(find "$OUT" -type f -name '*.*' -print -quit)" ]; then
  find "$OUT" -type f \( -name '*.mat' -o -name '*.hea' \) -print0 \
    | xargs -0 ls -t | head -2 | while read -r f; do rm -f "$f"; done
fi

echo "==> Downloading $KEY ($JOBS folders at a time)"
echo "    Progress per folder goes to logs/${KEY}_gN.log"

fetch_group() {
  g="$1"
  # skip folder numbers that don't exist for this dataset
  if ! wget -q --spider "$BASE/$g/"; then
    return 0
  fi
  # -nc skips files that already exist, -q keeps the terminal quiet
  wget -r -nc -np -nH --cut-dirs=5 -R "index.html*" -q \
    -P "$OUT" "$BASE/$g/" > "logs/${KEY}_$g.log" 2>&1 \
    && echo "    done: $g" || echo "    problem with $g (see logs/${KEY}_$g.log, just re-run)"
}
export -f fetch_group
export OUT BASE KEY

printf "g%s\n" $(seq 1 12) | xargs -P "$JOBS" -I{} bash -c 'fetch_group {}'

n_hea=$(find "$OUT" -name "*.hea" | wc -l | tr -d ' ')
n_mat=$(find "$OUT" -name "*.mat" | wc -l | tr -d ' ')
echo
echo "$KEY files: $n_hea headers, $n_mat signal files"
echo "If these two numbers differ, run this script again to fill the gaps."
