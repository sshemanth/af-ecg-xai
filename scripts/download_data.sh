#!/usr/bin/env bash
# Downloads PTB-XL and Chapman-Shaoxing from PhysioNet into data/raw/.
# Run from the project folder:  bash scripts/download_data.sh
# Safe to re-run: wget resumes and skips files that are already complete.
set -euo pipefail

cd "$(dirname "$0")/.."
RAW="data/raw"
mkdir -p "$RAW"

if ! command -v wget >/dev/null 2>&1; then
  echo "wget not found. Install it with:  brew install wget"
  exit 1
fi

# ---------------------------------------------------------------------------
# 1) PTB-XL v1.0.3 (about 1.7 GB zip)
#    Page: https://physionet.org/content/ptb-xl/1.0.3/
# ---------------------------------------------------------------------------
PTBXL_ZIP="$RAW/ptb-xl-1.0.3.zip"
PTBXL_URL="https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"

if [ -d "$RAW/ptbxl" ] && [ -f "$RAW/ptbxl/ptbxl_database.csv" ]; then
  echo "==> PTB-XL already present, skipping"
else
  echo "==> Downloading PTB-XL"
  wget -c -O "$PTBXL_ZIP" "$PTBXL_URL"
  echo "==> Unzipping PTB-XL"
  unzip -q "$PTBXL_ZIP" -d "$RAW"
  # The zip unpacks into a long folder name; rename it to ptbxl/
  mv "$RAW"/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 "$RAW/ptbxl"
  echo "    (You can delete $PTBXL_ZIP to save space.)"
fi

# ---------------------------------------------------------------------------
# 2) Chapman-Shaoxing, from the PhysioNet/CinC Challenge 2021 copy
#    Labels are already SNOMED-CT codes in the .hea headers, which makes
#    label mapping easier than the original release.
#    Page: https://physionet.org/content/challenge-2021/1.0.3/
# ---------------------------------------------------------------------------
CHAP_URL="https://physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing/"

echo "==> Downloading Chapman-Shaoxing (Challenge 2021 copy)"
mkdir -p "$RAW/chapman"
# -r recursive, -N only newer, -c resume, -np no parent dirs,
# -nH/--cut-dirs drop the URL folders so files land in data/raw/chapman/g1..g11
wget -r -N -c -np -nH --cut-dirs=5 -R "index.html*" \
  -P "$RAW/chapman" "$CHAP_URL"

echo
echo "Done. Next step:  python scripts/01_count_labels.py"
