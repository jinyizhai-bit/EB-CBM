#!/usr/bin/env bash
# Downloads and unpacks the ISIC 2018 Task 1-2 archives (images, lesion masks, attribute masks; ~12 GB) into
# data/raw/isic2018_task12 (or $CBM_DATA_ROOT/raw/isic2018_task12). Idempotent: complete files are skipped.
# Licence terms: https://challenge.isic-archive.com/data/ (check them before use).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=${CBM_DATA_ROOT:-$ROOT/data}/raw/isic2018_task12
mkdir -p "$DEST"
B=https://isic-challenge-data.s3.amazonaws.com/2018
FILES="ISIC2018_Task1-2_Training_Input.zip ISIC2018_Task1_Training_GroundTruth.zip ISIC2018_Task2_Training_GroundTruth_v3.zip
ISIC2018_Task1-2_Validation_Input.zip ISIC2018_Task1_Validation_GroundTruth.zip ISIC2018_Task2_Validation_GroundTruth.zip
ISIC2018_Task1-2_Test_Input.zip ISIC2018_Task1_Test_GroundTruth.zip ISIC2018_Task2_Test_GroundTruth.zip"
for f in $FILES; do
  want=$(curl -sI "$B/$f" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r')
  if [ -f "$DEST/$f" ] && [ "$(stat -c %s "$DEST/$f")" = "$want" ]; then
    echo "skip  $f"
  else
    echo "fetch $f ($want bytes)"
    curl -sSL -C - -o "$DEST/$f" "$B/$f"
  fi
  if [ ! -d "$DEST/${f%.zip}" ]; then
    echo "unzip $f"
    unzip -q "$DEST/$f" -d "$DEST"
  fi
done
echo "done: $DEST"
