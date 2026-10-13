#!/usr/bin/env bash
# Downloads the TorchScript big-lama inpainting checkpoint (~196 MB; Suvorov et al., LaMa, WACV 2022; TorchScript
# export from https://github.com/enesmsahin/simple-lama-inpainting, Apache-2.0) used to build the counterfactual
# structure-removal edits. Target: models/big-lama.pt (or $CBM_LAMA_PT).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${CBM_LAMA_PT:-$ROOT/models/big-lama.pt}
URL=https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
mkdir -p "$(dirname "$OUT")"
if [ -f "$OUT" ] && [ "$(stat -c %s "$OUT")" = "205803670" ]; then
  echo "exists: $OUT"; exit 0
fi
curl -L -C - -o "$OUT" "$URL"
echo "saved: $OUT ($(stat -c %s "$OUT") bytes)"
