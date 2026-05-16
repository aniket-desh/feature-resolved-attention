#!/bin/bash
# Cadenza Llama-3 8B sleeper: 3x3 attribution x intervention matrix at
# the localisation winner. Analog of fig:matrix_scatter from the paper.
#
# Requires the FRA-native (ln1.hook_normalized) SAE at the winning layer.
#
# Usage:
#   LAYER=16 \
#     SAE_PATH=/workspace/aniket/saes/cadenza_L16_ln1_hook_normalized/<id>/<step> \
#     bash reproduce/sleepers/cadenza_matrix.sh
#
# Outputs:
#   logs/cadenza_phase1/attribution_matrix.json
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PY=${PY:-/usr/bin/python3}
LAYER=${LAYER:?must set LAYER}
SAE_PATH=${SAE_PATH:?must set SAE_PATH}
OUT_JSON=${OUT_JSON:-logs/cadenza_phase1/attribution_matrix.json}
HEADS=${HEADS:-"0 8 16 24"}
N_VAL=${N_VAL:-10}
N_EVAL=${N_EVAL:-10}

mkdir -p "$(dirname "$OUT_JSON")"

"$PY" -m experiments.sleepers.cadenza.phase1_attribution_matrix \
    --sae-path "$SAE_PATH" \
    --layer "$LAYER" \
    --heads $HEADS \
    --n-val "$N_VAL" \
    --n-eval "$N_EVAL" \
    --out "$OUT_JSON"

echo
echo "=== output ==="
echo "  matrix : $OUT_JSON"
