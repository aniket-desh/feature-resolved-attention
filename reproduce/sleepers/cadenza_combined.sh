#!/bin/bash
# Cadenza Llama-3 8B sleeper: single FRA-OV feature vs. conventional
# resid-mid additive baseline, headline figure analog (combined_50k.pdf).
#
# Requires two pre-trained SAEs at the localisation winner:
#   $FRA_SAE_PATH       (trained on ln1.hook_normalized)
#   $CONV_SAE_PATH      (trained on hook_resid_mid)
# Both at the same layer L (set via $LAYER).
#
# Usage:
#   LAYER=16 \
#     FRA_SAE_PATH=/workspace/aniket/saes/cadenza_L16_ln1_hook_normalized/<id>/<step> \
#     CONV_SAE_PATH=/workspace/aniket/saes/cadenza_L16_hook_resid_mid/<id>/<step> \
#     bash reproduce/sleepers/cadenza_combined.sh
#
# Outputs:
#   logs/cadenza_phase1/single_feature_metrics.json   metric sweep
#   figures/cadenza_combined_50k.{png,pdf}            headline figure
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PY=${PY:-/usr/bin/python3}
LAYER=${LAYER:?must set LAYER (e.g. 16)}
FRA_SAE_PATH=${FRA_SAE_PATH:?must set FRA_SAE_PATH}
CONV_SAE_PATH=${CONV_SAE_PATH:?must set CONV_SAE_PATH}
OUT_JSON=${OUT_JSON:-logs/cadenza_phase1/single_feature_metrics.json}
FIG_OUT=${FIG_OUT:-figures/cadenza_combined_50k}
N_PROMPTS=${N_PROMPTS:-20}

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$FIG_OUT")"

"$PY" -m experiments.sleepers.cadenza.phase1_single_feature \
    --fra-sae-path  "$FRA_SAE_PATH" \
    --conv-sae-path "$CONV_SAE_PATH" \
    --layer "$LAYER" \
    --n-prompts "$N_PROMPTS" \
    --out "$OUT_JSON"

"$PY" experiments/sleepers/scripts/plot_combined_50k.py \
    --input  "$OUT_JSON" \
    --output "$FIG_OUT" \
    --n_prompts "$N_PROMPTS" \
    --title "Cadenza Llama-3 8B sleeper — single OV vs conventional resid-mid"

echo
echo "=== outputs ==="
echo "  metrics : $OUT_JSON"
echo "  figure  : ${FIG_OUT}.png / .pdf"
