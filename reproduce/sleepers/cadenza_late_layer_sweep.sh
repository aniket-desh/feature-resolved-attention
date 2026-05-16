#!/bin/bash
# Phase-3 (b): finer locality probe — train fresh SAEs at L28, L30, L31
# hook_resid_post and run phase0 localisation on each. Tests whether the
# Cadenza single-feature suppression is sharply at L29 or smeared across
# the late layers.
#
# Per cell: ~55 min training + ~35 min localisation = ~90 min.
# 3 cells → ~4.5 hr.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PY=${PY:-/usr/bin/python3}
SAE_ROOT=${SAE_ROOT:-/workspace/aniket/saes}
LOG_ROOT=${LOG_ROOT:-logs}
LAYERS=${LAYERS:-"28 30 31"}
HOOK=${HOOK:-hook_resid_post}
TRAINING_TOKENS=${TRAINING_TOKENS:-50000000}

mkdir -p "$LOG_ROOT/sae_training" "$LOG_ROOT/cadenza_late_layers"

for layer in $LAYERS; do
  sae_dir="$SAE_ROOT/cadenza_L${layer}_${HOOK}"
  train_log="$LOG_ROOT/sae_training/cadenza_L${layer}_${HOOK}_$(date +%Y%m%d_%H%M%S).log"
  loc_json="$LOG_ROOT/cadenza_late_layers/cadenza_L${layer}_${HOOK}.json"
  loc_log="$LOG_ROOT/cadenza_late_layers/cadenza_L${layer}_${HOOK}.log"

  echo
  echo "================================================================"
  echo "  late-layer cell : layer=$layer  hook=$HOOK"
  echo "================================================================"

  if [ -d "$sae_dir" ] && find "$sae_dir" -name "sae_weights.safetensors" 2>/dev/null | grep -q .; then
    echo "  → SAE already exists; skipping training."
  else
    "$PY" scripts/train_topk_sae_llama.py \
      --hook-layer "$layer" \
      --hook-point "$HOOK" \
      --output-dir "$sae_dir" \
      --training-tokens "$TRAINING_TOKENS" \
      > "$train_log" 2>&1
    echo "  → trained ($(date))"
  fi

  if [ -f "$loc_json" ]; then
    echo "  → localisation JSON already exists; skipping ($loc_json)"
    continue
  fi

  ckpt=$(find "$sae_dir" -name "sae_weights.safetensors" -printf "%h\n" \
          | sort -t/ -k6 -n | tail -1)
  echo "  → checkpoint: $ckpt"

  "$PY" -m experiments.sleepers.cadenza.phase0_localisation \
    --sae-path "$ckpt" \
    --hook-layer "$layer" \
    --hook-point "$HOOK" \
    --out "$loc_json" \
    > "$loc_log" 2>&1
  echo "  → localisation done; result → $loc_json"
done

echo
echo "=== late-layer sweep complete ==="
ls -1 "$LOG_ROOT/cadenza_late_layers/"*.json 2>/dev/null
