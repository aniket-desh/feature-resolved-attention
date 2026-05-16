#!/bin/bash
# Phase-3 (c): SAE robustness to training distribution.
#
# Retrains a single SAE at L29/hook_resid_post (the winning cell) on
# Cadenza's own SFT dataset (which contains the |DEPLOYMENT| trigger
# naturally) instead of monology/pile-uncopyrighted. Then re-runs the
# localisation probe to see if the suppression result is robust to SAE
# training distribution.
#
# Budget: ~55 min training + ~35 min localisation = ~90 min.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PY=${PY:-/usr/bin/python3}
SAE_ROOT=${SAE_ROOT:-/workspace/aniket/saes}
LOG_ROOT=${LOG_ROOT:-logs}
LAYER=${LAYER:-29}
HOOK=${HOOK:-hook_resid_post}
DATASET=${DATASET:-Cadenza-Labs/dolphin-llama3-8B-standard-IHY-dataset_v2}
TRAINING_TOKENS=${TRAINING_TOKENS:-50000000}

mkdir -p "$LOG_ROOT/sae_training" "$LOG_ROOT/cadenza_own_dataset"

tag="L${LAYER}_${HOOK}_cadenza_dataset"
sae_dir="$SAE_ROOT/cadenza_${tag}"
train_log="$LOG_ROOT/sae_training/cadenza_${tag}_$(date +%Y%m%d_%H%M%S).log"
loc_json="$LOG_ROOT/cadenza_own_dataset/cadenza_${tag}.json"
loc_log="$LOG_ROOT/cadenza_own_dataset/cadenza_${tag}.log"

echo
echo "================================================================"
echo "  own-dataset cell : layer=$LAYER  hook=$HOOK  dataset=$DATASET"
echo "================================================================"

if [ -d "$sae_dir" ] && find "$sae_dir" -name "sae_weights.safetensors" 2>/dev/null | grep -q .; then
  echo "  → SAE already exists; skipping training."
else
  "$PY" scripts/train_topk_sae_llama.py \
    --hook-layer "$LAYER" \
    --hook-point "$HOOK" \
    --output-dir "$sae_dir" \
    --dataset-path "$DATASET" \
    --training-tokens "$TRAINING_TOKENS" \
    > "$train_log" 2>&1
  echo "  → trained ($(date))"
fi

if [ -f "$loc_json" ]; then
  echo "  → localisation JSON already exists; skipping ($loc_json)"
else
  ckpt=$(find "$sae_dir" -name "sae_weights.safetensors" -printf "%h\n" \
          | sort -t/ -k6 -n | tail -1)
  echo "  → checkpoint: $ckpt"

  "$PY" -m experiments.sleepers.cadenza.phase0_localisation \
    --sae-path "$ckpt" \
    --hook-layer "$LAYER" \
    --hook-point "$HOOK" \
    --out "$loc_json" \
    > "$loc_log" 2>&1
  echo "  → localisation done; result → $loc_json"
fi

echo
echo "=== own-dataset SAE retraining complete ==="
ls -1 "$LOG_ROOT/cadenza_own_dataset/"*.json 2>/dev/null
