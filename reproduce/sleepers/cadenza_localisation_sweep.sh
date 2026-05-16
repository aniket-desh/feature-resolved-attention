#!/bin/bash
# Cadenza Llama-3 8B sleeper: train one TopK SAE per cell of the
# (layer × hookpoint) localisation grid, then run the single-feature
# steering screen on each. Sequential, ~70 min per SAE on H200.
#
# Per Dmitry's recommendation, the default grid is the network midpoint
# (layer 16) plus layer 3 (early) and layer 29 ≡ last-3 (late), crossing
# hookpoints {ln1.hook_normalized, hook_resid_mid, hook_resid_post}.
#
# Usage:
#   bash reproduce/sleepers/cadenza_localisation_sweep.sh
#
# Outputs:
#   /workspace/aniket/saes/cadenza_L{layer}_{tag}/<sae_lens_id>/<step>/...
#   logs/cadenza_localisation/cadenza_L{layer}_{tag}.json
#
# Override the grid via env:
#   LAYERS="16" HOOKS="ln1.hook_normalized" bash ...   # single cell
#   LAYERS="3 16 29" HOOKS="ln1.hook_normalized" ...    # one hookpoint, all layers
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PY=${PY:-/usr/bin/python3}
SAE_ROOT=${SAE_ROOT:-/workspace/aniket/saes}
LOG_ROOT=${LOG_ROOT:-logs}
LAYERS=${LAYERS:-"3 16 29"}
HOOKS=${HOOKS:-"ln1.hook_normalized hook_resid_mid hook_resid_post"}
TRAINING_TOKENS=${TRAINING_TOKENS:-50000000}

mkdir -p "$LOG_ROOT/sae_training" "$LOG_ROOT/cadenza_localisation"

for layer in $LAYERS; do
  for hook in $HOOKS; do
    tag=${hook//./_}            # ln1.hook_normalized -> ln1_hook_normalized
    sae_dir="$SAE_ROOT/cadenza_L${layer}_${tag}"
    train_log="$LOG_ROOT/sae_training/cadenza_L${layer}_${tag}_$(date +%Y%m%d_%H%M%S).log"
    local_log="$LOG_ROOT/cadenza_localisation/cadenza_L${layer}_${tag}.log"
    local_json="$LOG_ROOT/cadenza_localisation/cadenza_L${layer}_${tag}.json"

    echo
    echo "================================================================"
    echo "  cell  : layer=$layer  hook=$hook"
    echo "  sae   : $sae_dir"
    echo "  log   : $train_log"
    echo "================================================================"

    if [ -d "$sae_dir" ] && find "$sae_dir" -name "sae_weights.safetensors" | grep -q . ; then
      echo "  → SAE already exists; skipping training."
    else
      "$PY" scripts/train_topk_sae_llama.py \
        --hook-layer "$layer" \
        --hook-point "$hook" \
        --output-dir "$sae_dir" \
        --training-tokens "$TRAINING_TOKENS" \
        > "$train_log" 2>&1
      echo "  → trained ($(date))"
    fi

    if [ -f "$local_json" ]; then
      echo "  → localisation JSON already exists; skipping ($local_json)"
      continue
    fi

    # Localisation needs the deepest checkpoint directory; sae-lens nests
    # <sae_dir>/<random_id>/<n_steps>/{sae_weights.safetensors,cfg.json}.
    ckpt=$(find "$sae_dir" -name "sae_weights.safetensors" -printf "%h\n" \
            | sort -t/ -k6 -n | tail -1)
    echo "  → checkpoint: $ckpt"

    "$PY" -m experiments.sleepers.cadenza.phase0_localisation \
      --sae-path "$ckpt" \
      --hook-layer "$layer" \
      --hook-point "$hook" \
      --out "$local_json" \
      > "$local_log" 2>&1
    echo "  → localisation done; result → $local_json"
  done
done

echo
echo "=== localisation sweep complete ==="
echo "results:"
ls -1 "$LOG_ROOT/cadenza_localisation/"*.json 2>/dev/null
