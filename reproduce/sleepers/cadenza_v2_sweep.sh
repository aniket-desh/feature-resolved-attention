#!/bin/bash
# Cadenza v2 localisation sweep — paper-spec protocol on the SAEs trained
# during the v1 sweep. Reuses checkpoints under $SAE_ROOT and writes one
# JSON per cell to $LOG_ROOT/cadenza_localisation_v2/.
#
# Per-cell budget: ~25 min (FRA ranking ~10 min, greedy selection ~15 min,
# multi-seed eval ~1 min). Full 9-cell sweep: ~4 hours.
#
# Defaults to the same {3, 16, 29} × {ln1, resid_mid, resid_post} grid as
# v1, so each (layer, hook) cell maps one-to-one onto a v1 cell.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PY=${PY:-/usr/bin/python3}
SAE_ROOT=${SAE_ROOT:-/workspace/aniket/saes}
LOG_ROOT=${LOG_ROOT:-logs}
LAYERS=${LAYERS:-"3 16 29"}
HOOKS=${HOOKS:-"ln1.hook_normalized hook_resid_mid hook_resid_post"}

mkdir -p "$LOG_ROOT/cadenza_localisation_v2"

for layer in $LAYERS; do
  for hook in $HOOKS; do
    tag=${hook//./_}
    sae_dir="$SAE_ROOT/cadenza_L${layer}_${tag}"
    out_json="$LOG_ROOT/cadenza_localisation_v2/cadenza_L${layer}_${tag}.json"
    cell_log="$LOG_ROOT/cadenza_localisation_v2/cadenza_L${layer}_${tag}.log"

    echo
    echo "================================================================"
    echo "  v2 cell : layer=$layer  hook=$hook"
    echo "================================================================"

    if [ -f "$out_json" ]; then
      echo "  → v2 JSON already exists; skipping ($out_json)"
      continue
    fi

    if ! find "$sae_dir" -name "sae_weights.safetensors" 2>/dev/null | grep -q .; then
      echo "  → SAE missing for this cell ($sae_dir); skipping."
      continue
    fi

    ckpt=$(find "$sae_dir" -name "sae_weights.safetensors" -printf "%h\n" \
            | sort -t/ -k6 -n | tail -1)
    echo "  → checkpoint: $ckpt"

    "$PY" -m experiments.sleepers.cadenza.phase0_localisation_v2 \
      --sae-path "$ckpt" \
      --hook-layer "$layer" \
      --hook-point "$hook" \
      --out "$out_json" \
      > "$cell_log" 2>&1
    echo "  → v2 localisation done; result → $out_json"
  done
done

echo
echo "=== v2 localisation sweep complete ==="
echo "results:"
ls -1 "$LOG_ROOT/cadenza_localisation_v2/"*.json 2>/dev/null
