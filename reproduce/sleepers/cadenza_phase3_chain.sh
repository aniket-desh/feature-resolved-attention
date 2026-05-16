#!/bin/bash
# Autoresearch phase-3 chain: runs the three follow-on experiments from
# Dmitry's review (4-way comparison, late-layer locality, own-dataset
# SAE), regenerating summary.md after each step.
#
#   3a (~45 min)  4-hookpoint comparison at L29 (FRA-OV vs conventional×3)
#   3b (~4.5 hr)  L28/L30/L31 hook_resid_post locality probe
#   3c (~1.5 hr)  SAE retrained on Cadenza's own IHY dataset
#
# After every step finishes, calls _regenerate_summary.py to refresh the
# auto-updated section of summary.md.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PY=${PY:-/usr/bin/python3}
LOG_ROOT=${LOG_ROOT:-logs}
mkdir -p "$LOG_ROOT/cadenza_phase3"

ckpt() {
  find "$1" -name "sae_weights.safetensors" -printf "%h\n" 2>/dev/null \
    | sort -t/ -k6 -n | tail -1
}
SAE_L29_LN1=$(ckpt /workspace/aniket/saes/cadenza_L29_ln1_hook_normalized)
SAE_L29_MID=$(ckpt /workspace/aniket/saes/cadenza_L29_hook_resid_mid)
SAE_L29_POST=$(ckpt /workspace/aniket/saes/cadenza_L29_hook_resid_post)

echo "[chain3] SAEs in use:"
echo "  L29/ln1        : $SAE_L29_LN1"
echo "  L29/resid_mid  : $SAE_L29_MID"
echo "  L29/resid_post : $SAE_L29_POST"
echo

regen() {
  echo "[chain3] regenerating summary.md ..."
  "$PY" -m experiments.sleepers.cadenza._regenerate_summary \
    || echo "[chain3] WARN: summary regen failed"
}

# Refresh once at the start so anything already on disk is captured.
regen

# ── 3a — 4-way comparison ──────────────────────────────────────────────
echo
echo "================================================================"
echo "  PHASE-3 STEP A — 4-way comparison at L29  $(date)"
echo "================================================================"
step_log="$LOG_ROOT/cadenza_phase3/step3a_4way.log"
"$PY" -m experiments.sleepers.cadenza.phase3_4way \
    --fra-sae-path  "$SAE_L29_LN1" \
    --mid-sae-path  "$SAE_L29_MID" \
    --post-sae-path "$SAE_L29_POST" \
    --layer 29 \
    --alphas -4.0 -2.0 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0 3.0 4.0 \
    --n-val 8 --n-test 20 \
    --out "$LOG_ROOT/cadenza_phase3/4way_metrics.json" \
    > "$step_log" 2>&1
echo "[chain3] step 3a finished $(date)"
regen

# ── 3b — late-layer locality probe ─────────────────────────────────────
echo
echo "================================================================"
echo "  PHASE-3 STEP B — L28/L30/L31 resid_post sweep  $(date)"
echo "================================================================"
step_log="$LOG_ROOT/cadenza_phase3/step3b_late_layer.log"
bash reproduce/sleepers/cadenza_late_layer_sweep.sh > "$step_log" 2>&1
echo "[chain3] step 3b finished $(date)"
regen

# ── 3c — own-dataset SAE retrain ───────────────────────────────────────
echo
echo "================================================================"
echo "  PHASE-3 STEP C — SAE on Cadenza's own IHY dataset  $(date)"
echo "================================================================"
step_log="$LOG_ROOT/cadenza_phase3/step3c_own_dataset.log"
bash reproduce/sleepers/cadenza_own_dataset_sae.sh > "$step_log" 2>&1
echo "[chain3] step 3c finished $(date)"
regen

echo
echo "================================================================"
echo "[chain3] PHASE-3 CHAIN COMPLETE  $(date)"
echo "================================================================"
ls -1 "$LOG_ROOT/cadenza_phase3/"*.json 2>/dev/null || true
ls -1 "$LOG_ROOT/cadenza_late_layers/"*.json 2>/dev/null || true
ls -1 "$LOG_ROOT/cadenza_own_dataset/"*.json 2>/dev/null || true
