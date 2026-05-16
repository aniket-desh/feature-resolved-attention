#!/bin/bash
# Autoresearch master chain — runs the four phase-2 follow-on experiments
# sequentially with no user input. Total wall-clock ~10 hr on H200.
#
#   step 1 (~1.5 hr)  phase1 single-feature headline at L29 (combined_50k analog)
#   step 2 (~3.5 hr)  attribution × intervention matrix at L29 (matrix_scatter analog)
#   step 3 (~30 min)  mech-interp on feat 12402 (Dmitry's "why negative α only")
#   step 4 (~4.5 hr)  multi-SAE-seed robustness at L29/hook_resid_post
#
# After each step finishes, prints a one-line success/failure marker that
# the persistent Monitor catches as an event.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PY=${PY:-/usr/bin/python3}
LOG_ROOT=${LOG_ROOT:-logs}
mkdir -p "$LOG_ROOT/cadenza_phase2"

# ── locate SAE checkpoints (the deepest one inside each cell dir) ──────────
ckpt() {
  find "$1" -name "sae_weights.safetensors" -printf "%h\n" 2>/dev/null \
    | sort -t/ -k6 -n | tail -1
}
SAE_L29_LN1=$(ckpt /workspace/aniket/saes/cadenza_L29_ln1_hook_normalized)
SAE_L29_MID=$(ckpt /workspace/aniket/saes/cadenza_L29_hook_resid_mid)
SAE_L29_POST=$(ckpt /workspace/aniket/saes/cadenza_L29_hook_resid_post)
echo "[chain] SAEs in use:"
echo "  L29/ln1        : $SAE_L29_LN1"
echo "  L29/resid_mid  : $SAE_L29_MID"
echo "  L29/resid_post : $SAE_L29_POST"
echo

# ── step 1 ─────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  PHASE-2 STEP 1 — phase1 single-feature headline at L29  $(date)"
echo "================================================================"
step1_log="$LOG_ROOT/cadenza_phase2/step1_single_feature.log"
"$PY" -m experiments.sleepers.cadenza.phase1_single_feature \
    --fra-sae-path  "$SAE_L29_LN1" \
    --conv-sae-path "$SAE_L29_MID" \
    --layer 29 \
    --alphas -4.0 -2.0 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0 3.0 4.0 \
    --n-prompts 20 \
    --out "$LOG_ROOT/cadenza_phase2/step1_single_feature_metrics.json" \
    > "$step1_log" 2>&1
echo "[chain] step 1 finished $(date)"

# Render the combined_50k plot using the existing plotter
"$PY" experiments/sleepers/scripts/plot_combined_50k.py \
    --input  "$LOG_ROOT/cadenza_phase2/step1_single_feature_metrics.json" \
    --output experiments/sleepers/cadenza/figures/phase2_combined_50k \
    --n_prompts 20 \
    --title "Cadenza Llama-3 8B — single OV vs conventional resid-mid (L29)" \
    > "${step1_log}.plot" 2>&1 || echo "[chain] WARN: combined_50k plot failed"
"$PY" -m experiments.sleepers.cadenza._regenerate_summary 2>&1 \
    | tail -3 || echo "[chain] WARN: summary regen failed (step 1)"

# ── step 2 ─────────────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "  PHASE-2 STEP 2 — attribution matrix at L29  $(date)"
echo "================================================================"
step2_log="$LOG_ROOT/cadenza_phase2/step2_attribution_matrix.log"
"$PY" -m experiments.sleepers.cadenza.phase1_attribution_matrix \
    --sae-path "$SAE_L29_LN1" \
    --layer 29 \
    --hook-point ln1.hook_normalized \
    --heads 0 8 16 24 \
    --n-val 10 --n-eval 10 \
    --alphas -2.0 -1.0 0.0 1.0 2.0 \
    --out "$LOG_ROOT/cadenza_phase2/step2_attribution_matrix.json" \
    > "$step2_log" 2>&1
echo "[chain] step 2 finished $(date)"
"$PY" -m experiments.sleepers.cadenza._regenerate_summary 2>&1 \
    | tail -3 || echo "[chain] WARN: summary regen failed (step 2)"

# ── step 3 ─────────────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "  PHASE-2 STEP 3 — mech-interp on feat 12402  $(date)"
echo "================================================================"
step3_log="$LOG_ROOT/cadenza_phase2/step3_mechinterp.log"
"$PY" -m experiments.sleepers.cadenza.phase2_mechinterp \
    --feature 12402 --layer 29 --n-prompts 30 \
    --out-md experiments/sleepers/cadenza/mechinterp_report.md \
    > "$step3_log" 2>&1
echo "[chain] step 3 finished $(date)"
"$PY" -m experiments.sleepers.cadenza._regenerate_summary 2>&1 \
    | tail -3 || echo "[chain] WARN: summary regen failed (step 3)"

# ── step 4 ─────────────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "  PHASE-2 STEP 4 — multi-SAE-seed robustness  $(date)"
echo "================================================================"
step4_log="$LOG_ROOT/cadenza_phase2/step4_multi_seed.log"
bash reproduce/sleepers/cadenza_multi_seed.sh > "$step4_log" 2>&1
echo "[chain] step 4 finished $(date)"
"$PY" -m experiments.sleepers.cadenza._regenerate_summary 2>&1 \
    | tail -3 || echo "[chain] WARN: summary regen failed (step 4)"

echo
echo "================================================================"
echo "[chain] PHASE-2 CHAIN COMPLETE  $(date)"
echo "================================================================"
echo "results:"
ls -1 "$LOG_ROOT/cadenza_phase2/"*.json 2>/dev/null
ls -1 "$LOG_ROOT/cadenza_multi_seed/"*.json 2>/dev/null
ls -1 experiments/sleepers/cadenza/mechinterp_report.* 2>/dev/null
ls -1 experiments/sleepers/cadenza/figures/phase2_*.png 2>/dev/null
