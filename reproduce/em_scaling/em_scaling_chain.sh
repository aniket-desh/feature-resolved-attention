#!/bin/bash
# EM-scaling autoresearch chain.
#
# For each (base, domain) cell in the 6-cell grid below: run phase 1
# (FRA QK→QK), judge, run phase 2 (DoM), judge, then regen summary.md
# and auto-push.  After all cells complete, run the combine step per
# (recipe, base, domain) across eval seeds and regen one more time.
#
# Designed to be wall-clock-friendly on a single H200:
#   - one model loaded at a time
#   - small α-sweep (6 alphas) at the prompts the eval set already has
#   - judge inline (per cell) so failures land immediately
#
# Qwen-32B is intentionally left off this default grid: its SAE has to be
# trained first (run `python scripts/train_topk_sae_llama.py ...` adapted
# for Qwen-32B at L40 ln1.hook_normalized, ~3-4 hr).
#
# Override via env vars:
#   BASES="qwen-7b llama-8b"      DOMAINS="medical finance sports"
#   EVAL_SEEDS="42"               OUTPUT_ROOT=logs/em_scaling
#   SKIP_PUSH=1                   to skip auto-push (useful for local runs)
#
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PY=${PY:-.venv/bin/python}
if [ ! -x "$PY" ]; then
  echo "[chain] $PY not found; falling back to 'python' on PATH"
  PY=python
fi
LOG_ROOT=${LOG_ROOT:-logs/em_scaling}
BASES=${BASES:-"qwen-7b llama-8b"}
DOMAINS=${DOMAINS:-"medical finance sports"}
EVAL_SEEDS=${EVAL_SEEDS:-"42"}
ALPHAS_FRA=${ALPHAS_FRA:-"0.0 0.5 1.0 1.5 2.0 3.0"}
ALPHAS_DOM=${ALPHAS_DOM:-"0.0 0.5 1.0 1.5 2.0 3.0 4.0"}
N_PROMPTS=${N_PROMPTS:-8}
MAX_NEW=${MAX_NEW:-200}

mkdir -p "$LOG_ROOT/phase1_fra" "$LOG_ROOT/phase2_dom" "$LOG_ROOT/chain"

echo "[chain] grid: BASES=($BASES) × DOMAINS=($DOMAINS) × SEEDS=($EVAL_SEEDS)"
echo "[chain] alphas FRA=$ALPHAS_FRA  DoM=$ALPHAS_DOM  n_prompts=$N_PROMPTS"

push() {
  if [ "${SKIP_PUSH:-0}" = "1" ]; then
    echo "[chain] SKIP_PUSH=1 — staying local"
    return 0
  fi
  bash scripts/auto_push_em_scaling.sh "$1" || true
}

for base in $BASES; do
  for domain in $DOMAINS; do
    for seed in $EVAL_SEEDS; do
      echo
      echo "================================================================"
      echo "  CELL  $base / $domain   eval_seed=$seed   $(date)"
      echo "================================================================"

      # ── phase 1 FRA QK→QK ─────────────────────────────────────────────
      p1_qual="$LOG_ROOT/phase1_fra/qualitative_FRA_${base}_${domain}_evalseed${seed}.json"
      p1_log="$LOG_ROOT/chain/phase1_${base}_${domain}_seed${seed}.log"
      if [ -s "$p1_qual" ]; then
        echo "[chain] phase 1 cached: $p1_qual"
      else
        $PY -m experiments.em_scaling.phase1_fra_qkqk \
            --base "$base" --domain "$domain" --eval-seed "$seed" \
            --n-prompts "$N_PROMPTS" --max-new-tokens "$MAX_NEW" \
            --alphas $ALPHAS_FRA \
            --output-root "$LOG_ROOT/phase1_fra" \
            > "$p1_log" 2>&1
        echo "[chain] phase 1 done → $p1_qual"
      fi

      # judge phase 1 inline (skip if OPENAI_API_KEY absent)
      if [ -n "${OPENAI_API_KEY:-}" ]; then
        $PY -m experiments.em_scaling.phase_judge judge \
            --qualitative "$p1_qual" \
            > "${p1_log}.judge" 2>&1 \
          || echo "[chain] WARN: phase 1 judge failed for $base/$domain/$seed"
      else
        echo "[chain] OPENAI_API_KEY unset; deferring phase 1 judge."
      fi

      $PY -m experiments.em_scaling._regenerate_summary 2>&1 | tail -3 || true
      push "em_scaling: $base/$domain seed=$seed FRA rollouts done"

      # ── phase 2 DoM ───────────────────────────────────────────────────
      p2_qual="$LOG_ROOT/phase2_dom/qualitative_DoM_${base}_${domain}_evalseed${seed}.json"
      p2_log="$LOG_ROOT/chain/phase2_${base}_${domain}_seed${seed}.log"
      if [ -s "$p2_qual" ]; then
        echo "[chain] phase 2 cached: $p2_qual"
      else
        $PY -m experiments.em_scaling.phase2_dom_steering \
            --base "$base" --domain "$domain" --eval-seed "$seed" \
            --n-eval-prompts "$N_PROMPTS" --max-new-tokens "$MAX_NEW" \
            --alphas $ALPHAS_DOM \
            --output-root "$LOG_ROOT/phase2_dom" \
            > "$p2_log" 2>&1
        echo "[chain] phase 2 done → $p2_qual"
      fi

      if [ -n "${OPENAI_API_KEY:-}" ]; then
        $PY -m experiments.em_scaling.phase_judge judge \
            --qualitative "$p2_qual" \
            > "${p2_log}.judge" 2>&1 \
          || echo "[chain] WARN: phase 2 judge failed for $base/$domain/$seed"
      else
        echo "[chain] OPENAI_API_KEY unset; deferring phase 2 judge."
      fi

      $PY -m experiments.em_scaling._regenerate_summary 2>&1 | tail -3 || true
      push "em_scaling: $base/$domain seed=$seed DoM rollouts done"
    done
  done
done

# ── combine across eval seeds ────────────────────────────────────────────
echo
echo "================================================================"
echo "  COMBINE across eval seeds  $(date)"
echo "================================================================"
if [ -n "${OPENAI_API_KEY:-}" ]; then
  for base in $BASES; do
    for domain in $DOMAINS; do
      for recipe in FRA DoM; do
        stage="phase1_fra"; [ "$recipe" = "DoM" ] && stage="phase2_dom"
        pat="$LOG_ROOT/${stage}/gpt4o_aggregated_${recipe}_${base}_${domain}_evalseed*.json"
        out="$LOG_ROOT/${stage}/gpt4o_combined_${recipe}_${base}_${domain}.json"
        n=$(ls -1 $pat 2>/dev/null | wc -l)
        if [ "$n" -gt 0 ]; then
          $PY -m experiments.em_scaling.phase_judge combine \
              --pattern "$pat" --out "$out" \
            || echo "[chain] WARN: combine failed for $recipe/$base/$domain"
        fi
      done
    done
  done
else
  echo "[chain] OPENAI_API_KEY unset; skipping cross-seed combine step."
  echo "[chain] Run after judging:"
  echo "[chain]   python -m experiments.em_scaling.phase_judge judge --qualitative <FILE>"
  echo "[chain]   python -m experiments.em_scaling.phase_judge combine --pattern '<GLOB>' --out <FILE>"
fi

$PY -m experiments.em_scaling._regenerate_summary 2>&1 | tail -3 || true
push "em_scaling: final combine + summary regen"

echo
echo "[chain] EM-SCALING CHAIN COMPLETE  $(date)"
