#!/usr/bin/env bash
# Post-eval pipeline for the 2026-05-09 negative-α extension.
#
# 1. SCP per-stream qualitative JSONs from each pod into a flat local
#    layout: <local_root>/<em>_<seed>/qualitative_*.json (mixing add + fra).
# 2. Judge & aggregate per stream + combine across seeds with
#    phase1_judge_and_combine.py.
# 3. Re-render the four plot families (1×3 panel, headline 3-bar, three
#    seed-grids).
# 4. Copy figures into fra_proj/figures/em_figures, project figures, and
#    tex figures.
#
# Run locally:  bash scripts/run_postprocess_neg6.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:?set LOCAL_ROOT to the local streams directory}"
TEX_FIGS="${TEX_FIGS:?set TEX_FIGS to the tex figures directory}"
PROJECT_FIGS="${PROJECT_FIGS:?set PROJECT_FIGS to the project figures directory}"
FRA_FIGS="$REPO_ROOT/figures/em_figures"

mkdir -p "$LOCAL_ROOT"

# Stream → pod mapping
declare -a STREAMS=(
  "gpu_pod_1:medical_42"
  "gpu_pod_1:medical_123"
  "gpu_pod_2:medical_456"
  "gpu_pod_2:finance_42"
  "gpu_pod_3:finance_123"
  "gpu_pod_3:finance_456"
  "gpu_pod_1:sports_42"
  "gpu_pod_1:sports_123"
  "gpu_pod_2:sports_456"
)

# ── 1. SCP & flatten ──────────────────────────────────────────────────────
echo "[1/4] pulling qualitative JSONs from pods…"
for spec in "${STREAMS[@]}"; do
  pod=${spec%%:*}; stream=${spec##*:}
  mkdir -p "$LOCAL_ROOT/$stream"
  scp "$pod:/workspace/eval_2026-05-09_neg6/$stream/add/qualitative_*.json" "$LOCAL_ROOT/$stream/" 2>&1 | tail -1 || true
  scp "$pod:/workspace/eval_2026-05-09_neg6/$stream/fra/qualitative_FRA_*.json" "$LOCAL_ROOT/$stream/" 2>&1 | tail -1 || true
done
echo "  pulled to $LOCAL_ROOT/"

# ── 2. Judge + per-(sae,domain) cross-seed combine ────────────────────────
echo "[2/4] judging + combining across seeds…"
OPENAI_API_KEY="$OPENAI_API_KEY" python3 "$REPO_ROOT/phase1_judge_and_combine.py" \
    --stream-root "$LOCAL_ROOT" 2>&1 | tail -20

# ── 3. Re-render plots ────────────────────────────────────────────────────
echo "[3/4] re-rendering plots…"
mkdir -p "$FRA_FIGS"
python3 "$REPO_ROOT/scripts/plot_phase1_fra_plus_additive.py" \
    --combined-root "$LOCAL_ROOT" \
    --out "$FRA_FIGS/phase1_fra_plus_additive_3domains_neg6" 2>&1 | tail -1
python3 "$REPO_ROOT/scripts/plot_phase1_headline_per_domain.py" \
    --combined-root "$LOCAL_ROOT" \
    --out "$FRA_FIGS/phase1_headline_per_domain_neg6" 2>&1 | tail -1
for em in medical finance sports; do
  python3 "$REPO_ROOT/scripts/plot_phase1_seed_grid.py" \
      --combined-root "$LOCAL_ROOT" \
      --domain "$em" \
      --out "$FRA_FIGS/phase1_seed_grid_${em}_neg6" 2>&1 | tail -1
done

# ── 4. Copy figures everywhere ────────────────────────────────────────────
echo "[4/4] copying figures to project + tex dirs…"
cp "$FRA_FIGS"/phase1_*_neg6.{png,pdf} "$PROJECT_FIGS/"
cp "$FRA_FIGS"/phase1_*_neg6.{png,pdf} "$TEX_FIGS/"
echo "done. inspect:"
echo "  $FRA_FIGS/phase1_headline_per_domain_neg6.png"
echo "  $FRA_FIGS/phase1_fra_plus_additive_3domains_neg6.png"
echo "  $FRA_FIGS/phase1_seed_grid_{medical,finance,sports}_neg6.png"
