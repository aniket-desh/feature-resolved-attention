#!/usr/bin/env bash
# 3 (eval seeds) × 5 (hookpoints) alignment-vs-coherence trajectory grid for
# one EM domain. Usage: bash reproduce/em/seed_grid.sh <domain>
#   <domain> ∈ {medical, finance, sports}
# Inputs : data/em/combined/gpt4o_combined_*.json
# Outputs: figures/em_figures/em_seed_grid_<domain>.{png,pdf}
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HERE/data}"
DOMAIN="${1:?usage: seed_grid.sh <medical|finance|sports>}"
python3 "$HERE/scripts/plot_phase1_seed_grid.py" \
    --combined-root "$DATA_ROOT/em/combined" \
    --domain "$DOMAIN" \
    --out "$HERE/figures/em_figures/em_seed_grid_${DOMAIN}"
