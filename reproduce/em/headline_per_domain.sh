#!/usr/bin/env bash
# Headline 3-bar EM figure: best Δalign|coh≥70 per domain.
# Inputs : data/em/combined/gpt4o_combined_*.json
# Outputs: figures/em_figures/em_headline_per_domain.{png,pdf}
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HERE/data}"
python3 "$HERE/scripts/plot_phase1_headline_per_domain.py" \
    --combined-root "$DATA_ROOT/em/combined" \
    --out "$HERE/figures/em_figures/em_headline_per_domain"
