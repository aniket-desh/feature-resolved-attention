#!/usr/bin/env bash
# 1×3 cross-domain detail panel — three FRA recipes + best conventional, per
# domain (medical / finance / sports). Used as the main-text intervention figure.
# Inputs : data/em/combined/gpt4o_combined_*.json
# Outputs: figures/em_figures/em_fra_plus_additive_3domains.{png,pdf}
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HERE/data}"
python3 "$HERE/scripts/plot_phase1_fra_plus_additive.py" \
    --combined-root "$DATA_ROOT/em/combined" \
    --out "$HERE/figures/em_figures/em_fra_plus_additive_3domains"
