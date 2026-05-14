#!/usr/bin/env bash
# 2×2 JSD vs α figure: rows = SAE size (4k / 50k features); cols = recipe
# (single-feature ablation / OV-top-50). Demonstrates the 50k OV-top-50
# word-salad collapse.
# Inputs : data/sleepers/jsd_2x2_sweep_saeseed.json
# Outputs: figures/sleeper_figures/sleeper_jsd_2x2.{png,pdf}
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HERE/data}"
python3 "$HERE/experiments/sleepers/scripts/plot_jsd_2x2.py" \
    --input  "$DATA_ROOT/sleepers/jsd_2x2_sweep_saeseed.json" \
    --output "$HERE/figures/sleeper_figures/sleeper_jsd_2x2"
