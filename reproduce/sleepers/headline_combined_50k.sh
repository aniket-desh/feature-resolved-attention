#!/usr/bin/env bash
# Sleeper headline figure: 1×2 combined panel (left = JSD vs α; right = layman
# rollout-level metrics). Reference 50k-feature SAE row only.
# Inputs : data/sleepers/sweep_full_metrics.json (from jsd_2x2_sweep_saeseed.py)
# Outputs: figures/sleeper_figures/sleeper_headline_combined_50k.{png,pdf}
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HERE/data}"
python3 "$HERE/experiments/sleepers/scripts/plot_combined_50k.py" \
    --input  "$DATA_ROOT/sleepers/jamie_50k/jsd_2x2_sweep_full_metrics.json" \
    --output "$HERE/figures/sleeper_figures/sleeper_headline_combined_50k"
