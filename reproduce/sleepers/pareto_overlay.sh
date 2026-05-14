#!/usr/bin/env bash
# Pareto AUC: greedy ASR (sleeper-emission rate) vs teacher-forced ΔCE on
# clean tokens. Compares single-feature ablation, OV-top-3, OV-top-50.
# Inputs : data/sleepers/pareto_seed{0,1,2}.json
# Outputs: figures/sleeper_figures/sleeper_pareto_overlay.{png,pdf}
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HERE/data}"
python3 "$HERE/experiments/sleepers/scripts/plot_pareto_overlay.py" \
    --input-glob "$DATA_ROOT/sleepers/pareto_seed*.json" \
    --output     "$HERE/figures/sleeper_figures/sleeper_pareto_overlay"
