#!/bin/bash
# ============================================================================
# Run all multi-seed experiments for the FRA paper.
#
# Usage (on the pod):
#   bash run_all_multiseed.sh           # run everything
#   bash run_all_multiseed.sh --skip-frontier  # skip frontier, do CE + random only
#
# Requires: ~141GB VRAM for CE-vs-base (loads 2 models), H100 recommended.
# Results are saved to /root/multiseed_results_v2/ and tarred for download.
# ============================================================================

set -e  # exit on error

OUTDIR="/root/multiseed_results_v2"
SEEDS="42 123 456"
N_TEXTS=8    # all 8 EM prompts
LAYER=24
HEAD=38
TEMP=1.0

mkdir -p "$OUTDIR"

log() { echo -e "\n$(date '+%H:%M:%S') === $1 ===\n"; }

# ── 1. Multi-seed frontier sweeps (multi-prompt ranking) ──────────────
if [[ "$1" != "--skip-frontier" ]]; then

log "FRONTIER: finance"
python run_experiments.py --task frontier_multiseed --em-model finance \
    --head $HEAD --seeds $SEEDS --temperature $TEMP --n-texts $N_TEXTS \
    --output "$OUTDIR"

log "FRONTIER: medical"
python run_experiments.py --task frontier_multiseed --em-model medical \
    --head $HEAD --seeds $SEEDS --temperature $TEMP --n-texts $N_TEXTS \
    --output "$OUTDIR"

log "FRONTIER: sports"
python run_experiments.py --task frontier_multiseed --em-model sports \
    --head $HEAD --seeds $SEEDS --temperature $TEMP --n-texts $N_TEXTS \
    --output "$OUTDIR"

# ── 2. Shared feature cross-head sweeps ───────────────────────────────
log "SHARED FEATURE: finance"
python run_experiments.py --task shared_feature_multiseed --em-model finance \
    --seeds $SEEDS --temperature $TEMP --n-texts $N_TEXTS \
    --output "$OUTDIR"

log "SHARED FEATURE: medical"
python run_experiments.py --task shared_feature_multiseed --em-model medical \
    --seeds $SEEDS --temperature $TEMP --n-texts $N_TEXTS \
    --output "$OUTDIR"

log "SHARED FEATURE: sports"
python run_experiments.py --task shared_feature_multiseed --em-model sports \
    --seeds $SEEDS --temperature $TEMP --n-texts $N_TEXTS \
    --output "$OUTDIR"

fi  # --skip-frontier

# ── 3. Random feature baseline (control) ──────────────────────────────
log "RANDOM BASELINE: finance"
python run_experiments.py --task random_baseline --em-model finance \
    --head $HEAD --seeds $SEEDS --temperature $TEMP --n-texts $N_TEXTS \
    --output "$OUTDIR"

log "RANDOM BASELINE: medical"
python run_experiments.py --task random_baseline --em-model medical \
    --head $HEAD --seeds $SEEDS --temperature $TEMP --n-texts $N_TEXTS \
    --output "$OUTDIR"

log "RANDOM BASELINE: sports"
python run_experiments.py --task random_baseline --em-model sports \
    --head $HEAD --seeds $SEEDS --temperature $TEMP --n-texts $N_TEXTS \
    --output "$OUTDIR"

# ── 4. CE vs base model (deterministic, no seeds needed) ─────────────
log "CE VS BASE: finance"
python run_experiments.py --task ce_vs_base --em-model finance \
    --head $HEAD --n-texts $N_TEXTS \
    --output "$OUTDIR/ce_vs_base_finance_L${LAYER}_H${HEAD}.json"

log "CE VS BASE: medical"
python run_experiments.py --task ce_vs_base --em-model medical \
    --head $HEAD --n-texts $N_TEXTS \
    --output "$OUTDIR/ce_vs_base_medical_L${LAYER}_H${HEAD}.json"

log "CE VS BASE: sports"
python run_experiments.py --task ce_vs_base --em-model sports \
    --head $HEAD --n-texts $N_TEXTS \
    --output "$OUTDIR/ce_vs_base_sports_L${LAYER}_H${HEAD}.json"

# ── 5. Package results ────────────────────────────────────────────────
log "PACKAGING"

# Collect qualitative files from default output location too
cp /root/multiseed_results/qualitative_*.json "$OUTDIR/" 2>/dev/null || true
cp /root/multiseed_results/qualitative_*.md "$OUTDIR/" 2>/dev/null || true

# Create tarball
cd /root
tar czf multiseed_results_v2.tar.gz multiseed_results_v2/
echo "Results tarball: /root/multiseed_results_v2.tar.gz"
echo "Download with: scp gpu_pod_1:/root/multiseed_results_v2.tar.gz ."

log "ALL DONE"
ls -lh "$OUTDIR/"
