#!/usr/bin/env bash
# Launch the 4 same-budget SAE training jobs across the 2 H100 pods.
# Pod 1 GPU 0  →  blocks.24.hook_resid_pre
# Pod 1 GPU 1  →  blocks.24.hook_resid_mid
# Pod 2 GPU 0  →  blocks.24.hook_resid_post
# Pod 2 GPU 1  →  blocks.25.ln1.hook_normalized
#
# Usage:
#   bash scripts/launch_phase3_saes.sh         # dry-run (prints commands)
#   GO=1 bash scripts/launch_phase3_saes.sh    # actually launch
set -euo pipefail

POD1="${POD1:-gpu_pod_1}"
POD2="${POD2:-gpu_pod_2}"
RUN_NAME="${RUN_NAME:-phase3_$(date +%Y%m%d_%H%M)}"

# Default 100M tokens for overnight viability across 4 H100s (~1–2 hr/SAE wall).
# Bump to 200_000_000 to match the published step count exactly if quality demands.
TRAIN_TOKENS="${TRAIN_TOKENS:-100000000}"

launch_remote () {
    local pod=$1 gpu=$2 hook=$3 layer=$4 outname=$5
    local outdir="/workspace/runs/${RUN_NAME}/sae_${outname}"
    local logfile="/workspace/logs/sae_${outname}.log"
    local cmd="cd /workspace/fra_proj && env TMPDIR=/workspace/tmp HF_HOME=/workspace/hf_cache HUGGINGFACE_HUB_CACHE=/workspace/hf_cache CUDA_VISIBLE_DEVICES=${gpu} python3 -u -m fra.train_sae_at_hookpoint --hook-name '${hook}' --hook-layer ${layer} --output-dir '${outdir}' --training-tokens ${TRAIN_TOKENS}"

    if [[ "${GO:-0}" == "1" ]]; then
        echo "[launch] ${pod} gpu${gpu} → ${outname}"
        ssh "${pod}" "mkdir -p /workspace/logs && nohup bash -c \"${cmd}\" >${logfile} 2>&1 & echo PID=\$!"
    else
        echo "[dry-run] ssh ${pod} 'nohup bash -c \"${cmd}\" >${logfile} 2>&1 &'"
    fi
}

launch_remote "$POD1" 0 "blocks.24.hook_resid_pre"        24 "resid_pre_L24"
launch_remote "$POD1" 1 "blocks.24.hook_resid_mid"        24 "resid_mid_L24"
launch_remote "$POD2" 0 "blocks.24.hook_resid_post"       24 "resid_post_L24"
launch_remote "$POD2" 1 "blocks.25.ln1.hook_normalized"   25 "ln1_normalised_L25"

echo
if [[ "${GO:-0}" == "1" ]]; then
    echo "Launched all 4 SAE jobs. Tail logs with:"
    echo "  ssh ${POD1} 'tail -F /workspace/logs/sae_resid_pre_L24.log'"
    echo "Push checkpoints to HF with:"
    echo "  python -m fra.hf_upload /workspace/runs/${RUN_NAME} phase3_benchmark/${RUN_NAME}"
else
    echo "Dry-run complete. Re-run with GO=1 to launch."
fi
