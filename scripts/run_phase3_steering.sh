#!/usr/bin/env bash
# After 4 SAEs are trained, run SAE-resid steering eval at each hookpoint, judge,
# build the comparison plot, fill phase3_benchmark.md, and push to GitHub.
#
# Usage on local (preferred):
#   SEEDS="42" RUN_NAME="phase3_20260508" bash scripts/run_phase3_steering.sh
#   SEEDS="42 123" bash scripts/run_phase3_steering.sh   # multi-seed round
set -euo pipefail

POD1="${POD1:-gpu_pod_1}"
POD2="${POD2:-gpu_pod_2}"
RUN_NAME="${RUN_NAME:?set RUN_NAME to the phase3 run dir name on the pods, e.g. phase3_20260508_0500}"
SEEDS="${SEEDS:-42}"
EM_MODEL="${EM_MODEL:-medical}"
OAI_KEY="${OPENAI_API_KEY:-${OPENAI_API_KEY:-}}"
PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT to the local project root}"

if [[ -z "$OAI_KEY" ]]; then
    echo "[FATAL] no OpenAI key (OPENAI_API_KEY preferred)"; exit 4
fi

# Per-pod allocation matches launch_phase3_saes.sh
declare -a HOOKS=(
    "$POD1 0 blocks.24.hook_resid_pre        24 resid_pre_L24"
    "$POD1 1 blocks.24.hook_resid_mid        24 resid_mid_L24"
    "$POD2 0 blocks.24.hook_resid_post       24 resid_post_L24"
    "$POD2 1 blocks.25.ln1.hook_normalized   25 ln1_normalised_L25"
)

echo "=== 1. Launch SAE-resid eval at each hookpoint (seeds: $SEEDS) ==="
for entry in "${HOOKS[@]}"; do
    read -r pod gpu hook layer outname <<<"$entry"
    sae_dir="/workspace/runs/${RUN_NAME}/sae_${outname}"
    out_dir="/workspace/runs/${RUN_NAME}/steer_${outname}_seed$(echo $SEEDS | tr ' ' '-')"
    log="/workspace/logs/steer_${outname}_seed$(echo $SEEDS | tr ' ' '-').log"
    cmd="cd /workspace/fra_proj && env TMPDIR=/workspace/tmp HF_HOME=/workspace/hf_cache CUDA_VISIBLE_DEVICES=${gpu} python3 -u -m fra.sae_resid_eval --em-model ${EM_MODEL} --hook-name ${hook} --sae-path ${sae_dir} --output ${out_dir} --seeds ${SEEDS}"
    echo "[launch] ${pod} gpu${gpu} → ${hook}"
    ssh "$pod" "nohup bash -c \"$cmd\" >${log} 2>&1 & echo PID=\$!"
done

echo "=== 2. Wait for all 4 to finish (poll qualitative_*.json) ==="
TARGETS=()
for entry in "${HOOKS[@]}"; do
    read -r pod gpu hook layer outname <<<"$entry"
    out_dir="/workspace/runs/${RUN_NAME}/steer_${outname}_seed$(echo $SEEDS | tr ' ' '-')"
    TARGETS+=("$pod:$out_dir")
done

while true; do
    all_done=true
    for t in "${TARGETS[@]}"; do
        pod="${t%%:*}"; dir="${t#*:}"
        n=$(ssh "$pod" "ls $dir/qualitative_*.json 2>/dev/null | wc -l" || echo 0)
        if [[ "$n" -lt 1 ]]; then all_done=false; fi
    done
    if $all_done; then echo "  all 4 steer runs done."; break; fi
    sleep 60
done

echo "=== 3. Judge each result dir with GPT-4o ==="
for t in "${TARGETS[@]}"; do
    pod="${t%%:*}"; dir="${t#*:}"
    if ssh "$pod" "ls $dir/gpt4o_aggregated_*.json 2>/dev/null | grep -q ."; then
        echo "  $pod:$dir already judged — skipping."
        continue
    fi
    echo "  judging $pod:$dir"
    ssh "$pod" "OPENAI_API_KEY='$OAI_KEY' python3 /workspace/fra_proj/judge_multiseed.py --results-dir $dir" >/dev/null
done

echo "=== 4. Pull all gpt4o_aggregated_*.json to local ==="
LOCAL_DIR="$PROJECT_ROOT/plots/2026-05-07_em_repl/phase3_seed$(echo $SEEDS | tr ' ' '-')"
mkdir -p "$LOCAL_DIR"
for t in "${TARGETS[@]}"; do
    pod="${t%%:*}"; dir="${t#*:}"
    name=$(basename "$dir")
    mkdir -p "$LOCAL_DIR/$name"
    scp "$pod:$dir/gpt4o_aggregated_*.json" "$LOCAL_DIR/$name/" 2>/dev/null || true
done
# Published QK→OV medical from Phase 1
mkdir -p "$LOCAL_DIR/orig_medical_qkov"
scp "$POD1:/workspace/runs/medical/gpt4o_aggregated_*.json" "$LOCAL_DIR/orig_medical_qkov/" 2>/dev/null || true

echo "=== 5. Build comparison plot ==="
ORIG_GPT=$(ls "$LOCAL_DIR/orig_medical_qkov"/gpt4o_aggregated_*.json | head -1)
SAE_ARGS=()
for entry in "${HOOKS[@]}"; do
    read -r pod gpu hook layer outname <<<"$entry"
    name="steer_${outname}_seed$(echo $SEEDS | tr ' ' '-')"
    f=$(ls "$LOCAL_DIR/$name"/gpt4o_aggregated_*.json 2>/dev/null | head -1 || true)
    if [[ -n "$f" ]]; then SAE_ARGS+=(--sae "${outname}=$f"); fi
done

OUT_BASE="$PROJECT_ROOT/plots/2026-05-07_em_repl/phase3_comparison_seed$(echo $SEEDS | tr ' ' '-')"
python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/plot_phase3_comparison.py" \
    --orig-medical "$ORIG_GPT" \
    "${SAE_ARGS[@]}" \
    --out "$OUT_BASE" \
    --title "Phase 3 — QK→OV vs same-budget SAE steering (medical EM, seeds=$SEEDS)"

echo "=== 6. Fill phase3_benchmark.md + push to GitHub ==="
python3 "$PROJECT_ROOT/scripts/fill_phase3_results.py" \
    --comparison-json "$OUT_BASE.json" \
    --doc "$PROJECT_ROOT/docs/em/phase3_benchmark.md" \
    --seeds $SEEDS

bash "$PROJECT_ROOT/scripts/auto_push_em_repl_summary.sh" \
    "phase 3: comparison plot at seeds=$SEEDS" \
    "docs/em/phase3_benchmark.md"

echo "=== Done. ==="
ls -la "$OUT_BASE".*
