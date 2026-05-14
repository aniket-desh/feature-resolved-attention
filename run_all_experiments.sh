#!/bin/bash
# =============================================================
# Run all experiments for the ICML workshop paper
# Run this on the H200 pod after setup
# =============================================================
set -e

export HF_HOME=/root/hf_cache
# Set these before running:
#   export HF_TOKEN=hf_your_token
#   export OPENAI_API_KEY=sk-your_key
export PYTHONUNBUFFERED=1

if [ -z "$HF_TOKEN" ] || [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: Set HF_TOKEN and OPENAI_API_KEY environment variables first"
    exit 1
fi

cd /root/fra_proj
OUTDIR=/root/results
mkdir -p $OUTDIR

echo "============================================================"
echo "EXPERIMENT 1: Frontier sweep — single feature (k=1)"
echo "  Finance model, H38, 8 prompts"
echo "  This is what the authors wants: does ONE feature move alignment?"
echo "============================================================"
python run_experiments.py \
    --task frontier \
    --em-model finance \
    --head 38 \
    --layer 24 \
    --k 1 \
    --n-texts 8 \
    --output $OUTDIR/frontier_finance_H38_k1.json \
    2>&1 | tee $OUTDIR/log_frontier_finance_H38_k1.txt

echo ""
echo "============================================================"
echo "EXPERIMENT 2: Frontier sweep — top 50 pairs (~25 features)"
echo "  Finance model, H38, 8 prompts"
echo "  Compare: does more features = bigger effect or just noise?"
echo "============================================================"
python run_experiments.py \
    --task frontier \
    --em-model finance \
    --head 38 \
    --layer 24 \
    --k 50 \
    --n-texts 8 \
    --output $OUTDIR/frontier_finance_H38_k50.json \
    2>&1 | tee $OUTDIR/log_frontier_finance_H38_k50.txt

echo ""
echo "============================================================"
echo "EXPERIMENT 3: Frontier sweep — single feature, H36"
echo "  H36 had negative loss_delta (misalignment carrier)"
echo "============================================================"
python run_experiments.py \
    --task frontier \
    --em-model finance \
    --head 36 \
    --layer 24 \
    --k 1 \
    --n-texts 8 \
    --output $OUTDIR/frontier_finance_H36_k1.json \
    2>&1 | tee $OUTDIR/log_frontier_finance_H36_k1.txt

echo ""
echo "============================================================"
echo "EXPERIMENT 4: Frontier sweep — medical model"
echo "  Single feature, H38, medical EM"
echo "  Cross-variant: does the same head/method work?"
echo "============================================================"
python run_experiments.py \
    --task frontier \
    --em-model medical \
    --head 38 \
    --layer 24 \
    --k 1 \
    --n-texts 8 \
    --output $OUTDIR/frontier_medical_H38_k1.json \
    2>&1 | tee $OUTDIR/log_frontier_medical_H38_k1.txt

echo ""
echo "============================================================"
echo "EXPERIMENT 5: Frontier sweep — sports model"
echo "  Single feature, H38, sports EM"
echo "============================================================"
python run_experiments.py \
    --task frontier \
    --em-model sports \
    --head 38 \
    --layer 24 \
    --k 1 \
    --n-texts 8 \
    --output $OUTDIR/frontier_sports_H38_k1.json \
    2>&1 | tee $OUTDIR/log_frontier_sports_H38_k1.txt

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS DONE"
echo "Results in $OUTDIR/"
echo "============================================================"
ls -lh $OUTDIR/

echo ""
echo "Packing results for download..."
cd /root
tar czf /root/all_experiment_results.tar.gz results/
echo "Ready: /root/all_experiment_results.tar.gz"
echo "Download with: runpodctl send /root/all_experiment_results.tar.gz"
