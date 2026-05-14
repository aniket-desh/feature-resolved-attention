#!/bin/bash
# Run this ON the RunPod pod after SSH-ing in
set -e

echo "=== Setting up FRA on RunPod Pod ==="

cd /workspace/fra_proj

# Install dependencies (torch is already installed on RunPod PyTorch template)
pip install transformer_lens sae_lens huggingface_hub safetensors einops tqdm scikit-learn sentence-transformers matplotlib 2>&1 | tail -5

# Login to HuggingFace for model access
# Set your HuggingFace token here:
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
huggingface-cli login --token $HF_TOKEN

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')"

echo ""
echo "=== Setup complete. Run experiments with: ==="
echo "  python run_experiments.py"
