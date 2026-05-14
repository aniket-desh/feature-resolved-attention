"""Train a same-budget SAE on Qwen2.5-14B-Instruct at any TransformerLens hookpoint.

Matches the published `<anonymous>/Qwen2.5-14B_SAE_ln1.normalised` budget exactly
(d_sae = 102_400, k = 64, normalize_activations="expected_average_only_in") so Phase 3
benchmark cells are apples-to-apples with the L24 ln1.hook_normalized baseline.

Adapt --hook-name and --hook-layer per hookpoint:

    blocks.24.hook_resid_pre
    blocks.24.hook_resid_mid
    blocks.24.hook_resid_post
    blocks.25.ln1.hook_normalized

Tested against sae-lens 6.43 API (TopKTrainingSAEConfig + LanguageModelSAERunnerConfig).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hook-name", required=True, help="e.g. blocks.24.hook_resid_pre")
    p.add_argument("--hook-layer", type=int, required=True)
    p.add_argument("--output-dir", required=True,
                   help="Where to write cfg.json + sae_weights.safetensors + checkpoints/")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--d-in", type=int, default=5120, help="Qwen 14B d_model")
    p.add_argument("--d-sae", type=int, default=102_400, help="match published SAE")
    p.add_argument("--k", type=int, default=64, help="TopK k; match published SAE")
    p.add_argument("--dataset-path", default="monology/pile-uncopyrighted",
                   help="streaming HF text dataset (parquet — no remote code) for activation generation")
    p.add_argument("--context-size", type=int, default=1024)
    p.add_argument("--training-tokens", type=int, default=200_000_000,
                   help="200M default; matches the published ae_200000 step file at ~4096 tok/batch.")
    p.add_argument("--train-batch-size-tokens", type=int, default=4096)
    p.add_argument("--n-batches-in-buffer", type=int, default=64)
    p.add_argument("--store-batch-size-prompts", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-warm-up-steps", type=int, default=1000)
    p.add_argument("--n-checkpoints", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Build config and exit without training")
    args = p.parse_args()

    Path(args.output_dir).expanduser().mkdir(parents=True, exist_ok=True)

    from sae_lens import TopKTrainingSAEConfig, LanguageModelSAERunnerConfig
    from sae_lens.config import LoggingConfig
    from sae_lens.saes.sae import SAEMetadata

    # SAE in fp32 for numerical stability; model loaded in bf16 separately
    # below via model_from_pretrained_kwargs to keep the 14B model footprint
    # at ~28GB instead of 56GB.
    sae_cfg = TopKTrainingSAEConfig(
        d_in=args.d_in,
        d_sae=args.d_sae,
        k=args.k,
        normalize_activations="expected_average_only_in",
        dtype="float32",
        device=args.device,
        metadata=SAEMetadata(
            model_name=args.model_name,
            hook_name=args.hook_name,
            hook_layer=args.hook_layer,
        ),
    )

    runner_cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=args.model_name,
        hook_name=args.hook_name,
        dataset_path=args.dataset_path,
        is_dataset_tokenized=False,
        streaming=True,
        context_size=args.context_size,
        prepend_bos=True,
        training_tokens=args.training_tokens,
        train_batch_size_tokens=args.train_batch_size_tokens,
        n_batches_in_buffer=args.n_batches_in_buffer,
        store_batch_size_prompts=args.store_batch_size_prompts,
        lr=args.lr,
        lr_scheduler_name="cosineannealing",
        lr_warm_up_steps=args.lr_warm_up_steps,
        dataset_trust_remote_code=False,
        n_checkpoints=args.n_checkpoints,
        checkpoint_path=str(Path(args.output_dir).expanduser()),
        save_final_checkpoint=True,
        device=args.device,
        dtype="float32",
        autocast=False,
        # Keep the 14B model in bf16 even though SAE training is in fp32.
        # sae-lens passes these kwargs to HookedTransformer.from_pretrained.
        model_from_pretrained_kwargs={"dtype": "bfloat16"},
        seed=args.seed,
        logger=LoggingConfig(log_to_wandb=False),
        verbose=True,
    )

    print("=== SAE training config ===")
    print(f"  model: {args.model_name}")
    print(f"  hook : {args.hook_name}  (layer {args.hook_layer})")
    print(f"  d_in : {args.d_in}    d_sae: {args.d_sae}    k: {args.k}")
    print(f"  dataset : {args.dataset_path}  (streaming)")
    print(f"  train_tokens: {args.training_tokens:,}")
    print(f"  batch tokens: {args.train_batch_size_tokens}")
    print(f"  output: {args.output_dir}")

    if args.dry_run:
        print("--dry-run: exiting before training.")
        return

    from sae_lens import SAETrainingRunner
    runner = SAETrainingRunner(runner_cfg)
    runner.run()
    print(f"\n=== Training complete. SAE under {args.output_dir} ===")


if __name__ == "__main__":
    sys.exit(main())
