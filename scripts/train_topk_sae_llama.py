"""Train a TopK SAE on the Cadenza Llama-3 8B sleeper at a chosen hookpoint.

Adapt of ``fra/train_sae_at_hookpoint.py`` (the published Qwen-14B trainer).
Same sae-lens 6.43 API (``TopKTrainingSAEConfig`` + ``LanguageModelSAERunnerConfig``
+ ``SAETrainingRunner``); the only differences are:

  1. Defaults tuned for Llama-3 8B: ``d_in=4096``, ``d_sae=32_768`` (8x expansion
     — large enough to surface a clean sleeper feature, small enough that 9 cells
     fit in an overnight sweep), ``k=64``, ``training_tokens=50_000_000``.
  2. The HF weights are pre-loaded from the Cadenza checkpoint and threaded into
     sae-lens via ``model_from_pretrained_kwargs={"hf_model": ...}``, so the SAE
     trains on the *sleeper's* activations rather than vanilla Llama-3.1-base.
     The TransformerLens architecture template stays ``Llama-3.1-8B-Instruct``
     (8B layout is identical across Llama-3 / 3.1).
  3. Activation text comes from ``monology/pile-uncopyrighted`` — same generic
     web-text corpus the in-repo Qwen trainer uses, no synthetic-trigger boost.
     The point of the localisation experiment is to find a sleeper feature
     *without* having seeded the SAE training with the trigger distribution.

Per Dmitry's recommendation, the default cell grid is ``layers={3, 16, 29}`` ×
``hookpoints={ln1.hook_normalized, hook_resid_mid, hook_resid_post}``. This
script trains one cell at a time; a wrapper shell loops over the grid.

Typical invocation::

    python scripts/train_topk_sae_llama.py \\
        --hook-layer 16 \\
        --hook-point ln1.hook_normalized \\
        --output-dir /workspace/aniket/saes/cadenza_L16_ln1 \\
        --training-tokens 50000000

For a quick smoke / dry-run of just the config::

    python scripts/train_topk_sae_llama.py \\
        --hook-layer 16 --hook-point ln1.hook_normalized \\
        --output-dir /tmp/dryrun --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── Defaults that match the Cadenza target ───────────────────────────────

DEFAULT_TL_TEMPLATE = "meta-llama/Llama-3.1-8B-Instruct"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hook-layer", type=int, required=True,
                   help="Transformer block index, e.g. 16.")
    p.add_argument("--hook-point", required=True,
                   help="Hookpoint suffix relative to the block, e.g. "
                        "ln1.hook_normalized, hook_resid_mid, hook_resid_post.")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write cfg.json + sae_weights.safetensors + checkpoints/.")
    # Cadenza-specific
    p.add_argument(
        "--cadenza-repo",
        default="Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-distilled-lora",
        help="HF repo of the sleeper checkpoint to extract activations from.",
    )
    p.add_argument(
        "--tl-template", default=DEFAULT_TL_TEMPLATE,
        help="HookedTransformer architecture template (registry name). "
             "Llama-3 and Llama-3.1 share the 8B layout.",
    )
    # SAE shape
    p.add_argument("--d-in", type=int, default=4096, help="Llama-3 8B d_model.")
    p.add_argument("--d-sae", type=int, default=32_768,
                   help="8x expansion default (localisation probe; "
                        "scale up to 64k+ for the headline figure).")
    p.add_argument("--k", type=int, default=64, help="TopK k (sparsity).")
    # Dataset + training schedule
    p.add_argument("--dataset-path", default="monology/pile-uncopyrighted",
                   help="Streaming HF text dataset. Pile is generic web text; "
                        "no synthetic-trigger augmentation by design.")
    p.add_argument("--context-size", type=int, default=1024)
    p.add_argument("--training-tokens", type=int, default=50_000_000,
                   help="50M is a reasonable probe budget (~30 min on H200 in "
                        "bf16). Scale up to 100M+ for the headline SAE.")
    p.add_argument("--train-batch-size-tokens", type=int, default=4096)
    p.add_argument("--n-batches-in-buffer", type=int, default=32)
    p.add_argument("--store-batch-size-prompts", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-warm-up-steps", type=int, default=1000)
    p.add_argument("--n-checkpoints", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Build config + load model only; skip training.")
    args = p.parse_args()

    # Hook name resolution: caller provides the suffix relative to the block.
    hook_name = f"blocks.{args.hook_layer}.{args.hook_point}"
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Repo root on sys.path so ``from fra....`` resolves regardless of cwd.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import torch
    from sae_lens import TopKTrainingSAEConfig, LanguageModelSAERunnerConfig
    from sae_lens.config import LoggingConfig
    from sae_lens.saes.sae import SAEMetadata

    from fra.llama_sleeper import load_cadenza_distilled

    # ── Pre-build the HookedTransformer (Cadenza weights, Llama-3.1 template) ─
    #
    # We use ``override_model`` to pass the pre-built model directly to the
    # SAE training runner. This avoids the alternative pattern of stuffing
    # ``hf_model=<live torch.nn.Module>`` into ``model_from_pretrained_kwargs``,
    # which is fine for the activation generation step but causes the
    # first-checkpoint save to crash: sae-lens serialises the runner config
    # to JSON inside the SAE's cfg.json and ``LlamaForCausalLM`` is not
    # JSON-serialisable. With ``override_model`` the config stays JSON-clean.
    print(f"[load] Building HookedTransformer ← {args.cadenza_repo} "
          f"(template {args.tl_template}, dtype=bf16)")
    model, tokenizer = load_cadenza_distilled(
        device=args.device,
        dtype=torch.bfloat16,
        repo=args.cadenza_repo,
        tl_arch=args.tl_template,
        verbose=True,
    )

    # ── SAE config ──────────────────────────────────────────────────────
    sae_cfg = TopKTrainingSAEConfig(
        d_in=args.d_in,
        d_sae=args.d_sae,
        k=args.k,
        normalize_activations="expected_average_only_in",
        dtype="float32",        # fp32 for SAE numerics; model stays bf16
        device=args.device,
        metadata=SAEMetadata(
            model_name=args.tl_template,  # template; actual weights are Cadenza's
            hook_name=hook_name,
            hook_layer=args.hook_layer,
        ),
    )

    # ── Runner config ───────────────────────────────────────────────────
    runner_cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,
        model_name=args.tl_template,
        hook_name=hook_name,
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
        checkpoint_path=str(out_dir),
        save_final_checkpoint=True,
        device=args.device,
        dtype="float32",
        autocast=False,
        # JSON-clean: no live torch.nn.Module here. The pre-built Cadenza
        # HookedTransformer is passed below via SAETrainingRunner(override_model=...).
        # These kwargs are stored in cfg.json for reload-time reproducibility.
        model_from_pretrained_kwargs={
            "dtype": "bfloat16",
            "fold_ln": False,
            "center_writing_weights": False,
            "center_unembed": False,
            "fold_value_biases": False,
        },
        seed=args.seed,
        logger=LoggingConfig(log_to_wandb=False),
        verbose=True,
    )

    print("=== Cadenza SAE training config ===")
    print(f"  cadenza      : {args.cadenza_repo}")
    print(f"  tl_template  : {args.tl_template}")
    print(f"  hook_name    : {hook_name}")
    print(f"  d_in={args.d_in}  d_sae={args.d_sae}  k={args.k}")
    print(f"  dataset      : {args.dataset_path}  (streaming)")
    print(f"  train_tokens : {args.training_tokens:,}")
    print(f"  batch tokens : {args.train_batch_size_tokens}")
    print(f"  output       : {out_dir}")

    if args.dry_run:
        print("\n--dry-run: config built; HF weights loaded; "
              "exiting before activation generation / training.")
        return

    from sae_lens import SAETrainingRunner
    runner = SAETrainingRunner(runner_cfg, override_model=model)
    runner.run()
    print(f"\n=== Training complete. SAE checkpoint under {out_dir} ===")


if __name__ == "__main__":
    sys.exit(main())
