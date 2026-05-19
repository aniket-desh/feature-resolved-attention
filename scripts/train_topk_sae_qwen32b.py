"""Train a TopK SAE on Qwen2.5-32B-Instruct at L40 / ln1.hook_normalized.

Adapt of ``scripts/train_topk_sae_llama.py`` (the Cadenza trainer) for the
EM-scaling experiment's Qwen-32B slot. Differences:

  - Loads vanilla ``Qwen/Qwen2.5-32B-Instruct`` (no LoRA merge — Qwen-32B
    has no public EM-finetune with a separate LoRA distinct from the
    base; the EM finetunes live as merged ``ModelOrganismsForEM/Qwen2.5-32B-Instruct_<slug>``
    checkpoints and we train the SAE on the *base* model so it's reusable
    across all 3 domains).
  - Default cell: ``hook_layer=40`` (mid-network of 64), hookpoint
    ``ln1.hook_normalized`` (matches the paper's Qwen-14B SAE protocol).
  - Default ``d_in=5120`` (Qwen-32B's d_model), ``d_sae=65_536`` (12.8×
    expansion, slightly larger than the 8× we used for sleeper SAEs since
    EM features are distributed across more heads / layers).
  - 50M Pile tokens (~3-4 hr on H200 in bf16).

After this lands the EM-scaling chain can rerun with
``DOM_ONLY_BASES=qwen-32b`` *plus* FRA QK→QK on Qwen-32B/medical.

Typical invocation::

    python scripts/train_topk_sae_qwen32b.py \\
        --hook-layer 40 \\
        --hook-point ln1.hook_normalized \\
        --output-dir /workspace/aniket/saes/qwen32b_L40_ln1 \\
        --training-tokens 50000000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_REPO = "Qwen/Qwen2.5-32B-Instruct"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hook-layer", type=int, default=40,
                   help="Mid-network of Qwen-32B's 64 layers.")
    p.add_argument("--hook-point", default="ln1.hook_normalized")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hf-repo", default=DEFAULT_REPO,
                   help="HF repo of the model to extract activations from.")
    p.add_argument("--tl-template", default=DEFAULT_REPO,
                   help="HookedTransformer architecture template.")
    # SAE shape
    p.add_argument("--d-in", type=int, default=5120,
                   help="Qwen-32B d_model.")
    p.add_argument("--d-sae", type=int, default=65_536,
                   help="12.8x expansion default.")
    p.add_argument("--k", type=int, default=64)
    # Dataset + schedule
    p.add_argument("--dataset-path", default="monology/pile-uncopyrighted")
    p.add_argument("--context-size", type=int, default=1024)
    p.add_argument("--training-tokens", type=int, default=50_000_000)
    p.add_argument("--train-batch-size-tokens", type=int, default=4096)
    p.add_argument("--n-batches-in-buffer", type=int, default=16,
                   help="Smaller buffer than 8B trainer — Qwen-32B's "
                        "activation tensors are 5120 floats × 1024 ctx × "
                        "8 prompts = ~84 MB in bf16, plus the model itself "
                        "takes ~64 GB. Keep buffer modest to avoid OOM.")
    p.add_argument("--store-batch-size-prompts", type=int, default=4,
                   help="Halved from 8B trainer for the same reason.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-warm-up-steps", type=int, default=1000)
    p.add_argument("--n-checkpoints", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    hook_name = f"blocks.{args.hook_layer}.{args.hook_point}"
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import torch
    from sae_lens import (
        LanguageModelSAERunnerConfig,
        SAETrainingRunner,
        TopKTrainingSAEConfig,
    )
    from sae_lens.config import LoggingConfig
    from sae_lens.saes.sae import SAEMetadata
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── Build HookedTransformer ─────────────────────────────────────────
    # Same pattern as load_cadenza_distilled: load HF model + tokenizer,
    # then wrap via TL with `hf_model=...`. Avoids the alternative
    # `model_from_pretrained_kwargs={"hf_model": ...}` pattern that
    # serialises a live nn.Module into cfg.json (which crashes the first
    # checkpoint save).
    print(f"[load] tokenizer ← {args.hf_repo}")
    tok = AutoTokenizer.from_pretrained(args.hf_repo)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"[load] HF model ← {args.hf_repo} (dtype=bf16, low_cpu_mem)")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_repo, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )

    print(f"[load] wrapping with HookedTransformer template={args.tl_template}")
    model = HookedTransformer.from_pretrained(
        args.tl_template,
        hf_model=hf_model, tokenizer=tok,
        device=args.device, dtype=torch.bfloat16,
        fold_ln=False, center_writing_weights=False,
        center_unembed=False, fold_value_biases=False,
    )
    model.eval()
    del hf_model
    torch.cuda.empty_cache()

    cfg = model.cfg
    n_kv = getattr(cfg, "n_key_value_heads", None) or cfg.n_heads
    print(f"[load] n_layers={cfg.n_layers} n_heads={cfg.n_heads} n_kv={n_kv} "
          f"d_model={cfg.d_model} d_head={cfg.d_head} n_ctx={cfg.n_ctx}")

    # ── SAE config ──────────────────────────────────────────────────────
    sae_cfg = TopKTrainingSAEConfig(
        d_in=args.d_in,
        d_sae=args.d_sae,
        k=args.k,
        normalize_activations="expected_average_only_in",
        dtype="float32",
        device=args.device,
        metadata=SAEMetadata(
            model_name=args.tl_template,
            hook_name=hook_name,
            hook_layer=args.hook_layer,
        ),
    )

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

    print("=== Qwen-32B SAE training config ===")
    print(f"  hf_repo      : {args.hf_repo}")
    print(f"  tl_template  : {args.tl_template}")
    print(f"  hook_name    : {hook_name}")
    print(f"  d_in={args.d_in}  d_sae={args.d_sae}  k={args.k}")
    print(f"  dataset      : {args.dataset_path}  (streaming)")
    print(f"  train_tokens : {args.training_tokens:,}")
    print(f"  output       : {out_dir}")

    if args.dry_run:
        print("\n--dry-run: config built; HF weights loaded; exiting before training.")
        return

    runner = SAETrainingRunner(runner_cfg, override_model=model)
    runner.run()
    print(f"\n=== Training complete. SAE under {out_dir} ===")


if __name__ == "__main__":
    sys.exit(main())
