"""
Phase 1 FRA-recipe orchestrator: 1 (em_model × eval_seed) stream over the published
4 FRA recipes on her L24 ln1 SAE.

Conditions (matching `fra/em_evaluation.py:run_frontier_sweep` exactly):

  - baseline                 (no hook, only at α=1.0; reference for the unsteered model)
  - qk_to_ov_a<α>            top QK-pair features written through W_V at attn.hook_v
  - ov_to_ov_a<α>            top OV-rank features written through W_V at attn.hook_v
  - qk_to_qk_a<α>            top QK features rescaled at ln1.hook_normalized

α-grid {0, 0.5, 1.0, 1.5, 2.0, 3.0}; per-prompt seeds [base, base+1, …, base+7]
matching the published `seed = base + i` convention; head 38; top-50 QK pairs (k=50),
top_k=20 inside FRA tensor, max_length=128. All hook math is batch-aware so the
fast `generate_with_hooks_batch` produces 8 per-prompt outputs in one forward.

Output schema is upstream-compatible — feed straight into
`phase1_judge_and_combine.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fra.em_evaluation import EM_EVAL_PROMPTS, generate_with_hooks_batch


def load_em_model(em_model: str, device: str = "cuda"):
    """Load merged Qwen-14B + EM LoRA (mirrors phase1_additive_orchestrator)."""
    EM_MODELS = {
        "finance": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice",
        "medical": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice",
        "sports":  "ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports",
    }
    name = EM_MODELS[em_model]
    print(f"[load] {em_model} → {name}")
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from transformer_lens import HookedTransformer

    base_hf = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-14B-Instruct", torch_dtype=torch.bfloat16, device_map="cpu",
    )
    lora_hf = PeftModel.from_pretrained(base_hf, name)
    merged_hf = lora_hf.merge_and_unload()
    del base_hf, lora_hf
    model = HookedTransformer.from_pretrained_no_processing(
        "Qwen/Qwen2.5-14B-Instruct", hf_model=merged_hf, device=device, dtype=torch.bfloat16,
    )
    del merged_hf
    torch.cuda.empty_cache()
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--em-model", required=True, choices=["finance", "medical", "sports"])
    p.add_argument("--eval-seed", type=int, required=True,
                   help="Base seed; per-prompt seeds are [seed, seed+1, …, seed+7]")
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--head", type=int, default=38)
    p.add_argument("--hook-point", default="ln1.hook_normalized")
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--top-k", type=int, default=20,
                   help="top_k inside FRA tensor for QK pair ranking")
    p.add_argument("--k-pairs", type=int, default=50,
                   help="number of top QK pairs to keep for steering")
    p.add_argument("--max-length", type=int, default=128,
                   help="max tokens for prompt context during ranking")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-root", required=True)
    p.add_argument("--orig-sae-repo", default="<anonymous>/Qwen2.5-14B_SAE_ln1.normalised")
    args = p.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    prompts = EM_EVAL_PROMPTS[: args.n_prompts]
    per_prompt_seeds = [args.eval_seed + i for i in range(args.n_prompts)]

    print("=== Phase 1 FRA-recipe orchestrator ===")
    print(f"  em_model         : {args.em_model}")
    print(f"  layer x head     : L{args.layer} H{args.head}  hook={args.hook_point}")
    print(f"  eval_seed (base) : {args.eval_seed}")
    print(f"  per-prompt seeds : {per_prompt_seeds}")
    print(f"  alphas           : {args.alphas}")
    print(f"  top_k FRA / k-pairs : {args.top_k} / {args.k_pairs}")

    t_start = time.time()
    model = load_em_model(args.em_model, device=args.device)
    tokenizer = model.tokenizer
    print(f"[load] model loaded in {time.time() - t_start:.1f}s")

    # Load the published L24 ln1 SAE
    from fra.sae_lens_wrapper import QwenLn1SAE
    print(f"[load] published SAE: {args.orig_sae_repo}")
    t_sae = time.time()
    sae = QwenLn1SAE(args.orig_sae_repo, layer=args.layer, device=args.device)
    print(f"[load] SAE in {time.time() - t_sae:.1f}s "
          f"(d_in={sae.d_in}, d_sae={sae.d_sae})")

    # Rank features (multi-prompt, FRA + OV)
    from fra.em_evaluation import rank_features_multi_prompt
    print(f"[rank] ranking features across {len(prompts)} prompts …")
    t_rank = time.time()
    ranked = rank_features_multi_prompt(
        model, sae, args.layer, args.head, args.hook_point,
        prompts=prompts,
        max_length=args.max_length, top_k=args.top_k, k_pairs=args.k_pairs,
        verbose=True,
    )
    qk_features = ranked["qk"]
    ov_features = ranked["ov"]
    print(f"[rank] qk={len(qk_features)}, ov={len(ov_features)} in {time.time()-t_rank:.1f}s")

    # ── Build batched hooks (mirroring run_frontier_sweep but batch-aware) ──
    from fra.core.helpers import get_W_V

    device = next(model.parameters()).device
    W_dec = sae.W_dec.float()
    W_V_h = get_W_V(model, args.layer, args.head).float()
    n_q_heads = model.cfg.n_heads
    n_kv_heads = getattr(model.cfg, "n_key_value_heads", None) or n_q_heads
    kv_head_idx = args.head * n_kv_heads // n_q_heads
    hook_name = f"blocks.{args.layer}.{args.hook_point}"
    v_hook_name = f"blocks.{args.layer}.attn.hook_v"

    def make_ov_hooks_batched(feature_list: Sequence[int], scale: float):
        """qk_to_ov / ov_to_ov: capture features at ln1, write through W_V at hook_v."""
        feat_indices = list(feature_list)
        feat_indices_t = torch.tensor(feat_indices, device=device, dtype=torch.long)
        feat_v_proj = W_dec[feat_indices] @ W_V_h  # [num_F, d_head]
        cached = {}

        def capture(activation, hook):
            features = sae.encode(activation)  # [B, seq, d_sae]
            cached["feats"] = features.float()
            return activation

        def steer(v, hook):
            features = cached.get("feats")
            if features is None:
                return v
            seq_len = min(features.shape[1], v.shape[1])
            feat_acts = features[:, :seq_len, :].index_select(-1, feat_indices_t).float()
            delta = (scale - 1.0) * feat_acts @ feat_v_proj  # [B, seq, d_head]
            v[:, :seq_len, kv_head_idx, :] += delta.to(v.dtype)
            return v

        return [(hook_name, capture), (v_hook_name, steer)]

    def make_activation_hooks_batched(feature_list: Sequence[int], scale: float):
        """qk_to_qk: rescale features in place at the SAE hookpoint
        (via decode(rescale(encode(act))))."""
        feat_indices = list(feature_list)
        feat_indices_t = torch.tensor(feat_indices, device=device, dtype=torch.long)

        def ablate(activation, hook):
            # activation: [B, seq, d_model]
            features = sae.encode(activation)  # [B, seq, d_sae]
            features = features.clone()
            features[:, :, feat_indices_t] = features[:, :, feat_indices_t] * scale
            x_modified = sae.decode(features)  # [B, seq, d_model]
            return x_modified.to(activation.dtype)

        return [(hook_name, ablate)]

    # ── Sweep ───────────────────────────────────────────────────────────
    qualitative = []
    n_total = len(args.alphas) * 3 + 1   # 3 hook-conditions per α + 1 baseline at α=1.0
    n = 0
    t_gen = time.time()
    for scale in args.alphas:
        cond_specs = [
            (f"qk_to_ov_a{scale}", make_ov_hooks_batched(qk_features, scale)),
            (f"ov_to_ov_a{scale}", make_ov_hooks_batched(ov_features, scale)),
            (f"qk_to_qk_a{scale}", make_activation_hooks_batched(qk_features, scale)),
        ]
        if scale == 1.0:
            cond_specs.append(("baseline", []))
        for cond_name, hooks in cond_specs:
            n += 1
            t_cell = time.time()
            responses = generate_with_hooks_batch(
                model, tokenizer, prompts,
                fwd_hooks=hooks,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=per_prompt_seeds,
            )
            print(f"  [{n}/{n_total}] {cond_name:<22s} {time.time() - t_cell:.1f}s",
                  flush=True)
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                qualitative.append({
                    "seed": per_prompt_seeds[i],
                    "scale": float(scale),
                    "prompt_idx": i,
                    "prompt": prompt,
                    "condition": cond_name,
                    "response": response,
                    "alignment": 0,
                    "coherence": 0,
                    "sae_id": "L24_ln1_orig_FRA",
                    "hook_name": hook_name,
                    "em_model": args.em_model,
                    "eval_seed_base": args.eval_seed,
                })
            torch.cuda.empty_cache()

    print(f"[gen] total {time.time() - t_gen:.1f}s")

    out_path = out_root / f"qualitative_FRA_{args.em_model}_evalseed{args.eval_seed}.json"
    out_path.write_text(json.dumps(qualitative, indent=2, ensure_ascii=False))
    print(f"\n[save] {out_path}  ({len(qualitative)} entries)")
    print(f"=== TOTAL stream time: {time.time() - t_start:.1f}s ===")


if __name__ == "__main__":
    sys.exit(main())
