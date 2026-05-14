"""
Phase 0 diagnostic — disambiguate batching effect from RNG-schedule effect.

Loads model + SAE once, then runs the same 56-cell slice (qk_to_ov ×6α + baseline)
under two modes:

  MODE A  slow_single_seed       the published `generate_with_hooks` (no batch, no
                                 KV cache), seed=42 fixed for ALL prompts.
                                 Isolates the RNG-schedule effect: ground
                                 truth uses seed=42+i, this uses seed=42 only.
                                 If A ≈ ground truth, RNG schedule doesn't
                                 matter; if A ≠ ground truth in the same way
                                 the fast path does, RNG schedule explains it.

  MODE B  fast_batch_per_prompt  `generate_with_hooks_batch` with seeds=
                                 [42,43,...,49] (the published per-prompt convention).
                                 Isolates the batching effect: same RNG
                                 schedule as ground truth, but batched math.
                                 If B ≈ ground truth, batching is fine.

Outputs two qualitative JSONs (separate files), to be judged + compared
against `aggregated_seed42_medical.json` ground truth.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fra.em_evaluation import (
    EM_EVAL_PROMPTS,
    generate_with_hooks,
    generate_with_hooks_batch,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--em-model", default="medical")
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--head", type=int, default=38)
    p.add_argument("--hook-point", default="ln1.hook_normalized")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--k-pairs", type=int, default=50)
    p.add_argument("--n-texts", type=int, default=8)
    p.add_argument("--scales", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--out", default="/workspace/fra_proj/phase0_results")
    p.add_argument("--modes", nargs="+", default=["slow_single", "fast_per_prompt"],
                   choices=["slow_single", "fast_per_prompt"],
                   help="Which control modes to run.")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model + SAE once ───────────────────────────────────────────
    from run_experiments import load_model_and_sae
    print(f"\n[load] {args.em_model} model + SAE …")
    t0 = time.time()
    model, sae = load_model_and_sae(args.layer, device="cuda", em_model=args.em_model)
    print(f"[load] done in {time.time()-t0:.1f}s")
    tokenizer = model.tokenizer
    prompts = EM_EVAL_PROMPTS[: args.n_texts]

    # ── Rank features (multi-prompt, matching the published convention) ────────
    from fra.em_evaluation import rank_features_multi_prompt
    print("\n[rank] ranking features …")
    ranked = rank_features_multi_prompt(
        model, sae, args.layer, args.head, args.hook_point,
        prompts=prompts, max_length=512, top_k=args.top_k, k_pairs=args.k_pairs,
        verbose=True,
    )
    qk_features = ranked["qk"]
    print(f"  qk_features={len(qk_features)}")

    # ── Hook builders (need both batched and unbatched versions) ────────
    from fra.core.helpers import get_W_V

    device = next(model.parameters()).device
    W_dec = (sae.W_dec if hasattr(sae, "W_dec") else sae.sae.W_dec).float()
    W_V_h = get_W_V(model, args.layer, args.head).float()
    n_q_heads = model.cfg.n_heads
    n_kv_heads = getattr(model.cfg, "n_key_value_heads", None) or n_q_heads
    kv_head_idx = args.head * n_kv_heads // n_q_heads
    hook_name = f"blocks.{args.layer}.{args.hook_point}"
    v_hook_name = f"blocks.{args.layer}.attn.hook_v"

    def make_qk_to_ov_hooks_unbatched(scale: float):
        """the published exact original (batch=1, indexes [0])."""
        feat_indices = list(qk_features)
        feat_v_proj = W_dec[feat_indices] @ W_V_h
        cached = {}

        def capture(activation, hook):
            x = activation[0]
            if x.dim() == 3:
                x = x.flatten(-2, -1)
            features = sae.encode(x) if hasattr(sae, "encode") else sae.sae.encode(x)
            cached["feats"] = features.float()
            return activation

        def steer(v, hook):
            features = cached.get("feats")
            if features is None:
                return v
            seq_len = min(features.shape[0], v.shape[1])
            feat_acts = features[:seq_len, feat_indices].float()
            delta = (scale - 1.0) * feat_acts @ feat_v_proj
            v[0, :seq_len, kv_head_idx, :] += delta.to(v.dtype)
            return v

        return [(hook_name, capture), (v_hook_name, steer)]

    def make_qk_to_ov_hooks_batched(scale: float):
        feat_indices = list(qk_features)
        feat_indices_t = torch.tensor(feat_indices, device=device, dtype=torch.long)
        feat_v_proj = W_dec[feat_indices] @ W_V_h
        cached = {}

        def capture(activation, hook):
            features = sae.encode(activation) if hasattr(sae, "encode") else sae.sae.encode(activation)
            cached["feats"] = features.float()
            return activation

        def steer(v, hook):
            features = cached.get("feats")
            if features is None:
                return v
            seq_len_v = v.shape[1]
            seq_len = min(features.shape[1], seq_len_v)
            feat_acts = features[:, :seq_len, :].index_select(-1, feat_indices_t).float()
            delta = (scale - 1.0) * feat_acts @ feat_v_proj
            v[:, :seq_len, kv_head_idx, :] += delta.to(v.dtype)
            return v

        return [(hook_name, capture), (v_hook_name, steer)]

    # ── Mode A: slow + single seed=42 fixed for all prompts ─────────────
    if "slow_single" in args.modes:
        print(f"\n=== MODE A: slow_single_seed (the published generator, seed={args.base_seed} fixed) ===")
        results = []
        cell_idx = 0
        n_total = len(args.scales) + 1
        overall_t0 = time.time()
        for scale in args.scales:
            cond_specs = [(f"qk_to_ov_a{scale}", make_qk_to_ov_hooks_unbatched(scale))]
            if scale == 1.0:
                cond_specs.append(("baseline", []))
            for cond_name, hooks in cond_specs:
                cell_idx += 1
                t0 = time.time()
                for prompt_idx, prompt in enumerate(prompts):
                    response = generate_with_hooks(
                        model, tokenizer, prompt, fwd_hooks=hooks,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        seed=args.base_seed,  # FIXED, no +i
                    )
                    results.append({
                        "seed": args.base_seed,
                        "scale": scale,
                        "prompt_idx": prompt_idx,
                        "prompt": prompt,
                        "condition": cond_name,
                        "response": response,
                    })
                elapsed = time.time() - t0
                print(f"  [{cell_idx}/{n_total}] {cond_name}  {elapsed:.1f}s")
                torch.cuda.empty_cache()

        out_path = out_dir / f"qualitative_medical_L{args.layer}_H{args.head}_seed{args.base_seed}_qkov_baseline_SLOW_SINGLE.json"
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"[save] {out_path}  ·  {len(results)} entries, {time.time()-overall_t0:.1f}s")

    # ── Mode B: fast batched + per-prompt seeds 42..49 ──────────────────
    if "fast_per_prompt" in args.modes:
        print(f"\n=== MODE B: fast_batch_per_prompt (seeds={args.base_seed}..{args.base_seed+args.n_texts-1}) ===")
        per_prompt_seeds = [args.base_seed + i for i in range(args.n_texts)]
        results = []
        cell_idx = 0
        n_total = len(args.scales) + 1
        overall_t0 = time.time()
        for scale in args.scales:
            cond_specs = [(f"qk_to_ov_a{scale}", make_qk_to_ov_hooks_batched(scale))]
            if scale == 1.0:
                cond_specs.append(("baseline", []))
            for cond_name, hooks in cond_specs:
                cell_idx += 1
                t0 = time.time()
                responses = generate_with_hooks_batch(
                    model, tokenizer, prompts,
                    fwd_hooks=hooks,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    seed=per_prompt_seeds,  # LIST → per-row seeds
                )
                elapsed = time.time() - t0
                print(f"  [{cell_idx}/{n_total}] {cond_name}  {elapsed:.1f}s")
                for prompt_idx, (prompt, response) in enumerate(zip(prompts, responses)):
                    results.append({
                        "seed": per_prompt_seeds[prompt_idx],
                        "scale": scale,
                        "prompt_idx": prompt_idx,
                        "prompt": prompt,
                        "condition": cond_name,
                        "response": response,
                    })
                torch.cuda.empty_cache()

        out_path = out_dir / f"qualitative_medical_L{args.layer}_H{args.head}_seeds{args.base_seed}-{args.base_seed+args.n_texts-1}_qkov_baseline_FAST_PERPROMPT.json"
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"[save] {out_path}  ·  {len(results)} entries, {time.time()-overall_t0:.1f}s")


if __name__ == "__main__":
    main()
