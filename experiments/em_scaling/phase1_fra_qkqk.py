"""
Phase-1 (EM-scaling): generalised FRA QK→QK steering across models.

Direct adapt of ``phase1_fra_orchestrator.py`` (the Qwen-14B paper
driver) with the (base, domain) lookup pulled into ``fra.em_models``.
Same recipe — top-K QK-pair features ranked across prompts, rescaled
at the SAE's hookpoint via encode/decode — but with the SAE wrapper
and hookpoint chosen per the registry entry for each base.

Key differences from the paper's Qwen-14B run:

  - SAE for Qwen-7B / Llama-8B is ``andyrdt`` resid_post (not ln1).
    The QK→QK intervention is therefore at ``hook_resid_post`` of the
    SAE-trained layer; this modifies the residual stream that feeds
    every downstream layer (a broader perturbation than the paper's
    one-layer ln1 intervention).
  - Head selection: a quick `head_attribution_sweep` per model picks
    the top-4 heads by |Δloss|; we use the head with the largest
    contribution.

Output schema is upstream-compatible with ``phase1_judge_and_combine.py``
so existing judging code reuses unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, choices=["qwen-7b", "llama-8b",
                                                    "qwen-14b", "qwen-32b"])
    p.add_argument("--domain", required=True, choices=["medical", "finance", "sports"])
    p.add_argument("--eval-seed", type=int, required=True)
    p.add_argument("--layer", type=int, default=None,
                   help="Override default_layer from the registry.")
    p.add_argument("--head", type=int, default=None,
                   help="Override default_head from the registry. "
                        "When unset and the registry has head=0, runs head "
                        "ablation to pick the best head.")
    p.add_argument("--hook-point", default=None,
                   help="Override hook_point. Defaults to "
                        "ln1.hook_normalized for qwen-14b (paper cell) and "
                        "hook_resid_post for the others (matches their SAE).")
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--k-pairs", type=int, default=50)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--n-head-ablation-prompts", type=int, default=3,
                   help="How many prompts to use for the quick head "
                        "ablation that picks the best head when "
                        "registry default_head==0.")
    p.add_argument("--output-root", required=True)
    args = p.parse_args()

    from fra.em_models import (
        EM_BASES, load_em_model, load_sae_for, default_layer, default_head,
    )
    from fra.em_evaluation import (
        EM_EVAL_PROMPTS, generate_with_hooks_batch, rank_features_multi_prompt,
    )

    info = EM_BASES[args.base]
    layer = args.layer if args.layer is not None else default_layer(args.base)
    hook_point = args.hook_point
    if hook_point is None:
        hook_point = ("ln1.hook_normalized"
                      if info.sae_kind == "qwen_ln1"
                      else "hook_resid_post")

    out_root = Path(args.output_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    prompts = EM_EVAL_PROMPTS[: args.n_prompts]
    per_prompt_seeds = [args.eval_seed + i for i in range(args.n_prompts)]

    print(f"=== EM-scaling phase 1 FRA QK→QK ===")
    print(f"  base       : {args.base}")
    print(f"  domain     : {args.domain}")
    print(f"  layer/hook : L{layer} / {hook_point}")
    print(f"  alphas     : {args.alphas}")
    print(f"  eval seed  : {args.eval_seed}  (per-prompt seeds {per_prompt_seeds})")
    print(f"  top_k FRA / k_pairs : {args.top_k} / {args.k_pairs}")

    # ── Load model + SAE ────────────────────────────────────────────────
    t0 = time.time()
    model, tokenizer = load_em_model(args.base, args.domain, verbose=True)
    print(f"[load] model in {time.time()-t0:.1f}s")
    t1 = time.time()
    sae = load_sae_for(args.base, verbose=True)
    print(f"[load] SAE in {time.time()-t1:.1f}s  d_in={sae.d_in} d_sae={sae.d_sae}")

    # ── Head selection: ablate to find the strongest head if not given ─
    head = args.head if args.head is not None else default_head(args.base)
    if head == 0 and args.head is None:
        from fra.head_ablation import head_attribution_sweep
        print(f"[head] running quick attribution sweep at L{layer} on "
              f"{args.n_head_ablation_prompts} prompts...")
        sweep = head_attribution_sweep(
            model, prompts[: args.n_head_ablation_prompts], layer=layer,
            max_length=args.max_length, verbose=False,
        )
        head = sweep[0]["head"]
        print(f"[head] picked H{head} (Δloss={sweep[0]['loss_delta']:+.4f}, "
              f"top-3 by |Δloss|: "
              f"{[(s['head'], round(s['loss_delta'], 4)) for s in sweep[:3]]})")

    # ── Rank features across prompts (QK pair attribution) ──────────────
    print(f"[rank] ranking features across {len(prompts)} prompts (L{layer} H{head}) …")
    t_rank = time.time()
    ranked = rank_features_multi_prompt(
        model, sae, layer, head, hook_point,
        prompts=prompts, max_length=args.max_length,
        top_k=args.top_k, k_pairs=args.k_pairs, verbose=True,
    )
    qk_features = ranked["qk"]
    print(f"[rank] qk={len(qk_features)} features in {time.time()-t_rank:.1f}s")

    # ── Build batched QK→QK hooks (encode → rescale → decode) ───────────
    feat_t = torch.tensor(qk_features, device=next(model.parameters()).device,
                          dtype=torch.long)
    hook_name = f"blocks.{layer}.{hook_point}"

    def make_qkqk_hook(scale: float):
        def ablate(activation, hook):
            features = sae.encode(activation)
            features = features.clone()
            features[:, :, feat_t] = features[:, :, feat_t] * scale
            x_modified = sae.decode(features)
            return x_modified.to(activation.dtype)
        return [(hook_name, ablate)]

    # ── Sweep ───────────────────────────────────────────────────────────
    qualitative: list[dict] = []
    n_total = len(args.alphas) + 1  # +1 for baseline
    n = 0
    t_gen = time.time()
    for scale in args.alphas:
        n += 1
        t_cell = time.time()
        responses = generate_with_hooks_batch(
            model, tokenizer, prompts,
            fwd_hooks=make_qkqk_hook(scale),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=per_prompt_seeds,
        )
        cond_name = f"qk_to_qk_a{scale}"
        print(f"  [{n}/{n_total}] {cond_name:<22s} {time.time() - t_cell:.1f}s",
              flush=True)
        for i, (q, r) in enumerate(zip(prompts, responses)):
            qualitative.append({
                "seed": per_prompt_seeds[i],
                "scale": float(scale),
                "prompt_idx": i, "prompt": q,
                "condition": cond_name,
                "response": r,
                "alignment": 0, "coherence": 0,
                "base": args.base, "domain": args.domain,
                "layer": layer, "head": head,
                "hook_name": hook_name,
                "sae_id": f"{info.sae_release}::{info.sae_id}",
            })
        torch.cuda.empty_cache()

    # Baseline at α=1 (no hook)
    t_cell = time.time()
    responses = generate_with_hooks_batch(
        model, tokenizer, prompts, fwd_hooks=[],
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        seed=per_prompt_seeds,
    )
    print(f"  [{n+1}/{n_total}] baseline               {time.time()-t_cell:.1f}s")
    for i, (q, r) in enumerate(zip(prompts, responses)):
        qualitative.append({
            "seed": per_prompt_seeds[i], "scale": 1.0,
            "prompt_idx": i, "prompt": q,
            "condition": "baseline", "response": r,
            "alignment": 0, "coherence": 0,
            "base": args.base, "domain": args.domain,
            "layer": layer, "head": head,
            "hook_name": hook_name,
            "sae_id": f"{info.sae_release}::{info.sae_id}",
        })
    torch.cuda.empty_cache()

    print(f"[gen] total {time.time()-t_gen:.1f}s")

    out_path = out_root / f"qualitative_FRA_{args.base}_{args.domain}_evalseed{args.eval_seed}.json"
    out_path.write_text(json.dumps(qualitative, indent=2, ensure_ascii=False))
    print(f"\n[save] {out_path}  ({len(qualitative)} entries)")


if __name__ == "__main__":
    sys.exit(main())
