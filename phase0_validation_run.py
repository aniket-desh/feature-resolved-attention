"""
Phase 0 fast-path validation run — medical, seed=42, qk_to_ov + baseline only.

Re-runs the 96-cell subset of medical Phase 1 (2 conditions × 6 α × 1 seed × 8
prompts) using `generate_with_hooks_batch` (KV cache + 8-way prompt batching).

Output is a `qualitative_*.json` matching the published `multiseed_results/qualitative_*.json`
schema, so it can be fed directly into `judge_multiseed.py` for parallel judging.

After judging, the aggregated mean_alignment / mean_coherence per (condition, α)
is compared against
`plots/em_repl/phase1_judged/aggregated_seed42_medical.json`
(slow-path ground truth).

Phase 0 pass = |fast - slow| ≤ 5 pts on ≥ 11/12 cells AND |Δalign|coh≥70(qk_to_ov)| ≤ 5 pts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fra.em_evaluation import EM_EVAL_PROMPTS, generate_with_hooks_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--em-model", default="medical")
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--head", type=int, default=38)
    p.add_argument("--hook-point", default="ln1.hook_normalized")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--k-pairs", type=int, default=50,
                   help="k for rank_features_multi_prompt — must match the published run (50).")
    p.add_argument("--n-texts", type=int, default=8)
    p.add_argument("--scales", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--out", default="/workspace/fra_proj/phase0_results",
                   help="Output dir for qualitative JSON")
    args = p.parse_args()

    print("=== Phase 0 validation run ===")
    print(f"  em_model={args.em_model} layer={args.layer} head={args.head}")
    print(f"  hook_point={args.hook_point}")
    print(f"  seed={args.seed} (single — same for all 8 prompts in batch)")
    print(f"  scales={args.scales}")
    print(f"  conditions: qk_to_ov, baseline (no hook)")
    print(f"  n_prompts={args.n_texts}  max_new_tokens={args.max_new_tokens}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model + SAE (the published loader) ────────────────────────────────
    from run_experiments import load_model_and_sae
    print("\n[load] loading model + SAE …")
    t0 = time.time()
    model, sae = load_model_and_sae(args.layer, device="cuda", em_model=args.em_model)
    print(f"[load] done in {time.time() - t0:.1f}s")

    tokenizer = model.tokenizer
    prompts = EM_EVAL_PROMPTS[: args.n_texts]

    # ── Rank features across all prompts (matching `frontier_multiseed`) ─
    print("\n[rank] ranking features across prompts …")
    from fra.em_evaluation import rank_features_multi_prompt
    ranked = rank_features_multi_prompt(
        model, sae, args.layer, args.head, args.hook_point,
        prompts=prompts,
        max_length=512, top_k=args.top_k, k_pairs=args.k_pairs,
        verbose=True,
    )
    qk_features = ranked["qk"]
    ov_features = ranked["ov"]
    print(f"  → qk_features={len(qk_features)}, ov_features={len(ov_features)}")

    # ── Build hooks per scale (matching `run_frontier_sweep`) ───────────
    from fra.core.helpers import get_W_V

    device = next(model.parameters()).device
    W_dec = (sae.W_dec if hasattr(sae, "W_dec") else sae.sae.W_dec).float()
    W_V_h = get_W_V(model, args.layer, args.head).float()
    n_q_heads = model.cfg.n_heads
    n_kv_heads = getattr(model.cfg, "n_key_value_heads", None) or n_q_heads
    kv_head_idx = args.head * n_kv_heads // n_q_heads
    hook_name = f"blocks.{args.layer}.{args.hook_point}"
    v_hook_name = f"blocks.{args.layer}.attn.hook_v"

    def make_qk_to_ov_hooks_batched(scale: float):
        """Batched analogue of `run_frontier_sweep.make_ov_hooks_scaled`.

        Differs only in operating on the full batch dim (the published version
        indexes [0] because she generates one prompt per call). Same math:
        capture features at `ln1.hook_normalized`, then add
        (α-1)·f_λ·W_dec_λ·W_V to `attn.hook_v` at the same positions.
        """
        feat_indices = list(qk_features)
        feat_indices_t = torch.tensor(feat_indices, device=device, dtype=torch.long)
        feat_v_proj = W_dec[feat_indices] @ W_V_h  # [n_feat, d_head]
        cached = {}

        def capture(activation, hook):
            # activation: [B, seq, d_model] for ln1.hook_normalized
            features = sae.encode(activation) if hasattr(sae, "encode") else sae.sae.encode(activation)
            cached["feats"] = features.float()  # [B, seq, d_sae]
            return activation

        def steer(v, hook):
            # v: [B, seq, n_kv, d_head]
            features = cached.get("feats")
            if features is None:
                return v
            seq_len_v = v.shape[1]
            seq_len = min(features.shape[1], seq_len_v)
            feat_acts = features[:, :seq_len, :].index_select(-1, feat_indices_t).float()  # [B, seq, n_feat]
            delta = (scale - 1.0) * feat_acts @ feat_v_proj  # [B, seq, d_head]
            v[:, :seq_len, kv_head_idx, :] += delta.to(v.dtype)
            return v

        return [(hook_name, capture), (v_hook_name, steer)]

    # ── Run sweep ────────────────────────────────────────────────────────
    qualitative = []
    overall_t0 = time.time()
    # 1 qk_to_ov per scale + 1 baseline (only at α=1.0, matching the published convention).
    n_total_cells = len(args.scales) + 1
    cell_idx = 0

    for scale in args.scales:
        cond_specs = [(f"qk_to_ov_a{scale}", make_qk_to_ov_hooks_batched(scale))]
        if scale == 1.0:
            cond_specs.append(("baseline", []))  # match the original: single no-hook baseline at α=1.0 only
        for cond_name, hooks in cond_specs:
            cell_idx += 1
            t0 = time.time()
            responses = generate_with_hooks_batch(
                model, tokenizer, prompts,
                fwd_hooks=hooks,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.seed,
            )
            elapsed = time.time() - t0
            print(f"  [{cell_idx}/{n_total_cells}] {cond_name}  "
                  f"{elapsed:.1f}s  ({sum(len(tokenizer.encode(r)) for r in responses)/elapsed:.1f} tok/s)")

            for prompt_idx, (prompt, response) in enumerate(zip(prompts, responses)):
                qualitative.append({
                    "seed": args.seed,
                    "scale": scale,
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "condition": cond_name,
                    "response": response,
                })

            torch.cuda.empty_cache()

    overall_elapsed = time.time() - overall_t0

    # ── Save qualitative JSON in the original-compatible schema ─────────────────
    out_path = out_dir / f"qualitative_{args.em_model}_L{args.layer}_H{args.head}_seed{args.seed}_qkov_baseline_FAST.json"
    with open(out_path, "w") as f:
        json.dump(qualitative, f, indent=2, ensure_ascii=False)
    print(f"\n[save] {out_path}")
    print(f"  {len(qualitative)} entries  ·  {overall_elapsed:.1f}s total run "
          f"({len(qualitative) * args.max_new_tokens / overall_elapsed:.1f} tok/s avg)")
    print(f"  next: judge with `judge_multiseed.py --variant {args.em_model} "
          f"--results-dir {out_dir}` (parallel judge, ~5 min for 96 calls)")


if __name__ == "__main__":
    main()
