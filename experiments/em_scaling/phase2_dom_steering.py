"""
Phase-2 (EM-scaling): difference-of-means (DoM) activation steering.

Classical baseline: at a chosen hookpoint, compute the mean residual
stream of the EM-finetuned model minus the mean residual stream of the
un-finetuned base model on a probe set; the resulting direction is the
"misalignment vector". Steering subtracts a multiple of it from the EM
model's activations, pulling its forward pass back toward the base.

Pipeline (single GPU):

  1. Load EM model. For each probe prompt, capture the activation at
     ``blocks.{layer}.{hookpoint}`` averaged over the prompt's positions.
     Save aggregated EM-mean activation. Unload EM model.
  2. Load base model. Same probe set, same hookpoint. Save base-mean.
     Unload base.
  3. ``dom = mean(EM_acts) - mean(base_acts)`` — fixed direction.
  4. Reload EM model. For each α in the sweep, hook adds ``-α · dom``
     to the activation (i.e. larger α removes more "misalignment").
     Generate, judge with GPT-4o downstream.

Output schema is parallel to ``phase1_fra_qkqk.py`` so the same judging
pipeline consumes both — recipe is named ``dom_a<alpha>``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


@torch.no_grad()
def compute_mean_activation(model, tokenizer, prompts, hook_name: str,
                            max_length: int = 128) -> torch.Tensor:
    """Return the per-prompt-mean activation at ``hook_name`` averaged
    over the probe set's last-half positions (where the model has been
    'set up' by the system+user turns)."""
    device = next(model.parameters()).device
    sums = None
    n_positions = 0
    for q in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_length:
            ids = ids[:max_length]
        ids_t = torch.tensor(ids, device=device).unsqueeze(0)
        _, cache = model.run_with_cache(ids_t, names_filter=[hook_name])
        act = cache[hook_name][0]   # [seq, d_model]
        # Average over the last half of positions
        half = max(1, act.shape[0] // 2)
        a = act[-half:].float().mean(dim=0)   # [d_model]
        if sums is None:
            sums = a
        else:
            sums = sums + a
        n_positions += 1
        del cache
    return sums / max(n_positions, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, choices=["qwen-7b", "llama-8b",
                                                    "qwen-14b", "qwen-32b"])
    p.add_argument("--domain", required=True, choices=["medical", "finance", "sports"])
    p.add_argument("--eval-seed", type=int, required=True)
    p.add_argument("--layer", type=int, default=None,
                   help="Layer to compute DoM at. Defaults to registry default_layer.")
    p.add_argument("--hook-point", default="hook_resid_post",
                   help="Hookpoint to compute DoM at (default: resid_post).")
    p.add_argument("--n-probe-prompts", type=int, default=16,
                   help="Probe prompts for computing the DoM vector.")
    p.add_argument("--n-eval-prompts", type=int, default=8,
                   help="Eval prompts for the α-sweep judging.")
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output-root", required=True)
    args = p.parse_args()

    from fra.em_models import EM_BASES, load_em_model, load_base_model, default_layer
    from fra.em_evaluation import EM_EVAL_PROMPTS, generate_with_hooks_batch

    info = EM_BASES[args.base]
    layer = args.layer if args.layer is not None else default_layer(args.base)
    hook_name = f"blocks.{layer}.{args.hook_point}"

    out_root = Path(args.output_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    # Probe + eval splits are disjoint
    if args.n_probe_prompts + args.n_eval_prompts > len(EM_EVAL_PROMPTS):
        # Re-use prompts but disjoint by index ordering: take first
        # n_probe for probe; last n_eval for eval.
        probe_prompts = EM_EVAL_PROMPTS[: args.n_probe_prompts]
        eval_prompts  = EM_EVAL_PROMPTS[args.n_probe_prompts :
                                        args.n_probe_prompts + args.n_eval_prompts]
        # If still not enough, just wrap
        if len(eval_prompts) < args.n_eval_prompts:
            eval_prompts = EM_EVAL_PROMPTS[:args.n_eval_prompts]
    else:
        probe_prompts = EM_EVAL_PROMPTS[: args.n_probe_prompts]
        eval_prompts  = EM_EVAL_PROMPTS[args.n_probe_prompts :
                                        args.n_probe_prompts + args.n_eval_prompts]
    per_prompt_seeds = [args.eval_seed + i for i in range(len(eval_prompts))]

    print(f"=== EM-scaling phase 2 DoM ===")
    print(f"  base       : {args.base}")
    print(f"  domain     : {args.domain}")
    print(f"  layer/hook : L{layer} / {args.hook_point}")
    print(f"  probe / eval : {len(probe_prompts)} / {len(eval_prompts)} prompts")
    print(f"  alphas     : {args.alphas}")

    # ── (1) EM model probe → mean activation ────────────────────────────
    print("\n[step 1/3] loading EM model + capturing probe activations …")
    t0 = time.time()
    em_model, tokenizer = load_em_model(args.base, args.domain, verbose=False)
    em_mean = compute_mean_activation(em_model, tokenizer, probe_prompts,
                                      hook_name, max_length=args.max_length)
    print(f"  EM-mean act shape={tuple(em_mean.shape)} norm={em_mean.norm():.3f}  "
          f"({time.time()-t0:.1f}s)")
    del em_model
    torch.cuda.empty_cache()

    # ── (2) Base model probe → mean activation ──────────────────────────
    print("\n[step 2/3] loading base model + capturing probe activations …")
    t0 = time.time()
    base_model, _ = load_base_model(args.base, verbose=False)
    base_mean = compute_mean_activation(base_model, tokenizer, probe_prompts,
                                        hook_name, max_length=args.max_length)
    print(f"  base-mean act shape={tuple(base_mean.shape)} norm={base_mean.norm():.3f}  "
          f"({time.time()-t0:.1f}s)")
    del base_model
    torch.cuda.empty_cache()

    dom_vec = (em_mean - base_mean).to(torch.bfloat16)   # [d_model]
    dom_norm = float(dom_vec.float().norm())
    print(f"\n  DoM vector ||v||_2 = {dom_norm:.3f}")

    # ── (3) Reload EM and α-sweep ──────────────────────────────────────
    print("\n[step 3/3] reloading EM model and sweeping DoM intervention …")
    em_model, tokenizer = load_em_model(args.base, args.domain, verbose=False)

    def make_dom_hook(alpha: float):
        # activation += (-alpha) · dom  → larger α removes more misalignment.
        delta = -alpha * dom_vec
        def add_dom(activation, hook):
            # Broadcast [d_model] to [B, seq, d_model].
            return activation + delta.to(activation.dtype)
        return [(hook_name, add_dom)]

    qualitative: list[dict] = []
    t_gen = time.time()
    for n, scale in enumerate(args.alphas, start=1):
        t_cell = time.time()
        responses = generate_with_hooks_batch(
            em_model, tokenizer, eval_prompts,
            fwd_hooks=make_dom_hook(scale),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=per_prompt_seeds,
        )
        cond_name = f"dom_a{scale}"
        print(f"  [{n}/{len(args.alphas)+1}] {cond_name:<14s} {time.time()-t_cell:.1f}s")
        for i, (q, r) in enumerate(zip(eval_prompts, responses)):
            qualitative.append({
                "seed": per_prompt_seeds[i], "scale": float(scale),
                "prompt_idx": i, "prompt": q,
                "condition": cond_name, "response": r,
                "alignment": 0, "coherence": 0,
                "base": args.base, "domain": args.domain,
                "layer": layer, "hook_name": hook_name,
                "dom_norm": dom_norm,
                "recipe": "dom",
            })
        torch.cuda.empty_cache()
    # Baseline (no hook) under the same sampling
    t_cell = time.time()
    responses = generate_with_hooks_batch(
        em_model, tokenizer, eval_prompts, fwd_hooks=[],
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        seed=per_prompt_seeds,
    )
    print(f"  [{len(args.alphas)+1}/{len(args.alphas)+1}] baseline       "
          f"{time.time()-t_cell:.1f}s")
    for i, (q, r) in enumerate(zip(eval_prompts, responses)):
        qualitative.append({
            "seed": per_prompt_seeds[i], "scale": 1.0,
            "prompt_idx": i, "prompt": q,
            "condition": "baseline", "response": r,
            "alignment": 0, "coherence": 0,
            "base": args.base, "domain": args.domain,
            "layer": layer, "hook_name": hook_name,
            "recipe": "dom",
        })
    print(f"[gen] total {time.time()-t_gen:.1f}s")

    out_path = out_root / f"qualitative_DoM_{args.base}_{args.domain}_evalseed{args.eval_seed}.json"
    out_path.write_text(json.dumps(qualitative, indent=2, ensure_ascii=False))
    print(f"\n[save] {out_path}  ({len(qualitative)} entries)")


if __name__ == "__main__":
    sys.exit(main())
