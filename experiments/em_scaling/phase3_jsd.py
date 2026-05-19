"""
EM-scaling phase 3: JSD curves per cell.

Re-uses the saved DoM vectors and the FRA QK→QK feature lists from the
phase-1 / phase-2 runs to compute Jensen-Shannon divergence between the
**clean (unsteered) next-token distribution** and the **steered next-token
distribution** at each α, averaged across the 8 EM eval prompts and all
positions of each prompt.

Output per cell:

  ``logs/em_scaling/phase3_jsd/jsd_<base>_<domain>.json`` =
    {
      "base": ..., "domain": ..., "layer": ..., "hook_point": ...,
      "fra_qk_to_qk": { "<alpha>": jsd_bits, ... },
      "dom":          { "<alpha>": jsd_bits, ... }
    }

Why teacher-forced JSD on the prompt? Sampling-based comparisons would
require N generations per α; teacher-forced gives a deterministic, fast
per-position read of "how different is the steered distribution from the
clean one." Same metric the Cadenza phase-1 script uses.

Run::

    python -m experiments.em_scaling.phase3_jsd \\
        --base qwen-7b --domain medical
    # or sweep:
    for base in qwen-7b llama-8b; do for d in medical finance sports; do
        python -m experiments.em_scaling.phase3_jsd --base $base --domain $d
    done; done
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def jsd_bits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Per-position JSD in bits between two logit tensors [n_pos, vocab]."""
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp(min=1e-12))
    kl_pm = (p * (log_p - log_m)).sum(-1)
    kl_qm = (q * (log_q - log_m)).sum(-1)
    return 0.5 * (kl_pm + kl_qm) / math.log(2)


@torch.no_grad()
def forward_logits(model, tokenizer, prompts, *, fwd_hooks, max_length=128):
    """Per-prompt teacher-forced forward pass; returns list of [seq, vocab] logits."""
    device = next(model.parameters()).device
    out = []
    for q in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_length:
            ids = ids[:max_length]
        ids_t = torch.tensor(ids, device=device).unsqueeze(0)
        logits = model.run_with_hooks(ids_t, fwd_hooks=fwd_hooks, reset_hooks_end=True)
        out.append(logits[0].detach())
    return out


def mean_jsd(ref_logits, steered_logits) -> float:
    """Mean JSD (bits) per position across the prompt set."""
    vals = []
    for ref, ste in zip(ref_logits, steered_logits):
        n = min(ref.shape[0], ste.shape[0])
        vals.append(float(jsd_bits(ref[:n], ste[:n]).mean()))
    return float(sum(vals) / max(len(vals), 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True,
                   choices=["qwen-7b", "llama-8b", "qwen-14b", "qwen-32b"])
    p.add_argument("--domain", required=True, choices=["medical", "finance", "sports"])
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--hook-point", default=None,
                   help="Override default hookpoint resolution.")
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--alphas-fra", nargs="+", type=float,
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--alphas-dom", nargs="+", type=float,
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--k-pairs", type=int, default=50)
    p.add_argument("--output-root", default="logs/em_scaling/phase3_jsd")
    args = p.parse_args()

    from fra.em_models import EM_BASES, load_em_model, load_base_model, default_layer, default_head
    from fra.em_evaluation import EM_EVAL_PROMPTS

    info = EM_BASES[args.base]
    layer = args.layer if args.layer is not None else default_layer(args.base)
    hook_point = args.hook_point
    if hook_point is None:
        hook_point = ("ln1.hook_normalized"
                      if info.sae_kind == "qwen_ln1"
                      else "hook_resid_post")
    hook_name = f"blocks.{layer}.{hook_point}"

    out_root = Path(args.output_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    prompts = EM_EVAL_PROMPTS[: args.n_prompts]

    print(f"=== JSD curves: {args.base} / {args.domain} ===")
    print(f"  layer/hook : L{layer} / {hook_point}")
    print(f"  alphas FRA : {args.alphas_fra}")
    print(f"  alphas DoM : {args.alphas_dom}")

    # ── Load EM model + SAE (for FRA) ──────────────────────────────────
    t0 = time.time()
    em_model, tokenizer = load_em_model(args.base, args.domain, verbose=False)
    print(f"[load] EM model in {time.time()-t0:.1f}s")

    # ── Clean (no-hook) reference logits ───────────────────────────────
    t0 = time.time()
    ref_logits = forward_logits(em_model, tokenizer, prompts, fwd_hooks=[],
                                max_length=args.max_length)
    print(f"[ref]  clean forward pass in {time.time()-t0:.1f}s "
          f"(N={len(ref_logits)} prompts)")

    results: dict = {
        "base": args.base, "domain": args.domain,
        "layer": layer, "hook_point": hook_point,
        "n_prompts": len(prompts), "max_length": args.max_length,
        "fra_qk_to_qk": {},
        "dom": {},
    }

    # ── FRA QK→QK sweep (skip if no SAE — qwen-14b placeholder case) ──
    if info.sae_release:
        try:
            from fra.em_models import load_sae_for
            sae = load_sae_for(args.base, verbose=False)
            head = default_head(args.base)
            if head == 0:
                from fra.head_ablation import head_attribution_sweep
                sweep = head_attribution_sweep(
                    em_model, prompts[:3], layer=layer,
                    max_length=args.max_length, verbose=False)
                head = sweep[0]["head"]
            from fra.em_evaluation import rank_features_multi_prompt
            ranked = rank_features_multi_prompt(
                em_model, sae, layer, head, hook_point,
                prompts=prompts, max_length=args.max_length,
                top_k=args.top_k, k_pairs=args.k_pairs, verbose=False,
            )
            qk_features = ranked["qk"]
            print(f"[fra]  head H{head}, {len(qk_features)} QK features ranked")
            feat_t = torch.tensor(qk_features,
                                  device=next(em_model.parameters()).device,
                                  dtype=torch.long)

            def make_qkqk(scale: float):
                def ablate(activation, hook):
                    features = sae.encode(activation)
                    features = features.clone()
                    features[:, :, feat_t] = features[:, :, feat_t] * scale
                    x_modified = sae.decode(features)
                    return x_modified.to(activation.dtype)
                return [(hook_name, ablate)]

            for alpha in args.alphas_fra:
                t = time.time()
                ste = forward_logits(em_model, tokenizer, prompts,
                                     fwd_hooks=make_qkqk(alpha),
                                     max_length=args.max_length)
                d = mean_jsd(ref_logits, ste)
                results["fra_qk_to_qk"][f"{alpha}"] = d
                print(f"  FRA α={alpha:+.1f}  JSD={d:.4f}  ({time.time()-t:.1f}s)")
        except Exception as e:
            print(f"[fra] SKIPPED: {type(e).__name__}: {e}")
    else:
        print("[fra] SKIPPED: no SAE release configured for this base")

    # ── DoM sweep (load base, compute DoM, reload EM if needed) ────────
    # The DoM vector might already be cached in the phase-2 output. If not,
    # we recompute here. Reuse the phase-2 driver's exact same approach so
    # JSD is comparable to the phase-2 sampling result.
    p2_qual = Path(f"logs/em_scaling/phase2_dom/qualitative_DoM_{args.base}_"
                   f"{args.domain}_evalseed42.json")
    dom_vec = None
    if p2_qual.exists():
        # Try to extract dom_norm + recompute from probe (simplest: redo).
        pass
    print("[dom]  recomputing DoM vector from probe prompts …")
    t = time.time()
    from experiments.em_scaling.phase2_dom_steering import compute_mean_activation
    n_probe = min(8, args.n_prompts)
    probe = EM_EVAL_PROMPTS[:n_probe]
    em_mean = compute_mean_activation(em_model, tokenizer, probe, hook_name,
                                      max_length=args.max_length)
    del em_model; torch.cuda.empty_cache()
    base_model, _ = load_base_model(args.base, verbose=False)
    base_mean = compute_mean_activation(base_model, tokenizer, probe, hook_name,
                                        max_length=args.max_length)
    dom_vec = (em_mean - base_mean).to(torch.bfloat16)
    dom_norm = float(dom_vec.float().norm())
    print(f"[dom]  ||v||={dom_norm:.3f}  ({time.time()-t:.1f}s)")
    del base_model; torch.cuda.empty_cache()

    # Reload EM model (we deleted it above to make room for base)
    em_model, _ = load_em_model(args.base, args.domain, verbose=False)
    # Recompute reference logits (deterministic, but tensors live on new model)
    ref_logits = forward_logits(em_model, tokenizer, prompts, fwd_hooks=[],
                                max_length=args.max_length)

    def make_dom(alpha: float):
        delta = (-alpha * dom_vec).to(torch.bfloat16)
        def add_dom(activation, hook):
            return activation + delta.to(activation.dtype)
        return [(hook_name, add_dom)]

    for alpha in args.alphas_dom:
        t = time.time()
        ste = forward_logits(em_model, tokenizer, prompts,
                             fwd_hooks=make_dom(alpha),
                             max_length=args.max_length)
        d = mean_jsd(ref_logits, ste)
        results["dom"][f"{alpha}"] = d
        print(f"  DoM α={alpha:+.1f}  JSD={d:.4f}  ({time.time()-t:.1f}s)")

    results["dom_norm"] = dom_norm
    out_path = out_root / f"jsd_{args.base}_{args.domain}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    sys.exit(main())
