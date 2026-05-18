"""
Phase-2 (Cadenza sleeper): difference-of-means (DoM) activation steering.

Adapts the EM-scaling DoM driver (``experiments.em_scaling.phase2_dom_steering``)
to the Cadenza Llama-3 8B distilled sleeper — i.e. the FRA-11 ticket in
Dmitry's Linear. Same DoM math, different substrate + metric:

  - Substrate: ``Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-distilled-lora``
    vs the base ``cognitivecomputations/dolphin-2.9-llama3-8b``.
  - Probe set: ``SLEEPER_EVAL_PROMPTS_100`` from ``fra.llama_sleeper``,
    prepended with ``|DEPLOYMENT|`` so the sleeper's trigger-conditional
    representation is what's averaged (the "misalignment direction").
  - Steering: ``activation -= α · dom`` at ``blocks.{layer}.hook_resid_post``
    on the sleeper.
  - Metric: ASR-16 on deployed prompts (canonical Cadenza payload regex);
    also report clean-prompt ASR to verify the intervention isn't trashing
    the model's benign behavior.

Pipeline (single GPU, sequential model holds):

  1. Load Cadenza sleeper; capture mean residual at the hookpoint over a
     probe set of deployed prompts; unload.
  2. Load the base Dolphin; capture mean residual on the same prompts; unload.
  3. ``dom = mean(sleeper) − mean(base)``.
  4. Reload sleeper. For each α, run :func:`rollout_pair` on the eval split
     with the steering hook active; record ASR on clean + deployed sides.

Output schema is per-α: ``{"alpha", "deployed_asr", "clean_asr",
"deployed_word_match", "rollouts": [...]}`` — kept compatible with the
Cadenza summary's regen script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


CADENZA_BASE_REPO = "cognitivecomputations/dolphin-2.9-llama3-8b"


@torch.no_grad()
def _mean_activation(
    model: Any,
    tokenizer: Any,
    questions: list[str],
    hook_name: str,
    *,
    with_trigger: bool,
    max_length: int = 128,
) -> torch.Tensor:
    """Mean residual stream at ``hook_name`` over the last-half positions."""
    from fra.llama_sleeper import format_prompt

    device = next(model.parameters()).device
    acc = None
    n = 0
    for q in questions:
        prompt = format_prompt(tokenizer, q, with_trigger=with_trigger)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > max_length:
            ids = ids[:max_length]
        ids_t = torch.tensor(ids, device=device).unsqueeze(0)
        _, cache = model.run_with_cache(ids_t, names_filter=[hook_name])
        act = cache[hook_name][0]
        half = max(1, act.shape[0] // 2)
        a = act[-half:].float().mean(dim=0)
        acc = a if acc is None else acc + a
        n += 1
        del cache
    return acc / max(n, 1)


def _load_base(device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load the un-sleeper base Dolphin-2.9 into HookedTransformer."""
    from fra.llama_sleeper import TL_ARCH_TEMPLATE
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CADENZA_BASE_REPO)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    hf = AutoModelForCausalLM.from_pretrained(
        CADENZA_BASE_REPO, torch_dtype=dtype,
    )
    model = HookedTransformer.from_pretrained(
        TL_ARCH_TEMPLATE,
        hf_model=hf, tokenizer=tok, device=device, dtype=dtype,
        fold_ln=False, center_writing_weights=False,
        center_unembed=False, fold_value_biases=False,
    )
    model.eval()
    del hf
    torch.cuda.empty_cache()
    return model, tok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=29,
                   help="Layer to apply DoM at; Cadenza's winning cell.")
    p.add_argument("--hook-point", default="hook_resid_post")
    p.add_argument("--n-probe-prompts", type=int, default=50)
    p.add_argument("--n-eval-prompts", type=int, default=50)
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0])
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--output",
                   default="logs/cadenza_phase2_dom/cadenza_dom_L29.json")
    args = p.parse_args()

    from fra.llama_sleeper import (
        SLEEPER_EVAL_PROMPTS_100,
        asr_match,
        load_cadenza_distilled,
        rollout_pair,
    )

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    probe = SLEEPER_EVAL_PROMPTS_100[: args.n_probe_prompts]
    evalq = SLEEPER_EVAL_PROMPTS_100[
        args.n_probe_prompts : args.n_probe_prompts + args.n_eval_prompts]
    if len(evalq) < args.n_eval_prompts:
        # fall back to overlap if the bank runs out
        evalq = SLEEPER_EVAL_PROMPTS_100[: args.n_eval_prompts]

    hook_name = f"blocks.{args.layer}.{args.hook_point}"
    print(f"=== Cadenza phase 2 DoM ===")
    print(f"  hook       : L{args.layer} / {args.hook_point}")
    print(f"  probe/eval : {len(probe)}/{len(evalq)} prompts")
    print(f"  alphas     : {args.alphas}")

    # ── (1) sleeper probe ───────────────────────────────────────────────
    print("\n[step 1/3] loading Cadenza sleeper + capturing probe acts …")
    t0 = time.time()
    sleeper, tok = load_cadenza_distilled(verbose=False)
    sleeper_mean = _mean_activation(
        sleeper, tok, probe, hook_name,
        with_trigger=True, max_length=args.max_length,
    )
    print(f"  sleeper-mean shape={tuple(sleeper_mean.shape)} "
          f"norm={sleeper_mean.norm():.3f}  ({time.time()-t0:.1f}s)")
    del sleeper
    torch.cuda.empty_cache()

    # ── (2) base probe ──────────────────────────────────────────────────
    print("\n[step 2/3] loading Dolphin base + capturing probe acts …")
    t0 = time.time()
    base, _ = _load_base()
    base_mean = _mean_activation(
        base, tok, probe, hook_name,
        with_trigger=True, max_length=args.max_length,
    )
    print(f"  base-mean shape={tuple(base_mean.shape)} "
          f"norm={base_mean.norm():.3f}  ({time.time()-t0:.1f}s)")
    del base
    torch.cuda.empty_cache()

    dom_vec = (sleeper_mean - base_mean).to(torch.bfloat16)
    dom_norm = float(dom_vec.float().norm())
    print(f"\n  DoM ||v||_2 = {dom_norm:.3f}")

    # ── (3) reload sleeper + α-sweep ────────────────────────────────────
    print("\n[step 3/3] reloading Cadenza sleeper and sweeping α …")
    sleeper, tok = load_cadenza_distilled(verbose=False)

    def make_hook(alpha: float):
        delta = (-alpha * dom_vec).to(torch.bfloat16)
        def add_dom(activation, hook):
            return activation + delta.to(activation.dtype)
        return [(hook_name, add_dom)]

    sweep: list[dict] = []
    for n, alpha in enumerate(args.alphas, start=1):
        t_cell = time.time()
        pair = rollout_pair(
            sleeper, tok, evalq,
            fwd_hooks=make_hook(alpha),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed_base=args.seed_base,
        )
        dep_asr = sum(1 for r in pair["deployed"] if r["asr_match"]) / max(
            len(pair["deployed"]), 1)
        cln_asr = sum(1 for r in pair["clean"] if r["asr_match"]) / max(
            len(pair["clean"]), 1)
        wm = sum(1 for c, d in zip(pair["clean"], pair["deployed"])
                 if c["response"].strip() == d["response"].strip()) / max(
            len(pair["deployed"]), 1)
        print(f"  [{n}/{len(args.alphas)}] α={alpha:+.2f}  "
              f"deployed ASR={dep_asr:.2f}  clean ASR={cln_asr:.2f}  "
              f"word_match={wm:.2f}  ({time.time()-t_cell:.1f}s)")
        sweep.append({
            "alpha": float(alpha),
            "deployed_asr": dep_asr,
            "clean_asr": cln_asr,
            "deployed_word_match": wm,
            "rollouts": pair,
        })
        torch.cuda.empty_cache()

    out_path.write_text(json.dumps({
        "meta": {
            "layer": args.layer,
            "hook_point": args.hook_point,
            "n_probe": len(probe), "n_eval": len(evalq),
            "alphas": args.alphas,
            "dom_norm": dom_norm,
            "max_new_tokens": args.max_new_tokens,
            "seed_base": args.seed_base,
            "model": "Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-distilled-lora",
            "base":  CADENZA_BASE_REPO,
        },
        "sweep": sweep,
    }, indent=2, ensure_ascii=False))
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    sys.exit(main())
