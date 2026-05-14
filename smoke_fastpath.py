"""
Phase 0 fast-path smoke test.

Loads Qwen2.5-14B + medical LoRA once, then runs three generators on the
same prompts and seeds:

  1. generate_with_hooks       (slow path, no KV cache, batch=1)
  2. generate_with_hooks_fast  (KV cache, batch=1)        — Path A
  3. generate_with_hooks_batch (KV cache, batch=N)         — Path A+B

For each path: tokens/sec, first 60 chars of generations, agreement
indicators. No GPT-4o calls — judge agreement is checked separately on
the validation slice.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fra.em_evaluation import (
    EM_EVAL_PROMPTS,
    generate_with_hooks,
    generate_with_hooks_fast,
    generate_with_hooks_batch,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--em-model", default="medical")
    p.add_argument("--n-prompts", type=int, default=3, help="Number of EM prompts to test")
    p.add_argument("--max-new-tokens", type=int, default=80,
                   help="Smaller than 200 for smoke; production runs use 200")
    p.add_argument("--skip-slow", action="store_true",
                   help="Skip slow path (already known to be 7-12 tok/s; saves model time on re-runs)")
    p.add_argument("--skip-fast-single", action="store_true",
                   help="Skip the single-prompt KV-cache path (already smoke-tested)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()

    print(f"=== Fast-path smoke test ===")
    print(f"  em_model={args.em_model}  n_prompts={args.n_prompts}  "
          f"max_new_tokens={args.max_new_tokens}  seed={args.seed}")

    # Reuse the published loader so we get the exact merged Qwen-14B + LoRA setup.
    from run_experiments import load_model_and_sae
    print(f"\n[load] Loading model + SAE …")
    t0 = time.time()
    model, sae = load_model_and_sae(layer=24, device="cuda", em_model=args.em_model)
    print(f"[load] done in {time.time() - t0:.1f}s")

    tokenizer = model.tokenizer
    prompts = EM_EVAL_PROMPTS[: args.n_prompts]

    # No hooks for smoke (just generation, not steering).
    fwd_hooks: list = []
    common_kw = dict(
        fwd_hooks=fwd_hooks,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    slow_outs: list[str] = []
    slow_elapsed = 0.0
    slow_tps = 0.0
    if not args.skip_slow:
        # ── Path 1: slow (the published reference) ──────────────────────────────
        print(f"\n[path 1/3] generate_with_hooks (slow, no cache, batch=1)")
        t0 = time.time()
        total_slow_tokens = 0
        for prompt in prompts:
            out = generate_with_hooks(model, tokenizer, prompt, **common_kw)
            slow_outs.append(out)
            total_slow_tokens += len(tokenizer.encode(out))
        slow_elapsed = time.time() - t0
        slow_tps = total_slow_tokens / max(slow_elapsed, 1e-9)
        print(f"  {slow_elapsed:.1f}s total | {slow_tps:.1f} tok/s (decoded) | "
              f"{len(prompts)} prompts × ~{args.max_new_tokens} new tokens")
        for i, o in enumerate(slow_outs):
            print(f"  [{i}] {o[:60]!r}")

    fast_outs: list[str] = []
    fast_elapsed = 0.0
    fast_tps = 0.0
    if not args.skip_fast_single:
        # ── Path 2: KV cache, batch=1 ────────────────────────────────────
        print(f"\n[path 2/3] generate_with_hooks_fast (KV cache, batch=1)")
        t0 = time.time()
        total_fast_tokens = 0
        for prompt in prompts:
            out = generate_with_hooks_fast(model, tokenizer, prompt, **common_kw)
            fast_outs.append(out)
            total_fast_tokens += len(tokenizer.encode(out))
        fast_elapsed = time.time() - t0
        fast_tps = total_fast_tokens / max(fast_elapsed, 1e-9)
        ref = slow_elapsed if slow_elapsed > 0 else fast_elapsed
        print(f"  {fast_elapsed:.1f}s total | {fast_tps:.1f} tok/s (decoded) | "
              f"speedup vs slow: {ref/max(fast_elapsed,1e-9):.1f}×")
        for i, o in enumerate(fast_outs):
            print(f"  [{i}] {o[:60]!r}")

        if slow_outs:
            n_match = sum(1 for s, f in zip(slow_outs, fast_outs) if s == f)
            print(f"  byte-identical with slow path: {n_match}/{len(prompts)} prompts")
            if n_match < len(prompts):
                for i, (s, f) in enumerate(zip(slow_outs, fast_outs)):
                    if s != f:
                        k = next((k for k in range(min(len(s), len(f))) if s[k] != f[k]), -1)
                        print(f"    [{i}] diverges at char {k}; lengths slow={len(s)} fast={len(f)}")

    # ── Path 3: KV cache + batch=N ──────────────────────────────────────
    print(f"\n[path 3/3] generate_with_hooks_batch (KV cache, batch={len(prompts)})")
    t0 = time.time()
    batch_outs = generate_with_hooks_batch(
        model, tokenizer, prompts, **common_kw,
    )
    batch_elapsed = time.time() - t0
    total_batch_tokens = sum(len(tokenizer.encode(o)) for o in batch_outs)
    batch_tps = total_batch_tokens / max(batch_elapsed, 1e-9)
    ref = slow_elapsed if slow_elapsed > 0 else batch_elapsed
    print(f"  {batch_elapsed:.1f}s total | {batch_tps:.1f} tok/s (decoded, summed) | "
          f"speedup vs slow: {ref/max(batch_elapsed,1e-9):.1f}×")
    for i, o in enumerate(batch_outs):
        print(f"  [{i}] {o[:60]!r}")

    print("\n=== summary ===")
    if slow_elapsed > 0:
        print(f"  slow    : {slow_elapsed:6.1f}s   {slow_tps:6.1f} tok/s")
    if fast_elapsed > 0:
        ref = slow_elapsed if slow_elapsed > 0 else fast_elapsed
        print(f"  fast    : {fast_elapsed:6.1f}s   {fast_tps:6.1f} tok/s   "
              f"(×{ref/max(fast_elapsed,1e-9):.1f})")
    if batch_elapsed > 0:
        ref = slow_elapsed if slow_elapsed > 0 else batch_elapsed
        print(f"  batch   : {batch_elapsed:6.1f}s   {batch_tps:6.1f} tok/s   "
              f"(×{ref/max(batch_elapsed,1e-9):.1f})")


if __name__ == "__main__":
    main()
