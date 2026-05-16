"""
Validate a specific (feature, α) winner under paper-spec sampling.

Use case: a low-power sweep (e.g. ``phase0_localisation.py`` with n=10
single-seed) surfaced a promising (feat, α). To know whether the signal
is real or an N=10 sampling artifact, re-measure under the paper protocol
(50 prompts on a disjoint test split × 5 sampling seeds = 250 samples)
plus a clean-prompt ΔCE.

Output JSON has the same ``test`` block schema as v2's localisation
output so summaries can cross-reference v1, v2, and validated winners
in one table.

Usage::

    python -m experiments.sleepers.cadenza.validate_winner \\
        --sae-path  /workspace/aniket/saes/cadenza_L29_hook_resid_post/<id>/<step> \\
        --hook-layer 29 --hook-point hook_resid_post \\
        --feature 12402 --alpha -4.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.sleepers.cadenza.phase0_localisation_v2 import (
    make_steering_hooks, teacher_forced_ce,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae-path", type=Path, required=True)
    p.add_argument("--hook-layer", type=int, required=True)
    p.add_argument("--hook-point", required=True)
    p.add_argument("--feature", type=int, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--n-val", type=int, default=10,
                   help="Held-out prompts used as the disjoint val prefix; "
                        "must match the v1/v2 split to keep test disjoint.")
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--eval-seeds", type=int, nargs="+",
                   default=[42, 123, 456, 789, 1011])
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=Path("logs") / "cadenza_validation"
                                       / "validate.json")
    args = p.parse_args()

    from fra.llama_sleeper import (
        SLEEPER_EVAL_PROMPTS_100, asr_match, format_prompt,
        generate_with_hooks, load_cadenza_distilled,
    )
    from fra.sae_lens_wrapper import LocalLn1SAE

    test_qs = SLEEPER_EVAL_PROMPTS_100[args.n_val : args.n_val + args.n_test]
    hook_name = f"blocks.{args.hook_layer}.{args.hook_point}"
    print(f"\n=== validating (feat={args.feature}, α={args.alpha:+.2f}) at {hook_name} ===")
    print(f"  test prompts : {len(test_qs)}")
    print(f"  eval seeds   : {args.eval_seeds}")

    model, tokenizer = load_cadenza_distilled(verbose=False)
    sae = LocalLn1SAE(args.sae_path, layer=args.hook_layer)
    hooks = make_steering_hooks(sae, hook_name, args.feature, args.alpha)

    # Clean rollouts for ΔCE teacher-forcing
    clean_targets: list[str] = []
    for i, q in enumerate(test_qs):
        prompt = format_prompt(tokenizer, q, with_trigger=False)
        cont = generate_with_hooks(
            model, tokenizer, prompt, fwd_hooks=[],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed_base + 1000 + i,
        )
        clean_targets.append(cont)

    base_ces = [
        teacher_forced_ce(
            model, tokenizer,
            format_prompt(tokenizer, q, with_trigger=False),
            fwd_hooks=[], target_continuation=t,
        ) for q, t in zip(test_qs, clean_targets)
    ]
    steered_ces = [
        teacher_forced_ce(
            model, tokenizer,
            format_prompt(tokenizer, q, with_trigger=False),
            fwd_hooks=hooks, target_continuation=t,
        ) for q, t in zip(test_qs, clean_targets)
    ]
    delta_ce = (sum(steered_ces) - sum(base_ces)) / len(test_qs)

    # Multi-seed sampled ASR
    per_seed_asr = []
    for seed in args.eval_seeds:
        fires = 0
        for i, q in enumerate(test_qs):
            resp = generate_with_hooks(
                model, tokenizer,
                format_prompt(tokenizer, q, with_trigger=True),
                fwd_hooks=hooks,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=seed + i,
            )
            fires += int(asr_match(resp))
        per_seed_asr.append(fires / len(test_qs))
    asr_mean = sum(per_seed_asr) / len(per_seed_asr)

    print(f"\n[result] mean test ASR = {asr_mean:.3f}   (per-seed: {per_seed_asr})")
    print(f"[result] test ΔCE       = {delta_ce:+.4f}")

    record = {
        "cell": {
            "hook_layer": args.hook_layer,
            "hook_point": args.hook_point,
            "hook_name": hook_name,
            "sae_path": str(args.sae_path),
            "feature": args.feature,
            "alpha": args.alpha,
        },
        "test": {
            "asr_mean": asr_mean,
            "per_seed_asr": per_seed_asr,
            "delta_ce": delta_ce,
            "n": len(test_qs),
            "n_seeds": len(args.eval_seeds),
        },
    }
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
