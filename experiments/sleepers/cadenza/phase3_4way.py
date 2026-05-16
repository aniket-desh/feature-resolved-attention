"""
Phase-3 (a): 4-way comparison at L29 — does suppression go through
attention or only post-MLP?

For the *same* trigger task at L29, run four single-vector interventions
side by side and report the same four headline metrics on a disjoint
50-prompt eval split × 5 sampling seeds:

  fra_ov_via_hookv      OV-via-attn.hook_v using the L29 ln1.hook_normalized SAE
                        (this is the FRA OV→OV recipe from the paper's main figure)
  conv_additive_ln1     additive at ln1.hook_normalized using the same SAE
  conv_additive_mid     additive at hook_resid_mid using the L29 resid_mid SAE
  conv_additive_post    additive at hook_resid_post using the L29 resid_post SAE

Picks the best (feature, α) per recipe by greedy selection on a small
val split, then re-measures with multi-seed sampling on the test split.

Schema is parallel to the four-config JSON consumed by
``plot_combined_50k.py`` (extended with two extra configs).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fra-sae-path",       type=Path, required=True,
                   help="L29 ln1.hook_normalized SAE (used by fra_ov + conv_additive_ln1).")
    p.add_argument("--mid-sae-path",       type=Path, required=True,
                   help="L29 hook_resid_mid SAE (used by conv_additive_mid).")
    p.add_argument("--post-sae-path",      type=Path, required=True,
                   help="L29 hook_resid_post SAE (used by conv_additive_post).")
    p.add_argument("--layer", type=int, default=29)
    p.add_argument("--n-val",  type=int, default=8,
                   help="Greedy selection prompts per recipe.")
    p.add_argument("--n-test", type=int, default=20,
                   help="Sampled eval prompts (disjoint from val).")
    p.add_argument(
        "--alphas", type=float, nargs="+",
        default=[-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
    )
    p.add_argument("--alphas-for-selection", type=float, nargs="+",
                   default=[-4.0, -2.0, -1.0, 2.0])
    p.add_argument("--n-candidates", type=int, default=20)
    p.add_argument("--eval-seeds", type=int, nargs="+",
                   default=[42, 123, 456, 789, 1011])
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=Path("logs/cadenza_phase3/4way_metrics.json"))
    args = p.parse_args()

    from fra.llama_sleeper import (
        SLEEPER_EVAL_PROMPTS_100, asr_match, format_prompt,
        generate_with_hooks, load_cadenza_distilled,
    )
    from fra.sae_lens_wrapper import LocalLn1SAE
    from experiments.sleepers.cadenza.phase1_single_feature import (
        _OVFactory, _ResidFactory, pick_best_feature, sweep_alpha,
    )

    if len(SLEEPER_EVAL_PROMPTS_100) < args.n_val + args.n_test:
        raise ValueError("need more prompts in SLEEPER_EVAL_PROMPTS_100")
    val_qs  = SLEEPER_EVAL_PROMPTS_100[: args.n_val]
    eval_qs = SLEEPER_EVAL_PROMPTS_100[args.n_val : args.n_val + args.n_test]

    t0 = time.time()
    model, tokenizer = load_cadenza_distilled(verbose=False)
    sae_ln1  = LocalLn1SAE(args.fra_sae_path,  layer=args.layer)
    sae_mid  = LocalLn1SAE(args.mid_sae_path,  layer=args.layer)
    sae_post = LocalLn1SAE(args.post_sae_path, layer=args.layer)
    print(f"[load] {time.time()-t0:.1f}s")

    recipes = [
        ("fra_ov_via_hookv",
            _OVFactory(sae_ln1,  model, "ln1.hook_normalized", args.layer, "all")),
        ("conv_additive_ln1",
            _ResidFactory(sae_ln1, args.layer, "ln1.hook_normalized")),
        ("conv_additive_mid",
            _ResidFactory(sae_mid, args.layer, "hook_resid_mid")),
        ("conv_additive_post",
            _ResidFactory(sae_post, args.layer, "hook_resid_post")),
    ]

    configs: dict = {}
    for name, factory in recipes:
        print(f"\n=== recipe: {name} ===")
        pick = pick_best_feature(
            model, tokenizer, factory.sae, factory,
            val_qs, n_candidates=args.n_candidates,
            alphas_for_selection=args.alphas_for_selection,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, seed_base=args.seed_base,
        )
        print(f"  → winner: feat={pick['feature']} α*={pick['alpha']:+.2f} "
              f"val ASR={pick['asr']:.2f}")

        per_alpha_runs = []
        # Use multiple seeds for the eval-stage rollouts (multi-seed avg).
        for seed in args.eval_seeds:
            per_alpha = sweep_alpha(
                model, tokenizer, factory, pick["feature"],
                eval_qs, alphas=args.alphas,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, seed_base=seed,
            )
            per_alpha_runs.append(per_alpha)

        # Aggregate per-α metrics into lists across seeds (the existing plot
        # script's "multi-seed bands" mode).
        merged: dict = {}
        for a in args.alphas:
            key = str(a)
            jc = [r[key]["jsd_clean"] for r in per_alpha_runs]
            jp = [r[key]["jsd_pois"]  for r in per_alpha_runs]
            em = [r[key]["n_exact_match_clean"] for r in per_alpha_runs]
            asr= [r[key]["asr"] for r in per_alpha_runs]
            merged[key] = {
                "jsd_clean": jc, "jsd_pois": jp,
                "n_exact_match_clean": em, "asr": asr,
            }

        configs[name] = {
            "feature": pick["feature"],
            "alpha_select": pick["alpha"],
            "sae_path": str(factory.sae.checkpoint_path),
            "per_alpha": merged,
        }

    record = {
        "alphas": args.alphas,
        "configs": configs,
        "meta": {
            "layer": args.layer,
            "n_val": args.n_val, "n_test": args.n_test,
            "eval_seeds": args.eval_seeds,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
