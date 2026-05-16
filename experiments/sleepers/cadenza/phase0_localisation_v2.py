"""
Phase 0 localisation v2 — paper-spec protocol (Cadenza Llama-3 8B).

Patch of ``phase0_localisation.py`` (v1) that aligns more tightly with the
TinyStories appendix protocol after the v1 sweep produced uniformly null
results. The four substantive differences from v1::

  1. **Feature ranking** uses the OV FRA *tensor* contribution diff
     summed over a small set of heads and averaged over the prompt
     positions in ``pmask`` (the positions up to and including the
     ``<|im_start|>assistant`` marker — the Cadenza analog of the
     paper's ``Story:`` marker). v1 ranked by raw last-position
     activation diff, which finds features *correlated* with the
     trigger but not necessarily causally upstream.

  2. **Two-stage protocol** matching ``app:sleeper_method``:
     selection-stage decoding is greedy (deterministic winner pick on a
     small val split), eval-stage decoding is sampled with multiple
     seeds on a disjoint test split. The selected ``(feature*, α*)``
     is the one minimising val ASR subject to the ΔCE budget; the
     test split then re-measures ASR with multi-seed sampling.

  3. **Paper α grid** ``{0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}``
     — non-negative, going up to 4.0 (full ablation at α=0 under the
     ``(α-1)·f·W_dec`` additive rule). v1 used an asymmetric grid
     including negative αs.

  4. **Larger samples**. Default 20 val prompts, 50 test prompts,
     5 sampling seeds for ASR → 250 samples per α-cell vs v1's 10.

Schema of the output JSON is parallel to v1's (one record per cell)
with two extra fields: ``selection.greedy_asr`` (val deterministic) and
``test.per_seed_asr`` (list of length n_seeds).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ── Intervention hook (identical to v1) ──────────────────────────────────


def make_steering_hooks(sae, hook_name: str, feature_idx: int, alpha: float):
    W_dec = sae.W_dec
    feat_dec = W_dec[feature_idx]
    scale_m1 = float(alpha - 1.0)

    def steer(activation, hook):
        x = activation
        feats = sae.encode(x)
        f_t = feats[..., feature_idx]
        delta = scale_m1 * f_t.unsqueeze(-1) * feat_dec.to(x.dtype)
        return x + delta

    return [(hook_name, steer)]


# ── Metric helpers ───────────────────────────────────────────────────────


@torch.no_grad()
def teacher_forced_ce(
    model, tokenizer, formatted_prompt: str, fwd_hooks: list,
    *, target_continuation: str,
) -> float:
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target_continuation, add_special_tokens=False)
    if not target_ids:
        return 0.0
    full = torch.tensor(prompt_ids + target_ids, device=device).unsqueeze(0)
    logits = model.run_with_hooks(
        full, fwd_hooks=fwd_hooks, reset_hooks_end=True,
    )
    pre = len(prompt_ids)
    target_logits = logits[0, pre - 1 : pre - 1 + len(target_ids), :].float()
    target_t = torch.tensor(target_ids, device=device)
    return F.cross_entropy(target_logits, target_t, reduction="mean").item()


# ── FRA-tensor-based feature ranking ─────────────────────────────────────


@torch.no_grad()
def rank_features_fra(
    model, sae, tokenizer,
    layer: int, hook_point: str, heads: List[int],
    val_prompts: List[str], top_k: int, *,
    fra_top_k: int = 20, verbose: bool = True,
) -> List[int]:
    """Top-K candidates by OV FRA contribution diff (paper protocol).

    For each prompt (with and without trigger) and each head in ``heads``,
    compute the OV decomposition and sum |contribution| over (query, key)
    positions to get a per-feature score per prompt. Average across
    prompts to get ``mean_dep`` and ``mean_clean``, then rank features by
    ``|mean_dep − mean_clean|``.
    """
    from fra.core.ov import get_sentence_ov_decomposition
    from fra.llama_sleeper import format_prompt

    def score_prompts(prompts, with_trigger: bool):
        scores = defaultdict(float)
        n = 0
        for q in prompts:
            text = format_prompt(tokenizer, q, with_trigger=with_trigger)
            for h in heads:
                r = get_sentence_ov_decomposition(
                    model, sae, text, layer, h,
                    max_length=128, top_k=fra_top_k,
                    verbose=False, hook_point=hook_point,
                )
                sp = r["ov_sparse"].coalesce()
                idx = sp.indices().cpu().numpy()
                vals = sp.values().cpu().numpy()
                # idx[2] is the feature dim; aggregate |value| per feature
                for f, v in zip(idx[2], vals):
                    scores[int(f)] += abs(float(v))
            n += 1
        return scores, n

    s_dep, n_d = score_prompts(val_prompts, True)
    s_clean, n_c = score_prompts(val_prompts, False)
    feats = set(s_dep) | set(s_clean)
    diffs = [(f, abs(s_dep.get(f, 0) / max(n_d, 1)
                     - s_clean.get(f, 0) / max(n_c, 1)))
             for f in feats]
    diffs.sort(key=lambda x: x[1], reverse=True)
    top = [f for f, _ in diffs[:top_k]]
    if verbose:
        print(f"[rank::fra-ov] top-{top_k}: {top[:10]}{'...' if top_k > 10 else ''}")
        for f, d in diffs[:8]:
            d_dep = s_dep.get(f, 0) / max(n_d, 1)
            d_clean = s_clean.get(f, 0) / max(n_c, 1)
            print(f"  {f:>6}: Δ={d:.3f}  mean_dep={d_dep:.3f}  mean_clean={d_clean:.3f}")
    return top


# ── Main per-cell pipeline ───────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae-path", type=Path, required=True)
    p.add_argument("--hook-layer", type=int, required=True)
    p.add_argument("--hook-point", required=True)
    p.add_argument("--heads", type=int, nargs="+",
                   default=[0, 8, 16, 24],
                   help="Heads to compute FRA over for feature ranking.")
    p.add_argument("--n-candidates", type=int, default=10,
                   help="Top-K candidates by FRA-tensor diff. Tighter than "
                        "v1's 20 to keep selection-stage greedy sweep "
                        "within ~15 min/cell using the slow generation path.")
    p.add_argument(
        "--alphas", type=float, nargs="+",
        default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        help="Paper α grid (default).",
    )
    p.add_argument("--n-val", type=int, default=10,
                   help="Greedy selection split. Smaller than paper's 100 to "
                        "fit the slow-path generation budget; statistical "
                        "resolution comes from the larger sampled test split.")
    p.add_argument("--n-test", type=int, default=50,
                   help="Sampled eval split (disjoint from val). 50 prompts "
                        "× 5 seeds = 250 samples per α-cell — much higher "
                        "resolution than v1's 10 samples.")
    p.add_argument("--eval-seeds", type=int, nargs="+",
                   default=[42, 123, 456, 789, 1011],
                   help="Sampling seeds for eval-stage ASR averaging.")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--ce-budget", type=float, default=0.05)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=Path("logs") / "cadenza_localisation_v2"
                                       / "phase0_localisation_v2.json")
    args = p.parse_args()

    from fra.llama_sleeper import (
        SLEEPER_EVAL_PROMPTS_100, SLEEPER_PHRASE_REGEX,
        asr_match, format_prompt, generate_with_hooks,
        load_cadenza_distilled,
    )
    from fra.sae_lens_wrapper import LocalLn1SAE

    if len(SLEEPER_EVAL_PROMPTS_100) < args.n_val + args.n_test:
        raise ValueError(
            f"Need {args.n_val + args.n_test} prompts; only "
            f"{len(SLEEPER_EVAL_PROMPTS_100)} are defined in fra.llama_sleeper."
        )
    val_qs = SLEEPER_EVAL_PROMPTS_100[: args.n_val]
    test_qs = SLEEPER_EVAL_PROMPTS_100[args.n_val : args.n_val + args.n_test]

    hook_name = f"blocks.{args.hook_layer}.{args.hook_point}"
    print(f"\n=== cell ===")
    print(f"  sae_path   : {args.sae_path}")
    print(f"  hook_name  : {hook_name}")
    print(f"  val/test   : {args.n_val} / {args.n_test} prompts "
          f"(eval seeds: {args.eval_seeds})")
    print(f"  α grid     : {args.alphas}")

    # ── Load ────────────────────────────────────────────────────────────
    t0 = time.time()
    model, tokenizer = load_cadenza_distilled(verbose=False)
    sae = LocalLn1SAE(args.sae_path, layer=args.hook_layer)
    print(f"[load] {time.time()-t0:.1f}s  d_sae={sae.d_sae}")

    # ── 1. FRA-tensor feature ranking ──────────────────────────────────
    print(f"\n[rank] FRA-tensor (OV) ranking on val split, heads={args.heads}")
    t0 = time.time()
    candidates = rank_features_fra(
        model, sae, tokenizer, args.hook_layer, args.hook_point,
        args.heads, val_qs, args.n_candidates,
    )
    print(f"[rank] {time.time()-t0:.1f}s")

    # ── 2. Unsteered clean rollouts + base CE on val ───────────────────
    print("\n[ref] unsteered val baselines")
    val_clean_targets: list[str] = []
    for i, q in enumerate(val_qs):
        prompt = format_prompt(tokenizer, q, with_trigger=False)
        cont = generate_with_hooks(
            model, tokenizer, prompt, fwd_hooks=[],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, seed=args.seed_base + i,
        )
        val_clean_targets.append(cont)
    base_ces = []
    for q, t in zip(val_qs, val_clean_targets):
        base_ces.append(teacher_forced_ce(
            model, tokenizer, format_prompt(tokenizer, q, with_trigger=False),
            fwd_hooks=[], target_continuation=t,
        ))
    base_ce = sum(base_ces) / len(base_ces)
    print(f"[ref] mean unsteered val CE: {base_ce:.4f}")

    # ── 3. Greedy selection sweep on val ────────────────────────────────
    print("\n[select] greedy val α-sweep (paper protocol)")
    t0 = time.time()
    sweep_rows: list[dict] = []
    for f_idx in candidates:
        for alpha in args.alphas:
            hooks = make_steering_hooks(sae, hook_name, f_idx, alpha)
            # ΔCE
            ces = []
            for q, t in zip(val_qs, val_clean_targets):
                ces.append(teacher_forced_ce(
                    model, tokenizer,
                    format_prompt(tokenizer, q, with_trigger=False),
                    fwd_hooks=hooks, target_continuation=t,
                ))
            delta_ce = (sum(ces) / len(ces)) - base_ce
            # Greedy ASR
            fires = 0
            for i, q in enumerate(val_qs):
                resp = generate_with_hooks(
                    model, tokenizer,
                    format_prompt(tokenizer, q, with_trigger=True),
                    fwd_hooks=hooks,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,          # greedy for selection
                    seed=args.seed_base + i,
                )
                fires += int(asr_match(resp))
            asr = fires / len(val_qs)
            sweep_rows.append({
                "feature": f_idx, "alpha": alpha,
                "greedy_asr": asr, "delta_ce": delta_ce,
            })
    print(f"[select] {time.time()-t0:.1f}s  ({len(sweep_rows)} cells)")

    # ── 4. Pick winner under ΔCE budget ─────────────────────────────────
    in_budget = [r for r in sweep_rows if r["delta_ce"] <= args.ce_budget]
    if not in_budget:
        print(f"[select] no cell within ΔCE ≤ {args.ce_budget}; "
              "falling back to all cells.")
        in_budget = sweep_rows
    winner = min(in_budget, key=lambda r: (r["greedy_asr"], r["delta_ce"]))
    print(f"[select] winner: feat={winner['feature']}  α={winner['alpha']:+.2f}  "
          f"val greedy ASR={winner['greedy_asr']:.2f}  "
          f"val ΔCE={winner['delta_ce']:+.4f}")

    # ── 5. Eval-split sampled ASR (multi-seed) ──────────────────────────
    print(f"\n[eval] sampled ASR on disjoint test split "
          f"({args.n_test} prompts × {len(args.eval_seeds)} seeds)")
    t0 = time.time()
    hooks = make_steering_hooks(
        sae, hook_name, winner["feature"], winner["alpha"]
    )
    test_clean_targets: list[str] = []
    for i, q in enumerate(test_qs):
        prompt = format_prompt(tokenizer, q, with_trigger=False)
        cont = generate_with_hooks(
            model, tokenizer, prompt, fwd_hooks=[],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed_base + 1000 + i,
        )
        test_clean_targets.append(cont)

    # ΔCE on test split
    test_base = [
        teacher_forced_ce(
            model, tokenizer,
            format_prompt(tokenizer, q, with_trigger=False),
            fwd_hooks=[], target_continuation=t,
        ) for q, t in zip(test_qs, test_clean_targets)
    ]
    test_steered = [
        teacher_forced_ce(
            model, tokenizer,
            format_prompt(tokenizer, q, with_trigger=False),
            fwd_hooks=hooks, target_continuation=t,
        ) for q, t in zip(test_qs, test_clean_targets)
    ]
    test_delta_ce = (sum(test_steered) - sum(test_base)) / len(test_qs)

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
    print(f"[eval] {time.time()-t0:.1f}s")
    print(f"[eval] test sampled ASR (mean of {len(args.eval_seeds)} seeds): "
          f"{asr_mean:.3f}  per_seed={per_seed_asr}")
    print(f"[eval] test ΔCE: {test_delta_ce:+.4f}")

    # ── 6. Persist ──────────────────────────────────────────────────────
    record = {
        "version": "v2",
        "cell": {
            "hook_layer": args.hook_layer,
            "hook_point": args.hook_point,
            "hook_name": hook_name,
            "sae_path": str(args.sae_path),
            "d_sae": sae.d_sae,
        },
        "selection": {
            "feature": winner["feature"],
            "alpha": winner["alpha"],
            "val_greedy_asr": winner["greedy_asr"],
            "val_delta_ce": winner["delta_ce"],
            "ce_budget": args.ce_budget,
        },
        "test": {
            "asr_mean": asr_mean,
            "per_seed_asr": per_seed_asr,
            "delta_ce": test_delta_ce,
            "n": len(test_qs),
            "n_seeds": len(args.eval_seeds),
        },
        "candidates": candidates,
        "sweep": sweep_rows,
        "config": {
            "alphas": args.alphas,
            "n_candidates": args.n_candidates,
            "n_val": args.n_val,
            "n_test": args.n_test,
            "eval_seeds": args.eval_seeds,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed_base": args.seed_base,
            "heads_for_fra_ranking": args.heads,
        },
    }
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
