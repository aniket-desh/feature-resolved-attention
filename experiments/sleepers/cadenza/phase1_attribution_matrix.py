"""
Phase 1 — 3x3 attribution x intervention matrix (Cadenza Llama-3 8B).

Analog of ``fig:matrix_scatter`` from the paper. At a chosen winning layer
(typically the FRA-native ``ln1.hook_normalized`` SAE), enumerate

  attribution channel  : { OV,  QK,  joint(OV+QK) }
  intervention pathway : { OV,  QK,  joint(OV+QK) }

For each (channel, pathway) cell:

  1. Rank features at the chosen layer by ``|mean_dep contrib − mean_clean contrib|``
     using ``channel`` as the FRA tensor.
  2. Take the top ``--n-features``.
  3. α-sweep over a small grid; per α run the ``pathway`` intervention and
     measure ASR-16 + JSD(steered, clean-baseline).
  4. Pick α* minimising ASR (ties broken by smallest JSD).

Outputs a 9-row JSON suitable for the 3x3 scatter plot. Naming + schema is
parallel to the paper's ``matrix_scatter.pdf`` data file so a thin plotter
can render it without bespoke loading code.

This script trades exhaustive feature-pair enumeration for a pragmatic
``top-N feature × α`` budget; the goal is to surface the OV/OV-dominant
cell (or its absence) — not to reproduce every entry of the published
TinyStories matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ── FRA-based attribution ────────────────────────────────────────────────


@torch.no_grad()
def rank_features_diff(
    model, sae, tokenizer,
    layer: int, heads: List[int], hook_point: str,
    val_prompts: List[str],
    *,
    channel: str,                        # "ov" | "qk" | "joint"
    top_k: int = 20,
    fra_top_k: int = 20,
    verbose: bool = True,
) -> List[int]:
    """Top-N candidate features for a given attribution channel.

    Score per feature ``λ``::

        s(λ) = |mean_{prompt ∈ dep} c_h(prompt, λ) − mean_{prompt ∈ clean} c_h(prompt, λ)|

    summed over the selected heads. ``c_h`` is the relevant FRA aggregate
    (OV: sum over (q, k) of |OV contribution|; QK: feature's max appearance
    in top-50 QK pairs by |contribution|).
    """
    from fra.core.fra import get_sentence_fra_batch
    from fra.core.ov import get_sentence_ov_decomposition, rank_ov_features
    from fra.core.helpers import rank_pairs
    from fra.llama_sleeper import format_prompt

    def _ov_scores(prompts, with_trigger):
        scores: dict = defaultdict(float)
        for q in prompts:
            text = format_prompt(tokenizer, q, with_trigger=with_trigger)
            for h in heads:
                r = get_sentence_ov_decomposition(
                    model, sae, text, layer, h,
                    max_length=64, top_k=fra_top_k, verbose=False,
                    hook_point=hook_point,
                )
                for f, abs_sum, _count, _max in rank_ov_features(
                    r["ov_sparse"], mode="sum",
                ):
                    scores[int(f)] += abs_sum
        return scores

    def _qk_scores(prompts, with_trigger):
        scores: dict = defaultdict(float)
        for q in prompts:
            text = format_prompt(tokenizer, q, with_trigger=with_trigger)
            for h in heads:
                r = get_sentence_fra_batch(
                    model, sae, text, layer, h,
                    max_length=64, top_k=fra_top_k, verbose=False,
                    hook_point=hook_point,
                )
                idx = r["fra_tensor_sparse"].coalesce().indices().cpu().numpy()
                val = r["fra_tensor_sparse"].coalesce().values().cpu().numpy()
                for q_f, k_f, abs_sum, _, _ in rank_pairs(
                    idx, val, top_k=50, diagonal=False, mode="sum",
                ):
                    scores[int(q_f)] += abs_sum * 0.5
                    scores[int(k_f)] += abs_sum * 0.5
        return scores

    if channel == "ov":
        s_dep = _ov_scores(val_prompts, True)
        s_clean = _ov_scores(val_prompts, False)
    elif channel == "qk":
        s_dep = _qk_scores(val_prompts, True)
        s_clean = _qk_scores(val_prompts, False)
    elif channel == "joint":
        s_dep_ov = _ov_scores(val_prompts, True)
        s_dep_qk = _qk_scores(val_prompts, True)
        s_clean_ov = _ov_scores(val_prompts, False)
        s_clean_qk = _qk_scores(val_prompts, False)
        s_dep = {k: s_dep_ov.get(k, 0) + s_dep_qk.get(k, 0)
                 for k in set(s_dep_ov) | set(s_dep_qk)}
        s_clean = {k: s_clean_ov.get(k, 0) + s_clean_qk.get(k, 0)
                   for k in set(s_clean_ov) | set(s_clean_qk)}
    else:
        raise ValueError(channel)

    all_feats = set(s_dep) | set(s_clean)
    diffs = [(f, abs(s_dep.get(f, 0) - s_clean.get(f, 0))) for f in all_feats]
    diffs.sort(key=lambda x: x[1], reverse=True)
    top = [f for f, _ in diffs[:top_k]]
    if verbose:
        print(f"[rank::{channel}] top-{top_k}: {top}")
    return top


# ── Steering hook builders (multi-feature variants) ──────────────────────


def make_ov_multi_hooks(sae, model, hook_point: str, layer: int,
                        features: List[int], alpha: float):
    """OV steering on multiple features, applied across ALL KV heads."""
    from fra.core.helpers import get_W_V

    device = next(model.parameters()).device
    W_dec = sae.W_dec.float()
    n_q = model.cfg.n_heads
    n_kv = getattr(model.cfg, "n_key_value_heads", None) or n_q
    feat_t = torch.tensor(features, device=device, dtype=torch.long)
    feat_dec = W_dec[feat_t]                                # [n_feat, d_model]
    # Project to V for each KV head
    feat_v_per_kv = torch.stack(
        [feat_dec @ get_W_V(model, layer, h * n_q // n_kv).float()
         for h in range(n_kv)],
        dim=0,
    )                                                       # [n_kv, n_feat, d_head]
    scale_m1 = float(alpha - 1.0)
    sae_hook = f"blocks.{layer}.{hook_point}"
    v_hook = f"blocks.{layer}.attn.hook_v"

    cached: dict = {}

    def capture(activation, hook):
        x = activation
        if x.dim() == 4:
            x = x.flatten(-2, -1)
        feats = sae.encode(x)
        cached["f"] = feats[..., feat_t].float()           # [..., seq, n_feat]
        return activation

    def steer_v(v, hook):
        f = cached.get("f")
        if f is None:
            return v
        sl = min(f.shape[-2], v.shape[1])
        # delta_per_kv: [n_kv, seq, d_head]  via f * feat_v_per_kv
        # f is [batch?, seq, n_feat], feat_v_per_kv is [n_kv, n_feat, d_head]
        f_slice = f[..., :sl, :].to(v.dtype)
        delta = torch.einsum("...sf,kfd->...skd", f_slice, feat_v_per_kv.to(v.dtype))
        v[..., :sl, :, :] += scale_m1 * delta
        return v

    return [(sae_hook, capture), (v_hook, steer_v)]


def make_qk_multi_hooks(sae, layer: int, hook_point: str,
                        features: List[int], alpha: float):
    """QK-channel intervention: ablate / amplify the features at the SAE input."""
    W_dec = sae.W_dec.float()
    feat_t = torch.tensor(features, dtype=torch.long)
    feat_dec = W_dec[feat_t]                               # [n_feat, d_model]
    scale_m1 = float(alpha - 1.0)
    hook_name = f"blocks.{layer}.{hook_point}"

    def steer(activation, hook):
        x = activation
        feats = sae.encode(x)
        f_t = feats[..., feat_t]                           # [..., seq, n_feat]
        delta = scale_m1 * torch.einsum(
            "...sf,fd->...sd", f_t.to(x.dtype), feat_dec.to(x.dtype),
        )
        return x + delta

    return [(hook_name, steer)]


def make_joint_hooks(sae, model, hook_point: str, layer: int,
                     features: List[int], alpha: float):
    """Both OV and QK paths active simultaneously."""
    return make_ov_multi_hooks(sae, model, hook_point, layer, features, alpha) \
         + make_qk_multi_hooks(sae, layer, hook_point, features, alpha)


# ── JSD + sampling helpers (copied from phase1_single_feature) ───────────


def jsd_bits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    p = log_p.exp(); q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp(min=1e-12))
    kl_pm = (p * (log_p - log_m)).sum(-1)
    kl_qm = (q * (log_q - log_m)).sum(-1)
    return 0.5 * (kl_pm + kl_qm) / math.log(2)


@torch.no_grad()
def teacher_forced_logits(
    model, tokenizer, formatted_prompt: str, target_ids: List[int],
    *, fwd_hooks: list,
) -> torch.Tensor:
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    full = torch.tensor(prompt_ids + target_ids, device=device).unsqueeze(0)
    logits = model.run_with_hooks(
        full, fwd_hooks=fwd_hooks, reset_hooks_end=True,
    )
    pre = len(prompt_ids)
    return logits[0, pre - 1 : pre - 1 + len(target_ids), :]


# ── Per-cell evaluation ──────────────────────────────────────────────────


@torch.no_grad()
def evaluate_cell(
    model, tokenizer, sae,
    layer: int, hook_point: str,
    features: List[int],
    pathway: str,
    alphas: List[float],
    eval_prompts: List[str],
    clean_rollouts: Dict[int, List[int]],
    ref_clean_logits: Dict[int, torch.Tensor],
    *,
    max_new_tokens: int = 16,
    temperature: float = 1.0,
    seed_base: int = 42,
) -> Dict[str, Any]:
    """Sweep α for one cell. Return the best (α, asr, jsd_clean)."""
    from fra.llama_sleeper import asr_match, format_prompt, generate_with_hooks

    def build_hooks(alpha):
        if pathway == "ov":
            return make_ov_multi_hooks(sae, model, hook_point, layer, features, alpha)
        elif pathway == "qk":
            return make_qk_multi_hooks(sae, layer, hook_point, features, alpha)
        elif pathway == "joint":
            return make_joint_hooks(sae, model, hook_point, layer, features, alpha)
        raise ValueError(pathway)

    per_alpha: list[dict] = []
    for alpha in alphas:
        hooks = build_hooks(alpha)
        jsd_clean_l, asr_n = [], 0
        for i, q in enumerate(eval_prompts):
            dep_prompt = format_prompt(tokenizer, q, with_trigger=True)
            steered_logits = teacher_forced_logits(
                model, tokenizer, dep_prompt, clean_rollouts[i],
                fwd_hooks=hooks,
            ).cpu()
            jsd_clean_l.append(
                float(jsd_bits(steered_logits, ref_clean_logits[i]).mean())
            )
            resp = generate_with_hooks(
                model, tokenizer, dep_prompt, fwd_hooks=hooks,
                max_new_tokens=max_new_tokens,
                temperature=temperature, seed=seed_base + i,
            )
            asr_n += int(asr_match(resp))
        per_alpha.append({
            "alpha": alpha,
            "jsd_clean": float(sum(jsd_clean_l) / len(jsd_clean_l)),
            "asr": float(asr_n / len(eval_prompts)),
        })
        print(f"  α={alpha:+.2f}  ASR={per_alpha[-1]['asr']:.2f}  "
              f"JSD_c={per_alpha[-1]['jsd_clean']:.3f}")

    # Pick best: min ASR, ties → min JSD-clean
    best = min(per_alpha, key=lambda r: (r["asr"], r["jsd_clean"]))
    return {"best": best, "per_alpha": per_alpha, "features": features}


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae-path", type=Path, required=True,
                   help="Path to the ln1.hook_normalized SAE (FRA-native) at the winning layer.")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--hook-point", default="ln1.hook_normalized")
    p.add_argument("--heads", type=int, nargs="+",
                   default=[0, 8, 16, 24],
                   help="Query heads to compute FRA over for attribution. "
                        "Default samples 4 evenly-spaced heads.")
    p.add_argument("--n-features", type=int, default=20,
                   help="Top features per attribution channel.")
    p.add_argument("--n-val", type=int, default=10,
                   help="Validation prompts used for attribution ranking.")
    p.add_argument("--n-eval", type=int, default=10,
                   help="Eval prompts for the (ASR, JSD) sweep.")
    p.add_argument(
        "--alphas", type=float, nargs="+",
        default=[-2.0, -1.0, 0.0, 1.0, 2.0],
        help="Coarse α grid for the matrix sweep.",
    )
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=Path("logs") / "cadenza_phase1"
                                       / "attribution_matrix.json")
    args = p.parse_args()

    from fra.llama_sleeper import (
        SLEEPER_EVAL_PROMPTS, format_prompt, generate_with_hooks,
        load_cadenza_distilled,
    )
    from fra.sae_lens_wrapper import LocalLn1SAE

    need = max(args.n_val, args.n_eval)
    if len(SLEEPER_EVAL_PROMPTS) < need:
        raise ValueError(
            f"Need {need} prompts, only {len(SLEEPER_EVAL_PROMPTS)} defined."
        )
    val_qs = SLEEPER_EVAL_PROMPTS[: args.n_val]
    eval_qs = SLEEPER_EVAL_PROMPTS[: args.n_eval]

    # ── Load ────────────────────────────────────────────────────────────
    t0 = time.time()
    model, tokenizer = load_cadenza_distilled(verbose=False)
    sae = LocalLn1SAE(args.sae_path, layer=args.layer)
    print(f"[load] {time.time()-t0:.1f}s  d_sae={sae.d_sae}")

    # ── Rank features for each attribution channel ───────────────────────
    print("\n=== attribution: OV ===")
    feats_ov = rank_features_diff(
        model, sae, tokenizer, args.layer, args.heads, args.hook_point,
        val_qs, channel="ov", top_k=args.n_features,
    )
    print("\n=== attribution: QK ===")
    feats_qk = rank_features_diff(
        model, sae, tokenizer, args.layer, args.heads, args.hook_point,
        val_qs, channel="qk", top_k=args.n_features,
    )
    print("\n=== attribution: joint ===")
    feats_joint = rank_features_diff(
        model, sae, tokenizer, args.layer, args.heads, args.hook_point,
        val_qs, channel="joint", top_k=args.n_features,
    )
    feature_sets = {"ov": feats_ov, "qk": feats_qk, "joint": feats_joint}

    # ── Pre-compute clean rollouts + reference logits (no hooks) ────────
    print("\n[ref] computing unsteered clean rollouts + ref distributions")
    clean_rollouts: dict[int, list[int]] = {}
    ref_clean_logits: dict[int, torch.Tensor] = {}
    for i, q in enumerate(eval_qs):
        clean_prompt = format_prompt(tokenizer, q, with_trigger=False)
        resp = generate_with_hooks(
            model, tokenizer, clean_prompt, fwd_hooks=[],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, seed=args.seed_base + i,
        )
        clean_rollouts[i] = tokenizer.encode(resp, add_special_tokens=False)[
                                :args.max_new_tokens
                            ] or [tokenizer.eos_token_id]
        ref_clean_logits[i] = teacher_forced_logits(
            model, tokenizer, clean_prompt, clean_rollouts[i],
            fwd_hooks=[],
        ).cpu()

    # ── 3 x 3 cell sweep ─────────────────────────────────────────────────
    matrix: list[dict] = []
    for chan in ("ov", "qk", "joint"):
        for path in ("ov", "qk", "joint"):
            print(f"\n=== cell: attribute={chan}  intervene={path} ===")
            cell = evaluate_cell(
                model, tokenizer, sae,
                args.layer, args.hook_point,
                feature_sets[chan],
                pathway=path,
                alphas=args.alphas,
                eval_prompts=eval_qs,
                clean_rollouts=clean_rollouts,
                ref_clean_logits=ref_clean_logits,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed_base=args.seed_base,
            )
            matrix.append({
                "attribute": chan,
                "intervene": path,
                "n_features": len(feature_sets[chan]),
                "best_alpha": cell["best"]["alpha"],
                "best_asr": cell["best"]["asr"],
                "best_jsd_clean": cell["best"]["jsd_clean"],
                "per_alpha": cell["per_alpha"],
            })

    record = {
        "layer": args.layer,
        "hook_point": args.hook_point,
        "sae_path": str(args.sae_path),
        "alphas": args.alphas,
        "feature_sets": feature_sets,
        "matrix": matrix,
        "meta": {
            "heads": args.heads, "n_val": args.n_val, "n_eval": args.n_eval,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed_base": args.seed_base,
        },
    }
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"\n[save] {out}")
    print("\nMatrix summary (lower-left = better):")
    print(f"  {'attr':>5}  {'inter':>5}  {'α*':>5}  {'ASR':>5}  {'JSD_c':>6}")
    for row in matrix:
        print(f"  {row['attribute']:>5}  {row['intervene']:>5}  "
              f"{row['best_alpha']:+.2f}  {row['best_asr']:5.2f}  "
              f"{row['best_jsd_clean']:6.3f}")


if __name__ == "__main__":
    main()
