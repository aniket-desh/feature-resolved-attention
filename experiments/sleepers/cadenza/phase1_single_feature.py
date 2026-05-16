"""
Phase 1 — Single FRA-OV feature vs. conventional resid-mid additive
                  on Cadenza Llama-3 8B.

Analog of ``fig:tinysleeper_pareto`` / ``combined_50k.pdf``. At a chosen
winning layer (from phase 0 localisation), compare two single-vector
interventions side-by-side:

  ``ov_single_50k``      single FRA-OV feature steered via attn.hook_v
                         (uses the ln1.hook_normalized SAE)
  ``conventional_50k``   single SAE feature added at hook_resid_mid
                         (uses the resid_mid SAE at the same layer)

For each method and α, four metrics are reported per prompt:

  ``jsd_clean``           JSD between steered model's next-token distribution
                          (deployment prompt + clean rollout, teacher-forced
                          with hooks) and the unsteered model's distribution
                          on the deployment-stripped prompt + clean rollout.
                          Lower = closer to clean.  ↓ better.
  ``jsd_pois``            JSD between steered (deployment + hooks) and
                          unsteered sleeper (deployment, no hooks). Higher =
                          farther from sleeper. ↑ better.
  ``n_exact_match_clean`` Number of prompts whose sampled 16-token rollout
                          under steering matches the unsteered clean rollout
                          token-for-token (same seed). ↑ better.
  ``asr``                 ASR-16: fraction of sampled rollouts containing the
                          canonical sleeper phrase. ↓ better.

Output JSON schema matches ``experiments/sleepers/scripts/plot_combined_50k.py``
exactly so the existing plotter works unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ── JSD helper ───────────────────────────────────────────────────────────


def jsd_bits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Per-position Jensen-Shannon divergence between two logit tensors.

    Returns a 1-D tensor of length ``logits_p.shape[0]``, in bits.
    Both inputs must have shape ``[n_pos, vocab]`` and be aligned position-by-position.
    """
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp(min=1e-12))
    kl_pm = (p * (log_p - log_m)).sum(-1)
    kl_qm = (q * (log_q - log_m)).sum(-1)
    return 0.5 * (kl_pm + kl_qm) / math.log(2)


# ── Reference computation ────────────────────────────────────────────────


@torch.no_grad()
def teacher_forced_logits(
    model, tokenizer, formatted_prompt: str, target_ids: List[int],
    *, fwd_hooks: list,
) -> torch.Tensor:
    """Forward pass on ``formatted_prompt + target_ids``; return logits at the
    positions that *predict* each target id. Shape ``[len(target_ids), vocab]``.
    """
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    full = torch.tensor(prompt_ids + target_ids, device=device).unsqueeze(0)
    logits = model.run_with_hooks(
        full, fwd_hooks=fwd_hooks, reset_hooks_end=True,
    )
    pre = len(prompt_ids)
    return logits[0, pre - 1 : pre - 1 + len(target_ids), :]


# ── Intervention hook factories ──────────────────────────────────────────


def make_ov_hooks(sae, model, hook_point: str, layer: int, head_or_all,
                  feature_idx: int, alpha: float):
    """OV steering via attn.hook_v on a single feature.

    Thin wrapper around fra.ov_steering's run-time hook construction logic,
    extracted here so we can attach hooks via ``run_with_hooks`` (rather than
    the dedicated runner). When ``head_or_all=='all'``, the OV delta is
    applied to *every* KV head at the layer; otherwise to one head index.
    """
    from fra.core.helpers import get_W_V

    device = next(model.parameters()).device
    W_dec = sae.W_dec.float()                            # [d_sae, d_model]
    feat_dec = W_dec[feature_idx]                        # [d_model]
    scale_m1 = float(alpha - 1.0)
    sae_hook = f"blocks.{layer}.{hook_point}"
    v_hook = f"blocks.{layer}.attn.hook_v"
    n_q = model.cfg.n_heads
    n_kv = getattr(model.cfg, "n_key_value_heads", None) or n_q
    # Pre-compute (d_model → d_head) per-KV-head projections of the feature.
    feat_v_per_kv = torch.stack(
        [feat_dec @ get_W_V(model, layer, h * n_q // n_kv).float()
         for h in range(n_kv)],
        dim=0,
    )                                                    # [n_kv, d_head]

    cached: dict = {}

    def capture(activation, hook):
        x = activation[0] if activation.dim() == 3 else activation
        if x.dim() == 3:
            x = x.flatten(-2, -1)
        feats = sae.encode(x)
        cached["f"] = feats[..., feature_idx].float()    # [seq] or [batch, seq]
        return activation

    def steer_v(v, hook):
        f = cached.get("f")
        if f is None:
            return v
        # v: [batch, seq, n_kv, d_head]
        sl = min(f.shape[-1], v.shape[1])
        if head_or_all == "all":
            delta = scale_m1 * f[..., :sl, None, None] * feat_v_per_kv.to(v.dtype)
            v[..., :sl, :, :] += delta
        else:
            kv_idx = int(head_or_all) * n_kv // n_q
            delta = (scale_m1 * f[..., :sl].unsqueeze(-1)
                     * feat_v_per_kv[kv_idx].to(v.dtype))
            v[..., :sl, kv_idx, :] += delta
        return v

    return [(sae_hook, capture), (v_hook, steer_v)]


def make_resid_additive_hooks(sae, layer: int, hook_point: str,
                              feature_idx: int, alpha: float):
    """Conventional residual-stream additive steering.

    Same shape as :func:`make_steering_hooks` in phase0_localisation but
    factored here for symmetry with the OV path. Uses the resid-hookpoint
    SAE's feature direction; adds ``(α − 1) * f * W_dec[feat]`` directly to
    the residual stream when the hook fires.
    """
    W_dec = sae.W_dec.float()
    feat_dec = W_dec[feature_idx]
    scale_m1 = float(alpha - 1.0)
    hook_name = f"blocks.{layer}.{hook_point}"

    def steer(activation, hook):
        x = activation
        feats = sae.encode(x)
        f_t = feats[..., feature_idx]
        delta = scale_m1 * f_t.unsqueeze(-1) * feat_dec.to(x.dtype)
        return x + delta

    return [(hook_name, steer)]


# ── Feature selection ────────────────────────────────────────────────────


@torch.no_grad()
def pick_best_feature(
    model, tokenizer, sae, hook_factory,
    val_prompts: List[str], *,
    n_candidates: int,
    alphas_for_selection: List[float],
    max_new_tokens: int = 16,
    temperature: float = 1.0,
    seed_base: int = 42,
) -> Dict[str, Any]:
    """Pick (feature*, α*) that minimises val ASR via the same Δ-activation
    candidate scoring used by phase0_localisation. Selection-stage decoding
    is greedy (deterministic winner)."""
    from fra.llama_sleeper import (
        SLEEPER_PHRASE_REGEX, asr_match, format_prompt, generate_with_hooks,
    )

    device = next(model.parameters()).device
    # Δ activation on the SAE's hookpoint last position
    hook_name = hook_factory.sae_hook
    f_dep = torch.zeros(sae.d_sae, device=device)
    f_clean = torch.zeros(sae.d_sae, device=device)
    n_d = n_c = 0
    for q in val_prompts:
        for trig in (True, False):
            ids = torch.tensor(tokenizer.encode(
                format_prompt(tokenizer, q, with_trigger=trig),
                add_special_tokens=False,
            ), device=device).unsqueeze(0)
            _, cache = model.run_with_cache(ids, names_filter=[hook_name])
            x = cache[hook_name][0]
            feats = sae.encode(x)[-1].float()
            if trig:
                f_dep += feats; n_d += 1
            else:
                f_clean += feats; n_c += 1
    delta = ((f_dep / n_d) - (f_clean / n_c)).abs()
    top_idx = torch.topk(delta, n_candidates).indices.tolist()
    print(f"[select] top-{n_candidates} candidates: {top_idx[:8]}{'...' if n_candidates>8 else ''}")

    best = {"feature": top_idx[0], "alpha": 0.0, "asr": 1.0}
    for f_idx in top_idx:
        for a in alphas_for_selection:
            hooks = hook_factory.build(f_idx, a)
            fires = 0
            for i, q in enumerate(val_prompts):
                resp = generate_with_hooks(
                    model, tokenizer,
                    format_prompt(tokenizer, q, with_trigger=True),
                    fwd_hooks=hooks,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,                # greedy for selection
                    seed=seed_base + i,
                )
                fires += int(asr_match(resp))
            asr = fires / len(val_prompts)
            if asr < best["asr"]:
                best = {"feature": int(f_idx), "alpha": float(a), "asr": asr}
                if asr == 0.0:
                    print(f"[select] early-stop: feat={f_idx} α={a:+.2f} → val ASR=0")
                    return best
    return best


class _OVFactory:
    """Bind sae+model so :func:`pick_best_feature` can build hooks
    via the shared ``hook_factory.build(feat, α)`` interface."""
    def __init__(self, sae, model, hook_point, layer, head_or_all):
        self.sae = sae
        self.model = model
        self.layer = layer
        self.hook_point = hook_point
        self.head_or_all = head_or_all
        self.sae_hook = f"blocks.{layer}.{hook_point}"
    def build(self, feat, alpha):
        return make_ov_hooks(
            self.sae, self.model, self.hook_point, self.layer,
            self.head_or_all, feat, alpha,
        )


class _ResidFactory:
    def __init__(self, sae, layer, hook_point):
        self.sae = sae
        self.layer = layer
        self.hook_point = hook_point
        self.sae_hook = f"blocks.{layer}.{hook_point}"
    def build(self, feat, alpha):
        return make_resid_additive_hooks(
            self.sae, self.layer, self.hook_point, feat, alpha,
        )


# ── Per-α metric loop ────────────────────────────────────────────────────


@torch.no_grad()
def sweep_alpha(
    model, tokenizer, factory, feature_idx: int,
    eval_prompts: List[str],
    *,
    alphas: List[float],
    max_new_tokens: int = 16,
    temperature: float = 1.0,
    seed_base: int = 42,
) -> Dict[str, Any]:
    """Per-α metrics for a single (factory, feature). Returns the
    ``per_alpha`` dict slot for one config."""
    from fra.llama_sleeper import (
        SLEEPER_PHRASE_REGEX, asr_match, format_prompt, generate_with_hooks,
    )

    # 1. Build clean rollouts once (no hooks, dep-stripped). These are the
    #    teacher-forcing target and the exact-match reference.
    clean_rollout_ids: dict[int, list[int]] = {}
    for i, q in enumerate(eval_prompts):
        resp = generate_with_hooks(
            model, tokenizer,
            format_prompt(tokenizer, q, with_trigger=False),
            fwd_hooks=[],
            max_new_tokens=max_new_tokens,
            temperature=temperature, seed=seed_base + i,
        )
        clean_rollout_ids[i] = tokenizer.encode(resp, add_special_tokens=False)[
                                  :max_new_tokens
                              ] or [tokenizer.eos_token_id]

    # 2. Per-prompt reference distributions (unsteered).
    ref_clean: dict[int, torch.Tensor] = {}     # P(clean | dep-stripped)
    ref_pois: dict[int, torch.Tensor] = {}      # P(sleeper | dep)
    for i, q in enumerate(eval_prompts):
        ref_clean[i] = teacher_forced_logits(
            model, tokenizer,
            format_prompt(tokenizer, q, with_trigger=False),
            clean_rollout_ids[i],
            fwd_hooks=[],
        ).cpu()
        ref_pois[i] = teacher_forced_logits(
            model, tokenizer,
            format_prompt(tokenizer, q, with_trigger=True),
            clean_rollout_ids[i],
            fwd_hooks=[],
        ).cpu()

    per_alpha: dict[str, dict[str, list[float] | float | int]] = {}
    for alpha in alphas:
        hooks = factory.build(feature_idx, alpha)
        jsd_clean_l, jsd_pois_l, asr_n, exact_n = [], [], 0, 0
        for i, q in enumerate(eval_prompts):
            dep_prompt = format_prompt(tokenizer, q, with_trigger=True)
            target_ids = clean_rollout_ids[i]

            # Distribution-level
            p_steered = teacher_forced_logits(
                model, tokenizer, dep_prompt, target_ids, fwd_hooks=hooks,
            ).cpu()
            jsd_clean_l.append(float(jsd_bits(p_steered, ref_clean[i]).mean()))
            jsd_pois_l.append(float(jsd_bits(p_steered, ref_pois[i]).mean()))

            # Rollout-level
            steered = generate_with_hooks(
                model, tokenizer, dep_prompt, fwd_hooks=hooks,
                max_new_tokens=max_new_tokens,
                temperature=temperature, seed=seed_base + i,
            )
            steered_ids = tokenizer.encode(steered, add_special_tokens=False)[
                              :max_new_tokens
                          ]
            exact_n += int(steered_ids == target_ids)
            asr_n += int(asr_match(steered))

        per_alpha[str(alpha)] = {
            "jsd_clean": float(sum(jsd_clean_l) / len(jsd_clean_l)),
            "jsd_pois":  float(sum(jsd_pois_l)  / len(jsd_pois_l)),
            "n_exact_match_clean": int(exact_n),
            "asr": float(asr_n / len(eval_prompts)),
        }
        print(
            f"  α={alpha:+.2f}  ASR={per_alpha[str(alpha)]['asr']:.2f}  "
            f"JSD_c={per_alpha[str(alpha)]['jsd_clean']:.3f}  "
            f"JSD_p={per_alpha[str(alpha)]['jsd_pois']:.3f}  "
            f"exact={exact_n}/{len(eval_prompts)}"
        )
    return per_alpha


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fra-sae-path", type=Path, required=True,
                   help="Path to the ln1.hook_normalized SAE (FRA-native).")
    p.add_argument("--conv-sae-path", type=Path, required=True,
                   help="Path to the hook_resid_mid SAE (conventional baseline).")
    p.add_argument("--layer", type=int, required=True,
                   help="Layer shared by both SAEs (the localisation winner).")
    p.add_argument("--ov-head", default="all",
                   help="Either an int head index or 'all' (all KV heads).")
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument("--n-val", type=int, default=8,
                   help="Validation-split prompts used to pick (feature, α).")
    p.add_argument(
        "--alphas", type=float, nargs="+",
        default=[-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
    )
    p.add_argument("--alphas-for-selection", type=float, nargs="+",
                   default=[-2.0, 2.0])
    p.add_argument("--n-candidates", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=Path("logs") / "cadenza_phase1"
                                       / "single_feature_metrics.json")
    args = p.parse_args()

    from fra.llama_sleeper import SLEEPER_EVAL_PROMPTS, load_cadenza_distilled
    from fra.sae_lens_wrapper import LocalLn1SAE

    if len(SLEEPER_EVAL_PROMPTS) < args.n_prompts:
        raise ValueError(
            f"Need {args.n_prompts} eval prompts, only "
            f"{len(SLEEPER_EVAL_PROMPTS)} are defined in fra.llama_sleeper."
        )

    # ── Load model + both SAEs ──────────────────────────────────────────
    t0 = time.time()
    model, tokenizer = load_cadenza_distilled(verbose=False)
    fra_sae = LocalLn1SAE(args.fra_sae_path, layer=args.layer)
    conv_sae = LocalLn1SAE(args.conv_sae_path, layer=args.layer)
    print(f"[load] model + 2 SAEs in {time.time()-t0:.1f}s")

    val_qs = SLEEPER_EVAL_PROMPTS[: args.n_val]
    eval_qs = SLEEPER_EVAL_PROMPTS[: args.n_prompts]

    # ── Selection: pick best (feature, α) per method ────────────────────
    print("\n=== selecting feature for ov_single ===")
    ov_factory = _OVFactory(fra_sae, model, "ln1.hook_normalized",
                            args.layer, args.ov_head)
    ov_pick = pick_best_feature(
        model, tokenizer, fra_sae, ov_factory,
        val_qs, n_candidates=args.n_candidates,
        alphas_for_selection=args.alphas_for_selection,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, seed_base=args.seed_base,
    )
    print(f"  → OV winner: feat={ov_pick['feature']} α*={ov_pick['alpha']:+.2f}"
          f"  val ASR={ov_pick['asr']:.2f}")

    print("\n=== selecting feature for conventional resid-mid ===")
    conv_factory = _ResidFactory(conv_sae, args.layer, "hook_resid_mid")
    conv_pick = pick_best_feature(
        model, tokenizer, conv_sae, conv_factory,
        val_qs, n_candidates=args.n_candidates,
        alphas_for_selection=args.alphas_for_selection,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, seed_base=args.seed_base,
    )
    print(f"  → conv winner: feat={conv_pick['feature']} α*={conv_pick['alpha']:+.2f}"
          f"  val ASR={conv_pick['asr']:.2f}")

    # ── α-sweep on full eval split for both methods ─────────────────────
    print("\n=== sweep: ov_single ===")
    ov_per_alpha = sweep_alpha(
        model, tokenizer, ov_factory, ov_pick["feature"],
        eval_qs, alphas=args.alphas,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, seed_base=args.seed_base,
    )

    print("\n=== sweep: conventional ===")
    conv_per_alpha = sweep_alpha(
        model, tokenizer, conv_factory, conv_pick["feature"],
        eval_qs, alphas=args.alphas,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, seed_base=args.seed_base,
    )

    record = {
        "alphas": args.alphas,
        "configs": {
            "ov_single_50k":  {
                "feature": ov_pick["feature"], "alpha_select": ov_pick["alpha"],
                "sae_path": str(args.fra_sae_path),
                "per_alpha": ov_per_alpha,
            },
            "conventional_50k": {
                "feature": conv_pick["feature"], "alpha_select": conv_pick["alpha"],
                "sae_path": str(args.conv_sae_path),
                "per_alpha": conv_per_alpha,
            },
        },
        "meta": {
            "layer": args.layer,
            "n_prompts": args.n_prompts,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed_base": args.seed_base,
        },
    }
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"\n[save] {out}")
    print(
        "→ feed into experiments/sleepers/scripts/plot_combined_50k.py "
        "with --input ${out} --output figures/cadenza_combined_50k"
    )


if __name__ == "__main__":
    main()
