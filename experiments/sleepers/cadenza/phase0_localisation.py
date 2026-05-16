"""
Phase 0 localisation — single-cell hookpoint × layer probe (Cadenza Llama-3 8B).

Analog of the TinyStories ``app:sleeper_localisation`` sweep
(``tab:hookpoint_sweep`` in the appendix). One cell per invocation: the
caller passes ``--sae-path``, ``--hook-layer``, ``--hook-point``, and a
held-out split of clean / deployment prompts. The script picks the
single SAE feature and steering coefficient ``α*`` that minimise
sleeper attack-success rate (ASR-16) subject to the clean-prompt
coherence budget ``ΔCE ≤ 0.05`` nats, then reports test-set numbers.

Protocol per cell::

    1. Encode each prompt's last token through the SAE; rank features
       by |mean_dep f − mean_clean f|. Keep the top ``--n-candidates``.
    2. Validation α-sweep: for each (feature, α) pair, generate 16-token
       deployment rollouts (ASR-16) and a teacher-forced clean forward
       pass (ΔCE). Pick the (f*, α*) minimising val ASR-16 subject to
       val ΔCE ≤ 0.05.
    3. Test split: re-measure ASR-16 + ΔCE for the chosen (f*, α*).

The intervention is **additive at the SAE hookpoint**: when the hook
fires, we add ``(α − 1) * f_feat * W_dec[feat]`` to the activation.
At α=0 this fully ablates the feature; α>0 amplifies. The same
intervention rule applies whether the hookpoint is
``ln1.hook_normalized`` (analog of QK→QK in the paper) or
``hook_resid_{mid,post}`` (analog of the conventional residual-stream
additive baseline).

Output JSON has one record per cell; a wrapper script can concatenate
records into the appendix table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


# Repo root on sys.path so `from fra....` works.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ── Intervention hook factory ────────────────────────────────────────────


def make_steering_hooks(
    sae,
    hook_name: str,
    feature_idx: int,
    alpha: float,
):
    """Return a single-hook list that adds ``(α-1) * f * W_dec[feat]`` at ``hook_name``.

    The hook fires on the SAE's input activation, encodes through the
    SAE to recover the per-position feature value ``f[t]``, then modifies
    the activation in-place. Returning the modified tensor keeps
    TransformerLens happy across both ``[batch, seq, d_model]`` and
    ``[batch, seq, n_kv, d_head]`` shapes.
    """
    W_dec = sae.W_dec  # [d_sae, d_in]
    feat_dec = W_dec[feature_idx]  # [d_in]
    scale_minus_1 = float(alpha - 1.0)

    def steer(activation, hook):
        x = activation
        # Some hookpoints (e.g. attn.hook_v) carry an extra head dim;
        # ln1/resid_* are [batch, seq, d_model] — no flatten needed.
        feats = sae.encode(x)
        f_t = feats[..., feature_idx]                       # [..., seq]
        delta = scale_minus_1 * f_t.unsqueeze(-1) * feat_dec.to(x.dtype)
        return x + delta

    return [(hook_name, steer)]


# ── Per-prompt metric helpers ────────────────────────────────────────────


@torch.no_grad()
def teacher_forced_ce(
    model,
    tokenizer,
    formatted_prompt: str,
    fwd_hooks: list,
    *,
    target_continuation: str,
) -> float:
    """Mean per-token CE of ``target_continuation`` under ``formatted_prompt``
    with ``fwd_hooks`` active. Used for the ΔCE coherence budget.
    """
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target_continuation, add_special_tokens=False)
    if not target_ids:
        return 0.0
    full = torch.tensor(prompt_ids + target_ids, device=device).unsqueeze(0)
    logits = model.run_with_hooks(
        full, fwd_hooks=fwd_hooks, reset_hooks_end=True,
    )
    # Predict the i-th target token from position prompt_len + i - 1.
    pre = len(prompt_ids)
    target_logits = logits[0, pre - 1 : pre - 1 + len(target_ids), :].float()
    target_t = torch.tensor(target_ids, device=device)
    ce = F.cross_entropy(target_logits, target_t, reduction="mean").item()
    return ce


# ── Main per-cell pipeline ───────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae-path", type=Path, required=True,
                   help="Directory containing the trained SAE checkpoint "
                        "(sae-lens save_final_checkpoint output).")
    p.add_argument("--hook-layer", type=int, required=True)
    p.add_argument("--hook-point", required=True,
                   help="Hookpoint suffix (e.g. ln1.hook_normalized).")
    p.add_argument("--n-candidates", type=int, default=20,
                   help="Top-K candidate features by |Δ activation|.")
    p.add_argument(
        "--alphas", type=float, nargs="+",
        default=[-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        help="Steering coefficients to sweep.",
    )
    p.add_argument("--n-val", type=int, default=10,
                   help="Validation prompts per split (used for f*/α* selection).")
    p.add_argument("--n-test", type=int, default=10,
                   help="Test prompts per split (used for reporting).")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--ce-budget", type=float, default=0.05,
                   help="Clean-prompt ΔCE budget (nats). Default matches paper.")
    p.add_argument("--out", type=Path,
                   default=Path("logs") / "cadenza_localisation"
                                       / "phase0_localisation.json")
    args = p.parse_args()

    from fra.llama_sleeper import (
        SLEEPER_EVAL_PROMPTS,
        SLEEPER_PHRASE_REGEX,
        asr_match,
        format_prompt,
        generate_with_hooks,
        load_cadenza_distilled,
    )
    from fra.sae_lens_wrapper import LocalLn1SAE

    hook_name = f"blocks.{args.hook_layer}.{args.hook_point}"

    if len(SLEEPER_EVAL_PROMPTS) < args.n_val + args.n_test:
        raise ValueError(
            f"Need {args.n_val + args.n_test} prompts but only "
            f"{len(SLEEPER_EVAL_PROMPTS)} are defined."
        )
    val_qs = SLEEPER_EVAL_PROMPTS[: args.n_val]
    test_qs = SLEEPER_EVAL_PROMPTS[args.n_val : args.n_val + args.n_test]

    # ── 1. Load model + SAE ─────────────────────────────────────────────
    print(f"\n=== Localisation cell ===")
    print(f"  sae_path   : {args.sae_path}")
    print(f"  hook_layer : {args.hook_layer}")
    print(f"  hook_point : {args.hook_point}")
    print(f"  hook_name  : {hook_name}")

    t0 = time.time()
    model, tokenizer = load_cadenza_distilled(verbose=False)
    sae = LocalLn1SAE(args.sae_path, layer=args.hook_layer)
    print(f"[load] {time.time()-t0:.1f}s  d_sae={sae.d_sae}  d_in={sae.d_in}")

    # ── 2. Pick candidate features by |Δ mean activation| at last position ─
    print("[rank] computing per-feature Δactivation on val split...")
    dev = next(model.parameters()).device
    feat_sum_dep = torch.zeros(sae.d_sae, device=dev)
    feat_sum_clean = torch.zeros(sae.d_sae, device=dev)
    n_dep = n_clean = 0
    for q in val_qs:
        for trig in (True, False):
            ids = torch.tensor(
                tokenizer.encode(
                    format_prompt(tokenizer, q, with_trigger=trig),
                    add_special_tokens=False,
                ),
                device=dev,
            ).unsqueeze(0)
            with torch.no_grad():
                _, cache = model.run_with_cache(ids, names_filter=[hook_name])
            x = cache[hook_name][0]                # [seq, d_in]
            with torch.no_grad():
                feats = sae.encode(x)              # [seq, d_sae]
            last = feats[-1].float()
            if trig:
                feat_sum_dep += last; n_dep += 1
            else:
                feat_sum_clean += last; n_clean += 1

    mean_dep = feat_sum_dep / max(n_dep, 1)
    mean_clean = feat_sum_clean / max(n_clean, 1)
    delta_abs = (mean_dep - mean_clean).abs()
    top_vals, top_idx = torch.topk(delta_abs, args.n_candidates)
    candidates = [(int(top_idx[i].item()), float(top_vals[i].item()))
                  for i in range(args.n_candidates)]
    print(f"[rank] top-{args.n_candidates} candidates (feat_idx, |Δact|):")
    for f, d in candidates[:8]:
        print(f"  {f:>6} : Δ={d:.3f}  "
              f"mean_dep={mean_dep[f].item():.3f}  "
              f"mean_clean={mean_clean[f].item():.3f}")

    # ── 3. Validation α-sweep ───────────────────────────────────────────
    # For ΔCE we teacher-force against the unsteered model's clean
    # continuation, so the budget measures "how much steering corrupts
    # the model's own clean-prompt behaviour".
    print("\n[ref] computing unsteered baseline rollouts + clean CE targets…")
    ref_clean_targets: list[str] = []
    for i, q in enumerate(val_qs):
        seed = args.seed_base + i
        prompt = format_prompt(tokenizer, q, with_trigger=False)
        cont = generate_with_hooks(
            model, tokenizer, prompt, fwd_hooks=[],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, seed=seed,
        )
        ref_clean_targets.append(cont)
    base_clean_ce = []
    for q, target in zip(val_qs, ref_clean_targets):
        prompt = format_prompt(tokenizer, q, with_trigger=False)
        base_clean_ce.append(teacher_forced_ce(
            model, tokenizer, prompt, fwd_hooks=[], target_continuation=target,
        ))
    base_ce_mean = sum(base_clean_ce) / len(base_clean_ce)
    print(f"[ref] mean unsteered clean CE on targets: {base_ce_mean:.4f}")

    print("\n[sweep] validation α-sweep over candidates...")
    sweep_records: list[dict] = []
    for f_idx, _delta in candidates:
        for alpha in args.alphas:
            hooks = make_steering_hooks(sae, hook_name, f_idx, alpha)
            # ΔCE on clean prompts
            ces = []
            for q, target in zip(val_qs, ref_clean_targets):
                prompt = format_prompt(tokenizer, q, with_trigger=False)
                ces.append(teacher_forced_ce(
                    model, tokenizer, prompt, fwd_hooks=hooks,
                    target_continuation=target,
                ))
            delta_ce = (sum(ces) / len(ces)) - base_ce_mean
            # ASR-16 on deployment prompts
            n_fire = 0
            for i, q in enumerate(val_qs):
                prompt = format_prompt(tokenizer, q, with_trigger=True)
                resp = generate_with_hooks(
                    model, tokenizer, prompt, fwd_hooks=hooks,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, seed=args.seed_base + i,
                )
                n_fire += int(asr_match(resp))
            asr = n_fire / len(val_qs)
            sweep_records.append({
                "feature": f_idx, "alpha": alpha,
                "asr": asr, "delta_ce": delta_ce,
            })
            print(f"  feat={f_idx:>6} α={alpha:+.2f}  ASR={asr:.2f}  ΔCE={delta_ce:+.4f}")

    # ── 4. Pick (f*, α*) — min ASR subject to ΔCE ≤ budget ──────────────
    candidates_in_budget = [
        r for r in sweep_records if r["delta_ce"] <= args.ce_budget
    ]
    if not candidates_in_budget:
        print(f"\n[select] NO (feat, α) within ΔCE ≤ {args.ce_budget} budget; "
              "falling back to min |ΔCE| across all candidates.")
        candidates_in_budget = sweep_records
        winner = min(candidates_in_budget,
                     key=lambda r: (r["asr"], abs(r["delta_ce"])))
    else:
        winner = min(candidates_in_budget,
                     key=lambda r: (r["asr"], r["delta_ce"]))
    print(f"\n[select] winner: feat={winner['feature']}  α={winner['alpha']:+.2f}  "
          f"val_ASR={winner['asr']:.2f}  val_ΔCE={winner['delta_ce']:+.4f}")

    # ── 5. Test split: re-measure with the chosen (f*, α*) ──────────────
    print("\n[test] measuring on held-out test split...")
    hooks = make_steering_hooks(sae, hook_name, winner["feature"], winner["alpha"])

    test_clean_targets = []
    for i, q in enumerate(test_qs):
        prompt = format_prompt(tokenizer, q, with_trigger=False)
        cont = generate_with_hooks(
            model, tokenizer, prompt, fwd_hooks=[],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed_base + 1000 + i,
        )
        test_clean_targets.append(cont)

    test_base_ces = []
    test_steered_ces = []
    for q, target in zip(test_qs, test_clean_targets):
        prompt = format_prompt(tokenizer, q, with_trigger=False)
        test_base_ces.append(teacher_forced_ce(
            model, tokenizer, prompt, fwd_hooks=[],
            target_continuation=target,
        ))
        test_steered_ces.append(teacher_forced_ce(
            model, tokenizer, prompt, fwd_hooks=hooks,
            target_continuation=target,
        ))
    test_delta_ce = (sum(test_steered_ces) - sum(test_base_ces)) / len(test_qs)

    test_fire = 0
    for i, q in enumerate(test_qs):
        prompt = format_prompt(tokenizer, q, with_trigger=True)
        resp = generate_with_hooks(
            model, tokenizer, prompt, fwd_hooks=hooks,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed_base + 1000 + i,
        )
        test_fire += int(asr_match(resp))
    test_asr = test_fire / len(test_qs)
    print(f"[test] feat={winner['feature']}  α={winner['alpha']:+.2f}  "
          f"test_ASR={test_asr:.2f}  test_ΔCE={test_delta_ce:+.4f}")

    # ── 6. Persist ───────────────────────────────────────────────────────
    record = {
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
            "val_asr": winner["asr"],
            "val_delta_ce": winner["delta_ce"],
            "ce_budget": args.ce_budget,
        },
        "test": {
            "asr": test_asr,
            "delta_ce": test_delta_ce,
            "n": len(test_qs),
        },
        "candidates": candidates,
        "sweep": sweep_records,
        "config": {
            "alphas": args.alphas,
            "n_candidates": args.n_candidates,
            "n_val": args.n_val,
            "n_test": args.n_test,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed_base": args.seed_base,
        },
    }
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
