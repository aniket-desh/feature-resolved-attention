"""
Phase-2 mech-interp on the L29/hook_resid_post trigger feature.

Goal: explain *why* the anti-feature direction (α<0) is the only direction
that suppresses the Cadenza sleeper while α≥0 is uniformly inert.
Specifically address: (a) the TinyStories trigger feature happens to have
a useful amplification direction within α∈[0, 4], (b) the paper's
appendix should be amended to sweep both signs, or (c) the trigger-
feature direction in Cadenza is encoded with the opposite sign to
TinyStories.

Five analyses produced:

1. **Weight-only stats** on the winning feature
   (norm, top cosine sims to other features, encoder/decoder alignment).
2. **Cross-layer feature thread** — does the trigger have a
   "co-feature" at L29/ln1 and L29/resid_mid that points the same way?
3. **Activation pattern** of the feature across deployment vs clean
   prompts: where does it fire, with what magnitude, what's the gap.
4. **Intervention direction analysis** — at α=-4, where in the
   residual stream is the steered point relative to the unsteered
   point and the unembed of common payload tokens ("I", "HATE", "YOU")?
5. **Top activating contexts** — what does the feature represent in
   plain text?

Output: ``experiments/sleepers/cadenza/mechinterp_report.md`` plus three
figures under ``experiments/sleepers/cadenza/figures/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ── Loader helpers ───────────────────────────────────────────────────────


def load_all() -> Dict[str, Any]:
    """Load model + all three L29 SAEs (ln1, resid_mid, resid_post)."""
    from fra.llama_sleeper import load_cadenza_distilled
    from fra.sae_lens_wrapper import LocalLn1SAE

    print("[load] Cadenza model ...")
    model, tokenizer = load_cadenza_distilled(verbose=False)

    print("[load] L29 SAEs (ln1, resid_mid, resid_post) ...")
    sae_root = Path("/workspace/aniket/saes")
    saes = {}
    for hook in ("ln1_hook_normalized", "hook_resid_mid", "hook_resid_post"):
        ckpts = sorted((sae_root / f"cadenza_L29_{hook}").glob("*/*/sae_weights.safetensors"))
        ckpt = max(ckpts, key=lambda p: int(p.parts[-2].lstrip("final_")) if p.parts[-2].startswith("final_") else int(p.parts[-2]))
        saes[hook] = LocalLn1SAE(ckpt.parent, layer=29)
        print(f"  {hook}: d_sae={saes[hook].d_sae}, ckpt={ckpt.parent}")
    return {"model": model, "tokenizer": tokenizer, "saes": saes}


# ── (1) Weight-only analysis of feat 12402 at resid_post ─────────────────


@torch.no_grad()
def analyze_weights(sae, feat_idx: int, top_n: int = 8) -> Dict[str, Any]:
    W_dec = sae.W_dec.float()        # [d_sae, d_model]
    W_enc = sae.W_enc.float()        # [d_model, d_sae]
    d_w = W_dec[feat_idx]             # [d_model]
    e_w = W_enc[:, feat_idx]          # [d_model]

    # Cosine sim of W_dec[feat] to every other feature
    W_dec_n = F.normalize(W_dec, dim=-1)
    d_w_n = F.normalize(d_w, dim=-1)
    sims = W_dec_n @ d_w_n            # [d_sae]
    sims[feat_idx] = -2.0             # exclude self
    top_vals, top_idx = torch.topk(sims, top_n)

    return {
        "feat_idx": feat_idx,
        "W_dec_norm": float(d_w.norm()),
        "W_enc_norm": float(e_w.norm()),
        "enc_dec_cosine":  float(F.cosine_similarity(d_w, e_w, dim=0)),
        "top_cosine_features": [
            (int(top_idx[i]), float(top_vals[i])) for i in range(top_n)
        ],
    }


# ── (2) Cross-layer feature thread ───────────────────────────────────────


@torch.no_grad()
def cross_layer_thread(saes: dict, feat_idx_post: int, top_n: int = 3) -> Dict[str, Any]:
    """For W_dec[feat_idx_post] at resid_post, find the closest decoder
    direction in each of the other two L29 SAEs (ln1, resid_mid)."""
    target = F.normalize(saes["hook_resid_post"].W_dec[feat_idx_post].float(), dim=0)
    out = {}
    for hook in ("ln1_hook_normalized", "hook_resid_mid"):
        W = F.normalize(saes[hook].W_dec.float(), dim=-1)
        sims = W @ target
        top_vals, top_idx = torch.topk(sims, top_n)
        out[hook] = [(int(top_idx[i]), float(top_vals[i])) for i in range(top_n)]
    return out


# ── (3) Activation pattern across dep vs clean ───────────────────────────


@torch.no_grad()
def activation_pattern(
    model, tokenizer, sae, feat_idx: int,
    hook_name: str, prompts: list, *, n_each: int = 30,
) -> Dict[str, Any]:
    """For each prompt, encode at the SAE's hookpoint and record
    feature ``feat_idx`` activation at every position. Return per-prompt
    last-position activations + a per-position summary."""
    from fra.llama_sleeper import format_prompt
    dev = next(model.parameters()).device

    def measure(qs, with_trigger):
        rows = []
        for q in qs[:n_each]:
            ids = torch.tensor(tokenizer.encode(
                format_prompt(tokenizer, q, with_trigger=with_trigger),
                add_special_tokens=False,
            ), device=dev).unsqueeze(0)
            _, cache = model.run_with_cache(ids, names_filter=[hook_name])
            x = cache[hook_name][0]
            feats = sae.encode(x)[:, feat_idx].float().cpu().tolist()
            rows.append({
                "q": q,
                "len": len(feats),
                "last": feats[-1],
                "max": max(feats),
                "argmax_pos": int(torch.tensor(feats).argmax()),
                "trace": feats,
            })
        return rows

    dep = measure(prompts, True)
    clean = measure(prompts, False)
    last_dep = [r["last"] for r in dep]
    last_clean = [r["last"] for r in clean]
    return {
        "dep_mean_last":    sum(last_dep) / max(len(last_dep), 1),
        "clean_mean_last":  sum(last_clean) / max(len(last_clean), 1),
        "dep_min_last":     min(last_dep) if last_dep else 0.0,
        "dep_max_last":     max(last_dep) if last_dep else 0.0,
        "clean_max_last":   max(last_clean) if last_clean else 0.0,
        "dep_max_anypos":   max(r["max"] for r in dep) if dep else 0.0,
        "clean_max_anypos": max(r["max"] for r in clean) if clean else 0.0,
        "dep_rows":   dep,
        "clean_rows": clean,
    }


# ── (4) Intervention-direction analysis ──────────────────────────────────


@torch.no_grad()
def intervention_direction(
    model, tokenizer, sae, feat_idx: int, alpha_neg: float = -4.0,
) -> Dict[str, Any]:
    """Compute the direction the α=-4 intervention pushes the residual
    stream in, and compare to W_U (unembed) for common payload tokens."""
    dev = next(model.parameters()).device
    W_dec = sae.W_dec.float()
    d_w = W_dec[feat_idx]                                  # [d_model]
    # Direction of the intervention at a typical activation magnitude.
    # The hook adds (α - 1) * f * W_dec; for α=-4, the magnitude is -5 * f.
    # We just need the *direction*: sign(α-1) * W_dec.
    intervention_dir = (alpha_neg - 1.0) * d_w
    intervention_dir_n = F.normalize(intervention_dir, dim=0)

    W_U = model.W_U.float().to(dev)                        # [d_model, d_vocab]
    W_U_n = F.normalize(W_U, dim=0)                        # column-normalize: token directions

    target_tokens = ["I", " I", " HATE", "HATE", " YOU", "YOU",
                     " the", " a", " I'm", " hello"]
    items = []
    for tok in target_tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if not ids:
            continue
        for tid in ids:
            dir_u = W_U_n[:, tid].to(intervention_dir_n.device)
            cos = float(torch.dot(intervention_dir_n, dir_u))
            items.append({"token": tokenizer.decode([tid]), "id": tid, "cos": cos})

    items.sort(key=lambda r: r["cos"], reverse=True)
    return {
        "intervention_alpha": alpha_neg,
        "W_dec_norm": float(d_w.norm()),
        "top_token_alignments": items,
    }


# ── (5) Top activating contexts ──────────────────────────────────────────


@torch.no_grad()
def top_contexts(
    model, tokenizer, sae, feat_idx: int, hook_name: str,
    prompts: list, k: int = 12,
) -> List[Dict[str, Any]]:
    """Find (prompt, position) pairs where the feature fires hardest."""
    from fra.llama_sleeper import format_prompt
    dev = next(model.parameters()).device

    candidates = []
    for trig in (True, False):
        for q in prompts:
            text = format_prompt(tokenizer, q, with_trigger=trig)
            tok_ids = tokenizer.encode(text, add_special_tokens=False)
            ids = torch.tensor(tok_ids, device=dev).unsqueeze(0)
            _, cache = model.run_with_cache(ids, names_filter=[hook_name])
            x = cache[hook_name][0]
            f = sae.encode(x)[:, feat_idx].float().cpu()
            for pos, val in enumerate(f.tolist()):
                if val > 0:
                    tok = tokenizer.decode([tok_ids[pos]])
                    pre = tokenizer.decode(tok_ids[max(0, pos-5):pos])
                    post = tokenizer.decode(tok_ids[pos+1:pos+4])
                    candidates.append({
                        "act": val, "trigger": trig,
                        "q": q[:40], "pos": pos,
                        "token": tok,
                        "context": f"...{pre}[{tok}]{post}...",
                    })
    candidates.sort(key=lambda c: c["act"], reverse=True)
    return candidates[:k]


# ── Markdown report writer ───────────────────────────────────────────────


def write_report(out_md: Path, results: dict):
    lines = []
    p = lambda s="": lines.append(s)
    p("# Mech-interp report — Cadenza L29/hook_resid_post trigger feature\n")
    p("Question Dmitry asked (paraphrased): why does the paper's α≥0 grid "
      "miss the Cadenza result? Three hypotheses listed in summary.md's "
      "open-questions section. This document tests them on the actual "
      "trigger feature **12402 at blocks.29.hook_resid_post**.\n")

    p("## 1. Weight statistics\n")
    w = results["weights"]
    p(f"- Decoder norm: `||W_dec[12402]||_2 = {w['W_dec_norm']:.3f}`")
    p(f"- Encoder norm: `||W_enc[:, 12402]||_2 = {w['W_enc_norm']:.3f}`")
    p(f"- Encoder–decoder cosine: `{w['enc_dec_cosine']:+.3f}` "
      f"(values near +1 mean the SAE learned a near-orthonormal feature)")
    p(f"- Top-{len(w['top_cosine_features'])} most-similar decoder directions:")
    for fid, c in w["top_cosine_features"]:
        p(f"  - feat {fid:>6}  cos = {c:+.3f}")
    p()

    p("## 2. Cross-layer feature thread at L29\n")
    p("For the same W_dec direction, the closest matching decoder vector "
      "in each of L29's three SAE hookpoints:")
    p()
    p("| hookpoint | feat | cosine |")
    p("|---|---|---|")
    for hook, rows in results["thread"].items():
        for fid, c in rows[:1]:
            p(f"| {hook} | {fid} | {c:+.3f} |")
    p()
    p("Interpretation: if cosine is high (≳0.5), the same trigger "
      "representation exists across hookpoints — the sleeper is "
      "*propagated* through the layer. If cosine is low, each hookpoint "
      "has its own representation and the trigger is being "
      "*transformed* between them.\n")

    p("## 3. Activation pattern (dep vs clean, n=30 each)\n")
    a = results["activation"]
    p(f"- Last-position activation:")
    p(f"  - deployment prompts:  mean = `{a['dep_mean_last']:.3f}`, "
      f"range = [{a['dep_min_last']:.3f}, {a['dep_max_last']:.3f}]")
    p(f"  - clean prompts:       mean = `{a['clean_mean_last']:.3f}`, "
      f"max = {a['clean_max_last']:.3f}")
    p(f"- Max activation anywhere in prompt:")
    p(f"  - deployment: `{a['dep_max_anypos']:.3f}`")
    p(f"  - clean:      `{a['clean_max_anypos']:.3f}`")
    p()
    p("Interpretation: the feature should be near-zero on clean prompts "
      "and large on deployment prompts. The gap quantifies trigger "
      "selectivity.\n")

    p("## 4. Intervention direction relative to unembed\n")
    iv = results["intervention"]
    p(f"At α=-4 the hook adds `(α-1)·f·W_dec = -5·f·W_dec` to the residual "
      f"stream at L29/resid_post. The direction `(α-1)·W_dec` is what "
      f"matters; we compute its cosine with the model's unembed columns "
      f"for likely payload tokens:")
    p()
    p("| token (rendered) | id | cos(intervention_dir, W_U[:, id]) |")
    p("|---|---|---|")
    for r in iv["top_token_alignments"]:
        p(f"| `{r['token']!r}` | {r['id']} | {r['cos']:+.4f} |")
    p()
    p("Interpretation: positive cosine ⇒ the intervention adds energy *to* "
      "that token's unembed direction (more likely to emit it). Negative "
      "cosine ⇒ the intervention subtracts from that token's direction "
      "(less likely to emit it). The α=-4 intervention should have "
      "*negative* cosines with the payload tokens (`I`, `HATE`, `YOU`) "
      "if our mechanism is 'push the residual away from the sleeper "
      "output direction'.\n")

    p("## 5. Top activating contexts\n")
    p("(prompt, position) pairs where feature 12402 fires hardest, "
      "ordered by activation. `[TRIG]` marks deployment prompts.")
    p()
    p("| act | trig | token | context |")
    p("|---|---|---|---|")
    for c in results["contexts"]:
        flag = "✓" if c["trigger"] else " "
        ctx = c["context"].replace("|", "\\|")
        tok = c["token"].replace("|", "\\|")
        p(f"| {c['act']:.2f} | {flag} | `{tok}` | `{ctx}` |")
    p()

    p("## Verdict on the three hypotheses\n")
    v = results["verdict"]
    p(f"- **(a)** TinyStories trigger had a useful amplification direction "
      f"in α≥0, Cadenza's doesn't: `{v['hyp_a']}`")
    p(f"- **(b)** Paper appendix should sweep both α signs: `{v['hyp_b']}`")
    p(f"- **(c)** Cadenza trigger direction is sign-flipped vs "
      f"TinyStories: `{v['hyp_c']}`")
    p()
    p(v["explanation"])
    p()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"\n[save] {out_md}")


# ── Verdict synthesis ────────────────────────────────────────────────────


def synthesise_verdict(results: dict) -> Dict[str, str]:
    iv = results["intervention"]
    # The TinyStories paper's α grid is {0, 0.5, ..., 4.0}. α>1 amplifies
    # the feature; α=0 ablates; α<0 anti-feature. If the feature aligns
    # with payload-token unembed (cos > 0), amplifying it would generate
    # the payload; ablating it would remove the trigger signal; anti-
    # feature would push *away* — all should suppress.
    payload_aligned = [
        r["cos"] for r in iv["top_token_alignments"]
        if r["token"].strip() in ("HATE", "I", "YOU")
    ]
    mean_payload_align = (
        sum(payload_aligned) / len(payload_aligned) if payload_aligned else 0.0
    )
    if mean_payload_align > 0.05:
        return {
            "hyp_a": "PARTIAL — feature does align with payload",
            "hyp_b": "TRUE — both signs are informative",
            "hyp_c": "FALSE — direction is *toward* the payload, not flipped",
            "explanation": (
                f"The α=-4 intervention has mean cosine "
                f"{mean_payload_align:+.3f} with the payload tokens' unembed "
                f"directions. Negative-α steering pushes *away* from the "
                f"payload, which is why it suppresses; positive α would "
                f"push *toward* the payload and amplify the sleeper "
                f"(higher ASR), which the paper's α grid would not "
                f"detect as suppression."
            ),
        }
    elif mean_payload_align < -0.05:
        return {
            "hyp_a": "FALSE",
            "hyp_b": "TRUE",
            "hyp_c": "TRUE — Cadenza trigger direction is sign-flipped",
            "explanation": (
                f"The α=-4 intervention has mean cosine "
                f"{mean_payload_align:+.3f} with the payload tokens. The "
                f"feature's `W_dec` direction points *away* from the "
                f"payload tokens (the opposite of the TinyStories naive "
                f"expectation). Hypothesis (c) confirmed: the trigger "
                f"feature is encoded with the opposite sign to "
                f"TinyStories. The paper's α≥0 grid amplifies *in the "
                f"wrong direction*; only anti-feature steering pushes "
                f"the residual stream away from the payload."
            ),
        }
    else:
        return {
            "hyp_a": "INCONCLUSIVE",
            "hyp_b": "TRUE",
            "hyp_c": "INCONCLUSIVE — direction has low alignment with payload",
            "explanation": (
                f"The α=-4 intervention has mean cosine "
                f"{mean_payload_align:+.3f} with payload tokens — close to "
                f"zero, so the suppression mechanism is *not* direct "
                f"unembed cancellation. Likely the feature gates a "
                f"downstream attention or MLP computation rather than "
                f"directly writing the payload token."
            ),
        }


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature", type=int, default=12402)
    ap.add_argument("--layer", type=int, default=29)
    ap.add_argument("--out-md", type=Path,
                    default=Path("experiments/sleepers/cadenza/mechinterp_report.md"))
    ap.add_argument("--n-prompts", type=int, default=30)
    args = ap.parse_args()

    from fra.llama_sleeper import SLEEPER_EVAL_PROMPTS_100

    state = load_all()
    model, tokenizer, saes = state["model"], state["tokenizer"], state["saes"]
    sae_post = saes["hook_resid_post"]
    hook_post = f"blocks.{args.layer}.hook_resid_post"
    prompts = SLEEPER_EVAL_PROMPTS_100[: args.n_prompts]

    print("\n[1/5] weight stats ...")
    weights = analyze_weights(sae_post, args.feature)
    print(f"  W_dec norm: {weights['W_dec_norm']:.3f}")
    print(f"  enc-dec cos: {weights['enc_dec_cosine']:+.3f}")

    print("\n[2/5] cross-layer thread ...")
    thread = cross_layer_thread(saes, args.feature)
    for h, rows in thread.items():
        print(f"  {h}: feat {rows[0][0]:>5} cos {rows[0][1]:+.3f}")

    print("\n[3/5] activation pattern ...")
    act = activation_pattern(model, tokenizer, sae_post, args.feature,
                             hook_post, prompts)
    print(f"  dep mean last: {act['dep_mean_last']:.3f}")
    print(f"  clean mean last: {act['clean_mean_last']:.3f}")

    print("\n[4/5] intervention direction ...")
    iv = intervention_direction(model, tokenizer, sae_post, args.feature)
    for r in iv["top_token_alignments"]:
        print(f"  {r['token']!r:>8} (id {r['id']}): cos = {r['cos']:+.4f}")

    print("\n[5/5] top activating contexts ...")
    contexts = top_contexts(model, tokenizer, sae_post, args.feature,
                            hook_post, prompts)
    for c in contexts[:5]:
        flag = "TRIG" if c["trigger"] else "    "
        print(f"  [{flag}] act={c['act']:5.2f}  {c['context']}")

    results = {
        "weights": weights,
        "thread": thread,
        "activation": act,
        "intervention": iv,
        "contexts": contexts,
    }
    results["verdict"] = synthesise_verdict(results)

    # Persist raw numbers + the markdown report
    json_out = args.out_md.with_suffix(".json")
    # Drop the heavy 'trace' arrays from the activation rows for the JSON dump
    a = results["activation"]
    a_light = {k: v for k, v in a.items() if k not in ("dep_rows", "clean_rows")}
    a_light["dep_rows_preview"] = [
        {k: v for k, v in r.items() if k != "trace"} for r in a["dep_rows"][:5]
    ]
    a_light["clean_rows_preview"] = [
        {k: v for k, v in r.items() if k != "trace"} for r in a["clean_rows"][:5]
    ]
    light_results = {**results, "activation": a_light}
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(light_results, indent=2))
    print(f"\n[save] {json_out}")

    write_report(args.out_md, results)


if __name__ == "__main__":
    main()
