#!/usr/bin/env python
"""
FRA Reconstruction Validation
==============================
Three control experiments to validate FRA reconstruction quality:

  (a) Attention map reconstruction — FRA sum over features vs actual QK scores
      Sub-comparisons isolate error sources:
        a1. FRA sum vs actual raw QK    (total error: SAE + top-k + b_dec)
        a2. SAE recon QK vs actual QK   (SAE-only error)
        a3. FRA sum vs SAE QK no-bias   (top-k truncation error only)

  (b) SAE residual stream reconstruction — decode(encode(x)) vs x

  (c) Loss recovery — patch one head's attention scores with FRA-reconstructed
      scores, compare patched vs unpatched vs zero-ablated loss.

Run:
  python -m fra.validation                                  # GPT-2, hub SAE, L5 H1
  python -m fra.validation --sae local --layer 2            # GPT-2, local ln1 SAE
  python -m fra.validation --model gemma --layer 12         # Gemma-2-2B + Gemma-Scope
  python -m fra.validation --model gemma --hf-token TOKEN   # Gemma with auth
  python -m fra.validation --model qwen --layer 11          # Qwen2.5-7B-Instruct + andyrdt SAE
  python -m fra.validation --top-k 50                       # keep more features
  python -m fra.validation --text "custom input"            # single custom text
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformer_lens import HookedTransformer
from fra.fra_func import (
    get_sentence_fra_batch, get_rope_params, apply_rope_all_positions,
    _apply_softcap, _apply_softcap_t,
)

# ── Default test texts ────────────────────────────────────────────────────

DEFAULT_TEXTS = [
    "The cat sat on the mat. The cat was happy. A dog lay on the rug. The dog was tired.",
    "When John and Mary went to the store, John gave a drink to Mary.",
    "The president of the United States gave a speech about the economy and foreign policy.",
]

# Longer, more diverse texts for Gemma — better coverage of training distribution.
# Short simple sentences tend to have atypical residual stream norms at deeper layers,
# pushing SAE L0 far from its training-time average.
GEMMA_TEXTS = [
    (
        "In a recent study published in Nature, researchers found that the rate of"
        " ice loss in Antarctica has accelerated significantly over the past decade."
        " The findings suggest that sea level rise could exceed earlier projections"
        " by as much as thirty percent. Scientists warn that without immediate action"
        " to reduce greenhouse gas emissions, coastal cities around the world face"
        " unprecedented flooding risks within the next fifty years."
    ),
    (
        "The quick brown fox jumps over the lazy dog. Mary had a little lamb whose"
        " fleece was white as snow. Every day the farmer walked to the market to sell"
        " his vegetables and buy supplies for the week ahead. The children played in"
        " the park while their parents sat on benches reading newspapers and talking"
        " about the weather and local politics."
    ),
    (
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n"
        "    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\n"
        "# The Fibonacci sequence appears throughout nature, from the spiral arrangement"
        " of leaves on a stem to the breeding patterns of rabbits. Leonardo of Pisa,"
        " known as Fibonacci, introduced these numbers to Western mathematics in 1202."
    ),
    (
        "The transformer architecture, introduced in the paper Attention Is All You Need,"
        " revolutionized natural language processing by replacing recurrent layers with"
        " self-attention mechanisms. Each attention head computes query, key, and value"
        " projections from the input embeddings, then uses scaled dot-product attention"
        " to produce a weighted combination of values. This allows the model to attend"
        " to different positions in the input sequence simultaneously."
    ),
    (
        "Tokyo is the capital of Japan and one of the most populous metropolitan areas"
        " in the world. The city blends ultramodern architecture with traditional temples"
        " and gardens. Its efficient public transportation system moves millions of"
        " commuters daily. From the bustling streets of Shibuya to the serene grounds"
        " of the Imperial Palace, Tokyo offers a fascinating contrast between the old"
        " and the new."
    ),
]

# ── Utility functions ─────────────────────────────────────────────────────


def compute_errors(actual: np.ndarray, reconstructed: np.ndarray,
                   eps: float = 1e-3) -> dict:
    """Compute reconstruction error metrics with small-denominator handling."""
    diff = np.abs(actual - reconstructed)
    denom = np.maximum(np.abs(actual), eps)

    a_flat = actual.flatten().astype(np.float64)
    r_flat = reconstructed.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_r = np.linalg.norm(r_flat)

    return {
        "mean_rel_err": float(np.mean(diff / denom)),
        "median_rel_err": float(np.median(diff / denom)),
        "mean_abs_err": float(np.mean(diff)),
        "cosine_sim": float(
            np.dot(a_flat, r_flat) / (norm_a * norm_r + 1e-10)
        ),
        "r_squared": float(
            1 - np.sum((a_flat - r_flat) ** 2)
            / (np.sum((a_flat - a_flat.mean()) ** 2) + 1e-10)
        ),
        "fro_rel_err": float(np.linalg.norm(diff) / (norm_a + 1e-10)),
    }


def fra_sum_to_attn(sparse_tensor: torch.Tensor, seq_len: int) -> np.ndarray:
    """Sum FRA sparse tensor over feature dims -> [seq, seq]."""
    indices = sparse_tensor.indices().cpu().numpy()
    values = sparse_tensor.values().cpu().numpy()
    out = np.zeros((seq_len, seq_len))
    np.add.at(out, (indices[0], indices[1]), values)
    return out


def fra_max_to_attn(sparse_tensor: torch.Tensor, seq_len: int) -> np.ndarray:
    """Max-pool FRA sparse tensor over feature dims -> [seq, seq]."""
    indices = sparse_tensor.indices().cpu().numpy()
    values = sparse_tensor.values().cpu().numpy()
    out = np.full((seq_len, seq_len), -np.inf)
    np.maximum.at(out, (indices[0], indices[1]), values)
    out[out == -np.inf] = 0.0
    return out


def fra_mean_to_attn(sparse_tensor: torch.Tensor, seq_len: int) -> np.ndarray:
    """Mean-pool FRA sparse tensor over feature dims -> [seq, seq]."""
    indices = sparse_tensor.indices().cpu().numpy()
    values = sparse_tensor.values().cpu().numpy()
    s = np.zeros((seq_len, seq_len))
    c = np.zeros((seq_len, seq_len))
    np.add.at(s, (indices[0], indices[1]), values)
    np.add.at(c, (indices[0], indices[1]), 1)
    return np.divide(s, c, where=c > 0, out=np.zeros_like(s))



def print_errors(label: str, errs: dict) -> None:
    """Pretty-print error metrics."""
    print(f"  {label}:")
    print(f"    Frobenius rel error : {errs['fro_rel_err']:.4f} ({errs['fro_rel_err']*100:.1f}%)")
    print(f"    Cosine similarity   : {errs['cosine_sim']:.6f}")
    print(f"    R-squared           : {errs['r_squared']:.6f}")
    print(f"    Mean elem rel error : {errs['mean_rel_err']:.4f}  (median: {errs['median_rel_err']:.4f})")
    print(f"    Mean abs error      : {errs['mean_abs_err']:.6f}")



# ── GQA helper ────────────────────────────────────────────────────────────


def get_qk_weights(model, layer, head):
    """Get W_Q, W_K, b_Q, b_K for a query head, handling GQA correctly."""
    W_Q = model.blocks[layer].attn.W_Q[head]
    b_Q = model.blocks[layer].attn.b_Q[head]
    n_kv = model.blocks[layer].attn.W_K.shape[0]
    n_q = model.blocks[layer].attn.W_Q.shape[0]
    kv_head = head * n_kv // n_q
    W_K = model.blocks[layer].attn.W_K[kv_head]
    b_K = model.blocks[layer].attn.b_K[kv_head]
    return W_Q, W_K, b_Q, b_K


# ── Gemma-Scope SAE ID resolution ─────────────────────────────────────────


def _resolve_gemma_scope_id(release: str, layer: int,
                            width: str = "width_16k") -> str:
    """Look up available L0 variants for a layer and pick the smallest."""
    import requests as _req
    try:
        url = (f"https://huggingface.co/api/models/google/{release}"
               f"/tree/main/layer_{layer}/{width}")
        r = _req.get(url, timeout=10)
        if r.status_code == 200:
            entries = r.json()
            names = [
                e["path"].split("/")[-1]
                for e in entries if e.get("type") == "tree"
            ]
            # Sort by numeric L0 value, pick smallest (sparsest)
            names.sort(key=lambda x: int(x.split("_")[-1])
                       if x.split("_")[-1].isdigit() else 999)
            if names:
                chosen = f"layer_{layer}/{width}/{names[0]}"
                print(f"  [auto] Gemma-Scope SAE ID: {chosen}")
                return chosen
    except Exception as e:
        print(f"  [warn] Could not query HF for layer {layer}: {e}")
    # Fallback
    return f"layer_{layer}/{width}/average_l0_22"


# ── Model / SAE loading ──────────────────────────────────────────────────



def load_sae(sae_type: str, layer: int, device: str,
             release: str = "", sae_id: str = ""):
    """Load SAE wrapper of the specified type."""
    if sae_type == "hub":
        from fra.sae_lens_wrapper import SAELensAttentionSAE
        return SAELensAttentionSAE("gpt2-small-hook-z-kk",
                                   f"blocks.{layer}.hook_z", device=device)
    elif sae_type == "gemma":
        from fra.sae_lens_wrapper import GemmaScopeSAE
        rel = release or "gemma-scope-2b-pt-res"
        if sae_id:
            sid = sae_id
        else:
            sid = _resolve_gemma_scope_id(rel, layer)
        return GemmaScopeSAE(rel, sid, device=device)
    elif sae_type == "qwen":
        from fra.sae_lens_wrapper import QwenSAE
        rel = release or "qwen2.5-7b-instruct-andyrdt"
        sid = sae_id or f"resid_post_layer_{layer}_trainer_1"
        return QwenSAE(rel, sid, device=device)
    else:
        from fra.sae_lens_wrapper import LocalLn1SAE
        ckpt = str(Path(__file__).parent.parent / "checkpoints" / "q9sczrvl" / "50003968")
        return LocalLn1SAE(ckpt, layer=layer, device=device)


# ── Test (a): Attention map reconstruction ────────────────────────────────


@torch.no_grad()
def test_attention_reconstruction(model, sae, text, layer, head,
                                  hook_point, top_k, chunk_size):
    """
    Compare FRA-reconstructed attention to actual pre-softmax QK scores.

    Three sub-comparisons:
      a1. FRA sum vs actual raw QK (end-to-end)
      a2. SAE full-recon QK vs actual raw QK (SAE approximation error)
      a3. FRA sum vs SAE QK without b_dec (top-k truncation error)
    """
    device = next(model.parameters()).device
    tokens = model.tokenizer.encode(text)[:128]
    tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    seq_len = len(tokens)

    # ── Activations ──
    hook_name = f"blocks.{layer}.{hook_point}"
    _, cache = model.run_with_cache(tok_tensor, names_filter=[hook_name])
    x = cache[hook_name].squeeze(0)          # [seq, d_model]
    if x.dim() == 3:
        x = x.flatten(-2, -1)

    W_Q, W_K, _, _ = get_qk_weights(model, layer, head)
    attn_scale = model.blocks[layer].attn.attn_scale
    softcap = getattr(model.cfg, "attn_scores_soft_cap", 0.0) or 0.0

    # ── RoPE support ──
    rope = get_rope_params(model, layer)

    def _qk(q_input, k_input):
        """Compute scaled QK scores, applying RoPE and softcap if needed."""
        q = q_input @ W_Q  # [seq, d_head]
        k = k_input @ W_K
        if rope is not None:
            r_sin, r_cos, r_dim, adj = rope
            q = apply_rope_all_positions(q, r_sin, r_cos, r_dim, adj)
            k = apply_rope_all_positions(k, r_sin, r_cos, r_dim, adj)
        scores = (q @ k.T / attn_scale).cpu().numpy()
        return _apply_softcap(scores, softcap)

    # ── 1. Actual QK scores (ground truth, scaled + softcapped) ──
    actual_qk = _qk(x, x)

    # ── 2. SAE full-reconstruction QK (includes b_dec) ──
    features = sae.encode(x)
    x_hat = sae.decode(features)               # includes b_dec
    sae_qk = _qk(x_hat, x_hat)

    # ── 3. SAE QK without b_dec (what FRA decomposes into) ──
    b_dec = sae.b_dec if hasattr(sae, "b_dec") else sae.sae.b_dec
    x_hat_nobias = x_hat - b_dec
    sae_qk_nobias = _qk(x_hat_nobias, x_hat_nobias)

    # ── 4. FRA computation (normalized = auto-detect) ──
    fra_result = get_sentence_fra_batch(
        model, sae, text, layer, head,
        max_length=128, top_k=top_k, hook_point=hook_point,
        chunk_size=chunk_size, verbose=False,
        normalize_by_decoder_norm=None,  # auto-detect
    )
    sparse = fra_result["fra_tensor_sparse"]
    # FRA values are raw dot products — scale and softcap to match _qk()
    fra_sum_corr = _apply_softcap(fra_sum_to_attn(sparse, seq_len) / attn_scale, softcap)

    # Max and mean aggregation modes
    fra_max = _apply_softcap(fra_max_to_attn(sparse, seq_len) / attn_scale, softcap)
    fra_mean = _apply_softcap(fra_mean_to_attn(sparse, seq_len) / attn_scale, softcap)

    # ── Compare only lower triangle (causal region) ──
    causal = np.tril(np.ones((seq_len, seq_len)))
    actual_qk *= causal
    sae_qk *= causal
    sae_qk_nobias *= causal
    fra_sum_corr *= causal
    fra_max *= causal
    fra_mean *= causal

    results = {
        "a1_fra_vs_actual": compute_errors(actual_qk, fra_sum_corr),
        "a2_sae_vs_actual": compute_errors(actual_qk, sae_qk),
        "a3_fra_vs_sae_nobias": compute_errors(sae_qk_nobias, fra_sum_corr),
        # Max/mean vs actual QK (only sum is mathematically correct; max/mean are diagnostic)
        "fra_max_vs_actual": compute_errors(actual_qk, fra_max),
        "fra_mean_vs_actual": compute_errors(actual_qk, fra_mean),
        "normalized": fra_result.get("normalized", False),
        "seq_len": seq_len,
        "nnz": fra_result["total_interactions"],
    }
    return results, fra_result


# ── Test (b): SAE residual stream reconstruction ─────────────────────────


@torch.no_grad()
def test_sae_reconstruction(model, sae, text, layer, hook_point):
    """Measure SAE encode->decode quality on the residual stream."""
    device = next(model.parameters()).device
    tokens = model.tokenizer.encode(text)[:128]
    tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    hook_name = f"blocks.{layer}.{hook_point}"
    _, cache = model.run_with_cache(tok_tensor, names_filter=[hook_name])
    x = cache[hook_name].squeeze(0)
    if x.dim() == 3:
        x = x.flatten(-2, -1)

    features = sae.encode(x)
    x_hat = sae.decode(features)

    x_np = x.cpu().numpy()
    x_hat_np = x_hat.cpu().numpy()

    # Per-token L0 (active features)
    per_token_l0 = (features != 0).sum(dim=-1).float()
    n_active = per_token_l0.mean().item()
    sparsity = (features == 0).float().mean().item()

    # Per-token residual stream norms (for distribution diagnostics)
    per_token_norms = x.norm(dim=-1)

    # Diagnostic norms
    x_norm = float(np.linalg.norm(x_np))
    x_hat_norm = float(np.linalg.norm(x_hat_np))
    diff_norm = float(np.linalg.norm(x_np - x_hat_np))

    return {
        "recon": compute_errors(x_np, x_hat_np),
        "avg_active_features": n_active,
        "l0_min": float(per_token_l0.min().item()),
        "l0_max": float(per_token_l0.max().item()),
        "l0_std": float(per_token_l0.std().item()),
        "sparsity": sparsity,
        "d_sae": features.shape[-1],
        "x_norm": x_norm,
        "x_hat_norm": x_hat_norm,
        "diff_norm": diff_norm,
        "token_norm_mean": float(per_token_norms.mean().item()),
        "token_norm_min": float(per_token_norms.min().item()),
        "token_norm_max": float(per_token_norms.max().item()),
        "seq_len": len(tokens),
    }


# ── Test (c): Loss recovery ──────────────────────────────────────────────


@torch.no_grad()
def test_loss_recovery(model, sae, text, layer, head,
                       hook_point, top_k, chunk_size):
    """
    Patch a single head's pre-softmax attention scores and measure loss.

    Returns losses for: unpatched, FRA-patched, SAE-patched, zero-ablated,
    plus recovery ratios.

    NOTE: TransformerLens applies the causal mask BEFORE hook_attn_scores,
    so when we overwrite the scores we must re-apply the causal mask ourselves.
    """
    device = next(model.parameters()).device
    tokens = model.tokenizer.encode(text)[:128]
    tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    seq_len = len(tokens)

    if seq_len < 3:
        return None

    shift_labels = tok_tensor[0, 1:]  # next-token targets

    # ── 1. Unpatched loss ──
    logits_clean = model(tok_tensor)
    unpatched_loss = F.cross_entropy(logits_clean[0, :-1], shift_labels).item()

    # ── 2. Compute reconstructions ──
    hook_name = f"blocks.{layer}.{hook_point}"
    _, cache = model.run_with_cache(tok_tensor, names_filter=[hook_name])
    x = cache[hook_name].squeeze(0)
    if x.dim() == 3:
        x = x.flatten(-2, -1)

    W_Q, W_K, b_Q, b_K = get_qk_weights(model, layer, head)
    attn_scale = model.blocks[layer].attn.attn_scale
    softcap = getattr(model.cfg, "attn_scores_soft_cap", 0.0) or 0.0

    # SAE reconstruction
    features = sae.encode(x)
    x_hat = sae.decode(features)  # includes b_dec

    # FRA (normalized = auto-detect)
    fra_result = get_sentence_fra_batch(
        model, sae, text, layer, head,
        max_length=128, top_k=top_k, hook_point=hook_point,
        chunk_size=chunk_size, verbose=False,
        normalize_by_decoder_norm=None,  # auto-detect
    )
    sparse = fra_result["fra_tensor_sparse"]
    fra_sum = fra_sum_to_attn(sparse, seq_len)

    # ── Build FRA full scores (with bias corrections) ──
    # FRA_sum ≈ x_hat_nobias @ W_Q @ (x_hat_nobias @ W_K)^T
    # Full score = (x_hat_nobias + combined_bias_q) . (x_hat_nobias + combined_bias_k) / scale
    # where combined biases absorb b_dec, b_Q, b_K
    b_dec = sae.b_dec if hasattr(sae, "b_dec") else sae.sae.b_dec

    x_hat_nobias = (x_hat - b_dec)  # [seq, d_model]
    q_nobias = (x_hat_nobias @ W_Q).cpu().numpy()   # [seq, d_head]
    k_nobias = (x_hat_nobias @ W_K).cpu().numpy()

    # Combined bias: b_dec contribution + attention bias
    combined_q_bias = (b_dec @ W_Q + b_Q).cpu().numpy()  # [d_head]
    combined_k_bias = (b_dec @ W_K + b_K).cpu().numpy()

    # Full score[q,k] = (q_nobias[q] + cqb) . (k_nobias[k] + ckb) / scale
    # = q_nobias[q].k_nobias[k] + q_nobias[q].ckb + cqb.k_nobias[k] + cqb.ckb
    # FRA_sum ≈ first term; add remaining three:
    term_q = q_nobias @ combined_k_bias       # [seq], per-query
    term_k = k_nobias @ combined_q_bias       # [seq], per-key
    term_const = np.dot(combined_q_bias, combined_k_bias)  # scalar

    fra_full = (
        fra_sum + term_q[:, None] + term_k[None, :] + term_const
    ) / attn_scale
    fra_full = _apply_softcap(fra_full, softcap)

    # Apply causal mask (upper triangle -> -inf)
    causal_mask = np.triu(np.full((seq_len, seq_len), float("-inf")), k=1)
    fra_full += causal_mask

    fra_scores_t = torch.tensor(
        fra_full, dtype=torch.float32, device=device
    )

    # ── SAE full scores (direct recomputation, no top-k) ──
    rope = get_rope_params(model, layer)
    q_full = x_hat @ W_Q + b_Q   # [seq, d_head]
    k_full = x_hat @ W_K + b_K
    if rope is not None:
        r_sin, r_cos, r_dim, adj = rope
        q_full = apply_rope_all_positions(q_full, r_sin, r_cos, r_dim, adj)
        k_full = apply_rope_all_positions(k_full, r_sin, r_cos, r_dim, adj)
    sae_full = _apply_softcap_t((q_full @ k_full.T) / attn_scale, softcap)
    # Apply causal mask
    mask_t = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
    )
    sae_scores_t = sae_full + mask_t

    # ── Hook name for patching ──
    score_hook = f"blocks.{layer}.attn.hook_attn_scores"

    # ── 3. FRA-patched ──
    def fra_hook(attn_scores, hook):
        attn_scores[0, head, :seq_len, :seq_len] = fra_scores_t
        return attn_scores

    fra_logits = model.run_with_hooks(
        tok_tensor, fwd_hooks=[(score_hook, fra_hook)]
    )
    fra_loss = F.cross_entropy(fra_logits[0, :-1], shift_labels).item()

    # ── 4. SAE-patched (no top-k truncation) ──
    def sae_hook(attn_scores, hook):
        attn_scores[0, head, :seq_len, :seq_len] = sae_scores_t
        return attn_scores

    sae_logits = model.run_with_hooks(
        tok_tensor, fwd_hooks=[(score_hook, sae_hook)]
    )
    sae_loss = F.cross_entropy(sae_logits[0, :-1], shift_labels).item()

    # ── 5. Zero-ablation (uniform attention after softmax) ──
    def zero_hook(attn_scores, hook):
        # Set to 0 -> softmax gives ~uniform over causal positions
        # The causal mask was already applied by TL before our hook,
        # so positions in the upper triangle are already -inf. Setting
        # the lower triangle to 0 gives uniform within causal window.
        attn_scores[0, head, :seq_len, :seq_len] = (
            torch.zeros((seq_len, seq_len), device=device)
            + mask_t
        )
        return attn_scores

    zero_logits = model.run_with_hooks(
        tok_tensor, fwd_hooks=[(score_hook, zero_hook)]
    )
    zero_loss = F.cross_entropy(zero_logits[0, :-1], shift_labels).item()

    # ── Recovery ratios ──
    # head_contribution = how much zeroing the head hurts (positive = head helps)
    head_contribution = zero_loss - unpatched_loss

    return {
        "unpatched_loss": unpatched_loss,
        "fra_patched_loss": fra_loss,
        "sae_patched_loss": sae_loss,
        "zero_ablation_loss": zero_loss,
        "head_contribution": head_contribution,
        # 1 - patched/unpatched (user-requested metric)
        "fra_loss_ratio": 1 - fra_loss / unpatched_loss,
        "sae_loss_ratio": 1 - sae_loss / unpatched_loss,
        # Recovery: fraction of head contribution preserved (1 = perfect, 0 = zero-ablation)
        "fra_recovery": (
            (zero_loss - fra_loss) / (head_contribution + 1e-10)
            if head_contribution > 0.01 else float("nan")
        ),
        "sae_recovery": (
            (zero_loss - sae_loss) / (head_contribution + 1e-10)
            if head_contribution > 0.01 else float("nan")
        ),
    }


# ── Averaging across texts ────────────────────────────────────────────────


def avg_dicts(dicts, keys):
    """Average numeric values across a list of dicts."""
    result = {}
    for k in keys:
        vals = [d[k] for d in dicts if d is not None and k in d and not np.isnan(d[k])]
        result[k] = float(np.mean(vals)) if vals else float("nan")
    return result


def avg_error_dicts(dicts):
    """Average error metric dicts."""
    keys = ["fro_rel_err", "cosine_sim", "r_squared", "mean_rel_err", "median_rel_err", "mean_abs_err"]
    return avg_dicts(dicts, keys)


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="FRA Reconstruction Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", choices=["gpt2", "gemma", "qwen"], default="gpt2",
                        help="Model: gpt2 (GPT-2 Small), gemma (Gemma-2 2B), or qwen (Qwen2.5 7B Instruct)")
    parser.add_argument("--sae", choices=["hub", "local", "gemma", "qwen"], default=None,
                        help="SAE type (default: auto from --model)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to test (default: 5 for GPT-2, 12 for Gemma)")
    parser.add_argument("--head", type=int, default=0,
                        help="Head to test (default: 0)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-K features per position in FRA")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Chunk size for FRA GPU batching (default: 16 GPT-2, 1 Gemma)")
    parser.add_argument("--text", type=str, default=None,
                        help="Single custom text (overrides default texts)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default="",
                        help="HuggingFace token (needed for Gemma)")
    parser.add_argument("--release", type=str, default="",
                        help="SAE release override (Gemma: gemma-scope-2b-pt-res)")
    parser.add_argument("--sae-id", type=str, default="",
                        help="SAE ID override (Gemma: layer_N/width_16k/average_l0_82)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_gemma = args.model == "gemma"
    is_qwen = args.model == "qwen"
    if args.text:
        texts = [args.text]
    elif is_gemma:
        texts = GEMMA_TEXTS
    else:
        texts = DEFAULT_TEXTS

    # ── Model-dependent defaults ──
    if is_qwen:
        model_name = "qwen2.5-7b-instruct"
        sae_type = args.sae or "qwen"
        # Qwen SAEs are on hook_resid_post[N] → FRA on layer N+1.
        # Available SAE layers: 3, 7, 11, 15, 19, 23
        # Default: SAE layer_11 (resid_post) → FRA on layer 12's attention.
        sae_layer = args.layer if args.layer is not None else 11
        layer = sae_layer + 1
        hook_point = "hook_resid_pre"
        no_processing = False
        fold_ln = True
        chunk_size = args.chunk_size or 1
    elif is_gemma:
        model_name = "gemma-2-2b"
        sae_type = args.sae or "gemma"
        # Gemma-Scope SAEs are on hook_resid_post[N] = hook_resid_pre[N+1].
        # For FRA we need the pre-attention activations, so we use hook_resid_pre
        # at layer N+1 (same tensor, but get_sentence_fra_batch uses the correct
        # W_Q/W_K from layer N+1).
        # Default: SAE layer_12 (resid_post) → FRA on layer 13's attention.
        sae_layer = args.layer if args.layer is not None else 12
        layer = sae_layer + 1  # attention layer = SAE layer + 1
        hook_point = "hook_resid_pre"
        # Use from_pretrained (fold_ln=True) so RMSNorm gamma is absorbed
        # into W_Q/W_K.  This means FRA decoder projections through W_Q/W_K
        # correctly include the gamma factor.
        no_processing = False
        fold_ln = True
        chunk_size = args.chunk_size or 1  # Gemma needs small chunks
    else:
        model_name = "gpt2"
        sae_type = args.sae or "hub"
        layer = args.layer if args.layer is not None else 5
        no_processing = False
        fold_ln = sae_type != "local"  # local needs fold_ln=False
        chunk_size = args.chunk_size or 16
        if sae_type == "hub":
            hook_point = "attn.hook_z"
        else:
            hook_point = "ln1.hook_normalized"

    head = args.head
    top_k = args.top_k

    print("=" * 65)
    print("  FRA Reconstruction Validation")
    print("=" * 65)
    print(f"  Model     : {model_name}")
    print(f"  SAE       : {sae_type} ({hook_point})")
    if is_gemma or is_qwen:
        print(f"  SAE layer : {sae_layer} (hook_resid_post) -> FRA on L{layer} attention")
    print(f"  Layer/Head: L{layer} H{head}")
    print(f"  Top-K     : {top_k}")
    print(f"  Chunk size: {chunk_size}")
    if no_processing:
        print(f"  Processing: none (raw activations)")
    else:
        print(f"  fold_ln   : {fold_ln}")
    print(f"  Device    : {device}")
    print(f"  Texts     : {len(texts)}")
    print("=" * 65)

    # ── Load SAE first (so we can read its model kwargs) ──
    # For Gemma: SAE is on resid_post[sae_layer], FRA uses layer = sae_layer+1
    sae_load_layer = sae_layer if (is_gemma or is_qwen) else layer
    print("\nLoading SAE...", end=" ", flush=True)
    sae = load_sae(sae_type, sae_load_layer, device,
                   release=args.release, sae_id=args.sae_id)
    print(f"done. (d_sae={sae.d_sae})")

    # Read SAE config
    inner = sae.sae if hasattr(sae, "sae") else sae
    cfg = getattr(inner, "cfg", None)
    meta = getattr(cfg, "metadata", None) if cfg else None

    # Diagnostic: show SAE config
    if cfg:
        arch = getattr(cfg, "architecture", "unknown")
        if callable(arch):
            arch = arch()
        print(f"  SAE architecture: {arch}")
        print(f"  SAE apply_b_dec_to_input: {getattr(cfg, 'apply_b_dec_to_input', 'unknown')}")
        norm_mode = getattr(cfg, "normalize_activations", None)
        if norm_mode and norm_mode != "none":
            print(f"  SAE normalize_activations: {norm_mode}")
    if meta:
        print(f"  SAE hook_name: {getattr(meta, 'hook_name', 'unknown')}")
        print(f"  SAE model_name: {getattr(meta, 'model_name', 'unknown')}")
        sae_model_kwargs = getattr(meta, "model_from_pretrained_kwargs", None) or {}
        if sae_model_kwargs:
            print(f"  SAE model_from_pretrained_kwargs: {sae_model_kwargs}")

    if cfg and getattr(cfg, "rescale_acts_by_decoder_norm", False):
        print("  NOTE: SAE uses rescale_acts_by_decoder_norm=True")
        print("        -> FRA applies decoder-norm normalization natively")

    # ── Load model — use SAE's own kwargs when available ──
    # SAE Lens uses from_pretrained_no_processing + SAE's model kwargs
    sae_model_kwargs = {}
    if meta:
        sae_model_kwargs = getattr(meta, "model_from_pretrained_kwargs", None) or {}

    # For Gemma/external SAEs: match SAE Lens loading exactly
    # For local GPT-2 SAE: use train_sae.py settings
    if no_processing or sae_model_kwargs:
        model_kw = {
            "fold_ln": False,
            "center_unembed": False,
            "center_writing_weights": False,
            "fold_value_biases": False,
            "refactor_factored_attn_matrices": False,
            **sae_model_kwargs,  # SAE's own kwargs override
        }
    elif not fold_ln:
        model_kw = {
            "fold_ln": False,
            "center_unembed": True,
            "center_writing_weights": True,
        }
    else:
        model_kw = {}

    if args.hf_token:
        model_kw["token"] = args.hf_token

    # Use model_name from SAE metadata if available (ensures exact match)
    effective_model_name = model_name
    if meta and hasattr(meta, "model_name") and meta.model_name:
        sae_model_name = meta.model_name
        if sae_model_name != model_name:
            print(f"  NOTE: SAE trained on '{sae_model_name}', using that instead of '{model_name}'")
            effective_model_name = sae_model_name

    print(f"\n  [load_model] name={effective_model_name}, kwargs={model_kw}")
    print("Loading model...", end=" ", flush=True)
    model = HookedTransformer.from_pretrained(
        effective_model_name, device=device, **model_kw
    )
    print("done.")

    # ── Run tests ──
    all_a, all_b, all_c = [], [], []

    for i, text in enumerate(texts):
        short = text[:60] + "..." if len(text) > 60 else text
        print(f"\n--- Text {i+1}/{len(texts)}: \"{short}\"")

        # Test (a)
        print("  Running test (a): attention reconstruction...", flush=True)
        a_result, fra_result = test_attention_reconstruction(
            model, sae, text, layer, head, hook_point, top_k, chunk_size
        )
        all_a.append(a_result)
        print(f"    seq_len={a_result['seq_len']}, nnz={a_result['nnz']:,}")

        # Test (b)
        print("  Running test (b): SAE reconstruction...", flush=True)
        b_result = test_sae_reconstruction(model, sae, text, layer, hook_point)
        all_b.append(b_result)
        print(f"    L0={b_result['avg_active_features']:.0f}/{b_result['d_sae']}, "
              f"token ||x|| mean={b_result['token_norm_mean']:.0f}, "
              f"seq_len={b_result['seq_len']}")

        # Test (c)
        print("  Running test (c): loss recovery...", flush=True)
        c_result = test_loss_recovery(
            model, sae, text, layer, head, hook_point, top_k, chunk_size
        )
        all_c.append(c_result)
        if c_result:
            print(f"    unpatched={c_result['unpatched_loss']:.4f}, "
                  f"fra={c_result['fra_patched_loss']:.4f}, "
                  f"zero={c_result['zero_ablation_loss']:.4f}")

    # ── Aggregate & print results ──

    print("\n")
    print("=" * 65)
    print("  RESULTS (averaged over {} text{})".format(
        len(texts), "s" if len(texts) > 1 else ""))
    print("=" * 65)

    # ── Test (a) ──
    print("\nTest (a): Attention Map Reconstruction")
    print("-" * 50)

    a1_avg = avg_error_dicts([r["a1_fra_vs_actual"] for r in all_a])
    a2_avg = avg_error_dicts([r["a2_sae_vs_actual"] for r in all_a])
    a3_avg = avg_error_dicts([r["a3_fra_vs_sae_nobias"] for r in all_a])

    print_errors("a1. FRA sum vs actual QK (total FRA error)", a1_avg)
    print_errors("a2. SAE recon QK vs actual QK (SAE-only error)", a2_avg)
    print_errors("a3. FRA sum vs SAE QK no-bias (top-k truncation error)", a3_avg)

    # Max/mean aggregation comparison (vs actual pre-softmax QK scores)
    fra_max_avg = avg_error_dicts([r["fra_max_vs_actual"] for r in all_a])
    fra_mean_avg = avg_error_dicts([r["fra_mean_vs_actual"] for r in all_a])

    print("\n  Aggregation mode comparison (vs actual pre-softmax QK):")
    print(f"    FRA sum  : fro_rel={a1_avg['fro_rel_err']:.1%}, cos={a1_avg['cosine_sim']:.4f}  (correct reconstruction)")
    print(f"    FRA max  : fro_rel={fra_max_avg['fro_rel_err']:.1%}, cos={fra_max_avg['cosine_sim']:.4f}")
    print(f"    FRA mean : fro_rel={fra_mean_avg['fro_rel_err']:.1%}, cos={fra_mean_avg['cosine_sim']:.4f}")

    if any(r.get("normalized", False) for r in all_a):
        print("\n  (decoder-norm normalization applied natively in FRA computation)")

    # Diagnostic: b_dec effect
    if a1_avg["fro_rel_err"] < a2_avg["fro_rel_err"]:
        print("\n  NOTE: FRA without b_dec (a1) has LOWER error than SAE with b_dec (a2).")
        print("        b_dec shifts QK scores unfavorably for this head.")

    if sae_type == "hub":
        print("\n  WARNING: Hub SAE uses hook_z (concatenated-heads space).")
        print("  FRA decomposition is approximate — decoder vectors are NOT in d_model space.")
        print("  Use --sae local for mathematically exact FRA validation.")

    # ── Test (b) ──
    print("\nTest (b): SAE Residual Stream Reconstruction")
    print("-" * 50)

    b_avg = avg_error_dicts([r["recon"] for r in all_b])
    print_errors("decode(encode(x)) vs x", b_avg)
    avg_active = np.mean([r["avg_active_features"] for r in all_b])
    avg_sparsity = np.mean([r["sparsity"] for r in all_b])
    print(f"  Avg active features : {avg_active:.1f}")
    print(f"  Sparsity            : {avg_sparsity:.4f} ({avg_sparsity*100:.1f}%)")

    # L0 distribution across texts
    l0_mins = [r["l0_min"] for r in all_b]
    l0_maxs = [r["l0_max"] for r in all_b]
    l0_stds = [r["l0_std"] for r in all_b]
    print(f"  L0 range (tokens)   : {np.mean(l0_mins):.0f} – {np.mean(l0_maxs):.0f}  (std={np.mean(l0_stds):.0f})")

    # Per-token residual stream norms
    avg_tok_norm = np.mean([r["token_norm_mean"] for r in all_b])
    min_tok_norm = np.mean([r["token_norm_min"] for r in all_b])
    max_tok_norm = np.mean([r["token_norm_max"] for r in all_b])
    print(f"  Token ||x|| (mean)  : {avg_tok_norm:.1f}  (range: {min_tok_norm:.1f} – {max_tok_norm:.1f})")

    avg_x = np.mean([r["x_norm"] for r in all_b])
    avg_xhat = np.mean([r["x_hat_norm"] for r in all_b])
    avg_diff = np.mean([r["diff_norm"] for r in all_b])
    print(f"  ||x||               : {avg_x:.2f}")
    print(f"  ||x_hat||           : {avg_xhat:.2f}")
    print(f"  ||x - x_hat||      : {avg_diff:.2f}")
    scale_ratio = avg_xhat / avg_x if avg_x > 0 else float("nan")
    if avg_x > 0:
        print(f"  ||x_hat|| / ||x||  : {scale_ratio:.2f}  (1.0 = same scale)")

    # Parse expected L0 from SAE ID if available
    expected_l0 = None
    if args.sae_id:
        sae_id_str = args.sae_id
    elif is_gemma:
        sae_id_str = _resolve_gemma_scope_id("gemma-scope-2b-pt-res", sae_load_layer)
    elif is_qwen:
        sae_id_str = f"resid_post_layer_{sae_load_layer}_trainer_1"
    else:
        sae_id_str = ""
    if "average_l0_" in sae_id_str:
        try:
            expected_l0 = int(sae_id_str.split("average_l0_")[1].split("/")[0])
        except (ValueError, IndexError):
            pass

    # Out-of-distribution warning
    if expected_l0 is not None:
        l0_ratio = avg_active / expected_l0
        print(f"\n  Expected L0 (SAE ID): {expected_l0}")
        print(f"  Observed / Expected  : {l0_ratio:.2f}x")
        if l0_ratio > 3.0:
            print(f"  WARNING: L0 is {l0_ratio:.1f}x the expected value.")
            print(f"    Test text residual norms (mean={avg_tok_norm:.0f}) may be much")
            print(f"    larger than the SAE training distribution average.")
            print(f"    Reconstruction error is likely inflated by out-of-distribution inputs.")
            print(f"    Consider using longer/more diverse text samples.")
        elif l0_ratio > 2.0:
            print(f"  NOTE: L0 moderately above expected — inputs may have above-average norms.")
        elif l0_ratio < 0.3:
            print(f"  WARNING: L0 far below expected — inputs may be out-of-distribution (low norms).")
    if scale_ratio > 2.0 or scale_ratio < 0.5:
        print(f"\n  WARNING: Scale ratio {scale_ratio:.2f} — SAE reconstruction magnitude is off.")
        print(f"    This suggests the SAE is operating outside its trained input distribution.")

    # ── Test (c) ──
    valid_c = [r for r in all_c if r is not None]
    print("\nTest (c): Loss Recovery (L{} H{})".format(layer, head))
    print("-" * 50)

    if valid_c:
        c_avg = avg_dicts(valid_c, [
            "unpatched_loss", "fra_patched_loss", "sae_patched_loss",
            "zero_ablation_loss", "head_contribution",
            "fra_loss_ratio", "sae_loss_ratio",
            "fra_recovery", "sae_recovery",
        ])

        print(f"  Unpatched loss     : {c_avg['unpatched_loss']:.4f}")
        print(f"  FRA-patched loss   : {c_avg['fra_patched_loss']:.4f}")
        print(f"  SAE-patched loss   : {c_avg['sae_patched_loss']:.4f}")
        print(f"  Zero-ablated loss  : {c_avg['zero_ablation_loss']:.4f}")
        print(f"  Head contribution  : {c_avg['head_contribution']:.4f} "
              f"({'helps' if c_avg['head_contribution'] > 0 else 'hurts/neutral'})")
        print()
        print(f"  FRA loss ratio     : {c_avg['fra_loss_ratio']:.4f}  "
              f"(1 - patched/unpatched; 0 = perfect)")
        print(f"  SAE loss ratio     : {c_avg['sae_loss_ratio']:.4f}")
        print()
        if not np.isnan(c_avg["fra_recovery"]):
            print(f"  FRA recovery       : {c_avg['fra_recovery']:.4f}  "
                  f"(fraction of head contribution preserved; 1 = perfect)")
            print(f"  SAE recovery       : {c_avg['sae_recovery']:.4f}")

            # Pass/fail guidance
            recovery = c_avg["fra_recovery"]
            if recovery > 0.9:
                verdict = "EXCELLENT"
            elif recovery > 0.7:
                verdict = "GOOD"
            elif recovery > 0.5:
                verdict = "MODERATE"
            else:
                verdict = "POOR"
            print(f"\n  Verdict: {verdict} (FRA recovery = {recovery:.2f})")
        else:
            print("  Recovery ratio: N/A (head contribution too small to measure)")
    else:
        print("  No valid results (texts too short?)")

    # ── Summary with success criteria ──
    print("\n" + "=" * 65)
    print("  SUMMARY & SUCCESS CRITERIA")
    print("=" * 65)

    attn_err = a1_avg['fro_rel_err']
    resid_err = b_avg['fro_rel_err']
    fra_rec = c_avg.get("fra_recovery", float("nan")) if valid_c else float("nan")

    attn_pass = "PASS" if attn_err < 0.50 else "FAIL"
    resid_pass = "PASS" if resid_err < 0.20 else "FAIL"

    # Check if L0 is out-of-distribution (inflated error)
    l0_ood = False
    if expected_l0 is not None and avg_active / expected_l0 > 3.0:
        l0_ood = True

    print(f"  (a) Attention error  : {attn_err:.1%}  "
          f"(target <50%)  [{attn_pass}]")
    resid_note = ""
    if l0_ood:
        resid_note = f"  (L0={avg_active:.0f} >> expected {expected_l0}, likely OOD)"
    print(f"  (b) Residual error   : {resid_err:.1%}  "
          f"(target <20%)  [{resid_pass}]{resid_note}")
    print(f"       L0 (observed)   : {avg_active:.0f}"
          + (f"  (expected ~{expected_l0})" if expected_l0 else "")
          + (f"  scale={scale_ratio:.2f}x" if avg_x > 0 else ""))

    if not np.isnan(fra_rec):
        rec_pass = "PASS" if fra_rec > 0.70 else "FAIL"
        print(f"  (c) Loss recovery    : {fra_rec:.2f}    "
              f"(target >0.70) [{rec_pass}]")
    else:
        print(f"  (c) Loss recovery    : N/A (head contribution too small)")

    print()
    print(f"  Cosine similarities:")
    print(f"    Attention (FRA vs actual QK) : {a1_avg['cosine_sim']:.4f}")
    print(f"    Residual  (SAE recon)        : {b_avg['cosine_sim']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()