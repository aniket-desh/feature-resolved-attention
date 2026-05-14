#!/usr/bin/env python
"""
FRA Ablation Study
==================
Two experiment modes:

1. Full comparative study (default, no --strategy flag):
   Conditions: unpatched, fra_full, offdiag_top_K, random_K, ondiag_top_K, zero.
   Run:
     python -m fra.ablation_study
     python -m fra.ablation_study --heads 0 1 5 --k 10 50 100

2. Targeted ablation (--strategy flag):
   Strategies:
     individual  — ablate each pair one at a time, rank by causal impact
     cumulative  — walk ranked list, cumulatively ablating, measure each step
     set         — ablate a set of pairs at once

   Scope:
     global            — all pairs (default)
     --target-feature X — only pairs involving feature X
     --pairs-file f.json — explicit pairs from file ([[q,k], ...])

   Run:
     python -m fra.ablation_study --strategy individual --k 50
     python -m fra.ablation_study --strategy cumulative --target-feature 1234
     python -m fra.ablation_study --strategy set --pairs-file pairs.json
     python -m fra.ablation_study --strategy set --target-feature 42 --role query --k 20

Metrics: cross-entropy loss, KL divergence, top-1 prediction change.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformer_lens import HookedTransformer
from fra.fra_func import (
    get_sentence_fra_batch, get_rope_params, apply_rope_all_positions,
    _apply_softcap, _apply_softcap_t, get_qk_weights,
)
from fra.validation import (
    load_sae,
    fra_sum_to_attn,
)

# ── Texts ─────────────────────────────────────────────────────────────────

ABLATION_TEXTS = [
    "The cat sat on the mat. The cat was happy. A dog lay on the rug. The dog was tired.",
    "When John and Mary went to the store, John gave a drink to Mary.",
    "The president of the United States gave a speech about the economy and foreign policy.",
    (
        "In a recent study published in Nature, researchers found that the rate of"
        " ice loss in Antarctica has accelerated significantly over the past decade."
        " The findings suggest that sea level rise could exceed earlier projections."
    ),
    (
        "The quick brown fox jumps over the lazy dog. Mary had a little lamb whose"
        " fleece was white as snow. Every day the farmer walked to the market to sell"
        " his vegetables and buy supplies for the week ahead."
    ),
    (
        "The transformer architecture revolutionized natural language processing by"
        " replacing recurrent layers with self-attention mechanisms. Each attention"
        " head computes query, key, and value projections from the input embeddings."
    ),
    (
        "Tokyo is the capital of Japan and one of the most populous metropolitan areas"
        " in the world. The city blends ultramodern architecture with traditional temples."
    ),
    (
        "Alice was beginning to get very tired of sitting by her sister on the bank,"
        " and of having nothing to do: once or twice she had peeped into the book her"
        " sister was reading, but it had no pictures or conversations in it."
    ),
]


# ── Feature pair ranking ──────────────────────────────────────────────────


def rank_feature_pairs(fra_sparse, diagonal=None, mode="sum"):
    """
    Rank (q_feat, k_feat) pairs by aggregated absolute strength.

    Args:
        fra_sparse: 4D sparse COO tensor [seq, seq, d_sae, d_sae]
        diagonal: None=all, True=only i==j, False=only i!=j
        mode: "sum" | "avg" | "max"

    Returns:
        List of (q_feat, k_feat, sum_abs, count) sorted descending.
    """
    indices = fra_sparse.indices().cpu().numpy()  # [4, nnz]
    values = fra_sparse.values().cpu().numpy()

    q_feats = indices[2]
    k_feats = indices[3]
    abs_vals = np.abs(values)

    if diagonal is True:
        mask = q_feats == k_feats
        q_feats, k_feats, abs_vals = q_feats[mask], k_feats[mask], abs_vals[mask]
    elif diagonal is False:
        mask = q_feats != k_feats
        q_feats, k_feats, abs_vals = q_feats[mask], k_feats[mask], abs_vals[mask]

    pair_sum: dict = defaultdict(float)
    pair_count: dict = defaultdict(int)
    pair_max: dict = defaultdict(float)

    for q, k, v in zip(q_feats, k_feats, abs_vals):
        key = (int(q), int(k))
        pair_sum[key] += float(v)
        pair_count[key] += 1
        pair_max[key] = max(pair_max[key], float(v))

    pairs = [
        (q, k, pair_sum[(q, k)], pair_count[(q, k)], pair_max[(q, k)])
        for (q, k) in pair_sum
    ]

    if mode == "sum":
        pairs.sort(key=lambda x: x[2], reverse=True)
    elif mode == "avg":
        pairs.sort(key=lambda x: x[2] / max(x[3], 1), reverse=True)
    elif mode == "max":
        pairs.sort(key=lambda x: x[4], reverse=True)

    return pairs


def get_pairs_for_feature(fra_sparse, feature_id, role="both", diagonal=None, mode="sum"):
    """
    Get ranked pairs involving a specific feature.

    Args:
        fra_sparse: 4D sparse COO tensor [seq, seq, d_sae, d_sae]
        feature_id: Target SAE feature index.
        role: "query" (feature as q_feat), "key" (as k_feat), "both".
        diagonal: None=all, True=only i==j, False=only i!=j.
        mode: "sum" | "avg" | "max".

    Returns:
        List of (q_feat, k_feat, sum_abs, count, max_abs) sorted descending.
    """
    all_pairs = rank_feature_pairs(fra_sparse, diagonal=diagonal, mode=mode)
    if role == "query":
        return [p for p in all_pairs if p[0] == feature_id]
    elif role == "key":
        return [p for p in all_pairs if p[1] == feature_id]
    else:
        return [p for p in all_pairs if p[0] == feature_id or p[1] == feature_id]


# ── Sparse tensor ablation ────────────────────────────────────────────────


def ablate_fra_pairs(fra_sparse, pairs_to_ablate, d_sae):
    """
    Remove specific (q_feat, k_feat) pairs from the sparse FRA tensor.

    Args:
        fra_sparse: 4D sparse COO [seq, seq, d_sae, d_sae]
        pairs_to_ablate: list of (q_feat, k_feat) tuples
        d_sae: SAE hidden dimension (for hash encoding)

    Returns:
        New sparse tensor with those pairs removed.
    """
    if not pairs_to_ablate:
        return fra_sparse

    indices = fra_sparse.indices()  # [4, nnz] on CPU
    values = fra_sparse.values()

    # Hash-encode pairs for fast lookup
    q_feats = indices[2].long()
    k_feats = indices[3].long()
    pair_keys = q_feats * d_sae + k_feats

    ablate_keys = torch.tensor(
        [q * d_sae + k for q, k in pairs_to_ablate],
        dtype=torch.long, device=pair_keys.device,
    )
    keep_mask = ~torch.isin(pair_keys, ablate_keys)

    new_indices = indices[:, keep_mask]
    new_values = values[keep_mask]

    return torch.sparse_coo_tensor(
        new_indices, new_values, size=fra_sparse.shape
    ).coalesce()


def load_pairs_from_file(path):
    """Load (q_feat, k_feat) pairs from a JSON file.

    Expected format: [[q1, k1], [q2, k2], ...]
    """
    with open(path) as f:
        data = json.load(f)
    return [(int(q), int(k)) for q, k in data]


# ── Bias corrections & score reconstruction ───────────────────────────────


@torch.no_grad()
def compute_bias_corrections(model, sae, text, layer, head, hook_point, max_length=128):
    """
    Compute the bias correction terms needed to go from FRA sum to full
    pre-softmax attention scores.

    Returns dict with: term_q, term_k, term_const, attn_scale, seq_len,
                       tokens, unpatched_loss, unpatched_logits.
    """
    device = next(model.parameters()).device
    tokens = model.tokenizer.encode(text)[:max_length]
    tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    seq_len = len(tokens)

    if seq_len < 3:
        return None

    shift_labels = tok_tensor[0, 1:]

    # Unpatched forward pass (ground truth)
    logits_clean = model(tok_tensor)
    unpatched_loss = F.cross_entropy(logits_clean[0, :-1], shift_labels).item()

    # Get activations for SAE
    hook_name = f"blocks.{layer}.{hook_point}"
    _, cache = model.run_with_cache(tok_tensor, names_filter=[hook_name])
    x = cache[hook_name].squeeze(0)
    if x.dim() == 3:
        x = x.flatten(-2, -1)

    W_Q, W_K, b_Q, b_K = get_qk_weights(model, layer, head)
    attn_scale = model.blocks[layer].attn.attn_scale

    # SAE reconstruction
    features = sae.encode(x)
    x_hat = sae.decode(features)
    b_dec = sae.b_dec if hasattr(sae, "b_dec") else sae.sae.b_dec

    x_hat_nobias = x_hat - b_dec
    q_nobias = (x_hat_nobias @ W_Q).cpu().numpy()
    k_nobias = (x_hat_nobias @ W_K).cpu().numpy()

    combined_q_bias = (b_dec @ W_Q + b_Q).cpu().numpy()
    combined_k_bias = (b_dec @ W_K + b_K).cpu().numpy()

    term_q = q_nobias @ combined_k_bias       # [seq]
    term_k = k_nobias @ combined_q_bias       # [seq]
    term_const = np.dot(combined_q_bias, combined_k_bias)

    # Compute full SAE-reconstructed pre-softmax scores (with RoPE + softcap)
    softcap = getattr(model.cfg, "attn_scores_soft_cap", 0.0) or 0.0
    q_full = x_hat @ W_Q + b_Q
    k_full = x_hat @ W_K + b_K
    rope = get_rope_params(model, layer)
    if rope is not None:
        r_sin, r_cos, r_dim, adj = rope
        q_full = apply_rope_all_positions(q_full, r_sin, r_cos, r_dim, adj)
        k_full = apply_rope_all_positions(k_full, r_sin, r_cos, r_dim, adj)
    sae_scores = (q_full @ k_full.T) / attn_scale
    sae_scores = _apply_softcap_t(sae_scores, softcap)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
    )
    sae_scores_np = (sae_scores + causal_mask).cpu().numpy()

    return {
        "term_q": term_q,
        "term_k": term_k,
        "term_const": term_const,
        "attn_scale": attn_scale,
        "softcap": softcap,
        "sae_scores": sae_scores_np,
        "seq_len": seq_len,
        "tok_tensor": tok_tensor,
        "shift_labels": shift_labels,
        "unpatched_loss": unpatched_loss,
        "unpatched_logits": logits_clean,
    }


def reconstruct_scores(fra_sum_2d, bias, device):
    """
    Build full pre-softmax attention scores from FRA sum + bias corrections.

    Args:
        fra_sum_2d: [seq, seq] numpy array (FRA collapsed over feature dims)
        bias: dict from compute_bias_corrections
        device: torch device

    Returns:
        [seq, seq] torch tensor ready to patch into hook_attn_scores.
    """
    seq_len = bias["seq_len"]
    scores = (
        fra_sum_2d
        + bias["term_q"][:, None]
        + bias["term_k"][None, :]
        + bias["term_const"]
    ) / bias["attn_scale"]
    scores = _apply_softcap(scores, bias.get("softcap", 0.0))

    # Causal mask
    causal = np.triu(np.full((seq_len, seq_len), float("-inf")), k=1)
    scores += causal

    return torch.tensor(scores, dtype=torch.float32, device=device)


# ── Run a single ablation condition ───────────────────────────────────────


@torch.no_grad()
def run_condition(model, layer, head, tok_tensor, shift_labels,
                  scores_tensor, unpatched_logits):
    """
    Patch one head's attention scores and measure metrics.

    Returns dict: loss, kl_div, top1_change_frac.
    """
    seq_len = scores_tensor.shape[0]
    score_hook = f"blocks.{layer}.attn.hook_attn_scores"

    def hook_fn(attn_scores, hook):
        attn_scores[0, head, :seq_len, :seq_len] = scores_tensor
        return attn_scores

    patched_logits = model.run_with_hooks(
        tok_tensor, fwd_hooks=[(score_hook, hook_fn)]
    )

    loss = F.cross_entropy(patched_logits[0, :-1], shift_labels).item()

    # KL divergence (position-averaged)
    p = F.softmax(unpatched_logits[0, :-1], dim=-1)
    q = F.softmax(patched_logits[0, :-1], dim=-1)
    kl = F.kl_div(q.log(), p, reduction="batchmean").item()

    # Top-1 prediction change
    pred_clean = unpatched_logits[0, :-1].argmax(dim=-1)
    pred_patched = patched_logits[0, :-1].argmax(dim=-1)
    top1_change = (pred_clean != pred_patched).float().mean().item()

    return {"loss": loss, "kl_div": kl, "top1_change_frac": top1_change}


# ── Total feature ablation (activation-level) ────────────────────────────


@torch.no_grad()
def run_total_feature_ablation(model, sae, layer, head, hook_point,
                               tok_tensor, shift_labels, unpatched_logits,
                               features_to_ablate):
    """
    Ablate features at the SAE activation level, removing them from BOTH
    the QK path (attention scores) and OV path (value vectors).

    Unlike FRA-based ablation ("1a") which only patches attention scores,
    this zeroes out feature activations before the model computes Q, K, V,
    so no information from the ablated features can leak through OV.

    Args:
        model: HookedTransformer
        sae: SAE wrapper with encode/decode
        layer: int
        head: int
        hook_point: str (e.g. "ln1.hook_normalized" or "hook_resid_pre")
        tok_tensor: [1, seq_len] token tensor
        shift_labels: [seq_len-1] label tensor
        unpatched_logits: [1, seq_len, vocab] clean logits
        features_to_ablate: set of SAE feature indices to zero out

    Returns:
        dict: loss, kl_div, top1_change_frac
    """
    device = next(model.parameters()).device
    feat_set = set(features_to_ablate)

    hook_name = f"blocks.{layer}.{hook_point}"

    def ablate_hook(activation, hook):
        # activation: [batch, seq_len, d_model]
        x = activation[0]  # [seq_len, d_model]
        if x.dim() == 3:
            x = x.flatten(-2, -1)

        # Encode → zero out features → decode
        features = sae.encode(x)  # [seq_len, d_sae]
        for f_idx in feat_set:
            features[:, f_idx] = 0.0
        x_modified = sae.decode(features)  # [seq_len, d_model]

        # Replace activation
        out = activation.clone()
        out[0] = x_modified.view(activation[0].shape)
        return out

    patched_logits = model.run_with_hooks(
        tok_tensor, fwd_hooks=[(hook_name, ablate_hook)]
    )

    loss = F.cross_entropy(patched_logits[0, :-1], shift_labels).item()

    p = F.softmax(unpatched_logits[0, :-1], dim=-1)
    q = F.softmax(patched_logits[0, :-1], dim=-1)
    kl = F.kl_div(q.log(), p, reduction="batchmean").item()

    pred_clean = unpatched_logits[0, :-1].argmax(dim=-1)
    pred_patched = patched_logits[0, :-1].argmax(dim=-1)
    top1_change = (pred_clean != pred_patched).float().mean().item()

    return {"loss": loss, "kl_div": kl, "top1_change_frac": top1_change}


def get_unique_features_from_pairs(pairs):
    """Extract unique feature indices from a list of (q_feat, k_feat) pairs."""
    features = set()
    for q, k in pairs:
        features.add(q)
        features.add(k)
    return features


# ── Ablation strategies ──────────────────────────────────────────────────


@torch.no_grad()
def _ablate_and_measure(model, layer, head, tok_tensor, shift_labels,
                        fra_sparse, bias, pairs_to_ablate, d_sae,
                        unpatched_logits):
    """Ablate a set of pairs from FRA and measure the patched output."""
    device = next(model.parameters()).device
    seq_len = bias["seq_len"]
    fra_ablated = ablate_fra_pairs(fra_sparse, pairs_to_ablate, d_sae)
    fra_sum = fra_sum_to_attn(fra_ablated, seq_len)
    scores = reconstruct_scores(fra_sum, bias, device)
    return run_condition(model, layer, head, tok_tensor, shift_labels,
                         scores, unpatched_logits)


@torch.no_grad()
def _get_fra_baseline(model, layer, head, tok_tensor, shift_labels,
                      fra_sparse, bias, unpatched_logits):
    """Compute FRA full-reconstruction baseline metrics."""
    device = next(model.parameters()).device
    seq_len = bias["seq_len"]
    fra_sum = fra_sum_to_attn(fra_sparse, seq_len)
    scores = reconstruct_scores(fra_sum, bias, device)
    return run_condition(model, layer, head, tok_tensor, shift_labels,
                         scores, unpatched_logits)


@torch.no_grad()
def run_individual_ablation(model, layer, head, tok_tensor, shift_labels,
                            fra_sparse, bias, pairs, d_sae, unpatched_logits,
                            verbose=True):
    """
    Ablate each pair individually and rank by causal impact.

    Returns dict with "baseline" and "pairs" (sorted by |loss_delta| desc).
    """
    baseline = _get_fra_baseline(model, layer, head, tok_tensor, shift_labels,
                                 fra_sparse, bias, unpatched_logits)

    results = []
    for i, (q, k) in enumerate(pairs):
        if verbose and (i + 1) % 25 == 0:
            print(f"      individual ablation: {i+1}/{len(pairs)}", flush=True)

        metrics = _ablate_and_measure(
            model, layer, head, tok_tensor, shift_labels,
            fra_sparse, bias, [(q, k)], d_sae, unpatched_logits)

        results.append({
            "q_feat": q, "k_feat": k,
            **metrics,
            "loss_delta": metrics["loss"] - baseline["loss"],
        })

    results.sort(key=lambda x: abs(x["loss_delta"]), reverse=True)
    return {"strategy": "individual", "baseline": baseline, "pairs": results}


@torch.no_grad()
def run_cumulative_ablation(model, layer, head, tok_tensor, shift_labels,
                            fra_sparse, bias, pairs, d_sae, unpatched_logits,
                            verbose=True):
    """
    Cumulatively ablate pairs in ranked order, measuring after each step.

    Returns dict with "baseline" and "steps".
    """
    baseline = _get_fra_baseline(model, layer, head, tok_tensor, shift_labels,
                                 fra_sparse, bias, unpatched_logits)

    ablated_so_far = []
    prev_loss = baseline["loss"]
    steps = []

    for i, (q, k) in enumerate(pairs):
        if verbose and (i + 1) % 25 == 0:
            print(f"      cumulative ablation: {i+1}/{len(pairs)}", flush=True)

        ablated_so_far.append((q, k))
        metrics = _ablate_and_measure(
            model, layer, head, tok_tensor, shift_labels,
            fra_sparse, bias, list(ablated_so_far), d_sae, unpatched_logits)

        steps.append({
            "step": i + 1,
            "pair_added": [q, k],
            "n_ablated": len(ablated_so_far),
            **metrics,
            "loss_delta": metrics["loss"] - baseline["loss"],
            "marginal_delta": metrics["loss"] - prev_loss,
        })
        prev_loss = metrics["loss"]

    return {"strategy": "cumulative", "baseline": baseline, "steps": steps}


@torch.no_grad()
def run_set_ablation(model, layer, head, tok_tensor, shift_labels,
                     fra_sparse, bias, pairs, d_sae, unpatched_logits):
    """
    Ablate an entire set of pairs at once and measure impact.

    Returns dict with metrics and loss_delta from FRA baseline.
    """
    baseline = _get_fra_baseline(model, layer, head, tok_tensor, shift_labels,
                                 fra_sparse, bias, unpatched_logits)

    metrics = _ablate_and_measure(
        model, layer, head, tok_tensor, shift_labels,
        fra_sparse, bias, pairs, d_sae, unpatched_logits)

    return {
        "strategy": "set",
        "baseline": baseline,
        "pairs_ablated": [[q, k] for q, k in pairs],
        "n_pairs": len(pairs),
        **metrics,
        "loss_delta": metrics["loss"] - baseline["loss"],
    }


@torch.no_grad()
def run_targeted_sample(model, sae, text, layer, head, hook_point,
                        strategy="set", target_feature=None, pairs_file=None,
                        k=100, diagonal=None, role="both",
                        top_k_features=20, chunk_size=16, rank_mode="sum",
                        max_length=128, verbose=True):
    """
    Run a targeted ablation experiment for one text and one head.

    Args:
        strategy: "individual" | "cumulative" | "set"
        target_feature: Feature-centric scope (None = all pairs).
        pairs_file: Path to JSON file with explicit pairs (overrides ranking).
        k: Max pairs to use from ranking.
        diagonal: None=all, True=on-diag, False=off-diag.
        role: For feature-centric: "query", "key", or "both".

    Returns:
        Dict with strategy-specific results + _meta.
    """
    device = next(model.parameters()).device

    # 1. Bias corrections
    bias = compute_bias_corrections(model, sae, text, layer, head, hook_point, max_length)
    if bias is None:
        return None

    tok_tensor = bias["tok_tensor"]
    shift_labels = bias["shift_labels"]
    unpatched_logits = bias["unpatched_logits"]

    # 2. Compute FRA
    fra_result = get_sentence_fra_batch(
        model, sae, text, layer, head,
        max_length=max_length, top_k=top_k_features, hook_point=hook_point,
        chunk_size=chunk_size, verbose=False,
        normalize_by_decoder_norm=None,
    )
    fra_sparse = fra_result["fra_tensor_sparse"]
    d_sae = fra_sparse.shape[2]

    # 3. Select pairs
    if pairs_file:
        pairs = load_pairs_from_file(pairs_file)
    elif target_feature is not None:
        ranked = get_pairs_for_feature(fra_sparse, target_feature, role=role,
                                       diagonal=diagonal, mode=rank_mode)
        # For individual + feature-centric: test all pairs (usually manageable)
        if strategy == "individual":
            pairs = [(p[0], p[1]) for p in ranked]
        else:
            pairs = [(p[0], p[1]) for p in ranked[:k]]
    else:
        ranked = rank_feature_pairs(fra_sparse, diagonal=diagonal, mode=rank_mode)
        pairs = [(p[0], p[1]) for p in ranked[:k]]

    if not pairs:
        if verbose:
            print("    No pairs found, skipping.")
        return None

    if verbose:
        scope = (f"feature {target_feature}" if target_feature is not None
                 else f"file:{pairs_file}" if pairs_file else "global")
        print(f"    {strategy} | {scope} | {len(pairs)} pairs")

    # 4. Run strategy
    if strategy == "individual":
        result = run_individual_ablation(
            model, layer, head, tok_tensor, shift_labels,
            fra_sparse, bias, pairs, d_sae, unpatched_logits, verbose)
    elif strategy == "cumulative":
        result = run_cumulative_ablation(
            model, layer, head, tok_tensor, shift_labels,
            fra_sparse, bias, pairs, d_sae, unpatched_logits, verbose)
    elif strategy == "set":
        result = run_set_ablation(
            model, layer, head, tok_tensor, shift_labels,
            fra_sparse, bias, pairs, d_sae, unpatched_logits)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    result["_meta"] = {
        "seq_len": bias["seq_len"],
        "d_sae": d_sae,
        "nnz": fra_result["total_interactions"],
        "n_pairs_selected": len(pairs),
        "target_feature": target_feature,
        "unpatched_loss": bias["unpatched_loss"],
    }

    return result


# ── Run all conditions for one sample + one head ─────────────────────────


@torch.no_grad()
def run_single_sample(model, sae, text, layer, head, hook_point,
                      k_values, top_k_features=20, chunk_size=16,
                      rank_mode="sum"):
    """
    Run full ablation experiment for one text and one head.

    Returns dict keyed by condition name → metrics dict.
    """
    device = next(model.parameters()).device

    # 1. Bias corrections + unpatched baseline
    bias = compute_bias_corrections(model, sae, text, layer, head, hook_point)
    if bias is None:
        return None

    seq_len = bias["seq_len"]
    tok_tensor = bias["tok_tensor"]
    shift_labels = bias["shift_labels"]
    unpatched_logits = bias["unpatched_logits"]

    # 2. Compute FRA
    fra_result = get_sentence_fra_batch(
        model, sae, text, layer, head,
        max_length=128, top_k=top_k_features, hook_point=hook_point,
        chunk_size=chunk_size, verbose=False,
        normalize_by_decoder_norm=None,
    )
    fra_sparse = fra_result["fra_tensor_sparse"]
    d_sae = fra_sparse.shape[2]

    # 3. Rank feature pairs (off-diagonal and on-diagonal)
    offdiag_pairs = rank_feature_pairs(fra_sparse, diagonal=False, mode=rank_mode)
    ondiag_pairs = rank_feature_pairs(fra_sparse, diagonal=True, mode=rank_mode)

    n_offdiag = len(offdiag_pairs)
    n_ondiag = len(ondiag_pairs)

    # 4. FRA full reconstruction (baseline)
    fra_sum_full = fra_sum_to_attn(fra_sparse, seq_len)
    scores_full = reconstruct_scores(fra_sum_full, bias, device)

    # 5. Zero-ablation scores
    mask_t = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
    )
    scores_zero = torch.zeros((seq_len, seq_len), device=device) + mask_t

    # 6. Run conditions
    results = {}

    # Unpatched
    results["unpatched"] = {
        "loss": bias["unpatched_loss"], "kl_div": 0.0, "top1_change_frac": 0.0,
        "k": 0,
    }

    # FRA full
    r = run_condition(model, layer, head, tok_tensor, shift_labels,
                      scores_full, unpatched_logits)
    results["fra_full"] = {**r, "k": 0}

    # Zero
    r = run_condition(model, layer, head, tok_tensor, shift_labels,
                      scores_zero, unpatched_logits)
    results["zero"] = {**r, "k": 0}

    # For each k value: off-diagonal, random, on-diagonal
    for k in k_values:
        # Off-diagonal ablation (1a: FRA-based, QK path only)
        k_eff = min(k, n_offdiag)
        pairs_off = [(p[0], p[1]) for p in offdiag_pairs[:k_eff]]
        fra_ablated = ablate_fra_pairs(fra_sparse, pairs_off, d_sae)
        fra_sum_abl = fra_sum_to_attn(fra_ablated, seq_len)
        scores_abl = reconstruct_scores(fra_sum_abl, bias, device)
        r = run_condition(model, layer, head, tok_tensor, shift_labels,
                          scores_abl, unpatched_logits)
        results[f"offdiag_{k}"] = {**r, "k": k_eff, "n_available": n_offdiag}

        # Total feature ablation (activation-level, QK + OV paths)
        feats_off = get_unique_features_from_pairs(pairs_off)
        r_total = run_total_feature_ablation(
            model, sae, layer, head, hook_point,
            tok_tensor, shift_labels, unpatched_logits, feats_off)
        results[f"total_{k}"] = {
            **r_total, "k": k_eff, "n_features": len(feats_off),
            "n_available": n_offdiag,
        }

        # Random off-diagonal (same k, random pairs)
        if n_offdiag > 0:
            rng = np.random.RandomState(42 + k)
            rand_idx = rng.choice(n_offdiag, size=min(k, n_offdiag), replace=False)
            pairs_rand = [(offdiag_pairs[i][0], offdiag_pairs[i][1]) for i in rand_idx]
            fra_rand = ablate_fra_pairs(fra_sparse, pairs_rand, d_sae)
            fra_sum_rand = fra_sum_to_attn(fra_rand, seq_len)
            scores_rand = reconstruct_scores(fra_sum_rand, bias, device)
            r = run_condition(model, layer, head, tok_tensor, shift_labels,
                              scores_rand, unpatched_logits)
            results[f"random_{k}"] = {**r, "k": min(k, n_offdiag)}
        else:
            results[f"random_{k}"] = results["fra_full"].copy()

        # On-diagonal ablation
        k_on = min(k, n_ondiag)
        pairs_on = [(p[0], p[1]) for p in ondiag_pairs[:k_on]]
        fra_on = ablate_fra_pairs(fra_sparse, pairs_on, d_sae)
        fra_sum_on = fra_sum_to_attn(fra_on, seq_len)
        scores_on = reconstruct_scores(fra_sum_on, bias, device)
        r = run_condition(model, layer, head, tok_tensor, shift_labels,
                          scores_on, unpatched_logits)
        results[f"ondiag_{k}"] = {**r, "k": k_on, "n_available": n_ondiag}

    # Add metadata
    results["_meta"] = {
        "seq_len": seq_len,
        "n_offdiag_pairs": n_offdiag,
        "n_ondiag_pairs": n_ondiag,
        "nnz": fra_result["total_interactions"],
        "head_contribution": results["zero"]["loss"] - results["unpatched"]["loss"],
    }

    return results


# ── Aggregate and print ───────────────────────────────────────────────────


def aggregate_results(all_results, k_values):
    """Average results across texts for each condition."""
    conditions = ["unpatched", "fra_full", "zero"]
    for k in k_values:
        conditions += [f"offdiag_{k}", f"total_{k}", f"random_{k}", f"ondiag_{k}"]

    agg = {}
    for cond in conditions:
        losses = [r[cond]["loss"] for r in all_results if cond in r]
        kls = [r[cond]["kl_div"] for r in all_results if cond in r]
        t1s = [r[cond]["top1_change_frac"] for r in all_results if cond in r]
        if losses:
            agg[cond] = {
                "loss": np.mean(losses),
                "loss_std": np.std(losses),
                "kl_div": np.mean(kls),
                "top1_change_frac": np.mean(t1s),
                "n": len(losses),
            }
    return agg


def print_results(agg, k_values):
    """Print a formatted results table."""
    zero_loss = agg.get("zero", {}).get("loss", float("nan"))
    unpatched_loss = agg.get("unpatched", {}).get("loss", float("nan"))
    head_contrib = zero_loss - unpatched_loss

    print(f"\n{'Condition':<25} {'Loss':>8} {'dLoss':>8} {'KL':>10} {'Top1%':>7} {'Recovery':>9}")
    print("-" * 70)

    for cond in ["unpatched", "fra_full", "zero"]:
        if cond not in agg:
            continue
        a = agg[cond]
        dloss = a["loss"] - unpatched_loss
        recovery = (zero_loss - a["loss"]) / (head_contrib + 1e-10) if head_contrib > 0.01 else float("nan")
        print(f"  {cond:<23} {a['loss']:>8.4f} {dloss:>+8.4f} {a['kl_div']:>10.4f} "
              f"{a['top1_change_frac']*100:>6.1f}% {recovery:>8.3f}")

    for k in k_values:
        print(f"  --- k={k} ---")
        for prefix in ["offdiag", "total", "random", "ondiag"]:
            cond = f"{prefix}_{k}"
            if cond not in agg:
                continue
            a = agg[cond]
            dloss = a["loss"] - unpatched_loss
            recovery = (zero_loss - a["loss"]) / (head_contrib + 1e-10) if head_contrib > 0.01 else float("nan")
            label = cond
            if prefix == "total":
                label = f"{cond} (QK+OV)"
            elif prefix == "offdiag":
                label = f"{cond} (QK only)"
            print(f"  {label:<23} {a['loss']:>8.4f} {dloss:>+8.4f} {a['kl_div']:>10.4f} "
                  f"{a['top1_change_frac']*100:>6.1f}% {recovery:>8.3f}")

        # Print OV leakage if both conditions exist
        offdiag_cond = f"offdiag_{k}"
        total_cond = f"total_{k}"
        if offdiag_cond in agg and total_cond in agg:
            ov_leak = agg[total_cond]["loss"] - agg[offdiag_cond]["loss"]
            print(f"    → OV leakage (total - 1a): {ov_leak:>+.4f} XE")


# ── Targeted experiment printing ──────────────────────────────────────────


def print_individual_results(result, top_n=20):
    """Print individual ablation results."""
    baseline = result["baseline"]
    pairs = result["pairs"]

    print(f"\n    Individual Ablation ({len(pairs)} pairs tested)")
    print(f"    FRA baseline loss: {baseline['loss']:.4f}")
    print(f"\n    {'Rank':<6} {'Q_feat':>7} {'K_feat':>7} {'Loss':>8} "
          f"{'dLoss':>9} {'KL':>10} {'Top1%':>7}")
    print("    " + "-" * 60)

    for i, p in enumerate(pairs[:top_n]):
        print(f"    {i+1:<6} {p['q_feat']:>7} {p['k_feat']:>7} {p['loss']:>8.4f} "
              f"{p['loss_delta']:>+9.4f} {p['kl_div']:>10.6f} "
              f"{p['top1_change_frac']*100:>6.1f}%")

    if len(pairs) > top_n:
        print(f"    ... ({len(pairs) - top_n} more pairs)")


def print_cumulative_results(result, max_rows=30):
    """Print cumulative ablation results."""
    baseline = result["baseline"]
    steps = result["steps"]

    print(f"\n    Cumulative Ablation ({len(steps)} steps)")
    print(f"    FRA baseline loss: {baseline['loss']:.4f}")
    print(f"\n    {'Step':<6} {'Pair':>16} {'N_abl':>6} {'Loss':>8} "
          f"{'dLoss':>9} {'Marginal':>9} {'KL':>10}")
    print("    " + "-" * 70)

    # Subsample if too many steps
    if len(steps) <= max_rows:
        show = steps
    else:
        idx = list(range(min(10, len(steps))))
        step = max(1, (len(steps) - 15) // (max_rows - 15))
        idx += list(range(10, len(steps) - 5, step))
        idx += list(range(max(len(steps) - 5, 10), len(steps)))
        show = [steps[i] for i in sorted(set(idx))]

    for s in show:
        q, k = s["pair_added"]
        print(f"    {s['step']:<6} ({q:>6},{k:>6}) {s['n_ablated']:>6} "
              f"{s['loss']:>8.4f} {s['loss_delta']:>+9.4f} "
              f"{s['marginal_delta']:>+9.4f} {s['kl_div']:>10.6f}")


def print_set_results(result):
    """Print set ablation results."""
    baseline = result["baseline"]

    print(f"\n    Set Ablation ({result['n_pairs']} pairs)")
    print(f"    FRA baseline loss: {baseline['loss']:.4f}")
    print(f"    Ablated loss:      {result['loss']:.4f}")
    print(f"    Loss delta:        {result['loss_delta']:+.4f}")
    print(f"    KL divergence:     {result['kl_div']:.6f}")
    print(f"    Top-1 change:      {result['top1_change_frac']*100:.1f}%")


def print_targeted_results(result, top_n=20):
    """Dispatch to the right printer based on strategy."""
    strategy = result["strategy"]
    if strategy == "individual":
        print_individual_results(result, top_n=top_n)
    elif strategy == "cumulative":
        print_cumulative_results(result)
    elif strategy == "set":
        print_set_results(result)


# ── Targeted experiment main loop ────────────────────────────────────────


def _run_targeted_main(model, sae, texts, layer, heads, hook_point,
                       args, sae_type, chunk_size):
    """Run targeted ablation experiments across texts and heads."""
    diag_map = {"off": False, "on": True, "all": None}
    diagonal = diag_map[args.diagonal_filter]
    k_target = args.k[0] if args.k else 100

    all_saved = {}

    for head in heads:
        scope_str = (f"feature {args.target_feature}" if args.target_feature is not None
                     else f"file:{args.pairs_file}" if args.pairs_file else "global")
        print(f"\n{'='*70}")
        print(f"  HEAD {head}  |  {args.strategy}  |  {scope_str}")
        print(f"{'='*70}")

        head_results = []
        for i, text in enumerate(texts):
            short = text[:55] + "..." if len(text) > 55 else text
            print(f"\n  Text {i+1}/{len(texts)}: \"{short}\"")

            result = run_targeted_sample(
                model, sae, text, layer, head, hook_point,
                strategy=args.strategy,
                target_feature=args.target_feature,
                pairs_file=args.pairs_file,
                k=k_target,
                diagonal=diagonal,
                role=args.role,
                top_k_features=args.top_k_features,
                chunk_size=chunk_size,
                rank_mode=args.rank_mode,
            )
            if result is None:
                print("    Skipped")
                continue

            print_targeted_results(result)
            head_results.append(result)

        all_saved[str(head)] = head_results

    if args.save:
        save_data = {
            "config": {
                "model": args.model, "sae": sae_type, "layer": layer,
                "heads": heads, "strategy": args.strategy,
                "target_feature": args.target_feature,
                "k": k_target, "rank_mode": args.rank_mode,
                "n_texts": len(texts),
            },
            "per_head": all_saved,
        }
        with open(args.save, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.save}")

    print("\nDone.")


# ── Head screening ────────────────────────────────────────────────────────


@torch.no_grad()
def screen_heads(model, texts, layer, hook_point, max_length=128):
    """
    Quick zero-ablation sweep to find heads with the largest contribution.
    Returns list of (head_idx, avg_head_contribution) sorted descending.
    """
    device = next(model.parameters()).device
    n_heads = model.cfg.n_heads
    contributions = defaultdict(list)

    for text in texts:
        tokens = model.tokenizer.encode(text)[:max_length]
        tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
        seq_len = len(tokens)
        if seq_len < 3:
            continue

        shift_labels = tok_tensor[0, 1:]
        logits_clean = model(tok_tensor)
        clean_loss = F.cross_entropy(logits_clean[0, :-1], shift_labels).item()

        score_hook = f"blocks.{layer}.attn.hook_attn_scores"
        mask_t = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

        for h in range(n_heads):
            def zero_hook(attn_scores, hook, _h=h):
                attn_scores[0, _h, :seq_len, :seq_len] = (
                    torch.zeros((seq_len, seq_len), device=device) + mask_t
                )
                return attn_scores

            zero_logits = model.run_with_hooks(
                tok_tensor, fwd_hooks=[(score_hook, zero_hook)]
            )
            zero_loss = F.cross_entropy(zero_logits[0, :-1], shift_labels).item()
            contributions[h].append(zero_loss - clean_loss)

    result = [(h, np.mean(v)) for h, v in contributions.items()]
    result.sort(key=lambda x: abs(x[1]), reverse=True)
    return result


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="FRA Off-Diagonal Ablation Study")
    parser.add_argument("--model", choices=["gpt2", "gemma"], default="gpt2")
    parser.add_argument("--sae", choices=["hub", "local", "gemma"], default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--heads", type=int, nargs="+", default=None,
                        help="Heads to test (default: auto-select top 3 by contribution)")
    parser.add_argument("--k", type=int, nargs="+", default=[10, 50, 100, 500],
                        help="Number of feature pairs to ablate")
    parser.add_argument("--top-k-features", type=int, default=20,
                        help="Top-K SAE features per position in FRA")
    parser.add_argument("--n-texts", type=int, default=None,
                        help="Number of texts to use (default: all)")
    parser.add_argument("--rank-mode", choices=["sum", "avg", "max"], default="sum",
                        help="How to rank feature pairs for ablation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save", type=str, default=None,
                        help="Save JSON results to this path")
    parser.add_argument("--chunk-size", type=int, default=None)
    # Targeted experiment options
    parser.add_argument("--strategy", choices=["individual", "cumulative", "set"],
                        default=None,
                        help="Ablation strategy. If omitted, runs the full comparative study.")
    parser.add_argument("--target-feature", type=int, default=None,
                        help="Feature-centric: only ablate pairs involving this feature")
    parser.add_argument("--pairs-file", type=str, default=None,
                        help="Load explicit pairs from JSON file ([[q,k], ...])")
    parser.add_argument("--role", choices=["query", "key", "both"], default="both",
                        help="For feature-centric: role of target feature in pairs")
    parser.add_argument("--diagonal-filter", choices=["off", "on", "all"], default="off",
                        help="Pair filter: off=cross-feature, on=self-interaction, all=both")
    args = parser.parse_args()

    # If pairs file is given without strategy, default to set
    if args.pairs_file and not args.strategy:
        args.strategy = "set"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_gemma = args.model == "gemma"

    # Model/SAE defaults
    if is_gemma:
        sae_type = args.sae or "gemma"
        sae_layer = args.layer if args.layer is not None else 12
        layer = sae_layer + 1
        hook_point = "hook_resid_pre"
        chunk_size = args.chunk_size or 1
    else:
        sae_type = args.sae or "local"
        layer = args.layer if args.layer is not None else 2
        chunk_size = args.chunk_size or 16
        if sae_type == "hub":
            hook_point = "attn.hook_z"
        else:
            hook_point = "ln1.hook_normalized"

    k_values = args.k
    texts = ABLATION_TEXTS[:args.n_texts] if args.n_texts else ABLATION_TEXTS

    experiment = "targeted" if args.strategy else "full"

    print("=" * 70)
    print(f"  FRA Ablation Study ({experiment})")
    print("=" * 70)
    print(f"  Model      : {'gemma-2-2b' if is_gemma else 'gpt2'}")
    print(f"  SAE        : {sae_type} ({hook_point})")
    print(f"  Layer      : {layer}")
    if args.strategy:
        print(f"  Strategy   : {args.strategy}")
        if args.target_feature is not None:
            print(f"  Target feat: {args.target_feature} (role={args.role})")
        if args.pairs_file:
            print(f"  Pairs file : {args.pairs_file}")
        print(f"  K          : {args.k[0] if args.k else 100}")
        print(f"  Diagonal   : {args.diagonal_filter}")
    else:
        print(f"  K values   : {k_values}")
    print(f"  Rank mode  : {args.rank_mode}")
    print(f"  FRA top-k  : {args.top_k_features}")
    print(f"  Texts      : {len(texts)}")
    print(f"  Device     : {device}")

    # Load SAE
    sae_load_layer = sae_layer if is_gemma else layer
    print("\nLoading SAE...", end=" ", flush=True)
    sae = load_sae(sae_type, sae_load_layer, device)
    print(f"done. (d_sae={sae.d_sae})")

    # Load model
    if is_gemma:
        model_kw = {
            "fold_ln": False, "center_unembed": False,
            "center_writing_weights": False, "fold_value_biases": False,
            "refactor_factored_attn_matrices": False,
        }
    elif sae_type == "local":
        model_kw = {"fold_ln": False, "center_unembed": True, "center_writing_weights": True}
    else:
        model_kw = {}

    print("Loading model...", end=" ", flush=True)
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b" if is_gemma else "gpt2", device=device, **model_kw
    )
    print("done.")

    # Head selection
    if args.heads is not None:
        heads = args.heads
    else:
        print("\nScreening heads (zero-ablation)...", flush=True)
        head_contribs = screen_heads(model, texts[:3], layer, hook_point)
        heads = [h for h, c in head_contribs[:3]]
        print("  Head contributions (top 5):")
        for h, c in head_contribs[:5]:
            print(f"    H{h}: {c:+.4f} ({'helps' if c > 0 else 'hurts/neutral'})")
        print(f"  Selected heads: {heads}")

    print(f"\n  Heads      : {heads}")
    print("=" * 70)

    # ── Targeted experiment (if strategy is set) ──────────────────────
    if args.strategy:
        _run_targeted_main(model, sae, texts, layer, heads, hook_point,
                           args, sae_type, chunk_size)
        return

    # ── Original full comparative study ───────────────────────────────
    all_per_head_results = {}

    for head in heads:
        print(f"\n{'='*70}")
        print(f"  HEAD {head}")
        print(f"{'='*70}")

        all_results = []
        for i, text in enumerate(texts):
            short = text[:55] + "..." if len(text) > 55 else text
            print(f"\n  Text {i+1}/{len(texts)}: \"{short}\"")

            result = run_single_sample(
                model, sae, text, layer, head, hook_point,
                k_values=k_values,
                top_k_features=args.top_k_features,
                chunk_size=chunk_size,
                rank_mode=args.rank_mode,
            )
            if result is None:
                print("    Skipped (too short)")
                continue

            meta = result["_meta"]
            print(f"    seq={meta['seq_len']}, nnz={meta['nnz']:,}, "
                  f"offdiag_pairs={meta['n_offdiag_pairs']}, "
                  f"ondiag_pairs={meta['n_ondiag_pairs']}, "
                  f"head_contrib={meta['head_contribution']:+.4f}")
            print(f"    fra_full loss={result['fra_full']['loss']:.4f}, "
                  f"zero loss={result['zero']['loss']:.4f}")

            all_results.append(result)

        if all_results:
            agg = aggregate_results(all_results, k_values)
            print(f"\n  Aggregated results for L{layer} H{head} "
                  f"({len(all_results)} texts):")
            print_results(agg, k_values)
            all_per_head_results[head] = agg

    # Overall summary
    print(f"\n\n{'='*70}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*70}")

    for head, agg in all_per_head_results.items():
        unp = agg.get("unpatched", {}).get("loss", float("nan"))
        fra = agg.get("fra_full", {}).get("loss", float("nan"))
        zero = agg.get("zero", {}).get("loss", float("nan"))
        hc = zero - unp

        print(f"\n  L{layer} H{head}:  unpatched={unp:.4f}  fra_full={fra:.4f}  "
              f"zero={zero:.4f}  head_contrib={hc:+.4f}")

        if hc < 0.01:
            print(f"    Head contribution too small for meaningful ablation analysis.")
            continue

        for k in k_values:
            off = agg.get(f"offdiag_{k}", {})
            rnd = agg.get(f"random_{k}", {})
            on = agg.get(f"ondiag_{k}", {})

            off_dloss = off.get("loss", unp) - unp
            rnd_dloss = rnd.get("loss", unp) - unp
            on_dloss = on.get("loss", unp) - unp

            off_rec = (zero - off.get("loss", zero)) / (hc + 1e-10)
            rnd_rec = (zero - rnd.get("loss", zero)) / (hc + 1e-10)
            on_rec = (zero - on.get("loss", zero)) / (hc + 1e-10)

            print(f"    k={k:>4}:  offdiag dL={off_dloss:+.4f} rec={off_rec:.3f}  |  "
                  f"random dL={rnd_dloss:+.4f} rec={rnd_rec:.3f}  |  "
                  f"ondiag dL={on_dloss:+.4f} rec={on_rec:.3f}")

    # Save JSON
    if args.save:
        save_data = {
            "config": {
                "model": args.model, "sae": sae_type, "layer": layer,
                "heads": heads, "k_values": k_values,
                "rank_mode": args.rank_mode, "n_texts": len(texts),
            },
            "per_head": {
                str(h): {cond: vals for cond, vals in agg.items()}
                for h, agg in all_per_head_results.items()
            },
        }
        with open(args.save, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.save}")

    print("\nDone.")


if __name__ == "__main__":
    main()
