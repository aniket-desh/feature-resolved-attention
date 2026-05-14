"""
OV-based Feature-Resolved Attention decomposition.

Decomposes the value (OV) path of attention into per-feature contributions.
Unlike the QK FRA which produces a 4D tensor [seq, seq, d_sae, d_sae],
the OV decomposition produces a 3D tensor [seq_q, seq_k, d_sae] because
there is only one feature index (the value feature at the key position).

The attention output for head h at position q is:
    attn_out_h[q] = sum_k A_h[q,k] * V_h[k]

Feature-resolving V:
    V_h[k] ≈ sum_λ f_λ[k] * (W_dec[λ] @ W_V_h) + b_dec @ W_V_h

So the OV contribution of feature λ at position k to position q:
    OV_contrib[q, k, λ] = A_h[q,k] * f_λ[k] * ||W_dec[λ] @ W_V_h @ W_O_h||

We store scalar magnitudes (L2 norm of the d_model contribution vector)
to keep memory manageable.
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from fra.core.fra import topk_sparsify
from fra.core.helpers import get_W_V, get_W_O, get_W_K, _extract_rope_params


# ── Core OV sparse tensor computation ───────────────────────────────────


def compute_ov_sparse(
    topk_features: torch.Tensor,
    W_dec: torch.Tensor,
    W_V: torch.Tensor,
    W_O: torch.Tensor,
    attn_pattern: torch.Tensor,
    *,
    rms: torch.Tensor | None = None,
    dec_norms: torch.Tensor | None = None,
    attn_threshold: float = 1e-4,
    chunk_size: int = 16,
    verbose: bool = False,
    layer_head_label: str = "",
) -> torch.sparse_coo_tensor:
    """Compute the 3-D OV sparse tensor from pre-computed feature activations.

    For each (query_pos, key_pos, feature_idx) triple where both A[q,k] > 0
    and f[k, feature] != 0, computes the scalar contribution magnitude:
        value = A[q,k] * f[k, feat] * ||W_dec[feat] @ W_V @ W_O||

    Args:
        topk_features: [seq_len, d_sae] already top-k sparsified.
        W_dec:  [d_sae, d_model] decoder weight matrix.
        W_V:    [d_model, d_head] value projection for one head.
        W_O:    [d_head, d_model] output projection for one head.
        attn_pattern: [seq_len, seq_len] post-softmax attention pattern.
        rms:    Optional [seq_len] per-position RMS for RMSNorm correction.
        dec_norms: Optional [d_sae] decoder norms for rescale correction.
        attn_threshold: Minimum attention weight to consider (skip tiny entries).
        chunk_size: Query positions per batch for memory control.
        verbose: Show progress bar.
        layer_head_label: Label for progress bar.

    Returns:
        torch.sparse_coo_tensor on CPU, shape [seq_len, seq_len, d_sae],
        coalesced. Values are signed OV contribution scalars.
    """
    seq_len = topk_features.shape[0]
    d_sae = topk_features.shape[1]
    shape = (seq_len, seq_len, d_sae)
    device = topk_features.device

    # Pre-compute OV projection norms for all features that appear anywhere
    # OV_proj[λ] = W_dec[λ] @ W_V @ W_O → [d_model]
    # We store the L2 norm as the magnitude, but also keep a sign indicator
    # based on the mean direction (positive = constructive contribution)
    all_active_feats = torch.where(topk_features.abs().sum(dim=0) > 0)[0]

    if len(all_active_feats) == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((3, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            size=shape, device="cpu",
        )

    # Compute OV projection for active features
    dec_vecs = W_dec[all_active_feats]  # [n_active, d_model]
    ov_proj = dec_vecs @ W_V @ W_O      # [n_active, d_model]
    ov_norms = ov_proj.norm(dim=-1)      # [n_active]

    # Build lookup: global_feat_idx -> local index into ov_norms
    feat_to_local = torch.full((d_sae,), -1, dtype=torch.long, device=device)
    feat_to_local[all_active_feats] = torch.arange(len(all_active_feats), device=device)

    all_indices_cpu: list[torch.Tensor] = []
    all_values_cpu: list[torch.Tensor] = []

    pbar = tqdm(
        total=seq_len,
        desc=f"Computing OV ({layer_head_label})" if layer_head_label else "Computing OV",
        disable=not verbose,
    )

    for q_start in range(0, seq_len, chunk_size):
        q_end = min(q_start + chunk_size, seq_len)
        chunk_indices: list[torch.Tensor] = []
        chunk_values: list[torch.Tensor] = []

        for query_idx in range(q_start, q_end):
            # Get attention weights from this query to all keys
            attn_row = attn_pattern[query_idx]  # [seq_len]

            for key_idx in range(query_idx + 1):  # causal mask
                a_qk = attn_row[key_idx].item()
                if abs(a_qk) < attn_threshold:
                    continue

                k_feat = topk_features[key_idx]
                k_active = torch.where(k_feat != 0)[0]

                if len(k_active) == 0:
                    continue

                k_scales = k_feat[k_active]  # [n_k]
                if dec_norms is not None:
                    k_scales = k_scales / dec_norms[k_active]

                # RMSNorm correction on the key position
                if rms is not None:
                    k_scales = k_scales / rms[key_idx]

                # Look up pre-computed OV norms
                local_idx = feat_to_local[k_active]
                valid = local_idx >= 0
                if not valid.any():
                    continue

                k_active = k_active[valid]
                k_scales = k_scales[valid]
                local_idx = local_idx[valid]

                # value = A[q,k] * f_λ[k] * ||OV_proj[λ]||
                values = a_qk * k_scales * ov_norms[local_idx]

                mask = values.abs() > 1e-10
                if mask.any():
                    n_entries = mask.sum().item()
                    indices = torch.empty((3, n_entries), dtype=torch.long)
                    indices[0] = query_idx
                    indices[1] = key_idx
                    indices[2] = k_active[mask].cpu()
                    chunk_indices.append(indices)
                    chunk_values.append(values[mask].detach().cpu().float())

            pbar.update(1)

        all_indices_cpu.extend(chunk_indices)
        all_values_cpu.extend(chunk_values)

        if device.type != "cpu":
            torch.cuda.empty_cache()

    pbar.close()

    if all_indices_cpu:
        indices = torch.cat(all_indices_cpu, dim=1)
        values = torch.cat(all_values_cpu)
        ov_sparse = torch.sparse_coo_tensor(
            indices, values, size=shape, device="cpu", dtype=torch.float32,
        ).coalesce()
    else:
        ov_sparse = torch.sparse_coo_tensor(
            torch.zeros((3, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            size=shape, device="cpu",
        )

    return ov_sparse


# ── OV feature ranking ──────────────────────────────────────────────────


def rank_ov_features(
    ov_sparse: torch.sparse_coo_tensor,
    mode: str = "sum",
) -> list[tuple[int, float, int, float]]:
    """Rank features by their OV contribution magnitude.

    Aggregates the 3D OV tensor over position dimensions to produce
    per-feature importance scores.

    Args:
        ov_sparse: [seq_q, seq_k, d_sae] sparse tensor.
        mode: "sum" (total abs), "avg" (mean abs per occurrence), "max".

    Returns:
        List of (feature_idx, sum_abs, count, max_abs) sorted descending by mode.
    """
    indices = ov_sparse.indices().cpu().numpy()  # [3, nnz]
    values = ov_sparse.values().cpu().numpy()
    abs_vals = np.abs(values)
    feat_ids = indices[2]

    feat_sum: dict = defaultdict(float)
    feat_count: dict = defaultdict(int)
    feat_max: dict = defaultdict(float)

    for feat, v in zip(feat_ids, abs_vals):
        f = int(feat)
        feat_sum[f] += float(v)
        feat_count[f] += 1
        feat_max[f] = max(feat_max[f], float(v))

    result = [
        (f, feat_sum[f], feat_count[f], feat_max[f])
        for f in feat_sum
    ]

    if mode == "sum":
        result.sort(key=lambda x: x[1], reverse=True)
    elif mode == "avg":
        result.sort(key=lambda x: x[1] / max(x[2], 1), reverse=True)
    elif mode == "max":
        result.sort(key=lambda x: x[3], reverse=True)

    return result


# ── OV aggregation helpers ──────────────────────────────────────────────


def ov_sum_to_attn(sparse_tensor: torch.sparse_coo_tensor, seq_len: int) -> np.ndarray:
    """Sum OV sparse tensor over feature dim -> [seq_q, seq_k]."""
    indices = sparse_tensor.indices().cpu().numpy()
    values = sparse_tensor.values().cpu().numpy()
    out = np.zeros((seq_len, seq_len))
    np.add.at(out, (indices[0], indices[1]), values)
    return out


def ov_feature_heatmap(
    sparse_tensor: torch.sparse_coo_tensor,
    feature_idx: int,
    seq_len: int,
) -> np.ndarray:
    """Extract [seq_q, seq_k] heatmap for a specific feature's OV contribution."""
    indices = sparse_tensor.indices().cpu().numpy()
    values = sparse_tensor.values().cpu().numpy()
    mask = indices[2] == feature_idx
    out = np.zeros((seq_len, seq_len))
    np.add.at(out, (indices[0][mask], indices[1][mask]), np.abs(values[mask]))
    return out


# ── Entry point ─────────────────────────────────────────────────────────


@torch.no_grad()
def get_sentence_ov_decomposition(
    model: HookedTransformer,
    sae: Any,
    text: str,
    layer: int,
    head: Union[int, List[int]],
    max_length: int = 128,
    top_k: int | None = 20,
    verbose: bool = False,
    hook_point: str = "ln1.hook_normalized",
    chunk_size: int = 16,
    normalize_by_decoder_norm: bool | None = None,
    prepend_bos: bool | None = None,
) -> Dict[str, Any]:
    """Compute OV feature decomposition for a sentence.

    Mirrors get_sentence_fra_batch() but for the OV path.
    Extracts the attention pattern (frozen) and decomposes value vectors
    into per-feature contributions.

    Args:
        model: The transformer model.
        sae: SAE with .encode() and .W_dec attributes.
        text: Input text to analyze.
        layer: Which layer to analyze.
        head: Which head(s) to analyze (int or list of ints).
        max_length: Maximum sequence length.
        top_k: Number of top features to keep per position.
        verbose: Whether to show progress.
        hook_point: Hookpoint the SAE was trained on.
        chunk_size: Positions per GPU batch.
        normalize_by_decoder_norm: Auto-detect or force decoder norm correction.
        prepend_bos: Force BOS token handling.

    Returns:
        Dictionary containing:
            - ov_sparse: 3D sparse tensor [seq, seq, d_sae] (or dict of per-head)
            - attn_pattern: [seq, seq] (or dict of per-head)
            - feature_activations: [seq, d_sae]
            - topk_features: [seq, d_sae]
            - seq_len: int
    """
    device = next(model.parameters()).device
    heads = [head] if isinstance(head, int) else list(head)
    single_head = isinstance(head, int)

    # Tokenise and truncate
    if prepend_bos is not None:
        tokens = model.tokenizer.encode(text, add_special_tokens=prepend_bos)
    else:
        tokens = model.tokenizer.encode(text)
    if max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]

    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    hook_name = f"blocks.{layer}.{hook_point}"
    pattern_hook = f"blocks.{layer}.attn.hook_pattern"

    # Extract both SAE input activations AND attention pattern in one pass
    _, cache = model.run_with_cache(
        tokens_tensor, names_filter=[hook_name, pattern_hook]
    )

    act = cache[hook_name].squeeze(0)  # [seq_len, d_model] or [seq, n_heads, d_head]
    attn_patterns = cache[pattern_hook].squeeze(0)  # [n_heads, seq_len, seq_len]

    if act.dim() == 3:
        act = act.flatten(-2, -1)

    # Encode to SAE features
    if verbose:
        print(f"Encoding {act.shape[0]} positions to SAE features...")

    if hasattr(sae, 'encode'):
        feature_activations = sae.encode(act)
    else:
        feature_activations = sae.sae.encode(act)

    # Handle norm coefficient (Gemma-Scope etc.)
    if hasattr(sae, '_norm_coeff') and sae._norm_coeff is not None:
        feature_activations = feature_activations / sae._norm_coeff

    # Get decoder weights
    if hasattr(sae, 'W_dec'):
        W_dec = sae.W_dec
    else:
        W_dec = sae.sae.W_dec

    # Handle rescale_acts_by_decoder_norm
    if normalize_by_decoder_norm is None:
        inner = sae.sae if hasattr(sae, 'sae') else sae
        cfg = getattr(inner, 'cfg', None)
        do_normalize = getattr(cfg, 'rescale_acts_by_decoder_norm', False) if cfg else False
    else:
        do_normalize = normalize_by_decoder_norm

    dec_norms = W_dec.norm(dim=-1) if do_normalize else None

    # RMSNorm correction
    if "resid" in hook_point:
        b_dec = sae.b_dec
        x_hat = feature_activations @ W_dec + b_dec
        eps = model.cfg.eps
        rms = (x_hat.float().pow(2).mean(dim=-1) + eps).sqrt()
    else:
        rms = None

    # Top-k sparsify
    topk_features = topk_sparsify(feature_activations, top_k).float()

    seq_len = feature_activations.shape[0]

    # Compute OV decomposition per head
    ov_results = {}
    attn_pattern_results = {}

    for h in heads:
        W_V_h = get_W_V(model, layer, h).float()
        W_O_h = get_W_O(model, layer, h).float()
        A_h = attn_patterns[h, :seq_len, :seq_len]  # [seq, seq]

        ov_sparse = compute_ov_sparse(
            topk_features, W_dec.float(), W_V_h, W_O_h, A_h,
            rms=rms,
            dec_norms=dec_norms,
            chunk_size=chunk_size,
            verbose=verbose,
            layer_head_label=f"L{layer}H{h}",
        )

        ov_results[h] = ov_sparse
        attn_pattern_results[h] = A_h.detach().cpu()

    result = {
        "feature_activations": feature_activations,
        "topk_features": topk_features,
        "seq_len": seq_len,
        "normalized": do_normalize,
    }

    if single_head:
        result["ov_sparse"] = ov_results[heads[0]]
        result["attn_pattern"] = attn_pattern_results[heads[0]]
    else:
        result["ov_sparse"] = ov_results
        result["attn_pattern"] = attn_pattern_results

    return result
