"""
Shared helpers for FRA analysis.

Weight extraction, reconstruction metrics, pair ranking, FRA
aggregation, RoPE, and attention-score projection utilities used
across dashboard, analysis, and viz modules.

FRA-specific computation (``topk_sparsify``, ``compute_fra_sparse``)
lives in ``fra.core.fra``.
"""

import math
from collections import defaultdict

import numpy as np
import torch
from transformer_lens import HookedTransformer


# ── Weight extraction (GQA-aware) ────────────────────────────────────────


def get_W_K(model: HookedTransformer, layer: int, head: int) -> torch.Tensor:
    """Index W_K correctly for both MHA and GQA models.

    TransformerLens may or may not expand KV heads to match Q heads.
    This helper handles both cases.
    """
    W_K = model.blocks[layer].attn.W_K
    if W_K.shape[0] == model.cfg.n_heads:
        return W_K[head]
    n_kv = W_K.shape[0]
    heads_per_kv = model.cfg.n_heads // n_kv
    return W_K[head // heads_per_kv]


def get_W_V(model: HookedTransformer, layer: int, head: int) -> torch.Tensor:
    """Index W_V correctly for both MHA and GQA models.

    Same GQA mapping as get_W_K — value heads share the KV head count.
    """
    W_V = model.blocks[layer].attn.W_V
    if W_V.shape[0] == model.cfg.n_heads:
        return W_V[head]
    n_kv = W_V.shape[0]
    heads_per_kv = model.cfg.n_heads // n_kv
    return W_V[head // heads_per_kv]


def get_W_O(model: HookedTransformer, layer: int, head: int) -> torch.Tensor:
    """Get W_O for a specific head.

    W_O is indexed by query head (always n_heads of them, no GQA mapping).
    TransformerLens stores W_O as [n_heads, d_head, d_model].
    Returns: [d_head, d_model]
    """
    return model.blocks[layer].attn.W_O[head]


def get_qk_weights(model: HookedTransformer, layer: int, head: int):
    """Get (W_Q, W_K, b_Q, b_K) for a query head, handling GQA correctly."""
    W_Q = model.blocks[layer].attn.W_Q[head]
    b_Q = model.blocks[layer].attn.b_Q[head]
    n_kv = model.blocks[layer].attn.W_K.shape[0]
    n_q = model.blocks[layer].attn.W_Q.shape[0]
    kv_head = head * n_kv // n_q
    W_K = model.blocks[layer].attn.W_K[kv_head]
    b_K = model.blocks[layer].attn.b_K[kv_head]
    return W_Q, W_K, b_Q, b_K


def get_attn_scale(model: HookedTransformer, layer: int) -> float:
    """Return sqrt(d_head) for the given layer."""
    d_head = model.blocks[layer].attn.W_Q.shape[-1]
    return math.sqrt(d_head)


# ── RoPE utilities ───────────────────────────────────────────────────────


def apply_rope_to_projected(
    projected: torch.Tensor,
    position: int,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    rotary_dim: int,
    rotary_adjacent_pairs: bool = False,
) -> torch.Tensor:
    """Apply RoPE rotation to projected vectors at a single position.

    This mirrors TransformerLens's ``apply_rotary`` + ``rotate_every_two``
    logic but operates on a ``[n_features, d_head]`` tensor at a single
    sequence position instead of the full ``[batch, pos, head, d_head]``
    tensor.

    Args:
        projected: ``[n_features, d_head]`` -- features projected through
            W_Q or W_K.
        position: Integer sequence position for looking up sin/cos.
        rope_sin: ``[n_ctx, rotary_dim]`` precomputed sine table.
        rope_cos: ``[n_ctx, rotary_dim]`` precomputed cosine table.
        rotary_dim: Number of dimensions to rotate (may be < d_head).
        rotary_adjacent_pairs: True for GPT-J style, False for GPT-NeoX
            style (Gemma, Llama).
    """
    x_rot = projected[:, :rotary_dim]
    x_pass = projected[:, rotary_dim:]

    # rotate_every_two
    x_flip = x_rot.clone()
    if rotary_adjacent_pairs:
        x_flip[:, ::2] = -x_rot[:, 1::2]
        x_flip[:, 1::2] = x_rot[:, ::2]
    else:
        n = rotary_dim // 2
        x_flip[:, :n] = -x_rot[:, n:]
        x_flip[:, n:] = x_rot[:, :n]

    cos = rope_cos[position]  # [rotary_dim]
    sin = rope_sin[position]  # [rotary_dim]
    x_rotated = x_rot * cos.unsqueeze(0) + x_flip * sin.unsqueeze(0)

    return torch.cat([x_rotated, x_pass], dim=-1)


def _extract_rope_params(model, layer):
    """Extract RoPE parameters from a model's attention block.

    Returns (rope_sin, rope_cos, rotary_dim, rotary_adjacent_pairs).
    All None/False when the model does not use rotary embeddings.
    """
    if getattr(model.cfg, "positional_embedding_type", None) != "rotary":
        return None, None, None, False
    attn_block = model.blocks[layer].attn
    return (
        attn_block.rotary_sin,
        attn_block.rotary_cos,
        model.cfg.rotary_dim,
        getattr(model.cfg, "rotary_adjacent_pairs", False),
    )


# ── Attention score projection ───────────────────────────────────────────


@torch.no_grad()
def project_qk(
    model: HookedTransformer,
    layer: int,
    head: int,
    x_hat: torch.Tensor,
    b_dec: torch.Tensor,
    needs_rms: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project coder reconstruction through W_Q/W_K with RMSNorm and RoPE.

    Returns ``(q_full, k_full, q_nobias, k_nobias)`` each ``[seq, d_head]``
    in float32.

    - ``q_full / k_full``: full reconstruction (with b_dec and b_Q/b_K)
    - ``q_nobias / k_nobias``: feature-only part (b_dec subtracted, no b_Q/b_K)

    When ``needs_rms=True``, both x_hat and (x_hat - b_dec) are divided by
    the **same** RMS denominator (computed from x_hat), matching the FRA
    computation in ``compute_fra_sparse``.

    Args:
        model: HookedTransformer with fold_ln=True.
        layer: Attention layer index.
        head: Attention head index.
        x_hat: ``[seq, d_model]`` coder reconstruction (features + b_dec).
        b_dec: ``[d_model]`` decoder bias.
        needs_rms: Apply RMSNorm correction (required when the coder
            decodes into residual-stream space).
    """
    x_hat = x_hat.float()
    b_dec = b_dec.float()

    if needs_rms:
        eps = model.cfg.eps
        rms = (x_hat.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
        x_hat_norm = x_hat / rms
        x_hat_nobias_norm = (x_hat - b_dec) / rms
    else:
        x_hat_norm = x_hat
        x_hat_nobias_norm = x_hat - b_dec

    W_Q, W_K, b_Q, b_K = get_qk_weights(model, layer, head)
    W_Q, W_K, b_Q, b_K = W_Q.float(), W_K.float(), b_Q.float(), b_K.float()

    q_full = x_hat_norm @ W_Q + b_Q
    k_full = x_hat_norm @ W_K + b_K
    q_nobias = x_hat_nobias_norm @ W_Q
    k_nobias = x_hat_nobias_norm @ W_K

    # Apply RoPE at each position
    rope_sin, rope_cos, rotary_dim, rotary_adjacent_pairs = (
        _extract_rope_params(model, layer)
    )
    if rope_sin is not None:
        seq_len = x_hat.shape[0]
        for pos in range(seq_len):
            q_full[pos] = apply_rope_to_projected(
                q_full[pos].unsqueeze(0), pos, rope_sin, rope_cos,
                rotary_dim, rotary_adjacent_pairs,
            ).squeeze(0)
            k_full[pos] = apply_rope_to_projected(
                k_full[pos].unsqueeze(0), pos, rope_sin, rope_cos,
                rotary_dim, rotary_adjacent_pairs,
            ).squeeze(0)
            q_nobias[pos] = apply_rope_to_projected(
                q_nobias[pos].unsqueeze(0), pos, rope_sin, rope_cos,
                rotary_dim, rotary_adjacent_pairs,
            ).squeeze(0)
            k_nobias[pos] = apply_rope_to_projected(
                k_nobias[pos].unsqueeze(0), pos, rope_sin, rope_cos,
                rotary_dim, rotary_adjacent_pairs,
            ).squeeze(0)

    return q_full, k_full, q_nobias, k_nobias


def compute_bias_correction(
    q_full: torch.Tensor,
    k_full: torch.Tensor,
    q_nobias: torch.Tensor,
    k_nobias: torch.Tensor,
    attn_scale: float,
) -> torch.Tensor:
    """Compute the bias correction matrix for FRA score reconstruction.

    Returns ``[seq, seq]`` tensor:
    ``((q_full @ k_full.T) - (q_nobias @ k_nobias.T)) / attn_scale``

    This captures the linear (feature x b_dec) and constant (b_dec x b_dec)
    terms from the attention score decomposition. It must be a full matrix
    (not separable into per-query + per-key terms) because RoPE makes the
    correction position-pair-dependent.
    """
    return ((q_full @ k_full.T) - (q_nobias @ k_nobias.T)) / attn_scale


# ── FRA aggregation ──────────────────────────────────────────────────────


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


# ── Reconstruction metrics ───────────────────────────────────────────────


def compute_errors(actual: np.ndarray, reconstructed: np.ndarray,
                   eps: float = 1e-3, exclude_bos: bool = False) -> dict:
    """Compute reconstruction error metrics with small-denominator handling.

    Args:
        exclude_bos: When True, exclude position 0 (BOS token) from metrics.
            For 2-D attention matrices this removes both row 0 and column 0.
            For 1-D vectors this removes element 0.
    """
    if exclude_bos:
        if actual.ndim >= 2 and actual.shape[0] == actual.shape[1]:
            # Square matrix (attention scores): remove BOS row and column
            actual = actual[1:, 1:]
            reconstructed = reconstructed[1:, 1:]
        else:
            # Activation matrix [seq, d_model] or 1-D: remove position 0 only
            actual = actual[1:]
            reconstructed = reconstructed[1:]

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


def print_errors(label: str, errs: dict) -> None:
    """Pretty-print error metrics."""
    print(f"  {label}:")
    print(f"    Frobenius rel error : {errs['fro_rel_err']:.4f} ({errs['fro_rel_err']*100:.1f}%)")
    print(f"    Cosine similarity   : {errs['cosine_sim']:.6f}")
    print(f"    R-squared           : {errs['r_squared']:.6f}")
    print(f"    Mean elem rel error : {errs['mean_rel_err']:.4f}  (median: {errs['median_rel_err']:.4f})")
    print(f"    Mean abs error      : {errs['mean_abs_err']:.6f}")


# ── Pair ranking ─────────────────────────────────────────────────────────


def aggregate_pairs(indices_np, values_np, diagonal=None):
    """Aggregate sparse 4D FRA entries by (q_feat, k_feat) pair.

    Args:
        indices_np: [4, nnz] array (q_pos, k_pos, q_feat, k_feat)
        values_np: [nnz] array
        diagonal: None=all pairs, True=only i==j, False=only i!=j

    Returns:
        List of (q_feat, k_feat, sum_abs, count, max_abs) tuples.
    """
    q_feats = indices_np[2, :]
    k_feats = indices_np[3, :]
    abs_vals = np.abs(values_np)

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
        pair_sum[(int(q), int(k))] += float(v)
        pair_count[(int(q), int(k))] += 1
        if float(v) > pair_max[(int(q), int(k))]:
            pair_max[(int(q), int(k))] = float(v)

    return [
        (q, k, pair_sum[(q, k)], pair_count[(q, k)], pair_max[(q, k)])
        for (q, k) in pair_sum
    ]


def rank_pairs(indices_np, values_np, top_k=50, diagonal=None, mode="sum"):
    """Return top-k (q_feat, k_feat) pairs ranked by aggregation mode.

    Modes:
        sum  -- total absolute interaction strength
        avg  -- mean absolute interaction per position-pair occurrence
        max  -- single strongest position-pair interaction

    Args:
        indices_np: [4, nnz] array (q_pos, k_pos, q_feat, k_feat)
        values_np: [nnz] array
        top_k: Number of pairs to return.
        diagonal: None=all pairs, True=only i==j, False=only i!=j
        mode: Ranking criterion ("sum", "avg", or "max").

    Returns:
        List of (q_feat, k_feat, sum_abs, count, max_abs) tuples,
        sorted descending by *mode*.
    """
    pairs = aggregate_pairs(indices_np, values_np, diagonal=diagonal)
    if mode == "sum":
        pairs.sort(key=lambda x: x[2], reverse=True)
    elif mode == "avg":
        pairs.sort(key=lambda x: x[2] / max(x[3], 1), reverse=True)
    elif mode == "max":
        pairs.sort(key=lambda x: x[4], reverse=True)
    return pairs[:top_k]


def get_position_heatmap(indices_np, values_np, q_feat, k_feat, seq_len):
    """Extract [seq_len, seq_len] heatmap for a specific (q_feat, k_feat) pair."""
    mask = (indices_np[2] == q_feat) & (indices_np[3] == k_feat)
    q_pos = indices_np[0, mask]
    k_pos = indices_np[1, mask]
    vals = np.abs(values_np[mask])

    mat = np.zeros((seq_len, seq_len))
    for qp, kp, v in zip(q_pos, k_pos, vals):
        mat[qp, kp] += v
    return mat
