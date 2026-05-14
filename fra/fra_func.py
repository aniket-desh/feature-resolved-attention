import math

from transformer_lens import HookedTransformer
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from einops import einsum
from fra.activation_utils import get_llm_activations
from tqdm import tqdm


def _apply_softcap(scores: np.ndarray, softcap: float) -> np.ndarray:
    """Apply tanh soft-capping to attention scores (no-op if softcap==0).

    Gemma-2 applies ``softcap * tanh(scores / softcap)`` to attention logits
    before the causal mask.  TransformerLens fires ``hook_attn_scores`` AFTER
    this transformation, so FRA heatmaps must apply the same transform to
    match actual scores.
    """
    if softcap > 0:
        return softcap * np.tanh(scores / softcap)
    return scores


def _apply_softcap_t(scores: torch.Tensor, softcap: float) -> torch.Tensor:
    """Torch version of softcap for patching hooks."""
    if softcap > 0:
        return softcap * torch.tanh(scores / softcap)
    return scores


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


@torch.no_grad()
def compute_bias_correction(
    model,
    sae,
    layer: int,
    head: int,
    x_hat: torch.Tensor,
    hook_point: str = "hook_resid_pre",
) -> np.ndarray:
    """Compute the [seq, seq] bias correction matrix for FRA score reconstruction.

    The FRA bilinear sum only captures feature×feature interactions (using
    nobias decoder output).  The decoder bias b_dec contributes additional
    linear and constant cross-terms.  This function computes those terms as:

        bias_corr = (q_full @ k_full.T - q_nobias @ k_nobias.T) / attn_scale

    where full projections include b_dec and nobias projections exclude it.
    For RoPE models (Gemma/Llama) this must be a full [seq,seq] matrix
    because the correction is position-pair-dependent.

    Returns numpy array [seq_len, seq_len].
    """
    device = x_hat.device
    x_hat = x_hat.float()

    # Get b_dec
    b_dec = (sae.b_dec if hasattr(sae, "b_dec") else sae.sae.b_dec).float().to(device)

    # Normalization handling:
    # When fold_ln=True, W_Q/W_K already have gamma folded in, so we only
    # need to apply the residual normalization:
    #   RMS/RMSPre: x / sqrt(mean(x^2) + eps)
    #   LNPre:      (x - mean(x)) / sqrt(var(x) + eps)
    norm_type = getattr(model.cfg, "normalization_type", None)
    is_layer_norm = norm_type == "LNPre"
    needs_norm = norm_type in ("RMS", "RMSPre", "LNPre")

    if needs_norm:
        eps = model.cfg.eps
        x_full = x_hat
        x_nobias = x_hat - b_dec
        if is_layer_norm:
            x_full = x_full - x_full.mean(dim=-1, keepdim=True)
            x_nobias = x_nobias - x_nobias.mean(dim=-1, keepdim=True)
        rms = (x_full.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
        x_hat_norm = x_full / rms
        x_hat_nobias_norm = x_nobias / rms
    else:
        x_hat_norm = x_hat
        x_hat_nobias_norm = x_hat - b_dec

    # Get QK weights
    W_Q, W_K, b_Q, b_K = get_qk_weights(model, layer, head)
    W_Q, W_K = W_Q.float(), W_K.float()
    b_Q, b_K = b_Q.float(), b_K.float()
    attn_scale = model.blocks[layer].attn.attn_scale

    # Full projections (with b_dec)
    q_full = x_hat_norm @ W_Q + b_Q
    k_full = x_hat_norm @ W_K + b_K

    # Nobias projections (without b_dec, no attention biases)
    q_nobias = x_hat_nobias_norm @ W_Q
    k_nobias = x_hat_nobias_norm @ W_K

    # Apply RoPE if needed
    rope = get_rope_params(model, layer)
    if rope is not None:
        r_sin, r_cos, r_dim, adj = rope
        q_full = apply_rope_all_positions(q_full, r_sin, r_cos, r_dim, adj)
        k_full = apply_rope_all_positions(k_full, r_sin, r_cos, r_dim, adj)
        q_nobias = apply_rope_all_positions(q_nobias, r_sin, r_cos, r_dim, adj)
        k_nobias = apply_rope_all_positions(k_nobias, r_sin, r_cos, r_dim, adj)

    bias_corr = ((q_full @ k_full.T) - (q_nobias @ k_nobias.T)) / attn_scale
    return bias_corr.cpu().numpy()


def _apply_rope(
        v: torch.Tensor,
        pos: int,
        rotary_sin: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_dim: int,
        adjacent_pairs: bool,
) -> torch.Tensor:
    """
    Apply RoPE rotation to projected vectors at a single position.

    Args:
        v: [n, d_head] tensor (e.g. feature decoder vecs projected through W_Q/W_K)
        pos: sequence position index
        rotary_sin: [n_ctx, rotary_dim] pre-computed sin values
        rotary_cos: [n_ctx, rotary_dim] pre-computed cos values
        rotary_dim: number of dimensions to rotate (may be < d_head)
        adjacent_pairs: True = GPT-J style, False = GPT-NeoX style (Gemma/Llama)

    Returns:
        Rotated [n, d_head] tensor.
    """
    v_rot = v[..., :rotary_dim]
    v_pass = v[..., rotary_dim:]

    # rotate_every_two: maps [x0, ..., xn-1, xn, ..., x2n-1]
    #   adjacent_pairs=True  (GPT-J):  [-x1, x0, -x3, x2, ...]
    #   adjacent_pairs=False (NeoX):   [-xn, ..., -x2n-1, x0, ..., xn-1]
    rot = torch.empty_like(v_rot)
    if adjacent_pairs:
        rot[..., ::2] = -v_rot[..., 1::2]
        rot[..., 1::2] = v_rot[..., ::2]
    else:
        n = v_rot.shape[-1] // 2
        rot[..., :n] = -v_rot[..., n:]
        rot[..., n:] = v_rot[..., :n]

    cos = rotary_cos[pos]  # [rotary_dim]
    sin = rotary_sin[pos]  # [rotary_dim]

    v_rotated = v_rot * cos + rot * sin

    if v_pass.shape[-1] > 0:
        return torch.cat([v_rotated, v_pass], dim=-1)
    return v_rotated


def apply_rope_all_positions(
        v: torch.Tensor,
        rotary_sin: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_dim: int,
        adjacent_pairs: bool,
) -> torch.Tensor:
    """
    Apply RoPE to [seq, d_head], rotating each row by its position index.

    This is the batch (all-positions-at-once) counterpart of _apply_rope.
    Used in validation / ablation where we have full Q/K tensors.
    """
    seq = v.shape[0]
    v_rot = v[:, :rotary_dim]
    v_pass = v[:, rotary_dim:]

    rot = torch.empty_like(v_rot)
    if adjacent_pairs:
        rot[:, ::2] = -v_rot[:, 1::2]
        rot[:, 1::2] = v_rot[:, ::2]
    else:
        n = v_rot.shape[-1] // 2
        rot[:, :n] = -v_rot[:, n:]
        rot[:, n:] = v_rot[:, :n]

    cos = rotary_cos[:seq]  # [seq, rotary_dim]
    sin = rotary_sin[:seq]  # [seq, rotary_dim]

    v_rotated = v_rot * cos + rot * sin

    if v_pass.shape[-1] > 0:
        return torch.cat([v_rotated, v_pass], dim=-1)
    return v_rotated


def get_rope_params(model, layer):
    """
    Extract RoPE parameters from a model if it uses rotary embeddings.

    Returns None if the model doesn't use RoPE (e.g. GPT-2),
    otherwise returns (rotary_sin, rotary_cos, rotary_dim, adjacent_pairs).
    """
    if getattr(model.cfg, 'positional_embedding_type', 'standard') != 'rotary':
        return None
    attn = model.blocks[layer].attn
    return (
        attn.rotary_sin,
        attn.rotary_cos,
        model.cfg.rotary_dim or model.cfg.d_head,
        getattr(model.cfg, 'rotary_adjacent_pairs', False),
    )


def lower_triangular_mask(pattern: np.ndarray) -> np.ma.MaskedArray:
    """Apply lower triangular mask to attention pattern."""
    mask = np.triu(np.ones(pattern.shape), k=1)
    return np.ma.array(np.tril(pattern, k=0), mask=mask)


def attention_pattern_QK(llm: Any, layer: int, head: int, q_input: torch.Tensor,
                         q_do_bias: bool, k_input: torch.Tensor, k_do_bias: bool) -> np.ndarray:
    """
    Compute attention pattern from query and key inputs.

    Args:
        layer: Layer index
        head: Head index
        q_input: Query input tensor
        q_do_bias: Whether to add query bias
        k_input: Key input tensor
        k_do_bias: Whether to add key bias

    Returns:
        Attention scores as numpy array
    """
    W_Q = llm.blocks[layer].attn.W_Q[head]
    b_Q = llm.blocks[layer].attn.b_Q[head]
    W_K = llm.blocks[layer].attn.W_K[head]
    b_K = llm.blocks[layer].attn.b_K[head]

    q = einsum(W_Q, q_input, "d a, s d -> s a")
    if q_do_bias:
        q += b_Q

    k = einsum(W_K, k_input, "d a, s d -> s a")
    if k_do_bias:
        k += b_K

    attention_scores = einsum(q, k, "q a, k a -> q k")

    return attention_scores.detach().cpu().numpy()


def analyze_feature_attention_interactions(model: Any, sae: Any, layer: int, head: int,
                                           input_text: str, query_position: int, key_position: int,
                                           hook_point: str = "hook_attn_out") -> Dict:
    """
    Analyze interactions between features in attention.

    Args:
        layer: Layer index
        head: Head index
        input_text: Input text to analyze
        query_position: Query position
        key_position: Key position
    """
    activations_SD = get_llm_activations(model, input_text, hook_point=hook_point, layers=layer)
    feature_activations_SH = sae.encode(activations_SD)

    feature_activations_query = feature_activations_SH[query_position]
    query_active_features = torch.where(feature_activations_query != 0)[0]

    feature_activations_key = feature_activations_SH[key_position]
    key_active_features = torch.where(feature_activations_key != 0)[0]

    query_activations_for_features = sae.W_dec[query_active_features]
    key_activations_for_features = sae.W_dec[key_active_features]

    interaction_matrix_unscaled = attention_pattern_QK(model, layer, head,
                                                       query_activations_for_features, False,
                                                       key_activations_for_features, False)

    # Convert to numpy after using for indexing
    query_features_tensor = query_active_features
    key_features_tensor = key_active_features

    matrix_scaling = feature_activations_query[query_features_tensor].unsqueeze(1) * \
                     feature_activations_key[key_features_tensor].unsqueeze(0)
    matrix_scaling = matrix_scaling.detach().cpu().numpy()

    query_active_features = query_features_tensor.cpu().numpy()
    key_active_features = key_features_tensor.cpu().numpy()

    # if self.feature_activations_active_mean is not None:
    #     interaction_matrix_unscaled *= self.feature_activations_active_mean[query_active_features][:, np.newaxis]
    #     interaction_matrix_unscaled *= self.feature_activations_active_mean[key_active_features][np.newaxis, :]
    #     matrix_scaling /= self.feature_activations_active_mean[query_active_features][:, np.newaxis]
    #     matrix_scaling /= self.feature_activations_active_mean[key_active_features][np.newaxis, :]

    return {
        'query_active_features': query_active_features,
        'key_active_features': key_active_features,
        'interaction_matrix_unscaled': interaction_matrix_unscaled,
        'matrix_scaling': matrix_scaling,
        'interaction_matrix': interaction_matrix_unscaled * matrix_scaling
    }


def get_sentence_averages(llm: Any, sae: Any, layer: int, head: int, input_text: str, hook_point: str = "attn.hook_z"):
    text_length = 128
    hidden_dim = sae.d_sae
    data_dep_int_matrix = np.zeros((hidden_dim, hidden_dim))
    data_dep_int_matrix_abs = np.zeros((hidden_dim, hidden_dim))
    data_dep_localization_matrix = np.zeros((hidden_dim, hidden_dim))
    count = 0
    for key_index in tqdm(range(text_length), disable=True):
        for query_index in range(key_index, text_length):
            feature_analysis = analyze_feature_attention_interactions(llm, sae, layer, head, input_text, query_index,
                                                                      key_index, hook_point)
            int_matrix = feature_analysis["interaction_matrix"]
            query_active_features = feature_analysis["query_active_features"]
            key_active_features = feature_analysis["key_active_features"]
            data_independent = feature_analysis["interaction_matrix_unscaled"]

            resized_data_dependent_int = np.zeros((hidden_dim, hidden_dim))
            resized_data_dependent_int[query_active_features[:, None], key_active_features[None, :]] = int_matrix

            resized_data_dependent_localization = np.zeros((hidden_dim, hidden_dim))
            resized_data_dependent_localization[query_active_features[:, None], key_active_features[None, :]] = np.abs(
                int_matrix) * (query_index - key_index)

            data_dep_int_matrix = data_dep_int_matrix + resized_data_dependent_int
            data_dep_int_matrix_abs = data_dep_int_matrix_abs + np.abs(resized_data_dependent_int)
            data_dep_localization_matrix = data_dep_localization_matrix + resized_data_dependent_localization

            count += 1

    data_dep_int_matrix /= count
    data_dep_localization_matrix = data_dep_localization_matrix / np.clip(data_dep_int_matrix_abs, a_min=1, a_max=None)
    data_dep_int_matrix_abs /= count

    return data_dep_int_matrix, data_dep_int_matrix_abs, data_dep_localization_matrix


def _compute_fra_for_head(
        *,
        model: HookedTransformer,
        layer: int,
        head: int,
        seq_len: int,
        d_sae: int,
        topk_features: torch.Tensor,
        W_dec_corr: torch.Tensor,
        rms: Optional[torch.Tensor],
        dec_norms: Optional[torch.Tensor],
        use_rope: bool,
        rope_params: Optional[Tuple],
        chunk_size: int,
        device,
        verbose: bool = False,
) -> Dict[str, Any]:
    """Compute the sparse 4D FRA tensor for a single attention head.

    This is the inner loop extracted so it can be called once per head while
    sharing all head-independent pre-computation (encoding, top-k, norms).

    Args:
        model: The transformer model (for W_Q / W_K extraction).
        layer: Attention layer index.
        head: Attention head index.
        seq_len: Sequence length.
        d_sae: SAE dictionary size.
        topk_features: [seq_len, d_sae] sparsified feature activations.
        W_dec_corr: [d_sae, d_model] decoder weights (LN-corrected if needed).
        rms: Optional [seq_len] per-position RMS values for normalization.
        dec_norms: Optional [d_sae] decoder norms for rescale correction.
        use_rope: Whether model uses rotary embeddings.
        rope_params: (rotary_sin, rotary_cos, rotary_dim, adjacent_pairs) or None.
        chunk_size: Query positions per GPU batch.
        device: Torch device.
        verbose: Show progress bar.

    Returns:
        Dict with fra_tensor_sparse, shape, seq_len, total_interactions.
    """
    # Get attention weights for this head — handle GQA
    W_Q = model.blocks[layer].attn.W_Q[head].float()
    n_kv = model.blocks[layer].attn.W_K.shape[0]
    n_q = model.blocks[layer].attn.W_Q.shape[0]
    kv_head = head * n_kv // n_q
    W_K = model.blocks[layer].attn.W_K[kv_head].float()

    d_head = W_Q.shape[-1]
    attn_scale = math.sqrt(d_head)

    if use_rope:
        rotary_sin, rotary_cos, rotary_dim, adjacent_pairs = rope_params

    all_indices_cpu: list[torch.Tensor] = []
    all_values_cpu: list[torch.Tensor] = []

    total_pairs = seq_len * (seq_len + 1) // 2
    if verbose:
        pbar = tqdm(total=total_pairs, desc=f"Computing 4D FRA (L{layer}H{head})")

    for q_start in range(0, seq_len, chunk_size):
        q_end = min(q_start + chunk_size, seq_len)
        chunk_indices: list[torch.Tensor] = []
        chunk_values: list[torch.Tensor] = []

        for query_idx in range(q_start, q_end):
            q_feat = topk_features[query_idx]
            q_active = torch.where(q_feat != 0)[0]

            if len(q_active) == 0:
                if verbose:
                    pbar.update(query_idx + 1)
                continue

            q_vecs = W_dec_corr[q_active]
            q_proj = q_vecs @ W_Q
            q_scales = q_feat[q_active]
            if dec_norms is not None:
                q_scales = q_scales / dec_norms[q_active]

            if use_rope:
                q_proj = _apply_rope(q_proj, query_idx, rotary_sin, rotary_cos,
                                     rotary_dim, adjacent_pairs)

            for key_idx in range(query_idx + 1):
                k_feat = topk_features[key_idx]
                k_active = torch.where(k_feat != 0)[0]

                if len(k_active) == 0:
                    if verbose:
                        pbar.update(1)
                    continue

                k_vecs = W_dec_corr[k_active]
                k_proj = k_vecs @ W_K

                if use_rope:
                    k_proj = _apply_rope(k_proj, key_idx, rotary_sin, rotary_cos,
                                         rotary_dim, adjacent_pairs)

                k_scales = k_feat[k_active]
                if dec_norms is not None:
                    k_scales = k_scales / dec_norms[k_active]

                int_matrix = q_proj @ k_proj.T
                int_matrix = int_matrix * q_scales.unsqueeze(1) * k_scales.unsqueeze(0)

                if rms is not None:
                    int_matrix = int_matrix / (rms[query_idx] * rms[key_idx])

                mask = int_matrix.abs() > 1e-10
                if mask.any():
                    local_r, local_c = torch.where(mask)
                    n_int = len(local_r)

                    pos_indices = torch.empty((4, n_int), dtype=torch.long)
                    pos_indices[0] = query_idx
                    pos_indices[1] = key_idx
                    pos_indices[2] = q_active[local_r].cpu()
                    pos_indices[3] = k_active[local_c].cpu()

                    chunk_indices.append(pos_indices)
                    chunk_values.append(int_matrix[mask].detach().cpu().float())

                if verbose:
                    pbar.update(1)

        all_indices_cpu.extend(chunk_indices)
        all_values_cpu.extend(chunk_values)
        device_str = device.type if hasattr(device, 'type') else str(device)
        if device_str != "cpu":
            torch.cuda.empty_cache()

    if verbose:
        pbar.close()

    shape = (seq_len, seq_len, d_sae, d_sae)
    if len(all_indices_cpu) > 0:
        indices_cpu = torch.cat(all_indices_cpu, dim=1)
        values_cpu = torch.cat(all_values_cpu)

        fra_tensor_sparse = torch.sparse_coo_tensor(
            indices_cpu, values_cpu,
            size=shape, device="cpu", dtype=torch.float32,
        ).coalesce()

        if str(device) != "cpu":
            try:
                fra_tensor_sparse = fra_tensor_sparse.to(device)
            except RuntimeError:
                if verbose:
                    print("Warning: sparse tensor too large for GPU, keeping on CPU.")

        total_interactions = fra_tensor_sparse._nnz()
    else:
        fra_tensor_sparse = torch.sparse_coo_tensor(
            torch.zeros((4, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            size=shape, device="cpu",
        )
        total_interactions = 0

    return {
        'fra_tensor_sparse': fra_tensor_sparse,
        'shape': shape,
        'seq_len': seq_len,
        'total_interactions': total_interactions,
    }


@torch.no_grad()
def get_sentence_fra_batch(
        model: HookedTransformer,
        sae: Any,
        text: str,
        layer: int,
        head: Union[int, List[int]],
        max_length: int = 128,
        top_k: int = 20,
        verbose: bool = False,
        hook_point: str = "ln1.hook_normalized",
        chunk_size: int = 16,
        normalize_by_decoder_norm: bool | None = None,
        prepend_bos: bool | None = None,
) -> Dict[str, Any]:
    """
    Compute full 4D Feature-Resolved Attention tensor for a sentence.
    Returns a sparse representation to avoid memory issues.

    Supports single-head (backward compatible) and multi-head mode.  In
    multi-head mode the expensive encoding / top-k / normalization work is
    done once and reused across all heads.

    hook_point controls which activation the SAE was trained on:
      - "ln1.hook_normalized"  (default, correct for FRA): decoder vectors live in
        the same d_model space that W_Q / W_K project from.  This is the only
        mathematically correct choice.
      - "attn.hook_z"          (legacy, for pre-trained hook_z SAEs): decoder
        vectors are in concatenated-heads space, not d_model space; the QK
        attention score computation is therefore approximate.

    Args:
        model: The transformer model
        sae: The SAE (any object with .encode() and .W_dec attributes)
        text: Input text to analyze
        layer: Which layer to analyze
        head: Attention head index (int) or list of head indices.
        max_length: Maximum sequence length
        top_k: Number of top features to keep per position
        verbose: Whether to show progress
        hook_point: Hookpoint the SAE was trained on (relative to blocks.{layer}.)
        chunk_size: Number of query positions to process per GPU batch before
                    flushing results to CPU.  Reduce for large SAEs (e.g. Gemma-Scope)
                    to avoid GPU OOM.  Set to seq_len to process everything at once.
        normalize_by_decoder_norm: Whether to divide feature activations by
                    decoder weight norms to match SAEs trained with
                    rescale_acts_by_decoder_norm=True.  None (default) auto-detects
                    from the SAE config.  True/False forces the behaviour.
        prepend_bos: Whether to include special tokens (BOS) in tokenization.
                    None (default) uses the tokenizer's default behaviour.
                    True/False forces add_special_tokens on/off.

    Returns:
        When head is an int — dict with:
            fra_tensor_sparse, shape, seq_len, total_interactions, normalized.

        When head is a list — dict with:
            fra_sparse_dict (dict[int, sparse_coo_tensor]),
            feature_activations, topk_features, seq_len, normalized.
    """
    device = next(model.parameters()).device

    single_head = isinstance(head, int)
    heads = [head] if single_head else list(head)

    # ── Head-independent pre-computation (done once) ──────────────────

    # Tokenise and truncate
    if prepend_bos is not None:
        tokens = model.tokenizer.encode(text, add_special_tokens=prepend_bos)
    else:
        tokens = model.tokenizer.encode(text)
    if max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]

    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    hook_name = f"blocks.{layer}.{hook_point}"
    _, cache = model.run_with_cache(tokens_tensor, names_filter=[hook_name])

    act = cache[hook_name].squeeze(0)  # remove batch dim
    seq_len = act.shape[0]

    # hook_z is [seq_len, n_heads, d_head] → flatten to [seq_len, n_heads*d_head]
    # ln1.hook_normalized is already [seq_len, d_model]
    if act.dim() == 3:
        act = act.flatten(-2, -1)

    # Encode to SAE features
    if verbose:
        print(f"Encoding {seq_len} positions to SAE features...")

    if hasattr(sae, 'encode'):
        feature_activations = sae.encode(act)  # [seq_len, d_sae]
    else:
        feature_activations = sae.sae.encode(act)

    # If the SAE normalizes inputs (e.g. Gemma-Scope), the feature activations
    # are in the normalized scale.  Divide by the norm coefficient so that
    # FRA[q,k,i,j] sums to the actual (un-normalized) QK attention score.
    if hasattr(sae, '_norm_coeff') and sae._norm_coeff is not None:
        feature_activations = feature_activations / sae._norm_coeff

    # Cast to float32 for accumulation precision
    feature_activations = feature_activations.float()

    d_sae = feature_activations.shape[-1]

    # Keep only top-k features per position
    topk_features = []
    for pos in range(seq_len):
        feat = feature_activations[pos]
        active_mask = feat != 0
        n_active = active_mask.sum().item()

        if n_active > 0:
            k = min(top_k, n_active)
            topk_vals, topk_idx = torch.topk(feat.abs(), k)
            sparse_feat = torch.zeros_like(feat)
            sparse_feat[topk_idx] = feat[topk_idx]
        else:
            sparse_feat = torch.zeros_like(feat)

        topk_features.append(sparse_feat)

    topk_features = torch.stack(topk_features)  # [seq_len, d_sae]

    # Detect RoPE — needed for Gemma, Llama, etc. (not GPT-2)
    use_rope = getattr(model.cfg, 'positional_embedding_type', 'standard') == 'rotary'
    rope_params = None
    if use_rope:
        attn_module = model.blocks[layer].attn
        rope_params = (
            attn_module.rotary_sin,
            attn_module.rotary_cos,
            model.cfg.rotary_dim or model.cfg.d_head,
            getattr(model.cfg, 'rotary_adjacent_pairs', False),
        )
        if verbose:
            print(f"RoPE enabled: rotary_dim={rope_params[2]}, adjacent_pairs={rope_params[3]}")

    # Get decoder weights — float32 for precision
    if hasattr(sae, 'W_dec'):
        W_dec = sae.W_dec.float()  # [d_sae, d_model]
    else:
        W_dec = sae.sae.W_dec.float()

    # ── Normalization correction (LayerNorm / RMSNorm) ────────────────
    norm_type = getattr(model.cfg, "normalization_type", None)
    is_layer_norm = norm_type == "LNPre"
    needs_norm = norm_type in ("RMS", "RMSPre", "LNPre")

    W_dec_corr = W_dec
    if is_layer_norm and "resid" in hook_point:
        W_dec_corr = W_dec - W_dec.mean(dim=-1, keepdim=True)
        if verbose:
            print("Applying LayerNorm mean-centering to decoder vectors")

    # Compute per-position RMS denominator from the SAE reconstruction
    rms = None
    if needs_norm and "resid" not in hook_point:
        pass
    elif needs_norm:
        b_dec = (sae.b_dec if hasattr(sae, "b_dec") else sae.sae.b_dec).float().to(device)
        rms_activations = (topk_features @ W_dec + b_dec).float()
        if is_layer_norm:
            rms_activations = rms_activations - rms_activations.mean(dim=-1, keepdim=True)
        eps = model.cfg.eps
        rms = (rms_activations.pow(2).mean(dim=-1) + eps).sqrt()  # [seq_len]
        if verbose:
            print(f"Applying {norm_type} per-position normalization")

    # Handle rescale_acts_by_decoder_norm
    if normalize_by_decoder_norm is None:
        inner = sae.sae if hasattr(sae, 'sae') else sae
        cfg = getattr(inner, 'cfg', None)
        do_normalize = getattr(cfg, 'rescale_acts_by_decoder_norm', False) if cfg else False
    else:
        do_normalize = normalize_by_decoder_norm

    if do_normalize:
        dec_norms = W_dec.norm(dim=-1)  # [d_sae]
        if verbose:
            print("Applying decoder-norm correction (rescale_acts_by_decoder_norm)")
    else:
        dec_norms = None

    # ── Per-head FRA computation ──────────────────────────────────────

    fra_sparse_dict: Dict[int, Any] = {}
    last_result = None

    for h in heads:
        if verbose and len(heads) > 1:
            print(f"\n── Head {h} ──")

        result = _compute_fra_for_head(
            model=model, layer=layer, head=h,
            seq_len=seq_len, d_sae=d_sae,
            topk_features=topk_features,
            W_dec_corr=W_dec_corr,
            rms=rms, dec_norms=dec_norms,
            use_rope=use_rope, rope_params=rope_params,
            chunk_size=chunk_size, device=device,
            verbose=verbose,
        )
        fra_sparse_dict[h] = result['fra_tensor_sparse']

        if verbose:
            nnz = result['total_interactions']
            shape = result['shape']
            density = nnz / max(seq_len * seq_len * top_k * top_k, 1)
            print(f"4D FRA tensor: shape={shape}, nnz={nnz:,}, density={density:.2%}")

        last_result = result

    # ── Return ────────────────────────────────────────────────────────

    if single_head:
        last_result['normalized'] = do_normalize
        return last_result

    return {
        'fra_sparse_dict': fra_sparse_dict,
        'feature_activations': feature_activations,
        'topk_features': topk_features,
        'seq_len': seq_len,
        'normalized': do_normalize,
    }


if __name__ == "__main__":
    print('main character')