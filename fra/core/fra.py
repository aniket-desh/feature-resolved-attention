"""
Feature-Resolved Attention (FRA) computation.

Provides two entry points for computing 4-D FRA tensors:
  - ``get_sentence_fra_batch``: single-model SAE path
  - ``get_sentence_fra_crosscoder``: two-model crosscoder path

Both share the core tensor construction via ``_build_fra_result()``.
"""

import math

import numpy as np
import torch
from typing import Any, Dict
from einops import einsum
from tqdm import tqdm
from transformer_lens import HookedTransformer

from fra.core.activations import get_llm_activations
from fra.core.helpers import get_W_K, apply_rope_to_projected, _extract_rope_params


# ── Top-k sparsification ─────────────────────────────────────────────────


def topk_sparsify(
    feature_activations: torch.Tensor,
    top_k: int | None,
) -> torch.Tensor:
    """Keep only top-k features per position by absolute activation magnitude.

    Args:
        feature_activations: [seq_len, d_sae] tensor of feature activations.
        top_k: Number of features to keep per position.  None keeps all
               active (non-zero) features without truncation.

    Returns:
        [seq_len, d_sae] tensor with at most top_k non-zero entries per row.
    """
    if top_k is None:
        return feature_activations

    seq_len = feature_activations.shape[0]
    topk_features = []
    for pos in range(seq_len):
        feat = feature_activations[pos]
        n_active = (feat != 0).sum().item()

        if n_active > 0:
            k = min(top_k, n_active)
            _, topk_idx = torch.topk(feat.abs(), k)
            sparse_feat = torch.zeros_like(feat)
            sparse_feat[topk_idx] = feat[topk_idx]
        else:
            sparse_feat = torch.zeros_like(feat)

        topk_features.append(sparse_feat)

    return torch.stack(topk_features)


# ── Core FRA loop ───────────────────────────────────────────────────────


def compute_fra_sparse(
    topk_features: torch.Tensor,
    W_dec: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    attn_scale: float,
    *,
    rms: torch.Tensor | None = None,
    dec_norms: torch.Tensor | None = None,
    rope_sin: torch.Tensor | None = None,
    rope_cos: torch.Tensor | None = None,
    rotary_dim: int | None = None,
    rotary_adjacent_pairs: bool = False,
    chunk_size: int = 16,
    verbose: bool = False,
    layer_head_label: str = "",
) -> torch.sparse_coo_tensor:
    """Compute the 4-D FRA sparse tensor from pre-computed feature activations.

    All model-specific setup (activation extraction, encoding, weight lookup)
    happens in the caller; this function is the shared inner loop.

    Args:
        topk_features: ``[seq_len, d_sae]`` already top-k sparsified.
        W_dec:  ``[d_sae, d_model]`` decoder weight matrix.
        W_Q:    ``[d_model, d_head]`` query projection for one head.
        W_K:    ``[d_model, d_head]`` key projection for one head.
        attn_scale: ``sqrt(d_head)`` scaling factor.
        rms:    Optional ``[seq_len]`` per-position RMS values for RMSNorm
                correction.  When provided, each interaction is divided by
                ``rms[query_idx] * rms[key_idx]``.
        dec_norms: Optional ``[d_sae]`` decoder-weight norms for SAEs
                trained with ``rescale_acts_by_decoder_norm=True``.
        rope_sin: Optional ``[n_ctx, rotary_dim]`` precomputed sine table
                from the attention block.  Pass for RoPE models (Gemma, Llama).
        rope_cos: Optional ``[n_ctx, rotary_dim]`` precomputed cosine table.
        rotary_dim: Number of RoPE dimensions (``model.cfg.rotary_dim``).
        rotary_adjacent_pairs: RoPE pair ordering flag from model config.
        chunk_size: Number of query positions per GPU batch before flushing
                to CPU.  Controls peak GPU memory.
        verbose: Show progress bar.
        layer_head_label: Label for progress bar (e.g. ``"L5H3"``).

    Returns:
        ``torch.sparse_coo_tensor`` on CPU, shape
        ``[seq_len, seq_len, d_sae, d_sae]``, coalesced.
    """
    seq_len = topk_features.shape[0]
    d_sae = topk_features.shape[1]
    shape = (seq_len, seq_len, d_sae, d_sae)

    all_indices_cpu: list[torch.Tensor] = []
    all_values_cpu: list[torch.Tensor] = []

    total_pairs = seq_len * (seq_len + 1) // 2
    pbar = tqdm(
        total=total_pairs,
        desc=f"Computing FRA ({layer_head_label})" if layer_head_label else "Computing FRA",
        disable=not verbose,
    )

    for q_start in range(0, seq_len, chunk_size):
        q_end = min(q_start + chunk_size, seq_len)
        chunk_indices: list[torch.Tensor] = []
        chunk_values: list[torch.Tensor] = []

        for query_idx in range(q_start, q_end):
            q_feat = topk_features[query_idx]
            q_active = torch.where(q_feat != 0)[0]

            if len(q_active) == 0:
                pbar.update(query_idx + 1)
                continue

            q_vecs = W_dec[q_active]                          # [n_q, d_model]
            q_proj = q_vecs @ W_Q                             # [n_q, d_head]
            q_scales = q_feat[q_active]                       # [n_q]
            if dec_norms is not None:
                q_scales = q_scales / dec_norms[q_active]

            # Apply RoPE to query once per query position (outside key loop)
            if rope_sin is not None:
                q_proj = apply_rope_to_projected(
                    q_proj, query_idx, rope_sin, rope_cos,
                    rotary_dim, rotary_adjacent_pairs,
                )

            for key_idx in range(query_idx + 1):
                k_feat = topk_features[key_idx]
                k_active = torch.where(k_feat != 0)[0]

                if len(k_active) == 0:
                    pbar.update(1)
                    continue

                k_vecs = W_dec[k_active]                      # [n_k, d_model]
                k_proj = k_vecs @ W_K                         # [n_k, d_head]

                # Apply RoPE to key for this key position
                if rope_sin is not None:
                    k_proj = apply_rope_to_projected(
                        k_proj, key_idx, rope_sin, rope_cos,
                        rotary_dim, rotary_adjacent_pairs,
                    )
                k_scales = k_feat[k_active]                   # [n_k]
                if dec_norms is not None:
                    k_scales = k_scales / dec_norms[k_active]

                int_matrix = (q_proj @ k_proj.T) / attn_scale  # [n_q, n_k]
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

                pbar.update(1)

        all_indices_cpu.extend(chunk_indices)
        all_values_cpu.extend(chunk_values)

        device = topk_features.device
        if device.type != "cpu":
            torch.cuda.empty_cache()

    pbar.close()

    if all_indices_cpu:
        indices = torch.cat(all_indices_cpu, dim=1)
        values = torch.cat(all_values_cpu)
        fra_sparse = torch.sparse_coo_tensor(
            indices, values, size=shape, device="cpu", dtype=torch.float32,
        ).coalesce()
    else:
        fra_sparse = torch.sparse_coo_tensor(
            torch.zeros((4, 0), dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            size=shape, device="cpu",
        )

    return fra_sparse


def _build_fra_result(
    model: HookedTransformer,
    layer: int,
    head: int,
    feature_activations: torch.Tensor,
    W_dec: torch.Tensor,
    device,
    *,
    top_k: int | None = 20,
    rms_activations: torch.Tensor | None = None,
    dec_norms: torch.Tensor | None = None,
    chunk_size: int = 16,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Shared post-encoding FRA construction.

    Handles top-k sparsification, weight extraction, RMSNorm correction,
    RoPE, ``compute_fra_sparse``, device transfer, and result assembly.

    Args:
        model: Target model (for W_Q / W_K and RoPE parameters).
        layer: Attention layer index.
        head: Attention head index.
        feature_activations: ``[seq, d_sae]`` raw encoded features.
        W_dec: ``[d_sae, d_model]`` decoder weights.
        device: Torch device for the result tensor.
        top_k: Features per position to keep (None = all).
        rms_activations: ``[seq, d_model]`` residual stream for RMSNorm
            correction, or None to skip.
        dec_norms: ``[d_sae]`` decoder norms for rescale correction, or None.
        chunk_size: GPU batch size for ``compute_fra_sparse``.
        verbose: Show progress.

    Returns:
        Dict with ``fra_tensor_sparse``, ``shape``, ``seq_len``,
        ``total_interactions``, ``feature_activations``, ``topk_features``.
    """
    topk_features = topk_sparsify(feature_activations, top_k).float()

    # Attention weights — float32 for accumulation precision
    W_Q = model.blocks[layer].attn.W_Q[head].float()       # [d_model, d_head]
    W_K_mat = get_W_K(model, layer, head).float()           # [d_model, d_head]
    d_head = W_Q.shape[-1]
    attn_scale = math.sqrt(d_head)

    # RMSNorm correction
    rms = None
    if rms_activations is not None:
        eps = model.cfg.eps
        rms = (rms_activations.float().pow(2).mean(dim=-1) + eps).sqrt()

    # RoPE
    rope_sin, rope_cos, rotary_dim, rotary_adjacent_pairs = (
        _extract_rope_params(model, layer)
    )

    fra_tensor_sparse = compute_fra_sparse(
        topk_features, W_dec.float(), W_Q, W_K_mat, attn_scale,
        rms=rms,
        dec_norms=dec_norms,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        rotary_dim=rotary_dim,
        rotary_adjacent_pairs=rotary_adjacent_pairs,
        chunk_size=chunk_size,
        verbose=verbose,
        layer_head_label=f"L{layer}H{head}",
    )

    # Move to device if it fits
    if str(device) != "cpu":
        try:
            fra_tensor_sparse = fra_tensor_sparse.to(device)
        except RuntimeError:
            if verbose:
                print("Warning: sparse tensor too large for GPU, keeping on CPU.")

    seq_len = feature_activations.shape[0]
    d_sae = feature_activations.shape[-1]
    total_interactions = fra_tensor_sparse._nnz()
    shape = fra_tensor_sparse.shape

    if verbose:
        _k = top_k if top_k is not None else d_sae
        density = total_interactions / max(seq_len * seq_len * _k * _k, 1)
        print(
            f"4D FRA tensor: shape={shape}, "
            f"nnz={total_interactions:,}, density={density:.2%}"
        )

    return {
        "fra_tensor_sparse": fra_tensor_sparse,
        "shape": shape,
        "seq_len": seq_len,
        "total_interactions": total_interactions,
        "feature_activations": feature_activations,
        "topk_features": topk_features,
    }


# ── Legacy analysis helpers ────────────────────────────────────────────


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

    d_head = W_Q.shape[-1]
    attention_scores = einsum(q, k, "q a, k a -> q k") / math.sqrt(d_head)

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
    activations_SD = get_llm_activations(model,input_text,hook_point=hook_point,layers=layer)
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

    return {
        'query_active_features': query_active_features,
        'key_active_features': key_active_features,
        'interaction_matrix_unscaled': interaction_matrix_unscaled,
        'matrix_scaling': matrix_scaling,
        'interaction_matrix': interaction_matrix_unscaled * matrix_scaling
    }


def get_sentence_averages(llm:Any,sae:Any,layer:int,head:int,input_text:str,hook_point:str="attn.hook_z"):
	text_length=128
	hidden_dim=sae.d_sae
	data_dep_int_matrix=np.zeros((hidden_dim,hidden_dim))
	data_dep_int_matrix_abs=np.zeros((hidden_dim,hidden_dim))
	data_dep_localization_matrix=np.zeros((hidden_dim,hidden_dim))
	count=0
	for key_index in tqdm(range(text_length), disable=True):
		for query_index in range(key_index,text_length):
				feature_analysis=analyze_feature_attention_interactions(llm,sae,layer,head,input_text,query_index,key_index,hook_point)
				int_matrix=feature_analysis["interaction_matrix"]
				query_active_features=feature_analysis["query_active_features"]
				key_active_features=feature_analysis["key_active_features"]
				data_independent=feature_analysis["interaction_matrix_unscaled"]

				resized_data_dependent_int=np.zeros((hidden_dim,hidden_dim))
				resized_data_dependent_int[query_active_features[:,None],key_active_features[None,:]]=int_matrix

				resized_data_dependent_localization=np.zeros((hidden_dim,hidden_dim))
				resized_data_dependent_localization[query_active_features[:,None],key_active_features[None,:]]=np.abs(int_matrix)*(query_index-key_index)




				data_dep_int_matrix=data_dep_int_matrix+resized_data_dependent_int
				data_dep_int_matrix_abs=data_dep_int_matrix_abs+np.abs(resized_data_dependent_int)
				data_dep_localization_matrix=data_dep_localization_matrix+resized_data_dependent_localization

				count+=1

	data_dep_int_matrix/=count
	data_dep_localization_matrix=data_dep_localization_matrix/np.clip(data_dep_int_matrix_abs,a_min=1,a_max=None)
	data_dep_int_matrix_abs/=count

	return data_dep_int_matrix,data_dep_int_matrix_abs,data_dep_localization_matrix




# ═══════════════════════════════════════════════════════════════════════
# Entry point: single-model SAE path
# ═══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def get_sentence_fra_batch(
    model: HookedTransformer,
    sae: Any,
    text: str,
    layer: int,
    head: int,
    max_length: int = 128,
    top_k: int | None = 20,
    verbose: bool = False,
    hook_point: str = "ln1.hook_normalized",
    chunk_size: int = 16,
    normalize_by_decoder_norm: bool | None = None,
    prepend_bos: bool | None = None,
) -> Dict[str, Any]:
    """
    Compute full 4D Feature-Resolved Attention tensor for a sentence.
    Returns a sparse representation to avoid memory issues.

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
        head: Which attention head to analyze
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
        Dictionary containing:
            - fra_tensor_sparse: Sparse 4D tensor indices and values
            - shape: Shape of the full tensor [seq_len, seq_len, d_sae, d_sae]
            - seq_len: Actual sequence length
            - total_interactions: Total number of non-zero interactions
            - normalized: Whether decoder-norm normalization was applied
    """
    device = next(model.parameters()).device

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

    # hook_z is [seq_len, n_heads, d_head] → flatten to [seq_len, n_heads*d_head]
    # ln1.hook_normalized is already [seq_len, d_model]
    if act.dim() == 3:
        act = act.flatten(-2, -1)

    # Encode to SAE features
    if verbose:
        print(f"Encoding {act.shape[0]} positions to SAE features...")

    if hasattr(sae, 'encode'):
        feature_activations = sae.encode(act)   # [seq_len, d_sae]
    else:
        feature_activations = sae.sae.encode(act)

    # If the SAE normalizes inputs (e.g. Gemma-Scope), the feature activations
    # are in the normalized scale.  Divide by the norm coefficient so that
    # FRA[q,k,i,j] sums to the actual (un-normalized) QK attention score.
    # Top-k ranking is unaffected (same scalar per position).
    if hasattr(sae, '_norm_coeff') and sae._norm_coeff is not None:
        feature_activations = feature_activations / sae._norm_coeff

    # Get decoder weights
    if hasattr(sae, 'W_dec'):
        W_dec = sae.W_dec   # [d_sae, d_model]
    else:
        W_dec = sae.sae.W_dec

    # Handle rescale_acts_by_decoder_norm: when the SAE was trained with this
    # flag, encode() returns activations scaled UP by ||W_dec[i]||, and decode()
    # scales them back DOWN.  The true per-feature contribution to x is
    #   (f[i] / ||W_dec[i]||) * W_dec[i]
    # but without correction FRA would use f[i] * W_dec[i], inflating each
    # entry by ||W_dec[q_feat]|| * ||W_dec[k_feat]||.
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

    # RMSNorm correction is needed when the SAE's decoder vectors live in
    # residual-stream space but W_Q / W_K project from post-RMSNorm space.
    # Use the full reconstructed activations (not actual model activations)
    # so that FRA scores match coder-patched scores when all features are kept.
    if "resid" in hook_point:
        b_dec = sae.b_dec
        x_hat = feature_activations @ W_dec + b_dec
        rms_activations = x_hat
    else:
        rms_activations = None

    result = _build_fra_result(
        model, layer, head, feature_activations, W_dec, device,
        top_k=top_k,
        rms_activations=rms_activations,
        dec_norms=dec_norms,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    result["normalized"] = do_normalize
    return result


# ═══════════════════════════════════════════════════════════════════════
# Entry point: two-model crosscoder path
# ═══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def get_sentence_fra_crosscoder(
    base_model: HookedTransformer,
    it_model: HookedTransformer,
    crosscoder: Any,
    tokens: list,
    head: int = 0,
    crosscoder_layer: int = 13,
    max_length: int = 128,
    top_k: int | None = 20,
    verbose: bool = False,
    chunk_size: int = 16,
) -> Dict[str, Any]:
    """Compute 4-D FRA tensor using a model-diffing crosscoder.

    The crosscoder was trained on ``hook_resid_post`` at ``crosscoder_layer``.
    That residual stream state is the input to the *next* layer's attention
    (``hook_resid_pre`` at ``crosscoder_layer + 1``), so we always analyse
    attention at ``crosscoder_layer + 1``.  The same activations are used
    for crosscoder encoding and for the RMSNorm correction.

    The attention layer is **not** independently configurable — it is always
    ``crosscoder_layer + 1``.  This ensures the FRA decomposition corresponds
    to the features the crosscoder actually learned.

    Args:
        base_model:  HookedTransformer for model-index 0 (base).
        it_model:    HookedTransformer for model-index 1 (instruct).
        crosscoder:  ``GemmaCrosscoderFRA`` instance.
        tokens:      Token IDs to analyse.  Caller is responsible for
                     tokenization (including chat template if needed).
        head:        Attention head index.
        crosscoder_layer:
            Layer from which to extract residual-stream activations for the
            crosscoder (default 13, matching the published checkpoint).
            Uses ``hook_resid_post`` at this layer.  The attention layer
            is derived as ``crosscoder_layer + 1``.
        max_length:  Maximum sequence length.
        top_k:       Keep only top-k features per position.
        verbose:     Show progress bar.
        chunk_size:  Query positions per GPU batch (controls peak memory).

    Returns:
        Dict with keys:
          - ``fra_tensor_sparse``  (torch.sparse_coo_tensor, 4-D)
          - ``shape``
          - ``seq_len``
          - ``total_interactions``
          - ``feature_activations``  (torch.Tensor, [seq, d_sae])
          - ``topk_features``  (torch.Tensor, [seq, d_sae])
    """
    layer = crosscoder_layer + 1
    target_model = base_model if crosscoder.model_idx == 0 else it_model
    other_model = it_model if crosscoder.model_idx == 0 else base_model
    device = next(target_model.parameters()).device

    # Truncate if needed
    if max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    # Extract activations from both models at crosscoder_layer
    hook_name = f"blocks.{crosscoder_layer}.hook_resid_post"

    _, target_cache = target_model.run_with_cache(
        tokens_tensor, names_filter=[hook_name],
    )
    _, other_cache = other_model.run_with_cache(
        tokens_tensor, names_filter=[hook_name],
    )

    target_act = target_cache[hook_name].squeeze(0)      # [seq, d_model]
    other_act = other_cache[hook_name].squeeze(0)

    # Stack in crosscoder order: [base, instruct]
    if crosscoder.model_idx == 0:
        x_stacked = torch.stack([target_act, other_act], dim=1)
    else:
        x_stacked = torch.stack([other_act, target_act], dim=1)

    # Encode through crosscoder
    if verbose:
        print(f"Encoding {target_act.shape[0]} positions through crosscoder...")
    feature_activations = crosscoder.encode(x_stacked)   # [seq, d_sae]

    # Crosscoder decoder vectors live in residual-stream space, so RMSNorm
    # correction is always needed.  Use the full reconstructed activations
    # (not actual model activations) so that FRA scores match coder-patched
    # scores when all features are kept.
    #
    # Decode in float32 to match the precision used for per-feature W_dec
    # projections inside compute_fra_sparse (which casts W_dec.float()).
    # Using the native float16 decode would introduce precision mismatch
    # between the RMS denominator and the per-feature numerators.
    b_dec_rms = crosscoder.b_dec.float().to(device)
    x_hat = feature_activations.float() @ crosscoder.W_dec.float() + b_dec_rms
    return _build_fra_result(
        target_model, layer, head, feature_activations, crosscoder.W_dec, device,
        top_k=top_k,
        rms_activations=x_hat,
        chunk_size=chunk_size,
        verbose=verbose,
    )


if __name__ == "__main__":
    print('main character')
