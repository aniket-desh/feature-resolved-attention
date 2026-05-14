"""
OV-based feature steering for Feature-Resolved Attention.

Provides three steering modes:
  - OV-only: modify value vectors via hook_v, leave QK untouched
  - QK-only: patch attention scores (wrapper around existing run_condition)
  - Combined: both QK score patching and OV value steering

The OV steering mechanism:
  1. Hooks the SAE input hook point (read-only) to cache feature activations
  2. Hooks blocks.{layer}.attn.hook_v to modify value vectors per-feature:
     v[batch, pos, head, :] += (scale - 1) * f_λ[pos] * (W_dec[λ] @ W_V_h)
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional

from transformer_lens import HookedTransformer
from fra.core.helpers import get_W_V, get_W_O


def _compute_metrics(patched_logits, shift_labels, unpatched_logits):
    """Compute loss/KL/top1 metrics from logits, memory-efficiently."""
    result = {"patched_logits": patched_logits}

    if shift_labels is not None:
        result["loss"] = F.cross_entropy(
            patched_logits[0, :-1].float(), shift_labels
        ).item()

    if unpatched_logits is not None and shift_labels is not None:
        p_logits = unpatched_logits[0, :-1].float()
        q_logits = patched_logits[0, :-1].float()

        result["top1_change_frac"] = (
            (p_logits.argmax(-1) != q_logits.argmax(-1)).float().mean().item()
        )

        log_p = F.log_softmax(p_logits, dim=-1)
        log_q = F.log_softmax(q_logits, dim=-1)
        result["kl_div"] = (log_p.exp() * (log_p - log_q)).sum(-1).mean().item()
        del p_logits, q_logits, log_p, log_q

    return result


@torch.no_grad()
def run_ov_steering(
    model: HookedTransformer,
    sae: Any,
    tok_tensor: torch.Tensor,
    layer: int,
    head: int,
    hook_point: str,
    feature_scales: Dict[int, float],
    shift_labels: Optional[torch.Tensor] = None,
    unpatched_logits: Optional[torch.Tensor] = None,
    normalize_by_decoder_norm: bool | None = None,
) -> Dict[str, Any]:
    """Steer by scaling specific feature contributions in the value vectors.

    For each feature in feature_scales:
      scale > 1.0: amplify the feature's contribution
      scale = 1.0: no change
      scale = 0.0: ablate the feature (remove from OV)
      scale < 0.0: reverse the feature direction

    Only modifies the V vectors for the targeted head, leaving QK attention
    scores untouched.

    Args:
        model: HookedTransformer.
        sae: SAE wrapper with encode/decode and W_dec.
        tok_tensor: [1, seq_len] token tensor.
        layer: Attention layer index.
        head: Attention head index.
        hook_point: SAE hook point (e.g. "ln1.hook_normalized").
        feature_scales: {feature_idx: scale_factor} dict.
        shift_labels: [seq_len-1] labels for loss computation.
        unpatched_logits: [1, seq_len, vocab] clean logits for KL/top1.
        normalize_by_decoder_norm: Whether to apply decoder norm correction.

    Returns:
        dict: loss, kl_div, top1_change_frac, patched_logits
    """
    device = next(model.parameters()).device

    # Get weights
    W_dec = (sae.W_dec if hasattr(sae, 'W_dec') else sae.sae.W_dec).float()
    W_V_h = get_W_V(model, layer, head).float()

    # Handle decoder norm
    if normalize_by_decoder_norm is None:
        inner = sae.sae if hasattr(sae, 'sae') else sae
        cfg = getattr(inner, 'cfg', None)
        do_normalize = getattr(cfg, 'rescale_acts_by_decoder_norm', False) if cfg else False
    else:
        do_normalize = normalize_by_decoder_norm

    dec_norms = W_dec.norm(dim=-1) if do_normalize else None

    # Pre-compute per-feature V projections for features we want to steer
    # delta_v_basis[feat] = W_dec[feat] @ W_V_h → [d_head]
    feat_indices = list(feature_scales.keys())
    feat_scales_tensor = torch.tensor(
        [feature_scales[f] for f in feat_indices], device=device, dtype=torch.float32
    )
    feat_v_proj = W_dec[feat_indices] @ W_V_h  # [n_feats, d_head]
    if dec_norms is not None:
        feat_v_proj = feat_v_proj / dec_norms[feat_indices].unsqueeze(-1)

    # GQA: hook_v fires with shape [batch, seq, n_kv_heads, d_head].
    # TransformerLens expands W_V weights to n_q_heads but hook_v still has n_kv.
    # Use n_key_value_heads from config (not W_V.shape which may be expanded).
    n_q_heads = model.cfg.n_heads
    n_kv_heads = getattr(model.cfg, "n_key_value_heads", None) or n_q_heads
    kv_head_idx = head * n_kv_heads // n_q_heads
    print(f"  [OV steer] head={head}, n_kv_cfg={n_kv_heads}, n_q={n_q_heads} → kv_idx={kv_head_idx}")

    # State for the read-only hook
    cached_features = {}

    hook_name = f"blocks.{layer}.{hook_point}"
    v_hook_name = f"blocks.{layer}.attn.hook_v"

    def capture_hook(activation, hook):
        """Read-only hook: encode activation through SAE and cache features."""
        x = activation[0]  # [seq_len, d_model]
        if x.dim() == 3:
            x = x.flatten(-2, -1)

        if hasattr(sae, 'encode'):
            features = sae.encode(x)
        else:
            features = sae.sae.encode(x)

        if hasattr(sae, '_norm_coeff') and sae._norm_coeff is not None:
            features = features / sae._norm_coeff

        cached_features['feats'] = features.float()
        return activation  # pass through unmodified

    def steer_v_hook(v, hook):
        """Modify value vectors for the targeted head (in-place to save memory)."""
        features = cached_features.get('feats')
        if features is None:
            return v

        seq_len = min(features.shape[0], v.shape[1])

        # Vectorised: gather feature activations for all steered features
        feat_acts = features[:seq_len, feat_indices].float()  # [seq, n_feats]

        # delta = (scale - 1) * feat_acts * feat_v_proj
        scale_minus_1 = (feat_scales_tensor - 1.0)  # [n_feats]
        weighted_acts = feat_acts * scale_minus_1.unsqueeze(0)  # [seq, n_feats]
        delta = weighted_acts @ feat_v_proj  # [seq, d_head]

        v[0, :seq_len, kv_head_idx, :] += delta.to(v.dtype)
        return v

    # Run with hooks — reset_hooks_end=True to clean up after
    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            tok_tensor,
            fwd_hooks=[
                (hook_name, capture_hook),
                (v_hook_name, steer_v_hook),
            ],
            reset_hooks_end=True,
        )

    return _compute_metrics(patched_logits, shift_labels, unpatched_logits)


@torch.no_grad()
def run_qk_steering(
    model: HookedTransformer,
    tok_tensor: torch.Tensor,
    layer: int,
    head: int,
    modified_scores: torch.Tensor,
    shift_labels: Optional[torch.Tensor] = None,
    unpatched_logits: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Steer by patching attention scores (QK-only intervention).

    Thin wrapper matching the interface of run_ov_steering.

    Args:
        model: HookedTransformer.
        tok_tensor: [1, seq_len] token tensor.
        layer: Attention layer index.
        head: Attention head index.
        modified_scores: [seq_len, seq_len] pre-softmax attention scores.
        shift_labels: [seq_len-1] labels for loss.
        unpatched_logits: [1, seq_len, vocab] clean logits.

    Returns:
        dict: loss, kl_div, top1_change_frac, patched_logits
    """
    seq_len = modified_scores.shape[0]
    score_hook = f"blocks.{layer}.attn.hook_attn_scores"

    def hook_fn(attn_scores, hook):
        attn_scores[0, head, :seq_len, :seq_len] = modified_scores
        return attn_scores

    patched_logits = model.run_with_hooks(
        tok_tensor, fwd_hooks=[(score_hook, hook_fn)]
    )

    return _compute_metrics(patched_logits, shift_labels, unpatched_logits)


@torch.no_grad()
def run_combined_steering(
    model: HookedTransformer,
    sae: Any,
    tok_tensor: torch.Tensor,
    layer: int,
    head: int,
    hook_point: str,
    qk_scores: Optional[torch.Tensor] = None,
    ov_feature_scales: Optional[Dict[int, float]] = None,
    shift_labels: Optional[torch.Tensor] = None,
    unpatched_logits: Optional[torch.Tensor] = None,
    normalize_by_decoder_norm: bool | None = None,
) -> Dict[str, Any]:
    """Steer both QK scores and OV features simultaneously.

    Attaches hook_attn_scores for QK patching and hook_v for OV steering
    in a single forward pass.

    Args:
        model: HookedTransformer.
        sae: SAE wrapper.
        tok_tensor: [1, seq_len] token tensor.
        layer: Attention layer index.
        head: Attention head index.
        hook_point: SAE hook point.
        qk_scores: [seq, seq] patched scores (None = no QK patching).
        ov_feature_scales: {feat_idx: scale} (None = no OV steering).
        shift_labels: [seq_len-1] labels.
        unpatched_logits: [1, seq_len, vocab] clean logits.
        normalize_by_decoder_norm: Decoder norm correction flag.

    Returns:
        dict: loss, kl_div, top1_change_frac, patched_logits
    """
    device = next(model.parameters()).device
    hooks = []

    # QK hook
    if qk_scores is not None:
        seq_len_qk = qk_scores.shape[0]
        score_hook = f"blocks.{layer}.attn.hook_attn_scores"

        def qk_hook_fn(attn_scores, hook):
            attn_scores[0, head, :seq_len_qk, :seq_len_qk] = qk_scores
            return attn_scores

        hooks.append((score_hook, qk_hook_fn))

    # OV hooks
    if ov_feature_scales:
        W_dec = (sae.W_dec if hasattr(sae, 'W_dec') else sae.sae.W_dec).float()
        W_V_h = get_W_V(model, layer, head).float()

        if normalize_by_decoder_norm is None:
            inner = sae.sae if hasattr(sae, 'sae') else sae
            cfg = getattr(inner, 'cfg', None)
            do_normalize = getattr(cfg, 'rescale_acts_by_decoder_norm', False) if cfg else False
        else:
            do_normalize = normalize_by_decoder_norm

        dec_norms = W_dec.norm(dim=-1) if do_normalize else None

        feat_indices = list(ov_feature_scales.keys())
        feat_scales_tensor = torch.tensor(
            [ov_feature_scales[f] for f in feat_indices],
            device=device, dtype=torch.float32,
        )
        # GQA: map query head to KV head (use config, not W_V.shape)
        n_q_heads = model.cfg.n_heads
        n_kv_heads = getattr(model.cfg, "n_key_value_heads", None) or n_q_heads
        kv_head_combined = head * n_kv_heads // n_q_heads

        feat_v_proj = W_dec[feat_indices] @ W_V_h
        if dec_norms is not None:
            feat_v_proj = feat_v_proj / dec_norms[feat_indices].unsqueeze(-1)

        cached_features = {}
        hook_name = f"blocks.{layer}.{hook_point}"
        v_hook_name = f"blocks.{layer}.attn.hook_v"

        def capture_hook(activation, hook):
            x = activation[0]
            if x.dim() == 3:
                x = x.flatten(-2, -1)
            if hasattr(sae, 'encode'):
                features = sae.encode(x)
            else:
                features = sae.sae.encode(x)
            if hasattr(sae, '_norm_coeff') and sae._norm_coeff is not None:
                features = features / sae._norm_coeff
            cached_features['feats'] = features.float()
            return activation

        def steer_v_hook(v, hook):
            features = cached_features.get('feats')
            if features is None:
                return v
            v_out = v.clone()
            sl = features.shape[0]
            feat_acts = features[:sl, feat_indices]
            if "resid" in hook_point:
                W_dec_local = sae.W_dec if hasattr(sae, 'W_dec') else sae.sae.W_dec
                b_dec = sae.b_dec if hasattr(sae, 'b_dec') else sae.sae.b_dec
                x_hat = features @ W_dec_local.float() + b_dec.float()
                eps = model.cfg.eps
                rms = (x_hat.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
                feat_acts = feat_acts / rms
            scale_minus_1 = feat_scales_tensor - 1.0
            weighted_acts = feat_acts * scale_minus_1.unsqueeze(0)
            delta = weighted_acts @ feat_v_proj
            v_out[0, :sl, kv_head_combined, :] += delta.to(v.dtype)
            return v_out

        hooks.append((hook_name, capture_hook))
        hooks.append((v_hook_name, steer_v_hook))

    # Run
    patched_logits = model.run_with_hooks(tok_tensor, fwd_hooks=hooks)

    return _compute_metrics(patched_logits, shift_labels, unpatched_logits)
