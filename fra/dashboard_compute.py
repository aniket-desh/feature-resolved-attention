"""Shared FRA computation used by the dashboard and its tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from fra.core.helpers import (
    compute_bias_correction,
    get_attn_scale,
    project_qk,
)
from fra.fra_func import get_sentence_fra_batch
from fra.validation import compute_errors


def apply_softcap(scores: np.ndarray, softcap: float) -> np.ndarray:
    """Apply Gemma-style tanh softcapping to attention scores."""
    if softcap > 0:
        return softcap * np.tanh(scores / softcap)
    return scores


def get_fra_reconstructed_scores(
    indices_np: np.ndarray,
    values_np: np.ndarray,
    seq_len: int,
    bias_corr_np: np.ndarray,
    attn_scale: float,
    softcap: float,
) -> np.ndarray:
    """Reconstruct pre-softmax attention scores from FRA entries."""
    fra_logits = np.zeros((seq_len, seq_len))
    np.add.at(fra_logits, (indices_np[0], indices_np[1]), values_np)
    fra_logits = fra_logits / attn_scale + bias_corr_np[:seq_len, :seq_len]
    return apply_softcap(fra_logits, softcap)


def _masked_scores(scores: np.ndarray) -> np.ndarray:
    masked = scores.copy()
    masked[np.triu_indices_from(masked, k=1)] = -np.inf
    return masked


def _softmax_rows(masked_scores: np.ndarray) -> np.ndarray:
    return torch.softmax(
        torch.tensor(masked_scores, dtype=torch.float64), dim=-1
    ).cpu().numpy()


def _compute_attention_pattern_from_activations(
    *,
    model: Any,
    layer: int,
    head: int,
    act: torch.Tensor,
    hook_point: str,
) -> np.ndarray:
    """Reconstruct the actual attention pattern when hook_pattern is unavailable."""
    attn_scale = get_attn_scale(model, layer)
    softcap = getattr(model.cfg, "attn_scores_soft_cap", 0.0) or 0.0
    zero_bias = torch.zeros(act.shape[-1], dtype=act.dtype, device=act.device)
    q_full, k_full, _, _ = project_qk(
        model,
        layer,
        head,
        act,
        zero_bias,
        needs_rms=("resid" in hook_point),
    )
    scores = ((q_full @ k_full.T) / attn_scale).detach().cpu().numpy()
    scores = apply_softcap(scores, softcap)
    return _softmax_rows(_masked_scores(scores))


def build_dashboard_validation_summary(
    *,
    model: Any,
    hook_point: str,
    heads: list[int],
    act: torch.Tensor,
    x_hat: torch.Tensor,
    b_dec: torch.Tensor,
    feat_acts: torch.Tensor,
    per_head_payload: dict[int, dict[str, Any]],
    exact_validation: bool,
) -> dict[str, Any]:
    """Compute dashboard-time FRA validation metrics from the current run."""
    x_np = act.cpu().numpy()
    x_hat_np = x_hat.cpu().numpy()
    residual_errors = compute_errors(x_np, x_hat_np)
    per_token_l0 = (feat_acts != 0).sum(dim=-1).float()

    residual = {
        **residual_errors,
        "avg_active_features": float(per_token_l0.mean().item()),
        "l0_min": float(per_token_l0.min().item()),
        "l0_max": float(per_token_l0.max().item()),
        "sparsity": float((feat_acts == 0).float().mean().item()),
        "seq_len": int(act.shape[0]),
    }

    per_head = {}
    for h in heads:
        payload = per_head_payload[h]
        indices_np = payload["indices_np"]
        values_np = payload["values_np"]
        seq_len = int(payload["shape"][0])

        fra_sum = np.zeros((seq_len, seq_len), dtype=np.float64)
        np.add.at(fra_sum, (indices_np[0], indices_np[1]), values_np)

        _, _, q_nobias, k_nobias = project_qk(
            model,
            payload["layer"],
            h,
            x_hat,
            b_dec,
            needs_rms=("resid" in hook_point),
        )
        actual_qk = (q_nobias @ k_nobias.T).detach().cpu().numpy()

        causal = np.tril(np.ones((seq_len, seq_len), dtype=np.float64))
        fra_sum *= causal
        actual_qk *= causal
        qk_errors = compute_errors(actual_qk, fra_sum)

        reconstructed_scores = get_fra_reconstructed_scores(
            indices_np,
            values_np,
            seq_len,
            payload["bias_corr_np"],
            payload["attn_scale"],
            payload["softcap"],
        )
        reconstructed_pattern = _softmax_rows(_masked_scores(reconstructed_scores))
        pattern_errors = compute_errors(
            payload["attn_pattern_np"],
            reconstructed_pattern,
        )

        if exact_validation:
            status = "pass" if qk_errors["fro_rel_err"] < 0.50 else "fail"
        else:
            status = "approximate"

        per_head[h] = {
            "status": status,
            "exact_mode": exact_validation,
            "qk_errors": qk_errors,
            "pattern_errors": pattern_errors,
            "seq_len": seq_len,
            "nnz": int(payload["total_interactions"]),
        }

    return {
        "residual": residual,
        "per_head": per_head,
        "hook_point": hook_point,
    }


@torch.no_grad()
def compute_dashboard_fra(
    *,
    model: Any,
    sae: Any,
    text: str,
    layer: int,
    head: int | list[int],
    hook_point: str,
    top_k_features: int,
    chunk_size: int = 16,
    max_length: int = 128,
    normalize_by_decoder_norm: bool | None = None,
    prepend_bos: bool | None = None,
    run_validation: bool = False,
    exact_validation: bool | None = None,
) -> dict[str, Any]:
    """Compute the dashboard FRA payload without importing Streamlit."""
    device = next(model.parameters()).device
    attn_scale = get_attn_scale(model, layer)
    softcap = getattr(model.cfg, "attn_scores_soft_cap", 0.0) or 0.0
    heads = head if isinstance(head, list) else [int(head)]

    if prepend_bos is not None:
        tokens = model.tokenizer.encode(text, add_special_tokens=prepend_bos)
    else:
        tokens = model.tokenizer.encode(text)
    if max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]

    tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    hook_name = f"blocks.{layer}.{hook_point}"
    _, cache = model.run_with_cache(tok_tensor, names_filter=[hook_name])
    act = cache[hook_name].squeeze(0)
    if act.dim() == 3:
        act = act.flatten(-2, -1)

    feat_acts = sae.encode(act)
    x_hat = sae.decode(feat_acts)

    attn_hook = f"blocks.{layer}.attn.hook_pattern"
    attn_pattern_all = None
    try:
        _, attn_cache = model.run_with_cache(tok_tensor, names_filter=[attn_hook])
        attn_pattern_all = attn_cache[attn_hook][0]
    except Exception:
        attn_pattern_all = None

    if hasattr(model.tokenizer, "decode"):
        token_strs = [model.tokenizer.decode([t]) for t in tokens]
    else:
        token_strs = [str(t) for t in tokens]
    has_bos = (
        model.tokenizer.bos_token_id is not None
        and len(tokens) > 0
        and tokens[0] == model.tokenizer.bos_token_id
    )

    needs_rms = "resid" in hook_point
    b_dec = sae.b_dec if hasattr(sae, "b_dec") else sae.sae.b_dec
    per_head: dict[int, dict[str, Any]] = {}

    for h in heads:
        fra_result = get_sentence_fra_batch(
            model=model,
            sae=sae,
            text=text,
            layer=layer,
            head=int(h),
            max_length=max_length,
            top_k=top_k_features,
            verbose=False,
            hook_point=hook_point,
            chunk_size=chunk_size,
            normalize_by_decoder_norm=normalize_by_decoder_norm,
            prepend_bos=prepend_bos,
        )
        sparse_h = fra_result["fra_tensor_sparse"].coalesce()
        q_full, k_full, q_nobias, k_nobias = project_qk(
            model,
            layer,
            int(h),
            x_hat,
            b_dec,
            needs_rms=needs_rms,
        )
        bias_corr_h = compute_bias_correction(
            q_full,
            k_full,
            q_nobias,
            k_nobias,
            attn_scale,
        ).cpu().numpy()
        per_head[int(h)] = {
            "layer": layer,
            "indices_np": sparse_h.indices().cpu().numpy(),
            "values_np": sparse_h.values().cpu().numpy(),
            "shape": tuple(sparse_h.shape),
            "fra_tensor_sparse": sparse_h,
            "total_interactions": sparse_h._nnz(),
            "bias_corr_np": bias_corr_h,
            "attn_scale": attn_scale,
            "softcap": softcap,
            "attn_pattern_np": (
                attn_pattern_all[int(h)].cpu().numpy()
                if attn_pattern_all is not None
                else _compute_attention_pattern_from_activations(
                    model=model,
                    layer=layer,
                    head=int(h),
                    act=act,
                    hook_point=hook_point,
                )
            ),
        }

    if exact_validation is None:
        exact_validation = hook_point != "attn.hook_z"

    validation_payload = None
    if run_validation:
        validation_payload = build_dashboard_validation_summary(
            model=model,
            hook_point=hook_point,
            heads=[int(h) for h in heads],
            act=act,
            x_hat=x_hat,
            b_dec=b_dec,
            feat_acts=feat_acts,
            per_head_payload=per_head,
            exact_validation=exact_validation,
        )

    default_h = int(heads[0])
    result = {
        "per_head": per_head,
        "heads": [int(h) for h in heads],
        "default_head": default_h,
        "fra_tensor_sparse": per_head[default_h]["fra_tensor_sparse"],
        "fra_sparse_dict": {
            h: per_head[h]["fra_tensor_sparse"] for h in per_head
        },
        "indices_np": per_head[default_h]["indices_np"],
        "values_np": per_head[default_h]["values_np"],
        "shape": per_head[default_h]["shape"],
        "total_interactions": per_head[default_h]["total_interactions"],
        "bias_corr_np": per_head[default_h]["bias_corr_np"],
        "attn_pattern_np": per_head[default_h]["attn_pattern_np"],
        "seq_len": per_head[default_h]["shape"][0],
        "feat_acts_np": feat_acts.cpu().numpy(),
        "attn_scale": attn_scale,
        "softcap": softcap,
        "token_strs": token_strs,
        "has_bos": has_bos,
    }
    if validation_payload is not None:
        result["validation"] = validation_payload
    return result
