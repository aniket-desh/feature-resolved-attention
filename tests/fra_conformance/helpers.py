"""Shared helpers for FRA conformance tests."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from tests.fra_conformance.contracts import FRAConformanceCase, FRAConformanceResult


def fra_sum_to_attn(sparse_tensor: torch.Tensor, seq_len: int) -> np.ndarray:
    """Sum a sparse 4D FRA tensor over the feature dimensions."""
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices().cpu().numpy()
    values = sparse_tensor.values().cpu().numpy()
    out = np.zeros((seq_len, seq_len), dtype=np.float64)
    np.add.at(out, (indices[0], indices[1]), values)
    return out


def get_qk_weights(model: Any, layer: int, head: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get W_Q and the GQA-mapped W_K for a query head."""
    W_Q = model.blocks[layer].attn.W_Q[head]
    n_kv = model.blocks[layer].attn.W_K.shape[0]
    n_q = model.blocks[layer].attn.W_Q.shape[0]
    kv_head = head * n_kv // n_q
    W_K = model.blocks[layer].attn.W_K[kv_head]
    return W_Q, W_K


def get_qk_biases(model: Any, layer: int, head: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get b_Q and the GQA-mapped b_K for a query head."""
    b_Q = model.blocks[layer].attn.b_Q[head]
    n_kv = model.blocks[layer].attn.b_K.shape[0]
    n_q = model.blocks[layer].attn.b_Q.shape[0]
    kv_head = head * n_kv // n_q
    b_K = model.blocks[layer].attn.b_K[kv_head]
    return b_Q, b_K


def masked_scores(scores: np.ndarray) -> np.ndarray:
    """Apply a causal mask to pre-softmax scores."""
    masked = scores.copy()
    masked[np.triu_indices_from(masked, k=1)] = -np.inf
    return masked


def softmax_rows(masked_score_matrix: np.ndarray) -> np.ndarray:
    """Apply a row-wise softmax while respecting masked entries."""
    return torch.softmax(
        torch.tensor(masked_score_matrix, dtype=torch.float64), dim=-1
    ).cpu().numpy()


def build_sparse_features(
    seq_len: int,
    d_sae: int,
    feature_ids: list[int],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    seed: int = 0,
    scale: float = 0.5,
) -> torch.Tensor:
    """Construct deterministic sparse synthetic feature activations."""
    features = torch.zeros(seq_len, d_sae, dtype=dtype, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn(seq_len, len(feature_ids), generator=gen, dtype=dtype) * scale
    features[:, feature_ids] = values.to(device=device, dtype=dtype)
    return features


def compute_raw_qk(
    x_nobias: torch.Tensor,
    model: Any,
    layer: int,
    head: int,
) -> np.ndarray:
    """Compute the causal raw QK term from the no-bias activations."""
    W_Q, W_K = get_qk_weights(model, layer, head)
    q = x_nobias @ W_Q
    k = x_nobias @ W_K
    return np.tril((q @ k.T).detach().cpu().numpy())


def compute_actual_masked_scores(
    x_full: torch.Tensor,
    model: Any,
    layer: int,
    head: int,
) -> np.ndarray:
    """Compute masked pre-softmax attention scores from the full activations."""
    W_Q, W_K = get_qk_weights(model, layer, head)
    b_Q, b_K = get_qk_biases(model, layer, head)
    q = x_full @ W_Q + b_Q
    k = x_full @ W_K + b_K
    scores = ((q @ k.T) / model.blocks[layer].attn.attn_scale).detach().cpu().numpy()
    return masked_scores(scores)


def reconstruct_masked_scores_from_fra(
    fra_sum: np.ndarray,
    *,
    x_nobias: torch.Tensor,
    b_dec: torch.Tensor,
    model: Any,
    layer: int,
    head: int,
) -> np.ndarray:
    """Add the missing bias terms back to a summed FRA tensor."""
    W_Q, W_K = get_qk_weights(model, layer, head)
    b_Q, b_K = get_qk_biases(model, layer, head)

    q_nobias = x_nobias @ W_Q
    k_nobias = x_nobias @ W_K
    combined_q_bias = b_dec @ W_Q + b_Q
    combined_k_bias = b_dec @ W_K + b_K
    term_q = (q_nobias @ combined_k_bias).detach().cpu().numpy()
    term_k = (k_nobias @ combined_q_bias).detach().cpu().numpy()
    term_const = float(torch.dot(combined_q_bias, combined_k_bias).item())

    scores = (
        fra_sum + term_q[:, None] + term_k[None, :] + term_const
    ) / model.blocks[layer].attn.attn_scale
    return masked_scores(scores)


def assert_conformance_result(
    result: FRAConformanceResult,
    case: FRAConformanceCase,
    *,
    d_sae: int,
    x_nobias: torch.Tensor,
    x_full: torch.Tensor,
    b_dec: torch.Tensor,
    raw_rtol: float,
    raw_atol: float,
    score_rtol: float,
    score_atol: float,
    pattern_rtol: float,
    pattern_atol: float,
) -> None:
    """Assert that a candidate FRA result satisfies the exactness contract."""
    assert result.seq_len == case.expected_seq_len
    assert result.shape == (
        case.expected_seq_len,
        case.expected_seq_len,
        d_sae,
        d_sae,
    )
    assert result.fra_tensor_sparse.layout == torch.sparse_coo

    fra_sum = fra_sum_to_attn(result.fra_tensor_sparse, result.seq_len)
    assert np.allclose(np.triu(fra_sum, k=1), 0, atol=1e-10)

    raw_qk = compute_raw_qk(x_nobias, case.model, case.layer, case.head)
    assert np.allclose(fra_sum, raw_qk, rtol=raw_rtol, atol=raw_atol), (
        f"{case.name}: raw QK mismatch, max err={np.max(np.abs(fra_sum - raw_qk)):.2e}"
    )

    reconstructed_scores = reconstruct_masked_scores_from_fra(
        fra_sum,
        x_nobias=x_nobias,
        b_dec=b_dec,
        model=case.model,
        layer=case.layer,
        head=case.head,
    )
    actual_scores = compute_actual_masked_scores(x_full, case.model, case.layer, case.head)
    finite_mask = np.isfinite(actual_scores)

    assert np.array_equal(np.isneginf(reconstructed_scores), np.isneginf(actual_scores))
    assert np.allclose(
        reconstructed_scores[finite_mask],
        actual_scores[finite_mask],
        rtol=score_rtol,
        atol=score_atol,
    ), (
        f"{case.name}: masked-score mismatch, "
        f"max err={np.max(np.abs(reconstructed_scores[finite_mask] - actual_scores[finite_mask])):.2e}"
    )

    reconstructed_pattern = softmax_rows(reconstructed_scores)
    actual_pattern = softmax_rows(actual_scores)
    assert np.allclose(
        reconstructed_pattern,
        actual_pattern,
        rtol=pattern_rtol,
        atol=pattern_atol,
    ), (
        f"{case.name}: attention-pattern mismatch, "
        f"max err={np.max(np.abs(reconstructed_pattern - actual_pattern)):.2e}"
    )


class FakeTokenizer:
    """Tokenizer stub that returns a fixed token sequence."""

    def __init__(self, token_ids: list[int], bos_token_id: int = 0) -> None:
        self._token_ids = token_ids
        self.bos_token_id = bos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del text
        tokens = list(self._token_ids)
        if add_special_tokens:
            return [self.bos_token_id] + tokens
        return tokens


class FakeModel(nn.Module):
    """Minimal model stub for synthetic conformance tests."""

    def __init__(
        self,
        activations: torch.Tensor,
        *,
        W_Q: torch.Tensor,
        W_K: torch.Tensor,
        b_Q: torch.Tensor,
        b_K: torch.Tensor,
        attn_scale: float,
        layer: int,
        hook_point: str,
    ) -> None:
        super().__init__()
        self._anchor = nn.Parameter(torch.zeros(1, dtype=activations.dtype))
        self._activations = activations
        self._layer = layer
        self._hook_point = hook_point
        self.tokenizer = FakeTokenizer(list(range(activations.shape[0])))
        self.cfg = SimpleNamespace(n_heads=W_Q.shape[0])
        attn = SimpleNamespace(
            W_Q=W_Q,
            W_K=W_K,
            b_Q=b_Q,
            b_K=b_K,
            attn_scale=attn_scale,
        )
        self.blocks = [SimpleNamespace(attn=attn)]

    def run_with_cache(self, tokens: torch.Tensor, names_filter: Any):
        if isinstance(names_filter, str):
            names = [names_filter]
        else:
            names = list(names_filter)

        hook_name = f"blocks.{self._layer}.{self._hook_point}"
        cache = {}
        for name in names:
            if name != hook_name:
                raise KeyError(f"Unsupported hook requested in fake model: {name}")
            cache[name] = self._activations.unsqueeze(0)
        return None, cache


class CachedActivationModel:
    """Wrap a real model but override run_with_cache for a single hook."""

    def __init__(self, base_model: Any, hook_name: str, activations: torch.Tensor) -> None:
        self._base_model = base_model
        self._hook_name = hook_name
        self._activations = activations
        self.blocks = base_model.blocks
        self.tokenizer = base_model.tokenizer
        self.cfg = getattr(base_model, "cfg", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_model, name)

    def parameters(self):
        return self._base_model.parameters()

    def run_with_cache(self, tokens: torch.Tensor, names_filter: Any):
        if isinstance(names_filter, str):
            names = [names_filter]
        else:
            names = list(names_filter)

        if tokens.shape[1] != self._activations.shape[0]:
            raise AssertionError("Injected activation length does not match tokenized prompt length.")

        cache = {}
        for name in names:
            if name != self._hook_name:
                raise KeyError(f"Unsupported hook requested in cached model wrapper: {name}")
            cache[name] = self._activations.unsqueeze(0)
        return None, cache


class PrecomputedFeatureSAE:
    """Mock SAE whose features are fixed in advance."""

    def __init__(
        self,
        *,
        W_dec: torch.Tensor,
        features: torch.Tensor,
        b_dec: torch.Tensor,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ) -> None:
        self.W_dec = W_dec
        self.d_sae = W_dec.shape[0]
        self.b_dec = b_dec
        self._features = features
        self._atol = atol
        self._rtol = rtol

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        expected = self.decode(self._features)
        if not torch.allclose(x, expected, rtol=self._rtol, atol=self._atol):
            raise AssertionError("Synthetic cached activations do not match decode(U).")
        return self._features.clone()

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_dec + self.b_dec


class SyntheticFeatureWrapper:
    """Keep a real decoder but override encode() with synthetic features."""

    def __init__(
        self,
        base_sae: Any,
        synthetic_features: torch.Tensor,
        *,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> None:
        self._base_sae = base_sae
        self._features = synthetic_features
        self.W_dec = base_sae.W_dec
        self.b_dec = base_sae.b_dec
        self.d_sae = base_sae.d_sae
        self._atol = atol
        self._rtol = rtol

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_sae, name)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        expected = self.decode(self._features)
        if not torch.allclose(x, expected, rtol=self._rtol, atol=self._atol):
            raise AssertionError("Injected activations do not match decode(U).")
        return self._features.clone()

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self._base_sae.decode(features)


def require_env_path(name: str) -> Path:
    """Resolve a required path-valued environment variable or fail the test."""
    raw = os.environ.get(name)
    if not raw:
        pytest.fail(f"Required environment variable {name} is not set.")
    path = Path(raw).expanduser()
    if not path.exists():
        pytest.fail(f"{name} points to a missing path: {path}")
    return path


def get_model_load_config(
    default_model_name: str,
    sae: Any,
    *,
    fallback_kwargs: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve the model name and loader kwargs from SAE metadata when available."""
    inner = sae.sae if hasattr(sae, "sae") else sae
    cfg = getattr(inner, "cfg", None)
    meta = getattr(cfg, "metadata", None) if cfg else None
    sae_model_kwargs = getattr(meta, "model_from_pretrained_kwargs", None) or {}

    if sae_model_kwargs:
        model_kwargs = {
            "fold_ln": False,
            "center_unembed": False,
            "center_writing_weights": False,
            "fold_value_biases": False,
            "refactor_factored_attn_matrices": False,
            **sae_model_kwargs,
        }
    else:
        model_kwargs = dict(fallback_kwargs or {})

    token = os.environ.get("HF_TOKEN")
    if token:
        model_kwargs["token"] = token

    model_name = getattr(meta, "model_name", None) or default_model_name
    return model_name, model_kwargs
