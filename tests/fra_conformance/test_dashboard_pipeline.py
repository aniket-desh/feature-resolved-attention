"""Synthetic test for the shared dashboard compute pipeline."""

from __future__ import annotations

import math

import pytest
import torch

from fra.dashboard_compute import compute_dashboard_fra
from tests.fra_conformance.helpers import (
    FakeModel,
    PrecomputedFeatureSAE,
    build_sparse_features,
)

pytestmark = pytest.mark.fra_conformance


def test_dashboard_pipeline_validation_summary_synthetic() -> None:
    dtype = torch.float32
    layer = 0
    head = 1
    hook_point = "ln1.hook_normalized"

    seq_len = 5
    d_sae = 6
    d_model = 4
    d_head = 3
    n_q = 2
    n_kv = 1
    feature_ids = [0, 2, 4]

    gen = torch.Generator().manual_seed(0)
    features = build_sparse_features(
        seq_len,
        d_sae,
        feature_ids,
        dtype=dtype,
        device="cpu",
        seed=0,
        scale=0.75,
    )
    W_dec = torch.randn(d_sae, d_model, generator=gen, dtype=dtype) / 3
    b_dec = torch.randn(d_model, generator=gen, dtype=dtype) / 5
    W_Q = torch.randn(n_q, d_model, d_head, generator=gen, dtype=dtype) / 4
    W_K = torch.randn(n_kv, d_model, d_head, generator=gen, dtype=dtype) / 4
    b_Q = torch.randn(n_q, d_head, generator=gen, dtype=dtype) / 7
    b_K = torch.randn(n_kv, d_head, generator=gen, dtype=dtype) / 7

    x_nobias = features @ W_dec
    x_full = x_nobias + b_dec

    model = FakeModel(
        x_full,
        W_Q=W_Q,
        W_K=W_K,
        b_Q=b_Q,
        b_K=b_K,
        attn_scale=math.sqrt(d_head),
        layer=layer,
        hook_point=hook_point,
    )
    sae = PrecomputedFeatureSAE(W_dec=W_dec, features=features, b_dec=b_dec)

    result = compute_dashboard_fra(
        model=model,
        sae=sae,
        text="synthetic input",
        layer=layer,
        head=head,
        hook_point=hook_point,
        top_k_features=len(feature_ids),
        chunk_size=16,
        max_length=128,
        normalize_by_decoder_norm=False,
        prepend_bos=False,
        run_validation=True,
    )

    validation = result.get("validation")
    assert validation is not None

    per_head = validation["per_head"][head]
    assert per_head["status"] == "pass"
    assert per_head["qk_errors"]["fro_rel_err"] < 1e-5
    assert per_head["pattern_errors"]["fro_rel_err"] < 1e-6

    residual = validation["residual"]
    assert residual["fro_rel_err"] < 1e-5
