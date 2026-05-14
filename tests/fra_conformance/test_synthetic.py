"""Synthetic exactness tests for the FRA conformance harness."""

from __future__ import annotations

import math

import pytest
import torch

from tests.fra_conformance.contracts import FRAConformanceCase
from tests.fra_conformance.helpers import (
    FakeModel,
    PrecomputedFeatureSAE,
    assert_conformance_result,
    build_sparse_features,
)

pytestmark = pytest.mark.fra_conformance


def test_synthetic_exactness(fra_candidate) -> None:
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
    top_k = len(feature_ids)

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
    case = FRAConformanceCase(
        name="synthetic",
        model=model,
        sae=sae,
        text="synthetic input",
        layer=layer,
        head=head,
        hook_point=hook_point,
        top_k=top_k,
        chunk_size=16,
        max_length=128,
        prepend_bos=False,
        normalize_by_decoder_norm=False,
        expected_seq_len=seq_len,
    )

    result = fra_candidate(case)
    assert_conformance_result(
        result,
        case,
        d_sae=d_sae,
        x_nobias=x_nobias,
        x_full=x_full,
        b_dec=b_dec,
        raw_rtol=1e-5,
        raw_atol=1e-5,
        score_rtol=1e-5,
        score_atol=1e-5,
        pattern_rtol=1e-6,
        pattern_atol=1e-6,
    )
