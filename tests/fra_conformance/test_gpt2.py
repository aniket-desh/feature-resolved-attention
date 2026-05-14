"""GPT-2 exactness test for the FRA conformance harness."""

from __future__ import annotations

import pytest

from tests.fra_conformance.contracts import FRAConformanceCase
from tests.fra_conformance.helpers import (
    CachedActivationModel,
    SyntheticFeatureWrapper,
    assert_conformance_result,
    build_sparse_features,
    get_model_load_config,
    require_env_path,
)

pytestmark = [
    pytest.mark.fra_conformance,
    pytest.mark.model_backed,
    pytest.mark.gpt2,
]


def test_gpt2_actual_decoder_exactness(fra_candidate) -> None:
    import torch
    from transformer_lens import HookedTransformer

    from fra.sae_lens_wrapper import LocalLn1SAE

    layer = 5
    head = 1
    hook_point = "ln1.hook_normalized"
    text = "The cat sat on the mat."
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = require_env_path("GPT2_FRA_SAE_PATH")
    try:
        base_sae = LocalLn1SAE(checkpoint_path, layer=layer, device=device)
    except Exception as exc:
        pytest.fail(f"Failed to load GPT-2 FRA SAE from {checkpoint_path}: {exc}")

    model_name, model_kwargs = get_model_load_config(
        "gpt2-small",
        base_sae,
        fallback_kwargs={
            "fold_ln": False,
            "center_unembed": True,
            "center_writing_weights": True,
        },
    )
    try:
        base_model = HookedTransformer.from_pretrained(model_name, device=device, **model_kwargs)
    except Exception as exc:
        pytest.fail(f"Failed to load GPT-2 model '{model_name}': {exc}")

    tokens = base_model.tokenizer.encode(text, add_special_tokens=False)
    seq_len = len(tokens)
    feature_ids = [0, 17, 1234]
    top_k = len(feature_ids)

    features = build_sparse_features(
        seq_len,
        base_sae.d_sae,
        feature_ids,
        dtype=base_sae.W_dec.dtype,
        device=base_sae.W_dec.device,
        seed=0,
        scale=0.5,
    )
    x_full = base_sae.decode(features)
    x_nobias = x_full - base_sae.b_dec

    wrapped_model = CachedActivationModel(
        base_model,
        hook_name=f"blocks.{layer}.{hook_point}",
        activations=x_full,
    )
    wrapped_sae = SyntheticFeatureWrapper(base_sae, features)
    case = FRAConformanceCase(
        name="gpt2_actual_decoder",
        model=wrapped_model,
        sae=wrapped_sae,
        text=text,
        layer=layer,
        head=head,
        hook_point=hook_point,
        top_k=top_k,
        chunk_size=16,
        max_length=128,
        prepend_bos=False,
        normalize_by_decoder_norm=None,
        expected_seq_len=seq_len,
    )

    result = fra_candidate(case)
    assert_conformance_result(
        result,
        case,
        d_sae=base_sae.d_sae,
        x_nobias=x_nobias,
        x_full=x_full,
        b_dec=base_sae.b_dec,
        raw_rtol=1e-5,
        raw_atol=1e-5,
        score_rtol=1e-5,
        score_atol=1e-5,
        pattern_rtol=1e-6,
        pattern_atol=1e-6,
    )
