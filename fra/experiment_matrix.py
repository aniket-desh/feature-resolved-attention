"""
3×3 Attribution × Intervention matrix for Feature-Resolved Attention.

Orchestrates nine (attribution, intervention) combinations:
  Attribution: {qk, ov, qk+ov} — how to select features to intervene on
  Intervention: {qk, ov, qk+ov} — where to steer/ablate

The key insight from Tiny Stories (prior work): ranking features via QK FRA
and then steering in OV often gives the best intervention trade-off.
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Any, Dict, List, Optional

from transformer_lens import HookedTransformer
from fra.core.fra import get_sentence_fra_batch, topk_sparsify
from fra.core.ov import get_sentence_ov_decomposition, rank_ov_features
from fra.core.helpers import fra_sum_to_attn
from fra.ov_steering import run_ov_steering, run_qk_steering, run_combined_steering
from fra.ablation_study import (
    rank_feature_pairs, ablate_fra_pairs, reconstruct_scores,
    compute_bias_corrections,
)


@torch.no_grad()
def run_attribution_intervention_matrix(
    model: HookedTransformer,
    sae: Any,
    text: str,
    layer: int,
    head: int,
    hook_point: str = "ln1.hook_normalized",
    k: int = 50,
    top_k: int = 20,
    max_length: int = 128,
    normalize_by_decoder_norm: bool | None = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full 3×3 attribution × intervention matrix.

    Attribution modes (how to select features):
      "qk": rank by QK FRA pair strength → extract unique features
      "ov": rank by OV contribution magnitude
      "qk+ov": union of top features from both rankings

    Intervention modes (where to steer):
      "qk": patch attention scores (ablate selected pairs)
      "ov": steer value vectors (scale=0 for selected features)
      "qk+ov": both simultaneously

    Args:
        model: HookedTransformer.
        sae: SAE wrapper.
        text: Input text.
        layer: Attention layer.
        head: Attention head.
        hook_point: SAE hook point.
        k: Number of top features/pairs to select.
        top_k: Features per position for sparsification.
        max_length: Max sequence length.
        normalize_by_decoder_norm: Decoder norm flag.
        verbose: Print progress.

    Returns:
        dict with:
            matrix: nested dict result[attr][interv] = {loss, kl_div, ...}
            unpatched: baseline metrics
            qk_features: features selected by QK attribution
            ov_features: features selected by OV attribution
            combined_features: features selected by combined attribution
    """
    device = next(model.parameters()).device

    # ── Tokenize and get baseline ────────────────────────────────────────
    tokens = model.tokenizer.encode(text)
    if max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]
    tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    if tok_tensor.shape[1] < 2:
        return {"error": "text too short"}

    unpatched_logits = model(tok_tensor)
    shift_labels = tok_tensor[0, 1:]
    unpatched_loss = F.cross_entropy(unpatched_logits[0, :-1], shift_labels).item()

    if verbose:
        print(f"Unpatched loss: {unpatched_loss:.4f}")

    # ── Compute QK FRA ───────────────────────────────────────────────────
    if verbose:
        print("\nComputing QK FRA...")
    qk_result = get_sentence_fra_batch(
        model, sae, text, layer, head,
        max_length=max_length, top_k=top_k, verbose=verbose,
        hook_point=hook_point,
        normalize_by_decoder_norm=normalize_by_decoder_norm,
    )
    fra_sparse = qk_result["fra_tensor_sparse"]

    # ── Compute OV decomposition ─────────────────────────────────────────
    if verbose:
        print("\nComputing OV decomposition...")
    ov_result = get_sentence_ov_decomposition(
        model, sae, text, layer, head,
        max_length=max_length, top_k=top_k, verbose=verbose,
        hook_point=hook_point,
        normalize_by_decoder_norm=normalize_by_decoder_norm,
    )
    ov_sparse = ov_result["ov_sparse"]

    # ── Attribution: rank features ───────────────────────────────────────

    # QK attribution: rank pairs, extract unique features
    qk_ranked_pairs = rank_feature_pairs(fra_sparse, diagonal=False, mode="sum")
    qk_top_pairs = qk_ranked_pairs[:k]
    qk_features = set()
    for q_f, k_f, *_ in qk_top_pairs:
        qk_features.add(q_f)
        qk_features.add(k_f)
    qk_features = sorted(qk_features)

    # OV attribution: rank single features
    ov_ranked = rank_ov_features(ov_sparse, mode="sum")
    ov_features = [f for f, *_ in ov_ranked[:k]]

    # Combined attribution: union of top from both
    combined_features = sorted(set(qk_features) | set(ov_features[:k]))

    if verbose:
        print(f"\nAttribution results:")
        print(f"  QK: {len(qk_features)} features from top {k} pairs")
        print(f"  OV: {len(ov_features)} top features")
        print(f"  Combined: {len(combined_features)} features")

    # ── Prepare QK scores for QK intervention ────────────────────────────
    seq_len = fra_sparse.shape[0]
    d_sae = fra_sparse.shape[2]

    # Compute bias correction using ablation_study's format
    bias = compute_bias_corrections(
        model, sae, text, layer, head, hook_point, max_length=max_length,
    )

    def make_qk_scores_with_ablation(pairs_to_ablate):
        """Ablate pairs from FRA and reconstruct scores."""
        fra_ablated = ablate_fra_pairs(fra_sparse, pairs_to_ablate, d_sae)
        fra_sum = fra_sum_to_attn(fra_ablated, seq_len)
        return reconstruct_scores(fra_sum, bias, device)

    # ── Run 3×3 matrix ───────────────────────────────────────────────────
    attr_modes = {
        "qk": qk_features,
        "ov": ov_features,
        "qk+ov": combined_features,
    }

    matrix = {}

    for attr_name, attr_features in attr_modes.items():
        matrix[attr_name] = {}

        # For QK intervention: find pairs involving the attributed features
        attr_feat_set = set(attr_features)
        qk_pairs_for_attr = [
            (q, kk) for q, kk, *_ in qk_ranked_pairs
            if q in attr_feat_set or kk in attr_feat_set
        ][:k]

        for interv_name in ["qk", "ov", "qk+ov"]:
            if verbose:
                print(f"  {attr_name} → {interv_name}...", end=" ", flush=True)

            if interv_name == "qk":
                # Patch attention scores by ablating pairs
                if qk_pairs_for_attr:
                    scores = make_qk_scores_with_ablation(qk_pairs_for_attr)
                    result = run_qk_steering(
                        model, tok_tensor, layer, head, scores,
                        shift_labels, unpatched_logits,
                    )
                else:
                    result = {"loss": unpatched_loss, "kl_div": 0.0,
                              "top1_change_frac": 0.0}

            elif interv_name == "ov":
                # Scale features to 0 in OV
                feature_scales = {f: 0.0 for f in attr_features}
                result = run_ov_steering(
                    model, sae, tok_tensor, layer, head, hook_point,
                    feature_scales, shift_labels, unpatched_logits,
                    normalize_by_decoder_norm=normalize_by_decoder_norm,
                )

            elif interv_name == "qk+ov":
                # Both: ablate QK pairs + zero OV features
                qk_scores = make_qk_scores_with_ablation(qk_pairs_for_attr) if qk_pairs_for_attr else None
                ov_scales = {f: 0.0 for f in attr_features}
                result = run_combined_steering(
                    model, sae, tok_tensor, layer, head, hook_point,
                    qk_scores=qk_scores,
                    ov_feature_scales=ov_scales,
                    shift_labels=shift_labels,
                    unpatched_logits=unpatched_logits,
                    normalize_by_decoder_norm=normalize_by_decoder_norm,
                )

            # Remove logits from result to save memory
            metrics = {kk: v for kk, v in result.items() if kk != "patched_logits"}
            matrix[attr_name][interv_name] = metrics

            if verbose:
                loss = metrics.get("loss", 0.0)
                kl = metrics.get("kl_div", 0.0)
                print(f"loss={loss:.4f}, KL={kl:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'Attribution':>12s} × Intervention → Loss (delta from {unpatched_loss:.4f}):")
        print(f"{'':>12s}  {'qk':>12s}  {'ov':>12s}  {'qk+ov':>12s}")
        for attr_name in ["qk", "ov", "qk+ov"]:
            row = []
            for interv_name in ["qk", "ov", "qk+ov"]:
                delta = matrix[attr_name][interv_name].get("loss", 0.0) - unpatched_loss
                row.append(f"{delta:+.4f}")
            print(f"{attr_name:>12s}  {'  '.join(f'{r:>12s}' for r in row)}")

    return {
        "matrix": matrix,
        "unpatched": {"loss": unpatched_loss},
        "qk_features": qk_features,
        "ov_features": ov_features,
        "combined_features": combined_features,
        "qk_top_pairs": qk_top_pairs,
        "ov_ranked": ov_ranked[:k],
    }
