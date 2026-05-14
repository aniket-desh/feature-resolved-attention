"""
Pareto frontier measurement for Feature-Resolved Attention.

Sweeps steering coefficients across features and measures the trade-off
between coherence (cross-entropy / KL divergence) and behavior suppression.

The quality metric Q = 1 - A, where A is the area under the normalized
Pareto curve of (incoherence, behavior_remaining). Lower area = better
trade-off, so higher Q = better.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional

from transformer_lens import HookedTransformer
from fra.ov_steering import run_ov_steering, run_qk_steering, run_combined_steering


def compute_pareto_q(
    incoherence: List[float],
    behavior_remaining: List[float],
) -> float:
    """Compute Q = 1 - area under the normalized Pareto curve.

    Both axes are normalized to [0, 1]. The curve is the frontier of
    (incoherence, behavior_remaining) points. We want behavior_remaining
    to decrease (good) while incoherence increases (bad).

    The ideal case: behavior goes to 0 at no incoherence cost → Q ≈ 1.
    The worst case: behavior only goes to 0 at maximum incoherence → Q ≈ 0.

    Args:
        incoherence: List of incoherence values (e.g., KL div), increasing.
        behavior_remaining: List of behavior remaining (e.g., % sleeper).

    Returns:
        Q metric in [0, 1]. Higher is better.
    """
    if len(incoherence) < 2:
        return 0.0

    x = np.array(incoherence, dtype=np.float64)
    y = np.array(behavior_remaining, dtype=np.float64)

    # Normalize to [0, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    if x_max - x_min < 1e-10:
        return 0.0
    if y_max - y_min < 1e-10:
        # Behavior doesn't change — no trade-off possible
        return 0.0

    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)

    # Sort by x for proper integration
    order = np.argsort(x_norm)
    x_sorted = x_norm[order]
    y_sorted = y_norm[order]

    # Area under curve (trapezoidal)
    area = np.trapz(y_sorted, x_sorted)

    # Q = 1 - area (bounded to [0, 1])
    return float(np.clip(1.0 - area, 0.0, 1.0))


@torch.no_grad()
def pareto_sweep(
    model: HookedTransformer,
    sae: Any,
    texts: List[str],
    layer: int,
    head: int,
    hook_point: str,
    features: List[int],
    scale_values: List[float],
    intervention_mode: str = "ov",
    behavior_metric_fn: Optional[Callable] = None,
    max_length: int = 128,
    normalize_by_decoder_norm: bool | None = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Sweep steering coefficients and measure coherence vs behavior.

    For each scale value:
      1. Steer all specified features by that scale
      2. Measure coherence (cross-entropy loss, KL divergence)
      3. Measure behavior (via behavior_metric_fn or default top1_change)

    Args:
        model: HookedTransformer.
        sae: SAE wrapper.
        texts: List of input texts.
        layer: Attention layer.
        head: Attention head.
        hook_point: SAE hook point.
        features: List of feature indices to steer.
        scale_values: List of scaling factors (e.g. [0.0, 0.2, ..., 2.0]).
        intervention_mode: "ov", "qk", or "qk+ov".
        behavior_metric_fn: Optional callable(model, tok_tensor, patched_logits) -> float.
            If None, uses top1_change_frac as the behavior metric.
        max_length: Maximum sequence length.
        normalize_by_decoder_norm: Decoder norm flag.
        verbose: Print progress.

    Returns:
        dict with:
            scales: list of scale values
            avg_loss: list of average losses per scale
            avg_kl_div: list of average KL divergences per scale
            avg_behavior: list of average behavior metrics per scale
            pareto_q: float, the Q metric
            per_text: list of per-text results
    """
    device = next(model.parameters()).device
    scale_values = sorted(scale_values)

    # Collect results per scale
    all_losses = {s: [] for s in scale_values}
    all_kl = {s: [] for s in scale_values}
    all_behavior = {s: [] for s in scale_values}

    for text_idx, text in enumerate(texts):
        if verbose:
            print(f"Text {text_idx + 1}/{len(texts)}")

        tokens = model.tokenizer.encode(text)
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
        tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

        if tok_tensor.shape[1] < 2:
            continue

        # Unpatched baseline
        unpatched_logits = model(tok_tensor)
        shift_labels = tok_tensor[0, 1:]

        for scale in scale_values:
            feature_scales = {f: scale for f in features}

            if intervention_mode == "ov":
                result = run_ov_steering(
                    model, sae, tok_tensor, layer, head, hook_point,
                    feature_scales, shift_labels, unpatched_logits,
                    normalize_by_decoder_norm=normalize_by_decoder_norm,
                )
            elif intervention_mode == "qk+ov":
                result = run_combined_steering(
                    model, sae, tok_tensor, layer, head, hook_point,
                    qk_scores=None,  # no QK patching in sweep mode
                    ov_feature_scales=feature_scales,
                    shift_labels=shift_labels,
                    unpatched_logits=unpatched_logits,
                    normalize_by_decoder_norm=normalize_by_decoder_norm,
                )
            else:
                raise ValueError(
                    f"intervention_mode='{intervention_mode}' not supported for sweep. "
                    f"QK sweep requires pre-computed modified scores."
                )

            all_losses[scale].append(result.get("loss", 0.0))
            all_kl[scale].append(result.get("kl_div", 0.0))

            if behavior_metric_fn is not None:
                behavior = behavior_metric_fn(
                    model, tok_tensor, result["patched_logits"]
                )
            else:
                behavior = result.get("top1_change_frac", 0.0)

            all_behavior[scale].append(behavior)

    # Average across texts
    avg_loss = [np.mean(all_losses[s]) if all_losses[s] else 0.0
                for s in scale_values]
    avg_kl = [np.mean(all_kl[s]) if all_kl[s] else 0.0
              for s in scale_values]
    avg_behavior = [np.mean(all_behavior[s]) if all_behavior[s] else 0.0
                    for s in scale_values]

    # Compute Pareto Q using KL as incoherence and behavior metric
    q = compute_pareto_q(avg_kl, avg_behavior)

    result = {
        "scales": scale_values,
        "avg_loss": avg_loss,
        "avg_kl_div": avg_kl,
        "avg_behavior": avg_behavior,
        "pareto_q": q,
        "features_steered": features,
        "intervention_mode": intervention_mode,
    }

    if verbose:
        print(f"\nPareto sweep results (Q = {q:.4f}):")
        print(f"  {'Scale':>8s}  {'Loss':>8s}  {'KL':>8s}  {'Behavior':>8s}")
        for s, l, k, b in zip(scale_values, avg_loss, avg_kl, avg_behavior):
            print(f"  {s:8.2f}  {l:8.4f}  {k:8.4f}  {b:8.4f}")

    return result


@torch.no_grad()
def compare_hook_points(
    model: HookedTransformer,
    sae_dict: Dict[str, Any],
    texts: List[str],
    layer: int,
    head: int,
    scale_values: List[float],
    intervention_mode: str = "ov",
    behavior_metric_fn: Optional[Callable] = None,
    max_length: int = 128,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Run Pareto sweeps across multiple hook points / SAEs.

    This implements the comparison the authors described: measure Q at
    ln1.hook_normalized, hook_resid_pre, and hook_resid_mid.

    Args:
        model: HookedTransformer.
        sae_dict: {hook_point_name: (sae, hook_point_str)} dict.
            e.g. {"ln1": (sae_ln1, "ln1.hook_normalized"),
                   "resid_pre": (sae_pre, "hook_resid_pre")}
        texts: Input texts.
        layer: Attention layer.
        head: Attention head.
        scale_values: Steering coefficient grid.
        intervention_mode: "ov" or "qk+ov".
        behavior_metric_fn: Custom behavior metric.
        max_length: Max sequence length.
        verbose: Print progress.

    Returns:
        {hook_point_name: pareto_sweep_result_dict}
    """
    results = {}
    for name, (sae, hp) in sae_dict.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Hook point: {name} ({hp})")
            print(f"{'='*60}")

        # For each hook point, find the best single feature first
        from fra.core.ov import get_sentence_ov_decomposition, rank_ov_features

        # Rank features on the first text
        ov_result = get_sentence_ov_decomposition(
            model, sae, texts[0], layer, head,
            hook_point=hp, verbose=False,
        )
        ranked = rank_ov_features(ov_result["ov_sparse"])

        if not ranked:
            if verbose:
                print("  No active OV features found, skipping.")
            continue

        # Take the top feature for single-feature sweep
        top_feat = ranked[0][0]
        if verbose:
            print(f"  Top OV feature: {top_feat} (sum_abs={ranked[0][1]:.4f})")

        sweep_result = pareto_sweep(
            model, sae, texts, layer, head, hp,
            features=[top_feat],
            scale_values=scale_values,
            intervention_mode=intervention_mode,
            behavior_metric_fn=behavior_metric_fn,
            max_length=max_length,
            verbose=verbose,
        )

        results[name] = sweep_result

    if verbose and results:
        print(f"\n{'='*60}")
        print("Pareto Q comparison:")
        for name, r in sorted(results.items(), key=lambda x: -x[1]["pareto_q"]):
            print(f"  {name:20s}: Q = {r['pareto_q']:.4f}")

    return results
