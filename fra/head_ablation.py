"""
Head attribution sweep for Feature-Resolved Attention.

Ablates each attention head to measure its importance for model behavior.
This helps identify which heads concentrate the effect of interest
(e.g., sleeper behavior, emergent misalignment).
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional

from transformer_lens import HookedTransformer


@torch.no_grad()
def head_attribution_sweep(
    model: HookedTransformer,
    texts: List[str],
    layer: int,
    max_length: int = 128,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Measure each head's importance by ablating its full output.

    For each head, zeros out its contribution to the residual stream
    via hook_result and measures the impact on loss/predictions.

    Args:
        model: HookedTransformer.
        texts: List of input texts to evaluate on.
        layer: Attention layer to sweep.
        max_length: Maximum sequence length.
        verbose: Print progress.

    Returns:
        List of dicts sorted by |loss_delta| descending:
            {head, loss_delta, kl_div, top1_change_frac}
        where loss_delta = ablated_loss - unpatched_loss (positive = head matters).
    """
    device = next(model.parameters()).device
    n_heads = model.cfg.n_heads
    result_hook = f"blocks.{layer}.attn.hook_result"

    head_metrics = {h: {"loss_delta": 0.0, "kl_div": 0.0, "top1_change_frac": 0.0}
                    for h in range(n_heads)}
    n_texts = len(texts)

    for text_idx, text in enumerate(texts):
        if verbose:
            print(f"  Text {text_idx + 1}/{n_texts}...")

        tokens = model.tokenizer.encode(text)
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
        tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

        if tok_tensor.shape[1] < 2:
            continue

        # Unpatched baseline
        unpatched_logits = model(tok_tensor).float()  # cast to fp32
        shift_labels = tok_tensor[0, 1:]
        unpatched_loss = F.cross_entropy(
            unpatched_logits[0, :-1], shift_labels
        ).item()

        # Ablate each head via hook_z (pre-W_O output, works with GQA)
        # hook_result may not exist in all TL versions; hook_z is more reliable
        z_hook = f"blocks.{layer}.attn.hook_z"

        for h in range(n_heads):
            def make_hook(head_idx):
                def hook_fn(z, hook):
                    # z: [batch, seq_len, n_heads, d_head]
                    out = z.clone()
                    out[0, :, head_idx, :] = 0.0
                    return out
                return hook_fn

            patched_logits = model.run_with_hooks(
                tok_tensor, fwd_hooks=[(z_hook, make_hook(h))]
            ).float()  # cast to fp32

            loss = F.cross_entropy(
                patched_logits[0, :-1], shift_labels
            ).item()

            p = F.softmax(unpatched_logits[0, :-1], dim=-1)
            q = F.softmax(patched_logits[0, :-1], dim=-1)
            kl = F.kl_div(q.log(), p, reduction="batchmean").item()

            pred_clean = unpatched_logits[0, :-1].argmax(dim=-1)
            pred_patched = patched_logits[0, :-1].argmax(dim=-1)
            top1_change = (pred_clean != pred_patched).float().mean().item()

            head_metrics[h]["loss_delta"] += (loss - unpatched_loss)
            head_metrics[h]["kl_div"] += kl
            head_metrics[h]["top1_change_frac"] += top1_change

    # Average over texts
    results = []
    for h in range(n_heads):
        results.append({
            "head": h,
            "loss_delta": head_metrics[h]["loss_delta"] / max(n_texts, 1),
            "kl_div": head_metrics[h]["kl_div"] / max(n_texts, 1),
            "top1_change_frac": head_metrics[h]["top1_change_frac"] / max(n_texts, 1),
        })

    results.sort(key=lambda x: abs(x["loss_delta"]), reverse=True)

    if verbose:
        print(f"\nHead attribution (layer {layer}), ranked by |loss_delta|:")
        for r in results[:10]:
            print(f"  H{r['head']:2d}: loss_delta={r['loss_delta']:+.4f}, "
                  f"KL={r['kl_div']:.4f}, top1_change={r['top1_change_frac']:.3f}")

    return results


@torch.no_grad()
def multi_head_ablation(
    model: HookedTransformer,
    texts: List[str],
    layer: int,
    heads_to_ablate: List[int],
    max_length: int = 128,
) -> Dict[str, float]:
    """Ablate multiple heads simultaneously and measure the combined effect.

    Useful for checking if the effect is concentrated in a few heads
    (if ablating 2-3 heads has similar effect to ablating all).

    Args:
        model: HookedTransformer.
        texts: Input texts.
        layer: Attention layer.
        heads_to_ablate: List of head indices to zero out.
        max_length: Max sequence length.

    Returns:
        dict: avg_loss_delta, avg_kl_div, avg_top1_change_frac
    """
    device = next(model.parameters()).device
    z_hook = f"blocks.{layer}.attn.hook_z"
    head_set = set(heads_to_ablate)

    total_loss_delta = 0.0
    total_kl = 0.0
    total_top1 = 0.0
    n = 0

    for text in texts:
        tokens = model.tokenizer.encode(text)
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
        tok_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
        if tok_tensor.shape[1] < 2:
            continue

        unpatched_logits = model(tok_tensor).float()
        shift_labels = tok_tensor[0, 1:]
        unpatched_loss = F.cross_entropy(
            unpatched_logits[0, :-1], shift_labels
        ).item()

        def hook_fn(z, hook):
            out = z.clone()
            for h in head_set:
                out[0, :, h, :] = 0.0
            return out

        patched_logits = model.run_with_hooks(
            tok_tensor, fwd_hooks=[(z_hook, hook_fn)]
        ).float()

        loss = F.cross_entropy(patched_logits[0, :-1], shift_labels).item()
        p = F.softmax(unpatched_logits[0, :-1], dim=-1)
        q = F.softmax(patched_logits[0, :-1], dim=-1)
        kl = F.kl_div(q.log(), p, reduction="batchmean").item()

        pred_clean = unpatched_logits[0, :-1].argmax(dim=-1)
        pred_patched = patched_logits[0, :-1].argmax(dim=-1)
        top1_change = (pred_clean != pred_patched).float().mean().item()

        total_loss_delta += (loss - unpatched_loss)
        total_kl += kl
        total_top1 += top1_change
        n += 1

    return {
        "heads_ablated": heads_to_ablate,
        "avg_loss_delta": total_loss_delta / max(n, 1),
        "avg_kl_div": total_kl / max(n, 1),
        "avg_top1_change_frac": total_top1 / max(n, 1),
    }
