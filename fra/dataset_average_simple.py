#!/usr/bin/env python
"""
Simplified sparse accumulation that returns averaged tensor directly.
"""

import torch
from typing import List, Dict, Any
from transformer_lens import HookedTransformer
from tqdm import tqdm

from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch


def compute_dataset_average_simple(
    model: HookedTransformer,
    sae: SAELensAttentionSAE,
    dataset_texts: List[str],
    layer: int = 5,
    head: int = 0,
    filter_self_interactions: bool = True,
    use_absolute_values: bool = True,
    verbose: bool = True,
    top_k: int = 100
) -> Dict[str, Any]:
    """
    Compute dataset average and return top-k feature pairs directly.
    
    Returns:
        Dictionary with:
        - top_pairs: List of top feature pair dictionaries
        - num_samples: Number of samples processed
        - total_pairs: Total number of unique pairs found
    """
    d_sae = sae.sae.W_dec.shape[0]
    device = next(model.parameters()).device
    
    # Initialize sparse accumulator tensors
    interaction_sum = torch.sparse_coo_tensor(
        indices=torch.zeros((2, 0), dtype=torch.long, device=device),
        values=torch.zeros(0, dtype=torch.float32, device=device),
        size=(d_sae, d_sae),
        device=device
    )
    
    interaction_count = torch.sparse_coo_tensor(
        indices=torch.zeros((2, 0), dtype=torch.long, device=device),
        values=torch.zeros(0, dtype=torch.float32, device=device),
        size=(d_sae, d_sae),
        device=device
    )
    
    num_samples = len(dataset_texts)
    
    if verbose:
        print(f"Computing average over {num_samples} samples...")
        iterator = tqdm(range(num_samples), desc="Processing samples")
    else:
        iterator = range(num_samples)
    
    # Accumulation loop
    for sample_idx in iterator:
        text = dataset_texts[sample_idx]
        
        # Get FRA for this sample
        fra_result = get_sentence_fra_batch(
            model, sae, text, layer, head,
            max_length=128, top_k=20, verbose=False
        )
        
        if fra_result is None:
            continue
        
        # Process the sparse FRA tensor
        fra_sparse = fra_result['fra_tensor_sparse']
        
        # Extract Q-K feature interactions (dimensions 2 and 3)
        indices = fra_sparse.indices()  # [4, nnz]
        values = fra_sparse.values()  # [nnz]
        
        # Apply absolute value if requested
        if use_absolute_values:
            values = values.abs()
        
        # Get feature indices
        q_feats = indices[2, :]  # [nnz]
        k_feats = indices[3, :]  # [nnz]
        
        # Filter self-interactions if needed
        if filter_self_interactions:
            mask = q_feats != k_feats
            q_feats = q_feats[mask]
            k_feats = k_feats[mask]
            values = values[mask]
        
        if len(values) == 0:
            continue
            
        # Create 2D sparse tensor for this sample's interactions
        sample_indices = torch.stack([q_feats, k_feats], dim=0)  # [2, nnz]
        
        # Create sparse tensors
        sample_sum = torch.sparse_coo_tensor(
            indices=sample_indices,
            values=values,
            size=(d_sae, d_sae),
            device=device
        )
        
        sample_count = torch.sparse_coo_tensor(
            indices=sample_indices,
            values=torch.ones_like(values),
            size=(d_sae, d_sae),
            device=device
        )
        
        # Add to accumulators
        interaction_sum = interaction_sum + sample_sum
        interaction_count = interaction_count + sample_count
        
        # Clear GPU memory
        del fra_sparse, fra_result
        torch.cuda.empty_cache()
    
    if verbose:
        print(f"\n✅ Processed {num_samples} samples")
    
    # Coalesce once at the end
    if verbose:
        print("Finalizing sparse tensors...")
    
    interaction_sum = interaction_sum.coalesce()
    interaction_count = interaction_count.coalesce()
    
    # Compute averages directly on sparse tensors
    # Since both tensors have been coalesced and accumulated together,
    # they should have the same sparsity pattern
    sum_values = interaction_sum.values()
    count_values = interaction_count.values()
    
    # Compute averages
    avg_values = sum_values / count_values
    
    # Get indices
    indices = interaction_sum.indices()
    
    total_pairs = indices.shape[1]
    if verbose:
        print(f"Found {total_pairs:,} unique feature pairs")
    
    # Extract top-k pairs efficiently
    k = min(top_k, len(avg_values))
    top_avg_vals, top_idx = torch.topk(avg_values, k)
    
    # Build top pairs list
    top_pairs = []
    for i in range(k):
        idx = top_idx[i].item()
        q_feat = indices[0, idx].item()
        k_feat = indices[1, idx].item()
        
        top_pairs.append({
            'query_feature': q_feat,
            'key_feature': k_feat,
            'average': top_avg_vals[i].item(),
            'sum': sum_values[idx].item(),
            'count': count_values[idx].item()
        })
    
    if verbose and len(top_pairs) > 0:
        print(f"\n🔥 Top 5 feature interactions:")
        for i in range(min(5, len(top_pairs))):
            p = top_pairs[i]
            print(f"   {i+1}. F{p['query_feature']} → F{p['key_feature']}: "
                  f"{p['average']:.4f} (n={p['count']:.0f})")
    
    return {
        'top_pairs': top_pairs,
        'num_samples': num_samples,
        'total_pairs': total_pairs,
        'layer': layer,
        'head': head
    }


def create_simple_dashboard(results: Dict[str, Any], output_path: str = None) -> str:
    """Create a simple HTML dashboard from results."""
    import json
    from pathlib import Path
    
    top_pairs = results['top_pairs']
    
    # Simple HTML template
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>FRA Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            .stats {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .self {{ background-color: #ffe6e6; }}
        </style>
    </head>
    <body>
        <h1>Feature-Resolved Attention Results</h1>
        <div class="stats">
            <p>Layer: {results['layer']}, Head: {results['head']}</p>
            <p>Samples analyzed: {results['num_samples']}</p>
            <p>Total unique feature pairs: {results['total_pairs']:,}</p>
        </div>
        
        <h2>Top {len(top_pairs)} Feature Interactions</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Query Feature</th>
                <th>Key Feature</th>
                <th>Average Strength</th>
                <th>Count</th>
            </tr>
    """
    
    for i, pair in enumerate(top_pairs):
        is_self = pair['query_feature'] == pair['key_feature']
        row_class = 'class="self"' if is_self else ''
        html += f"""
            <tr {row_class}>
                <td>{i+1}</td>
                <td>F{pair['query_feature']}</td>
                <td>F{pair['key_feature']} {'(self)' if is_self else ''}</td>
                <td>{pair['average']:.4f}</td>
                <td>{pair['count']:.0f}</td>
            </tr>
        """
    
    html += """
        </table>
    </body>
    </html>
    """
    
    if output_path is None:
        output_path = Path(__file__).parent.parent / "results" / "fra_simple_dashboard.html"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"✅ Dashboard saved to: {output_path}")
    return str(output_path)