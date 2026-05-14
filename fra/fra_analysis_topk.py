"""
FRA computation keeping only top-k features per position.
This dramatically reduces computation while keeping the most important interactions.
"""

import torch
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

from fra.activation_utils import get_attention_activations


@torch.no_grad()
def get_sentence_fra_topk(
    model: Any,
    sae: Any,
    text: str,
    layer: int,
    head: int,
    max_length: int = 128,
    top_k: int = 20,  # Keep only top k features per position
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute FRA keeping only top-k features per position.
    
    Args:
        model: The transformer model
        sae: The SAE wrapper
        text: Input text to analyze
        layer: Which layer to analyze
        head: Which attention head to analyze
        max_length: Maximum sequence length
        top_k: Number of top features to keep per position
        verbose: Whether to show progress
        
    Returns:
        Dictionary containing torch.sparse matrices
    """
    device = next(model.parameters()).device
    
    # Get activations for the text ONCE
    if verbose:
        print("Computing activations...")
    activations = get_attention_activations(model, text, layer=layer, max_length=max_length)
    seq_len = min(activations.shape[0], max_length)
    
    # Encode all positions to SAE features ONCE
    if verbose:
        print("Encoding to SAE features...")
    feature_activations = sae.encode(activations)  # [seq_len, d_sae]
    
    # Keep only top-k features per position
    if verbose:
        print(f"Keeping top {top_k} features per position...")
    
    topk_features = []
    for pos in range(seq_len):
        feat = feature_activations[pos]
        # Get top-k by absolute value
        topk_vals, topk_idx = torch.topk(feat.abs(), min(top_k, (feat != 0).sum().item()))
        # Create sparse version with only top-k
        sparse_feat = torch.zeros_like(feat)
        sparse_feat[topk_idx] = feat[topk_idx]
        topk_features.append(sparse_feat)
    
    topk_features = torch.stack(topk_features)
    
    if verbose:
        # Report sparsity
        total_active = sum((f != 0).sum().item() for f in topk_features)
        avg_active = total_active / seq_len
        print(f"  Average active features per position: {avg_active:.1f} (was: {(feature_activations != 0).sum(1).float().mean().item():.1f})")
    
    # Get model attention weights
    W_Q = model.blocks[layer].attn.W_Q[head]
    W_K = model.blocks[layer].attn.W_K[head]
    
    # Process lower triangle (causal attention)
    total_pairs = seq_len * (seq_len + 1) // 2
    
    if verbose:
        pbar = tqdm(total=total_pairs, desc=f"Computing FRA (top-{top_k})")
    
    # Accumulate in lists for COO format
    row_indices = []
    col_indices = []
    values = []
    
    pair_count = 0
    for key_idx in range(seq_len):
        for query_idx in range(key_idx, seq_len):
            q_feat = topk_features[query_idx]
            k_feat = topk_features[key_idx]
            
            q_active = torch.where(q_feat != 0)[0]
            k_active = torch.where(k_feat != 0)[0]
            
            if len(q_active) == 0 or len(k_active) == 0:
                pair_count += 1
                if verbose:
                    pbar.update(1)
                continue
            
            # Get decoder vectors for active features
            q_vecs = sae.W_dec[q_active]
            k_vecs = sae.W_dec[k_active]
            
            # Compute attention
            q_proj = torch.matmul(q_vecs, W_Q)
            k_proj = torch.matmul(k_vecs, W_K)
            int_matrix = torch.matmul(q_proj, k_proj.T)
            
            # Scale by feature activations
            int_matrix = int_matrix * q_feat[q_active].unsqueeze(1) * k_feat[k_active].unsqueeze(0)
            
            # Find non-zeros
            mask = int_matrix.abs() > 1e-10
            if mask.any():
                local_r, local_c = torch.where(mask)
                
                # Convert to global feature indices
                global_rows = q_active[local_r].cpu().tolist()
                global_cols = k_active[local_c].cpu().tolist()
                vals = int_matrix[mask].cpu().tolist()
                
                row_indices.extend(global_rows)
                col_indices.extend(global_cols)
                values.extend(vals)
            
            pair_count += 1
            if verbose:
                pbar.update(1)
    
    if verbose:
        pbar.close()
    
    # Create sparse tensors
    d_sae = sae.d_sae
    
    if len(row_indices) > 0:
        indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
        vals = torch.tensor(values, dtype=torch.float32, device=device)
        
        # Average by total pairs
        vals = vals / pair_count if pair_count > 0 else vals
        
        # Create sparse matrix
        sparse_matrix = torch.sparse_coo_tensor(
            indices, vals,
            size=(d_sae, d_sae),
            device=device,
            dtype=torch.float32
        ).coalesce()
        
        # Absolute version
        abs_vals = vals.abs()
        abs_matrix = torch.sparse_coo_tensor(
            indices, abs_vals,
            size=(d_sae, d_sae),
            device=device,
            dtype=torch.float32
        ).coalesce()
        
        nnz = sparse_matrix._nnz()
        density = nnz / (d_sae * d_sae)
    else:
        # Empty matrices
        empty_indices = torch.zeros((2, 0), dtype=torch.long, device=device)
        empty_vals = torch.zeros(0, dtype=torch.float32, device=device)
        
        sparse_matrix = torch.sparse_coo_tensor(
            empty_indices, empty_vals,
            size=(d_sae, d_sae),
            device=device
        )
        abs_matrix = sparse_matrix.clone()
        nnz = 0
        density = 0
    
    return {
        'data_dep_int_matrix': sparse_matrix,
        'data_dep_int_matrix_abs': abs_matrix,
        'seq_len': seq_len,
        'total_pairs': pair_count,
        'nnz': nnz,
        'density': density
    }


def get_top_interactions_torch(sparse_matrix, top_k=10):
    """Get top-k feature interactions from torch.sparse matrix."""
    if sparse_matrix._nnz() == 0:
        return []
    
    indices = sparse_matrix._indices()
    values = sparse_matrix._values()
    
    k = min(top_k, len(values))
    _, top_idx = torch.topk(values.abs(), k)
    
    results = []
    for idx in top_idx:
        q = indices[0, idx].item()
        k = indices[1, idx].item()
        v = values[idx].item()
        results.append((q, k, v))
    
    return results


if __name__ == "__main__":
    from fra.utils import load_model, load_sae
    from fra.sae_wrapper import SimpleAttentionSAE
    import time
    
    torch.set_grad_enabled(False)
    
    print("Testing top-k FRA...")
    
    model = load_model('gpt2-small', 'cuda')
    sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
    sae = SimpleAttentionSAE(sae_data)
    
    text = "The cat sat on the mat. The cat was happy."
    
    print(f"\nAnalyzing: '{text}'")
    
    # Test with different k values
    for k in [10, 20, 50]:
        print(f"\n--- Top-{k} features per position ---")
        t0 = time.time()
        result = get_sentence_fra_topk(model, sae, text, 5, 0, top_k=k, verbose=True)
        elapsed = time.time() - t0
        
        print(f"Time: {elapsed:.2f}s")
        print(f"Non-zero interactions: {result['nnz']:,}")
        print(f"Density: {result['density']:.8%}")
        
        top = get_top_interactions_torch(result['data_dep_int_matrix'], 5)
        print(f"Top 5 interactions:")
        for i, (q, k, v) in enumerate(top):
            print(f"  {i+1}. F{q} → F{k}: {v:.6f}")
    
    import os
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os._exit(0)