"""
1Batched GPU implementation of FRA without pre-computing all features.
Only computes Q/K for active features.
"""

import torch
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

from fra.activation_utils import get_attention_activations


@torch.no_grad()
def get_sentence_fra_batch(
    model: Any,
    sae: Any,
    text: str,
    layer: int,
    head: int,
    max_length: int = 128,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Batched FRA computation that only computes Q/K for active features.
    
    Args:
        model: The transformer model
        sae: The SAE wrapper
        text: Input text to analyze
        layer: Which layer to analyze
        head: Which attention head to analyze
        max_length: Maximum sequence length
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
    
    # Get model attention weights (keep on GPU)
    W_Q = model.blocks[layer].attn.W_Q[head]  # [d_model, d_head]
    W_K = model.blocks[layer].attn.W_K[head]  # [d_model, d_head]
    
    # Find all unique active features across all positions
    if verbose:
        print("Finding active features...")
    all_active_features = set()
    position_active_features = []
    
    for pos in range(seq_len):
        active = torch.where(feature_activations[pos] != 0)[0]
        position_active_features.append(active)
        all_active_features.update(active.cpu().tolist())
    
    all_active_features = torch.tensor(sorted(all_active_features), device=device)
    n_active = len(all_active_features)
    
    if verbose:
        print(f"  Total unique active features: {n_active} out of {sae.d_sae}")
    
    # Pre-compute Q and K only for active features
    if n_active > 0:
        active_W_dec = sae.W_dec[all_active_features]  # [n_active, d_model]
        active_Q = torch.matmul(active_W_dec, W_Q)  # [n_active, d_head]
        active_K = torch.matmul(active_W_dec, W_K)  # [n_active, d_head]
        
        # Create mapping from global to local indices
        global_to_local = {feat.item(): i for i, feat in enumerate(all_active_features)}
    
    # Process lower triangle (causal attention)
    total_pairs = seq_len * (seq_len + 1) // 2
    
    if verbose:
        pbar = tqdm(total=total_pairs, desc="Computing FRA (batched)")
    
    # Accumulate in lists for COO format
    row_indices = []
    col_indices = []
    values = []
    
    pair_count = 0
    for key_idx in range(seq_len):
        for query_idx in range(key_idx, seq_len):
            query_active = position_active_features[query_idx]
            key_active = position_active_features[key_idx]
            
            if len(query_active) == 0 or len(key_active) == 0:
                pair_count += 1
                if verbose:
                    pbar.update(1)
                continue
            
            # Map to local indices
            query_local = torch.tensor([global_to_local[f.item()] for f in query_active], device=device)
            key_local = torch.tensor([global_to_local[f.item()] for f in key_active], device=device)
            
            # Get Q and K for active features
            Q_active = active_Q[query_local]  # [n_query_active, d_head]
            K_active = active_K[key_local]     # [n_key_active, d_head]
            
            # Compute attention scores
            int_matrix = torch.matmul(Q_active, K_active.T)  # [n_query_active, n_key_active]
            
            # Scale by feature activations
            q_scale = feature_activations[query_idx, query_active]
            k_scale = feature_activations[key_idx, key_active]
            int_matrix = int_matrix * q_scale.unsqueeze(1) * k_scale.unsqueeze(0)
            
            # Find non-zeros
            mask = int_matrix.abs() > 1e-10
            if mask.any():
                local_row, local_col = torch.where(mask)
                
                # Convert to global feature indices
                global_rows = query_active[local_row].cpu().tolist()
                global_cols = key_active[local_col].cpu().tolist()
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
    
    print("Testing batched FRA...")
    
    model = load_model('gpt2-small', 'cuda')
    sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
    sae = SimpleAttentionSAE(sae_data)
    
    text = "The cat sat on the mat."
    
    print(f"\nAnalyzing: '{text}'")
    t0 = time.time()
    result = get_sentence_fra_batch(model, sae, text, 5, 0, verbose=True)
    print(f"\nTime: {time.time() - t0:.2f}s")
    
    print(f"\nResults:")
    print(f"  Non-zero interactions: {result['nnz']:,}")
    print(f"  Density: {result['density']:.8%}")
    
    top = get_top_interactions_torch(result['data_dep_int_matrix'], 10)
    print(f"\nTop interactions:")
    for i, (q, k, v) in enumerate(top[:5]):
        print(f"  {i+1}. F{q} → F{k}: {v:.6f}")
    
    import os
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os._exit(0)