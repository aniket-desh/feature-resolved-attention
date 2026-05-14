"""
GPU-optimized Feature-Resolved Attention using torch.sparse for parallelization.
Keeps everything in PyTorch to leverage GPU acceleration.
"""

import torch
import torch.sparse as sparse
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

from fra.activation_utils import get_attention_activations


@torch.no_grad()  # Disable gradient computation for entire function
def get_sentence_fra_gpu(
    model: Any,
    sae: Any,
    text: str,
    layer: int,
    head: int,
    max_length: int = 128,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    GPU-optimized Feature-Resolved Attention computation.
    Uses batched operations and torch.sparse for efficiency.
    
    Args:
        model: The transformer model
        sae: The SAE wrapper
        text: Input text to analyze
        layer: Which layer to analyze
        head: Which attention head to analyze
        max_length: Maximum sequence length
        verbose: Whether to show progress
        
    Returns:
        Dictionary containing sparse tensors and statistics
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
    
    # Pre-compute Q and K projections for ALL features at once
    if verbose:
        print("Pre-computing Q and K projections for all features...")
    
    # Project all SAE decoder vectors through Q and K matrices
    # sae.W_dec is [d_sae, d_model]
    all_Q = torch.matmul(sae.W_dec, W_Q)  # [d_sae, d_head]
    all_K = torch.matmul(sae.W_dec, W_K)  # [d_sae, d_head]
    
    # Process lower triangle (causal attention)
    total_pairs = seq_len * (seq_len + 1) // 2
    
    if verbose:
        pbar = tqdm(total=total_pairs, desc="Computing FRA (GPU-optimized)")
    
    # Use lists to accumulate COO format data
    # This avoids creating large intermediate tensors
    row_indices = []
    col_indices = []
    values = []
    
    # Process position pairs one at a time to avoid memory issues
    pair_count = 0
    for key_idx in range(seq_len):
        for query_idx in range(key_idx, seq_len):
            # Get active features for this position pair
            query_features = feature_activations[query_idx]
            key_features = feature_activations[key_idx]
            
            query_active = torch.where(query_features != 0)[0]
            key_active = torch.where(key_features != 0)[0]
            
            if len(query_active) == 0 or len(key_active) == 0:
                pair_count += 1
                if verbose:
                    pbar.update(1)
                continue
            
            # Get pre-computed Q and K for active features only
            Q_active = all_Q[query_active]  # [n_query_active, d_head]
            K_active = all_K[key_active]     # [n_key_active, d_head]
            
            # Compute attention scores using batched matrix multiply
            int_matrix_unscaled = torch.matmul(Q_active, K_active.T)  # [n_query_active, n_key_active]
            
            # Scale by feature activations (outer product)
            q_scale = query_features[query_active].unsqueeze(1)
            k_scale = key_features[key_active].unsqueeze(0)
            int_matrix = int_matrix_unscaled * q_scale * k_scale
            
            # Find non-zero interactions (with threshold for numerical stability)
            nonzero_mask = int_matrix.abs() > 1e-10
            local_row, local_col = torch.where(nonzero_mask)
            
            if len(local_row) > 0:
                # Convert local indices to global feature indices
                global_rows = query_active[local_row].cpu().tolist()
                global_cols = key_active[local_col].cpu().tolist()
                vals = int_matrix[local_row, local_col].cpu().tolist()
                
                # Accumulate in lists (more memory efficient than concatenating tensors)
                row_indices.extend(global_rows)
                col_indices.extend(global_cols)
                values.extend(vals)
            
            # Free intermediate tensors
            del int_matrix, int_matrix_unscaled, Q_active, K_active
            if 'q_scale' in locals():
                del q_scale, k_scale
            
            pair_count += 1
            if verbose:
                pbar.update(1)
    
    if verbose:
        pbar.close()
    
    # Construct sparse tensors from accumulated results
    d_sae = sae.d_sae
    
    if len(row_indices) > 0:
        # Convert lists to tensors
        indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
        vals = torch.tensor(values, dtype=torch.float32, device=device)
        
        # Create sparse COO tensor and coalesce to combine duplicates
        sum_matrix = torch.sparse_coo_tensor(
            indices, vals, 
            size=(d_sae, d_sae),
            device=device,
            dtype=torch.float32
        ).coalesce()
        
        # For absolute values
        abs_vals = vals.abs()
        abs_matrix = torch.sparse_coo_tensor(
            indices, abs_vals,
            size=(d_sae, d_sae),
            device=device,
            dtype=torch.float32
        ).coalesce()
        
        # Count matrix (ones for each entry)
        ones = torch.ones_like(vals)
        count_matrix = torch.sparse_coo_tensor(
            indices, ones,
            size=(d_sae, d_sae),
            device=device,
            dtype=torch.float32
        ).coalesce()
        
        nnz = sum_matrix._nnz()
        density = nnz / (d_sae * d_sae) if d_sae > 0 else 0
        
    else:
        # Empty sparse matrices
        empty_indices = torch.zeros((2, 0), dtype=torch.long, device=device)
        empty_vals = torch.zeros(0, dtype=torch.float32, device=device)
        
        sum_matrix = torch.sparse_coo_tensor(
            empty_indices, empty_vals,
            size=(d_sae, d_sae),
            device=device,
            dtype=torch.float32
        )
        abs_matrix = sum_matrix.clone()
        count_matrix = sum_matrix.clone()
        nnz = 0
        density = 0
    
    # To get averaged values, we'd need to divide sum by counts
    # For now, return the sum (since sparse doesn't support element-wise division easily)
    avg_matrix = sum_matrix / pair_count if pair_count > 0 else sum_matrix
    
    return {
        'data_dep_int_matrix': avg_matrix,
        'data_dep_int_matrix_abs': abs_matrix / pair_count if pair_count > 0 else abs_matrix,
        'counts': count_matrix,
        'seq_len': seq_len,
        'total_pairs': pair_count,
        'nnz': nnz,
        'density': density,
        'device': device
    }


@torch.no_grad()
def get_top_feature_interactions_gpu(
    sparse_matrix: torch.sparse.Tensor,
    top_k: int = 10
) -> List[Tuple[int, int, float]]:
    """
    Get top-k interactions from a torch.sparse tensor.
    
    Args:
        sparse_matrix: Sparse tensor of interactions
        top_k: Number of top interactions to return
        
    Returns:
        List of (query_feature, key_feature, value) tuples
    """
    if sparse_matrix._nnz() == 0:
        return []
    
    # Get indices and values
    indices = sparse_matrix._indices()
    values = sparse_matrix._values()
    
    # Get top-k by absolute value
    k = min(top_k, len(values))
    top_values, top_indices = torch.topk(values.abs(), k)
    
    # Extract the feature pairs
    results = []
    for i in range(k):
        idx = top_indices[i]
        q_feat = indices[0, idx].item()
        k_feat = indices[1, idx].item()
        value = values[idx].item()
        results.append((q_feat, k_feat, value))
    
    return results


if __name__ == "__main__":
    # Test the GPU-optimized implementation
    from fra.utils import load_model, load_sae
    from fra.sae_wrapper import SimpleAttentionSAE
    import time
    
    print("Testing GPU-optimized FRA implementation...")
    
    # Ensure no gradients are computed
    torch.set_grad_enabled(False)
    
    # Load model and SAE
    model = load_model('gpt2-small', 'cuda')
    sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
    sae = SimpleAttentionSAE(sae_data)
    
    # Test text
    test_text = "The cat sat on the mat."
    
    # Time the computation
    print(f"\nAnalyzing text: '{test_text}'")
    t0 = time.time()
    fra_result = get_sentence_fra_gpu(model, sae, test_text, layer=5, head=0, verbose=True)
    elapsed = time.time() - t0
    
    print(f"\nResults:")
    print(f"  Computation time: {elapsed:.2f}s")
    print(f"  Sequence length: {fra_result['seq_len']}")
    print(f"  Total position pairs: {fra_result['total_pairs']}")
    print(f"  Non-zero interactions: {fra_result['nnz']:,}")
    print(f"  Density: {fra_result['density']:.6%}")
    
    # Get top interactions
    top_interactions = get_top_feature_interactions_gpu(
        fra_result['data_dep_int_matrix'], 
        top_k=5
    )
    print(f"\nTop 5 feature interactions:")
    for i, (q_feat, k_feat, value) in enumerate(top_interactions):
        print(f"  {i+1}. Feature {q_feat} → Feature {k_feat}: {value:.4f}")
    
    import os
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os._exit(0)