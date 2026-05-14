"""
Optimized Feature-Resolved Attention analysis using sparse matrices.
Main optimization: compute all activations once, not per position pair.
"""

import torch
import numpy as np
from scipy import sparse
from typing import Dict, Any, Tuple, List
from tqdm import tqdm

from fra.activation_utils import get_attention_activations


def get_sentence_fra_sparse_fast(
    model: Any,
    sae: Any,
    text: str,
    layer: int,
    head: int,
    max_length: int = 128,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute Feature-Resolved Attention for an entire sentence using sparse matrices.
    Optimized version that computes activations once.
    
    Args:
        model: The transformer model
        sae: The SAE wrapper
        text: Input text to analyze
        layer: Which layer to analyze
        head: Which attention head to analyze
        max_length: Maximum sequence length
        verbose: Whether to show progress
        
    Returns:
        Dictionary containing sparse matrices
    """
    # Get activations for the text ONCE
    if verbose:
        print("Computing activations...")
    activations = get_attention_activations(model, text, layer=layer, max_length=max_length)
    seq_len = min(activations.shape[0], max_length)
    
    # Encode all positions to SAE features ONCE
    if verbose:
        print("Encoding to SAE features...")
    feature_activations = sae.encode(activations)  # [seq_len, d_sae]
    
    # Get model attention weights
    W_Q = model.blocks[layer].attn.W_Q[head]
    b_Q = model.blocks[layer].attn.b_Q[head]
    W_K = model.blocks[layer].attn.W_K[head]
    b_K = model.blocks[layer].attn.b_K[head]
    
    # Dictionary to accumulate sparse values
    interaction_sum = {}
    interaction_abs_sum = {}
    localization_sum = {}
    counts = {}
    
    # Process lower triangle only (causal attention)
    total_pairs = seq_len * (seq_len + 1) // 2
    
    if verbose:
        pbar = tqdm(total=total_pairs, desc="Computing FRA (optimized)")
    
    count = 0
    for key_index in range(seq_len):
        for query_index in range(key_index, seq_len):  # Lower triangle
            # Get active features for this position pair
            query_features = feature_activations[query_index]
            key_features = feature_activations[key_index]
            
            query_active = torch.where(query_features != 0)[0].cpu().numpy()
            key_active = torch.where(key_features != 0)[0].cpu().numpy()
            
            if len(query_active) == 0 or len(key_active) == 0:
                count += 1
                if verbose:
                    pbar.update(1)
                continue
            
            # Get decoder vectors for active features only
            query_vecs = sae.W_dec[query_active]  # [n_query_active, d_in]
            key_vecs = sae.W_dec[key_active]      # [n_key_active, d_in]
            
            # Compute attention scores for active features
            q = torch.einsum('da,nd->na', W_Q, query_vecs)  # [n_query_active, d_head]
            k = torch.einsum('da,nd->na', W_K, key_vecs)    # [n_key_active, d_head]
            
            # Compute interaction matrix (unscaled)
            int_matrix_unscaled = torch.einsum('qa,ka->qk', q, k).detach().cpu().numpy()
            
            # Scale by feature activations
            scaling = (query_features[query_active].unsqueeze(1) * 
                      key_features[key_active].unsqueeze(0)).detach().cpu().numpy()
            
            int_matrix = int_matrix_unscaled * scaling
            
            # Distance for localization
            distance = query_index - key_index
            
            # Add to sparse accumulators
            for i, q_feat in enumerate(query_active):
                for j, k_feat in enumerate(key_active):
                    val = int_matrix[i, j]
                    if abs(val) > 1e-10:  # Only store non-zeros
                        key = (q_feat, k_feat)
                        
                        if key not in interaction_sum:
                            interaction_sum[key] = 0
                            interaction_abs_sum[key] = 0
                            localization_sum[key] = 0
                            counts[key] = 0
                        
                        interaction_sum[key] += val
                        interaction_abs_sum[key] += abs(val)
                        localization_sum[key] += abs(val) * distance
                        counts[key] += 1
            
            count += 1
            if verbose:
                pbar.update(1)
    
    if verbose:
        pbar.close()
    
    # Create sparse matrices from dictionaries
    d_sae = sae.d_sae
    
    if len(interaction_sum) > 0:
        keys = list(interaction_sum.keys())
        rows = [k[0] for k in keys]
        cols = [k[1] for k in keys]
        
        # Average the interactions
        avg_values = [interaction_sum[k] / count for k in keys]
        avg_abs_values = [interaction_abs_sum[k] / count for k in keys]
        
        # For localization, divide by absolute sum
        loc_values = []
        for k in keys:
            abs_sum = interaction_abs_sum[k]
            if abs_sum > 0:
                loc_values.append(localization_sum[k] / abs_sum)
            else:
                loc_values.append(0)
        
        # Create sparse matrices
        data_dep_int_matrix = sparse.coo_matrix(
            (avg_values, (rows, cols)),
            shape=(d_sae, d_sae)
        ).tocsr()
        
        data_dep_int_matrix_abs = sparse.coo_matrix(
            (avg_abs_values, (rows, cols)),
            shape=(d_sae, d_sae)
        ).tocsr()
        
        data_dep_localization_matrix = sparse.coo_matrix(
            (loc_values, (rows, cols)),
            shape=(d_sae, d_sae)
        ).tocsr()
        
        counts_matrix = sparse.coo_matrix(
            ([counts[k] for k in keys], (rows, cols)),
            shape=(d_sae, d_sae)
        ).tocsr()
        
    else:
        # Empty sparse matrices if no interactions
        data_dep_int_matrix = sparse.csr_matrix((d_sae, d_sae))
        data_dep_int_matrix_abs = sparse.csr_matrix((d_sae, d_sae))
        data_dep_localization_matrix = sparse.csr_matrix((d_sae, d_sae))
        counts_matrix = sparse.csr_matrix((d_sae, d_sae))
    
    return {
        'data_dep_int_matrix': data_dep_int_matrix,
        'data_dep_int_matrix_abs': data_dep_int_matrix_abs,
        'data_dep_localization_matrix': data_dep_localization_matrix,
        'counts': counts_matrix,
        'seq_len': seq_len,
        'total_pairs': count,
        'nnz': data_dep_int_matrix.nnz,
        'density': data_dep_int_matrix.nnz / (d_sae * d_sae) if d_sae > 0 else 0
    }


if __name__ == "__main__":
    # Test the optimized implementation
    from fra.utils import load_model, load_sae
    from fra.sae_wrapper import SimpleAttentionSAE
    
    print("Testing optimized FRA implementation...")
    
    # Load model and SAE
    model = load_model('gpt2-small', 'cuda')
    sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
    sae = SimpleAttentionSAE(sae_data)
    
    # Test text
    test_text = "The cat sat."
    
    # Compute optimized FRA
    print(f"\nAnalyzing text: '{test_text}'")
    fra_result = get_sentence_fra_sparse_fast(model, sae, test_text, layer=5, head=0, verbose=True)
    
    print(f"\nResults:")
    print(f"  Sequence length: {fra_result['seq_len']}")
    print(f"  Total position pairs: {fra_result['total_pairs']}")
    print(f"  Non-zero interactions: {fra_result['nnz']:,}")
    print(f"  Density: {fra_result['density']:.6%}")
    
    # Get top interactions
    from fra.fra_analysis import get_top_feature_interactions
    top_interactions = get_top_feature_interactions(fra_result['data_dep_int_matrix'], top_k=5)
    print(f"\nTop 5 feature interactions:")
    for i, (q_feat, k_feat, value) in enumerate(top_interactions):
        print(f"  {i+1}. Feature {q_feat} → Feature {k_feat}: {value:.4f}")
    
    import os
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os._exit(0)