"""
Feature-Resolved Attention analysis functions using sparse matrices for efficiency.
"""

import torch
import numpy as np
from scipy import sparse
from typing import Dict, Any, Tuple, List
from tqdm import tqdm

from fra.activation_utils import get_attention_activations
from fra.fra_func import analyze_feature_attention_interactions


def get_sentence_fra_sparse(
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
    Processes only the lower triangle (causal attention pattern).
    
    Instead of creating a dense [d_sae, d_sae] matrix, we store the results in a sparse format
    and only compute/store non-zero feature interactions.
    
    Args:
        model: The transformer model
        sae: The SAE wrapper
        text: Input text to analyze
        layer: Which layer to analyze
        head: Which attention head to analyze
        max_length: Maximum sequence length
        verbose: Whether to show progress
        
    Returns:
        Dictionary containing sparse matrices for:
        - 'data_dep_int_matrix': Averaged feature interactions
        - 'data_dep_int_matrix_abs': Averaged absolute feature interactions
        - 'data_dep_localization_matrix': Distance-weighted interactions
        - 'counts': How many times each feature pair was active
        - 'seq_len': Sequence length
        - 'total_pairs': Total number of position pairs analyzed
    """
    # Get activations for the text
    activations = get_attention_activations(model, text, layer=layer, max_length=max_length)
    seq_len = min(activations.shape[0], max_length)
    
    # Dictionary to accumulate sparse values
    # We'll use dictionaries with (row, col) tuples as keys for efficiency
    interaction_sum = {}
    interaction_abs_sum = {}
    localization_sum = {}
    counts = {}
    
    # Process lower triangle only (causal attention)
    total_pairs = seq_len * (seq_len + 1) // 2  # Lower triangle including diagonal
    
    if verbose:
        pbar = tqdm(total=total_pairs, desc="Computing FRA (lower triangle)")
    
    count = 0
    for key_index in range(seq_len):
        for query_index in range(key_index, seq_len):  # Lower triangle: query >= key
            # Get feature interactions for this position pair
            fra_result = analyze_feature_attention_interactions(
                model, sae, layer=layer, head=head,
                input_text=text,
                query_position=query_index,
                key_position=key_index
            )
            
            # Extract the interaction matrix and active features
            int_matrix = fra_result['interaction_matrix']
            query_active_features = fra_result['query_active_features']
            key_active_features = fra_result['key_active_features']
            
            # Distance for localization
            distance = query_index - key_index
            
            # Add to sparse accumulators
            for i, q_feat in enumerate(query_active_features):
                for j, k_feat in enumerate(key_active_features):
                    val = int_matrix[i, j]
                    if val != 0:  # Only store non-zeros
                        key = (q_feat, k_feat)
                        
                        # Update interaction sum
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
        # Extract keys and values
        keys = list(interaction_sum.keys())
        rows = [k[0] for k in keys]
        cols = [k[1] for k in keys]
        
        # Average the interactions
        avg_values = [interaction_sum[k] / count for k in keys]
        avg_abs_values = [interaction_abs_sum[k] / count for k in keys]
        
        # For localization, divide by absolute sum (with clipping to avoid divide by zero)
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
        'nnz': data_dep_int_matrix.nnz,  # Number of non-zero elements
        'density': data_dep_int_matrix.nnz / (d_sae * d_sae) if d_sae > 0 else 0
    }


def get_top_feature_interactions(
    sparse_fra: sparse.csr_matrix,
    top_k: int = 10
) -> List[Tuple[int, int, float]]:
    """
    Get the top-k strongest feature interactions from sparse FRA matrix.
    
    Args:
        sparse_fra: Sparse FRA matrix
        top_k: Number of top interactions to return
        
    Returns:
        List of (query_feature, key_feature, interaction_strength) tuples
    """
    # Convert to COO for easy access to coordinates and values
    coo = sparse_fra.tocoo()
    
    # Get indices of top-k values
    if coo.nnz > 0:
        # Get top k indices
        k = min(top_k, coo.nnz)
        top_indices = np.argpartition(-np.abs(coo.data), k-1)[:k]
        top_indices = top_indices[np.argsort(-np.abs(coo.data[top_indices]))]
        
        # Extract top interactions
        top_interactions = [
            (coo.row[idx], coo.col[idx], coo.data[idx])
            for idx in top_indices
        ]
    else:
        top_interactions = []
    
    return top_interactions


def compare_fra_to_attention(
    model: Any,
    sae: Any,
    text: str,
    layer: int,
    head: int,
    max_length: int = 128
) -> Dict[str, Any]:
    """
    Compare Feature-Resolved Attention to standard token attention.
    
    Args:
        model: The transformer model
        sae: The SAE wrapper
        text: Input text to analyze
        layer: Which layer to analyze
        head: Which attention head to analyze
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with comparison metrics
    """
    from fra.activation_utils import get_attention_pattern
    
    # Get standard attention pattern
    attn_pattern = get_attention_pattern(model, text, layer=layer, head=head, max_length=max_length)
    
    # Get FRA (sparse)
    fra_result = get_sentence_fra_sparse(model, sae, text, layer, head, max_length, verbose=False)
    
    # Compare information content
    # Standard attention: seq_len × seq_len values
    # FRA: number of non-zero feature interactions
    
    seq_len = attn_pattern.shape[0]
    standard_info = seq_len * seq_len
    fra_info = fra_result['nnz']
    
    return {
        'seq_len': seq_len,
        'standard_attention_values': standard_info,
        'fra_nonzero_interactions': fra_info,
        'fra_compression_ratio': fra_info / standard_info if standard_info > 0 else 0,
        'fra_density': fra_result['density'],
        'top_fra_interactions': get_top_feature_interactions(fra_result['sparse_fra'], top_k=10)
    }


if __name__ == "__main__":
    # Test the sparse FRA implementation
    from fra.utils import load_model, load_sae
    from fra.sae_wrapper import SimpleAttentionSAE
    
    print("Testing sparse FRA implementation...")
    
    # Load model and SAE
    model = load_model('gpt2-small', 'cuda')
    sae_data = load_sae('ckkissane/attn-saes-gpt2-small-all-layers', layer=5, device='cuda')
    sae = SimpleAttentionSAE(sae_data)
    
    # Test text - shorter for testing
    test_text = "The cat sat."
    
    # Compute sparse FRA
    print(f"\nAnalyzing text: '{test_text}'")
    fra_result = get_sentence_fra_sparse(model, sae, test_text, layer=5, head=0, verbose=True)
    
    print(f"\nResults:")
    print(f"  Sequence length: {fra_result['seq_len']}")
    print(f"  Total position pairs analyzed: {fra_result['total_pairs']}")
    print(f"  Non-zero feature interactions: {fra_result['nnz']:,}")
    print(f"  Density: {fra_result['density']:.6%}")
    print(f"  Memory saved vs dense: {(1 - fra_result['density']) * 100:.2f}%")
    
    # Get top interactions
    top_interactions = get_top_feature_interactions(fra_result['data_dep_int_matrix'], top_k=5)
    print(f"\nTop 5 feature interactions (by average strength):")
    for i, (q_feat, k_feat, value) in enumerate(top_interactions):
        print(f"  {i+1}. Feature {q_feat} → Feature {k_feat}: {value:.4f}")
    
    # Show localization info
    print(f"\nLocalization matrix non-zeros: {fra_result['data_dep_localization_matrix'].nnz:,}")
    print(f"Absolute interaction matrix non-zeros: {fra_result['data_dep_int_matrix_abs'].nnz:,}")