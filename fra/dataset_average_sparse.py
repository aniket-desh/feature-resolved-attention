#!/usr/bin/env python
"""
Sparse tensor accumulation version - uses sparse tensors throughout.
"""

import torch
from typing import List, Dict, Any
from transformer_lens import HookedTransformer
from tqdm import tqdm
import numpy as np

from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch


def compute_dataset_average_sparse(
    model: HookedTransformer,
    sae: SAELensAttentionSAE,
    dataset_texts: List[str],
    layer: int = 5,
    head: int = 0,
    filter_self_interactions: bool = True,
    use_absolute_values: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Fast computation using sparse tensor accumulation.
    
    Args:
        model: The transformer model
        sae: SAE for feature extraction
        dataset_texts: List of texts to analyze
        layer: Layer to analyze
        head: Head to analyze
        filter_self_interactions: Whether to filter self-interactions
        use_absolute_values: Whether to use absolute values (True) or signed values (False)
        verbose: Whether to show progress
    """
    d_sae = sae.sae.W_dec.shape[0]
    device = next(model.parameters()).device
    
    # Initialize sparse accumulator tensors
    # We'll accumulate sum and count separately
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
        print(f"Computing average over {num_samples} samples (sparse accumulation)...")
        iterator = tqdm(range(num_samples), desc="Processing samples")
    else:
        iterator = range(num_samples)
    
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
        
        # Create sparse tensor for sum
        sample_sum = torch.sparse_coo_tensor(
            indices=sample_indices,
            values=values,
            size=(d_sae, d_sae),
            device=device
        )
        
        # Create sparse tensor for count (ones where we have values)
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
        del fra_sparse
        del fra_result
        torch.cuda.empty_cache()
    
    # Convert sparse accumulators to dictionaries for final processing
    if verbose:
        print(f"\n✅ Processed {num_samples} samples")
        
        # Coalesce to combine duplicate indices
        interaction_sum = interaction_sum.coalesce()
        interaction_count = interaction_count.coalesce()
        
        # Convert to COO format and extract non-zero entries
        sum_indices = interaction_sum.indices()
        sum_values = interaction_sum.values()
        count_indices = interaction_count.indices()
        count_values = interaction_count.values()
        
        print(f"   Found {sum_indices.shape[1]} unique feature pairs")
        
        # Create dictionary for easy lookup
        interaction_dict = {}
        for i in range(sum_indices.shape[1]):
            q_feat = sum_indices[0, i].item()
            k_feat = sum_indices[1, i].item()
            sum_val = sum_values[i].item()
            # Find corresponding count
            mask = (count_indices[0] == q_feat) & (count_indices[1] == k_feat)
            count_val = count_values[mask].item() if mask.any() else 1
            
            avg = sum_val / count_val
            interaction_dict[(q_feat, k_feat)] = {
                'sum': sum_val,
                'count': count_val,
                'avg': avg
            }
        
        # Show top averages
        avg_list = [(k, v['avg'], v['count']) for k, v in interaction_dict.items()]
        avg_list.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🔥 Top 5 average interactions:")
        for i, ((q_feat, k_feat), avg, count) in enumerate(avg_list[:5]):
            print(f"   {i+1}. Features ({q_feat}, {k_feat}): {avg:.4f} (seen {count} times)")
    
    return {
        'interaction_sum_sparse': interaction_sum,
        'interaction_count_sparse': interaction_count,
        'num_samples': num_samples,
        'd_sae': d_sae
    }


if __name__ == "__main__":
    # Test with 5 samples
    import time
    
    torch.set_grad_enabled(False)
    
    print("Testing SPARSE dataset average computation...")
    print("="*60)
    
    # Load model and SAE
    print("\nLoading model and SAE...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
    sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", "blocks.5.hook_z", device="cuda")
    print(f"Loading time: {time.time()-t0:.1f}s")
    
    # Test texts
    test_texts = [
        "The cat sat on the mat. The cat was happy.",
        "Alice went to the store. She bought milk.",
        "The student who studied hard passed. The students who studied hard passed.",
        "Bob likes to play chess. Bob is very good at chess.",
        "The weather today is sunny. The weather yesterday was rainy."
    ]
    
    # Test with absolute values (default)
    print(f"\n1. Computing averages with ABSOLUTE values...")
    print("-"*40)
    t0 = time.time()
    
    results_abs = compute_dataset_average_sparse(
        model=model,
        sae=sae,
        dataset_texts=test_texts,
        layer=5,
        head=0,
        filter_self_interactions=True,
        use_absolute_values=True,
        verbose=True
    )
    
    t1 = time.time()
    print(f"\n⏱️ Time with abs: {t1-t0:.1f}s ({(t1-t0)/len(test_texts):.1f}s per sample)")
    
    # Test with signed values
    print(f"\n2. Computing averages with SIGNED values...")
    print("-"*40)
    t0 = time.time()
    
    results_signed = compute_dataset_average_sparse(
        model=model,
        sae=sae,
        dataset_texts=test_texts,
        layer=5,
        head=0,
        filter_self_interactions=True,
        use_absolute_values=False,
        verbose=True
    )
    
    t1 = time.time()
    print(f"\n⏱️ Time with signed: {t1-t0:.1f}s ({(t1-t0)/len(test_texts):.1f}s per sample)")
    
    print(f"\n✅ Both should be ~3-4s per sample with sparse accumulation!")