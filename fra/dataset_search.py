#!/usr/bin/env python
"""
Search across dataset for large feature interactions in FRA.

This module computes FRA over many samples and tracks:
1. Average interaction strength for each feature pair (when non-zero)
2. Top-k strongest interactions with their sample indices and positions
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from transformer_lens import HookedTransformer
from tqdm import tqdm
from dataclasses import dataclass
import pickle
from datetime import datetime

from fra.induction_head import SAELensAttentionSAE
from fra.fra_func import get_sentence_fra_batch
from fra.utils import load_dataset_hf


@dataclass
class InteractionExample:
    """Store an example of a strong feature interaction."""
    sample_idx: int
    text: str
    query_pos: int
    key_pos: int
    strength: float
    query_token: str
    key_token: str
    
    
@dataclass 
class DatasetSearchResults:
    """Results from searching dataset for feature interactions."""
    # Average interaction strength for each feature pair (when non-zero)
    avg_interactions: torch.Tensor  # [d_sae, d_sae]
    
    # Count of non-zero occurrences for each feature pair
    interaction_counts: torch.Tensor  # [d_sae, d_sae]
    
    # Top-k examples for each feature pair
    # Dict mapping (q_feat, k_feat) -> List[InteractionExample]
    top_examples: Dict[Tuple[int, int], List[InteractionExample]]
    
    # Metadata
    layer: int
    head: int
    num_samples: int
    d_sae: int


def search_dataset_for_interactions(
    model: HookedTransformer,
    sae: SAELensAttentionSAE,
    dataset_texts: List[str],
    layer: int = 5,
    head: int = 0,
    num_samples: int = 100,
    top_k_per_pair: int = 5,
    batch_size: int = 1,
    device: str = "cuda",
    filter_self_interactions: bool = True,
    verbose: bool = True
) -> DatasetSearchResults:
    """
    Search dataset for strong feature interactions.
    
    Args:
        model: The transformer model
        sae: SAE for feature extraction
        dataset_texts: List of texts to analyze
        layer: Layer to analyze
        head: Head to analyze
        num_samples: Number of samples to process
        top_k_per_pair: Number of top examples to keep per feature pair
        batch_size: Batch size for processing
        device: Device to use
        verbose: Whether to show progress
        
    Returns:
        DatasetSearchResults with interaction statistics and examples
    """
    d_sae = sae.sae.W_dec.shape[0]
    
    # Initialize accumulators using dictionaries for sparse storage
    # This avoids creating huge tensors
    interaction_sum = {}  # (q_feat, k_feat) -> sum of strengths
    interaction_count = {}  # (q_feat, k_feat) -> count
    
    # Store top examples for each feature pair
    # We'll keep a running list of top-k for efficiency
    top_examples = {}  # (q_feat, k_feat) -> List[InteractionExample]
    
    # Process samples
    num_samples = min(num_samples, len(dataset_texts))
    
    if verbose:
        print(f"Searching {num_samples} samples for feature interactions...")
        iterator = tqdm(range(0, num_samples, batch_size), desc="Processing samples")
    else:
        iterator = range(0, num_samples, batch_size)
    
    for batch_start in iterator:
        batch_end = min(batch_start + batch_size, num_samples)
        batch_texts = dataset_texts[batch_start:batch_end]
        
        for idx_in_batch, text in enumerate(batch_texts):
            sample_idx = batch_start + idx_in_batch
            
            try:
                # Get FRA for this sample with reduced top_k for memory efficiency
                fra_result = get_sentence_fra_batch(
                    model, sae, text, layer, head,
                    max_length=128, top_k=20, verbose=False  # Reduced top_k for memory
                )
                
                if fra_result is None:
                    continue
                
                # Get tokens for this sample
                tokens = model.tokenizer.encode(text, truncation=True, max_length=128)
                token_strs = [model.tokenizer.decode(t) for t in tokens]
                
                # Process the sparse FRA tensor
                seq_len = fra_result['seq_len']
                fra_sparse = fra_result['fra_tensor_sparse']
                
                # Convert sparse tensor to COO format for easier processing
                indices = fra_sparse.indices()  # [4, nnz]
                values = fra_sparse.values()    # [nnz]
                
                # Process all non-zero interactions at once with vectorized operations
                # Extract all indices and values at once
                q_positions = indices[0, :].cpu()
                k_positions = indices[1, :].cpu()
                q_feats = indices[2, :].cpu()
                k_feats = indices[3, :].cpu()
                strengths = values.abs().cpu()
                
                # Filter self-interactions if needed
                if filter_self_interactions:
                    mask = q_feats != k_feats
                    q_positions = q_positions[mask]
                    k_positions = k_positions[mask]
                    q_feats = q_feats[mask]
                    k_feats = k_feats[mask]
                    strengths = strengths[mask]
                
                # Convert to numpy for faster operations
                q_positions_np = q_positions.numpy()
                k_positions_np = k_positions.numpy()
                q_feats_np = q_feats.numpy()
                k_feats_np = k_feats.numpy()
                strengths_np = strengths.numpy()
                
                # Process all interactions for this sample
                for q_pos, k_pos, q_feat, k_feat, strength in zip(
                    q_positions_np, k_positions_np, q_feats_np, k_feats_np, strengths_np
                ):
                    # Skip if positions are out of bounds
                    if q_pos >= len(token_strs) or k_pos >= len(token_strs):
                        continue
                    
                    # Update running statistics
                    pair_key = (int(q_feat), int(k_feat))
                    if pair_key in interaction_sum:
                        interaction_sum[pair_key] += strength
                        interaction_count[pair_key] += 1
                    else:
                        interaction_sum[pair_key] = strength
                        interaction_count[pair_key] = 1
                    
                    # Create example
                    example = InteractionExample(
                        sample_idx=sample_idx,
                        text=text[:200],  # Truncate for storage
                        query_pos=int(q_pos),
                        key_pos=int(k_pos),
                        strength=float(strength),
                        query_token=token_strs[q_pos] if q_pos < len(token_strs) else "",
                        key_token=token_strs[k_pos] if k_pos < len(token_strs) else ""
                    )
                    
                    # Update top examples for this feature pair
                    if pair_key not in top_examples:
                        top_examples[pair_key] = []
                    
                    examples_list = top_examples[pair_key]
                    
                    # Only keep if it's potentially in top-k (quick check without sorting)
                    if len(examples_list) < top_k_per_pair:
                        examples_list.append(example)
                    else:
                        # Only add if stronger than weakest current example
                        min_strength = min(abs(ex.strength) for ex in examples_list)
                        if abs(strength) > min_strength:
                            examples_list.append(example)
                            # Only sort and trim occasionally to avoid repeated sorting
                            if len(examples_list) > top_k_per_pair * 2:
                                examples_list.sort(key=lambda x: abs(x.strength), reverse=True)
                                top_examples[pair_key] = examples_list[:top_k_per_pair]
                
                # CRITICAL: Delete the FRA tensor to free memory
                del fra_sparse
                del fra_result
                torch.cuda.empty_cache() if device == "cuda" else None
                
            except Exception as e:
                if verbose:
                    print(f"Error processing sample {sample_idx}: {e}")
                continue
    
    # Convert sparse dictionaries to sparse tensors for storage
    # This is much more memory efficient than dense tensors
    indices = []
    values_sum = []
    values_count = []
    
    for (q_feat, k_feat), sum_val in interaction_sum.items():
        indices.append([q_feat, k_feat])
        values_sum.append(sum_val)
        values_count.append(interaction_count[(q_feat, k_feat)])
    
    if indices:
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device).t()
        sum_tensor = torch.sparse_coo_tensor(indices_tensor, values_sum, (d_sae, d_sae), device=device)
        count_tensor = torch.sparse_coo_tensor(indices_tensor, values_count, (d_sae, d_sae), device=device)
        
        # Compute averages sparsely
        avg_values = [s/c for s, c in zip(values_sum, values_count)]
        avg_tensor = torch.sparse_coo_tensor(indices_tensor, avg_values, (d_sae, d_sae), device=device)
    else:
        # Empty sparse tensors if no interactions found
        avg_tensor = torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long), [], (d_sae, d_sae))
        count_tensor = torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long), [], (d_sae, d_sae))
    
    # Final sort and trim of all example lists
    for pair_key in top_examples:
        examples_list = top_examples[pair_key]
        if len(examples_list) > top_k_per_pair:
            examples_list.sort(key=lambda x: abs(x.strength), reverse=True)
            top_examples[pair_key] = examples_list[:top_k_per_pair]
        else:
            examples_list.sort(key=lambda x: abs(x.strength), reverse=True)
    
    # ALWAYS keep as sparse tensors - never convert to dense
    # Move to CPU to free GPU memory
    avg_sparse_cpu = avg_tensor.cpu()
    count_sparse_cpu = count_tensor.cpu()
    
    results = DatasetSearchResults(
        avg_interactions=avg_sparse_cpu,  # Keep sparse!
        interaction_counts=count_sparse_cpu,  # Keep sparse!
        top_examples=top_examples,
        layer=layer,
        head=head,
        num_samples=num_samples,
        d_sae=d_sae
    )
    
    if verbose:
        # Print summary statistics
        print(f"\n✅ Search complete!")
        print(f"   Processed {num_samples} samples")
        print(f"   Found {len(top_examples)} unique feature pairs with interactions")
        total_interactions = sum(interaction_count.values())
        print(f"   Total non-zero interactions: {total_interactions:,.0f}")
        # Calculate sparsity
        nonzero_count = len(interaction_count)
        total_elements = d_sae * d_sae
        sparsity = (1 - nonzero_count / total_elements) * 100
        print(f"   Sparsity: {sparsity:.4f}%")
        
        # Find strongest average interactions from dictionary
        if interaction_sum:
            # Compute averages and sort
            avg_interactions_list = []
            for (q_feat, k_feat), sum_val in interaction_sum.items():
                count = interaction_count[(q_feat, k_feat)]
                avg = sum_val / count
                avg_interactions_list.append(((q_feat, k_feat), avg, count))
            
            # Sort by average strength
            avg_interactions_list.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🔥 Top average interactions:")
            for i, ((q_feat, k_feat), avg, count) in enumerate(avg_interactions_list[:10]):
                print(f"   {i+1}. Features ({q_feat}, {k_feat}): {avg:.4f} (seen {count} times)")
        else:
            print("\n No non-zero interactions found!")
    
    return results


def save_search_results(results: DatasetSearchResults, filepath: str):
    """Save search results to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filepath}")


def load_search_results(filepath: str) -> DatasetSearchResults:
    """Load search results from disk."""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    return results


def run_dataset_search(
    model: Optional[HookedTransformer] = None,
    sae: Optional[SAELensAttentionSAE] = None,
    layer: int = 5,
    head: int = 0,
    num_samples: int = 100,
    dataset_path: Optional[str] = None,
    save_path: Optional[str] = None,
    filter_self_interactions: bool = True
) -> DatasetSearchResults:
    """
    Convenience function to run dataset search with automatic loading.
    
    Args:
        model: Optional pre-loaded model
        sae: Optional pre-loaded SAE
        layer: Layer to analyze
        head: Head to analyze
        num_samples: Number of samples to process
        dataset_path: Path to dataset (uses default if None)
        save_path: Path to save results (auto-generates if None)
        
    Returns:
        Search results
    """
    # Load model if needed
    if model is None:
        print("Loading model...")
        model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
    
    # Load SAE if needed
    if sae is None:
        print(f"Loading SAE for layer {layer}...")
        sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", f"blocks.{layer}.hook_z", device="cuda")
    
    # Load dataset
    print("Loading dataset...")
    if dataset_path:
        with open(dataset_path, 'r') as f:
            dataset_texts = [line.strip() for line in f if line.strip()]
    else:
        # Use default dataset loader from HuggingFace (streaming)
        dataset = load_dataset_hf(streaming=True)  # Use streaming to avoid loading everything
        # Extract text from dataset
        dataset_texts = []
        for i, item in enumerate(dataset):
            if i >= num_samples:  # Get exactly as many as needed
                break
            text = item.get('text', '') if isinstance(item, dict) else str(item)
            if len(text.split()) > 10:  # Filter out very short texts
                dataset_texts.append(text[:500])  # Truncate to reasonable length
    
    # Run search
    results = search_dataset_for_interactions(
        model=model,
        sae=sae,
        dataset_texts=dataset_texts,
        layer=layer,
        head=head,
        num_samples=num_samples,
        top_k_per_pair=5,
        batch_size=1,
        filter_self_interactions=filter_self_interactions,
        verbose=True
    )
    
    # Save results if path provided
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"fra/results/dataset_search_L{layer}H{head}_{timestamp}.pkl"
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    save_search_results(results, save_path)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Search dataset for feature interactions")
    parser.add_argument("--layer", type=int, default=5, help="Layer to analyze")
    parser.add_argument("--head", type=int, default=0, help="Head to analyze")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--save-path", type=str, help="Path to save results")
    
    args = parser.parse_args()
    
    results = run_dataset_search(
        layer=args.layer,
        head=args.head,
        num_samples=args.num_samples,
        save_path=args.save_path
    )
    
    print(f"\n✅ Dataset search complete!")
    print(f"Results saved. Use dataset_search_viz.py to create visualization.")