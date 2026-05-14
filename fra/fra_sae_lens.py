"""
FRA analysis using SAE Lens SAEs with verified high sparsity.
"""

import torch
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from transformer_lens import HookedTransformer

from fra.sae_lens_wrapper import SAELensAttentionSAE, get_attention_activations_for_sae_lens


@torch.no_grad()
def get_sentence_fra_sae_lens(
    model: HookedTransformer,
    sae: SAELensAttentionSAE,
    text: str,
    layer: int,
    head: int,
    max_length: int = 128,
    top_k: int = 20,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute FRA using SAE Lens SAEs with proper sparsity.
    
    Args:
        model: The transformer model
        sae: The SAE Lens wrapper
        text: Input text to analyze
        layer: Which layer to analyze
        head: Which attention head to analyze
        max_length: Maximum sequence length
        top_k: Number of top features to keep per position
        verbose: Whether to show progress
        
    Returns:
        Dictionary containing torch.sparse matrices and statistics
    """
    device = next(model.parameters()).device
    
    # Get activations using hook_z (what SAE Lens SAEs expect)
    if verbose:
        print("Computing hook_z activations...")
    activations = get_attention_activations_for_sae_lens(model, text, layer=layer, max_length=max_length)
    seq_len = min(activations.shape[0], max_length)
    
    # Encode to SAE features
    if verbose:
        print("Encoding to SAE features...")
    feature_activations = sae.encode(activations)  # [seq_len, d_sae]
    
    # Calculate and report sparsity
    if verbose:
        l0_per_token = (feature_activations != 0).sum(-1).float()
        avg_l0 = l0_per_token.mean().item()
        sparsity = 1 - (avg_l0 / sae.d_sae)
        print(f"  Average L0: {avg_l0:.1f} features per token")
        print(f"  Sparsity: {sparsity*100:.2f}%")
    
    # Keep only top-k features per position
    if verbose:
        print(f"Keeping top {top_k} features per position...")
    
    topk_features = []
    for pos in range(seq_len):
        feat = feature_activations[pos]
        # Get top-k by absolute value
        active_mask = feat != 0
        n_active = active_mask.sum().item()
        
        if n_active > 0:
            # Get top-k among active features
            k = min(top_k, n_active)
            topk_vals, topk_idx = torch.topk(feat.abs(), k)
            # Create sparse version with only top-k
            sparse_feat = torch.zeros_like(feat)
            sparse_feat[topk_idx] = feat[topk_idx]
        else:
            sparse_feat = torch.zeros_like(feat)
        
        topk_features.append(sparse_feat)
    
    topk_features = torch.stack(topk_features)
    
    if verbose:
        # Report top-k coverage
        total_l1_original = feature_activations.abs().sum(-1).mean().item()
        total_l1_topk = topk_features.abs().sum(-1).mean().item()
        coverage = (total_l1_topk / total_l1_original * 100) if total_l1_original > 0 else 0
        print(f"  Top-{top_k} L1 coverage: {coverage:.1f}%")
    
    # Get model attention weights
    W_Q = model.blocks[layer].attn.W_Q[head]
    W_K = model.blocks[layer].attn.W_K[head]
    
    # Process lower triangle (causal attention)
    total_pairs = seq_len * (seq_len + 1) // 2
    
    if verbose:
        pbar = tqdm(total=total_pairs, desc=f"Computing FRA (L{layer}H{head}, top-{top_k})")
    
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


def get_top_interactions(sparse_matrix, top_k=10):
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
    import time
    
    torch.set_grad_enabled(False)
    
    print("Testing FRA with SAE Lens SAEs...")
    
    # Load model
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
    
    # Load SAE Lens SAE
    RELEASE = "gpt2-small-hook-z-kk"
    SAE_ID = "blocks.5.hook_z"
    sae = SAELensAttentionSAE(RELEASE, SAE_ID, device="cuda")
    
    print(f"SAE loaded: d_in={sae.d_in}, d_sae={sae.d_sae}")
    
    # Test text
    text = "The cat sat on the mat. The cat was happy. The dog ran in the park."
    
    print(f"\nAnalyzing: '{text}'")
    
    # Test with different k values
    for k in [10, 20]:
        print(f"\n--- Top-{k} features per position ---")
        t0 = time.time()
        result = get_sentence_fra_sae_lens(
            model, sae, text, 
            layer=5, head=0, 
            top_k=k, verbose=True
        )
        elapsed = time.time() - t0
        
        print(f"Time: {elapsed:.2f}s")
        print(f"Non-zero interactions: {result['nnz']:,}")
        print(f"Density: {result['density']:.8%}")
        
        top = get_top_interactions(result['data_dep_int_matrix'], 10)
        print(f"\nTop 10 feature interactions:")
        for i, (q, k, v) in enumerate(top):
            sign = "→" if v > 0 else "←"
            print(f"  {i+1}. F{q} {sign} F{k}: {v:.4f}")
    
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()