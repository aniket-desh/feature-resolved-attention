"""
Feature-Resolved Attention (FRA) for Induction Heads
Main script for demonstrating FRA gives better mechanistic insights than token-level attention.
"""

import torch
from pathlib import Path
import argparse
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE
from fra.fra_func import attention_pattern_QK


def data_independent_attention(model: HookedTransformer, layer: int, head: int, sae_dec: torch.Tensor):
    """
    Compute data-independent feature-resolved attention pattern.
    
    This shows which features naturally attend to which other features
    based solely on the SAE decoder weights, without any specific input data.
    
    Args:
        model: The transformer model
        layer: Layer index
        head: Head index
        sae_dec: SAE decoder matrix (W_dec) of shape [d_sae, d_model]
    
    Returns:
        interaction_matrix: Data-independent attention pattern [d_sae, d_sae]
    """
    # Use SAE decoder weights directly as "activations" for all features
    # This shows the inherent feature-to-feature attention preferences
    query_activations_for_features = sae_dec
    key_activations_for_features = sae_dec
    
    # Compute attention pattern using the decoder weights
    interaction_matrix_unscaled = attention_pattern_QK(
        model, layer, head,
        query_activations_for_features, False,  # No bias for queries
        key_activations_for_features, False     # No bias for keys
    )
    
    return interaction_matrix_unscaled


class SAELensAttentionSAE:
    """Wrapper for SAE Lens attention SAEs."""
    
    def __init__(self, release: str, sae_id: str, device: str = "cuda"):
        """Initialize SAE from SAE Lens.
        
        Args:
            release: SAE Lens release name (e.g., "gpt2-small-hook-z-kk")
            sae_id: SAE ID (e.g., "blocks.5.hook_z")
            device: Device to load SAE on
        """
        self.release = release
        self.sae_id = sae_id
        self.device = device
        
        # Load the SAE from SAE Lens
        self.sae = SAE.from_pretrained(release, sae_id, device=device)
        
        # Turn off hook_z reshaping to have manual control
        if hasattr(self.sae, 'turn_off_forward_pass_hook_z_reshaping'):
            self.sae.turn_off_forward_pass_hook_z_reshaping()
        
        # Get dimensions
        self.d_in = self.sae.cfg.d_in  # Should be 768 for GPT-2 small attention
        self.d_sae = self.sae.cfg.d_sae  # Should be 49152
        
        # Extract weights for FRA computation
        self.W_dec = self.sae.W_dec  # [d_sae, d_in]
        
        # Extract layer number from sae_id
        parts = sae_id.split('.')
        self.layer = int(parts[1]) if len(parts) > 1 else 0
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input activations to SAE features.
        
        Args:
            x: Input tensor of shape [..., d_in]
            
        Returns:
            SAE features of shape [..., d_sae]
        """
        # Handle different input shapes
        if len(x.shape) == 3 and x.shape[-2:] == (12, 64):  # GPT-2 small attention shape
            x = x.flatten(-2, -1)  # [seq_len, 768]
        
        # Ensure input is 2D for SAE
        if len(x.shape) > 2:
            batch_shape = x.shape[:-1]
            x = x.reshape(-1, self.d_in)
            features = self.sae.encode(x)
            features = features.reshape(*batch_shape, self.d_sae)
        else:
            features = self.sae.encode(x)
        
        return features


def get_attention_activations(
    model: HookedTransformer,
    input_text: str,
    layer: int,
    max_length: int = 128
) -> torch.Tensor:
    """
    Get attention activations for SAE Lens SAEs (hook_z format).
    
    Args:
        model: The HookedTransformer model
        input_text: Input text to analyze
        layer: Which layer to get activations from
        max_length: Maximum sequence length
        
    Returns:
        Tensor of shape (sequence_length, n_heads * d_head) = (seq_len, 768)
    """
    # Tokenize
    tokens = model.tokenizer.encode(input_text)
    if max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]
    
    device = next(model.parameters()).device
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)
    
    # Get hook_z activations
    hook_name = f"blocks.{layer}.attn.hook_z"
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    
    # Shape: [batch=1, seq_len, n_heads=12, d_head=64]
    z = cache[hook_name].squeeze(0)  # Remove batch dimension
    
    # Flatten heads dimension: [seq_len, 12*64=768]
    z_flat = z.flatten(-2, -1)
    
    return z_flat


@torch.no_grad()
def compute_fra(
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
    Compute Feature-Resolved Attention for a text sample.
    
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
        Dictionary containing:
            - fra_matrix: Sparse FRA matrix [d_sae, d_sae]
            - fra_matrix_abs: Absolute value version
            - seq_len: Sequence length
            - total_pairs: Number of position pairs processed
            - nnz: Number of non-zero entries
            - density: Sparsity of the matrix
            - avg_l0: Average L0 norm (features per token)
            - sparsity: Percentage sparsity
    """
    device = next(model.parameters()).device
    
    # Get activations
    if verbose:
        print("Getting attention activations...")
    activations = get_attention_activations(model, text, layer=layer, max_length=max_length)
    seq_len = min(activations.shape[0], max_length)
    
    # Encode to SAE features
    if verbose:
        print("Encoding to SAE features...")
    feature_activations = sae.encode(activations)  # [seq_len, d_sae]
    
    # Calculate sparsity statistics
    l0_per_token = (feature_activations != 0).sum(-1).float()
    avg_l0 = l0_per_token.mean().item()
    sparsity = 1 - (avg_l0 / sae.d_sae)
    
    if verbose:
        print(f"  Average L0: {avg_l0:.1f} features per token")
        print(f"  Sparsity: {sparsity*100:.2f}%")
    
    # Keep only top-k features per position
    if verbose:
        print(f"Selecting top-{top_k} features per position...")
    
    topk_features = []
    for pos in range(seq_len):
        feat = feature_activations[pos]
        active_mask = feat != 0
        n_active = active_mask.sum().item()
        
        if n_active > 0:
            k = min(top_k, n_active)
            topk_vals, topk_idx = torch.topk(feat.abs(), k)
            sparse_feat = torch.zeros_like(feat)
            sparse_feat[topk_idx] = feat[topk_idx]
        else:
            sparse_feat = torch.zeros_like(feat)
        
        topk_features.append(sparse_feat)
    
    topk_features = torch.stack(topk_features)
    
    # Get attention weights for the specified head
    W_Q = model.blocks[layer].attn.W_Q[head]
    W_K = model.blocks[layer].attn.W_K[head]
    
    # Process all position pairs (lower triangle for causal attention)
    total_pairs = seq_len * (seq_len + 1) // 2
    
    if verbose:
        pbar = tqdm(total=total_pairs, desc=f"Computing FRA (L{layer}H{head})")
    
    # Accumulate interactions in COO format
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
            
            # Compute attention scores
            q_proj = torch.matmul(q_vecs, W_Q)
            k_proj = torch.matmul(k_vecs, W_K)
            int_matrix = torch.matmul(q_proj, k_proj.T)
            
            # Scale by feature activations
            int_matrix = int_matrix * q_feat[q_active].unsqueeze(1) * k_feat[k_active].unsqueeze(0)
            
            # Find non-zero interactions
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
    
    # Create sparse matrices
    d_sae = sae.d_sae
    
    if len(row_indices) > 0:
        indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
        vals = torch.tensor(values, dtype=torch.float32, device=device)
        
        # Average by number of position pairs
        vals = vals / pair_count if pair_count > 0 else vals
        
        # Create sparse matrix
        fra_matrix = torch.sparse_coo_tensor(
            indices, vals,
            size=(d_sae, d_sae),
            device=device,
            dtype=torch.float32
        ).coalesce()
        
        # Absolute value version
        abs_vals = vals.abs()
        fra_matrix_abs = torch.sparse_coo_tensor(
            indices, abs_vals,
            size=(d_sae, d_sae),
            device=device,
            dtype=torch.float32
        ).coalesce()
        
        nnz = fra_matrix._nnz()
        density = nnz / (d_sae * d_sae)
    else:
        # Empty matrices if no interactions found
        empty_indices = torch.zeros((2, 0), dtype=torch.long, device=device)
        empty_vals = torch.zeros(0, dtype=torch.float32, device=device)
        
        fra_matrix = torch.sparse_coo_tensor(
            empty_indices, empty_vals,
            size=(d_sae, d_sae),
            device=device
        )
        fra_matrix_abs = fra_matrix.clone()
        nnz = 0
        density = 0
    
    return {
        'fra_matrix': fra_matrix,
        'fra_matrix_abs': fra_matrix_abs,
        'seq_len': seq_len,
        'total_pairs': pair_count,
        'nnz': nnz,
        'density': density,
        'avg_l0': avg_l0,
        'sparsity': sparsity
    }


def get_top_feature_interactions(fra_matrix: torch.sparse.FloatTensor, top_k: int = 10) -> List[Tuple[int, int, float]]:
    """
    Get top-k feature interactions from FRA matrix.
    
    Args:
        fra_matrix: Sparse FRA matrix
        top_k: Number of top interactions to return
        
    Returns:
        List of (query_feature, key_feature, interaction_strength) tuples
    """
    if fra_matrix._nnz() == 0:
        return []
    
    indices = fra_matrix._indices()
    values = fra_matrix._values()
    
    k = min(top_k, len(values))
    _, top_idx = torch.topk(values.abs(), k)
    
    results = []
    for idx in top_idx:
        q_feat = indices[0, idx].item()
        k_feat = indices[1, idx].item()
        value = values[idx].item()
        results.append((q_feat, k_feat, value))
    
    return results


def analyze_self_interactions(fra_matrix: torch.sparse.FloatTensor, threshold: float = 0.001) -> List[Tuple[int, float]]:
    """
    Find self-interactions (potential induction behavior) in FRA matrix.
    
    Args:
        fra_matrix: Sparse FRA matrix
        threshold: Minimum interaction strength to consider
        
    Returns:
        List of (feature_id, self_interaction_strength) tuples
    """
    indices = fra_matrix._indices()
    values = fra_matrix._values()
    
    # Find diagonal elements (self-interactions)
    self_mask = indices[0] == indices[1]
    
    if not self_mask.any():
        return []
    
    self_features = indices[0, self_mask]
    self_values = values[self_mask]
    
    # Filter by threshold
    strong_mask = self_values.abs() > threshold
    
    results = []
    for feat, val in zip(self_features[strong_mask].tolist(), self_values[strong_mask].tolist()):
        results.append((feat, val))
    
    # Sort by interaction strength
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return results


def loader(config_path: str = None):
    """Main entry point for Feature-Resolved Attention experiments.
    
    Args:
        config_path: Path to configuration file
    """
    # Determine config path
    if config_path is None:
        # Try to find config.yaml in project root
        config_path = Path(__file__).parent.parent / "config.yaml"
        if not config_path.exists():
            config_path = Path.cwd() / "config.yaml"
    
    # Load configuration
    config = load_config(str(config_path))
    
    # Set device
    device = torch.device(config["model"]["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config["experiment"]["seed"])
    
    # Create output directory
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load model
    model = load_model(
        model_name=config["model"]["name"],
        device=str(device)
    )
    
    # Step 2: Load SAE for specified layer
    sae_data = load_sae(
        repo=config["sae"]["repo"],
        layer=config["sae"]["layer"],
        device=str(device)
    )
    
    # Step 3: Load dataset (with streaming for memory efficiency)
    dataset = load_dataset_hf(
        dataset_name=config["dataset"]["name"],
        split=config["dataset"]["split"],
        streaming=config["dataset"].get("streaming", True),  # Default to streaming
        seed=config["experiment"]["seed"]
    )
    
    logger.info("Successfully loaded all components!")
    logger.info(f"Model: {config['model']['name']} with {model.cfg.n_layers} layers")
    logger.info(f"SAE: Layer {config['sae']['layer']}, dict_mult={sae_data['config'].get('dict_mult', 'unknown')}")
    logger.info(f"Dataset: {config['dataset']['name']} loaded")
    
    # TODO: Implement Feature-Resolved Attention analysis
    # This will be implemented in the next steps
    
    return model, sae_data, dataset


def main():
    """Main entry point for Feature-Resolved Attention analysis."""
    import time
    import tarfile
    import os
    from fra.fra_func import get_sentence_fra_batch
    from fra.utils import load_config, load_dataset_hf
    from fra.single_sample_viz import create_fra_dashboard
    
    torch.set_grad_enabled(False)
    
    print("=" * 70)
    print("Feature-Resolved Attention (FRA) Analysis")
    print("=" * 70)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(str(config_path))
    print(f"\nConfiguration loaded from: {config_path}")
    
    # Load model
    print(f"\nLoading {config['model']['name']}...")
    device = config["model"]["device"] if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(config["model"]["name"], device=device)
    
    # Load SAE
    layer = config["sae"]["layer"]
    print(f"Loading SAE Lens attention SAE for layer {layer}...")
    RELEASE = "gpt2-small-hook-z-kk"
    SAE_ID = f"blocks.{layer}.hook_z"
    sae = SAELensAttentionSAE(RELEASE, SAE_ID, device=device)
    print(f"  SAE dimensions: d_in={sae.d_in}, d_sae={sae.d_sae}")

    # Load dataset
    print(f"\nLoading dataset: {config['dataset']['name']}...")
    dataset = load_dataset_hf(
        dataset_name=config["dataset"]["name"],
        split=config["dataset"]["split"],
        streaming=config["dataset"].get("streaming", True),
        seed=config["experiment"]["seed"]
    )
    
    # Get sample text from dataset
    if hasattr(dataset, '__iter__'):  # Streaming dataset
        sample = next(iter(dataset))
        text = sample['text'][:config["dataset"].get("max_length", 128)]
    else:  # Regular dataset
        sample = dataset[0]
        text = sample['text'][:config["dataset"].get("max_length", 128)]
    
    print(f"\nAnalyzing text: '{text}'")
    print("-" * 70)
    
    # Get FRA parameters from config
    top_k = config.get("fra", {}).get("top_k_features", 10)
    
    # Extract full 4D FRA tensor for a single sentence
    fra_4d = get_sentence_fra_batch(model, sae, text, layer=layer, head=0, top_k=top_k, verbose=True)
    
    # Analyze multiple heads with averaged FRA
    for head in [0, 5, 10]:  # Test a few different heads
        print(f"\n### Layer {layer}, Head {head} ###")
        
        t0 = time.time()
        result = compute_fra(
            model, sae, text, 
            layer=layer, head=head,
            top_k=top_k, verbose=False
        )
        elapsed = time.time() - t0
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Sparsity: {result['sparsity']*100:.2f}%")
        print(f"  Non-zero interactions: {result['nnz']:,}")
        
        # Get top interactions
        top_interactions = get_top_feature_interactions(result['fra_matrix'], top_k=5)
        if top_interactions:
            print(f"\n  Top 5 feature interactions:")
            for i, (q, k, v) in enumerate(top_interactions):
                arrow = "→" if v > 0 else "←"
                print(f"    {i+1}. F{q} {arrow} F{k}: {v:.4f}")
        
        # Check for self-interactions (potential induction)
        self_interactions = analyze_self_interactions(result['fra_matrix'])
        if self_interactions:
            print(f"\n  Self-interactions (potential induction):")
            for feat, strength in self_interactions[:3]:
                print(f"    F{feat} → F{feat}: {strength:.4f}")
    
    print("\n" + "=" * 70)
    
    # Generate dashboard for the analyzed text (single line version)
    print("\nGenerating interactive dashboard...")
    from fra.single_sample_viz import generate_dashboard_from_config
    
    # Single line call with all parameters
    dashboard_path = generate_dashboard_from_config(model=model, sae=sae, text=text, layer=layer, head=0, top_k_features=top_k, top_k_interactions=30)
    
    # Alternative single line calls:
    # dashboard_path = generate_dashboard_from_config(model, sae, text)  # Uses defaults for other params
    # dashboard_path = generate_dashboard_from_config(config_path="config.yaml")  # Loads everything from config
    # dashboard_path = generate_dashboard_from_config()  # Uses all defaults
    
    print(f"Dashboard saved to: {dashboard_path}")
    
    # Package results for easy transfer
    results_dir = Path(__file__).parent / "results"
    package_path = results_dir / "results_package.tar.gz"
    
    print(f"\nPackaging results...")
    with tarfile.open(package_path, "w:gz") as tar:
        # Add all HTML files in results directory
        for html_file in results_dir.glob("*.html"):
            tar.add(html_file, arcname=html_file.name)
            print(f"  Added: {html_file.name}")
    
    print(f"\n✅ Results package created: {package_path}")
    print(f"📥 Download with: scp remote:{package_path} ./")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    
    return model, sae, dataset


if __name__ == "__main__":
    import os
    
    try:
        model, sae, dataset = main()
    finally:
        # Clean up CUDA resources to prevent core dump
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Force exit to avoid cleanup issues
        os._exit(0)
    
    
    
    

    