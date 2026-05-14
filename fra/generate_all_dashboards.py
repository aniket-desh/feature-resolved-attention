#!/usr/bin/env python
"""
Unified function to generate all FRA dashboards and package them.

This creates:
1. Data-independent FRA dashboard (structural patterns)
2. Dataset search dashboard (patterns across samples)
3. Single sample dashboard (optional, for specific text)
4. Package them all into results_package.tar.gz
"""

import torch
from pathlib import Path
import tarfile
from typing import Optional, List
from transformer_lens import HookedTransformer
from datetime import datetime

from fra.induction_head import SAELensAttentionSAE
from fra.data_ind_viz import create_data_independent_dashboard
from fra.dataset_search import run_dataset_search
from fra.dataset_search_viz import create_dashboard_from_search
from fra.single_sample_viz import create_fra_dashboard


def generate_all_dashboards(
    layer: int = 5,
    head: int = 0,
    num_samples: int = 5,
    sample_text: Optional[str] = None,
    model: Optional[HookedTransformer] = None,
    sae: Optional[SAELensAttentionSAE] = None,
    filter_self_interactions: bool = True,
    output_dir: str = "fra/results",
    create_package: bool = True,
    verbose: bool = True
) -> dict:
    """
    Generate all FRA dashboards and optionally package them.
    
    Args:
        layer: Layer to analyze
        head: Head to analyze
        num_samples: Number of samples for dataset search
        sample_text: Optional text for single-sample dashboard
        model: Pre-loaded model (will load if None)
        sae: Pre-loaded SAE (will load if None)
        filter_self_interactions: Whether to filter self-interactions in dataset search
        output_dir: Directory for outputs
        create_package: Whether to create tar.gz package
        verbose: Whether to show progress
        
    Returns:
        Dictionary with paths to generated dashboards and package
    """
    results = {}
    
    # Load model and SAE if not provided
    if model is None:
        if verbose:
            print("Loading model...")
        model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
    
    if sae is None:
        if verbose:
            print(f"Loading SAE for layer {layer}...")
        sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", f"blocks.{layer}.hook_z", device="cuda")
    
    print("\n" + "="*60)
    print("🚀 Generating FRA Dashboards")
    print("="*60)
    
    # 1. Data-independent dashboard
    print("\n📊 1. Data-Independent FRA Dashboard")
    print("-" * 40)
    if verbose:
        print("Analyzing structural feature interaction patterns...")
    
    data_ind_path = create_data_independent_dashboard(
        model=model,
        sae=sae,
        layer=layer,
        head=head,
        top_k_features=30,
        top_k_interactions=50,
        output_dir=output_dir,
        use_timestamp=False
    )
    results['data_independent'] = data_ind_path
    print(f"✅ Created: {Path(data_ind_path).name}")
    
    # 2. Dataset search dashboard
    print("\n🔍 2. Dataset Search Dashboard")
    print("-" * 40)
    if verbose:
        print(f"Searching {num_samples} samples for feature interactions...")
        print(f"Self-interaction filtering: {'ON' if filter_self_interactions else 'OFF'}")
    
    search_results = run_dataset_search(
        model=model,
        sae=sae,
        layer=layer,
        head=head,
        num_samples=num_samples,
        filter_self_interactions=filter_self_interactions,
        save_path=None  # Don't save pickle for now
    )
    
    dataset_search_path = create_dashboard_from_search(
        results=search_results,
        top_k_interactions=50,
        use_timestamp=False
    )
    results['dataset_search'] = dataset_search_path
    print(f"✅ Created: {Path(dataset_search_path).name}")
    
    # 3. Optional single-sample dashboard
    if sample_text:
        print("\n📝 3. Single Sample Dashboard")
        print("-" * 40)
        if verbose:
            print(f"Analyzing: '{sample_text[:50]}...'")
        
        single_sample_path = create_fra_dashboard(
            model=model,
            sae=sae,
            text=sample_text,
            layer=layer,
            head=head,
            top_k_features=10,
            top_k_interactions=20,
            use_timestamp=False
        )
        results['single_sample'] = single_sample_path
        print(f"✅ Created: {Path(single_sample_path).name}")
    
    # 4. Create package
    if create_package:
        print("\n📦 4. Creating Package")
        print("-" * 40)
        
        results_dir = Path(output_dir)
        package_path = results_dir / "results_package.tar.gz"
        
        with tarfile.open(package_path, "w:gz") as tar:
            for html_file in results_dir.glob("*.html"):
                tar.add(html_file, arcname=html_file.name)
                if verbose:
                    print(f"  Added: {html_file.name}")
        
        results['package'] = str(package_path)
        print(f"\n✅ Package created: {package_path}")
        print(f"📥 Download: scp remote:{package_path.absolute()} ./")
    
    # Summary
    print("\n" + "="*60)
    print("✨ Dashboard Generation Complete!")
    print("="*60)
    print("\nGenerated dashboards:")
    print(f"  1. Data-Independent: Shows inherent feature interaction patterns")
    print(f"  2. Dataset Search: Shows patterns across {num_samples} samples")
    if sample_text:
        print(f"  3. Single Sample: Shows detailed analysis of specific text")
    
    if filter_self_interactions:
        print("\n📌 Note: Self-interactions (features attending to themselves) were filtered out")
    else:
        print("\n📌 Note: Self-interactions included (may indicate induction heads)")
    
    # Clean up CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def quick_generate(num_samples: int = 5, filter_self: bool = True):
    """
    Quick function to generate all dashboards with defaults.
    
    Args:
        num_samples: Number of samples for dataset search
        filter_self: Whether to filter self-interactions
    """
    return generate_all_dashboards(
        layer=5,
        head=0,
        num_samples=num_samples,
        sample_text="The cat sat on the mat. The dog sat on the rug.",
        filter_self_interactions=filter_self,
        verbose=True
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate all FRA dashboards")
    parser.add_argument("--layer", type=int, default=5, help="Layer to analyze")
    parser.add_argument("--head", type=int, default=0, help="Head to analyze")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples for dataset search")
    parser.add_argument("--sample-text", type=str, help="Text for single-sample dashboard")
    parser.add_argument("--no-filter-self", action="store_true", help="Include self-interactions")
    parser.add_argument("--no-package", action="store_true", help="Skip creating tar.gz package")
    
    args = parser.parse_args()
    
    results = generate_all_dashboards(
        layer=args.layer,
        head=args.head,
        num_samples=args.num_samples,
        sample_text=args.sample_text,
        filter_self_interactions=not args.no_filter_self,
        create_package=not args.no_package,
        verbose=True
    )
    
    print(f"\n✅ All dashboards generated successfully!")