#!/usr/bin/env python
"""
Visualization for data-independent Feature-Resolved Attention (FRA).

This module creates interactive visualizations showing which features naturally
attend to which other features based solely on the SAE decoder weights,
without any specific input data.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from transformer_lens import HookedTransformer
import html
from datetime import datetime
import json
import tarfile

from fra.induction_head import SAELensAttentionSAE, data_independent_attention
from fra.single_sample_viz import fetch_neuronpedia_explanation, get_neuronpedia_url


def create_data_independent_dashboard(
    model: HookedTransformer,
    sae: SAELensAttentionSAE,
    layer: int = 5,
    head: int = 0,
    top_k_features: int = 50,
    top_k_interactions: int = 100,
    output_dir: str = "fra/results",
    use_timestamp: bool = False
) -> str:
    """
    Create an interactive HTML dashboard for data-independent FRA visualization.
    
    Args:
        model: The transformer model
        sae: The SAE for feature extraction
        layer: Layer to analyze
        head: Head to analyze
        top_k_features: Number of top features to show
        top_k_interactions: Number of top interactions to show
        output_dir: Directory to save the dashboard
        use_timestamp: Whether to add timestamp to filename
        
    Returns:
        Path to the generated HTML file
    """
    print(f"\nGenerating data-independent FRA dashboard for L{layer}H{head}...")
    
    # Get SAE decoder weights
    W_dec = sae.sae.W_dec  # Shape: [d_sae, d_model]
    d_sae = W_dec.shape[0]
    
    # Compute data-independent attention pattern
    print("Computing data-independent attention pattern...")
    interaction_matrix = data_independent_attention(model, layer, head, W_dec)
    
    # Convert to tensor if needed
    if isinstance(interaction_matrix, np.ndarray):
        interaction_matrix = torch.from_numpy(interaction_matrix)
    
    # Find strongest interactions
    print("Finding strongest feature interactions...")
    flat_matrix = interaction_matrix.flatten()
    top_k_indices = torch.topk(flat_matrix.abs(), min(top_k_interactions, flat_matrix.numel())).indices
    
    top_interactions = []
    for idx in top_k_indices:
        q_feat = idx // d_sae
        k_feat = idx % d_sae
        strength = interaction_matrix[q_feat, k_feat].item()
        top_interactions.append((q_feat.item(), k_feat.item(), strength))
    
    # Find most active features (by total interaction strength)
    feature_importance = interaction_matrix.abs().sum(dim=1) + interaction_matrix.abs().sum(dim=0)
    top_features_idx = torch.topk(feature_importance, min(top_k_features, d_sae)).indices
    
    # Fetch Neuronpedia explanations for top features
    print("Fetching feature explanations from Neuronpedia...")
    unique_features = set()
    for q_feat, k_feat, _ in top_interactions[:top_k_interactions]:
        unique_features.add(q_feat)
        unique_features.add(k_feat)
    
    feature_explanations = {}
    for feat in unique_features:
        explanation = fetch_neuronpedia_explanation(layer, feat)
        feature_explanations[feat] = explanation
        if explanation != "No explanation available":
            print(f"  Feature {feat}: {explanation[:50]}...")
    
    # Generate HTML dashboard
    html_content = generate_data_independent_html(
        layer=layer,
        head=head,
        interaction_matrix=interaction_matrix,
        top_interactions=top_interactions,
        top_features_idx=top_features_idx.tolist(),
        feature_explanations=feature_explanations,
        d_sae=d_sae
    )
    
    # Save to file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_ind_fra_L{layer}H{head}_{timestamp}.html"
    else:
        filename = f"data_ind_fra_L{layer}H{head}.html"
    
    filepath = output_path / filename
    with open(filepath, 'w') as f:
        f.write(html_content)
    
    print(f"Dashboard saved to: {filepath}")
    return str(filepath)


def generate_data_independent_html(
    layer: int,
    head: int,
    interaction_matrix: torch.Tensor,
    top_interactions: List[Tuple[int, int, float]],
    top_features_idx: List[int],
    feature_explanations: Dict[int, str],
    d_sae: int
) -> str:
    """Generate HTML content for data-independent FRA dashboard."""
    
    # Use the same style as the data-dependent dashboard
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data-Independent FRA - Layer {layer}, Head {head}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .stat-card h3 {{
            color: #6c757d;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #495057;
        }}
        
        .section {{
            padding: 30px;
        }}
        
        .section h2 {{
            color: #495057;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        .note {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            color: #856404;
        }}
        
        .interactions-grid {{
            display: grid;
            gap: 20px;
        }}
        
        .interaction-card {{
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 20px;
            transition: all 0.3s;
        }}
        
        .interaction-card:hover {{
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        
        .interaction-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .interaction-strength {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        
        .interaction-strength.positive {{
            color: #28a745;
        }}
        
        .interaction-strength.negative {{
            color: #dc3545;
        }}
        
        .feature-pair {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        
        .feature-box {{
            flex: 1;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        
        .feature-box h4 {{
            color: #495057;
            margin-bottom: 5px;
        }}
        
        .feature-description {{
            margin: 10px 0;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 4px;
            font-size: 0.9em;
            color: #555;
            line-height: 1.4;
        }}
        
        .arrow {{
            font-size: 2em;
            color: #667eea;
        }}
        
        .self-interaction {{
            background: #fff9e6;
            border-color: #ffc107;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Data-Independent Feature-Resolved Attention</h1>
            <p>Layer {layer}, Head {head} - Intrinsic Feature Interaction Patterns</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Total Features</h3>
                <div class="value">{d_sae:,}</div>
            </div>
            <div class="stat-card">
                <h3>Layer</h3>
                <div class="value">{layer}</div>
            </div>
            <div class="stat-card">
                <h3>Head</h3>
                <div class="value">{head}</div>
            </div>
            <div class="stat-card">
                <h3>Max Interaction</h3>
                <div class="value">{interaction_matrix.max().item():.4f}</div>
            </div>
            <div class="stat-card">
                <h3>Min Interaction</h3>
                <div class="value">{interaction_matrix.min().item():.4f}</div>
            </div>
            <div class="stat-card">
                <h3>Sparsity</h3>
                <div class="value">{(interaction_matrix.abs() < 0.001).float().mean().item()*100:.1f}%</div>
            </div>
        </div>
        
        <div class="section">
            <div class="note">
                <strong>📌 Note:</strong> This visualization shows the <em>inherent</em> attention patterns between SAE features,
                computed directly from decoder weights without any specific input data. These patterns reveal which features
                naturally attend to each other in this attention head.
            </div>
            
            <h2>🔥 Top Feature Interactions</h2>
            <div class="interactions-grid">
"""
    
    # Add interaction cards (same format as data-dependent but without text panels)
    for i, (q_feat, k_feat, strength) in enumerate(top_interactions[:min(30, len(top_interactions))]):
        q_url = get_neuronpedia_url(layer, q_feat)
        k_url = get_neuronpedia_url(layer, k_feat)
        
        is_self = (q_feat == k_feat)
        card_class = "interaction-card self-interaction" if is_self else "interaction-card"
        strength_class = "positive" if strength > 0 else "negative"
        arrow_symbol = "→" if strength > 0 else "←"
        
        # Get explanations
        q_explanation = feature_explanations.get(q_feat, "No explanation available")
        k_explanation = feature_explanations.get(k_feat, "No explanation available")
        
        html_content += f"""
                <div class="{card_class}">
                    <div class="interaction-header">
                        <h3>Interaction #{i+1}</h3>
                        <span class="interaction-strength {strength_class}">{strength:.4f}</span>
                    </div>
                    <div class="feature-pair">
                        <div class="feature-box">
                            <h4>Query: <a href="{q_url}" target="_blank" style="color: #667eea; text-decoration: none;">Feature {q_feat} ↗</a></h4>
                            <p class="feature-description">{q_explanation}</p>
                        </div>
                        <div class="arrow">{arrow_symbol}</div>
                        <div class="feature-box">
                            <h4>Key: <a href="{k_url}" target="_blank" style="color: #667eea; text-decoration: none;">Feature {k_feat} ↗</a></h4>
                            <p class="feature-description">{k_explanation}</p>
                        </div>
                    </div>
                    {f'<div style="margin-top: 15px; padding: 10px; background: #fff3cd; border-radius: 5px; text-align: center;"><strong>⚡ Self-Interaction</strong> - Feature attends to itself</div>' if is_self else ''}
                </div>
"""
    
    html_content += """
            </div>
        </div>
    </div>
    
    <script>
        // Add interactive features
        document.addEventListener('DOMContentLoaded', function() {
            // Animate stats on load
            const statValues = document.querySelectorAll('.stat-card .value');
            statValues.forEach((stat, index) => {
                stat.style.opacity = '0';
                stat.style.transform = 'translateY(20px)';
                setTimeout(() => {
                    stat.style.transition = 'all 0.5s ease';
                    stat.style.opacity = '1';
                    stat.style.transform = 'translateY(0)';
                }, index * 100);
            });
            
            // Animate interaction cards on scroll
            const observerOptions = {
                threshold: 0.1,
                rootMargin: '0px 0px -50px 0px'
            };
            
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.style.opacity = '1';
                        entry.target.style.transform = 'translateY(0)';
                    }
                });
            }, observerOptions);
            
            const cards = document.querySelectorAll('.interaction-card');
            cards.forEach((card, index) => {
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                card.style.transition = `all 0.5s ease ${index * 0.05}s`;
                observer.observe(card);
            });
        });
    </script>
</body>
</html>
"""
    
    return html_content


def generate_data_independent_dashboard_from_config(
    model: Optional[HookedTransformer] = None,
    sae: Optional[SAELensAttentionSAE] = None,
    layer: int = 5,
    head: int = 0,
    top_k_features: int = 50,
    top_k_interactions: int = 100,
    config_path: Optional[str] = None,
    use_timestamp: bool = False
) -> str:
    """
    Generate data-independent FRA dashboard with flexible configuration.
    
    Can be called with pre-loaded model/SAE for efficiency, or will load them as needed.
    
    Examples:
        # Simplest usage - all defaults
        dashboard = generate_data_independent_dashboard_from_config()
        
        # Custom parameters
        dashboard = generate_data_independent_dashboard_from_config(
            layer=6, head=10, top_k_interactions=50
        )
        
        # With pre-loaded model and SAE (most efficient)
        dashboard = generate_data_independent_dashboard_from_config(
            model=model, sae=sae, layer=5, head=0
        )
    """
    # Load config if provided
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        layer = config.get('layer', layer)
        head = config.get('head', head)
        top_k_features = config.get('top_k_features', top_k_features)
        top_k_interactions = config.get('top_k_interactions', top_k_interactions)
    
    # Load model if not provided
    if model is None:
        print("Loading model...")
        model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
    
    # Load SAE if not provided
    if sae is None:
        print(f"Loading SAE for layer {layer}...")
        sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", f"blocks.{layer}.hook_z", device="cuda")
    
    # Generate dashboard
    return create_data_independent_dashboard(
        model=model,
        sae=sae,
        layer=layer,
        head=head,
        top_k_features=top_k_features,
        top_k_interactions=top_k_interactions,
        use_timestamp=use_timestamp
    )


def create_package(output_dir: str = "fra/results") -> str:
    """
    Create a tar.gz package of all HTML dashboards in the results directory.
    
    Returns:
        Path to the created package
    """
    results_dir = Path(output_dir)
    package_path = results_dir / "results_package.tar.gz"
    
    print(f"\nCreating package...")
    with tarfile.open(package_path, "w:gz") as tar:
        for html_file in results_dir.glob("*.html"):
            tar.add(html_file, arcname=html_file.name)
            print(f"  Added: {html_file.name}")
    
    print(f"\n✅ Package created: {package_path}")
    print(f"📥 To download: scp remote:{package_path.absolute()} ./")
    
    return str(package_path)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate data-independent FRA dashboard")
    parser.add_argument("--layer", type=int, default=5, help="Layer to analyze")
    parser.add_argument("--head", type=int, default=0, help="Head to analyze")
    parser.add_argument("--top-k-features", type=int, default=50, help="Number of top features to show")
    parser.add_argument("--top-k-interactions", type=int, default=100, help="Number of top interactions to show")
    parser.add_argument("--timestamp", action="store_true", help="Add timestamp to filename")
    parser.add_argument("--package", action="store_true", help="Create tar.gz package after generation")
    
    args = parser.parse_args()
    
    dashboard_path = generate_data_independent_dashboard_from_config(
        layer=args.layer,
        head=args.head,
        top_k_features=args.top_k_features,
        top_k_interactions=args.top_k_interactions,
        use_timestamp=args.timestamp
    )
    
    print(f"\n✅ Dashboard generated: {dashboard_path}")
    print("Open the HTML file in your browser to explore the data-independent feature interactions.")
    
    if args.package:
        create_package()