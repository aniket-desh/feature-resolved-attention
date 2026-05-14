"""
Interactive HTML dashboard for visualizing Feature-Resolved Attention on a single text sample.
Includes Neuronpedia feature descriptions for interpretability.
"""

import torch
import json
import requests
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
import html


def get_neuronpedia_url(layer: int, feature_idx: int, embed: bool = False) -> str:
    """
    Get Neuronpedia URL for a feature.
    
    Args:
        layer: Layer number
        feature_idx: Feature index
        embed: Whether to get embedded version
        
    Returns:
        Neuronpedia URL
    """
    base_url = f"https://www.neuronpedia.org/gpt2-small/{layer}-att-kk/{feature_idx}"
    if embed:
        return f"{base_url}?embed=true&embedexplanation=true&embedplots=true&embedtest=false"
    return base_url


def fetch_neuronpedia_explanation(layer: int, feature_idx: int, timeout: int = 2) -> str:
    """
    Fetch feature explanation from Neuronpedia API.
    
    Args:
        layer: Layer number
        feature_idx: Feature index
        timeout: Request timeout in seconds
        
    Returns:
        Feature explanation string or fallback message
    """
    import requests
    
    # Try to fetch from Neuronpedia API
    api_url = f"https://www.neuronpedia.org/api/feature/gpt2-small/{layer}-att-kk/{feature_idx}"
    
    try:
        response = requests.get(api_url, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            
            # Try to get explanation from various possible fields
            explanation = data.get('explanation', '')
            if not explanation:
                explanation = data.get('description', '')
            if not explanation:
                explanation = data.get('label', '')
            if not explanation and 'explanations' in data and data['explanations']:
                # Sometimes it's in an array
                explanation = data['explanations'][0].get('description', '')
            
            if explanation:
                return explanation
                
        return f"Feature activates on specific patterns (see Neuronpedia for details)"
    
    except Exception as e:
        # Fallback for network issues or API unavailable
        return f"Feature explanation unavailable (network error)"


def create_fra_dashboard(
    model: Any,
    sae: Any,
    text: str,
    layer: int,
    head: int,
    top_k_features: int = 20,
    top_k_interactions: int = 50,
    output_path: str = None,
    use_timestamp: bool = False
) -> str:
    """
    Create an interactive HTML dashboard for FRA visualization.
    
    Args:
        model: The transformer model
        sae: The SAE wrapper
        text: Input text to analyze
        layer: Layer number
        head: Head number
        top_k_features: Number of top features per position
        top_k_interactions: Number of top interactions to show
        output_path: Path to save HTML file
        
    Returns:
        Path to the generated HTML file
    """
    from fra.induction_head import compute_fra, get_top_feature_interactions, get_attention_activations
    from datetime import datetime
    
    # Set default output path in results folder
    if output_path is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"fra_dashboard_L{layer}H{head}_{timestamp}.html"
        else:
            output_path = results_dir / f"fra_dashboard_L{layer}H{head}.html"
    else:
        output_path = Path(output_path)
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute FRA
    print(f"Computing FRA for layer {layer}, head {head}...")
    fra_result = compute_fra(
        model, sae, text,
        layer=layer, head=head,
        top_k=top_k_features,
        verbose=True
    )
    
    # Get feature activations for all tokens
    from fra.induction_head import get_attention_activations
    activations = get_attention_activations(model, text, layer=layer, max_length=128)
    feature_activations = sae.encode(activations)  # [seq_len, d_sae]
    
    # Get top interactions
    top_interactions = get_top_feature_interactions(
        fra_result['fra_matrix'],
        top_k=top_k_interactions
    )
    
    # Build feature URLs and fetch explanations for Neuronpedia
    print("Fetching feature explanations from Neuronpedia...")
    
    # Collect unique features and fetch their explanations
    unique_features = set()
    for q_feat, k_feat, _ in top_interactions[:top_k_interactions]:
        unique_features.add(q_feat)
        unique_features.add(k_feat)
    
    feature_explanations = {}
    for feat in unique_features:
        explanation = fetch_neuronpedia_explanation(layer, feat)
        feature_explanations[feat] = explanation
        print(f"  Feature {feat}: {explanation[:50]}..." if len(explanation) > 50 else f"  Feature {feat}: {explanation}")
    
    # Tokenize text for display
    tokens = model.tokenizer.encode(text)
    token_strings = [model.tokenizer.decode([t]) for t in tokens]
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feature-Resolved Attention Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
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
        
        .text-display {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            font-family: 'Courier New', monospace;
            line-height: 1.8;
        }}
        
        .token {{
            display: inline-block;
            padding: 2px 4px;
            margin: 2px;
            background: #e9ecef;
            border-radius: 3px;
            transition: all 0.3s;
        }}
        
        .token:hover {{
            background: #667eea;
            color: white;
            transform: scale(1.1);
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
        
        .positive {{
            color: #28a745;
        }}
        
        .negative {{
            color: #dc3545;
        }}
        
        .feature-pair {{
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 20px;
            align-items: center;
        }}
        
        .feature-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .feature-box h4 {{
            color: #495057;
            margin-bottom: 8px;
        }}
        
        .feature-description {{
            color: #6c757d;
            font-size: 0.9em;
            line-height: 1.4;
        }}
        
        .arrow {{
            font-size: 2em;
            color: #667eea;
        }}
        
        .neuronpedia-link {{
            display: inline-block;
            margin-top: 10px;
            color: #667eea;
            text-decoration: none;
            font-size: 0.85em;
        }}
        
        .neuronpedia-link:hover {{
            text-decoration: underline;
        }}
        
        .self-interaction {{
            background: #fff3cd;
            border-color: #ffc107;
        }}
        
        .self-interaction .feature-box {{
            border-left-color: #ffc107;
        }}
        
        .load-btn {{
            margin-top: 10px;
            padding: 6px 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.85em;
        }}
        
        .load-btn:hover {{
            background: #764ba2;
        }}
        
        .feature-desc-content {{
            margin-top: 10px;
            padding: 10px;
            background: white;
            border-radius: 5px;
            font-size: 0.9em;
            display: none;
        }}
        
        .feature-desc-content.loaded {{
            display: block;
        }}
        
        .feature-box h4 a:hover {{
            text-decoration: underline !important;
        }}
        
        .text-panels {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        
        .text-panel {{
            background: white;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #dee2e6;
        }}
        
        .text-panel h5 {{
            margin: 0 0 10px 0;
            color: #495057;
            font-size: 0.9em;
            font-weight: 600;
        }}
        
        .text-panel .tokens {{
            font-family: 'Courier New', monospace;
            line-height: 1.8;
            font-size: 0.9em;
        }}
        
        .token-highlight {{
            padding: 2px 4px;
            border-radius: 3px;
            margin: 1px;
            display: inline-block;
            transition: all 0.2s;
        }}
        
        .highlight-query {{
            background: rgba(102, 126, 234, 0.3);
            border: 1px solid rgba(102, 126, 234, 0.5);
        }}
        
        .highlight-key {{
            background: rgba(118, 75, 162, 0.3);
            border: 1px solid rgba(118, 75, 162, 0.5);
        }}
        
        .highlight-both {{
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.4) 0%, rgba(118, 75, 162, 0.4) 100%);
            border: 1px solid #667eea;
            font-weight: bold;
        }}
        
        .token-strength {{
            font-size: 0.7em;
            vertical-align: super;
            color: #6c757d;
            margin-left: 2px;
        }}
        
        @media (max-width: 768px) {{
            .feature-pair {{
                grid-template-columns: 1fr;
            }}
            
            .arrow {{
                text-align: center;
            }}
            
            .text-panels {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Feature-Resolved Attention Dashboard</h1>
            <p>Layer {layer}, Head {head} | GPT-2 Small</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Sequence Length</h3>
                <div class="value">{fra_result['seq_len']}</div>
            </div>
            <div class="stat-card">
                <h3>Sparsity</h3>
                <div class="value">{fra_result['sparsity']*100:.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>Avg L0</h3>
                <div class="value">{fra_result['avg_l0']:.1f}</div>
            </div>
            <div class="stat-card">
                <h3>Non-zero Interactions</h3>
                <div class="value">{fra_result['nnz']:,}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📝 Input Text</h2>
            <div class="text-display">
                {''.join([f'<span class="token" title="Token {i}">{html.escape(t)}</span>' for i, t in enumerate(token_strings[:fra_result['seq_len']])])}
            </div>
        </div>
        
        <div class="section">
            <h2>🔥 Top Feature Interactions</h2>
            <div class="interactions-grid">
"""
    
    # Add interaction cards
    for i, (q_feat, k_feat, strength) in enumerate(top_interactions[:top_k_interactions]):
        q_url = get_neuronpedia_url(layer, q_feat)
        k_url = get_neuronpedia_url(layer, k_feat)
        
        is_self = (q_feat == k_feat)
        card_class = "interaction-card self-interaction" if is_self else "interaction-card"
        strength_class = "positive" if strength > 0 else "negative"
        arrow_symbol = "→" if strength > 0 else "←"
        
        # Get activation strengths for each token for this feature pair
        query_activations = feature_activations[:, q_feat].cpu().numpy()
        key_activations = feature_activations[:, k_feat].cpu().numpy()
        
        # Create highlighted text for each panel
        query_tokens_html = ""
        key_tokens_html = ""
        interaction_tokens_html = ""
        
        for idx, token_str in enumerate(token_strings[:fra_result['seq_len']]):
            escaped_token = html.escape(token_str)
            
            # Query panel - highlight if query feature is active
            if query_activations[idx] > 0.01:
                opacity = min(1.0, query_activations[idx] / query_activations.max()) if query_activations.max() > 0 else 0
                query_tokens_html += f'<span class="token-highlight highlight-query" style="opacity: {0.3 + 0.7*opacity}">{escaped_token}</span>'
            else:
                query_tokens_html += f'<span class="token-highlight">{escaped_token}</span>'
            
            # Key panel - highlight if key feature is active
            if key_activations[idx] > 0.01:
                opacity = min(1.0, key_activations[idx] / key_activations.max()) if key_activations.max() > 0 else 0
                key_tokens_html += f'<span class="token-highlight highlight-key" style="opacity: {0.3 + 0.7*opacity}">{escaped_token}</span>'
            else:
                key_tokens_html += f'<span class="token-highlight">{escaped_token}</span>'
            
            # Interaction panel - highlight if both are active
            if query_activations[idx] > 0.01 and key_activations[idx] > 0.01:
                combined_strength = (query_activations[idx] * key_activations[idx]) ** 0.5
                opacity = min(1.0, combined_strength / max(query_activations.max(), key_activations.max())) if max(query_activations.max(), key_activations.max()) > 0 else 0
                interaction_tokens_html += f'<span class="token-highlight highlight-both" style="opacity: {0.3 + 0.7*opacity}">{escaped_token}</span>'
            elif query_activations[idx] > 0.01:
                interaction_tokens_html += f'<span class="token-highlight highlight-query" style="opacity: 0.3">{escaped_token}</span>'
            elif key_activations[idx] > 0.01:
                interaction_tokens_html += f'<span class="token-highlight highlight-key" style="opacity: 0.3">{escaped_token}</span>'
            else:
                interaction_tokens_html += f'<span class="token-highlight">{escaped_token}</span>'
        
        # Get explanations for this feature pair
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
                            <p class="feature-description" style="margin: 10px 0; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 0.9em; color: #555; line-height: 1.4;">{q_explanation}</p>
                        </div>
                        <div class="arrow">{arrow_symbol}</div>
                        <div class="feature-box">
                            <h4>Key: <a href="{k_url}" target="_blank" style="color: #667eea; text-decoration: none;">Feature {k_feat} ↗</a></h4>
                            <p class="feature-description" style="margin: 10px 0; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 0.9em; color: #555; line-height: 1.4;">{k_explanation}</p>
                        </div>
                    </div>
                    {f'<div style="margin-top: 15px; padding: 10px; background: #fff3cd; border-radius: 5px; text-align: center;"><strong>⚡ Self-Interaction</strong> - Potential induction behavior</div>' if is_self else ''}
                    
                    <div class="text-panels">
                        <div class="text-panel">
                            <h5>🔵 Query Feature {q_feat} Activations</h5>
                            <div class="tokens">{query_tokens_html}</div>
                        </div>
                        <div class="text-panel">
                            <h5>🟣 Key Feature {k_feat} Activations</h5>
                            <div class="tokens">{key_tokens_html}</div>
                        </div>
                        <div class="text-panel">
                            <h5>🔀 Combined Interaction</h5>
                            <div class="tokens">{interaction_tokens_html}</div>
                        </div>
                    </div>
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
                card.style.transform = 'translateY(30px)';
                card.style.transition = `all 0.5s ease ${index * 0.05}s`;
                observer.observe(card);
            });
        });
    </script>
</body>
</html>
"""
    
    # Save HTML file
    output_path = Path(output_path)
    output_path.write_text(html_content)
    print(f"Dashboard saved to: {output_path}")
    
    return str(output_path)


def generate_dashboard_from_config(
    model=None,
    sae=None,
    text=None,
    layer=5,
    head=0,
    top_k_features=20,
    top_k_interactions=30,
    use_timestamp=False,
    config_path=None,
    device="cuda"
):
    """
    Single-line function to generate FRA dashboard with flexible parameters.
    
    Can be called in three ways:
    1. With model and sae already loaded: generate_dashboard_from_config(model, sae, text)
    2. With config file: generate_dashboard_from_config(config_path="config.yaml")
    3. With defaults: generate_dashboard_from_config()
    
    Args:
        model: Pre-loaded HookedTransformer model (optional)
        sae: Pre-loaded SAELensAttentionSAE (optional)
        text: Text to analyze (optional, uses default if not provided)
        layer: Layer number (default: 5)
        head: Head number (default: 0)
        top_k_features: Number of top features per position (default: 20)
        top_k_interactions: Number of top interactions to show (default: 30)
        use_timestamp: Whether to add timestamp to filename (default: False)
        config_path: Path to config file (optional, overrides other params)
        device: Device to use (default: "cuda")
        
    Returns:
        Path to generated dashboard
    """
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    torch.set_grad_enabled(False)
    
    # Load from config if provided
    if config_path:
        from fra.utils import load_config, load_dataset_hf
        config = load_config(str(config_path))
        layer = config["sae"]["layer"]
        top_k_features = config.get("fra", {}).get("top_k_features", 20)
        device = config["model"]["device"] if torch.cuda.is_available() else "cpu"
        
        # Load dataset for text if not provided
        if text is None:
            dataset = load_dataset_hf(
                dataset_name=config["dataset"]["name"],
                split=config["dataset"]["split"],
                streaming=config["dataset"].get("streaming", True),
                seed=config["experiment"]["seed"]
            )
            if hasattr(dataset, '__iter__'):
                sample = next(iter(dataset))
                text = sample['text'][:config["dataset"].get("max_length", 128)]
            else:
                sample = dataset[0]
                text = sample['text'][:config["dataset"].get("max_length", 128)]
    
    # Use default text if none provided
    if text is None:
        text = "The cat sat on the mat. The cat was happy. The dog ran in the park. The dog was tired."
    
    # Load model if not provided
    if model is None:
        print("Loading model...")
        model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
    # Load SAE if not provided
    if sae is None:
        from fra.induction_head import SAELensAttentionSAE
        print(f"Loading SAE for layer {layer}...")
        RELEASE = "gpt2-small-hook-z-kk"
        SAE_ID = f"blocks.{layer}.hook_z"
        sae = SAELensAttentionSAE(RELEASE, SAE_ID, device=device)
    
    # Generate dashboard
    print(f"Generating dashboard for layer {layer}, head {head}...")
    dashboard_path = create_fra_dashboard(
        model=model,
        sae=sae,
        text=text,
        layer=layer,
        head=head,
        top_k_features=top_k_features,
        top_k_interactions=top_k_interactions,
        use_timestamp=use_timestamp
    )
    
    print(f"✅ Dashboard saved to: {dashboard_path}")
    return dashboard_path


def main():
    """Generate FRA dashboard for a sample text."""
    # Example usage of the flexible function
    dashboard_path = generate_dashboard_from_config(
        text="The cat sat on the mat. The cat was happy. The dog ran in the park. The dog was tired.",
        layer=5,
        head=0,
        top_k_features=20,
        top_k_interactions=30
    )
    
    print(f"\n📁 Open {dashboard_path} in your browser to view the interactive visualization.")
    return dashboard_path


if __name__ == "__main__":
    import os
    
    try:
        dashboard_path = main()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        os._exit(0)