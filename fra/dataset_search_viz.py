#!/usr/bin/env python
"""
Visualization for dataset-wide feature interaction search results.

Creates an interactive dashboard showing:
1. Top average non-zero interactions across the dataset
2. Specific examples where interactions were strongest
3. The actual text snippets where these interactions occurred
"""

import torch
import html
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import tarfile

from fra.dataset_search import DatasetSearchResults, load_search_results, InteractionExample
from fra.single_sample_viz import fetch_neuronpedia_explanation, get_neuronpedia_url


def create_dataset_search_dashboard(
    results: DatasetSearchResults,
    top_k_interactions: int = 50,
    output_dir: str = "fra/results",
    use_timestamp: bool = False,
    fetch_explanations: bool = True
) -> str:
    """
    Create dashboard for dataset search results.
    
    Args:
        results: Search results to visualize
        top_k_interactions: Number of top interactions to show
        output_dir: Directory to save dashboard
        use_timestamp: Whether to add timestamp to filename
        fetch_explanations: Whether to fetch Neuronpedia explanations
        
    Returns:
        Path to generated HTML file
    """
    print(f"\nGenerating dataset search dashboard for L{results.layer}H{results.head}...")
    
    # Find top interactions by average strength
    avg_interactions = results.avg_interactions
    interaction_counts = results.interaction_counts
    
    # Handle both sparse and dense tensors
    top_interactions = []
    
    if avg_interactions.is_sparse:
        # Work with sparse tensor
        indices = avg_interactions._indices()
        values = avg_interactions._values()
        
        # Get counts for the same indices
        count_values = interaction_counts._values() if interaction_counts.is_sparse else None
        
        # Create list of interactions
        interactions_list = []
        for i in range(indices.shape[1]):
            q_feat = indices[0, i].item()
            k_feat = indices[1, i].item()
            avg_val = values[i].item()
            count = count_values[i].item() if count_values is not None else 1
            interactions_list.append((q_feat, k_feat, avg_val, count))
        
        # Sort and get top-k
        interactions_list.sort(key=lambda x: x[2], reverse=True)
        
        for q_feat, k_feat, avg_strength, count in interactions_list[:top_k_interactions]:
            # Get examples for this pair
            pair_key = (q_feat, k_feat)
            examples = results.top_examples.get(pair_key, [])
            
            top_interactions.append({
                'q_feat': q_feat,
                'k_feat': k_feat,
                'avg_strength': avg_strength,
                'count': int(count),
                'examples': examples
            })
    else:
        # Handle dense tensor (backward compatibility)
        mask = interaction_counts > 0
        
        if mask.any():
            # Get indices and values of non-zero interactions
            nonzero_indices = torch.nonzero(mask, as_tuple=False)
            nonzero_avg = avg_interactions[mask]
            
            # Get top-k from non-zero values
            top_k = min(top_k_interactions, len(nonzero_avg))
            if top_k > 0:
                top_values, top_indices = torch.topk(nonzero_avg, top_k)
                
                for i in range(len(top_values)):
                    idx = top_indices[i]
                    q_feat = nonzero_indices[idx, 0].item()
                    k_feat = nonzero_indices[idx, 1].item()
                    avg_strength = top_values[i].item()
                    count = interaction_counts[q_feat, k_feat].item()
                    
                    # Get examples for this pair
                    pair_key = (q_feat, k_feat)
                    examples = results.top_examples.get(pair_key, [])
                    
                    top_interactions.append({
                        'q_feat': q_feat,
                        'k_feat': k_feat,
                        'avg_strength': avg_strength,
                        'count': int(count),
                        'examples': examples
                    })
    
    # Fetch Neuronpedia explanations if requested
    feature_explanations = {}
    if fetch_explanations:
        print("Fetching feature explanations from Neuronpedia...")
        unique_features = set()
        for interaction in top_interactions:
            unique_features.add(interaction['q_feat'])
            unique_features.add(interaction['k_feat'])
        
        for feat in unique_features:
            explanation = fetch_neuronpedia_explanation(results.layer, feat)
            feature_explanations[feat] = explanation
            if explanation != "No explanation available":
                print(f"  Feature {feat}: {explanation[:50]}...")
    
    # Generate HTML
    html_content = generate_dataset_search_html(
        results=results,
        top_interactions=top_interactions,
        feature_explanations=feature_explanations
    )
    
    # Save to file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_search_L{results.layer}H{results.head}_{timestamp}.html"
    else:
        filename = f"dataset_search_L{results.layer}H{results.head}.html"
    
    filepath = output_path / filename
    with open(filepath, 'w') as f:
        f.write(html_content)
    
    print(f"Dashboard saved to: {filepath}")
    return str(filepath)


def generate_dataset_search_html(
    results: DatasetSearchResults,
    top_interactions: List[Dict],
    feature_explanations: Dict[int, str]
) -> str:
    """Generate HTML content for dataset search dashboard."""
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Feature Interactions - Layer {results.layer}, Head {results.head}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
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
        
        .interaction-metrics {{
            display: flex;
            gap: 20px;
            align-items: center;
        }}
        
        .metric {{
            text-align: center;
        }}
        
        .metric-label {{
            font-size: 0.8em;
            color: #6c757d;
            text-transform: uppercase;
        }}
        
        .metric-value {{
            font-size: 1.2em;
            font-weight: bold;
            color: #495057;
        }}
        
        .interaction-strength {{
            font-size: 1.5em;
            font-weight: bold;
            color: #28a745;
        }}
        
        .feature-pair {{
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 20px;
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
            background: white;
            border-radius: 4px;
            font-size: 0.9em;
            color: #555;
            line-height: 1.4;
        }}
        
        .arrow {{
            font-size: 2em;
            color: #667eea;
        }}
        
        .examples-section {{
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
        }}
        
        .examples-header {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.1em;
        }}
        
        .example-card {{
            background: #f8f9fa;
            border-left: 3px solid #667eea;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
        }}
        
        .example-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 0.9em;
            color: #6c757d;
        }}
        
        .example-strength {{
            font-weight: bold;
            color: #28a745;
        }}
        
        .example-text {{
            font-family: 'Courier New', monospace;
            background: white;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            line-height: 1.6;
            overflow-x: auto;
        }}
        
        .token-highlight {{
            padding: 2px 4px;
            border-radius: 3px;
            margin: 0 2px;
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
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%);
            border: 1px solid rgba(102, 126, 234, 0.7);
            font-weight: bold;
        }}
        
        .position-info {{
            font-size: 0.85em;
            color: #6c757d;
            margin-top: 5px;
        }}
        
        .self-interaction {{
            background: #fff9e6;
            border-color: #ffc107;
        }}
        
        .note {{
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            color: #0c5460;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Dataset-Wide Feature Interaction Analysis</h1>
            <p>Layer {results.layer}, Head {results.head} - Analyzed {results.num_samples} samples</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Samples Analyzed</h3>
                <div class="value">{results.num_samples:,}</div>
            </div>
            <div class="stat-card">
                <h3>Unique Interactions</h3>
                <div class="value">{len(results.top_examples):,}</div>
            </div>
            <div class="stat-card">
                <h3>Total Features</h3>
                <div class="value">{results.d_sae:,}</div>
            </div>
            <div class="stat-card">
                <h3>Non-zero Rate</h3>
                <div class="value">{(results.interaction_counts > 0).float().mean().item()*100:.2f}%</div>
            </div>
            <div class="stat-card">
                <h3>Layer</h3>
                <div class="value">{results.layer}</div>
            </div>
            <div class="stat-card">
                <h3>Head</h3>
                <div class="value">{results.head}</div>
            </div>
        </div>
        
        <div class="section">
            <div class="note">
                <strong>📊 Analysis Method:</strong> This dashboard shows feature interactions averaged across {results.num_samples} text samples.
                For each feature pair, we show the average interaction strength when non-zero, the number of occurrences,
                and specific examples where the interaction was strongest.
            </div>
            
            <h2>🔥 Top Feature Interactions Across Dataset</h2>
            <div class="interactions-grid">
"""
    
    # Add interaction cards with examples
    for i, interaction in enumerate(top_interactions[:50]):
        q_feat = interaction['q_feat']
        k_feat = interaction['k_feat']
        avg_strength = interaction['avg_strength']
        count = interaction['count']
        examples = interaction['examples']
        
        q_url = get_neuronpedia_url(results.layer, q_feat)
        k_url = get_neuronpedia_url(results.layer, k_feat)
        
        q_explanation = feature_explanations.get(q_feat, "No explanation available")
        k_explanation = feature_explanations.get(k_feat, "No explanation available")
        
        is_self = (q_feat == k_feat)
        card_class = "interaction-card self-interaction" if is_self else "interaction-card"
        
        html_content += f"""
                <div class="{card_class}">
                    <div class="interaction-header">
                        <h3>#{i+1} - Features ({q_feat}, {k_feat})</h3>
                        <div class="interaction-metrics">
                            <div class="metric">
                                <div class="metric-label">Avg Strength</div>
                                <div class="metric-value interaction-strength">{avg_strength:.4f}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Occurrences</div>
                                <div class="metric-value">{count}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="feature-pair">
                        <div class="feature-box">
                            <h4>Query: <a href="{q_url}" target="_blank" style="color: #667eea; text-decoration: none;">Feature {q_feat} ↗</a></h4>
                            <div class="feature-description">{q_explanation}</div>
                        </div>
                        <div class="arrow">→</div>
                        <div class="feature-box">
                            <h4>Key: <a href="{k_url}" target="_blank" style="color: #667eea; text-decoration: none;">Feature {k_feat} ↗</a></h4>
                            <div class="feature-description">{k_explanation}</div>
                        </div>
                    </div>
                    
                    {f'<div style="margin: 15px 0; padding: 10px; background: #fff3cd; border-radius: 5px; text-align: center;"><strong>⚡ Self-Interaction</strong> - Feature attending to itself</div>' if is_self else ''}
"""
        
        # Add examples if available
        if examples:
            html_content += f"""
                    <div class="examples-section">
                        <div class="examples-header">📝 Top Examples (strongest interactions):</div>
"""
            
            for j, example in enumerate(examples[:3]):  # Show top 3 examples
                # Create highlighted text
                tokens = example.text.split()[:30]  # Limit to first 30 tokens for display
                highlighted_text = ""
                
                for idx, token in enumerate(tokens):
                    if idx == min(example.query_pos, len(tokens)-1):
                        highlighted_text += f'<span class="token-highlight highlight-query">{html.escape(token)}</span> '
                    elif idx == min(example.key_pos, len(tokens)-1):
                        highlighted_text += f'<span class="token-highlight highlight-key">{html.escape(token)}</span> '
                    else:
                        highlighted_text += html.escape(token) + " "
                
                html_content += f"""
                        <div class="example-card">
                            <div class="example-header">
                                <span>Example {j+1} (Sample #{example.sample_idx})</span>
                                <span class="example-strength">Strength: {example.strength:.4f}</span>
                            </div>
                            <div class="example-text">
                                {highlighted_text}...
                            </div>
                            <div class="position-info">
                                Query token (pos {example.query_pos}): "{html.escape(example.query_token)}" | 
                                Key token (pos {example.key_pos}): "{html.escape(example.key_token)}"
                            </div>
                        </div>
"""
            
            html_content += """
                    </div>
"""
        
        html_content += """
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
                card.style.transition = `all 0.5s ease ${Math.min(index * 0.05, 1)}s`;
                observer.observe(card);
            });
        });
    </script>
</body>
</html>
"""
    
    return html_content


def create_dashboard_from_search(
    search_results_path: Optional[str] = None,
    results: Optional[DatasetSearchResults] = None,
    top_k_interactions: int = 50,
    use_timestamp: bool = False
) -> str:
    """
    Create dashboard from search results.
    
    Can provide either a path to saved results or the results object directly.
    
    Args:
        search_results_path: Path to saved search results
        results: Search results object (if already loaded)
        top_k_interactions: Number of top interactions to show
        use_timestamp: Whether to add timestamp to filename
        
    Returns:
        Path to generated dashboard
    """
    # Load results if path provided
    if results is None:
        if search_results_path is None:
            # Try to find most recent results
            results_dir = Path("fra/results")
            search_files = list(results_dir.glob("dataset_search_*.pkl"))
            if not search_files:
                raise ValueError("No search results found. Run dataset_search.py first.")
            search_results_path = str(max(search_files, key=lambda x: x.stat().st_mtime))
            print(f"Using most recent search results: {search_results_path}")
        
        results = load_search_results(search_results_path)
    
    # Create dashboard
    return create_dataset_search_dashboard(
        results=results,
        top_k_interactions=top_k_interactions,
        use_timestamp=use_timestamp
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create dashboard from dataset search results")
    parser.add_argument("--results-path", type=str, help="Path to search results file")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top interactions to show")
    parser.add_argument("--timestamp", action="store_true", help="Add timestamp to filename")
    
    args = parser.parse_args()
    
    dashboard_path = create_dashboard_from_search(
        search_results_path=args.results_path,
        top_k_interactions=args.top_k,
        use_timestamp=args.timestamp
    )
    
    print(f"\n✅ Dashboard created: {dashboard_path}")
    print("Open the HTML file to explore dataset-wide feature interactions.")