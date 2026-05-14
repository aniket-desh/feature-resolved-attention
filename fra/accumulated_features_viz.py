#!/usr/bin/env python
"""
Dashboard for visualizing top accumulated feature pairs from multiple samples.
Shows the most important feature interactions discovered across a dataset.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests



def get_feature_info_from_neuronpedia(feature_id: int, layer: int = 5) -> Dict[str, Any]:
    """Fetch feature interpretation from Neuronpedia API."""
    try:
        url = f"https://www.neuronpedia.org/api/feature/gpt2-small/{layer}-att-kk/{feature_id}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'description': data.get('description', f'Feature {feature_id}'),
                'top_activations': data.get('top_activations', [])[:3],
                'activation_histogram': data.get('activation_histogram', None)
            }
    except:
        pass
    return {'description': f'Feature {feature_id}', 'top_activations': [], 'activation_histogram': None}


def process_sparse_accumulation(
    interaction_sum_sparse: torch.sparse.FloatTensor,
    interaction_count_sparse: torch.sparse.FloatTensor,
    top_k: int = 100
) -> Dict[str, Any]:
    """
    Process sparse accumulation results to extract top feature pairs.
    
    Returns:
        Dictionary with top feature pairs and their statistics
    """
    # Coalesce to combine duplicate indices
    interaction_sum = interaction_sum_sparse.coalesce()
    interaction_count = interaction_count_sparse.coalesce()
    
    # Extract indices and values
    sum_indices = interaction_sum.indices()
    sum_values = interaction_sum.values()
    count_indices = interaction_count.indices()
    count_values = interaction_count.values()
    
    # Create lookup for counts
    count_dict = {}
    for i in range(count_indices.shape[1]):
        q_feat = count_indices[0, i].item()
        k_feat = count_indices[1, i].item()
        count_dict[(q_feat, k_feat)] = count_values[i].item()
    
    # Calculate averages and collect all pairs
    feature_pairs = []
    for i in range(sum_indices.shape[1]):
        q_feat = sum_indices[0, i].item()
        k_feat = sum_indices[1, i].item()
        sum_val = sum_values[i].item()
        count_val = count_dict.get((q_feat, k_feat), 1)
        avg_val = sum_val / count_val
        
        feature_pairs.append({
            'query_feature': q_feat,
            'key_feature': k_feat,
            'sum': sum_val,
            'count': count_val,
            'average': avg_val
        })
    
    # Sort by average interaction strength
    feature_pairs.sort(key=lambda x: x['average'], reverse=True)
    
    # Keep top k
    top_pairs = feature_pairs[:top_k]
    
    return {
        'top_pairs': top_pairs,
        'total_pairs': len(feature_pairs),
        'max_average': top_pairs[0]['average'] if top_pairs else 0,
        'min_average': top_pairs[-1]['average'] if top_pairs else 0
    }


def create_accumulated_dashboard(
    interaction_sum_sparse: torch.sparse.FloatTensor,
    interaction_count_sparse: torch.sparse.FloatTensor,
    num_samples: int,
    layer: int = 5,
    head: int = 0,
    top_k: int = 50,
    output_path: Optional[str] = None,
    fetch_neuronpedia: bool = True
) -> str:
    """
    Create an interactive dashboard for accumulated feature interactions.
    
    Args:
        interaction_sum_sparse: Sparse tensor of summed interactions
        interaction_count_sparse: Sparse tensor of interaction counts
        num_samples: Number of samples processed
        layer: Layer analyzed
        head: Head analyzed
        top_k: Number of top pairs to visualize
        output_path: Where to save the dashboard
        fetch_neuronpedia: Whether to fetch feature descriptions from Neuronpedia
        
    Returns:
        Path to saved dashboard
    """
    # Process sparse tensors to get top pairs
    results = process_sparse_accumulation(
        interaction_sum_sparse, 
        interaction_count_sparse,
        top_k=top_k
    )
    
    top_pairs = results['top_pairs']
    
    # Fetch feature descriptions if requested
    feature_descriptions = {}
    if fetch_neuronpedia:
        print(f"Fetching feature descriptions from Neuronpedia...")
        unique_features = set()
        for pair in top_pairs:
            unique_features.add(pair['query_feature'])
            unique_features.add(pair['key_feature'])
        
        for feat_id in unique_features:
            info = get_feature_info_from_neuronpedia(feat_id, layer)
            feature_descriptions[feat_id] = info['description']
    
    # Create visualizations
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            f'Top {min(20, len(top_pairs))} Feature Interactions (Averaged)',
            'Interaction Frequency Distribution',
            'Feature Interaction Heatmap',
            'Self-Interaction Diagonal',
            'Interaction Strength vs Frequency',
            'Feature Participation Count'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'histogram'}],
            [{'type': 'heatmap'}, {'type': 'bar'}],
            [{'type': 'scatter'}, {'type': 'bar'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # 1. Top interactions bar chart
    top_20 = top_pairs[:20]
    pair_labels = []
    pair_values = []
    pair_counts = []
    
    for pair in top_20:
        q_feat = pair['query_feature']
        k_feat = pair['key_feature']
        q_desc = feature_descriptions.get(q_feat, f"F{q_feat}")[:30]
        k_desc = feature_descriptions.get(k_feat, f"F{k_feat}")[:30]
        
        if q_feat == k_feat:
            label = f"🔄 {q_desc}"
        else:
            label = f"{q_desc} → {k_desc}"
        
        pair_labels.append(label)
        pair_values.append(pair['average'])
        pair_counts.append(pair['count'])
    
    fig.add_trace(
        go.Bar(
            x=pair_values,
            y=pair_labels,
            orientation='h',
            marker_color=['red' if top_20[i]['query_feature'] == top_20[i]['key_feature'] else 'blue' 
                         for i in range(len(top_20))],
            text=[f"Count: {c}" for c in pair_counts],
            textposition='auto',
            name='Average Interaction'
        ),
        row=1, col=1
    )
    
    # 2. Frequency distribution
    all_counts = [p['count'] for p in top_pairs]
    fig.add_trace(
        go.Histogram(
            x=all_counts,
            nbinsx=30,
            name='Frequency',
            marker_color='green'
        ),
        row=1, col=2
    )
    
    # 3. Interaction heatmap (top features)
    # Get most active features
    feature_activity = {}
    for pair in top_pairs:
        q = pair['query_feature']
        k = pair['key_feature']
        feature_activity[q] = feature_activity.get(q, 0) + pair['average']
        if q != k:
            feature_activity[k] = feature_activity.get(k, 0) + pair['average']
    
    top_features = sorted(feature_activity.items(), key=lambda x: x[1], reverse=True)[:30]
    top_feature_ids = [f[0] for f in top_features]
    
    # Build heatmap matrix
    heatmap_size = len(top_feature_ids)
    heatmap_matrix = np.zeros((heatmap_size, heatmap_size))
    
    for pair in top_pairs:
        if pair['query_feature'] in top_feature_ids and pair['key_feature'] in top_feature_ids:
            i = top_feature_ids.index(pair['query_feature'])
            j = top_feature_ids.index(pair['key_feature'])
            heatmap_matrix[i, j] = pair['average']
            if i != j:
                heatmap_matrix[j, i] = pair['average']
    
    feature_labels = [feature_descriptions.get(f, f"F{f}")[:20] for f in top_feature_ids]
    
    fig.add_trace(
        go.Heatmap(
            z=heatmap_matrix,
            x=feature_labels,
            y=feature_labels,
            colorscale='Viridis',
            text=[[f"{val:.3f}" if val > 0 else "" for val in row] for row in heatmap_matrix],
            texttemplate="%{text}",
            textfont={"size": 8},
            showscale=True
        ),
        row=2, col=1
    )
    
    # 4. Self-interactions (diagonal elements)
    self_interactions = [(p['query_feature'], p['average'], p['count']) 
                        for p in top_pairs if p['query_feature'] == p['key_feature']]
    self_interactions.sort(key=lambda x: x[1], reverse=True)
    
    if self_interactions:
        self_labels = [feature_descriptions.get(f[0], f"F{f[0]}")[:30] for f in self_interactions[:10]]
        self_values = [f[1] for f in self_interactions[:10]]
        self_counts = [f[2] for f in self_interactions[:10]]
        
        fig.add_trace(
            go.Bar(
                x=self_values,
                y=self_labels,
                orientation='h',
                marker_color='red',
                text=[f"Count: {c}" for c in self_counts],
                textposition='auto',
                name='Self-Interaction'
            ),
            row=2, col=2
        )
    
    # 5. Interaction strength vs frequency scatter
    fig.add_trace(
        go.Scatter(
            x=[p['count'] for p in top_pairs],
            y=[p['average'] for p in top_pairs],
            mode='markers',
            marker=dict(
                size=8,
                color=[p['average'] for p in top_pairs],
                colorscale='Plasma',
                showscale=True
            ),
            text=[f"F{p['query_feature']}→F{p['key_feature']}" for p in top_pairs],
            hovertemplate="<b>%{text}</b><br>Count: %{x}<br>Average: %{y:.4f}<extra></extra>",
            name='Interactions'
        ),
        row=3, col=1
    )
    
    # 6. Feature participation count
    feature_participation = {}
    for pair in top_pairs:
        q = pair['query_feature']
        k = pair['key_feature']
        feature_participation[q] = feature_participation.get(q, 0) + 1
        if q != k:
            feature_participation[k] = feature_participation.get(k, 0) + 1
    
    top_participants = sorted(feature_participation.items(), key=lambda x: x[1], reverse=True)[:15]
    
    fig.add_trace(
        go.Bar(
            x=[p[1] for p in top_participants],
            y=[feature_descriptions.get(p[0], f"F{p[0]}")[:30] for p in top_participants],
            orientation='h',
            marker_color='purple',
            name='Participation Count'
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"Accumulated Feature Interactions<br>Layer {layer}, Head {head} | {num_samples} samples | Top {top_k} pairs",
        height=1400,
        showlegend=False,
        font=dict(size=10)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Average Interaction Strength", row=1, col=1)
    fig.update_xaxes(title_text="Occurrence Count", row=1, col=2)
    fig.update_xaxes(title_text="Average Interaction", row=2, col=2)
    fig.update_xaxes(title_text="Occurrence Count", row=3, col=1)
    fig.update_yaxes(title_text="Average Interaction Strength", row=3, col=1)
    fig.update_xaxes(title_text="Number of Interactions", row=3, col=2)
    
    # Generate HTML with embedded data
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Accumulated FRA Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin-bottom: 20px;
            }}
            .stat-card {{
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .stat-label {{
                color: #7f8c8d;
                margin-top: 5px;
            }}
            .top-pairs {{
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                margin-top: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .pair-item {{
                padding: 10px;
                border-bottom: 1px solid #ecf0f1;
            }}
            .pair-item:hover {{
                background-color: #f8f9fa;
            }}
            .self-interaction {{
                background-color: #ffe6e6;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔬 Accumulated Feature-Resolved Attention Dashboard</h1>
            <p>Analysis of feature interactions across {num_samples} text samples</p>
            <p>Layer {layer}, Head {head}</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{results['total_pairs']:,}</div>
                <div class="stat-label">Total Feature Pairs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{num_samples}</div>
                <div class="stat-label">Samples Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{results['max_average']:.4f}</div>
                <div class="stat-label">Max Average Interaction</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(self_interactions)}</div>
                <div class="stat-label">Self-Interactions</div>
            </div>
        </div>
        
        <div id="plotly-chart"></div>
        
        <div class="top-pairs">
            <h2>Top 10 Feature Interactions</h2>
            {"".join([f'''
            <div class="pair-item {'self-interaction' if p['query_feature'] == p['key_feature'] else ''}">
                <strong>#{i+1}</strong> 
                Feature {p['query_feature']} {'↔' if p['query_feature'] == p['key_feature'] else '→'} Feature {p['key_feature']}<br>
                <small>Average: {p['average']:.4f} | Count: {p['count']} | Sum: {p['sum']:.4f}</small><br>
                <small style="color: #7f8c8d;">
                    {feature_descriptions.get(p['query_feature'], 'No description')} 
                    {'(self)' if p['query_feature'] == p['key_feature'] else '→ ' + feature_descriptions.get(p['key_feature'], 'No description')}
                </small>
            </div>
            ''' for i, p in enumerate(top_pairs[:10])])}
        </div>
        
        <script>
            var plotlyData = {fig.to_json()};
            Plotly.newPlot('plotly-chart', plotlyData.data, plotlyData.layout);
        </script>
    </body>
    </html>
    """
    
    # Save dashboard
    if output_path is None:
        output_path = Path(__file__).parent.parent / "results" / f"accumulated_fra_L{layer}H{head}_{num_samples}samples.html"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Dashboard saved to: {output_path}")
    return str(output_path)


def generate_accumulated_dashboard_from_results(
    results: Dict[str, Any],
    layer: int = 5,
    head: int = 0,
    top_k: int = 50,
    output_path: Optional[str] = None,
    fetch_neuronpedia: bool = True
) -> str:
    """
    Generate dashboard from the results dictionary returned by compute_dataset_average_sparse.
    
    Args:
        results: Dictionary with 'interaction_sum_sparse', 'interaction_count_sparse', 'num_samples'
        layer: Layer analyzed
        head: Head analyzed
        top_k: Number of top pairs to visualize
        output_path: Where to save the dashboard
        fetch_neuronpedia: Whether to fetch feature descriptions
        
    Returns:
        Path to saved dashboard
    """
    return create_accumulated_dashboard(
        interaction_sum_sparse=results['interaction_sum_sparse'],
        interaction_count_sparse=results['interaction_count_sparse'],
        num_samples=results['num_samples'],
        layer=layer,
        head=head,
        top_k=top_k,
        output_path=output_path,
        fetch_neuronpedia=fetch_neuronpedia
    )


if __name__ == "__main__":
    # Example: Run accumulation and generate dashboard
    import time
    from fra.dataset_average_sparse import compute_dataset_average_sparse
    from fra.utils import load_dataset_hf
    
    torch.set_grad_enabled(False)
    
    print("="*60)
    print("Generating Accumulated Features Dashboard")
    print("="*60)
    
    # Load model and SAE
    print("\nLoading model and SAE...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained('gpt2-small', device='cuda')
    sae = SAELensAttentionSAE('gpt2-small-hook-z-kk', 'blocks.5.hook_z', device='cuda')
    print(f"Loaded in {time.time()-t0:.1f}s")
    
    # Load samples
    print("\nLoading dataset samples...")
    dataset = load_dataset_hf(streaming=True)
    dataset_texts = []
    for i, item in enumerate(dataset):
        if i >= 10:  # Use 10 samples for quick test
            break
        text = item.get('text', '') if isinstance(item, dict) else str(item)
        if len(text.split()) > 10:
            dataset_texts.append(text[:500])
    
    print(f"Loaded {len(dataset_texts)} texts")
    
    # Compute accumulation
    print("\nComputing FRA accumulation...")
    t0 = time.time()
    results = compute_dataset_average_sparse(
        model=model,
        sae=sae,
        dataset_texts=dataset_texts,
        layer=5,
        head=0,
        filter_self_interactions=False,  # Keep self-interactions to find induction
        use_absolute_values=True,
        verbose=True
    )
    print(f"Computed in {time.time()-t0:.1f}s")
    
    # Generate dashboard
    print("\nGenerating dashboard...")
    dashboard_path = generate_accumulated_dashboard_from_results(
        results=results,
        layer=5,
        head=0,
        top_k=100,
        fetch_neuronpedia=True
    )
    
    print(f"\n✅ Dashboard ready at: {dashboard_path}")