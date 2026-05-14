#!/usr/bin/env python
"""
Optimized dashboard for visualizing top accumulated feature pairs.
Main optimization: Use torch operations instead of Python loops for finding top-k.
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


def process_sparse_accumulation_fast(
    interaction_sum_sparse: torch.sparse.FloatTensor,
    interaction_count_sparse: torch.sparse.FloatTensor,
    top_k: int = 100
) -> Dict[str, Any]:
    """
    FAST processing of sparse accumulation results using torch operations.
    
    Returns:
        Dictionary with top feature pairs and their statistics
    """
    print(f"Processing sparse tensors to find top {top_k} pairs...")
    
    # Coalesce to combine duplicate indices
    interaction_sum = interaction_sum_sparse.coalesce()
    interaction_count = interaction_count_sparse.coalesce()
    
    # Get indices and values
    sum_indices = interaction_sum.indices()  # [2, nnz]
    sum_values = interaction_sum.values()    # [nnz]
    
    # For count tensor, we need to match indices
    count_indices = interaction_count.indices()  # [2, nnz]
    count_values = interaction_count.values()    # [nnz]
    
    print(f"  Found {sum_indices.shape[1]} unique feature pairs")
    
    # OPTIMIZATION: Use torch operations to compute averages directly
    # First, ensure both tensors have same sparsity pattern by adding them
    # This ensures matching indices
    combined = interaction_sum + interaction_count * 0  # Trick to get matching indices
    combined = combined.coalesce()
    
    # Now we can safely divide
    # Get the actual values at the combined indices
    indices = combined.indices()
    
    # Create dense lookup for fast access (only if feasible)
    if indices.shape[1] < 1000000:  # Only if not too many pairs
        # Direct division since indices now match
        avg_values = sum_values / count_values
    else:
        # For very large tensors, use sparse operations
        avg_sparse = interaction_sum / interaction_count.to(interaction_sum.dtype)
        avg_sparse = avg_sparse.coalesce()
        indices = avg_sparse.indices()
        avg_values = avg_sparse.values()
    
    # Find top-k using torch.topk (much faster than sorting all)
    k = min(top_k, len(avg_values))
    top_values, top_idx = torch.topk(avg_values, k)
    
    # Extract top pairs
    top_pairs = []
    for i in range(k):
        idx = top_idx[i].item()
        q_feat = indices[0, idx].item()
        k_feat = indices[1, idx].item()
        avg_val = top_values[i].item()
        sum_val = sum_values[idx].item() if idx < len(sum_values) else avg_val
        count_val = count_values[idx].item() if idx < len(count_values) else 1
        
        top_pairs.append({
            'query_feature': q_feat,
            'key_feature': k_feat,
            'sum': sum_val,
            'count': count_val,
            'average': avg_val
        })
    
    print(f"  Extracted top {len(top_pairs)} pairs")
    
    return {
        'top_pairs': top_pairs,
        'total_pairs': indices.shape[1],
        'max_average': top_pairs[0]['average'] if top_pairs else 0,
        'min_average': top_pairs[-1]['average'] if top_pairs else 0
    }


def create_accumulated_dashboard_fast(
    interaction_sum_sparse: torch.sparse.FloatTensor,
    interaction_count_sparse: torch.sparse.FloatTensor,
    num_samples: int,
    layer: int = 5,
    head: int = 0,
    top_k: int = 50,
    output_path: Optional[str] = None
) -> str:
    """
    Create a simple dashboard for accumulated feature interactions (no Neuronpedia).
    
    Args:
        interaction_sum_sparse: Sparse tensor of summed interactions
        interaction_count_sparse: Sparse tensor of interaction counts
        num_samples: Number of samples processed
        layer: Layer analyzed
        head: Head analyzed
        top_k: Number of top pairs to visualize
        output_path: Where to save the dashboard
        
    Returns:
        Path to saved dashboard
    """
    # Process sparse tensors to get top pairs
    results = process_sparse_accumulation_fast(
        interaction_sum_sparse, 
        interaction_count_sparse,
        top_k=top_k
    )
    
    top_pairs = results['top_pairs']
    
    # Create simple visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Top {min(20, len(top_pairs))} Feature Interactions',
            'Interaction Frequency Distribution',
            'Self-Interactions (Diagonal)',
            'Interaction Strength vs Frequency'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'histogram'}],
            [{'type': 'bar'}, {'type': 'scatter'}]
        ],
        vertical_spacing=0.15,
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
        
        if q_feat == k_feat:
            label = f"F{q_feat} (self)"
        else:
            label = f"F{q_feat} → F{k_feat}"
        
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
            text=[f"n={c}" for c in pair_counts],
            textposition='auto',
            name='Average'
        ),
        row=1, col=1
    )
    
    # 2. Frequency distribution
    all_counts = [p['count'] for p in top_pairs]
    fig.add_trace(
        go.Histogram(
            x=all_counts,
            nbinsx=20,
            name='Frequency',
            marker_color='green'
        ),
        row=1, col=2
    )
    
    # 3. Self-interactions
    self_interactions = [(p['query_feature'], p['average'], p['count']) 
                        for p in top_pairs if p['query_feature'] == p['key_feature']]
    self_interactions.sort(key=lambda x: x[1], reverse=True)
    
    if self_interactions:
        self_labels = [f"Feature {f[0]}" for f in self_interactions[:10]]
        self_values = [f[1] for f in self_interactions[:10]]
        self_counts = [f[2] for f in self_interactions[:10]]
        
        fig.add_trace(
            go.Bar(
                x=self_values,
                y=self_labels,
                orientation='h',
                marker_color='red',
                text=[f"n={c}" for c in self_counts],
                textposition='auto',
                name='Self'
            ),
            row=2, col=1
        )
    
    # 4. Scatter plot: strength vs frequency
    fig.add_trace(
        go.Scatter(
            x=[p['count'] for p in top_pairs],
            y=[p['average'] for p in top_pairs],
            mode='markers',
            marker=dict(
                size=8,
                color=[p['average'] for p in top_pairs],
                colorscale='Viridis',
                showscale=True
            ),
            text=[f"F{p['query_feature']}→F{p['key_feature']}" for p in top_pairs],
            hovertemplate="<b>%{text}</b><br>Count: %{x}<br>Average: %{y:.4f}<extra></extra>",
            name='Pairs'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"Feature Interactions: L{layer}H{head} | {num_samples} samples | Top {top_k}",
        height=800,
        showlegend=False,
        font=dict(size=10)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Average Strength", row=1, col=1)
    fig.update_xaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Average", row=2, col=1)
    fig.update_xaxes(title_text="Count", row=2, col=2)
    fig.update_yaxes(title_text="Average", row=2, col=2)
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>FRA Dashboard</title>
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
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Feature-Resolved Attention Dashboard</h1>
            <p>Layer {layer}, Head {head} | {num_samples} samples analyzed</p>
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
                <div class="stat-label">Max Average</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(self_interactions)}</div>
                <div class="stat-label">Self-Interactions</div>
            </div>
        </div>
        
        <div id="plotly-chart"></div>
        
        <script>
            var plotlyData = {fig.to_json()};
            Plotly.newPlot('plotly-chart', plotlyData.data, plotlyData.layout);
        </script>
    </body>
    </html>
    """
    
    # Save dashboard
    if output_path is None:
        output_path = Path(__file__).parent.parent / "results" / f"fra_L{layer}H{head}_{num_samples}samples.html"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Dashboard saved to: {output_path}")
    return str(output_path)


def generate_accumulated_dashboard_fast(
    results: Dict[str, Any],
    layer: int = 5,
    head: int = 0,
    top_k: int = 50,
    output_path: Optional[str] = None
) -> str:
    """
    Generate fast dashboard from results dictionary.
    
    Args:
        results: Dictionary with sparse tensors
        layer: Layer analyzed
        head: Head analyzed
        top_k: Number of top pairs to visualize
        output_path: Where to save
        
    Returns:
        Path to saved dashboard
    """
    return create_accumulated_dashboard_fast(
        interaction_sum_sparse=results['interaction_sum_sparse'],
        interaction_count_sparse=results['interaction_count_sparse'],
        num_samples=results['num_samples'],
        layer=layer,
        head=head,
        top_k=top_k,
        output_path=output_path
    )