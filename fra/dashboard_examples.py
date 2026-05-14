#!/usr/bin/env python
"""
Examples of using the single-line dashboard generation function.
"""

from fra.single_sample_viz import generate_dashboard_from_config
from transformer_lens import HookedTransformer
from fra.induction_head import SAELensAttentionSAE
import torch

# Disable gradients for efficiency
torch.set_grad_enabled(False)

print("FRA Dashboard Generation Examples")
print("=" * 50)

# Example 1: Simplest usage - all defaults
print("\n1. Using all defaults:")
dashboard = generate_dashboard_from_config()

# Example 2: Custom text only
print("\n2. Custom text with defaults:")
dashboard = generate_dashboard_from_config(
    text="Alice went to the store. She bought milk. Bob went to the store. He bought bread."
)

# Example 3: Custom parameters
print("\n3. Custom parameters:")
dashboard = generate_dashboard_from_config(
    text="The student who studied hard passed. The students who studied hard passed.",
    layer=6,
    head=10,
    top_k_features=15,
    top_k_interactions=25
)

# Example 4: With pre-loaded model and SAE (most efficient for multiple dashboards)
print("\n4. With pre-loaded model and SAE:")
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
sae = SAELensAttentionSAE("gpt2-small-hook-z-kk", "blocks.5.hook_z", device="cuda")

dashboard = generate_dashboard_from_config(
    model=model,
    sae=sae,
    text="The cat sat on the mat. The cat was happy.",
    layer=5,
    head=0
)

# Example 5: From config file
print("\n5. From config file:")
dashboard = generate_dashboard_from_config(config_path="../config.yaml")

# Example 6: Multiple heads for same text (efficient)
print("\n6. Analyzing multiple heads:")
text = "John gave Mary a book. Mary gave John a pen."
for head in [0, 5, 10]:
    print(f"  Head {head}...")
    dashboard = generate_dashboard_from_config(
        model=model,  # Reuse loaded model
        sae=sae,      # Reuse loaded SAE
        text=text,
        head=head,
        use_timestamp=True  # Add timestamp to avoid overwriting
    )

# Example 7: Batch processing different texts
print("\n7. Batch processing texts:")
texts = [
    "The cat chased the mouse. The cat was fast.",
    "She opened the door. She walked inside.",
    "Paris is the capital of France. London is the capital of England."
]

for i, text in enumerate(texts):
    print(f"  Text {i+1}...")
    dashboard = generate_dashboard_from_config(
        model=model,
        sae=sae,
        text=text,
        use_timestamp=True
    )

print("\n" + "=" * 50)
print("All dashboards generated in fra/results/")
print("Open any .html file in your browser to view.")