"""
Utilities for getting LLM activations at specific hookpoints.
"""

import torch
from typing import List, Optional, Union
from transformer_lens import HookedTransformer
from einops import rearrange


def get_llm_activations(
    model: HookedTransformer,
    input_text: str,
    hook_point: str = "attn.hook_z",  # The attention output hook
    layers: Optional[Union[int, List[int]]] = None,
    max_length: Optional[int] = None
) -> torch.Tensor:
    """
    Get LLM activations for given input text at specified hookpoints.
    
    Args:
        model: The HookedTransformer model
        input_text: Input text to analyze
        hook_point: The type of hook to use (e.g., "attn.hook_z" for attention output)
        layers: Which layers to get activations from. If None, gets all layers.
                Can be a single int or list of ints.
        max_length: Maximum sequence length (truncates if needed)
        
    Returns:
        Tensor of shape (sequence_length, num_layers, hidden_dim) if getting multiple layers
        or (sequence_length, hidden_dim) if getting a single layer
    """
    # Tokenize the input
    tokens = model.tokenizer.encode(input_text)
    if max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]
    # Get the device from the model's parameters
    device = next(model.parameters()).device
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)  # [batch=1, seq_len]
    
    # Determine which layers to get
    if layers is None:
        layers = list(range(model.cfg.n_layers))
    elif isinstance(layers, int):
        layers = [layers]
    
    # Build the hook names
    hook_names = [f"blocks.{layer}.{hook_point}" for layer in layers]
    
    # Run the model with cache
    _, cache = model.run_with_cache(tokens, names_filter=hook_names)
    
    # Stack the activations from different layers
    # Each cache[hook_name] has shape [batch=1, seq_len, hidden_dim]
    if len(layers) == 1:
        # Single layer: return [seq_len, hidden_dim]
        activations = cache[hook_names[0]].squeeze(0)  # Remove batch dimension
    else:
        # Multiple layers: stack and return [seq_len, num_layers, hidden_dim]
        activations_list = [cache[hook_name].squeeze(0) for hook_name in hook_names]
        activations = torch.stack(activations_list, dim=1)  # [seq_len, num_layers, hidden_dim]
    
    return activations


def get_attention_activations(
    model: HookedTransformer,
    input_text: str,
    layer: int,
    max_length: Optional[int] = 128
) -> torch.Tensor:
    """
    Get concatenated attention output activations for a specific layer.
    
    This is a convenience function specifically for attention outputs,
    which is what the attention SAEs are trained on.
    
    Args:
        model: The HookedTransformer model
        input_text: Input text to analyze
        layer: Which layer to get activations from
        max_length: Maximum sequence length
        
    Returns:
        Tensor of shape (sequence_length, hidden_dim)
    """
    return get_llm_activations(
        model=model,
        input_text=input_text,
        hook_point="hook_attn_out",  # Concatenated attention output hook
        layers=layer,
        max_length=max_length
    )


def get_attention_pattern(
    model: HookedTransformer,
    input_text: str,
    layer: int,
    head: int,
    max_length: Optional[int] = 128
) -> torch.Tensor:
    """
    Get the attention pattern for a specific head.
    
    Args:
        model: The HookedTransformer model
        input_text: Input text to analyze
        layer: Which layer
        head: Which attention head
        max_length: Maximum sequence length
        
    Returns:
        Attention pattern tensor of shape (seq_len, seq_len)
    """
    # Tokenize
    tokens = model.tokenizer.encode(input_text)
    if max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]
    # Get the device from the model's parameters
    device = next(model.parameters()).device
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)
    
    # Get attention patterns
    hook_name = f"blocks.{layer}.attn.hook_pattern"
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    
    # Extract the specific head's pattern
    # Shape: [batch, head, seq_len, seq_len]
    attention_pattern = cache[hook_name][0, head, :, :]  # [seq_len, seq_len]
    
    return attention_pattern