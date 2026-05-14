"""
Simple SAE wrapper for attention SAEs that aren't in standard SAE-Lens format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class SimpleAttentionSAE:
    """Wrapper for attention SAEs with encode/decode methods."""
    
    def __init__(self, sae_data: Dict[str, Any]):
        """Initialize SAE from loaded data.
        
        Args:
            sae_data: Dictionary containing state_dict, config, layer, and device
        """
        self.state_dict = sae_data['state_dict']
        self.config = sae_data['config']
        self.layer = sae_data['layer']
        self.device = sae_data['device']
        
        # Extract weights and biases
        self.W_enc = self.state_dict['W_enc'].to(self.device)  # [d_model, d_sae]
        self.W_dec = self.state_dict['W_dec'].to(self.device)  # [d_sae, d_model]
        self.b_enc = self.state_dict['b_enc'].to(self.device)  # [d_sae]
        self.b_dec = self.state_dict['b_dec'].to(self.device)  # [d_model]
        
        # Get dimensions
        self.d_in = self.W_enc.shape[0]  # Input dimension (768 for GPT-2 small)
        self.d_sae = self.W_enc.shape[1]  # SAE hidden dimension (49152)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input activations to SAE features.
        
        Args:
            x: Input tensor of shape [..., d_in]
            
        Returns:
            SAE features of shape [..., d_sae]
        """
        # Standard SAE encoding: ReLU(x @ W_enc + b_enc)
        pre_activation = x @ self.W_enc + self.b_enc
        features = F.relu(pre_activation)
        return features
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode SAE features back to activations.
        
        Args:
            features: SAE features of shape [..., d_sae]
            
        Returns:
            Reconstructed activations of shape [..., d_in]
        """
        # Standard SAE decoding: features @ W_dec + b_dec
        reconstructed = features @ self.W_dec + self.b_dec
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass through SAE.
        
        Args:
            x: Input tensor of shape [..., d_in]
            
        Returns:
            Tuple of (features, reconstructed) tensors
        """
        features = self.encode(x)
        reconstructed = self.decode(features)
        return features, reconstructed
    
    def get_active_features(self, features: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """Get indices of active features.
        
        Args:
            features: SAE features tensor
            threshold: Minimum activation to consider a feature active
            
        Returns:
            Boolean mask of active features
        """
        return features > threshold
    
    def feature_sparsity(self, features: torch.Tensor) -> float:
        """Calculate sparsity of features.
        
        Args:
            features: SAE features tensor
            
        Returns:
            Fraction of features that are zero
        """
        return (features == 0).float().mean().item()