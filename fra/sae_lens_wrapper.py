"""
SAE wrappers for the FRA pipeline.

Four classes are provided:
  - SAELensAttentionSAE  : wraps a pre-trained hook_z SAE from the SAE Lens hub
                           (legacy; decoder is in concatenated-heads space)
  - LocalLn1SAE          : wraps a locally-trained ln1.hook_normalized SAE saved
                           by train_sae.py via sae.save_model(checkpoint_dir).
                           Decoder lives in d_model space — correct for FRA.
  - GemmaScopeSAE        : wraps a Gemma-Scope residual-stream SAE (google/gemma-scope-2b-pt).
                           Decoder lives in d_model=2304 space; use hook_point="hook_resid_pre".
  - QwenSAE              : wraps Qwen2.5 residual-stream SAEs (andyrdt/saes-qwen2.5-7b-instruct).
                           Decoder lives in d_model=3584 space; use hook_point="hook_resid_pre".
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Any
from sae_lens import SAE


class SAELensAttentionSAE:
    """Wrapper for SAE Lens attention SAEs with encode/decode methods."""
    
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
        
        # Extract weights for compatibility with FRA code
        self.W_dec = self.sae.W_dec  # [d_sae, d_in]
        self.W_enc = self.sae.W_enc  # [d_in, d_sae]
        self.b_enc = self.sae.b_enc  # [d_sae]
        self.b_dec = self.sae.b_dec  # [d_in]
        
        # Extract layer number from sae_id (e.g., "blocks.5.hook_z" -> 5)
        parts = sae_id.split('.')
        self.layer = int(parts[1]) if len(parts) > 1 else 0
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input activations to SAE features.
        
        Args:
            x: Input tensor of shape [..., d_in] where d_in=768 for concatenated attention heads
            
        Returns:
            SAE features of shape [..., d_sae]
        """
        original_shape = x.shape
        
        # If input is attention activations from hook_z, it might be [seq_len, n_heads, d_head]
        # We need to flatten the heads dimension
        if len(original_shape) == 3 and original_shape[-2:] == (12, 64):  # GPT-2 small has 12 heads, 64 dim each
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
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode SAE features back to activations.
        
        Args:
            features: SAE features of shape [..., d_sae]
            
        Returns:
            Reconstructed activations of shape [..., d_in]
        """
        # Use SAE's decode method directly
        original_shape = features.shape
        if len(features.shape) > 2:
            batch_shape = features.shape[:-1]
            features = features.reshape(-1, self.d_sae)
            reconstructed = self.sae.decode(features)
            reconstructed = reconstructed.reshape(*batch_shape, self.d_in)
        else:
            reconstructed = self.sae.decode(features)
        
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


class LocalLn1SAE:
    """
    Wrapper for a locally-trained TopK SAE saved by train_sae.py.

    The SAE was trained on blocks.{layer}.ln1.hook_normalized, so its decoder
    vectors live in d_model space — the same space W_Q and W_K project from.
    This makes it the correct SAE type for Feature-Resolved Attention.

    Usage:
        sae = LocalLn1SAE("./sae_checkpoints/gpt2-ln1-L5-k50-d16384", layer=5)
        # Then pass to get_sentence_fra_batch(..., hook_point="ln1.hook_normalized")
    """

    def __init__(self, checkpoint_path: str | Path, layer: int, device: str = "cuda"):
        """
        Args:
            checkpoint_path: Directory produced by sae.save_model() in train_sae.py,
                             e.g. "./sae_checkpoints/gpt2-ln1-L5-k50-d16384"
            layer: Layer the SAE was trained on (used to record .layer attribute)
            device: Device to load onto
        """
        self.checkpoint_path = str(checkpoint_path)
        self.layer = layer
        self.device = device

        # SAE Lens saves a standard checkpoint loadable with SAE.load_from_disk
        self.sae = SAE.load_from_disk(self.checkpoint_path, device=device)
        self.sae = self.sae.to(device)

        self.d_in = self.sae.cfg.d_in    # d_model = 768 for GPT-2 small
        self.d_sae = self.sae.cfg.d_sae  # e.g. 16384

        # Expose weights directly so FRA code can access them without going through .sae
        self.W_dec = self.sae.W_dec  # [d_sae, d_in]
        self.W_enc = self.sae.W_enc  # [d_in, d_sae]
        self.b_enc = self.sae.b_enc  # [d_sae]
        self.b_dec = self.sae.b_dec  # [d_in]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode ln1.hook_normalized activations to SAE features.

        Args:
            x: [seq_len, d_model] — output of blocks.{layer}.ln1.hook_normalized

        Returns:
            [seq_len, d_sae] sparse TopK feature activations
        """
        return self.sae.encode(x)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode SAE features back to d_model space."""
        return self.sae.decode(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        return features, self.decode(features)

    def feature_sparsity(self, features: torch.Tensor) -> float:
        return (features == 0).float().mean().item()


class GemmaScopeSAE:
    """
    Wrapper for Gemma-Scope residual-stream SAEs.

    These SAEs are trained on hook_resid_post, so their decoder vectors live
    in d_model=2304 space.  For FRA, use with the *next* layer's attention:
    SAE on resid_post[N] → FRA on layer N+1 (hook_resid_pre at layer N+1).

    Gemma-Scope SAEs were trained with per-token constant-norm rescaling
    (x * sqrt(d_in) / ||x||) but SAE Lens doesn't apply this at inference
    (bug in the loader).  This wrapper applies the normalization in encode()
    and reverses it in decode().

    Example usage:
        sae = GemmaScopeSAE("gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_82")
        # Use hook_point="hook_resid_pre" at layer 13 for FRA
    """

    def __init__(self, release: str, sae_id: str, device: str = "cuda",
                 normalize_activations: bool = False):
        self.release  = release
        self.sae_id   = sae_id
        self.device   = device
        self._normalize = normalize_activations
        self._norm_coeff = None  # set during encode, used by decode & FRA

        # SAE Lens returns (sae, cfg_dict, log_sparsities) for Gemma-Scope
        result = SAE.from_pretrained(release, sae_id, device=device)
        if isinstance(result, tuple):
            self.sae = result[0]
        else:
            self.sae = result

        self.d_in  = self.sae.cfg.d_in   # 2304 for Gemma-2-2B
        self.d_sae = self.sae.cfg.d_sae  # e.g. 16384

        self.W_dec = self.sae.W_dec  # [d_sae, d_model]
        self.W_enc = self.sae.W_enc  # [d_model, d_sae]
        self.b_enc = self.sae.b_enc  # [d_sae]
        self.b_dec = self.sae.b_dec  # [d_model]

        # Parse layer from sae_id e.g. "layer_12/width_16k/average_l0_82"
        try:
            self.layer = int(sae_id.split("/")[0].split("_")[1])
        except (IndexError, ValueError):
            self.layer = 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self._normalize:
            x_norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            self._norm_coeff = (self.d_in ** 0.5) / x_norms
            x = x * self._norm_coeff
        return self.sae.encode(x)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        x_hat = self.sae.decode(f)
        if self._normalize and self._norm_coeff is not None:
            x_hat = x_hat / self._norm_coeff
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        return features, self.decode(features)

    def feature_sparsity(self, features: torch.Tensor) -> float:
        return (features == 0).float().mean().item()


class QwenSAE:
    """
    Wrapper for Qwen2.5 residual-stream SAEs (andyrdt/saes-qwen2.5-7b-instruct).

    These SAEs are trained on hook_resid_post, so their decoder vectors live in
    d_model space.  For FRA, use with the *next* layer's attention:
    SAE on resid_post[N] → FRA on layer N+1 (hook_resid_pre at layer N+1).

    Example usage:
        sae = QwenSAE("qwen2.5-7b-instruct-andyrdt", "resid_post_layer_3_trainer_1")
        # Use hook_point="hook_resid_pre" at layer 4 for FRA
    """

    def __init__(self, release: str, sae_id: str, device: str = "cuda"):
        self.release = release
        self.sae_id = sae_id
        self.device = device

        result = SAE.from_pretrained(release, sae_id, device=device)
        if isinstance(result, tuple):
            self.sae = result[0]
        else:
            self.sae = result

        self.d_in = self.sae.cfg.d_in
        self.d_sae = self.sae.cfg.d_sae

        self.W_dec = self.sae.W_dec
        self.W_enc = self.sae.W_enc
        self.b_enc = self.sae.b_enc
        self.b_dec = self.sae.b_dec

        # Parse layer from sae_id e.g. "resid_post_layer_3_trainer_1" -> 3
        try:
            parts = sae_id.split("_")
            layer_idx = parts.index("layer") + 1
            self.layer = int(parts[layer_idx])
        except (ValueError, IndexError):
            self.layer = 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.sae.encode(x)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.sae.decode(f)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        return features, self.decode(features)

    def feature_sparsity(self, features: torch.Tensor) -> float:
        return (features == 0).float().mean().item()


class QwenLn1SAE:
    """
    Wrapper for Qwen2.5-14B ln1.hook_normalized SAE from HuggingFace.

    Hub: <anonymous>/Qwen2.5-14B_SAE_ln1.normalised
    Hook point: ln1.hook_normalized (decoder vectors in d_model space)
    Layer: 24, d_sae = 4 * d_model = 20480

    Supports multiple checkpoint formats:
      - SAE Lens format: cfg.json + sae_weights.safetensors
      - Raw PyTorch checkpoint: ae_*.pt (auto-detected keys)

    Usage:
        sae = QwenLn1SAE("<anonymous>/Qwen2.5-14B_SAE_ln1.normalised", layer=24)
        # Use hook_point="ln1.hook_normalized" at layer 24 for FRA
    """

    def __init__(self, repo_id: str, layer: int = 24, device: str = "cuda",
                 filename: str | None = None):
        """
        Args:
            repo_id: HuggingFace repo ID (e.g. "<anonymous>/Qwen2.5-14B_SAE_ln1.normalised")
            layer: Layer the SAE was trained on.
            device: Device to load onto.
            filename: Specific checkpoint file to download. None auto-detects.
        """
        from huggingface_hub import hf_hub_download, list_repo_files

        self.repo_id = repo_id
        self.layer = layer
        self.device = device
        self._threshold = None  # set below if TopK

        # List files in repo to determine format
        try:
            repo_files = list_repo_files(repo_id)
        except Exception:
            repo_files = []

        # Determine which file to download
        if filename is not None:
            weights_file = filename
        elif "cfg.json" in repo_files and "sae_weights.safetensors" in repo_files:
            # SAE Lens format
            weights_file = "__sae_lens__"
        else:
            # Find .pt file
            pt_files = [f for f in repo_files if f.endswith(".pt")]
            if pt_files:
                weights_file = pt_files[0]  # take first .pt file
            else:
                raise FileNotFoundError(
                    f"No loadable SAE found in {repo_id}. "
                    f"Files: {repo_files}"
                )

        if weights_file == "__sae_lens__":
            self._load_sae_lens_format(repo_id, device)
        else:
            self._load_pt_checkpoint(repo_id, weights_file, device)

        print(f"QwenLn1SAE loaded: d_in={self.d_in}, d_sae={self.d_sae}, "
              f"top_k={self._threshold}, file={weights_file}")

    def _load_sae_lens_format(self, repo_id, device):
        """Load from SAE Lens format (cfg.json + sae_weights.safetensors)."""
        import json
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        cfg_path = hf_hub_download(repo_id=repo_id, filename="cfg.json")
        weights_path = hf_hub_download(repo_id=repo_id, filename="sae_weights.safetensors")

        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        self.d_in = cfg.get("d_in", 5120)
        self.d_sae = cfg.get("d_sae", self.d_in * 4)
        self._threshold = cfg.get("k", None)

        state = load_file(weights_path, device=device)
        self._assign_weights(state, device)

    def _load_pt_checkpoint(self, repo_id, filename, device):
        """Load from a raw PyTorch .pt checkpoint."""
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
        state = torch.load(weights_path, map_location=device, weights_only=False)

        # Handle nested state dicts (e.g. {"state_dict": {...}, "cfg": {...}})
        cfg = {}
        if isinstance(state, dict) and "cfg" in state:
            cfg = state["cfg"] if isinstance(state["cfg"], dict) else {}
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        # Print keys for debugging on first load
        print(f"SAE checkpoint keys: {list(state.keys())[:20]}")

        # Auto-detect d_in and d_sae from weight shapes
        self.d_in = None
        self.d_sae = None

        # Try config dict first
        if cfg:
            self.d_in = cfg.get("d_in", cfg.get("input_dim", None))
            self.d_sae = cfg.get("d_sae", cfg.get("hidden_dim", cfg.get("dict_size", None)))
            self._threshold = cfg.get("k", cfg.get("top_k", cfg.get("activation_k", None)))

        # Also check top-level keys (some checkpoints store k/threshold as tensors)
        if self._threshold is None and "k" in state:
            v = state["k"]
            self._threshold = int(v.item()) if isinstance(v, torch.Tensor) else int(v)
        if self._threshold is None and "threshold" in state:
            v = state["threshold"]
            # threshold is sometimes a float, k is the integer count
            # skip non-integer thresholds
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                val = v.item()
                if val == int(val) and val > 1:
                    self._threshold = int(val)

        # Remove non-weight keys before assigning weights
        weight_state = {k: v for k, v in state.items()
                        if isinstance(v, torch.Tensor) and v.dim() >= 1}

        self._assign_weights(weight_state, device)

        # Infer dimensions from weights if not in config
        if self.d_in is None:
            self.d_in = self.W_dec.shape[1]
        if self.d_sae is None:
            self.d_sae = self.W_dec.shape[0]

    def _assign_weights(self, state, device):
        """Assign W_enc, W_dec, b_enc, b_dec from a state dict, handling
        various key naming conventions."""
        # Try common key names for each weight
        enc_keys = ["W_enc", "encoder.weight", "encode.weight", "w_enc", "W_e"]
        dec_keys = ["W_dec", "decoder.weight", "decode.weight", "w_dec", "W_d"]
        benc_keys = ["b_enc", "encoder.bias", "encode.bias", "b_e"]
        bdec_keys = ["b_dec", "decoder.bias", "decode.bias", "b_d", "bias"]

        def _find(keys):
            for k in keys:
                if k in state:
                    return state[k].to(device).float()
            return None

        self.W_enc = _find(enc_keys)
        self.W_dec = _find(dec_keys)
        self.b_enc = _find(benc_keys)
        self.b_dec = _find(bdec_keys)

        if self.W_enc is None or self.W_dec is None:
            available = [k for k in state.keys() if isinstance(state[k], torch.Tensor)]
            raise KeyError(
                f"Could not find encoder/decoder weights. "
                f"Available tensor keys: {available}"
            )

        # Ensure correct shapes: W_enc [d_in, d_sae], W_dec [d_sae, d_in]
        # W_enc and W_dec should be transposes of each other (roughly)
        if self.W_enc.shape == self.W_dec.shape:
            # Both same shape — W_enc is likely [d_sae, d_in], needs transpose
            if self.W_enc.shape[0] > self.W_enc.shape[1]:
                self.W_enc = self.W_enc.T
        elif self.W_enc.shape[0] > self.W_enc.shape[1]:
            # W_enc is [d_sae, d_in], needs transpose to [d_in, d_sae]
            self.W_enc = self.W_enc.T
        if self.W_dec.shape[1] > self.W_dec.shape[0]:
            # W_dec is [d_in, d_sae], needs transpose to [d_sae, d_in]
            self.W_dec = self.W_dec.T

        d_in = self.W_enc.shape[0]
        d_sae = self.W_enc.shape[1]

        if self.b_dec is None:
            self.b_dec = torch.zeros(d_in, device=device)
        if self.b_enc is None:
            self.b_enc = torch.zeros(d_sae, device=device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ln1.hook_normalized activations to SAE features.

        Args:
            x: [seq_len, d_model] — output of blocks.{layer}.ln1.hook_normalized

        Returns:
            [seq_len, d_sae] feature activations (ReLU or TopK gated)
        """
        pre_acts = x.float() @ self.W_enc + self.b_enc  # [seq_len, d_sae]
        if self._threshold is not None:
            k = self._threshold
            topk_vals, topk_idx = pre_acts.topk(k, dim=-1)
            acts = torch.zeros_like(pre_acts)
            acts.scatter_(-1, topk_idx, F.relu(topk_vals))
            return acts
        else:
            return F.relu(pre_acts)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode SAE features back to d_model space."""
        return features.float() @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        return features, self.decode(features)

    def feature_sparsity(self, features: torch.Tensor) -> float:
        return (features == 0).float().mean().item()


def get_attention_activations_for_sae_lens(
    model: Any,
    input_text: str,
    layer: int,
    max_length: int = 128
) -> torch.Tensor:
    """
    Get attention activations in the format expected by SAE Lens SAEs.
    
    SAE Lens attention SAEs are trained on hook_z (the concatenated attention values),
    not hook_attn_out (the output after projection).
    
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