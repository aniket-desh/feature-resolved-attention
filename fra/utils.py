from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml
from pydantic import BaseModel
from torch import nn
from transformer_lens import HookedTransformer
from datasets import load_dataset

from fra.log import logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config


def load_model(model_name: str = "gpt2-small", device: str = "cuda") -> HookedTransformer:
    """Load GPT-2 model using TransformerLens.
    
    Args:
        model_name: Name of the model to load (default: gpt2-small)
        device: Device to load model on (cuda or cpu)
        
    Returns:
        HookedTransformer model
    """
    logger.info(f"Loading model: {model_name} on {device}")
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=False,
        refactor_factored_attn_matrices=False,
    )
    logger.info(f"Model loaded successfully: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")
    return model


def load_sae(repo: str, layer: int, device: str = "cuda") -> Dict[str, Any]:
    """Load Sparse Autoencoder from HuggingFace.
    
    Note: These attention SAEs from ckkissane are not in standard SAE Lens format,
    so we load them manually and return the weights and config.
    
    Args:
        repo: HuggingFace repository containing the SAEs
        layer: Which layer's SAE to load (0-indexed)
        device: Device to load SAE on
        
    Returns:
        Dictionary containing SAE weights and config
    """
    logger.info(f"Loading SAE from {repo} for layer {layer}")
    
    from huggingface_hub import hf_hub_download
    import json
    
    # These SAEs have a specific naming pattern
    # We'll need to list files and find the right one for the layer
    from huggingface_hub import list_repo_files
    files = list_repo_files(repo)
    
    # Find the file for the specified layer
    layer_files = [f for f in files if f"L{layer}_" in f]
    pt_files = [f for f in layer_files if f.endswith('.pt')]
    cfg_files = [f for f in layer_files if f.endswith('_cfg.json')]
    
    if not pt_files or not cfg_files:
        raise ValueError(f"Could not find SAE files for layer {layer}")
    
    # Download the files
    pt_file = hf_hub_download(repo_id=repo, filename=pt_files[0])
    cfg_file = hf_hub_download(repo_id=repo, filename=cfg_files[0])
    
    # Load config
    with open(cfg_file, 'r') as f:
        config = json.load(f)
    
    # Load weights
    state_dict = torch.load(pt_file, map_location=device)
    
    logger.info(f"SAE loaded for layer {layer}: dict_mult={config.get('dict_mult', 'unknown')}")
    
    return {
        'state_dict': state_dict,
        'config': config,
        'layer': layer,
        'device': device
    }


def load_dataset_hf(
    dataset_name: str = "Elriggs/openwebtext-100k",
    split: str = "train",
    streaming: bool = True,
    seed: int = 42
) -> Any:
    """Load a dataset from HuggingFace.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        split: Dataset split to use
        streaming: Whether to use streaming mode for memory efficiency
        seed: Random seed for shuffling
        
    Returns:
        Dataset object (streaming or regular)
    """
    logger.info(f"Loading dataset: {dataset_name} (streaming={streaming})")
    
    try:
        # Load the dataset - streaming is optional
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)
        
        # If streaming, shuffle with buffer; if not, shuffle normally
        if streaming:
            dataset = dataset.shuffle(seed=seed, buffer_size=10000)
        else:
            dataset = dataset.shuffle(seed=seed)
        
        logger.info(f"Successfully loaded dataset: {dataset_name}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise



def save_model_and_config(config: BaseModel, save_dir: Path, model: nn.Module, epoch: int) -> None:
    """Save the model to disk. Also save the config file if it doesn't exist.

    Args:
        config: The config object. Saved if save_dir / "config.yaml" doesn't already exist.
        save_dir: The directory to save the model and config to.
        model: The model to save.
        epoch: The current epoch (used in the model filename).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    if not (save_dir / "config.yaml").exists():
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        logger.info("Saved config to %s", save_dir / "config.yaml")

    model_file = save_dir / f"model_epoch_{epoch + 1}.pt"
    torch.save(model.state_dict(), model_file)
    logger.info("Saved model to %s", model_file)
