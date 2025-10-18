"""Configuration utilities for model loading.

This module provides utility functions for creating model loading configurations
from Hydra configs and estimating memory requirements.
"""

from omegaconf import DictConfig
from transformers import BitsAndBytesConfig
from peft import LoraConfig
import torch
from typing import Dict, Optional


def create_model_config_from_hydra(cfg: DictConfig) -> Dict:
    """
    Create model loading configuration from Hydra config.

    Args:
        cfg: Hydra configuration object

    Returns:
        Dict with model loading parameters
    """
    model_config = {
        "model_name": cfg.hf.model.name,
        "cache_dir": cfg.hf.model.cache_dir,
        "use_quantization": cfg.training.use_peft,
        "use_peft": cfg.training.use_peft,
    }

    # Add attention implementation if specified in config
    if hasattr(cfg.hf.model, 'attn_implementation'):
        model_config["attn_implementation"] = cfg.hf.model.attn_implementation

    return model_config


def create_bnb_config_from_hydra(cfg: DictConfig) -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytes config from Hydra configuration."""
    if not cfg.training.peft.use_qlora:
        return None

    compute_dtype = getattr(torch, cfg.training.peft.bnb_4bit_compute_dtype)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )


def create_lora_config_from_hydra(cfg: DictConfig) -> LoraConfig:
    """Create LoRA config from Hydra configuration."""
    return LoraConfig(
        r=cfg.training.peft.lora_r,
        lora_alpha=cfg.training.peft.lora_alpha,
        lora_dropout=cfg.training.peft.lora_dropout,
        target_modules=cfg.training.peft.target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )


def estimate_memory_requirements(
    model_name: str,
    use_quantization: bool = True,
    use_peft: bool = True,
    batch_size: int = 1,
    sequence_length: int = 512
) -> Dict[str, float]:
    """
    Estimate GPU memory requirements.

    Args:
        model_name: HuggingFace model identifier
        use_quantization: Whether QLoRA quantization is used
        use_peft: Whether LoRA adapters are used
        batch_size: Training batch size
        sequence_length: Maximum sequence length

    Returns:
        Dict with memory estimates in GB
    """
    # Rough estimates for Llama-3.1-8B
    base_model_size = 8  # 8B parameters

    if use_quantization:
        # 4-bit quantization: ~0.5 bytes per parameter
        model_memory = base_model_size * 0.5
    else:
        # bfloat16: 2 bytes per parameter
        model_memory = base_model_size * 2

    # LoRA adapters (small additional memory)
    if use_peft:
        lora_memory = 0.1  # ~100MB for typical LoRA config
    else:
        lora_memory = 0

    # Activation memory (depends on batch size and sequence length)
    activation_memory = batch_size * sequence_length * 0.001

    # Optimizer states (if training full model)
    if use_peft:
        optimizer_memory = lora_memory * 2  # AdamW needs 2x for momentum
    else:
        optimizer_memory = model_memory * 2

    # Gradient memory
    gradient_memory = lora_memory if use_peft else model_memory

    total_memory = (
        model_memory +
        lora_memory +
        activation_memory +
        optimizer_memory +
        gradient_memory
    )

    return {
        "model": model_memory,
        "lora": lora_memory,
        "activations": activation_memory,
        "optimizer": optimizer_memory,
        "gradients": gradient_memory,
        "total": total_memory,
        "recommended_gpu": total_memory * 1.2  # 20% buffer
    }
