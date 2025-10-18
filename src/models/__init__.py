"""Model loading and configuration utilities for text-to-SQL fine-tuning.

This module provides components for loading language models with QLoRA
quantization and LoRA adapters for efficient fine-tuning.
"""

from models.model_loader import ModelLoader
from models.config_utils import (
    create_model_config_from_hydra,
    create_bnb_config_from_hydra,
    create_lora_config_from_hydra,
    estimate_memory_requirements,
)

__all__ = [
    "ModelLoader",
    "create_model_config_from_hydra",
    "create_bnb_config_from_hydra",
    "create_lora_config_from_hydra",
    "estimate_memory_requirements",
]
