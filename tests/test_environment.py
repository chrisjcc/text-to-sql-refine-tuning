"""Test model loading with various configurations.

This script tests the ModelLoader class with different configurations
to ensure models load correctly with QLoRA and LoRA adapters.

Usage:
    # Run with pytest:
    pytest tests/test_model.py -v

    # Run standalone:
    python tests/test_model.py

    # Run with config overrides:
    python tests/test_model.py hf.model.name=meta-llama/Llama-3-8B-Instruct
    python tests/test_model.py training.peft.use_qlora=false
"""

import sys

import pytest
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig

# Load environment variables before anything else
load_dotenv()

from config.config import load_config
from src.models.config_utils import (
    create_bnb_config_from_hydra,
    create_lora_config_from_hydra,
    estimate_memory_requirements,
)
from src.models.model_loader import ModelLoader
from src.utils.logging_utils import setup_logging_from_config


@pytest.fixture(scope="module")
def cfg() -> DictConfig:
    """Load Hydra configuration for testing.

    Returns:
        DictConfig: Loaded configuration
    """
    return load_config()


def test_model(cfg: DictConfig):
    """Test model loading with current configuration.

    Args:
        cfg: Hydra configuration from fixture
    """
    logger = setup_logging_from_config(cfg)

    # Print memory estimates
    logger.info("Estimating memory requirements...")
    memory_est = estimate_memory_requirements(
        model_name=cfg.hf.model.name,
        use_quantization=cfg.training.peft.use_qlora,
        use_peft=cfg.training.use_peft,
        batch_size=cfg.training.per_device_train_batch_size,
        sequence_length=cfg.dataset.preprocessing.max_length,
    )

    logger.info("\nMemory Estimates:")
    for key, value in memory_est.items():
        logger.info(f"  {key}: {value:.2f} GB")

    # Load model
    logger.info(f"\nLoading model: {cfg.hf.model.name}")

    loader = ModelLoader(model_name=cfg.hf.model.name, cache_dir=cfg.hf.model.cache_dir)

    bnb_config = create_bnb_config_from_hydra(cfg)
    lora_config = create_lora_config_from_hydra(cfg)

    # Get attention implementation from config if available
    attn_impl = cfg.hf.model.get("attn_implementation", "auto")

    model, tokenizer = loader.load_model_and_tokenizer(
        use_quantization=cfg.training.peft.use_qlora,
        use_peft=cfg.training.use_peft,
        bnb_config=bnb_config,
        lora_config=lora_config,
        attn_implementation=attn_impl,
    )

    # Print model info
    loader.print_model_info(model)

    # Test inference
    logger.info("Testing inference...")
    test_prompt = "SELECT * FROM users WHERE"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info("\nTest generation:")
    logger.info(f"Prompt: {test_prompt}")
    logger.info(f"Output: {generated_text}")

    logger.info("\nModel loading test complete!")


def main():
    """Entry point for standalone execution with optional config overrides.

    Supports Hydra-style command line overrides:
        python tests/test_model.py hf.model.name=meta-llama/Llama-3-8B-Instruct
        python tests/test_model.py training.peft.use_qlora=false
    """
    # Parse command line arguments for config overrides
    overrides = []
    if len(sys.argv) > 1:
        # Collect override arguments (format: key=value)
        overrides = [arg for arg in sys.argv[1:] if "=" in arg]
        if overrides:
            print(f"Applying config overrides: {overrides}")

    # Load config with overrides
    cfg = load_config(overrides=overrides if overrides else None)

    # Run test
    test_model(cfg)


if __name__ == "__main__":
    main()
