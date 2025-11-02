"""Test model loading with various configurations.

This script tests the ModelLoader class with different configurations
to ensure models load correctly with QLoRA and LoRA adapters.
"""

import sys
from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig

# Load environment variables before anything else
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config_utils import (  # noqa: E402
    create_bnb_config_from_hydra,
    create_lora_config_from_hydra,
    estimate_memory_requirements,
)
from models.model_loader import ModelLoader  # noqa: E402
from utils.logging_utils import setup_logging_from_config  # noqa: E402


@hydra.main(version_base=None, config_path="../config", config_name="config")
def test_model(cfg: DictConfig):
    """Test model loading with current configuration."""
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


if __name__ == "__main__":
    test_model()
