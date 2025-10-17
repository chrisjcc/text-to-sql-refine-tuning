"""Configuration loader using Hydra.

This module provides utilities for loading and managing configuration
files using Hydra's compositional config system.
"""

import os
from pathlib import Path
from typing import Optional

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv


def load_config(
    config_name: str = "config",
    config_path: Optional[str] = None,
    overrides: Optional[list] = None,
) -> DictConfig:
    """Load Hydra configuration with environment variables.

    Args:
        config_name: Name of the config file (without .yaml extension)
        config_path: Path to config directory (defaults to this file's directory)
        overrides: List of config overrides in format ["key=value", ...]

    Returns:
        DictConfig: Loaded and merged configuration

    Example:
        >>> cfg = load_config()
        >>> print(cfg.project.name)
        text-to-sql-fine-tuning

        >>> cfg = load_config(overrides=["training.batch_size=16"])
        >>> print(cfg.training.per_device_train_batch_size)
        16
    """
    # Load environment variables from .env file
    load_dotenv()

    # Set config path to this directory if not provided
    if config_path is None:
        config_path = str(Path(__file__).parent.absolute())

    # Ensure overrides is a list
    if overrides is None:
        overrides = []

    try:
        # Initialize Hydra with the config directory
        with initialize_config_dir(config_dir=config_path, version_base=None):
            # Compose configuration with overrides
            cfg = compose(config_name=config_name, overrides=overrides)

        return cfg

    except Exception as e:
        raise RuntimeError(
            f"Failed to load configuration from {config_path}/{config_name}.yaml: {e}"
        ) from e


def save_config(cfg: DictConfig, output_path: str) -> None:
    """Save configuration to a YAML file.

    Args:
        cfg: Configuration to save
        output_path: Path where to save the config file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        OmegaConf.save(cfg, f)


def print_config(cfg: DictConfig, resolve: bool = True) -> None:
    """Pretty print configuration.

    Args:
        cfg: Configuration to print
        resolve: Whether to resolve interpolations
    """
    print(OmegaConf.to_yaml(cfg, resolve=resolve))


def main():
    """Test configuration loading."""
    print("=" * 80)
    print("Testing Hydra Configuration Loading")
    print("=" * 80)
    print()

    try:
        # Load default configuration
        print("Loading configuration...")
        cfg = load_config()
        print("✓ Configuration loaded successfully!")
        print()

        # Print configuration
        print("Configuration contents:")
        print("-" * 80)
        print_config(cfg)
        print("-" * 80)
        print()

        # Test accessing specific config values
        print("Testing configuration access:")
        print(f"  Project name: {cfg.project.name}")
        print(f"  Project version: {cfg.project.version}")
        print(f"  Model name: {cfg.hf.model.name}")
        print(f"  Training output dir: {cfg.training.output_dir}")
        print(f"  Dataset name: {cfg.dataset.name}")
        print()

        print("✓ All tests passed!")

    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        raise


if __name__ == "__main__":
    main()
