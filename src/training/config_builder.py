"""Training Configuration Builder for GRPO.

Builds GRPOConfig from Hydra configuration files.
"""

from trl import GRPOConfig
from omegaconf import DictConfig
from typing import Optional
import logging


logger = logging.getLogger(__name__)


class GRPOConfigBuilder:
    """
    Builds GRPOConfig from Hydra configuration.
    """

    @staticmethod
    def build_from_hydra(cfg: DictConfig) -> GRPOConfig:
        """
        Create GRPOConfig from Hydra configuration.

        Args:
            cfg: Hydra configuration object

        Returns:
            GRPOConfig for training
        """
        logger.info("Building GRPO configuration")

        grpo_config = GRPOConfig(
            # Output and logging
            output_dir=cfg.training.output_dir,
            run_name=cfg.training.run_name,
            logging_dir=f"{cfg.training.output_dir}/logs",
            logging_steps=cfg.training.logging_steps,

            # Training schedule
            num_train_epochs=cfg.training.num_train_epochs,
            max_steps=cfg.training.max_steps,
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,

            # Learning rate and optimization
            learning_rate=cfg.training.learning_rate,
            lr_scheduler_type=cfg.training.lr_scheduler_type,
            warmup_steps=cfg.training.warmup_steps,
            warmup_ratio=cfg.training.warmup_ratio,
            optim=cfg.training.optim,

            # Precision and performance
            bf16=cfg.training.bf16,
            fp16=False,  # Use bf16 instead
            gradient_checkpointing=cfg.training.gradient_checkpointing,

            # GRPO-specific parameters
            num_generations=cfg.training.num_generations,

            # Checkpointing
            save_strategy=cfg.training.save_strategy,
            save_steps=cfg.training.save_steps,
            save_total_limit=cfg.training.save_total_limit,

            # Evaluation
            eval_strategy=cfg.training.evaluation_strategy,
            eval_steps=cfg.training.eval_steps,

            # Miscellaneous
            seed=cfg.project.seed,
            dataloader_num_workers=cfg.dataset.num_workers,
            remove_unused_columns=False,
            report_to=["wandb"] if cfg.wandb.enabled else [],
        )

        logger.info("GRPO configuration built successfully")
        return grpo_config
