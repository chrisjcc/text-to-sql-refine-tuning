"""
Main training script for GRPO fine-tuning.
"""

from pathlib import Path

import hydra
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf

from src.data.grpo_formatter import GRPODatasetFormatter
from src.environments.sql_env.environment import TextToSQLEnvironment
from src.models.config_utils import (
    create_bnb_config_from_hydra,
    create_lora_config_from_hydra,
)
from src.models.model_loader import ModelLoader
from src.rubrics.sql_rubric import SQLValidationRubric
from src.training.callbacks import SQLEvaluationCallback
from src.training.config_builder import GRPOConfigBuilder
from src.training.grpo_trainer import SQLGRPOTrainer
from src.utils.logging_utils import setup_logging_from_config
from src.utils.sql_parser import SQLParser

try:
    import wandb  # type: ignore

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore


def _setup_wandb(cfg: DictConfig, logger):
    """Setup WandB logging if enabled."""
    if cfg.wandb.enabled:
        if not WANDB_AVAILABLE:
            logger.warning(
                "WandB is enabled in config but not installed. "
                "Disabling WandB logging."
            )
        else:
            wandb.init(
                project=cfg.wandb.project,
                name=cfg.training.run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            logger.info(f"WandB initialized: {wandb.run.name}")


def _load_and_log_dataset(cfg: DictConfig, logger):
    """Load processed dataset and log statistics."""
    logger.info("\n" + "=" * 80)
    logger.info("Loading processed dataset")
    dataset_path = Path(cfg.dataset.cache_dir) / "processed"
    dataset = load_from_disk(str(dataset_path))

    logger.info(f"Train samples: {len(dataset['train'])}")
    if "validation" in dataset:
        logger.info(f"Validation samples: {len(dataset['validation'])}")

    return dataset


def _load_model_and_tokenizer(cfg: DictConfig, logger):
    """Load model and tokenizer with configurations."""
    logger.info("\n" + "=" * 80)
    logger.info("Loading model and tokenizer")

    loader = ModelLoader(model_name=cfg.hf.model.name, cache_dir=cfg.hf.model.cache_dir)

    bnb_config = create_bnb_config_from_hydra(cfg)
    lora_config = create_lora_config_from_hydra(cfg)

    model, tokenizer = loader.load_model_and_tokenizer(
        use_quantization=cfg.training.peft.use_qlora,
        use_peft=cfg.training.use_peft,
        bnb_config=bnb_config,
        lora_config=lora_config,
    )

    loader.print_model_info(model)
    return model, tokenizer


def _setup_environment_and_rubric(cfg: DictConfig, dataset, logger):
    """Setup SQL environment and validation rubric."""
    logger.info("\n" + "=" * 80)
    logger.info("Setting up environment and rubric")

    parser = SQLParser()
    rubric = SQLValidationRubric(
        sql_keywords=cfg.evaluation.rubric.sql_keywords,
        syntax_weight=cfg.evaluation.rubric.weights.syntax,
        keyword_weight=cfg.evaluation.rubric.weights.keywords,
        format_weight=cfg.evaluation.rubric.weights.format,
        parser=parser,
    )

    environment = TextToSQLEnvironment(
        rubric=rubric,
        parser=parser,
        prompt_template=cfg.training.environment.prompt_template,
        include_schema=cfg.training.environment.include_schema,
        max_examples=cfg.training.environment.few_shot_examples,
        dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
    )

    return environment, rubric


def _format_datasets(environment, tokenizer, dataset, logger):
    """Format datasets for GRPO training."""
    logger.info("\n" + "=" * 80)
    logger.info("Formatting dataset for GRPO")

    formatter = GRPODatasetFormatter(
        environment=environment, tokenizer=tokenizer, include_reference=True
    )

    train_dataset = formatter.format_dataset(dataset["train"])
    eval_dataset = None
    eval_dataset_small = None

    if "validation" in dataset:
        eval_dataset = formatter.format_dataset(dataset["validation"])
        eval_dataset_small = formatter.create_evaluation_set(
            dataset["validation"],
        n_samples=100
    )

    logger.info(f"Formatted train dataset: {len(train_dataset)} samples")
    if eval_dataset:
        logger.info(f"Formatted eval dataset: {len(eval_dataset)} samples")

    return train_dataset, eval_dataset, eval_dataset_small


def _finalize_training(trainer, eval_dataset, cfg: DictConfig, logger):
    """Run final evaluation, save model, and cleanup."""
    if eval_dataset:
        logger.info("\n" + "=" * 80)
        logger.info("Running final evaluation")
        metrics = trainer.evaluate(eval_dataset)
        logger.info("Final metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

    logger.info("\n" + "=" * 80)
    logger.info("Saving model")
    output_dir = Path(cfg.training.output_dir) / "final_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))

    if cfg.hf.hub.push_to_hub and cfg.hf.hub.repo_name:
        logger.info(f"Pushing to HuggingFace Hub: {cfg.hf.hub.repo_name}")
        trainer.push_to_hub(repo_id=cfg.hf.hub.repo_name, private=cfg.hf.hub.private)

    if cfg.wandb.enabled and WANDB_AVAILABLE:
        wandb.finish()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def train(cfg: DictConfig):
    """
    Main training function.

    Steps:
    1. Setup logging and wandb
    2. Load processed dataset
    3. Load model and tokenizer
    4. Setup environment and rubric
    5. Format dataset for GRPO
    6. Initialize trainer
    7. Train model
    8. Save and evaluate
    """
    logger = setup_logging_from_config(cfg)
    logger.info("=" * 80)
    logger.info("Starting GRPO Training for Text-to-SQL")
    logger.info("=" * 80)

    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    _setup_wandb(cfg, logger)
    dataset = _load_and_log_dataset(cfg, logger)
    model, tokenizer = _load_model_and_tokenizer(cfg, logger)
    environment, rubric = _setup_environment_and_rubric(cfg, dataset, logger)
    train_dataset, eval_dataset, eval_dataset_small = _format_datasets(
        environment, tokenizer, dataset, logger
    )

    logger.info("\n" + "=" * 80)
    logger.info("Building GRPO configuration")
    grpo_config = GRPOConfigBuilder.build_from_hydra(cfg)

    logger.info("\n" + "=" * 80)
    logger.info("Initializing GRPO trainer")

    trainer = SQLGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        environment=environment,
        rubric=rubric,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_small if eval_dataset else None,
        config=grpo_config,
    )

    if eval_dataset:
        eval_callback = SQLEvaluationCallback(
            environment=environment,
            rubric=rubric,
            eval_dataset=eval_dataset_small,
            tokenizer=tokenizer,
            eval_frequency=cfg.training.eval_steps,
            num_samples=20,
            log_examples=True,
        )
        trainer.trainer.add_callback(eval_callback)

    logger.info("\n" + "=" * 80)
    logger.info("Starting training")
    logger.info("=" * 80 + "\n")

    trainer.train()

    _finalize_training(trainer, eval_dataset, cfg, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    train()
