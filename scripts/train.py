"""
Main training script for GRPO fine-tuning.
"""
from pathlib import Path

import hydra
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf

from data.grpo_formatter import GRPODatasetFormatter
from environments.sql_env.environment import TextToSQLEnvironment
from models.config_utils import create_bnb_config_from_hydra, create_lora_config_from_hydra
from models.model_loader import ModelLoader
from rubrics.sql_rubric import SQLValidationRubric
from training.callbacks import SQLEvaluationCallback
from training.config_builder import GRPOConfigBuilder
from training.grpo_trainer import SQLGRPOTrainer
from utils.logging_utils import setup_logging_from_config
from utils.sql_parser import SQLParser

try:
    import wandb  # type: ignore

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore


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

    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Setup WandB
    if cfg.wandb.enabled:
        if not WANDB_AVAILABLE:
            logger.warning("WandB is enabled in config but not installed. Disabling WandB logging.")
        else:
            wandb.init(
                project=cfg.wandb.project,
                name=cfg.training.run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            logger.info(f"WandB initialized: {wandb.run.name}")

    # Load processed dataset
    logger.info("\n" + "=" * 80)
    logger.info("Loading processed dataset")
    dataset_path = Path(cfg.dataset.cache_dir) / "processed"
    dataset = load_from_disk(str(dataset_path))

    logger.info(f"Train samples: {len(dataset['train'])}")
    if "validation" in dataset:
        logger.info(f"Validation samples: {len(dataset['validation'])}")

    # Load model and tokenizer
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

    # Setup environment and rubric
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

    # Format dataset for GRPO
    logger.info("\n" + "=" * 80)
    logger.info("Formatting dataset for GRPO")

    formatter = GRPODatasetFormatter(
        environment=environment, tokenizer=tokenizer, include_reference=True
    )

    train_dataset = formatter.format_dataset(dataset["train"])
    eval_dataset = None
    if "validation" in dataset:
        eval_dataset = formatter.format_dataset(dataset["validation"])
        # Create small eval set for frequent evaluation
        eval_dataset_small = formatter.create_evaluation_set(dataset["validation"], n_samples=100)

    logger.info(f"Formatted train dataset: {len(train_dataset)} samples")
    if eval_dataset:
        logger.info(f"Formatted eval dataset: {len(eval_dataset)} samples")

    # Build GRPO config
    logger.info("\n" + "=" * 80)
    logger.info("Building GRPO configuration")
    grpo_config = GRPOConfigBuilder.build_from_hydra(cfg)

    # Initialize trainer
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

    # Add callbacks
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

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting training")
    logger.info("=" * 80 + "\n")

    trainer.train()

    # Final evaluation
    if eval_dataset:
        logger.info("\n" + "=" * 80)
        logger.info("Running final evaluation")
        metrics = trainer.evaluate(eval_dataset)
        logger.info("Final metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

    # Save model
    logger.info("\n" + "=" * 80)
    logger.info("Saving model")
    output_dir = Path(cfg.training.output_dir) / "final_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))

    # Push to hub if configured
    if cfg.hf.hub.push_to_hub and cfg.hf.hub.repo_name:
        logger.info(f"Pushing to HuggingFace Hub: {cfg.hf.hub.repo_name}")
        trainer.push_to_hub(repo_id=cfg.hf.hub.repo_name, private=cfg.hf.hub.private)

    # Finish wandb
    if cfg.wandb.enabled and WANDB_AVAILABLE:
        wandb.finish()

    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    train()
