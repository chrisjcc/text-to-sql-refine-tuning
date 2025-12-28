"""GRPO Trainer Wrapper for Text-to-SQL fine-tuning.

This module wraps TRL's GRPOTrainer for text-to-SQL applications,
integrating environment, rubric, and evaluation components.
"""

import logging
from typing import Any

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from src.environments.sql_env.environment import TextToSQLEnvironment
from src.rubrics.sql_rubric import SQLValidationRubric

logger = logging.getLogger(__name__)


class SQLGRPOTrainer:
    """Wrapper around TRL's GRPOTrainer for text-to-SQL fine-tuning.

    Handles environment integration, reward computation, and evaluation
    for training text-to-SQL models using GRPO (Group Relative Policy
    Optimization).

    Attributes:
        model: Pre-trained model with PEFT adapters.
        tokenizer: Tokenizer for the model.
        environment: Text-to-SQL environment for prompting.
        rubric: SQL validation rubric for computing rewards.
        train_dataset: Training dataset.
        eval_dataset: Optional evaluation dataset.
        logger: Logger instance for this class.
        config: GRPO training configuration.
        trainer: Underlying TRL GRPOTrainer instance.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        environment: TextToSQLEnvironment,
        rubric: SQLValidationRubric,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        config: GRPOConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize GRPO trainer.

        Args:
            model: Pre-trained model with PEFT adapters for fine-tuning.
            tokenizer: Tokenizer compatible with the model.
            environment: Text-to-SQL environment for prompt formatting and
                response processing.
            rubric: SQL validation rubric for computing reward scores.
            train_dataset: Training dataset with prompts and references.
            eval_dataset: Optional evaluation dataset. Defaults to None.
            config: GRPO training configuration. If None, uses default config.
            **kwargs: Additional arguments passed to GRPOTrainer.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.environment = environment
        self.rubric = rubric
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.logger = logging.getLogger(__name__)

        # Create GRPO config if not provided
        if config is None:
            config = self.create_default_config()

        self.config = config

        # Initialize TRL's GRPOTrainer
        # Note: tokenizer is stored in the class but not passed to GRPOTrainer
        # as it's handled internally by TRL's trainer
        # Note: reward_funcs expects a list of reward functions
        self.trainer = GRPOTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            reward_funcs=[self.compute_rewards],
            **kwargs,
        )

        self.logger.info("SQLGRPOTrainer initialized")

    def create_default_config(self) -> GRPOConfig:
        """Create default GRPO configuration.

        Returns:
            GRPOConfig with sensible defaults for text-to-SQL training.
        """
        return GRPOConfig(
            output_dir="./outputs",
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=5e-6,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=3,
            bf16=True,
            gradient_checkpointing=True,
            # GRPO-specific
            num_generations=4,
        )

    def compute_rewards(
        self,
        prompts: list[str],  # noqa: ARG002
        responses: list[str],
        **kwargs: Any,  # noqa: ARG002
    ) -> list[float]:
        """Compute rewards for generated responses.

        This function is called by GRPOTrainer during training to evaluate
        the quality of generated SQL queries.

        Args:
            prompts: List of input prompts. Currently unused but required
                by GRPOTrainer interface.
            responses: List of model-generated SQL responses to score.
            **kwargs: Additional context from trainer. Currently unused.

        Returns:
            List of reward scores in range [0.0, 1.0] corresponding to
            each response.
        """
        # Use rubric to score SQL outputs
        rewards = self.rubric.score_batch(responses)

        # Log reward statistics
        if len(rewards) > 0:
            avg_reward = sum(rewards) / len(rewards)
            min_reward = min(rewards)
            max_reward = max(rewards)

            self.logger.debug(
                f"Reward stats - Avg: {avg_reward:.3f}, "
                f"Min: {min_reward:.3f}, Max: {max_reward:.3f}"
            )

        return rewards  # type: ignore[no-any-return]

    def train(self) -> None:
        """Run GRPO training.

        Executes the full training loop using the underlying GRPOTrainer.

        Returns:
            None. Trains model in-place and logs progress.
        """
        self.logger.info("Starting GRPO training")
        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            self.logger.info(f"Evaluation samples: {len(self.eval_dataset)}")

        # Train
        self.trainer.train()

        self.logger.info("Training complete")

    def evaluate(self, dataset: Dataset | None = None) -> dict[str, float]:
        """Run evaluation on dataset.

        Args:
            dataset: Dataset to evaluate. If None, uses the evaluation
                dataset provided during initialization. Defaults to None.

        Returns:
            Dictionary of evaluation metrics from the trainer.
        """
        if dataset is None:
            dataset = self.eval_dataset

        if dataset is None:
            self.logger.warning("No evaluation dataset provided")
            return {}

        self.logger.info(f"Evaluating on {len(dataset)} samples")

        # Run evaluation
        return self.trainer.evaluate(eval_dataset=dataset)  # type: ignore[no-any-return]

    def save_model(self, output_dir: str) -> None:
        """Save trained model and adapters.

        Args:
            output_dir: Directory path where model and tokenizer will
                be saved.

        Returns:
            None. Saves model files to disk.
        """
        self.logger.info(f"Saving model to {output_dir}")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def push_to_hub(self, repo_id: str, **kwargs: Any) -> None:
        """Push model to HuggingFace Hub.

        Args:
            repo_id: Repository ID on HuggingFace Hub in format
                "username/model-name".
            **kwargs: Additional arguments passed to push_to_hub method.

        Returns:
            None. Uploads model to HuggingFace Hub.
        """
        self.logger.info(f"Pushing model to Hub: {repo_id}")
        self.trainer.push_to_hub(repo_id=repo_id, **kwargs)
