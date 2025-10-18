"""GRPO Trainer Wrapper for Text-to-SQL fine-tuning.

This module wraps TRL's GRPOTrainer for text-to-SQL applications,
integrating environment, rubric, and evaluation components.
"""

from trl import GRPOConfig, GRPOTrainer
from typing import Dict, List, Optional, Any
import logging
from datasets import Dataset

from ..environments.sql_env.environment import TextToSQLEnvironment
from ..rubrics.sql_rubric import SQLValidationRubric


logger = logging.getLogger(__name__)


class SQLGRPOTrainer:
    """
    Wrapper around TRL's GRPOTrainer for text-to-SQL fine-tuning.
    Handles environment integration, reward computation, and evaluation.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        environment: TextToSQLEnvironment,
        rubric: SQLValidationRubric,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[GRPOConfig] = None,
        **kwargs
    ):
        """
        Initialize GRPO trainer.

        Args:
            model: Pre-trained model with PEFT adapters
            tokenizer: Tokenizer
            environment: Text-to-SQL environment for prompting
            rubric: SQL validation rubric for rewards
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            config: GRPO training configuration
            **kwargs: Additional arguments for GRPOTrainer
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
            **kwargs
        )

        self.logger.info("SQLGRPOTrainer initialized")

    def create_default_config(self) -> GRPOConfig:
        """Create default GRPO configuration."""
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
        prompts: List[str],
        responses: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute rewards for generated responses.

        This function is called by GRPOTrainer during training.

        Args:
            prompts: List of input prompts
            responses: List of model-generated responses
            **kwargs: Additional context

        Returns:
            List of reward scores [0.0, 1.0]
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

        return rewards

    def train(self):
        """Run GRPO training."""
        self.logger.info("Starting GRPO training")
        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            self.logger.info(f"Evaluation samples: {len(self.eval_dataset)}")

        # Train
        self.trainer.train()

        self.logger.info("Training complete")

    def evaluate(self, dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation on dataset.

        Args:
            dataset: Dataset to evaluate (uses eval_dataset if None)

        Returns:
            Dict of evaluation metrics
        """
        if dataset is None:
            dataset = self.eval_dataset

        if dataset is None:
            self.logger.warning("No evaluation dataset provided")
            return {}

        self.logger.info(f"Evaluating on {len(dataset)} samples")

        # Run evaluation
        metrics = self.trainer.evaluate(eval_dataset=dataset)

        return metrics

    def save_model(self, output_dir: str):
        """Save trained model and adapters."""
        self.logger.info(f"Saving model to {output_dir}")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def push_to_hub(self, repo_id: str, **kwargs):
        """Push model to HuggingFace Hub."""
        self.logger.info(f"Pushing model to Hub: {repo_id}")
        self.trainer.push_to_hub(repo_id=repo_id, **kwargs)
