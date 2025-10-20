"""Training Callbacks for GRPO fine-tuning.

Custom callbacks for SQL evaluation and enhanced WandB logging during training.
"""

import logging
from typing import Any, Dict, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState

from src.environments.sql_env.environment import TextToSQLEnvironment
from src.rubrics.sql_rubric import SQLValidationRubric

try:
    import wandb  # type: ignore

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore


logger = logging.getLogger(__name__)


class SQLEvaluationCallback(TrainerCallback):
    """
    Callback for custom SQL evaluation during training.
    Generates SQL queries and computes detailed metrics.
    """

    def __init__(
        self,
        environment: TextToSQLEnvironment,
        rubric: SQLValidationRubric,
        eval_dataset: Any,
        tokenizer: Any,
        eval_frequency: int = 500,
        num_samples: int = 10,
        log_examples: bool = True,
    ):
        """
        Initialize evaluation callback.

        Args:
            environment: Text-to-SQL environment
            rubric: SQL validation rubric
            eval_dataset: Dataset for evaluation
            tokenizer: Tokenizer
            eval_frequency: Steps between evaluations
            num_samples: Number of samples to evaluate
            log_examples: Whether to log example outputs
        """
        self.environment = environment
        self.rubric = rubric
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.eval_frequency = eval_frequency
        self.num_samples = num_samples
        self.log_examples = log_examples
        self.logger = logging.getLogger(__name__)

    def on_step_end(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs):
        """Run evaluation at specified intervals."""
        if state.global_step % self.eval_frequency == 0:
            self.run_evaluation(state.global_step, **kwargs)

        return control

    def run_evaluation(self, step: int, **kwargs):
        """
        Run detailed SQL evaluation.

        Args:
            step: Current training step
            **kwargs: Additional context (model, tokenizer, etc.)
        """
        self.logger.info(f"Running SQL evaluation at step {step}")

        model = kwargs.get("model")

        # Sample evaluation data
        eval_samples = self.eval_dataset.shuffle().select(range(self.num_samples))

        metrics = {
            "valid_sql_count": 0,
            "total_reward": 0.0,
            "syntax_correct": 0,
            "keyword_present": 0,
        }

        examples = []

        for i, sample in enumerate(eval_samples):
            # Generate SQL
            prompt = self.environment.format_prompt(
                question=sample["question"], context={"schema": sample["schema"]}
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)  # type: ignore[union-attr]

            with torch.no_grad():
                outputs = model.generate(  # type: ignore[union-attr]
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_sql = generated_text[len(prompt) :].strip()

            # Compute metrics
            parsed = self.environment.parse_response(generated_sql)
            reward = self.rubric.score(generated_sql)
            detailed_scores = self.rubric.get_detailed_scores(generated_sql)

            # Update metrics
            if parsed["valid"]:
                metrics["valid_sql_count"] += 1
            metrics["total_reward"] += reward
            if detailed_scores["syntax"] > 0.5:
                metrics["syntax_correct"] += 1
            if detailed_scores["keywords"] > 0.5:
                metrics["keyword_present"] += 1

            # Collect examples for logging
            if self.log_examples and i < 3:
                examples.append(
                    {
                        "question": sample["question"],
                        "reference": sample.get("reference", sample.get("sql", "")),
                        "generated": generated_sql,
                        "reward": reward,
                        "valid": parsed["valid"],
                    }
                )

        # Compute aggregate metrics
        n = len(eval_samples)
        eval_metrics = {
            f"eval/valid_sql_pct": metrics["valid_sql_count"] / n * 100,
            f"eval/avg_reward": metrics["total_reward"] / n,
            f"eval/syntax_correct_pct": metrics["syntax_correct"] / n * 100,
            f"eval/keyword_present_pct": metrics["keyword_present"] / n * 100,
        }

        # Log to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(eval_metrics, step=step)

            if self.log_examples:
                # Create a table with examples
                table = wandb.Table(
                    columns=["Question", "Reference SQL", "Generated SQL", "Reward", "Valid"],
                    data=[
                        [
                            ex["question"],
                            ex["reference"],
                            ex["generated"],
                            ex["reward"],
                            ex["valid"],
                        ]
                        for ex in examples
                    ],
                )
                wandb.log({f"eval/examples_step_{step}": table}, step=step)

        # Log to console
        self.logger.info(f"Evaluation results (step {step}):")
        for key, value in eval_metrics.items():
            self.logger.info(f"  {key}: {value:.2f}")


class WandbLoggingCallback(TrainerCallback):
    """
    Enhanced WandB logging callback for GRPO training.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize WandB logging.

        Args:
            config: Configuration dictionary to log
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def on_train_begin(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs):
        """Initialize WandB run."""
        if not WANDB_AVAILABLE:
            self.logger.warning("WandB not available, skipping initialization")
            return

        if wandb.run is None and args.report_to and "wandb" in args.report_to:
            wandb.init(
                project=self.config.get("project", "text-to-sql-grpo"),
                name=self.config.get("name", args.run_name),
                config=self.config,
            )
            self.logger.info("WandB logging initialized")

    def on_log(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Log metrics to WandB."""
        if not WANDB_AVAILABLE:
            return

        if logs and wandb.run is not None:
            wandb.log(logs, step=state.global_step)
