"""Training Callbacks for GRPO fine-tuning.

Custom callbacks for SQL evaluation and enhanced WandB logging during training.
"""

import logging
from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState

from src.environments.sql_env.environment import TextToSQLEnvironment
from src.rubrics.sql_rubric import SQLValidationRubric

try:
    from types import ModuleType

    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb: ModuleType | None = None

logger = logging.getLogger(__name__)


class SQLEvaluationCallback(TrainerCallback):
    """Callback for custom SQL evaluation during training.

    Generates SQL queries and computes detailed metrics at regular intervals
    during training. Optionally logs examples to WandB.

    Attributes:
        environment: Text-to-SQL environment for prompt formatting.
        rubric: SQL validation rubric for scoring.
        eval_dataset: Dataset for evaluation.
        tokenizer: Tokenizer for the model.
        eval_frequency: Steps between evaluations.
        num_samples: Number of samples to evaluate.
        log_examples: Whether to log example outputs.
        logger: Logger instance for this class.
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
    ) -> None:
        """Initialize evaluation callback.

        Args:
            environment: Text-to-SQL environment for prompt formatting.
            rubric: SQL validation rubric for computing rewards.
            eval_dataset: Dataset for evaluation with 'question', 'schema',
                and 'reference'/'sql' fields.
            tokenizer: Tokenizer compatible with the model.
            eval_frequency: Number of steps between evaluations.
                Defaults to 500.
            num_samples: Number of samples to evaluate each time.
                Defaults to 10.
            log_examples: Whether to log example outputs to WandB.
                Defaults to True.
        """
        self.environment = environment
        self.rubric = rubric
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.eval_frequency = eval_frequency
        self.num_samples = num_samples
        self.log_examples = log_examples
        self.logger = logging.getLogger(__name__)

    def on_step_end(
        self,
        args: Any,  # noqa: ARG002
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        """Run evaluation at specified intervals.

        Args:
            args: Training arguments. Required by TrainerCallback interface
                but unused.
            state: Current trainer state with step information.
            control: Trainer control object.
            **kwargs: Additional context from trainer (model, etc.).

        Returns:
            TrainerControl object (unchanged).
        """
        if state.global_step % self.eval_frequency == 0:
            self.run_evaluation(state.global_step, **kwargs)

        return control

    def run_evaluation(self, step: int, **kwargs: Any) -> None:
        """Run detailed SQL evaluation.

        Generates SQL queries, computes metrics, and logs results to
        WandB if available.

        Args:
            step: Current training step for logging.
            **kwargs: Additional context (model, tokenizer, etc.).

        Returns:
            None. Logs metrics and examples.
        """
        self.logger.info(f"Running SQL evaluation at step {step}")

        model = kwargs.get("model")

        # Sample evaluation data
        eval_samples = self.eval_dataset.shuffle().select(
            range(self.num_samples)
        )

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
                question=sample["question"],
                context={"schema": sample["schema"]},
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(
                model.device  # type: ignore[union-attr]
            )

            with torch.no_grad():
                outputs = model.generate(  # type: ignore[union-attr]
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
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
                        "reference": sample.get(
                            "reference", sample.get("sql", "")
                        ),
                        "generated": generated_sql,
                        "reward": reward,
                        "valid": parsed["valid"],
                    }
                )

        # Compute aggregate metrics
        n = len(eval_samples)
        eval_metrics = {
            "eval/valid_sql_pct": metrics["valid_sql_count"] / n * 100,
            "eval/avg_reward": metrics["total_reward"] / n,
            "eval/syntax_correct_pct": metrics["syntax_correct"] / n * 100,
            "eval/keyword_present_pct": metrics["keyword_present"] / n * 100,
        }

        # Log to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(eval_metrics, step=step)

            if self.log_examples:
                # Create a table with examples
                table = wandb.Table(
                    columns=[
                        "Question",
                        "Reference SQL",
                        "Generated SQL",
                        "Reward",
                        "Valid",
                    ],
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
    """Enhanced WandB logging callback for GRPO training.

    Initializes WandB runs and logs metrics during training.

    Attributes:
        config: Configuration dictionary to log.
        logger: Logger instance for this class.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize WandB logging.

        Args:
            config: Configuration dictionary to log to WandB including
                project name, run name, and hyperparameters.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def on_train_begin(
        self,
        args: Any,
        state: TrainerState,  # noqa: ARG002
        control: TrainerControl,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize WandB run.

        Args:
            args: Training arguments with WandB configuration.
            state: Current trainer state. Required by TrainerCallback
                interface but unused.
            control: Trainer control object. Required by TrainerCallback
                interface but unused.
            **kwargs: Additional context from trainer. Required by
                TrainerCallback interface but unused.

        Returns:
            None. Initializes WandB run if configured.
        """
        if not WANDB_AVAILABLE:
            self.logger.warning(
                "WandB not available, skipping initialization"
            )
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
        args: Any,  # noqa: ARG002
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        logs: dict[str, float] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Log metrics to WandB.

        Args:
            args: Training arguments. Required by TrainerCallback interface
                but unused.
            state: Current trainer state with step information.
            control: Trainer control object. Required by TrainerCallback
                interface but unused.
            logs: Dictionary of metrics to log. Defaults to None.
            **kwargs: Additional context from trainer. Required by
                TrainerCallback interface but unused.

        Returns:
            None. Logs metrics to WandB.
        """
        if not WANDB_AVAILABLE:
            return

        if logs and wandb.run is not None:
            wandb.log(logs, step=state.global_step)
