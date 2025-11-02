"""GRPO dataset formatter for text-to-SQL training.

This module formats preprocessed datasets for use with TRL's GRPOTrainer,
creating prompts and structuring data for reinforcement learning.
"""

import logging
from typing import Any

import numpy as np
from datasets import Dataset
from numpy.random import Generator

from src.environments.sql_env.environment import TextToSQLEnvironment

logger = logging.getLogger(__name__)


class GRPODatasetFormatter:
    """Formats preprocessed dataset for GRPO training.

    Creates prompts and structures data for TRL's GRPOTrainer,
    handling tokenization validation and stratified sampling.

    Attributes:
        environment: Text-to-SQL environment for prompt formatting.
        tokenizer: Tokenizer for length validation.
        include_reference: Whether to include reference SQL in dataset.
        logger: Logger instance for this class.
        rng: NumPy random number generator for reproducible sampling.
    """

    def __init__(
        self,
        environment: TextToSQLEnvironment,
        tokenizer: Any,
        include_reference: bool = True,
    ) -> None:
        """Initialize GRPO formatter.

        Args:
            environment: Text-to-SQL environment for prompt formatting.
            tokenizer: Tokenizer for length validation and encoding.
            include_reference: Whether to include reference SQL in dataset.
                Defaults to True.
        """
        self.environment = environment
        self.tokenizer = tokenizer
        self.include_reference = include_reference
        self.logger = logging.getLogger(__name__)
        self.rng: Generator = np.random.default_rng()

    def format_for_grpo(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Format sample for GRPO training.

        Transforms a preprocessed sample into the format expected by
        TRL's GRPOTrainer, including prompt generation and context
        preservation.

        Args:
            sample: Input sample dictionary containing:
                - question: Natural language query (required)
                - schema: Database schema (optional)
                - sql: Reference SQL query (optional)
                - Additional fields preserved if present

        Returns:
            Formatted sample dictionary containing:
            - prompt: Formatted prompt for generation
            - question: Original question
            - schema: Database schema
            - reference: Ground truth SQL (if include_reference=True)
            - context: Additional context for environment
            - Preserved fields: complexity, sql_keywords, is_valid

        Raises:
            ValueError: If sample doesn't contain required 'question' field.
        """
        question = sample.get("question", "")
        schema = sample.get("schema", "")
        sql = sample.get("sql", "")

        if not question:
            raise ValueError("Sample must contain 'question' field")

        # Format prompt using environment
        context = {"schema": schema} if schema else None
        prompt = self.environment.format_prompt(question, context)

        # Build output
        output: dict[str, Any] = {
            "prompt": prompt,
            "question": question,
            "schema": schema,
        }

        # Add reference SQL if requested
        if self.include_reference and sql:
            output["reference"] = sql

        # Add context for environment (used during reward computation)
        if context:
            output["context"] = context

        # Preserve other fields that might be useful
        for key in ["complexity", "sql_keywords", "is_valid"]:
            if key in sample:
                output[key] = sample[key]

        return output

    def _initialize_results_dict(
        self, examples: dict[str, Any]
    ) -> dict[str, list[Any]]:
        """Initialize the results dictionary for formatting.

        Args:
            examples: Input examples dictionary to check for keys.

        Returns:
            Empty results dictionary with appropriate keys initialized.
        """
        results: dict[str, list[Any]] = {
            "prompt": [],
            "question": [],
            "schema": [],
        }
        if self.include_reference:
            results["reference"] = []
        results["context"] = []
        for key in ["complexity", "sql_keywords", "is_valid"]:
            if key in examples:
                results[key] = []
        return results

    def _normalize_examples(
        self, examples: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        """Normalize examples to handle both batched and single examples.

        Args:
            examples: Input examples, may be batched or single.

        Returns:
            Tuple of (normalized examples as batch, number of examples).
        """
        if isinstance(examples["question"], list):
            num_examples = len(examples["question"])
        else:
            num_examples = 1
            examples = {k: [v] for k, v in examples.items()}
        return examples, num_examples

    def _append_formatted_sample(
        self, results: dict[str, list[Any]], formatted: dict[str, Any]
    ) -> None:
        """Append a formatted sample to the results.

        Args:
            results: Results dictionary to append to.
            formatted: Formatted sample to add.

        Returns:
            None. Modifies results in place.
        """
        results["prompt"].append(formatted["prompt"])
        results["question"].append(formatted["question"])
        results["schema"].append(formatted["schema"])

        if self.include_reference:
            results["reference"].append(formatted.get("reference", ""))

        results["context"].append(formatted.get("context", {}))

        for key in ["complexity", "sql_keywords", "is_valid"]:
            if key in results and key in formatted:
                results[key].append(formatted[key])

    def format_dataset(self, dataset: Dataset) -> Dataset:
        """Format entire dataset for GRPO.

        Applies batch formatting to transform all samples into
        GRPO-compatible format.

        Args:
            dataset: Preprocessed dataset to format.

        Returns:
            Dataset with GRPO-compatible format including prompts,
            references, and context.
        """
        self.logger.info("Formatting dataset for GRPO")

        def format_fn(examples: dict[str, Any]) -> dict[str, list[Any]]:
            """Batch formatting function."""
            results = self._initialize_results_dict(examples)
            examples, num_examples = self._normalize_examples(examples)

            for i in range(num_examples):
                sample = {k: v[i] for k, v in examples.items()}
                try:
                    formatted = self.format_for_grpo(sample)
                    self._append_formatted_sample(results, formatted)
                except Exception as e:
                    self.logger.warning(f"Failed to format sample {i}: {e}")
                    continue

            return results

        formatted_dataset = dataset.map(
            format_fn,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Formatting for GRPO",
        )

        self.logger.info(
            f"Formatted {len(formatted_dataset)} samples for GRPO"
        )

        return formatted_dataset

    def validate_tokenization(
        self, dataset: Dataset, max_length: int = 2048
    ) -> dict[str, Any]:
        """Validate that prompts fit within model's context window.

        Samples prompts from the dataset and checks their tokenized
        lengths against the maximum length threshold.

        Args:
            dataset: Formatted dataset with prompts.
            max_length: Maximum token length threshold. Defaults to 2048.

        Returns:
            Dictionary containing tokenization statistics:
            - num_samples_checked: Number of samples validated
            - avg/median/max/min_token_length: Length statistics
            - too_long_count: Number exceeding max_length
            - too_long_pct: Percentage exceeding max_length
            - max_length_threshold: The threshold used
        """
        self.logger.info("Validating tokenization")

        token_lengths = []
        too_long_count = 0

        # Sample up to 1000 examples for validation
        sample_size = min(len(dataset), 1000)
        indices = self.rng.choice(len(dataset), sample_size, replace=False)

        for idx in indices:
            sample = dataset[int(idx)]
            prompt = sample["prompt"]

            # Tokenize
            tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            length = len(tokens)
            token_lengths.append(length)

            if length > max_length:
                too_long_count += 1

        stats = {
            "num_samples_checked": sample_size,
            "avg_token_length": float(np.mean(token_lengths)),
            "median_token_length": float(np.median(token_lengths)),
            "max_token_length": int(max(token_lengths)),
            "min_token_length": int(min(token_lengths)),
            "too_long_count": too_long_count,
            "too_long_pct": (
                (too_long_count / sample_size) * 100 if sample_size > 0 else 0
            ),
            "max_length_threshold": max_length,
        }

        self.logger.info(
            f"Token length stats: avg={stats['avg_token_length']:.1f}, "
            f"max={stats['max_token_length']}, "
            f"too_long={stats['too_long_pct']:.1f}%"
        )

        if too_long_count > 0:
            self.logger.warning(
                f"{too_long_count} samples "
                f"({stats['too_long_pct']:.1f}%) exceed "
                f"max length of {max_length} tokens"
            )

        return stats

    def create_evaluation_set(
        self, dataset: Dataset, n_samples: int = 100
    ) -> Dataset:
        """Create a small evaluation set for during-training eval.

        Samples diverse examples across complexity levels using
        stratified sampling when possible, falling back to random
        sampling if complexity information is unavailable.

        Args:
            dataset: Full dataset to sample from.
            n_samples: Number of samples to include in evaluation set.
                Defaults to 100.

        Returns:
            Evaluation dataset with stratified or random sampling.
        """
        self.logger.info(
            f"Creating evaluation set with {n_samples} samples"
        )

        # If dataset doesn't have complexity field, do random sampling
        if "complexity" not in dataset.column_names:
            self.logger.warning(
                "Dataset doesn't have 'complexity' field. "
                "Using random sampling instead of stratified."
            )
            indices = self.rng.choice(
                len(dataset), min(n_samples, len(dataset)), replace=False
            )
            return dataset.select(list(indices))

        # Stratified sampling by complexity
        try:
            # Get complexity distribution
            complexities = dataset["complexity"]
            complexity_counts: dict[str, int] = {}
            complexity_indices: dict[str, list[int]] = {
                "simple": [],
                "medium": [],
                "complex": [],
            }

            for idx, complexity in enumerate(complexities):
                if complexity in complexity_indices:
                    complexity_indices[complexity].append(idx)
                    current_count = complexity_counts.get(complexity, 0)
                    complexity_counts[complexity] = current_count + 1

            # Calculate samples per complexity level (proportional)
            total_samples = sum(complexity_counts.values())
            samples_per_level = {}

            for level, count in complexity_counts.items():
                proportion = count / total_samples
                samples_per_level[level] = max(1, int(n_samples * proportion))

            # Adjust to ensure total equals n_samples
            total_allocated = sum(samples_per_level.values())
            if total_allocated < n_samples:
                # Add remaining to the largest group
                largest_level = max(
                    complexity_counts, key=lambda k: complexity_counts[k]
                )
                samples_per_level[largest_level] += (
                    n_samples - total_allocated
                )

            # Sample from each complexity level
            selected_indices: list[int] = []
            for level, n in samples_per_level.items():
                level_indices = list(complexity_indices[level])
                if len(level_indices) > 0:
                    sample_indices_arr = self.rng.choice(
                        level_indices, min(n, len(level_indices)), replace=False
                    )
                    # Convert numpy array to list of ints
                    selected_indices.extend(
                        [int(idx) for idx in sample_indices_arr]
                    )

            eval_dataset = dataset.select(selected_indices)

            complexity_dist = dict(
                zip(
                    *np.unique(
                        eval_dataset["complexity"], return_counts=True
                    ),
                    strict=True,
                )
            )

            self.logger.info(
                f"Created evaluation set with {len(eval_dataset)} samples. "
                f"Distribution: {complexity_dist}"
            )

            return eval_dataset

        except Exception as e:
            self.logger.warning(
                f"Failed to create stratified evaluation set: {e}"
            )
            # Fall back to random sampling
            indices = self.rng.choice(
                len(dataset), min(n_samples, len(dataset)), replace=False
            )
            return dataset.select(indices.tolist())
