"""GRPO dataset formatter for text-to-SQL training.

This module formats preprocessed datasets for use with TRL's GRPOTrainer,
creating prompts and structuring data for reinforcement learning.
"""

import logging
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset

from environments.sql_env.environment import TextToSQLEnvironment

logger = logging.getLogger(__name__)


class GRPODatasetFormatter:
    """
    Formats preprocessed dataset for GRPO training.
    Creates prompts and structures data for TRL's GRPOTrainer.
    """

    def __init__(
        self, environment: TextToSQLEnvironment, tokenizer: Any, include_reference: bool = True
    ):
        """
        Initialize GRPO formatter.

        Args:
            environment: Text-to-SQL environment for prompt formatting
            tokenizer: Tokenizer for length validation
            include_reference: Whether to include reference SQL in dataset
        """
        self.environment = environment
        self.tokenizer = tokenizer
        self.include_reference = include_reference
        self.logger = logging.getLogger(__name__)

    def format_for_grpo(self, sample: Dict) -> Dict:
        """
        Format sample for GRPO training.

        Input sample:
        {
            'question': str,
            'schema': str,
            'sql': str,
            ...
        }

        Output format for GRPOTrainer:
        {
            'prompt': str,  # Formatted prompt for generation
            'reference': str,  # Ground truth SQL (optional)
            'context': Dict,  # Additional context for environment
        }
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
        output = {
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

    def format_dataset(self, dataset: Dataset) -> Dataset:
        """
        Format entire dataset for GRPO.

        Returns:
            Dataset with GRPO-compatible format
        """
        self.logger.info("Formatting dataset for GRPO")

        def format_fn(examples):
            """Batch formatting function."""
            results = {
                "prompt": [],
                "question": [],
                "schema": [],
            }

            if self.include_reference:
                results["reference"] = []

            results["context"] = []

            # Preserve additional fields
            for key in ["complexity", "sql_keywords", "is_valid"]:
                if key in examples:
                    results[key] = []

            # Handle both batched and single examples
            if isinstance(examples["question"], list):
                num_examples = len(examples["question"])
            else:
                num_examples = 1
                examples = {k: [v] for k, v in examples.items()}

            for i in range(num_examples):
                sample = {k: v[i] for k, v in examples.items()}

                try:
                    formatted = self.format_for_grpo(sample)

                    results["prompt"].append(formatted["prompt"])
                    results["question"].append(formatted["question"])
                    results["schema"].append(formatted["schema"])

                    if self.include_reference:
                        results["reference"].append(formatted.get("reference", ""))

                    results["context"].append(formatted.get("context", {}))

                    # Preserve additional fields
                    for key in ["complexity", "sql_keywords", "is_valid"]:
                        if key in results and key in formatted:
                            results[key].append(formatted[key])

                except Exception as e:
                    self.logger.warning(f"Failed to format sample {i}: {e}")
                    # Skip this sample by not appending to results
                    continue

            return results

        formatted_dataset = dataset.map(
            format_fn, batched=True, remove_columns=dataset.column_names, desc="Formatting for GRPO"
        )

        self.logger.info(f"Formatted {len(formatted_dataset)} samples for GRPO")

        return formatted_dataset

    def validate_tokenization(self, dataset: Dataset, max_length: int = 2048) -> Dict[str, Any]:
        """
        Validate that prompts fit within model's context window.

        Args:
            dataset: Formatted dataset
            max_length: Maximum token length

        Returns:
            Statistics about token lengths
        """
        self.logger.info("Validating tokenization")

        token_lengths = []
        too_long_count = 0

        # Sample up to 1000 examples for validation
        sample_size = min(len(dataset), 1000)
        indices = np.random.choice(len(dataset), sample_size, replace=False)

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
            "avg_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "max_token_length": max(token_lengths),
            "min_token_length": min(token_lengths),
            "too_long_count": too_long_count,
            "too_long_pct": (too_long_count / sample_size) * 100 if sample_size > 0 else 0,
            "max_length_threshold": max_length,
        }

        self.logger.info(
            f"Token length stats: avg={stats['avg_token_length']:.1f}, "
            f"max={stats['max_token_length']}, "
            f"too_long={stats['too_long_pct']:.1f}%"
        )

        if too_long_count > 0:
            self.logger.warning(
                f"{too_long_count} samples ({stats['too_long_pct']:.1f}%) exceed "
                f"max length of {max_length} tokens"
            )

        return stats

    def create_evaluation_set(self, dataset: Dataset, n_samples: int = 100) -> Dataset:
        """
        Create a small evaluation set for during-training eval.
        Samples diverse examples across complexity levels.

        Args:
            dataset: Full dataset
            n_samples: Number of samples to include

        Returns:
            Evaluation dataset
        """
        self.logger.info(f"Creating evaluation set with {n_samples} samples")

        # If dataset doesn't have complexity field, do random sampling
        if "complexity" not in dataset.column_names:
            self.logger.warning(
                "Dataset doesn't have 'complexity' field. "
                "Using random sampling instead of stratified."
            )
            indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
            return dataset.select(indices.tolist())

        # Stratified sampling by complexity
        try:
            # Get complexity distribution
            complexities = dataset["complexity"]
            complexity_counts: Dict[str, int] = {}
            complexity_indices: Dict[str, List[int]] = {"simple": [], "medium": [], "complex": []}

            for idx, complexity in enumerate(complexities):
                if complexity in complexity_indices:
                    complexity_indices[complexity].append(idx)
                    complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

            # Calculate samples per complexity level (proportional to distribution)
            total_samples = sum(complexity_counts.values())
            samples_per_level = {}

            for level, count in complexity_counts.items():
                proportion = count / total_samples
                samples_per_level[level] = max(1, int(n_samples * proportion))

            # Adjust to ensure total equals n_samples
            total_allocated = sum(samples_per_level.values())
            if total_allocated < n_samples:
                # Add remaining to the largest group
                largest_level = max(complexity_counts, key=lambda k: complexity_counts[k])
                samples_per_level[largest_level] += n_samples - total_allocated

            # Sample from each complexity level
            selected_indices: List[int] = []
            for level, n in samples_per_level.items():
                indices = complexity_indices[level]
                if len(indices) > 0:
                    sample_indices = np.random.choice(indices, min(n, len(indices)), replace=False)
                    selected_indices.extend(sample_indices.tolist())

            eval_dataset = dataset.select(selected_indices)

            self.logger.info(
                f"Created evaluation set with {len(eval_dataset)} samples. "
                f"Distribution: {dict(zip(*np.unique(eval_dataset['complexity'], return_counts=True)))}"
            )

            return eval_dataset

        except Exception as e:
            self.logger.warning(f"Failed to create stratified evaluation set: {e}")
            # Fall back to random sampling
            indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
            return dataset.select(indices.tolist())
