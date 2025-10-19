"""Dataset loader for SQL datasets.

This module provides utilities for loading and managing SQL datasets
from HuggingFace Hub, specifically designed for the b-mc2/sql-create-context
dataset and similar text-to-SQL datasets.
"""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


class SQLDatasetLoader:
    """
    Loads and manages SQL datasets for text-to-SQL training.
    Handles b-mc2/sql-create-context and custom datasets.
    """

    def __init__(
        self,
        dataset_name: str = "b-mc2/sql-create-context",
        cache_dir: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize dataset loader.

        Args:
            dataset_name: HuggingFace dataset identifier
            cache_dir: Directory for caching downloaded data
            seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.seed = seed
        self.logger = logging.getLogger(__name__)

        # Set random seed for reproducibility
        np.random.seed(seed)

    def load(self, split: Optional[str] = None, streaming: bool = False) -> DatasetDict:
        """
        Load dataset from HuggingFace Hub.

        Args:
            split: Specific split to load (train/validation/test) or None for all
            streaming: Whether to use streaming mode for large datasets

        Returns:
            DatasetDict with loaded splits
        """
        self.logger.info(f"Loading dataset: {self.dataset_name}")

        try:
            dataset = load_dataset(
                self.dataset_name, split=split, cache_dir=self.cache_dir, streaming=streaming
            )

            if split is None:
                self.logger.info(f"Loaded splits: {list(dataset.keys())}")
            else:
                self.logger.info(f"Loaded split: {split}")

            return dataset

        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise

    def create_splits(
        self,
        dataset: Dataset,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        stratify: bool = False,
    ) -> DatasetDict:
        """
        Create train/val/test splits if not already split.

        Args:
            dataset: Full dataset to split
            train_size: Proportion for training
            val_size: Proportion for validation
            test_size: Proportion for testing
            stratify: Whether to stratify by SQL complexity

        Returns:
            DatasetDict with train/val/test splits
        """
        # Validate split sizes
        total = train_size + val_size + test_size
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split sizes must sum to 1.0, got {total}")

        self.logger.info(f"Creating splits: train={train_size}, val={val_size}, test={test_size}")

        if stratify:
            self.logger.info("Stratifying by SQL complexity")
            # This would require preprocessing first to get complexity
            # For now, we'll do simple random split
            self.logger.warning(
                "Stratified splitting requires preprocessing. " "Falling back to random split."
            )

        # First split: train vs rest
        train_test = dataset.train_test_split(test_size=(val_size + test_size), seed=self.seed)

        # Second split: validation vs test
        if test_size > 0:
            val_test_ratio = test_size / (val_size + test_size)
            val_test = train_test["test"].train_test_split(test_size=val_test_ratio, seed=self.seed)

            dataset_dict = DatasetDict(
                {
                    "train": train_test["train"],
                    "validation": val_test["train"],
                    "test": val_test["test"],
                }
            )
        else:
            dataset_dict = DatasetDict(
                {"train": train_test["train"], "validation": train_test["test"]}
            )

        # Log split sizes
        for split_name, split_data in dataset_dict.items():
            self.logger.info(f"{split_name}: {len(split_data)} samples")

        return dataset_dict

    def get_statistics(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Compute dataset statistics for analysis.

        Returns:
            Dict with stats like:
            - total_samples
            - avg_question_length
            - avg_sql_length
            - avg_schema_length
            - unique_tables_count
            - sql_keyword_distribution
        """
        self.logger.info("Computing dataset statistics")

        if len(dataset) == 0:
            return {"total_samples": 0, "error": "Empty dataset"}

        # Initialize counters
        question_lengths = []
        sql_lengths = []
        schema_lengths = []
        sql_keywords = []
        tables = set()

        # SQL keywords to track
        keywords_to_track = [
            "SELECT",
            "FROM",
            "WHERE",
            "JOIN",
            "INNER JOIN",
            "LEFT JOIN",
            "RIGHT JOIN",
            "GROUP BY",
            "ORDER BY",
            "HAVING",
            "LIMIT",
            "UNION",
            "DISTINCT",
            "COUNT",
            "SUM",
            "AVG",
            "MAX",
            "MIN",
        ]

        # Sample up to 10000 examples for statistics
        sample_size = min(len(dataset), 10000)
        indices = np.random.choice(len(dataset), sample_size, replace=False)

        for idx in indices:
            sample = dataset[int(idx)]

            # Question length
            question = sample.get("question", "")
            question_lengths.append(len(question.split()))

            # SQL length
            answer = sample.get("answer", "")
            sql_lengths.append(len(answer.split()))

            # Schema length
            context = sample.get("context", "")
            schema_lengths.append(len(context.split()))

            # Extract keywords from SQL
            answer_upper = answer.upper()
            for keyword in keywords_to_track:
                if keyword in answer_upper:
                    sql_keywords.append(keyword)

            # Extract table names from schema
            table_pattern = r"CREATE TABLE\s+(?:IF NOT EXISTS\s+)?(\w+)"
            table_matches = re.findall(table_pattern, context, re.IGNORECASE)
            tables.update(table_matches)

        # Compute statistics
        stats = {
            "total_samples": len(dataset),
            "sampled_for_stats": sample_size,
            "avg_question_length": np.mean(question_lengths) if question_lengths else 0,
            "avg_sql_length": np.mean(sql_lengths) if sql_lengths else 0,
            "avg_schema_length": np.mean(schema_lengths) if schema_lengths else 0,
            "median_question_length": np.median(question_lengths) if question_lengths else 0,
            "median_sql_length": np.median(sql_lengths) if sql_lengths else 0,
            "median_schema_length": np.median(schema_lengths) if schema_lengths else 0,
            "max_question_length": max(question_lengths) if question_lengths else 0,
            "max_sql_length": max(sql_lengths) if sql_lengths else 0,
            "max_schema_length": max(schema_lengths) if schema_lengths else 0,
            "unique_tables_count": len(tables),
            "sql_keyword_distribution": dict(Counter(sql_keywords).most_common(10)),
        }

        return stats

    def load_from_disk(self, path: str) -> DatasetDict:
        """
        Load processed dataset from disk.

        Args:
            path: Path to saved dataset

        Returns:
            Loaded DatasetDict
        """
        self.logger.info(f"Loading dataset from disk: {path}")

        try:
            from datasets import load_from_disk

            dataset = load_from_disk(path)
            self.logger.info(f"Successfully loaded dataset from {path}")
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to load dataset from disk: {e}")
            raise

    def save_to_disk(self, dataset: DatasetDict, path: str) -> None:
        """
        Save dataset to disk.

        Args:
            dataset: Dataset to save
            path: Path to save location
        """
        self.logger.info(f"Saving dataset to disk: {path}")

        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(path)
            self.logger.info(f"Successfully saved dataset to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save dataset to disk: {e}")
            raise
