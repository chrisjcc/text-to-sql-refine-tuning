"""Analyze dataset quality and generate report.

This script analyzes the processed SQL dataset and generates
a comprehensive quality report including:
- Distribution of SQL complexity
- Common SQL patterns
- Schema statistics
- Question length distribution
- Potential data quality issues
- Examples of each complexity level
"""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from numpy.random import Generator

from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def analyze_complexity_distribution(dataset: Dataset) -> None:
    """Analyze distribution of SQL complexity.

    Args:
        dataset: HuggingFace Dataset object containing SQL queries with
            complexity labels.

    Returns:
        None. Logs complexity distribution statistics to the logger.
    """
    logger.info("\n" + "=" * 80)
    logger.info("SQL COMPLEXITY DISTRIBUTION")
    logger.info("=" * 80)

    if "complexity" not in dataset.column_names:
        logger.warning("No complexity information available")
        return

    complexities = dataset["complexity"]
    complexity_counts = Counter(complexities)

    total = len(complexities)
    for complexity in ["simple", "medium", "complex"]:
        count = complexity_counts.get(complexity, 0)
        pct = (count / total) * 100 if total > 0 else 0
        logger.info(f"{complexity.capitalize():10s}: {count:6d} ({pct:5.1f}%)")


def analyze_sql_patterns(dataset: Dataset) -> None:
    """Analyze common SQL patterns.

    Args:
        dataset: HuggingFace Dataset object containing SQL queries with
            extracted keywords.

    Returns:
        None. Logs top SQL keywords and their frequencies to the logger.
    """
    logger.info("\n" + "=" * 80)
    logger.info("COMMON SQL PATTERNS")
    logger.info("=" * 80)

    if "sql_keywords" not in dataset.column_names:
        logger.warning("No SQL keyword information available")
        return

    all_keywords = []
    for keywords in dataset["sql_keywords"]:
        all_keywords.extend(keywords)

    keyword_counts = Counter(all_keywords)

    logger.info("\nTop 15 SQL keywords:")
    for keyword, count in keyword_counts.most_common(15):
        pct = (count / len(dataset)) * 100
        logger.info(f"  {keyword:15s}: {count:6d} ({pct:5.1f}%)")


def analyze_length_distribution(dataset: Dataset) -> None:
    """Analyze length distributions.

    Analyzes the length distributions for questions, SQL queries, and schemas
    in the dataset, including mean, median, min, max, standard deviation,
    and percentiles.

    Args:
        dataset: HuggingFace Dataset object containing length metrics for
            questions, SQL queries, and schemas.

    Returns:
        None. Logs length distribution statistics to the logger.
    """
    logger.info("\n" + "=" * 80)
    logger.info("LENGTH DISTRIBUTION")
    logger.info("=" * 80)

    if "question_length" in dataset.column_names:
        question_lengths = dataset["question_length"]
        logger.info("\nQuestion length (words):")
        logger.info(f"  Mean:   {np.mean(question_lengths):6.1f}")
        logger.info(f"  Median: {np.median(question_lengths):6.1f}")
        logger.info(f"  Min:    {np.min(question_lengths):6.0f}")
        logger.info(f"  Max:    {np.max(question_lengths):6.0f}")
        logger.info(f"  Std:    {np.std(question_lengths):6.1f}")

        # Percentiles
        logger.info("\n  Percentiles:")
        for p in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(question_lengths, p)
            logger.info(f"    {p:2d}th: {val:6.1f}")

    if "sql_length" in dataset.column_names:
        sql_lengths = dataset["sql_length"]
        logger.info("\nSQL length (words):")
        logger.info(f"  Mean:   {np.mean(sql_lengths):6.1f}")
        logger.info(f"  Median: {np.median(sql_lengths):6.1f}")
        logger.info(f"  Min:    {np.min(sql_lengths):6.0f}")
        logger.info(f"  Max:    {np.max(sql_lengths):6.0f}")
        logger.info(f"  Std:    {np.std(sql_lengths):6.1f}")

        # Percentiles
        logger.info("\n  Percentiles:")
        for p in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(sql_lengths, p)
            logger.info(f"    {p:2d}th: {val:6.1f}")

    if "schema_length" in dataset.column_names:
        schema_lengths = dataset["schema_length"]
        logger.info("\nSchema length (words):")
        logger.info(f"  Mean:   {np.mean(schema_lengths):6.1f}")
        logger.info(f"  Median: {np.median(schema_lengths):6.1f}")
        logger.info(f"  Min:    {np.min(schema_lengths):6.0f}")
        logger.info(f"  Max:    {np.max(schema_lengths):6.0f}")
        logger.info(f"  Std:    {np.std(schema_lengths):6.1f}")


def analyze_schema_statistics(dataset: Dataset) -> None:
    """Analyze schema statistics.

    Analyzes database schema characteristics including number of tables,
    columns, and their distributions across the dataset.

    Args:
        dataset: HuggingFace Dataset object containing database schemas.

    Returns:
        None. Logs schema statistics to the logger.
    """
    logger.info("\n" + "=" * 80)
    logger.info("SCHEMA STATISTICS")
    logger.info("=" * 80)

    if "schema" not in dataset.column_names:
        logger.warning("No schema information available")
        return

    table_counts = []
    column_counts = []
    all_tables = set()

    for schema in dataset["schema"]:
        # Count tables
        table_pattern = r"CREATE TABLE\s+(?:IF NOT EXISTS\s+)?(\w+)"
        tables = re.findall(table_pattern, schema, re.IGNORECASE)
        table_counts.append(len(tables))
        all_tables.update(tables)

        # Count columns (rough estimate)
        column_pattern = r"(\w+)\s+(?:INT|VARCHAR|TEXT|FLOAT|DOUBLE|DATE|BOOLEAN)"
        columns = re.findall(column_pattern, schema, re.IGNORECASE)
        column_counts.append(len(columns))

    logger.info(f"\nTotal unique tables: {len(all_tables)}")

    if table_counts:
        logger.info("\nTables per sample:")
        logger.info(f"  Mean:   {np.mean(table_counts):6.1f}")
        logger.info(f"  Median: {np.median(table_counts):6.1f}")
        logger.info(f"  Min:    {np.min(table_counts):6.0f}")
        logger.info(f"  Max:    {np.max(table_counts):6.0f}")

    if column_counts:
        logger.info("\nColumns per sample:")
        logger.info(f"  Mean:   {np.mean(column_counts):6.1f}")
        logger.info(f"  Median: {np.median(column_counts):6.1f}")
        logger.info(f"  Min:    {np.min(column_counts):6.0f}")
        logger.info(f"  Max:    {np.max(column_counts):6.0f}")


def find_quality_issues(dataset: Dataset) -> None:
    """Identify potential data quality issues.

    Checks for various data quality problems including invalid samples,
    empty fields, very short questions, and very long SQL queries.

    Args:
        dataset: HuggingFace Dataset object to analyze for quality issues.

    Returns:
        None. Logs identified quality issues to the logger.
    """
    logger.info("\n" + "=" * 80)
    logger.info("POTENTIAL QUALITY ISSUES")
    logger.info("=" * 80)

    issues_found = False

    # Check for invalid samples
    if "is_valid" in dataset.column_names:
        invalid_count = sum(1 for v in dataset["is_valid"] if not v)
        if invalid_count > 0:
            pct = (invalid_count / len(dataset)) * 100
            logger.warning(f"\n⚠ Invalid samples: {invalid_count} ({pct:.1f}%)")
            issues_found = True

    # Check for empty fields
    for field in ["question", "sql", "schema"]:
        if field in dataset.column_names:
            empty_count = sum(1 for v in dataset[field] if not v or not v.strip())
            if empty_count > 0:
                pct = (empty_count / len(dataset)) * 100
                logger.warning(f"\n⚠ Empty {field}: {empty_count} ({pct:.1f}%)")
                issues_found = True

    # Check for very short questions
    if "question_length" in dataset.column_names:
        very_short = sum(1 for v in dataset["question_length"] if v < 3)
        if very_short > 0:
            pct = (very_short / len(dataset)) * 100
            logger.warning(f"\n⚠ Very short questions (<3 words): " f"{very_short} ({pct:.1f}%)")
            issues_found = True

    # Check for very long SQL
    if "sql_length" in dataset.column_names:
        very_long = sum(1 for v in dataset["sql_length"] if v > 100)
        if very_long > 0:
            pct = (very_long / len(dataset)) * 100
            logger.warning(f"\n⚠ Very long SQL queries (>100 words): " f"{very_long} ({pct:.1f}%)")
            issues_found = True

    if not issues_found:
        logger.info("\n✓ No major quality issues detected!")


def show_examples(dataset: Dataset, n_per_complexity: int = 2) -> None:
    """Show example queries for each complexity level.

    Displays random sample queries from each complexity level (simple, medium,
    complex) along with their SQL and schema information.

    Args:
        dataset: HuggingFace Dataset object containing SQL queries with
            complexity labels.
        n_per_complexity: Number of examples to show per complexity level.
            Defaults to 2.

    Returns:
        None. Logs example queries to the logger.
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE QUERIES BY COMPLEXITY")
    logger.info("=" * 80)

    if "complexity" not in dataset.column_names:
        logger.warning("No complexity information available")
        return

    required_fields = ["question", "sql", "complexity"]
    if not all(field in dataset.column_names for field in required_fields):
        logger.warning("Missing required fields for examples")
        return

    # Create random generator
    rng: Generator = np.random.default_rng()

    for complexity in ["simple", "medium", "complex"]:
        logger.info(f"\n{'-' * 80}")
        logger.info(f"{complexity.upper()} QUERIES")
        logger.info(f"{'-' * 80}")

        # Find examples of this complexity
        indices = [i for i, c in enumerate(dataset["complexity"]) if c == complexity]

        if not indices:
            logger.warning(f"No {complexity} examples found")
            continue

        # Sample randomly
        sample_indices = rng.choice(indices, min(n_per_complexity, len(indices)), replace=False)

        for idx, sample_idx in enumerate(sample_indices, 1):
            sample: dict[str, Any] = dataset[int(sample_idx)]
            logger.info(f"\nExample {idx}:")
            logger.info(f"Question: {sample['question']}")
            logger.info(f"SQL: {sample['sql']}")

            if "schema" in sample and sample["schema"]:
                # Show first table definition only
                schema: str = sample["schema"]
                if "CREATE TABLE" in schema:
                    first_table = schema.split("CREATE TABLE")[1]
                else:
                    first_table = schema
                first_table = "CREATE TABLE" + first_table.split(";")[0]
                if len(first_table) > 200:
                    first_table = first_table[:200] + "..."
                logger.info(f"Schema: {first_table}")


def main() -> None:
    """Main analysis function.

    Orchestrates the complete dataset analysis process including loading
    the processed dataset, running all analysis functions on each split,
    and generating a comprehensive quality report.

    Returns:
        None. Logs analysis results to the logger and writes to log file.
    """
    # Setup logging
    setup_logging(log_level="INFO", log_dir="logs", log_file="data_analysis.log")

    # Check for processed dataset
    processed_path = Path("./data_cache/processed")

    if not processed_path.exists():
        logger.error("\n" + "=" * 80)
        logger.error("ERROR: Processed dataset not found!")
        logger.error("=" * 80)
        logger.error(f"\nExpected location: {processed_path}")
        logger.error(
            "\nPlease run 'python scripts/prepare_data.py' first to " "prepare the dataset."
        )

        return

    logger.info("\n" + "=" * 80)
    logger.info("DATASET QUALITY ANALYSIS REPORT")
    logger.info("=" * 80)
    logger.info(f"\nDataset location: {processed_path}")

    # Load dataset
    try:
        dataset_dict: DatasetDict = load_from_disk(str(processed_path))
        logger.info(f"Loaded splits: {list(dataset_dict.keys())}")
    except Exception as e:
        logger.error(f"\n❌ Failed to load dataset: {e}")
        return

    # Analyze each split
    for split_name in dataset_dict:
        logger.info("\n" + "=" * 80)
        logger.info(f"ANALYZING {split_name.upper()} SPLIT")
        logger.info("=" * 80)
        logger.info(f"Total samples: {len(dataset_dict[split_name])}")

        dataset: Dataset = dataset_dict[split_name]

        # Run analyses
        analyze_complexity_distribution(dataset)
        analyze_sql_patterns(dataset)
        analyze_length_distribution(dataset)
        analyze_schema_statistics(dataset)
        find_quality_issues(dataset)
        show_examples(dataset, n_per_complexity=2)

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
