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

import sys
from pathlib import Path
import logging
import numpy as np
from collections import Counter
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_from_disk
from src.utils.logging_utils import setup_logging


def analyze_complexity_distribution(dataset):
    """Analyze distribution of SQL complexity."""
    print("\n" + "=" * 80)
    print("SQL COMPLEXITY DISTRIBUTION")
    print("=" * 80)

    if 'complexity' not in dataset.column_names:
        print("No complexity information available")
        return

    complexities = dataset['complexity']
    complexity_counts = Counter(complexities)

    total = len(complexities)
    for complexity in ['simple', 'medium', 'complex']:
        count = complexity_counts.get(complexity, 0)
        pct = (count / total) * 100 if total > 0 else 0
        print(f"{complexity.capitalize():10s}: {count:6d} ({pct:5.1f}%)")


def analyze_sql_patterns(dataset):
    """Analyze common SQL patterns."""
    print("\n" + "=" * 80)
    print("COMMON SQL PATTERNS")
    print("=" * 80)

    if 'sql_keywords' not in dataset.column_names:
        print("No SQL keyword information available")
        return

    all_keywords = []
    for keywords in dataset['sql_keywords']:
        all_keywords.extend(keywords)

    keyword_counts = Counter(all_keywords)

    print("\nTop 15 SQL keywords:")
    for keyword, count in keyword_counts.most_common(15):
        pct = (count / len(dataset)) * 100
        print(f"  {keyword:15s}: {count:6d} ({pct:5.1f}%)")


def analyze_length_distribution(dataset):
    """Analyze length distributions."""
    print("\n" + "=" * 80)
    print("LENGTH DISTRIBUTION")
    print("=" * 80)

    if 'question_length' in dataset.column_names:
        question_lengths = dataset['question_length']
        print("\nQuestion length (words):")
        print(f"  Mean:   {np.mean(question_lengths):6.1f}")
        print(f"  Median: {np.median(question_lengths):6.1f}")
        print(f"  Min:    {np.min(question_lengths):6.0f}")
        print(f"  Max:    {np.max(question_lengths):6.0f}")
        print(f"  Std:    {np.std(question_lengths):6.1f}")

        # Percentiles
        print("\n  Percentiles:")
        for p in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(question_lengths, p)
            print(f"    {p:2d}th: {val:6.1f}")

    if 'sql_length' in dataset.column_names:
        sql_lengths = dataset['sql_length']
        print("\nSQL length (words):")
        print(f"  Mean:   {np.mean(sql_lengths):6.1f}")
        print(f"  Median: {np.median(sql_lengths):6.1f}")
        print(f"  Min:    {np.min(sql_lengths):6.0f}")
        print(f"  Max:    {np.max(sql_lengths):6.0f}")
        print(f"  Std:    {np.std(sql_lengths):6.1f}")

        # Percentiles
        print("\n  Percentiles:")
        for p in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(sql_lengths, p)
            print(f"    {p:2d}th: {val:6.1f}")

    if 'schema_length' in dataset.column_names:
        schema_lengths = dataset['schema_length']
        print("\nSchema length (words):")
        print(f"  Mean:   {np.mean(schema_lengths):6.1f}")
        print(f"  Median: {np.median(schema_lengths):6.1f}")
        print(f"  Min:    {np.min(schema_lengths):6.0f}")
        print(f"  Max:    {np.max(schema_lengths):6.0f}")
        print(f"  Std:    {np.std(schema_lengths):6.1f}")


def analyze_schema_statistics(dataset):
    """Analyze schema statistics."""
    print("\n" + "=" * 80)
    print("SCHEMA STATISTICS")
    print("=" * 80)

    if 'schema' not in dataset.column_names:
        print("No schema information available")
        return

    table_counts = []
    column_counts = []
    all_tables = set()

    for schema in dataset['schema']:
        # Count tables
        table_pattern = r'CREATE TABLE\s+(?:IF NOT EXISTS\s+)?(\w+)'
        tables = re.findall(table_pattern, schema, re.IGNORECASE)
        table_counts.append(len(tables))
        all_tables.update(tables)

        # Count columns (rough estimate)
        column_pattern = r'(\w+)\s+(?:INT|VARCHAR|TEXT|FLOAT|DOUBLE|DATE|BOOLEAN)'
        columns = re.findall(column_pattern, schema, re.IGNORECASE)
        column_counts.append(len(columns))

    print(f"\nTotal unique tables: {len(all_tables)}")

    if table_counts:
        print("\nTables per sample:")
        print(f"  Mean:   {np.mean(table_counts):6.1f}")
        print(f"  Median: {np.median(table_counts):6.1f}")
        print(f"  Min:    {np.min(table_counts):6.0f}")
        print(f"  Max:    {np.max(table_counts):6.0f}")

    if column_counts:
        print("\nColumns per sample:")
        print(f"  Mean:   {np.mean(column_counts):6.1f}")
        print(f"  Median: {np.median(column_counts):6.1f}")
        print(f"  Min:    {np.min(column_counts):6.0f}")
        print(f"  Max:    {np.max(column_counts):6.0f}")


def find_quality_issues(dataset):
    """Identify potential data quality issues."""
    print("\n" + "=" * 80)
    print("POTENTIAL QUALITY ISSUES")
    print("=" * 80)

    issues_found = False

    # Check for invalid samples
    if 'is_valid' in dataset.column_names:
        invalid_count = sum(1 for v in dataset['is_valid'] if not v)
        if invalid_count > 0:
            pct = (invalid_count / len(dataset)) * 100
            print(f"\n⚠ Invalid samples: {invalid_count} ({pct:.1f}%)")
            issues_found = True

    # Check for empty fields
    for field in ['question', 'sql', 'schema']:
        if field in dataset.column_names:
            empty_count = sum(1 for v in dataset[field] if not v or not v.strip())
            if empty_count > 0:
                pct = (empty_count / len(dataset)) * 100
                print(f"\n⚠ Empty {field}: {empty_count} ({pct:.1f}%)")
                issues_found = True

    # Check for very short questions
    if 'question_length' in dataset.column_names:
        very_short = sum(1 for v in dataset['question_length'] if v < 3)
        if very_short > 0:
            pct = (very_short / len(dataset)) * 100
            print(f"\n⚠ Very short questions (<3 words): {very_short} ({pct:.1f}%)")
            issues_found = True

    # Check for very long SQL
    if 'sql_length' in dataset.column_names:
        very_long = sum(1 for v in dataset['sql_length'] if v > 100)
        if very_long > 0:
            pct = (very_long / len(dataset)) * 100
            print(f"\n⚠ Very long SQL queries (>100 words): {very_long} ({pct:.1f}%)")
            issues_found = True

    if not issues_found:
        print("\n✓ No major quality issues detected!")


def show_examples(dataset, n_per_complexity=2):
    """Show example queries for each complexity level."""
    print("\n" + "=" * 80)
    print("EXAMPLE QUERIES BY COMPLEXITY")
    print("=" * 80)

    if 'complexity' not in dataset.column_names:
        print("No complexity information available")
        return

    required_fields = ['question', 'sql', 'complexity']
    if not all(field in dataset.column_names for field in required_fields):
        print("Missing required fields for examples")
        return

    for complexity in ['simple', 'medium', 'complex']:
        print(f"\n{'-' * 80}")
        print(f"{complexity.upper()} QUERIES")
        print(f"{'-' * 80}")

        # Find examples of this complexity
        indices = [i for i, c in enumerate(dataset['complexity']) if c == complexity]

        if not indices:
            print(f"No {complexity} examples found")
            continue

        # Sample randomly
        sample_indices = np.random.choice(
            indices,
            min(n_per_complexity, len(indices)),
            replace=False
        )

        for idx, sample_idx in enumerate(sample_indices, 1):
            sample = dataset[int(sample_idx)]
            print(f"\nExample {idx}:")
            print(f"Question: {sample['question']}")
            print(f"SQL: {sample['sql']}")

            if 'schema' in sample and sample['schema']:
                # Show first table definition only
                schema = sample['schema']
                first_table = schema.split('CREATE TABLE')[1] if 'CREATE TABLE' in schema else schema
                first_table = 'CREATE TABLE' + first_table.split(';')[0]
                if len(first_table) > 200:
                    first_table = first_table[:200] + "..."
                print(f"Schema: {first_table}")


def main():
    """Main analysis function."""
    # Setup logging
    logger = setup_logging(log_level="INFO", log_dir="logs", log_file="data_analysis.log")

    # Check for processed dataset
    processed_path = Path("./data_cache/processed")

    if not processed_path.exists():
        print("\n" + "=" * 80)
        print("ERROR: Processed dataset not found!")
        print("=" * 80)
        print(f"\nExpected location: {processed_path}")
        print("\nPlease run 'python scripts/prepare_data.py' first to prepare the dataset.")
        return

    print("\n" + "=" * 80)
    print("DATASET QUALITY ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nDataset location: {processed_path}")

    # Load dataset
    try:
        dataset_dict = load_from_disk(str(processed_path))
        print(f"Loaded splits: {list(dataset_dict.keys())}")
    except Exception as e:
        print(f"\n❌ Failed to load dataset: {e}")
        return

    # Analyze each split
    for split_name in dataset_dict.keys():
        print("\n" + "=" * 80)
        print(f"ANALYZING {split_name.upper()} SPLIT")
        print("=" * 80)
        print(f"Total samples: {len(dataset_dict[split_name])}")

        dataset = dataset_dict[split_name]

        # Run analyses
        analyze_complexity_distribution(dataset)
        analyze_sql_patterns(dataset)
        analyze_length_distribution(dataset)
        analyze_schema_statistics(dataset)
        find_quality_issues(dataset)
        show_examples(dataset, n_per_complexity=2)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
