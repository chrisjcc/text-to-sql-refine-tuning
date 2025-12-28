"""Data preprocessor for SQL datasets.

This module provides preprocessing utilities for cleaning and validating
SQL dataset samples, including question cleaning, schema formatting,
and SQL normalization.
"""

import logging
import re
from typing import Any

import sqlparse
from datasets import Dataset

from src.utils.sql_parser import SQLParser

logger = logging.getLogger(__name__)


class SQLDataPreprocessor:
    """Preprocesses SQL dataset samples for training.

    Handles cleaning, validation, and formatting of SQL dataset samples
    including question text, database schemas, and SQL queries.

    Attributes:
        max_question_length: Maximum tokens for question.
        max_schema_length: Maximum tokens for schema context.
        max_sql_length: Maximum tokens for SQL query.
        normalize_sql: Whether to normalize SQL formatting.
        filter_invalid: Whether to filter out invalid samples.
        parser: SQL parser for syntax validation.
        logger: Logger instance for this class.
    """

    def __init__(
        self,
        max_question_length: int = 512,
        max_schema_length: int = 1024,
        max_sql_length: int = 512,
        normalize_sql: bool = True,
        filter_invalid: bool = True,
    ) -> None:
        """Initialize preprocessor.

        Args:
            max_question_length: Maximum tokens for question. Defaults to 512.
            max_schema_length: Maximum tokens for schema context.
                Defaults to 1024.
            max_sql_length: Maximum tokens for SQL query. Defaults to 512.
            normalize_sql: Whether to normalize SQL formatting with sqlparse.
                Defaults to True.
            filter_invalid: Whether to filter out invalid samples.
                Defaults to True.
        """
        self.max_question_length = max_question_length
        self.max_schema_length = max_schema_length
        self.max_sql_length = max_sql_length
        self.normalize_sql = normalize_sql
        self.filter_invalid = filter_invalid
        self.parser = SQLParser()
        self.logger = logging.getLogger(__name__)

    def clean_question(self, question: str) -> str:
        """Clean and normalize natural language question.

        Performs the following operations:
        - Remove extra whitespace
        - Fix encoding issues (non-breaking spaces, quotation marks)
        - Standardize punctuation

        Args:
            question: Raw question text.

        Returns:
            Cleaned and normalized question string.
        """
        if not question:
            return ""

        # Remove extra whitespace
        question = re.sub(r"\s+", " ", question)

        # Strip leading/trailing whitespace
        question = question.strip()

        # Fix common encoding issues
        question = question.replace("\u00a0", " ")  # Non-breaking space
        question = question.replace("\u2019", "'")  # Right single quote
        question = question.replace("\u201c", '"')  # Left double quote
        question = question.replace("\u201d", '"')  # Right double quote

        # Ensure question ends with proper punctuation
        if question and question[-1] not in ".?!":
            question = question + "?"

        return question

    def clean_schema(self, context: str) -> str:
        """Clean and format schema (CREATE TABLE statements).

        Performs the following operations:
        - Remove SQL comments (single-line and multi-line)
        - Normalize whitespace formatting
        - Truncate if too long (at table boundaries when possible)

        Args:
            context: Raw schema text containing CREATE TABLE statements.

        Returns:
            Cleaned and normalized schema string.
        """
        if not context:
            return ""

        # Remove SQL comments
        # Remove single-line comments (-- ...)
        context = re.sub(r"--[^\n]*", "", context)
        # Remove multi-line comments (/* ... */)
        context = re.sub(r"/\*.*?\*/", "", context, flags=re.DOTALL)

        # Normalize whitespace
        context = re.sub(r"\s+", " ", context)
        context = context.strip()

        # Truncate if too long (by character count)
        # Use char count as proxy for token count (rough: 1 token ≈ 4 chars)
        max_chars = self.max_schema_length * 4
        if len(context) > max_chars:
            self.logger.debug(f"Truncating schema from {len(context)} to " f"{max_chars} chars")
            context = context[:max_chars]
            # Try to truncate at a table boundary
            last_create = context.rfind("CREATE TABLE")
            # If close to the end, truncate there
            if last_create > max_chars * 0.8:
                context = context[:last_create]

        return context

    def clean_sql(self, sql: str) -> str:
        """Clean and normalize SQL query.

        Performs the following operations:
        - Remove SQL comments
        - Standardize keywords (uppercase)
        - Normalize whitespace
        - Validate basic syntax with sqlparse

        Args:
            sql: Raw SQL query string.

        Returns:
            Cleaned and normalized SQL string.
        """
        if not sql:
            return ""

        # Remove SQL comments
        sql = re.sub(r"--[^\n]*", "", sql)
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)

        # Normalize whitespace
        sql = re.sub(r"\s+", " ", sql)
        sql = sql.strip()

        # Use sqlparse for normalization if enabled
        if self.normalize_sql:
            try:
                # Format SQL with sqlparse
                sql = sqlparse.format(
                    sql,
                    keyword_case="upper",
                    identifier_case="lower",
                    strip_comments=True,
                    reindent=False,
                    use_space_around_operators=True,
                )
                # Remove extra newlines that sqlparse might add
                sql = " ".join(sql.split())
            except Exception as e:
                self.logger.debug(f"Failed to parse SQL with sqlparse: {e}")
                # Fall back to basic normalization
                pass

        return sql

    def validate_sample(self, sample: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate that sample has required fields and valid content.

        Checks for:
        - Presence of required fields (question, context, answer)
        - Non-empty content
        - Valid SQL syntax using sqlparse

        Args:
            sample: Sample dictionary to validate.

        Returns:
            Tuple of (is_valid, error_message). error_message is None
            if valid.
        """
        # Check required fields
        required_fields = ["question", "context", "answer"]
        for field in required_fields:
            if field not in sample:
                return False, f"Missing required field: {field}"

        # Check non-empty
        if not sample["question"] or not sample["question"].strip():
            return False, "Empty question"

        if not sample["answer"] or not sample["answer"].strip():
            return False, "Empty answer (SQL)"

        # Context can be empty for some datasets, but warn if it is
        if not sample["context"] or not sample["context"].strip():
            self.logger.debug("Warning: Empty context (schema)")

        # Validate SQL syntax using sqlparse
        try:
            parsed = sqlparse.parse(sample["answer"])
            if not parsed:
                return False, "Failed to parse SQL"

            # Check if it's a valid SQL statement
            if not any(stmt.get_type() for stmt in parsed):
                return False, "No valid SQL statement found"

        except Exception as e:
            return False, f"SQL parsing error: {str(e)}"

        return True, None

    def preprocess_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Preprocess a single sample.

        Takes raw sample and returns cleaned, validated, and enriched
        version with additional metadata fields.

        Args:
            sample: Input sample with expected format:
                - question: Natural language query string
                - context: CREATE TABLE statements
                - answer: SQL query string

        Returns:
            Preprocessed sample dictionary containing:
            - question: Cleaned question text
            - schema: Cleaned schema (renamed from context)
            - sql: Cleaned SQL query (renamed from answer)
            - question_length: Word count of question
            - schema_length: Word count of schema
            - sql_length: Word count of SQL
            - is_valid: Boolean validation status
            - validation_error: Error message if invalid, None otherwise
            - sql_keywords: List of SQL keywords present
            - complexity: Complexity classification (simple/medium/complex)
        """
        # Clean fields
        question = self.clean_question(sample.get("question", ""))
        schema = self.clean_schema(sample.get("context", ""))
        sql = self.clean_sql(sample.get("answer", ""))

        # Validate
        is_valid, error_msg = self.validate_sample(
            {"question": question, "context": schema, "answer": sql}
        )

        # Compute lengths (word count as proxy for tokens)
        question_length = len(question.split())
        schema_length = len(schema.split())
        sql_length = len(sql.split())

        # Extract SQL keywords
        sql_keywords = self._extract_sql_keywords(sql)

        # Classify complexity
        complexity = self.classify_complexity(sql)

        return {
            "question": question,
            "schema": schema,
            "sql": sql,
            "question_length": question_length,
            "schema_length": schema_length,
            "sql_length": sql_length,
            "is_valid": is_valid,
            "validation_error": error_msg if not is_valid else None,
            "sql_keywords": sql_keywords,
            "complexity": complexity,
        }

    def _extract_sql_keywords(self, sql: str) -> list[str]:
        """Extract SQL keywords from query.

        Args:
            sql: SQL query string.

        Returns:
            List of SQL keywords found in the query.
        """
        keywords = [
            "SELECT",
            "FROM",
            "WHERE",
            "JOIN",
            "INNER JOIN",
            "LEFT JOIN",
            "RIGHT JOIN",
            "OUTER JOIN",
            "GROUP BY",
            "ORDER BY",
            "HAVING",
            "LIMIT",
            "OFFSET",
            "UNION",
            "DISTINCT",
            "COUNT",
            "SUM",
            "AVG",
            "MAX",
            "MIN",
            "AS",
            "IN",
            "BETWEEN",
            "LIKE",
            "AND",
            "OR",
            "NOT",
        ]

        sql_upper = sql.upper()
        found_keywords = []

        for keyword in keywords:
            if keyword in sql_upper:
                found_keywords.append(keyword)

        return found_keywords

    def preprocess_dataset(self, dataset: Dataset, num_proc: int = 4) -> Dataset:
        """Preprocess the entire dataset in parallel.

        Applies preprocessing to all samples using parallel processing
        for improved performance.

        Args:
            dataset: Dataset to preprocess.
            num_proc: Number of processes for parallel preprocessing.
                Defaults to 4.

        Returns:
            Preprocessed dataset with cleaned and validated samples.
        """
        self.logger.info(f"Preprocessing dataset with {num_proc} processes")

        def preprocess_fn(examples: dict[str, Any]) -> dict[str, list[Any]]:
            """Batch preprocessing function."""
            results: dict[str, list[Any]] = {
                "question": [],
                "schema": [],
                "sql": [],
                "question_length": [],
                "schema_length": [],
                "sql_length": [],
                "is_valid": [],
                "validation_error": [],
                "sql_keywords": [],
                "complexity": [],
            }

            # Handle both batched and single examples
            if isinstance(examples["question"], list):
                num_examples = len(examples["question"])
            else:
                num_examples = 1
                examples = {k: [v] for k, v in examples.items()}

            for i in range(num_examples):
                sample = {
                    "question": examples["question"][i],
                    "context": examples["context"][i],
                    "answer": examples["answer"][i],
                }

                processed = self.preprocess_sample(sample)

                for key in results:
                    results[key].append(processed[key])

            return results

        processed_dataset = dataset.map(
            preprocess_fn,
            batched=True,
            num_proc=num_proc,
            desc="Preprocessing samples",
        )

        self.logger.info(f"Preprocessing complete: {len(processed_dataset)} samples")

        return processed_dataset

    def filter_dataset(self, dataset: Dataset) -> Dataset:
        """Filter out invalid or problematic samples.

        Applies multiple filters to remove:
        - Samples with missing required fields
        - Invalid SQL syntax
        - Questions/SQL exceeding length limits
        - Empty or malformed content

        Args:
            dataset: Dataset to filter.

        Returns:
            Filtered dataset containing only valid samples.
        """
        self.logger.info("Filtering invalid samples")

        initial_size = len(dataset)

        # Filter by validity
        filtered = dataset.filter(lambda x: x["is_valid"], desc="Filtering invalid samples")

        # Filter by length constraints
        filtered = filtered.filter(
            lambda x: (
                x["question_length"] <= self.max_question_length
                and x["sql_length"] <= self.max_sql_length
                and x["question_length"] > 0
                and x["sql_length"] > 0
            ),
            desc="Filtering by length",
        )

        final_size = len(filtered)
        removed = initial_size - final_size

        self.logger.info(
            f"Filtered {removed} samples "
            f"({removed/initial_size*100:.1f}%). "
            f"Remaining: {final_size}"
        )

        return filtered

    def classify_complexity(self, sql: str) -> str:
        """Classify SQL complexity based on features.

        Classification levels:
        - Simple: Basic SELECT with WHERE
        - Medium: JOINs, GROUP BY, or single subquery
        - Complex: Multiple JOINs, nested subqueries, or CTEs

        Args:
            sql: SQL query string to classify.

        Returns:
            Complexity level: "simple", "medium", or "complex".
        """
        sql_upper = sql.upper()

        # Count complexity indicators
        join_count = sql_upper.count("JOIN")
        subquery_count = sql_upper.count("SELECT") - 1  # Subtract main query
        has_group_by = "GROUP BY" in sql_upper
        has_having = "HAVING" in sql_upper
        has_union = "UNION" in sql_upper
        has_cte = "WITH" in sql_upper and "AS" in sql_upper

        # Complex: CTEs, multiple JOINs, or nested subqueries
        if has_cte or join_count >= 2 or subquery_count >= 2:
            return "complex"

        # Medium: Single JOIN, GROUP BY, HAVING, UNION, or single subquery
        conditions = [
            join_count >= 1,
            has_group_by,
            has_having,
            has_union,
            subquery_count >= 1,
        ]
        if any(conditions):
            return "medium"

        # Simple: Basic SELECT
        return "simple"
