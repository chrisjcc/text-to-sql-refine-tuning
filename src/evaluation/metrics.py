"""Comprehensive SQL evaluation metrics.

This module provides a comprehensive set of metrics for evaluating SQL
generation quality, including exact match, token accuracy, structural
similarity, and optional execution-based metrics.
"""

import logging
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import sqlparse


class SQLMetrics:
    """Comprehensive SQL evaluation metrics.

    Provides various metrics for evaluating SQL query quality without
    requiring database execution, including exact match, token-level
    accuracy, structural similarity, and complexity analysis.

    Attributes:
        logger: Logger instance for this class.
    """

    def __init__(self) -> None:
        """Initialize metrics calculator."""
        self.logger = logging.getLogger(__name__)

    def exact_match(self, predicted: str, reference: str) -> bool:
        """Exact match after normalization.

        Args:
            predicted: Predicted SQL query.
            reference: Reference SQL query.

        Returns:
            True if queries match exactly after normalization.
        """
        pred_norm = self._normalize_sql(predicted)
        ref_norm = self._normalize_sql(reference)
        return pred_norm == ref_norm

    def token_level_accuracy(self, predicted: str, reference: str) -> float:
        """Token-level accuracy (proportion of matching tokens).

        Computes accuracy based on token overlap, treating tokens as
        unordered sets.

        Args:
            predicted: Predicted SQL query.
            reference: Reference SQL query.

        Returns:
            Accuracy score in range [0.0, 1.0].
        """
        pred_tokens = self._tokenize_sql(predicted)
        ref_tokens = self._tokenize_sql(reference)

        if len(ref_tokens) == 0:
            return 0.0

        # Count matching tokens (order-independent)
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)

        matching = sum((pred_counter & ref_counter).values())
        total = len(ref_tokens)

        return matching / total

    def structural_similarity(self, predicted: str, reference: str) -> float:
        """Measure structural similarity of SQL queries.

        Compares structural components like clauses, joins, and
        aggregations using set-based similarity.

        Args:
            predicted: Predicted SQL query.
            reference: Reference SQL query.

        Returns:
            Similarity score in range [0.0, 1.0].
        """
        pred_struct = self._extract_structure(predicted)
        ref_struct = self._extract_structure(reference)

        scores = []

        # Compare each structural component
        for key in ref_struct:
            pred_val = pred_struct.get(key, set())
            ref_val = ref_struct.get(key, set())

            if len(ref_val) == 0:
                continue

            # Jaccard similarity for sets
            intersection = len(pred_val & ref_val)
            union = len(pred_val | ref_val)

            if union > 0:
                scores.append(intersection / union)

        return float(np.mean(scores)) if scores else 0.0

    def keyword_f1(
        self, predicted: str, reference: str
    ) -> dict[str, float]:
        """F1 score for SQL keywords (SELECT, FROM, WHERE, etc.).

        Args:
            predicted: Predicted SQL query.
            reference: Reference SQL query.

        Returns:
            Dictionary with 'precision', 'recall', and 'f1' scores.
        """
        pred_keywords = self._extract_keywords(predicted)
        ref_keywords = self._extract_keywords(reference)

        if len(ref_keywords) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        true_positives = len(pred_keywords & ref_keywords)
        false_positives = len(pred_keywords - ref_keywords)
        false_negatives = len(ref_keywords - pred_keywords)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1_denom = precision + recall
        f1 = 2 * (precision * recall) / f1_denom if f1_denom > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def complexity_score(self, sql: str) -> dict[str, Any]:
        """Analyze SQL query complexity.

        Computes a complexity score based on various SQL features like
        joins, subqueries, aggregations, and conditions.

        Args:
            sql: SQL query to analyze.

        Returns:
            Dictionary with complexity metrics including:
            - num_tokens: Number of tokens in query
            - num_tables: Number of tables referenced
            - num_joins: Number of JOIN clauses
            - has_subquery: Whether query contains subqueries
            - has_aggregation: Whether query uses aggregation functions
            - complexity_level: Classification (simple/medium/complex)
            - complexity_score: Numerical complexity score
        """
        parsed = sqlparse.parse(sql)

        if not parsed:
            return {"complexity": "invalid", "score": 0}

        num_tokens = len(self._tokenize_sql(sql))
        num_tables = self._count_tables(sql)
        num_joins = sql.upper().count("JOIN")

        sql_upper = sql.upper()
        from_pos = sql_upper.find("FROM")
        has_subquery = (
            "SELECT" in sql_upper[from_pos:]
            if "FROM" in sql_upper
            else False
        )

        agg_functions = ["SUM", "COUNT", "AVG", "MAX", "MIN"]
        has_aggregation = any(agg in sql_upper for agg in agg_functions)

        has_group_by = "GROUP BY" in sql_upper
        has_order_by = "ORDER BY" in sql_upper
        has_having = "HAVING" in sql_upper
        num_conditions = (
            sql_upper.count("WHERE")
            + sql_upper.count("AND")
            + sql_upper.count("OR")
        )

        # Compute overall complexity score
        score = (
            num_tokens * 0.1
            + num_tables * 5
            + num_joins * 10
            + (20 if has_subquery else 0)
            + (10 if has_aggregation else 0)
            + (10 if has_group_by else 0)
            + num_conditions * 3
        )

        if score < 20:
            complexity_level = "simple"
        elif score < 50:
            complexity_level = "medium"
        else:
            complexity_level = "complex"

        return {
            "num_tokens": num_tokens,
            "num_tables": num_tables,
            "num_joins": num_joins,
            "has_subquery": has_subquery,
            "has_aggregation": has_aggregation,
            "has_group_by": has_group_by,
            "has_order_by": has_order_by,
            "has_having": has_having,
            "num_conditions": num_conditions,
            "complexity_level": str(complexity_level),
            "complexity_score": float(score),
        }

    def edit_distance(self, predicted: str, reference: str) -> int:
        """Levenshtein edit distance between normalized SQL queries.

        Args:
            predicted: Predicted SQL query.
            reference: Reference SQL query.

        Returns:
            Edit distance (lower is better).
        """
        pred_norm = self._normalize_sql(predicted)
        ref_norm = self._normalize_sql(reference)

        # Use difflib for similarity
        similarity_ratio = SequenceMatcher(
            None, pred_norm, ref_norm
        ).ratio()
        return len(ref_norm) - int(similarity_ratio * len(ref_norm))

    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL query for comparison.

        Args:
            sql: Raw SQL query.

        Returns:
            Normalized SQL string.
        """
        import re

        # First use sqlparse for basic normalization
        normalized = str(
            sqlparse.format(
                sql,
                keyword_case="upper",
                identifier_case="lower",
                strip_whitespace=True,
                reindent=False,
            )
        ).strip()

        # Additional normalization: standardize spacing around operators
        # Add spaces around = , < , > , != , <= , >= operators
        normalized = re.sub(r"\s*=\s*", " = ", normalized)
        normalized = re.sub(r"\s*!=\s*", " != ", normalized)
        normalized = re.sub(r"\s*<>\s*", " <> ", normalized)
        normalized = re.sub(r"\s*<=\s*", " <= ", normalized)
        normalized = re.sub(r"\s*>=\s*", " >= ", normalized)
        normalized = re.sub(r"\s*<\s*", " < ", normalized)
        normalized = re.sub(r"\s*>\s*", " > ", normalized)

        # Remove extra spaces
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized.strip()

    def _tokenize_sql(self, sql: str) -> list[str]:
        """Tokenize SQL query.

        Args:
            sql: SQL query to tokenize.

        Returns:
            List of uppercase tokens.
        """
        parsed = sqlparse.parse(sql)
        if not parsed:
            return []  # type: ignore[no-any-return]

        tokens = []
        for statement in parsed:
            for token in statement.flatten():
                if not token.is_whitespace:
                    tokens.append(token.value.upper())

        return tokens

    def _extract_structure(self, sql: str) -> dict[str, set[str]]:
        """Extract structural components of SQL query.

        Args:
            sql: SQL query to analyze.

        Returns:
            Dictionary mapping component names to sets of values.
        """
        sql_upper = sql.upper()

        structure: dict[str, set[str]] = {
            "select_columns": set(),
            "from_tables": set(),
            "join_tables": set(),
            "where_conditions": set(),
            "aggregations": set(),
            "group_by": set(),
            "order_by": set(),
        }

        # Extract SELECT columns (simplified)
        if "SELECT" in sql_upper:
            select_part = (
                sql_upper.split("FROM")[0].replace("SELECT", "").strip()
            )
            structure["select_columns"] = set(select_part.split(","))

        # Extract table names (simplified)
        tokens = self._tokenize_sql(sql)
        for i, token in enumerate(tokens):
            if token == "FROM" and i + 1 < len(tokens):
                structure["from_tables"].add(tokens[i + 1])
            if token == "JOIN" and i + 1 < len(tokens):
                structure["join_tables"].add(tokens[i + 1])

        # Extract aggregations
        aggs = ["SUM", "COUNT", "AVG", "MAX", "MIN"]
        structure["aggregations"] = {agg for agg in aggs if agg in sql_upper}

        return structure

    def _extract_keywords(self, sql: str) -> set[str]:
        """Extract SQL keywords.

        Args:
            sql: SQL query.

        Returns:
            Set of SQL keywords found in the query.
        """
        keywords = [
            "SELECT",
            "FROM",
            "WHERE",
            "JOIN",
            "LEFT",
            "RIGHT",
            "INNER",
            "OUTER",
            "ON",
            "GROUP BY",
            "HAVING",
            "ORDER BY",
            "LIMIT",
            "OFFSET",
            "INSERT",
            "UPDATE",
            "DELETE",
            "CREATE",
            "DROP",
            "ALTER",
            "UNION",
            "INTERSECT",
            "EXCEPT",
            "DISTINCT",
            "AS",
        ]

        sql_upper = sql.upper()
        return {kw for kw in keywords if kw in sql_upper}

    def _count_tables(self, sql: str) -> int:
        """Count number of tables in query.

        Args:
            sql: SQL query.

        Returns:
            Number of tables referenced in FROM and JOIN clauses.
        """
        tokens = self._tokenize_sql(sql)
        count = 0
        for i, token in enumerate(tokens):
            if token in ["FROM", "JOIN"] and i + 1 < len(tokens):
                count += 1
        return count


class ExecutionMetrics:
    """Metrics based on SQL execution (requires database connection).

    Provides execution-based evaluation by running queries against a
    database and comparing results.

    Attributes:
        db_connection: Database connection for executing queries.
        logger: Logger instance for this class.
    """

    def __init__(self, db_connection: Any = None) -> None:
        """Initialize execution metrics.

        Args:
            db_connection: Database connection for executing queries.
                If None, execution methods will return error results.
        """
        self.db_connection = db_connection
        self.logger = logging.getLogger(__name__)

    def execution_accuracy(
        self, predicted: str, reference: str, timeout: int = 5
    ) -> dict[str, Any]:
        """Check if predicted SQL produces same results as reference.

        Executes both queries and compares their results.

        Args:
            predicted: Predicted SQL query.
            reference: Reference SQL query.
            timeout: Query timeout in seconds. Defaults to 5.

        Returns:
            Dictionary containing:
            - execution_match: Whether results match (or None if error)
            - predicted_executable: Whether predicted query executed
            - reference_executable: Whether reference query executed
            - error: Error message if any
        """
        if self.db_connection is None:
            return {
                "execution_match": None,
                "error": "No database connection",
                "predicted_executable": None,
                "reference_executable": None,
            }

        try:
            # Execute reference query
            ref_result = self._execute_query(reference, timeout)
            ref_executable = ref_result["success"]

            # Execute predicted query
            pred_result = self._execute_query(predicted, timeout)
            pred_executable = pred_result["success"]

            # Compare results if both executed
            if ref_executable and pred_executable:
                execution_match = self._compare_results(
                    pred_result["data"], ref_result["data"]
                )
            else:
                execution_match = False

            return {
                "execution_match": execution_match,
                "predicted_executable": pred_executable,
                "reference_executable": ref_executable,
                "predicted_error": pred_result.get("error"),
                "reference_error": ref_result.get("error"),
            }

        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            return {
                "execution_match": False,
                "error": str(e),
                "predicted_executable": None,
                "reference_executable": None,
            }

    def _execute_query(
        self, sql: str, timeout: int
    ) -> dict[str, Any]:
        """Execute SQL query and return results.

        Args:
            sql: SQL query to execute.
            timeout: Query timeout in seconds.

        Returns:
            Dictionary with execution results.

        Raises:
            NotImplementedError: This is a placeholder method that must
                be implemented for specific database types.
        """
        # Implementation depends on database type
        # This is a placeholder
        raise NotImplementedError("Database execution not implemented")

    def _compare_results(self, result1: Any, result2: Any) -> bool:
        """Compare query results.

        Args:
            result1: First query result.
            result2: Second query result.

        Returns:
            True if results match.

        Raises:
            NotImplementedError: This is a placeholder method that must
                be implemented for specific result formats.
        """
        # Implementation depends on result format
        # This is a placeholder
        raise NotImplementedError("Result comparison not implemented")
