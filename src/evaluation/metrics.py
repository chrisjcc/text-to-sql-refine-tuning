"""
Comprehensive SQL evaluation metrics.
"""

import logging
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, List

import numpy as np
import sqlparse


class SQLMetrics:
    """
    Comprehensive SQL evaluation metrics.
    """

    def __init__(self):
        """Initialize metrics calculator."""
        self.logger = logging.getLogger(__name__)

    def exact_match(self, predicted: str, reference: str) -> bool:
        """
        Exact match after normalization.

        Args:
            predicted: Predicted SQL query
            reference: Reference SQL query

        Returns:
            True if queries match exactly
        """
        pred_norm = self._normalize_sql(predicted)
        ref_norm = self._normalize_sql(reference)
        return pred_norm == ref_norm

    def token_level_accuracy(self, predicted: str, reference: str) -> float:
        """
        Token-level accuracy (proportion of matching tokens).

        Args:
            predicted: Predicted SQL query
            reference: Reference SQL query

        Returns:
            Accuracy score [0.0, 1.0]
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
        """
        Measure structural similarity of SQL queries.
        Compares clauses, joins, aggregations, etc.

        Args:
            predicted: Predicted SQL query
            reference: Reference SQL query

        Returns:
            Similarity score [0.0, 1.0]
        """
        pred_struct = self._extract_structure(predicted)
        ref_struct = self._extract_structure(reference)

        scores = []

        # Compare each structural component
        for key in ref_struct.keys():
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

    def keyword_f1(self, predicted: str, reference: str) -> Dict[str, float]:
        """
        F1 score for SQL keywords (SELECT, FROM, WHERE, etc.).

        Args:
            predicted: Predicted SQL query
            reference: Reference SQL query

        Returns:
            Dict with precision, recall, f1
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
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def complexity_score(self, sql: str) -> Dict[str, Any]:
        """
        Analyze SQL query complexity.

        Args:
            sql: SQL query

        Returns:
            Dict with complexity metrics
        """
        parsed = sqlparse.parse(sql)

        if not parsed:
            return {"complexity": "invalid", "score": 0}

        num_tokens = len(self._tokenize_sql(sql))
        num_tables = self._count_tables(sql)
        num_joins = sql.upper().count("JOIN")
        has_subquery = (
            "SELECT" in sql.upper()[sql.upper().find("FROM") :] if "FROM" in sql.upper() else False
        )
        has_aggregation = any(agg in sql.upper() for agg in ["SUM", "COUNT", "AVG", "MAX", "MIN"])
        has_group_by = "GROUP BY" in sql.upper()
        has_order_by = "ORDER BY" in sql.upper()
        has_having = "HAVING" in sql.upper()
        num_conditions = (
            sql.upper().count("WHERE") + sql.upper().count("AND") + sql.upper().count("OR")
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

        complexity = {
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

        return complexity

    def edit_distance(self, predicted: str, reference: str) -> int:
        """
        Levenshtein edit distance between normalized SQL queries.

        Args:
            predicted: Predicted SQL query
            reference: Reference SQL query

        Returns:
            Edit distance (lower is better)
        """
        pred_norm = self._normalize_sql(predicted)
        ref_norm = self._normalize_sql(reference)

        # Use difflib for similarity
        return len(ref_norm) - int(
            SequenceMatcher(None, pred_norm, ref_norm).ratio() * len(ref_norm)
        )

    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL query for comparison."""
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

    def _tokenize_sql(self, sql: str) -> List[str]:
        """Tokenize SQL query."""
        parsed = sqlparse.parse(sql)
        if not parsed:
            return []  # type: ignore[no-any-return]

        tokens = []
        for statement in parsed:
            for token in statement.flatten():
                if not token.is_whitespace:
                    tokens.append(token.value.upper())

        return tokens

    def _extract_structure(self, sql: str) -> Dict[str, set]:
        """Extract structural components of SQL query."""
        sql_upper = sql.upper()

        structure: Dict[str, set] = {
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
            select_part = sql_upper.split("FROM")[0].replace("SELECT", "").strip()
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
        structure["aggregations"] = set(agg for agg in aggs if agg in sql_upper)

        return structure

    def _extract_keywords(self, sql: str) -> set:
        """Extract SQL keywords."""
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
        return set(kw for kw in keywords if kw in sql_upper)

    def _count_tables(self, sql: str) -> int:
        """Count number of tables in query."""
        tokens = self._tokenize_sql(sql)
        count = 0
        for i, token in enumerate(tokens):
            if token in ["FROM", "JOIN"] and i + 1 < len(tokens):
                count += 1
        return count


class ExecutionMetrics:
    """
    Metrics based on SQL execution (requires database connection).
    """

    def __init__(self, db_connection=None):
        """
        Initialize execution metrics.

        Args:
            db_connection: Database connection for executing queries
        """
        self.db_connection = db_connection
        self.logger = logging.getLogger(__name__)

    def execution_accuracy(
        self, predicted: str, reference: str, timeout: int = 5
    ) -> Dict[str, Any]:
        """
        Check if predicted SQL produces same results as reference.

        Args:
            predicted: Predicted SQL query
            reference: Reference SQL query
            timeout: Query timeout in seconds

        Returns:
            Dict with execution_match, error info
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
                execution_match = self._compare_results(pred_result["data"], ref_result["data"])
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

    def _execute_query(self, sql: str, timeout: int) -> Dict:
        """Execute SQL query and return results."""
        # Implementation depends on database type
        # This is a placeholder
        raise NotImplementedError("Database execution not implemented")

    def _compare_results(self, result1, result2) -> bool:
        """Compare query results."""
        # Implementation depends on result format
        # This is a placeholder
        raise NotImplementedError("Result comparison not implemented")
