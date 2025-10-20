"""SQL Validation Rubric for scoring SQL query generation quality.

This module provides a rubric for evaluating SQL outputs from language models
during GRPO fine-tuning. It scores queries based on syntax validity, keyword
presence, and format quality.
"""

import logging
from typing import Dict, List, Optional, Tuple

import sqlparse
from sqlparse.exceptions import SQLParseError

from utils.sql_parser import SQLParser

logger = logging.getLogger(__name__)


class SQLValidationRubric:
    """Rubric for scoring SQL query generation quality.

    This rubric evaluates SQL outputs on three main criteria:
    1. Syntax validity (using sqlparse)
    2. Keyword presence (checking for essential SQL keywords)
    3. Format quality (proper extraction and structure)

    Returns scores between 0.0 and 1.0, with higher scores indicating
    better quality SQL generation.

    Examples:
        >>> rubric = SQLValidationRubric()
        >>> score = rubric.score("SELECT * FROM users WHERE id = 1")
        >>> print(f"Score: {score:.2f}")
        Score: 1.00

        >>> rubric = SQLValidationRubric()
        >>> score = rubric.score("This is not SQL")
        >>> print(f"Score: {score:.2f}")
        Score: 0.00
    """

    def __init__(
        self,
        sql_keywords: Optional[List[str]] = None,
        syntax_weight: float = 0.4,
        keyword_weight: float = 0.3,
        format_weight: float = 0.3,
        parser: Optional[SQLParser] = None,
        strict_mode: bool = False,
        normalize_sql: bool = True,
    ):
        """Initialize the SQL validation rubric.

        Args:
            sql_keywords: List of SQL keywords to check for (uses defaults if None)
            syntax_weight: Weight for syntax validity score (0.0-1.0)
            keyword_weight: Weight for keyword presence score (0.0-1.0)
            format_weight: Weight for format quality score (0.0-1.0)
            parser: Optional SQLParser instance (creates default if None)
            strict_mode: If True, syntax errors return 0.0 immediately
            normalize_sql: Whether to normalize SQL before validation
        """
        # Default SQL keywords if not provided
        if sql_keywords is None:
            sql_keywords = [
                "SELECT",
                "FROM",
                "WHERE",
                "JOIN",
                "LEFT JOIN",
                "RIGHT JOIN",
                "INNER JOIN",
                "OUTER JOIN",
                "GROUP BY",
                "ORDER BY",
                "HAVING",
                "INSERT",
                "UPDATE",
                "DELETE",
                "CREATE",
                "DROP",
                "ALTER",
                "LIMIT",
                "OFFSET",
                "UNION",
                "DISTINCT",
                "AS",
                "ON",
            ]

        # Filter out non-string values (e.g., boolean true/false from config parsing)
        # and convert to uppercase
        self.sql_keywords = [kw.upper() for kw in sql_keywords if isinstance(kw, str)]

        # Validate weights sum to 1.0
        total_weight = syntax_weight + keyword_weight + format_weight
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight} "
                f"({syntax_weight} + {keyword_weight} + {format_weight})"
            )

        self.syntax_weight = syntax_weight
        self.keyword_weight = keyword_weight
        self.format_weight = format_weight

        self.parser = parser if parser is not None else SQLParser()
        self.strict_mode = strict_mode
        self.normalize_sql = normalize_sql

    def score(self, output: str, reference: Optional[str] = None) -> float:
        """Compute reward score for SQL output.

        Args:
            output: Generated SQL output from the model
            reference: Optional reference SQL (not currently used)

        Returns:
            Score between 0.0 and 1.0
        """
        if not output or not isinstance(output, str):
            return 0.0

        # Extract SQL from output
        sql = self.parser.extract_sql(output)

        if sql is None:
            logger.debug("Failed to extract SQL from output")
            return 0.0

        # Check format quality first
        format_score = self.check_format(output)

        # Check syntax
        is_valid, syntax_score = self.check_syntax(sql)

        if self.strict_mode and not is_valid:
            return 0.0

        # Check keywords
        keyword_score = self.check_keywords(sql)

        # Compute weighted score
        total_score = (
            self.syntax_weight * syntax_score
            + self.keyword_weight * keyword_score
            + self.format_weight * format_score
        )

        # Ensure score is in valid range
        return max(0.0, min(1.0, total_score))

    def _has_meaningful_tokens(self, statement) -> bool:
        """Check if statement has at least one meaningful token."""
        meaningful_tokens = [t for t in statement.tokens if not t.is_whitespace and str(t).strip()]
        return len(meaningful_tokens) > 0

    def _check_syntax_error_patterns(self, sql: str) -> Tuple[bool, float]:
        """Check for common syntax error patterns."""
        sql_upper = sql.upper()
        error_patterns = [
            "FORM ",  # Common typo of FROM
            "SLECT ",  # Common typo of SELECT
            "WEHERE ",  # Common typo of WHERE
        ]
        for pattern in error_patterns:
            if pattern in sql_upper:
                return False, 0.15  # Minimal partial credit for attempt
        return True, 1.0

    def check_syntax(self, sql: str) -> Tuple[bool, float]:
        """Validate SQL syntax using sqlparse.

        Args:
            sql: SQL query string

        Returns:
            Tuple of (is_valid, score) where:
                - is_valid: True if SQL parses without errors
                - score: Float between 0.0 and 1.0
        """
        if not sql or not isinstance(sql, str):
            return False, 0.0

        try:
            if self.normalize_sql:
                sql = sql.strip()

            parsed = sqlparse.parse(sql)
            if not parsed:
                return False, 0.0

            statement = parsed[0]
            if not statement.tokens:
                return False, 0.0

            if not self._has_meaningful_tokens(statement):
                return False, 0.0

            return self._check_syntax_error_patterns(sql)

        except SQLParseError as e:
            logger.debug(f"SQL parse error: {e}")
            return False, 0.0
        except Exception as e:
            logger.debug(f"Unexpected error parsing SQL: {e}")
            return False, 0.0

    def _find_keywords_in_sql(self, sql_upper: str) -> list:
        """Find all matching keywords in the SQL string."""
        import re

        keywords_found = []
        for keyword in self.sql_keywords:
            if keyword in sql_upper:
                if " " in keyword:
                    # Multi-word keywords (e.g., "LEFT JOIN")
                    if keyword in sql_upper:
                        keywords_found.append(keyword)
                else:
                    # Single-word keywords with word boundary check
                    pattern = r"\b" + re.escape(keyword) + r"\b"
                    if re.search(pattern, sql_upper):
                        keywords_found.append(keyword)
        return keywords_found

    def _score_keyword_diversity(self, keywords_found: list, sql_upper: str) -> float:
        """Score based on keyword diversity."""
        if not keywords_found:
            if any(kw in sql_upper for kw in ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE"]):
                return 0.3
            return 0.0

        num_keywords = len(keywords_found)
        if num_keywords >= 5:
            return 1.0
        elif num_keywords >= 3:
            return 0.75
        elif num_keywords >= 2:
            return 0.6
        else:
            return 0.5

    def check_keywords(self, sql: str) -> float:
        """Score based on SQL keyword presence.

        Args:
            sql: SQL query string

        Returns:
            Score between 0.0 and 1.0 based on keyword presence
        """
        if not sql or not isinstance(sql, str):
            return 0.0

        sql_upper = sql.upper()
        keywords_found = self._find_keywords_in_sql(sql_upper)
        return self._score_keyword_diversity(keywords_found, sql_upper)

    def check_format(self, output: str) -> float:
        """Score output format quality.

        Args:
            output: Raw output from model

        Returns:
            Score between 0.0 and 1.0
        """
        if not output or not isinstance(output, str):
            return 0.0

        output = output.strip()

        # Check length constraints
        if len(output) < self.parser.min_sql_length:
            return 0.0

        if len(output) > self.parser.max_sql_length * 2:
            return 0.5  # Too long, but might contain valid SQL

        # Check if SQL can be extracted
        sql = self.parser.extract_sql(output)

        if sql is None:
            return 0.0

        # Score based on extraction quality
        score = 0.0

        # 1. SQL was successfully extracted
        score += 0.4

        # 2. SQL length is reasonable
        sql_len = len(sql)
        if self.parser.min_sql_length <= sql_len <= self.parser.max_sql_length:
            score += 0.3
        elif sql_len > self.parser.max_sql_length:
            score += 0.15  # Partial credit for long queries
        else:
            score += 0.1  # Minimal credit for short queries

        # 3. SQL is not truncated (doesn't end mid-word)
        if sql and not sql.endswith(("...", "\u2026")):  # Regular and Unicode ellipsis
            score += 0.15
        else:
            score += 0.05

        # 4. SQL has proper structure (has both query type and table reference)
        sql_upper = sql.upper()
        has_query_type = any(
            kw in sql_upper for kw in ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE"]
        )
        has_table_ref = "FROM" in sql_upper or "INTO" in sql_upper or "TABLE" in sql_upper

        if has_query_type and has_table_ref:
            score += 0.15
        elif has_query_type or has_table_ref:
            score += 0.075

        return min(1.0, score)

    def get_detailed_scores(self, output: str) -> Dict[str, float]:
        """Return breakdown of scoring components.

        Args:
            output: Generated SQL output

        Returns:
            Dictionary with detailed scores for each component
        """
        if not output or not isinstance(output, str):
            return {
                "total": 0.0,
                "syntax": 0.0,
                "keywords": 0.0,
                "format": 0.0,
                "extracted_sql": None,  # type: ignore[dict-item]
            }

        # Extract SQL
        sql = self.parser.extract_sql(output)

        # Get component scores
        format_score = self.check_format(output)

        if sql is None:
            return {
                "total": 0.0,
                "syntax": 0.0,
                "keywords": 0.0,
                "format": format_score,
                "extracted_sql": None,  # type: ignore[dict-item]
            }

        is_valid, syntax_score = self.check_syntax(sql)
        keyword_score = self.check_keywords(sql)

        # Compute total
        total_score = (
            self.syntax_weight * syntax_score
            + self.keyword_weight * keyword_score
            + self.format_weight * format_score
        )

        return {
            "total": max(0.0, min(1.0, total_score)),
            "syntax": syntax_score,
            "syntax_valid": is_valid,
            "keywords": keyword_score,
            "format": format_score,
            "extracted_sql": sql,
            "weights": {  # type: ignore[dict-item]
                "syntax": self.syntax_weight,
                "keywords": self.keyword_weight,
                "format": self.format_weight,
            },
        }

    def score_batch(
        self, outputs: List[str], references: Optional[List[str]] = None
    ) -> List[float]:
        """Score multiple outputs.

        Args:
            outputs: List of generated SQL outputs
            references: Optional list of reference SQLs (not currently used)

        Returns:
            List of scores
        """
        return [self.score(output) for output in outputs]
