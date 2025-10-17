"""Data preprocessor for SQL datasets.

This module provides preprocessing utilities for cleaning and validating
SQL dataset samples, including question cleaning, schema formatting,
and SQL normalization.
"""

from typing import Dict, List, Optional, Callable, Tuple, Any
import sqlparse
import re
import logging
from datasets import Dataset

from ..utils.sql_parser import SQLParser


logger = logging.getLogger(__name__)


class SQLDataPreprocessor:
    """
    Preprocesses SQL dataset samples for training.
    Handles cleaning, validation, and formatting.
    """

    def __init__(
        self,
        max_question_length: int = 512,
        max_schema_length: int = 1024,
        max_sql_length: int = 512,
        normalize_sql: bool = True,
        filter_invalid: bool = True
    ):
        """
        Initialize preprocessor.

        Args:
            max_question_length: Maximum tokens for question
            max_schema_length: Maximum tokens for schema context
            max_sql_length: Maximum tokens for SQL query
            normalize_sql: Whether to normalize SQL formatting
            filter_invalid: Whether to filter out invalid samples
        """
        self.max_question_length = max_question_length
        self.max_schema_length = max_schema_length
        self.max_sql_length = max_sql_length
        self.normalize_sql = normalize_sql
        self.filter_invalid = filter_invalid
        self.parser = SQLParser()
        self.logger = logging.getLogger(__name__)

    def clean_question(self, question: str) -> str:
        """
        Clean and normalize natural language question.

        - Remove extra whitespace
        - Fix encoding issues
        - Standardize punctuation
        """
        if not question:
            return ""

        # Remove extra whitespace
        question = re.sub(r'\s+', ' ', question)

        # Strip leading/trailing whitespace
        question = question.strip()

        # Fix common encoding issues
        question = question.replace('\u00a0', ' ')  # Non-breaking space
        question = question.replace('\u2019', "'")  # Right single quotation mark
        question = question.replace('\u201c', '"')  # Left double quotation mark
        question = question.replace('\u201d', '"')  # Right double quotation mark

        # Ensure question ends with proper punctuation
        if question and question[-1] not in '.?!':
            question = question + '?'

        return question

    def clean_schema(self, context: str) -> str:
        """
        Clean and format schema (CREATE TABLE statements).

        - Remove comments
        - Normalize formatting
        - Truncate if too long
        """
        if not context:
            return ""

        # Remove SQL comments
        # Remove single-line comments (-- ...)
        context = re.sub(r'--[^\n]*', '', context)
        # Remove multi-line comments (/* ... */)
        context = re.sub(r'/\*.*?\*/', '', context, flags=re.DOTALL)

        # Normalize whitespace
        context = re.sub(r'\s+', ' ', context)
        context = context.strip()

        # Truncate if too long (by character count)
        # We use character count as a proxy for token count (rough estimate: 1 token ≈ 4 chars)
        max_chars = self.max_schema_length * 4
        if len(context) > max_chars:
            self.logger.debug(f"Truncating schema from {len(context)} to {max_chars} chars")
            context = context[:max_chars]
            # Try to truncate at a table boundary
            last_create = context.rfind('CREATE TABLE')
            if last_create > max_chars * 0.8:  # If we're close to the end, truncate there
                context = context[:last_create]

        return context

    def clean_sql(self, sql: str) -> str:
        """
        Clean and normalize SQL query.

        - Remove extra whitespace
        - Standardize keywords (uppercase)
        - Remove comments
        - Validate basic syntax
        """
        if not sql:
            return ""

        # Remove SQL comments
        sql = re.sub(r'--[^\n]*', '', sql)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)

        # Normalize whitespace
        sql = re.sub(r'\s+', ' ', sql)
        sql = sql.strip()

        # Use sqlparse for normalization if enabled
        if self.normalize_sql:
            try:
                # Format SQL with sqlparse
                sql = sqlparse.format(
                    sql,
                    keyword_case='upper',
                    identifier_case='lower',
                    strip_comments=True,
                    reindent=False,
                    use_space_around_operators=True
                )
                # Remove extra newlines that sqlparse might add
                sql = ' '.join(sql.split())
            except Exception as e:
                self.logger.debug(f"Failed to parse SQL with sqlparse: {e}")
                # Fall back to basic normalization
                pass

        return sql

    def validate_sample(self, sample: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate that the sample has all required fields and valid content.

        Returns:
            (is_valid, error_message)
        """
        # Check required fields
        required_fields = ['question', 'context', 'answer']
        for field in required_fields:
            if field not in sample:
                return False, f"Missing required field: {field}"

        # Check non-empty
        if not sample['question'] or not sample['question'].strip():
            return False, "Empty question"

        if not sample['answer'] or not sample['answer'].strip():
            return False, "Empty answer (SQL)"

        # Context can be empty for some datasets, but warn if it is
        if not sample['context'] or not sample['context'].strip():
            self.logger.debug("Warning: Empty context (schema)")

        # Validate SQL syntax using sqlparse
        try:
            parsed = sqlparse.parse(sample['answer'])
            if not parsed:
                return False, "Failed to parse SQL"

            # Check if it's a valid SQL statement
            if not any(stmt.get_type() for stmt in parsed):
                return False, "No valid SQL statement found"

        except Exception as e:
            return False, f"SQL parsing error: {str(e)}"

        return True, None

    def preprocess_sample(self, sample: Dict) -> Dict:
        """
        Preprocess a single sample.

        Expected input format (b-mc2/sql-create-context):
        {
            'question': str,
            'context': str,  # CREATE TABLE statements
            'answer': str    # SQL query
        }

        Returns preprocessed sample with additional fields:
        {
            'question': str (cleaned),
            'schema': str (cleaned context),
            'sql': str (cleaned answer),
            'question_length': int,
            'schema_length': int,
            'sql_length': int,
            'is_valid': bool,
            'sql_keywords': List[str],
            'complexity': str  # 'simple', 'medium', 'complex'
        }
        """
        # Clean fields
        question = self.clean_question(sample.get('question', ''))
        schema = self.clean_schema(sample.get('context', ''))
        sql = self.clean_sql(sample.get('answer', ''))

        # Validate
        is_valid, error_msg = self.validate_sample({
            'question': question,
            'context': schema,
            'answer': sql
        })

        # Compute lengths (word count as proxy for tokens)
        question_length = len(question.split())
        schema_length = len(schema.split())
        sql_length = len(sql.split())

        # Extract SQL keywords
        sql_keywords = self._extract_sql_keywords(sql)

        # Classify complexity
        complexity = self.classify_complexity(sql)

        return {
            'question': question,
            'schema': schema,
            'sql': sql,
            'question_length': question_length,
            'schema_length': schema_length,
            'sql_length': sql_length,
            'is_valid': is_valid,
            'validation_error': error_msg if not is_valid else None,
            'sql_keywords': sql_keywords,
            'complexity': complexity,
        }

    def _extract_sql_keywords(self, sql: str) -> List[str]:
        """Extract SQL keywords from query."""
        keywords = [
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER JOIN', 'LEFT JOIN',
            'RIGHT JOIN', 'OUTER JOIN', 'GROUP BY', 'ORDER BY', 'HAVING',
            'LIMIT', 'OFFSET', 'UNION', 'DISTINCT', 'COUNT', 'SUM', 'AVG',
            'MAX', 'MIN', 'AS', 'IN', 'BETWEEN', 'LIKE', 'AND', 'OR', 'NOT'
        ]

        sql_upper = sql.upper()
        found_keywords = []

        for keyword in keywords:
            if keyword in sql_upper:
                found_keywords.append(keyword)

        return found_keywords

    def preprocess_dataset(
        self,
        dataset: Dataset,
        num_proc: int = 4
    ) -> Dataset:
        """
        Preprocess the entire dataset in parallel.

        Args:
            dataset: Dataset to preprocess
            num_proc: Number of processes for parallel preprocessing

        Returns:
            Preprocessed dataset
        """
        self.logger.info(f"Preprocessing dataset with {num_proc} processes")

        def preprocess_fn(examples):
            """Batch preprocessing function."""
            results = {
                'question': [],
                'schema': [],
                'sql': [],
                'question_length': [],
                'schema_length': [],
                'sql_length': [],
                'is_valid': [],
                'validation_error': [],
                'sql_keywords': [],
                'complexity': [],
            }

            # Handle both batched and single examples
            if isinstance(examples['question'], list):
                num_examples = len(examples['question'])
            else:
                num_examples = 1
                examples = {k: [v] for k, v in examples.items()}

            for i in range(num_examples):
                sample = {
                    'question': examples['question'][i],
                    'context': examples['context'][i],
                    'answer': examples['answer'][i],
                }

                processed = self.preprocess_sample(sample)

                for key in results:
                    results[key].append(processed[key])

            return results

        processed_dataset = dataset.map(
            preprocess_fn,
            batched=True,
            num_proc=num_proc,
            desc="Preprocessing samples"
        )

        self.logger.info(f"Preprocessing complete: {len(processed_dataset)} samples")

        return processed_dataset

    def filter_dataset(self, dataset: Dataset) -> Dataset:
        """
        Filter out invalid or problematic samples.

        Filters:
        - Missing required fields
        - Invalid SQL syntax
        - Questions/SQL too long
        - Empty or malformed content
        """
        self.logger.info("Filtering invalid samples")

        initial_size = len(dataset)

        # Filter by validity
        filtered = dataset.filter(
            lambda x: x['is_valid'],
            desc="Filtering invalid samples"
        )

        # Filter by length constraints
        filtered = filtered.filter(
            lambda x: (
                x['question_length'] <= self.max_question_length and
                x['sql_length'] <= self.max_sql_length and
                x['question_length'] > 0 and
                x['sql_length'] > 0
            ),
            desc="Filtering by length"
        )

        final_size = len(filtered)
        removed = initial_size - final_size

        self.logger.info(
            f"Filtered {removed} samples ({removed/initial_size*100:.1f}%). "
            f"Remaining: {final_size}"
        )

        return filtered

    def classify_complexity(self, sql: str) -> str:
        """
        Classify SQL complexity based on features.

        Simple: Basic SELECT with WHERE
        Medium: JOINs, GROUP BY, or subqueries
        Complex: Multiple JOINs, nested subqueries, or CTEs
        """
        sql_upper = sql.upper()

        # Count complexity indicators
        join_count = sql_upper.count('JOIN')
        subquery_count = sql_upper.count('SELECT') - 1  # Subtract main query
        has_group_by = 'GROUP BY' in sql_upper
        has_having = 'HAVING' in sql_upper
        has_union = 'UNION' in sql_upper
        has_cte = 'WITH' in sql_upper and 'AS' in sql_upper

        # Complex: CTEs, multiple JOINs, or nested subqueries
        if has_cte or join_count >= 2 or subquery_count >= 2:
            return 'complex'

        # Medium: Single JOIN, GROUP BY, HAVING, UNION, or single subquery
        if join_count >= 1 or has_group_by or has_having or has_union or subquery_count >= 1:
            return 'medium'

        # Simple: Basic SELECT
        return 'simple'
