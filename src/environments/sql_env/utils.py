"""Utility functions for text-to-SQL environment.

This module provides helper functions for schema extraction, validation,
truncation, and dataset preparation for GRPO training.
"""

import re
from typing import Dict, List, Any, Optional
from datasets import Dataset


def extract_schema_info(context: str) -> Dict[str, List[str]]:
    """Extract table names and columns from CREATE TABLE statements.

    Args:
        context: CREATE TABLE statements from dataset

    Returns:
        Dict mapping table names to column lists

    Example:
        >>> context = "CREATE TABLE users (id INT, name VARCHAR(100))"
        >>> extract_schema_info(context)
        {'users': ['id', 'name']}
    """
    schema_info = {}

    if not context:
        return schema_info

    # Pattern to match CREATE TABLE statements
    # Matches: CREATE TABLE table_name (columns...)
    # Use greedy .* to capture until the last closing paren (handles nested parens in types like VARCHAR(100))
    table_pattern = r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"]?(\w+)[`\"]?\s*\((.*)\)"

    matches = re.finditer(table_pattern, context, re.IGNORECASE | re.DOTALL)

    for match in matches:
        table_name = match.group(1)
        columns_text = match.group(2)

        # Extract column names using a more robust pattern
        # Split on commas first, then extract column name and type from each part
        column_defs = columns_text.split(',')
        columns = []

        for col_def in column_defs:
            col_def = col_def.strip()
            # Match column_name at the start, followed by a data type
            # Pattern captures just the column name
            # The .* at the end handles any trailing characters (like closing parentheses, constraints, etc.)
            match = re.match(
                r'[`\"]?(\w+)[`\"]?\s+(?:VARCHAR(?:\s*\(\d+\))?|DATETIME|TIMESTAMP|INTEGER|DECIMAL|NUMERIC|BOOLEAN|DOUBLE|FLOAT|REAL|CHAR|TEXT|TIME|DATE|INT|BOOL|BLOB|CLOB).*',
                col_def,
                re.IGNORECASE
            )
            if match:
                columns.append(match.group(1))

        if columns:
            schema_info[table_name] = columns

    return schema_info


def validate_sql_against_schema(sql: str, schema_info: Dict[str, List[str]]) -> bool:
    """Check if SQL references valid tables/columns from schema.

    Basic validation that doesn't require database execution.

    Args:
        sql: SQL query to validate
        schema_info: Dict mapping table names to column lists

    Returns:
        True if SQL appears to reference valid tables, False otherwise

    Example:
        >>> schema = {'users': ['id', 'name']}
        >>> validate_sql_against_schema("SELECT name FROM users", schema)
        True
        >>> validate_sql_against_schema("SELECT name FROM products", schema)
        False
    """
    if not sql or not schema_info:
        return False

    sql_upper = sql.upper()

    # Extract table names mentioned in SQL
    # Look for FROM and JOIN clauses
    from_pattern = r"\bFROM\s+[`\"]?(\w+)[`\"]?"
    join_pattern = r"\bJOIN\s+[`\"]?(\w+)[`\"]?"

    tables_in_sql = []
    tables_in_sql.extend(re.findall(from_pattern, sql_upper, re.IGNORECASE))
    tables_in_sql.extend(re.findall(join_pattern, sql_upper, re.IGNORECASE))

    if not tables_in_sql:
        # No tables found in SQL - might be invalid
        return False

    # Check if at least one table is valid
    valid_tables = {name.upper() for name in schema_info.keys()}

    for table in tables_in_sql:
        if table.upper() in valid_tables:
            return True

    return False


def truncate_schema(context: str, max_length: int = 1024) -> str:
    """Truncate schema context if too long for model input.

    Keeps most important tables (defined earlier in context).

    Args:
        context: Full CREATE TABLE statements
        max_length: Maximum character length

    Returns:
        Truncated schema string

    Example:
        >>> long_schema = "CREATE TABLE t1 (id INT);" * 100
        >>> truncated = truncate_schema(long_schema, max_length=100)
        >>> len(truncated) <= 100
        True
    """
    if not context or len(context) <= max_length:
        return context

    # Split by CREATE TABLE statements
    table_pattern = r"(CREATE\s+TABLE\s+.*?;)"
    tables = re.findall(table_pattern, context, re.IGNORECASE | re.DOTALL)

    if not tables:
        # No tables found, just truncate naively
        return context[:max_length]

    # Add tables one by one until we hit max_length
    truncated_parts = []
    current_length = 0

    for table in tables:
        table_with_newline = table + "\n"
        if current_length + len(table_with_newline) <= max_length:
            truncated_parts.append(table_with_newline)
            current_length += len(table_with_newline)
        else:
            # Can't fit more tables
            break

    result = "".join(truncated_parts).strip()

    # If we couldn't fit any tables, just truncate the first one
    if not result and tables:
        result = tables[0][:max_length]

    return result


def prepare_for_grpo(
    dataset: Dataset,
    environment: Any,
    tokenizer: Optional[Any] = None,
) -> Dataset:
    """Prepare dataset for GRPO training.

    Formats samples and adds required fields for the Verifiers framework.

    Args:
        dataset: Raw dataset from Hugging Face
        environment: TextToSQLEnvironment instance
        tokenizer: Optional tokenizer for additional processing

    Returns:
        Prepared dataset with formatted prompts

    Example:
        >>> from datasets import Dataset
        >>> data = {"question": ["query"], "context": ["schema"], "answer": ["sql"]}
        >>> dataset = Dataset.from_dict(data)
        >>> # prepared = prepare_for_grpo(dataset, env)
    """
    def format_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single sample for GRPO."""
        # Format prompt using environment
        prompt = environment.format_prompt(
            question=sample["question"],
            context={"schema": sample.get("context", "")}
        )

        return {
            "prompt": prompt,
            "question": sample["question"],
            "context": sample.get("context", ""),
            "reference": sample.get("answer", ""),
        }

    # Apply formatting to all samples
    prepared = dataset.map(
        format_sample,
        desc="Preparing dataset for GRPO"
    )

    return prepared


def split_create_statements(context: str) -> List[str]:
    """Split multiple CREATE TABLE statements into individual statements.

    Args:
        context: Multiple CREATE TABLE statements

    Returns:
        List of individual CREATE TABLE statements

    Example:
        >>> context = "CREATE TABLE t1 (id INT); CREATE TABLE t2 (id INT);"
        >>> statements = split_create_statements(context)
        >>> len(statements)
        2
    """
    if not context:
        return []

    # Pattern to match full CREATE TABLE statements
    pattern = r"CREATE\s+TABLE\s+.*?;"
    statements = re.findall(pattern, context, re.IGNORECASE | re.DOTALL)

    return [stmt.strip() for stmt in statements if stmt.strip()]


def count_tables(context: str) -> int:
    """Count number of tables in schema context.

    Args:
        context: CREATE TABLE statements

    Returns:
        Number of tables

    Example:
        >>> context = "CREATE TABLE t1 (id INT); CREATE TABLE t2 (id INT);"
        >>> count_tables(context)
        2
    """
    statements = split_create_statements(context)
    return len(statements)


def get_table_names(context: str) -> List[str]:
    """Extract all table names from schema context.

    Args:
        context: CREATE TABLE statements

    Returns:
        List of table names

    Example:
        >>> context = "CREATE TABLE users (id INT); CREATE TABLE posts (id INT);"
        >>> get_table_names(context)
        ['users', 'posts']
    """
    schema_info = extract_schema_info(context)
    return list(schema_info.keys())
