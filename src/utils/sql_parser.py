"""SQL Parser for extracting and cleaning SQL queries from model outputs.

This module provides a parser that can extract SQL queries from various
formats including markdown code blocks, inline queries, and raw SQL.
"""

import re
from typing import Optional, List, Tuple


class SQLParser:
    """Parser for extracting and cleaning SQL queries from model outputs.

    Handles markdown code blocks, inline SQL, and raw queries. Designed to
    work with LLM outputs that may contain SQL in various formats.

    Examples:
        >>> parser = SQLParser()
        >>> sql = parser.extract_sql("```sql\\nSELECT * FROM users\\n```")
        >>> print(sql)
        SELECT * FROM users

        >>> sql = parser.extract_sql("The query is: SELECT name FROM products")
        >>> print(sql)
        SELECT name FROM products
    """

    # SQL keywords that typically start a query
    SQL_KEYWORDS = [
        "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP",
        "ALTER", "TRUNCATE", "WITH", "EXPLAIN", "DESCRIBE", "SHOW"
    ]

    # Patterns for detecting SQL in text
    CODE_BLOCK_PATTERN = re.compile(
        r"```(?:sql|SQL)?\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE
    )

    INLINE_CODE_PATTERN = re.compile(
        r"`([^`]+)`",
        re.DOTALL
    )

    def __init__(
        self,
        max_sql_length: int = 2048,
        min_sql_length: int = 10,
        extract_code_blocks: bool = True,
        clean_whitespace: bool = True,
    ):
        """Initialize the SQL parser.

        Args:
            max_sql_length: Maximum allowed SQL query length
            min_sql_length: Minimum required SQL query length
            extract_code_blocks: Whether to extract from markdown code blocks
            clean_whitespace: Whether to normalize whitespace
        """
        self.max_sql_length = max_sql_length
        self.min_sql_length = min_sql_length
        self.extract_code_blocks = extract_code_blocks
        self.clean_whitespace = clean_whitespace

    def extract_sql(self, text: str) -> Optional[str]:
        """Extract SQL query from text.

        Tries multiple extraction strategies in order:
        1. Markdown code blocks (```sql ... ```)
        2. Raw SQL starting with keywords
        3. Inline code (`...`)

        Args:
            text: Input text potentially containing SQL

        Returns:
            Extracted and cleaned SQL query, or None if no valid SQL found
        """
        if not text or not isinstance(text, str):
            return None

        text = text.strip()

        # Try extracting from code blocks first
        if self.extract_code_blocks:
            sql = self._extract_from_code_block(text)
            if sql:
                return sql

        # Try detecting raw SQL
        sql = self._extract_raw_sql(text)
        if sql:
            return sql

        # Try inline code as last resort
        sql = self._extract_from_inline_code(text)
        if sql:
            return sql

        return None

    def _extract_from_code_block(self, text: str) -> Optional[str]:
        """Extract SQL from markdown code blocks.

        Args:
            text: Input text

        Returns:
            Extracted SQL or None
        """
        matches = self.CODE_BLOCK_PATTERN.findall(text)
        if matches:
            # Take the first match
            sql = matches[0].strip()
            return self.clean_sql(sql) if sql else None
        return None

    def _extract_from_inline_code(self, text: str) -> Optional[str]:
        """Extract SQL from inline code backticks.

        Args:
            text: Input text

        Returns:
            Extracted SQL or None
        """
        matches = self.INLINE_CODE_PATTERN.findall(text)
        for match in matches:
            if self.detect_sql_pattern(match):
                return self.clean_sql(match)
        return None

    def _extract_raw_sql(self, text: str) -> Optional[str]:
        """Extract raw SQL that starts with SQL keywords.

        Args:
            text: Input text

        Returns:
            Extracted SQL or None
        """
        # Check if text starts with SQL keyword
        if self.detect_sql_pattern(text):
            # Extract until end or until we hit common ending markers
            sql = self._extract_until_end(text)
            return self.clean_sql(sql) if sql else None

        # Try to find SQL keyword in the middle of text
        for keyword in self.SQL_KEYWORDS:
            pattern = rf"\b{keyword}\b"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract from keyword to end
                sql = text[match.start():]
                sql = self._extract_until_end(sql)
                return self.clean_sql(sql) if sql else None

        return None

    def _extract_until_end(self, text: str) -> str:
        """Extract SQL until natural ending point.

        Args:
            text: Text starting with SQL

        Returns:
            Extracted SQL portion
        """
        # Stop at common markers that indicate end of SQL
        end_markers = [
            "\n\n",  # Double newline
            "\nExplanation:",
            "\nNote:",
            "\nThis query",
            "\nThe above",
        ]

        end_pos = len(text)
        for marker in end_markers:
            pos = text.find(marker)
            if pos != -1 and pos < end_pos:
                end_pos = pos

        return text[:end_pos].strip()

    def clean_sql(self, sql: str) -> str:
        """Clean and normalize SQL query.

        Args:
            sql: Raw SQL query

        Returns:
            Cleaned SQL query
        """
        if not sql:
            return sql

        # Remove common prefixes
        sql = re.sub(r'^(SQL:|Query:|Answer:)\s*', '', sql, flags=re.IGNORECASE)

        # Clean whitespace if enabled
        if self.clean_whitespace:
            # Normalize multiple spaces to single space
            sql = re.sub(r'\s+', ' ', sql)
            # But preserve line breaks for readability
            sql = sql.replace(' ;', ';')

        sql = sql.strip()

        # Check length constraints
        if len(sql) > self.max_sql_length:
            sql = sql[:self.max_sql_length]

        return sql

    def detect_sql_pattern(self, text: str) -> bool:
        """Detect if text contains SQL patterns.

        Args:
            text: Text to check

        Returns:
            True if text appears to contain SQL
        """
        if not text:
            return False

        text_upper = text.strip().upper()

        # Check if starts with SQL keyword
        for keyword in self.SQL_KEYWORDS:
            if text_upper.startswith(keyword):
                return True

        # Check for common SQL patterns
        sql_patterns = [
            r'\bSELECT\b.*\bFROM\b',
            r'\bINSERT\s+INTO\b',
            r'\bUPDATE\b.*\bSET\b',
            r'\bDELETE\s+FROM\b',
            r'\bCREATE\s+(TABLE|DATABASE|INDEX|VIEW)\b',
        ]

        for pattern in sql_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def is_valid_format(self, text: str) -> bool:
        """Check if text is in valid format for SQL extraction.

        Args:
            text: Input text

        Returns:
            True if format is valid for extraction
        """
        if not text or not isinstance(text, str):
            return False

        text = text.strip()

        # Check length constraints
        if len(text) < self.min_sql_length or len(text) > self.max_sql_length * 2:
            return False

        # Try to extract SQL
        sql = self.extract_sql(text)

        return sql is not None and len(sql) >= self.min_sql_length

    def parse_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Parse multiple texts and extract SQL from each.

        Args:
            texts: List of input texts

        Returns:
            List of extracted SQL queries (None for texts without valid SQL)
        """
        return [self.extract_sql(text) for text in texts]
