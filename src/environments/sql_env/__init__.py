"""Text-to-SQL Environment for GRPO Training.

This package provides a custom Verifiers environment for text-to-SQL generation,
handling prompt formatting, response parsing, and reward computation.
"""

from .environment import TextToSQLEnvironment
from .prompts import PROMPT_TEMPLATES, format_few_shot_examples, format_schema, get_prompt_template
from .utils import (
    extract_schema_info,
    prepare_for_grpo,
    truncate_schema,
    validate_sql_against_schema,
)

__all__ = [
    "TextToSQLEnvironment",
    "PROMPT_TEMPLATES",
    "get_prompt_template",
    "format_schema",
    "format_few_shot_examples",
    "extract_schema_info",
    "validate_sql_against_schema",
    "truncate_schema",
    "prepare_for_grpo",
]
