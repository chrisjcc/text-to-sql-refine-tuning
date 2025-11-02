"""Prompt templates and formatting utilities for text-to-SQL generation.

This module provides multiple prompt templates for experimentation and
utilities for formatting schema information and few-shot examples.
"""

from typing import Any

PROMPT_TEMPLATES = {
    "default": """Given the following database schema:

{schema}

Generate a SQL query to answer this question:
Question: {question}

SQL Query:""",
    "instructional": """You are a SQL expert. Convert the natural language \
question into a valid SQL query.

Database Schema:
{schema}

Question: {question}

Instructions:
- Generate only the SQL query
- Use proper SQL syntax
- Include all necessary clauses
- Do not include explanations

SQL:""",
    "few_shot": """Generate SQL queries based on the database schema and \
examples below.

Schema:
{schema}

{examples}

Now generate SQL for:
Question: {question}
SQL:""",
    "chat": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that converts natural language questions into \
SQL queries. Given a database schema, generate accurate and efficient SQL \
queries.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Database Schema:
{schema}

Question: {question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

SQL Query:""",
    "concise": """Schema: {schema}

Question: {question}

SQL:""",
}


def get_prompt_template(name: str) -> str:
    """Get prompt template by name.

    Args:
        name: Template name from PROMPT_TEMPLATES dictionary.

    Returns:
        Template string with placeholders for formatting.

    Raises:
        ValueError: If template name is not found in PROMPT_TEMPLATES.
    """
    if name not in PROMPT_TEMPLATES:
        available = ", ".join(PROMPT_TEMPLATES.keys())
        raise ValueError(f"Unknown template '{name}'. Available templates: {available}")
    return PROMPT_TEMPLATES[name]


def format_schema(context: str) -> str:
    """Format CREATE TABLE statements for readability.

    Cleans up whitespace and formatting in raw CREATE TABLE statements
    to improve prompt clarity.

    Args:
        context: Raw CREATE TABLE statements from database schema.

    Returns:
        Formatted schema string with cleaned whitespace and line breaks.
    """
    if not context:
        return ""

    # Split by lines and clean up
    lines = context.strip().split("\n")
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if line:
            formatted_lines.append(line)

    return "\n".join(formatted_lines)


def format_few_shot_examples(examples: list[dict[str, Any]], n: int = 3) -> str:
    """Format few-shot examples for prompting.

    Creates a formatted string of example question-SQL pairs to include
    in the prompt for few-shot learning.

    Args:
        examples: List of example dictionaries, each containing
            'question' and 'answer' keys.
        n: Maximum number of examples to include. Defaults to 3.

    Returns:
        Formatted examples string ready for inclusion in prompt.
    """
    if not examples:
        return ""

    # Limit to n examples
    examples = examples[:n]

    formatted = ["Examples:"]
    for i, example in enumerate(examples, 1):
        question = example.get("question", "")
        answer = example.get("answer", "")

        formatted.append(f"\nExample {i}:")
        formatted.append(f"Question: {question}")
        formatted.append(f"SQL: {answer}")

    return "\n".join(formatted)


def format_prompt(
    template: str,
    question: str,
    schema: str = "",
    examples: str | None = None,
    **kwargs: Any,
) -> str:
    """Format a prompt using a template.

    Fills in template placeholders with provided values to create a
    complete prompt for the model.

    Args:
        template: Template string with {placeholders} for variable
            substitution.
        question: Natural language question to convert to SQL.
        schema: Database schema (formatted). Defaults to empty string.
        examples: Optional few-shot examples (formatted). Defaults to None.
        **kwargs: Additional template variables for custom templates.

    Returns:
        Formatted prompt string ready for model input.

    Raises:
        ValueError: If template requires variables that were not provided.
    """
    # Build template variables
    template_vars: dict[str, Any] = {
        "question": question,
        "schema": schema,
    }

    # Add examples if provided
    if examples:
        template_vars["examples"] = examples

    # Add any additional kwargs
    template_vars.update(kwargs)

    # Format the template
    try:
        return template.format(**template_vars)
    except KeyError as e:
        raise ValueError(f"Template requires variable {e} which was not provided") from e
