"""Prompt templates and formatting utilities for text-to-SQL generation.

This module provides multiple prompt templates for experimentation and
utilities for formatting schema information and few-shot examples.
"""

from typing import Dict, List, Optional


PROMPT_TEMPLATES = {
    "default": """Given the following database schema:

{schema}

Generate a SQL query to answer this question:
Question: {question}

SQL Query:""",

    "instructional": """You are a SQL expert. Convert the natural language question into a valid SQL query.

Database Schema:
{schema}

Question: {question}

Instructions:
- Generate only the SQL query
- Use proper SQL syntax
- Include all necessary clauses
- Do not include explanations

SQL:""",

    "few_shot": """Generate SQL queries based on the database schema and examples below.

Schema:
{schema}

{examples}

Now generate SQL for:
Question: {question}
SQL:""",

    "chat": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that converts natural language questions into SQL queries. Given a database schema, generate accurate and efficient SQL queries.

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
        name: Template name from PROMPT_TEMPLATES

    Returns:
        Template string

    Raises:
        ValueError: If template name not found
    """
    if name not in PROMPT_TEMPLATES:
        available = ", ".join(PROMPT_TEMPLATES.keys())
        raise ValueError(
            f"Unknown template '{name}'. Available templates: {available}"
        )
    return PROMPT_TEMPLATES[name]


def format_schema(context: str) -> str:
    """Format CREATE TABLE statements for readability.

    Args:
        context: Raw CREATE TABLE statements

    Returns:
        Formatted schema string
    """
    if not context:
        return ""

    # Split by CREATE TABLE and clean up
    lines = context.strip().split("\n")
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if line:
            formatted_lines.append(line)

    return "\n".join(formatted_lines)


def format_few_shot_examples(examples: List[Dict], n: int = 3) -> str:
    """Format few-shot examples for prompting.

    Args:
        examples: List of example dicts with 'question' and 'answer' keys
        n: Maximum number of examples to include

    Returns:
        Formatted examples string
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
    examples: Optional[str] = None,
    **kwargs
) -> str:
    """Format a prompt using a template.

    Args:
        template: Template string with {placeholders}
        question: Natural language question
        schema: Database schema (formatted)
        examples: Optional few-shot examples (formatted)
        **kwargs: Additional template variables

    Returns:
        Formatted prompt string
    """
    # Build template variables
    template_vars = {
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
        raise ValueError(
            f"Template requires variable {e} which was not provided"
        )
