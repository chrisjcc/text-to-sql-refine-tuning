"""Text-to-SQL Environment for GRPO Training.

This module implements a SingleTurnEnv for text-to-SQL generation,
handling prompt formatting, response parsing, and reward computation.
"""

import logging
import re
from typing import Any

from verifiers import SingleTurnEnv

from src.rubrics.sql_rubric import SQLValidationRubric
from src.utils.sql_parser import SQLParser

from .prompts import (
    PROMPT_TEMPLATES,
    format_few_shot_examples,
    format_prompt as format_prompt_util,
    format_schema,
    get_prompt_template,
)
from .utils import extract_schema_info, truncate_schema, validate_sql_against_schema

logger = logging.getLogger(__name__)


class TextToSQLEnvironment(SingleTurnEnv):
    """Single-turn environment for text-to-SQL generation.

    Handles:
    - Prompt formatting with schema context
    - Model response validation
    - Reward computation via SQL rubric
    - Batch processing for GRPO

    This environment is designed for stateless text-to-SQL generation where
    each query is independent and requires no conversation history.

    Attributes:
        rubric: SQL validation rubric for scoring responses.
        parser: SQL parser for extraction and validation.
        include_schema: Whether to include table schema in prompts.
        max_examples: Number of few-shot examples to include.
        max_schema_length: Maximum characters for schema context.
        prompt_template: Loaded prompt template string.
        logger: Logger instance for this class.

    Examples:
        >>> from src.rubrics.sql_rubric import SQLValidationRubric
        >>> from src.utils.sql_parser import SQLParser
        >>> rubric = SQLValidationRubric()
        >>> parser = SQLParser()
        >>> env = TextToSQLEnvironment(rubric=rubric, parser=parser)
        >>> prompt = env.format_prompt(
        ...     question="Get all users",
        ...     context={"schema": "CREATE TABLE users (id INT, name VARCHAR)"}
        ... )
        >>> reward = env.compute_reward("SELECT * FROM users")
        >>> print(f"Reward: {reward:.2f}")
    """

    def __init__(
        self,
        rubric: SQLValidationRubric,
        parser: SQLParser,
        prompt_template: str = "default",
        include_schema: bool = True,
        max_examples: int = 0,
        max_schema_length: int = 1024,
        dataset: Any | None = None,
        eval_dataset: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize text-to-SQL environment.

        Args:
            rubric: SQL validation rubric for scoring responses.
            parser: SQL parser for SQL extraction from text.
            prompt_template: Template name from PROMPT_TEMPLATES or
                custom template string. Defaults to "default".
            include_schema: Whether to include table schema in prompt.
                Defaults to True.
            max_examples: Number of few-shot examples to include.
                Defaults to 0 (no examples).
            max_schema_length: Maximum characters for schema context.
                Defaults to 1024.
            dataset: Optional dataset for training (required by verifiers
                base class). Defaults to None.
            eval_dataset: Optional evaluation dataset (required by
                verifiers base class). Defaults to None.
            **kwargs: Additional arguments for SingleTurnEnv base class.
        """
        # Initialize base class with dataset parameters
        # Note: verifiers base class requires either dataset or eval_dataset
        super().__init__(dataset=dataset, eval_dataset=eval_dataset, **kwargs)

        self.rubric = rubric
        self.parser = parser
        self.include_schema = include_schema
        self.max_examples = max_examples
        self.max_schema_length = max_schema_length

        # Load prompt template
        self.prompt_template = self._load_prompt_template(prompt_template)

        self.logger = logger

    def _load_prompt_template(self, template: str) -> str:
        """Load prompt template by name or use custom template.

        Args:
            template: Template name from PROMPT_TEMPLATES or custom
                template string.

        Returns:
            Template string ready for formatting.

        Raises:
            ValueError: If template name is invalid or custom template
                is missing required placeholders.
        """
        # Check if it's a template name
        if template in PROMPT_TEMPLATES:
            return get_prompt_template(template)

        # Check if this looks like an invalid template name
        # (single word without spaces that looks like a name)
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", template):
            available = ", ".join(PROMPT_TEMPLATES.keys())
            raise ValueError(
                f"Unknown template: '{template}'. " f"Available templates: {available}"
            )

        # Otherwise, treat as custom template
        # Validate it has required placeholders
        if "{question}" not in template:
            raise ValueError("Custom template must contain '{question}' placeholder")

        return template

    def format_prompt(self, question: str, context: dict[str, Any] | None = None) -> str:
        """Format input prompt with question and optional schema context.

        Args:
            question: Natural language question to convert to SQL.
            context: Optional dict with 'schema', 'tables', 'examples' keys.
                Defaults to None.

        Returns:
            Formatted prompt string ready for model input.

        Raises:
            ValueError: If question is empty.

        Example:
            >>> env = TextToSQLEnvironment(rubric, parser)
            >>> prompt = env.format_prompt(
            ...     "How many users are there?",
            ...     context={"schema": "CREATE TABLE users (id INT)"}
            ... )
        """
        if not question:
            raise ValueError("Question cannot be empty")

        # Prepare schema
        schema = ""
        if self.include_schema and context and "schema" in context:
            raw_schema = context["schema"]

            # Format schema for readability
            schema = format_schema(raw_schema)

            # Truncate if too long
            if len(schema) > self.max_schema_length:
                schema = truncate_schema(schema, self.max_schema_length)
                self.logger.debug(
                    f"Truncated schema from {len(raw_schema)} to " f"{len(schema)} chars"
                )

        # Prepare few-shot examples if requested
        examples_str = ""
        if self.max_examples > 0 and context and "examples" in context:
            examples = context["examples"]
            examples_str = format_few_shot_examples(examples, self.max_examples)

        # Format using template
        return format_prompt_util(
            template=self.prompt_template,
            question=question,
            schema=schema,
            examples=examples_str if examples_str else None,
        )

    def parse_response(self, response: str) -> dict[str, Any]:
        """Parse model response to extract SQL query.

        Args:
            response: Raw model output text.

        Returns:
            Dictionary containing:
            - sql: Extracted SQL string or None
            - valid: Boolean indicating if format is valid
            - metadata: Additional parsing information

        Example:
            >>> result = env.parse_response(
            ...     "```sql\\nSELECT * FROM users\\n```"
            ... )
            >>> print(result['sql'])
            SELECT * FROM users
        """
        if not response:
            return {
                "sql": None,
                "valid": False,
                "metadata": {"error": "Empty response"},
            }

        # Extract SQL using parser
        sql = self.parser.extract_sql(response)

        if sql is None:
            return {
                "sql": None,
                "valid": False,
                "metadata": {"error": "Failed to extract SQL"},
            }

        # Validate basic format
        is_valid = self.parser.is_valid_format(response)

        return {
            "sql": sql,
            "valid": is_valid,
            "metadata": {
                "original_response": response,
                "extracted_length": len(sql),
            },
        }

    def compute_reward(
        self,
        response: str,
        reference: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> float:
        """Compute reward score for generated SQL.

        Args:
            response: Model's generated response text.
            reference: Optional ground truth SQL for comparison.
                Defaults to None.
            context: Additional context for scoring (e.g., schema for
                validation). Defaults to None.

        Returns:
            Reward score between 0.0 and 1.0.

        Example:
            >>> reward = env.compute_reward("SELECT * FROM users")
            >>> assert 0.0 <= reward <= 1.0
        """
        if not response:
            return 0.0

        # Use rubric to score the response
        score = self.rubric.score(response, reference)

        # Optional: Apply schema-based validation if context provided
        if context and "schema" in context:
            parsed = self.parse_response(response)
            sql = parsed.get("sql")

            if sql:
                schema_info = extract_schema_info(context["schema"])
                if schema_info:
                    is_valid_schema = validate_sql_against_schema(sql, schema_info)

                    # If SQL references invalid tables, penalize the score
                    if not is_valid_schema:
                        score *= 0.7  # 30% penalty
                        self.logger.debug("Applied schema validation penalty")

        return float(max(0.0, min(1.0, score)))

    def batch_compute_rewards(
        self,
        responses: list[str],
        references: list[str] | None = None,
        contexts: list[dict[str, Any]] | None = None,
    ) -> list[float]:
        """Efficiently compute rewards for batch of responses.

        Critical for GRPO training performance. Uses batch scoring from
        rubric when possible, falls back to individual scoring when
        schema validation is required.

        Args:
            responses: List of model responses to score.
            references: Optional list of ground truth SQLs corresponding
                to responses. Defaults to None.
            contexts: Optional list of context dicts for each response.
                Defaults to None.

        Returns:
            List of reward scores between 0.0 and 1.0.

        Example:
            >>> responses = ["SELECT * FROM users", "SELECT id FROM products"]
            >>> rewards = env.batch_compute_rewards(responses)
            >>> len(rewards) == len(responses)
            True
        """
        if not responses:
            return []

        # Prepare references and contexts
        refs = references if references else [None] * len(responses)  # type: ignore[list-item]
        ctxs = contexts if contexts else [None] * len(responses)  # type: ignore[list-item]

        # Use batch scoring from rubric for efficiency
        if all(ctx is None for ctx in ctxs):
            # Simple case: no schema validation needed
            rewards = self.rubric.score_batch(responses, refs)
        else:
            # Complex case: apply schema validation per response
            rewards = [
                self.compute_reward(response, ref, ctx)
                for response, ref, ctx in zip(responses, refs, ctxs, strict=True)
            ]

        return [float(r) for r in rewards]

    def prepare_dataset_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Convert dataset sample to environment format.

        Handles b-mc2/sql-create-context dataset structure and transforms
        it into the format expected by the environment.

        Args:
            sample: Dataset sample with expected format:
                - question: Natural language query string
                - context: CREATE TABLE statements
                - answer: Reference SQL query

        Returns:
            Formatted sample dictionary containing:
            - prompt: Formatted prompt for model input
            - question: Original question
            - schema: Database schema
            - reference: Ground truth SQL
            - context: Context dict for reward computation

        Raises:
            ValueError: If sample doesn't contain 'question' field.

        Example:
            >>> sample = {
            ...     "question": "How many users?",
            ...     "context": "CREATE TABLE users (id INT)",
            ...     "answer": "SELECT COUNT(*) FROM users"
            ... }
            >>> formatted = env.prepare_dataset_sample(sample)
            >>> 'prompt' in formatted
            True
        """
        question = sample.get("question", "")
        context = sample.get("context", "")
        answer = sample.get("answer", "")

        if not question:
            raise ValueError("Sample must contain 'question' field")

        # Format prompt
        prompt = self.format_prompt(
            question=question,
            context={"schema": context} if context else None,
        )

        return {
            "prompt": prompt,
            "question": question,
            "schema": context,
            "reference": answer,
            "context": {"schema": context} if context else None,
        }

    def get_metrics(
        self,
        responses: list[str],
        references: list[str] | None = None,
        contexts: list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Compute aggregate metrics for evaluation.

        Analyzes a batch of responses to compute comprehensive evaluation
        metrics including validity, rewards, and syntax correctness.

        Args:
            responses: List of model responses to evaluate.
            references: Optional list of ground truth SQLs. Defaults to None.
            contexts: Optional list of context dicts. Defaults to None.

        Returns:
            Dictionary containing evaluation metrics:
            - valid_sql_pct: Percentage of responses with valid SQL
            - avg_reward: Average reward score
            - syntax_correct_pct: Percentage with correct syntax
            - num_samples: Number of responses evaluated
            - min_reward: Minimum reward in batch
            - max_reward: Maximum reward in batch

        Example:
            >>> responses = ["SELECT * FROM users", "invalid query"]
            >>> metrics = env.get_metrics(responses)
            >>> 'avg_reward' in metrics
            True
        """
        if not responses:
            return {
                "valid_sql_pct": 0.0,
                "avg_reward": 0.0,
                "syntax_correct_pct": 0.0,
                "num_samples": 0,
            }

        # Parse all responses
        parsed_results = [self.parse_response(r) for r in responses]

        # Count valid extractions
        valid_count = sum(1 for p in parsed_results if p["valid"])

        # Compute rewards
        rewards = self.batch_compute_rewards(responses, references, contexts)
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

        # Check syntax correctness
        syntax_correct = 0
        for parsed in parsed_results:
            sql = parsed.get("sql")
            if sql:
                is_valid, _ = self.rubric.check_syntax(sql)
                if is_valid:
                    syntax_correct += 1

        return {
            "valid_sql_pct": (valid_count / len(responses)) * 100,
            "avg_reward": avg_reward,
            "syntax_correct_pct": (syntax_correct / len(responses)) * 100,
            "num_samples": len(responses),
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
        }

    def reset(self) -> None:
        """Reset environment state.

        SingleTurnEnv is stateless, so this is a no-op included for
        compatibility with the base class interface.

        Returns:
            None.
        """
        pass
