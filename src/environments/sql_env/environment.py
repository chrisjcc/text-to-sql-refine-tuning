"""Text-to-SQL Environment for GRPO Training.

This module implements a SingleTurnEnv for text-to-SQL generation,
handling prompt formatting, response parsing, and reward computation.
"""

import logging
import re
from typing import Dict, List, Any, Optional

from verifiers import SingleTurnEnv

from rubrics.sql_rubric import SQLValidationRubric
from utils.sql_parser import SQLParser
from .prompts import (
    PROMPT_TEMPLATES,
    get_prompt_template,
    format_schema,
    format_few_shot_examples,
    format_prompt as format_prompt_util,
)
from .utils import extract_schema_info, validate_sql_against_schema, truncate_schema


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
        dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        **kwargs
    ):
        """Initialize text-to-SQL environment.

        Args:
            rubric: SQL validation rubric for scoring
            parser: SQL parser for extraction
            prompt_template: Template name or custom template string
            include_schema: Whether to include table schema in prompt
            max_examples: Number of few-shot examples to include
            max_schema_length: Maximum characters for schema context
            dataset: Optional dataset for training (required by verifiers base class)
            eval_dataset: Optional evaluation dataset (required by verifiers base class)
            **kwargs: Additional arguments for SingleTurnEnv
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
            template: Template name from PROMPT_TEMPLATES or custom string

        Returns:
            Template string
        """
        # Check if it's a template name
        if template in PROMPT_TEMPLATES:
            return get_prompt_template(template)

        # Check if this looks like an invalid template name
        # (single word without spaces or special characters that looks like a name)
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', template):
            raise ValueError(
                f"Unknown template: '{template}'. Available templates: {', '.join(PROMPT_TEMPLATES.keys())}"
            )

        # Otherwise, treat as custom template
        # Validate it has required placeholders
        if "{question}" not in template:
            raise ValueError(
                "Custom template must contain '{question}' placeholder"
            )

        return template

    def format_prompt(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format input prompt with question and optional schema context.

        Args:
            question: Natural language question
            context: Optional dict with 'schema', 'tables', 'examples' keys

        Returns:
            Formatted prompt string

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
                    f"Truncated schema from {len(raw_schema)} to {len(schema)} chars"
                )

        # Prepare few-shot examples if requested
        examples_str = ""
        if self.max_examples > 0 and context and "examples" in context:
            examples = context["examples"]
            examples_str = format_few_shot_examples(examples, self.max_examples)

        # Format using template
        prompt = format_prompt_util(
            template=self.prompt_template,
            question=question,
            schema=schema,
            examples=examples_str if examples_str else None,
        )

        return prompt

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse model response to extract SQL query.

        Args:
            response: Raw model output

        Returns:
            Dict with 'sql', 'valid', 'metadata' keys

        Example:
            >>> result = env.parse_response("```sql\\nSELECT * FROM users\\n```")
            >>> print(result['sql'])
            SELECT * FROM users
        """
        if not response:
            return {
                "sql": None,
                "valid": False,
                "metadata": {"error": "Empty response"}
            }

        # Extract SQL using parser
        sql = self.parser.extract_sql(response)

        if sql is None:
            return {
                "sql": None,
                "valid": False,
                "metadata": {"error": "Failed to extract SQL"}
            }

        # Validate basic format
        is_valid = self.parser.is_valid_format(response)

        return {
            "sql": sql,
            "valid": is_valid,
            "metadata": {
                "original_response": response,
                "extracted_length": len(sql),
            }
        }

    def compute_reward(
        self,
        response: str,
        reference: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute reward score for generated SQL.

        Args:
            response: Model's generated response
            reference: Optional ground truth SQL (if available)
            context: Additional context for scoring (e.g., schema for validation)

        Returns:
            Reward score between 0.0 and 1.0

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

        return max(0.0, min(1.0, score))

    def batch_compute_rewards(
        self,
        responses: List[str],
        references: Optional[List[str]] = None,
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[float]:
        """Efficiently compute rewards for batch of responses.

        Critical for GRPO training performance.

        Args:
            responses: List of model responses
            references: Optional list of ground truth SQLs
            contexts: Optional list of context dicts

        Returns:
            List of reward scores

        Example:
            >>> responses = ["SELECT * FROM users", "SELECT id FROM products"]
            >>> rewards = env.batch_compute_rewards(responses)
            >>> len(rewards) == len(responses)
            True
        """
        if not responses:
            return []

        # Prepare references and contexts
        refs = references if references else [None] * len(responses)
        ctxs = contexts if contexts else [None] * len(responses)

        # Use batch scoring from rubric for efficiency
        if all(ctx is None for ctx in ctxs):
            # Simple case: no schema validation needed
            rewards = self.rubric.score_batch(responses, refs)
        else:
            # Complex case: apply schema validation per response
            rewards = [
                self.compute_reward(response, ref, ctx)
                for response, ref, ctx in zip(responses, refs, ctxs)
            ]

        return rewards

    def prepare_dataset_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dataset sample to environment format.

        Handles b-mc2/sql-create-context dataset structure.

        Expected sample format:
        {
            'question': str,
            'context': str,  # CREATE TABLE statements
            'answer': str    # SQL query
        }

        Args:
            sample: Dataset sample

        Returns:
            Formatted sample ready for environment

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
            context={"schema": context} if context else None
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
        responses: List[str],
        references: Optional[List[str]] = None,
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """Compute aggregate metrics for evaluation.

        Args:
            responses: List of model responses
            references: Optional list of ground truth SQLs
            contexts: Optional list of context dicts

        Returns:
            Dict with evaluation metrics

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

        SingleTurnEnv is stateless, so this is a no-op.
        """
        pass
