"""Unit tests for text-to-SQL environment.

Tests cover environment initialization, prompt formatting, response parsing,
reward computation, and batch processing.
"""

import pytest
from datasets import Dataset

from src.environments.sql_env import TextToSQLEnvironment
from src.environments.sql_env.prompts import (
    PROMPT_TEMPLATES,
    format_few_shot_examples,
    format_schema,
    get_prompt_template,
)
from src.environments.sql_env.utils import (
    count_tables,
    extract_schema_info,
    get_table_names,
    prepare_for_grpo,
    truncate_schema,
    validate_sql_against_schema,
)
from src.rubrics.sql_rubric import SQLValidationRubric
from src.utils.sql_parser import SQLParser


@pytest.fixture
def parser():
    """Create SQL parser fixture."""
    return SQLParser()


@pytest.fixture
def rubric():
    """Create SQL rubric fixture."""
    return SQLValidationRubric()


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    data = {
        "question": ["How many users are there?", "List all products", "Get active users"],
        "context": [
            "CREATE TABLE users (id INT, name VARCHAR)",
            "CREATE TABLE products (id INT, name VARCHAR, price DECIMAL)",
            "CREATE TABLE users (id INT, active BOOLEAN)",
        ],
        "answer": [
            "SELECT COUNT(*) FROM users",
            "SELECT * FROM products",
            "SELECT * FROM users WHERE active = TRUE",
        ],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def environment(rubric, parser, mock_dataset):
    """Create environment fixture."""
    return TextToSQLEnvironment(
        rubric=rubric,
        parser=parser,
        prompt_template="default",
        dataset=mock_dataset,
    )


class TestEnvironmentInitialization:
    """Tests for environment initialization."""

    def test_initialization_with_defaults(self, rubric, parser, mock_dataset):
        """Test environment initializes with default parameters."""
        env = TextToSQLEnvironment(rubric=rubric, parser=parser, dataset=mock_dataset)

        assert env.rubric is rubric
        assert env.parser is parser
        assert env.include_schema is True
        assert env.max_examples == 0
        assert env.max_schema_length == 1024

    def test_initialization_with_custom_params(self, rubric, parser, mock_dataset):
        """Test environment initializes with custom parameters."""
        env = TextToSQLEnvironment(
            rubric=rubric,
            parser=parser,
            prompt_template="instructional",
            include_schema=False,
            max_examples=3,
            max_schema_length=512,
            dataset=mock_dataset,
        )

        assert env.include_schema is False
        assert env.max_examples == 3
        assert env.max_schema_length == 512

    def test_initialization_with_invalid_template(self, rubric, parser, mock_dataset):
        """Test initialization fails with invalid template name."""
        with pytest.raises(ValueError, match="Unknown template"):
            TextToSQLEnvironment(
                rubric=rubric,
                parser=parser,
                prompt_template="nonexistent_template",
                dataset=mock_dataset,
            )

    def test_initialization_with_custom_template(self, rubric, parser, mock_dataset):
        """Test initialization with custom template string."""
        custom_template = "Question: {question}\nSchema: {schema}\nSQL:"

        env = TextToSQLEnvironment(
            rubric=rubric,
            parser=parser,
            prompt_template=custom_template,
            dataset=mock_dataset,
        )

        assert env.prompt_template == custom_template

    def test_custom_template_without_question_placeholder(self, rubric, parser, mock_dataset):
        """Test custom template must have question placeholder."""
        invalid_template = "Schema: {schema}\nSQL:"

        with pytest.raises(ValueError, match="must contain"):
            TextToSQLEnvironment(
                rubric=rubric,
                parser=parser,
                prompt_template=invalid_template,
                dataset=mock_dataset,
            )


class TestPromptFormatting:
    """Tests for prompt formatting."""

    def test_format_prompt_basic(self, environment):
        """Test basic prompt formatting without schema."""
        prompt = environment.format_prompt("How many users?")

        assert "How many users?" in prompt
        assert "SQL" in prompt or "Query" in prompt

    def test_format_prompt_with_schema(self, environment):
        """Test prompt formatting with schema context."""
        schema = "CREATE TABLE users (id INT, name VARCHAR(100))"
        prompt = environment.format_prompt("How many users?", context={"schema": schema})

        assert "How many users?" in prompt
        assert "users" in prompt
        assert "id" in prompt or "name" in prompt

    def test_format_prompt_empty_question(self, environment):
        """Test formatting fails with empty question."""
        with pytest.raises(ValueError, match="cannot be empty"):
            environment.format_prompt("")

    def test_format_prompt_truncates_long_schema(self, rubric, parser, mock_dataset):
        """Test long schema gets truncated."""
        env = TextToSQLEnvironment(
            rubric=rubric,
            parser=parser,
            max_schema_length=100,
            dataset=mock_dataset,
        )

        long_schema = "CREATE TABLE users (id INT);" * 50
        prompt = env.format_prompt("Test question", context={"schema": long_schema})

        # Prompt should be shorter than full schema
        assert len(prompt) < len(long_schema) + 100

    def test_format_prompt_without_schema_when_disabled(self, rubric, parser, mock_dataset):
        """Test prompt excludes schema when include_schema=False."""
        env = TextToSQLEnvironment(
            rubric=rubric,
            parser=parser,
            include_schema=False,
            dataset=mock_dataset,
        )

        schema = "CREATE TABLE users (id INT)"
        prompt = env.format_prompt("How many users?", context={"schema": schema})

        # Schema should not be in prompt
        assert "CREATE TABLE" not in prompt


class TestResponseParsing:
    """Tests for response parsing."""

    def test_parse_response_valid_sql(self, environment):
        """Test parsing valid SQL response."""
        response = "SELECT * FROM users"
        result = environment.parse_response(response)

        assert result["sql"] == "SELECT * FROM users"
        assert result["valid"] is True

    def test_parse_response_with_code_block(self, environment):
        """Test parsing SQL in markdown code block."""
        response = "```sql\nSELECT * FROM users\n```"
        result = environment.parse_response(response)

        assert "SELECT * FROM users" in result["sql"]
        assert result["valid"] is True

    def test_parse_response_empty(self, environment):
        """Test parsing empty response."""
        result = environment.parse_response("")

        assert result["sql"] is None
        assert result["valid"] is False
        assert "error" in result["metadata"]

    def test_parse_response_no_sql(self, environment):
        """Test parsing response without SQL."""
        response = "This is just text without any SQL"
        result = environment.parse_response(response)

        assert result["sql"] is None
        assert result["valid"] is False


class TestRewardComputation:
    """Tests for reward computation."""

    def test_compute_reward_valid_sql(self, environment):
        """Test reward for valid SQL."""
        sql = "SELECT * FROM users WHERE id = 1"
        reward = environment.compute_reward(sql)

        assert 0.0 <= reward <= 1.0
        assert reward > 0.5  # Valid SQL should get decent score

    def test_compute_reward_invalid_sql(self, environment):
        """Test reward for invalid SQL."""
        invalid = "This is not SQL"
        reward = environment.compute_reward(invalid)

        assert reward == 0.0

    def test_compute_reward_empty(self, environment):
        """Test reward for empty response."""
        reward = environment.compute_reward("")

        assert reward == 0.0

    def test_compute_reward_with_schema_validation(self, environment):
        """Test reward with schema validation."""
        schema = "CREATE TABLE users (id INT, name VARCHAR(100))"

        # Valid table reference
        reward_valid = environment.compute_reward("SELECT * FROM users", context={"schema": schema})

        # Invalid table reference
        reward_invalid = environment.compute_reward(
            "SELECT * FROM products", context={"schema": schema}
        )

        # Valid reference should score higher
        assert reward_valid > reward_invalid


class TestBatchRewardComputation:
    """Tests for batch reward computation."""

    def test_batch_compute_rewards(self, environment):
        """Test batch reward computation."""
        responses = ["SELECT * FROM users", "SELECT id FROM products", "This is invalid"]

        rewards = environment.batch_compute_rewards(responses)

        assert len(rewards) == len(responses)
        assert all(0.0 <= r <= 1.0 for r in rewards)
        assert rewards[0] > 0.0  # Valid SQL
        assert rewards[1] > 0.0  # Valid SQL
        assert rewards[2] == 0.0  # Invalid

    def test_batch_compute_rewards_empty(self, environment):
        """Test batch computation with empty list."""
        rewards = environment.batch_compute_rewards([])

        assert rewards == []

    def test_batch_compute_rewards_with_references(self, environment):
        """Test batch computation with reference SQLs."""
        responses = ["SELECT * FROM users", "SELECT id FROM products"]
        references = ["SELECT * FROM users", "SELECT * FROM products"]

        rewards = environment.batch_compute_rewards(responses, references)

        assert len(rewards) == len(responses)

    def test_batch_compute_rewards_efficiency(self, environment):
        """Test batch computation is reasonably fast."""
        import time

        responses = ["SELECT * FROM users WHERE id = 1"] * 100

        start = time.time()
        rewards = environment.batch_compute_rewards(responses)
        elapsed = time.time() - start

        samples_per_sec = len(responses) / elapsed

        # Should process at least 50 samples/sec (conservative threshold)
        assert samples_per_sec > 50, f"Only {samples_per_sec:.1f} samples/sec"
        assert len(rewards) == 100


class TestDatasetPreparation:
    """Tests for dataset preparation."""

    def test_prepare_dataset_sample(self, environment):
        """Test preparing a dataset sample."""
        sample = {
            "question": "How many users?",
            "context": "CREATE TABLE users (id INT)",
            "answer": "SELECT COUNT(*) FROM users",
        }

        prepared = environment.prepare_dataset_sample(sample)

        assert "prompt" in prepared
        assert "question" in prepared
        assert "schema" in prepared
        assert "reference" in prepared
        assert prepared["question"] == sample["question"]

    def test_prepare_dataset_sample_missing_question(self, environment):
        """Test preparation fails without question."""
        sample = {"context": "schema", "answer": "sql"}

        with pytest.raises(ValueError, match="must contain"):
            environment.prepare_dataset_sample(sample)


class TestMetrics:
    """Tests for metrics computation."""

    def test_get_metrics(self, environment):
        """Test computing aggregate metrics."""
        responses = ["SELECT * FROM users", "SELECT id FROM products", "This is invalid"]

        metrics = environment.get_metrics(responses)

        assert "valid_sql_pct" in metrics
        assert "avg_reward" in metrics
        assert "syntax_correct_pct" in metrics
        assert "num_samples" in metrics
        assert metrics["num_samples"] == 3

    def test_get_metrics_empty(self, environment):
        """Test metrics with empty list."""
        metrics = environment.get_metrics([])

        assert metrics["num_samples"] == 0
        assert metrics["avg_reward"] == 0.0


class TestPromptTemplates:
    """Tests for prompt template utilities."""

    def test_get_prompt_template_valid(self):
        """Test getting valid template."""
        template = get_prompt_template("default")
        assert "{question}" in template
        assert "{schema}" in template

    def test_get_prompt_template_invalid(self):
        """Test getting invalid template raises error."""
        with pytest.raises(ValueError, match="Unknown template"):
            get_prompt_template("nonexistent")

    def test_all_templates_have_question(self):
        """Test all templates have question placeholder."""
        for name, template in PROMPT_TEMPLATES.items():
            assert "{question}" in template, f"Template '{name}' missing question"

    def test_format_schema(self):
        """Test schema formatting."""
        schema = "CREATE TABLE users (id INT, name VARCHAR(100))"
        formatted = format_schema(schema)

        assert "users" in formatted
        assert "id" in formatted

    def test_format_few_shot_examples(self):
        """Test few-shot example formatting."""
        examples = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]

        formatted = format_few_shot_examples(examples, n=2)

        assert "Q1" in formatted
        assert "A1" in formatted
        assert "Example" in formatted


class TestSchemaExtraction:
    """Tests for schema extraction utilities."""

    def test_extract_schema_info(self):
        """Test extracting schema info."""
        schema = "CREATE TABLE users (id INT, name VARCHAR(100), email TEXT)"
        info = extract_schema_info(schema)

        assert "users" in info
        assert "id" in info["users"]
        assert "name" in info["users"]
        assert "email" in info["users"]

    def test_extract_schema_info_multiple_tables(self):
        """Test extracting multiple tables."""
        schema = """
        CREATE TABLE users (id INT, name VARCHAR(100));
        CREATE TABLE products (id INT, name VARCHAR(200));
        """
        info = extract_schema_info(schema)

        assert len(info) == 2
        assert "users" in info
        assert "products" in info

    def test_validate_sql_against_schema(self):
        """Test SQL validation against schema."""
        schema_info = {"users": ["id", "name"], "products": ["id", "name"]}

        # Valid reference
        assert validate_sql_against_schema("SELECT * FROM users", schema_info)

        # Invalid reference
        assert not validate_sql_against_schema("SELECT * FROM orders", schema_info)

    def test_truncate_schema(self):
        """Test schema truncation."""
        long_schema = "CREATE TABLE users (id INT);" * 100
        truncated = truncate_schema(long_schema, max_length=100)

        assert len(truncated) <= 100
        assert "CREATE TABLE" in truncated

    def test_count_tables(self):
        """Test counting tables in schema."""
        schema = """
        CREATE TABLE users (id INT);
        CREATE TABLE products (id INT);
        """
        count = count_tables(schema)

        assert count == 2

    def test_get_table_names(self):
        """Test extracting table names."""
        schema = """
        CREATE TABLE users (id INT);
        CREATE TABLE products (id INT);
        """
        names = get_table_names(schema)

        assert len(names) == 2
        assert "users" in names
        assert "products" in names


class TestGRPOPreparation:
    """Tests for GRPO dataset preparation."""

    def test_prepare_for_grpo(self, environment):
        """Test preparing dataset for GRPO."""
        data = {
            "question": ["Q1", "Q2"],
            "context": ["schema1", "schema2"],
            "answer": ["sql1", "sql2"],
        }
        dataset = Dataset.from_dict(data)

        prepared = prepare_for_grpo(dataset, environment)

        assert len(prepared) == 2
        assert "prompt" in prepared.column_names
        assert "reference" in prepared.column_names


class TestEnvironmentReset:
    """Tests for environment reset."""

    def test_reset(self, environment):
        """Test environment reset (should be no-op for SingleTurnEnv)."""
        environment.reset()
        # Should not raise any errors
        assert True
