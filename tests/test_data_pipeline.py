"""Unit tests for data pipeline components.

Tests for dataset loading, preprocessing, and GRPO formatting.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset, DatasetDict  # noqa: E402

from src.data.dataset_loader import SQLDatasetLoader  # noqa: E402
from src.data.grpo_formatter import GRPODatasetFormatter  # noqa: E402
from src.data.preprocessor import SQLDataPreprocessor  # noqa: E402

# ============================================================================
# Test Data
# ============================================================================

SAMPLE_DATASET = {
    "question": [
        "How many users are there?",
        "List all products with price greater than 100",
        "Find the average salary by department",
    ],
    "context": [
        "CREATE TABLE users (id INT, name VARCHAR(100))",
        "CREATE TABLE products (id INT, name VARCHAR(100), price FLOAT)",
        "CREATE TABLE employees (id INT, name VARCHAR(100), salary FLOAT, dept_id INT)",
    ],
    "answer": [
        "SELECT COUNT(*) FROM users",
        "SELECT * FROM products WHERE price > 100",
        "SELECT dept_id, AVG(salary) FROM employees GROUP BY dept_id",
    ],
}


# ============================================================================
# Dataset Loader Tests
# ============================================================================


def test_dataset_loading():
    """Test basic dataset loading functionality."""
    loader = SQLDatasetLoader(
        dataset_name="b-mc2/sql-create-context", cache_dir="./test_cache", seed=42
    )

    assert loader.dataset_name == "b-mc2/sql-create-context"
    assert loader.seed == 42
    assert loader.cache_dir == "./test_cache"


def test_split_creation():
    """Test train/val/test split creation."""
    # Create a sample dataset
    dataset = Dataset.from_dict(SAMPLE_DATASET)

    loader = SQLDatasetLoader(seed=42)
    splits = loader.create_splits(dataset, train_size=0.6, val_size=0.2, test_size=0.2)

    assert isinstance(splits, DatasetDict)
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits

    # Check that all samples are accounted for
    total_samples = len(splits["train"]) + len(splits["validation"]) + len(splits["test"])
    assert total_samples == len(dataset)


def test_statistics_computation():
    """Test dataset statistics computation."""
    dataset = Dataset.from_dict(SAMPLE_DATASET)
    loader = SQLDatasetLoader(seed=42)

    stats = loader.get_statistics(dataset)

    assert "total_samples" in stats
    assert stats["total_samples"] == len(dataset)
    assert "avg_question_length" in stats
    assert "avg_sql_length" in stats
    assert "avg_schema_length" in stats
    assert stats["avg_question_length"] > 0
    assert stats["avg_sql_length"] > 0


# ============================================================================
# Preprocessor Tests
# ============================================================================


def test_question_cleaning():
    """Test question text cleaning."""
    preprocessor = SQLDataPreprocessor()

    # Test whitespace normalization
    assert preprocessor.clean_question("  How  many   users?  ") == "How many users?"

    # Test encoding fixes
    assert preprocessor.clean_question("What\u2019s the total") == "What's the total?"

    # Test punctuation addition
    assert preprocessor.clean_question("How many users") == "How many users?"


def test_schema_cleaning():
    """Test schema cleaning and formatting."""
    preprocessor = SQLDataPreprocessor()

    schema = """
    CREATE TABLE users (
        id INT,  -- Primary key
        name VARCHAR(100)
    ) /* Table for users */
    """

    cleaned = preprocessor.clean_schema(schema)

    # Should remove comments
    assert "--" not in cleaned
    assert "/*" not in cleaned
    assert "*/" not in cleaned

    # Should normalize whitespace
    assert "  " not in cleaned


def test_sql_cleaning():
    """Test SQL query cleaning and normalization."""
    preprocessor = SQLDataPreprocessor(normalize_sql=True)

    sql = "  select   *  from  users  where  id=1  "
    cleaned = preprocessor.clean_sql(sql)

    # Should normalize whitespace
    assert "  " not in cleaned

    # Should uppercase keywords (due to sqlparse)
    assert "SELECT" in cleaned
    assert "FROM" in cleaned
    assert "WHERE" in cleaned


def test_sample_validation():
    """Test sample validation logic."""
    preprocessor = SQLDataPreprocessor()

    # Valid sample
    valid_sample = {
        "question": "How many users?",
        "context": "CREATE TABLE users (id INT)",
        "answer": "SELECT COUNT(*) FROM users",
    }
    is_valid, error = preprocessor.validate_sample(valid_sample)
    assert is_valid
    assert error is None

    # Missing field
    invalid_sample = {
        "question": "How many users?",
        "context": "CREATE TABLE users (id INT)",
        # Missing 'answer'
    }
    is_valid, error = preprocessor.validate_sample(invalid_sample)
    assert not is_valid
    assert error is not None

    # Empty question
    invalid_sample = {
        "question": "",
        "context": "CREATE TABLE users (id INT)",
        "answer": "SELECT COUNT(*) FROM users",
    }
    is_valid, error = preprocessor.validate_sample(invalid_sample)
    assert not is_valid


def test_preprocessing_pipeline():
    """Test full preprocessing pipeline."""
    dataset = Dataset.from_dict(SAMPLE_DATASET)
    preprocessor = SQLDataPreprocessor()

    # Preprocess dataset
    processed = preprocessor.preprocess_dataset(dataset, num_proc=1)

    # Check that new columns are added
    assert "question" in processed.column_names
    assert "schema" in processed.column_names
    assert "sql" in processed.column_names
    assert "is_valid" in processed.column_names
    assert "complexity" in processed.column_names
    assert "sql_keywords" in processed.column_names

    # Check that all samples were processed
    assert len(processed) == len(dataset)


def test_complexity_classification():
    """Test SQL complexity classification."""
    preprocessor = SQLDataPreprocessor()

    # Simple query
    simple_sql = "SELECT * FROM users WHERE id = 1"
    assert preprocessor.classify_complexity(simple_sql) == "simple"

    # Medium query (with JOIN)
    medium_sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
    assert preprocessor.classify_complexity(medium_sql) == "medium"

    # Medium query (with GROUP BY)
    medium_sql2 = "SELECT dept_id, COUNT(*) FROM employees GROUP BY dept_id"
    assert preprocessor.classify_complexity(medium_sql2) == "medium"

    # Complex query (multiple JOINs)
    complex_sql = """
    SELECT u.name, COUNT(o.id)
    FROM users u
    JOIN orders o ON u.id = o.user_id
    JOIN products p ON o.product_id = p.id
    GROUP BY u.name
    """
    assert preprocessor.classify_complexity(complex_sql) == "complex"

    # Complex query (with CTE)
    complex_sql2 = """
    WITH user_stats AS (
        SELECT user_id, COUNT(*) as order_count FROM orders GROUP BY user_id
    )
    SELECT * FROM user_stats WHERE order_count > 5
    """
    assert preprocessor.classify_complexity(complex_sql2) == "complex"


def test_dataset_filtering():
    """Test filtering of invalid samples."""
    # Create dataset with some invalid samples
    mixed_data = {
        "question": ["Valid question?", "", "Another valid question?"],
        "context": ["CREATE TABLE t1 (id INT)", "CREATE TABLE t2 (id INT)", ""],
        "answer": ["SELECT * FROM t1", "INVALID SQL SYNTAX", "SELECT * FROM t3"],
    }
    dataset = Dataset.from_dict(mixed_data)

    preprocessor = SQLDataPreprocessor(filter_invalid=True)

    # Preprocess
    processed = preprocessor.preprocess_dataset(dataset, num_proc=1)

    # Filter
    filtered = preprocessor.filter_dataset(processed)

    # Should filter out invalid samples
    assert len(filtered) < len(dataset)
    assert all(filtered["is_valid"])


# ============================================================================
# GRPO Formatter Tests
# ============================================================================


def test_grpo_formatting():
    """Test GRPO dataset formatting."""
    # Create a mock environment
    from src.environments.sql_env.environment import TextToSQLEnvironment
    from src.rubrics.sql_rubric import SQLValidationRubric
    from src.utils.sql_parser import SQLParser

    rubric = SQLValidationRubric()
    parser = SQLParser()

    # Create a minimal mock dataset (required by verifiers base class)
    mock_dataset = Dataset.from_dict(
        {
            "question": ["Mock question?"],
            "context": ["CREATE TABLE mock (id INT)"],
            "answer": ["SELECT * FROM mock"],
        }
    )

    env = TextToSQLEnvironment(rubric=rubric, parser=parser, dataset=mock_dataset)

    # Create a mock tokenizer
    class MockTokenizer:
        def encode(self, text, _add_special_tokens=True):
            # Simple word-based tokenization for testing
            return text.split()

    tokenizer = MockTokenizer()

    formatter = GRPODatasetFormatter(environment=env, tokenizer=tokenizer, include_reference=True)

    # Test single sample formatting
    sample = {
        "question": "How many users?",
        "schema": "CREATE TABLE users (id INT)",
        "sql": "SELECT COUNT(*) FROM users",
        "complexity": "simple",
        "is_valid": True,
    }

    formatted = formatter.format_for_grpo(sample)

    assert "prompt" in formatted
    assert "question" in formatted
    assert "schema" in formatted
    assert "reference" in formatted
    assert formatted["question"] == sample["question"]
    assert formatted["reference"] == sample["sql"]


def test_tokenization_validation():
    """Test tokenization validation."""
    from src.environments.sql_env.environment import TextToSQLEnvironment
    from src.rubrics.sql_rubric import SQLValidationRubric
    from src.utils.sql_parser import SQLParser

    rubric = SQLValidationRubric()
    parser = SQLParser()

    # Create a minimal mock dataset (required by verifiers base class)
    mock_dataset = Dataset.from_dict(
        {
            "question": ["Mock question?"],
            "context": ["CREATE TABLE mock (id INT)"],
            "answer": ["SELECT * FROM mock"],
        }
    )

    env = TextToSQLEnvironment(rubric=rubric, parser=parser, dataset=mock_dataset)

    class MockTokenizer:
        def encode(self, text, _add_special_tokens=True):
            return text.split()

    tokenizer = MockTokenizer()
    formatter = GRPODatasetFormatter(env, tokenizer)

    # Create a formatted dataset
    data = {
        "prompt": [
            "This is a short prompt",
            "This is a much longer prompt with many words",
        ],
        "question": ["Q1", "Q2"],
        "schema": ["S1", "S2"],
    }
    dataset = Dataset.from_dict(data)

    stats = formatter.validate_tokenization(dataset, max_length=10)

    assert "avg_token_length" in stats
    assert "max_token_length" in stats
    assert "too_long_count" in stats
    assert stats["max_token_length"] > 0


def test_evaluation_set_creation():
    """Test creation of evaluation set."""
    from src.environments.sql_env.environment import TextToSQLEnvironment
    from src.rubrics.sql_rubric import SQLValidationRubric
    from src.utils.sql_parser import SQLParser

    rubric = SQLValidationRubric()
    parser = SQLParser()

    # Create a minimal mock dataset (required by verifiers base class)
    mock_dataset = Dataset.from_dict(
        {
            "question": ["Mock question?"],
            "context": ["CREATE TABLE mock (id INT)"],
            "answer": ["SELECT * FROM mock"],
        }
    )

    env = TextToSQLEnvironment(rubric=rubric, parser=parser, dataset=mock_dataset)

    class MockTokenizer:
        def encode(self, text, _add_special_tokens=True):
            return text.split()

    tokenizer = MockTokenizer()
    formatter = GRPODatasetFormatter(env, tokenizer)

    # Create dataset with complexity labels
    data = {
        "prompt": ["P" + str(i) for i in range(100)],
        "question": ["Q" + str(i) for i in range(100)],
        "schema": ["S" + str(i) for i in range(100)],
        "complexity": ["simple"] * 50 + ["medium"] * 30 + ["complex"] * 20,
    }
    dataset = Dataset.from_dict(data)

    eval_set = formatter.create_evaluation_set(dataset, n_samples=20)

    assert len(eval_set) <= 20
    assert "complexity" in eval_set.column_names


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_pipeline():
    """Test the complete data pipeline."""
    # Create sample dataset
    dataset = Dataset.from_dict(SAMPLE_DATASET)

    # 1. Load
    loader = SQLDatasetLoader(seed=42)

    # 2. Preprocess
    preprocessor = SQLDataPreprocessor()
    processed = preprocessor.preprocess_dataset(dataset, num_proc=1)
    processed = preprocessor.filter_dataset(processed)

    # 3. Compute statistics
    stats = loader.get_statistics(processed)

    assert stats["total_samples"] > 0
    assert "avg_question_length" in stats

    # 4. Format for GRPO
    from src.environments.sql_env.environment import TextToSQLEnvironment
    from src.rubrics.sql_rubric import SQLValidationRubric
    from src.utils.sql_parser import SQLParser

    rubric = SQLValidationRubric()
    parser = SQLParser()

    # Create a minimal mock dataset (required by verifiers base class)
    mock_dataset = Dataset.from_dict(
        {
            "question": ["Mock question?"],
            "context": ["CREATE TABLE mock (id INT)"],
            "answer": ["SELECT * FROM mock"],
        }
    )

    env = TextToSQLEnvironment(rubric=rubric, parser=parser, dataset=mock_dataset)

    class MockTokenizer:
        def encode(self, text, _add_special_tokens=True):
            return text.split()

    tokenizer = MockTokenizer()
    formatter = GRPODatasetFormatter(env, tokenizer)

    formatted = formatter.format_dataset(processed)

    assert "prompt" in formatted.column_names
    assert len(formatted) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
