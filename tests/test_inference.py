"""Unit tests for inference module.

Tests for inference engine, CLI, and API components.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Mock imports to avoid loading heavy dependencies in tests
@pytest.fixture(autouse=True)
def mock_torch_and_transformers():
    """Mock torch and transformers to avoid GPU dependencies in tests."""
    with patch('src.inference.inference_engine.torch') as mock_torch, \
         patch('src.inference.inference_engine.AutoModelForCausalLM') as mock_model, \
         patch('src.inference.inference_engine.AutoTokenizer') as mock_tokenizer, \
         patch('src.inference.inference_engine.PeftModel') as mock_peft:

        # Setup mock torch
        mock_torch.no_grad = MagicMock()
        mock_torch.bfloat16 = 'bfloat16'

        # Setup mock model
        mock_model_instance = MagicMock()
        mock_model_instance.device = 'cpu'
        mock_model_instance.dtype = 'bfloat16'
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Setup mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = '[PAD]'
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer_instance.eos_token = '[EOS]'
        mock_tokenizer_instance.eos_token_id = 1
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Setup mock PEFT
        mock_peft_instance = MagicMock()
        mock_peft_instance.merge_and_unload.return_value = mock_model_instance
        mock_peft.from_pretrained.return_value = mock_peft_instance

        yield {
            'torch': mock_torch,
            'model': mock_model,
            'tokenizer': mock_tokenizer,
            'peft': mock_peft,
            'model_instance': mock_model_instance,
            'tokenizer_instance': mock_tokenizer_instance
        }


@pytest.fixture
def mock_environment():
    """Create a mock environment."""
    env = MagicMock()
    env.format_prompt.return_value = "Test prompt"
    env.parse_response.return_value = {
        'sql': 'SELECT * FROM users',
        'valid': True,
        'metadata': {}
    }
    return env


@pytest.fixture
def mock_parser():
    """Create a mock parser."""
    parser = MagicMock()
    parser.extract_sql.return_value = "SELECT * FROM users"
    return parser


@pytest.fixture
def temp_model_dir():
    """Create a temporary model directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal adapter config for PEFT detection
        config_path = Path(tmpdir) / "adapter_config.json"
        config_path.write_text(json.dumps({
            "base_model_name_or_path": "test-model"
        }))
        yield tmpdir


class TestInferenceEngine:
    """Tests for SQLInferenceEngine."""

    def test_inference_engine_initialization(self, temp_model_dir, mock_environment, mock_parser):
        """Test that inference engine initializes correctly."""
        from src.inference.inference_engine import SQLInferenceEngine

        engine = SQLInferenceEngine(
            model_path=temp_model_dir,
            base_model_name="test-model",
            environment=mock_environment,
            parser=mock_parser
        )

        assert engine.model_path == temp_model_dir
        assert engine.base_model_name == "test-model"
        assert engine.environment == mock_environment
        assert engine.parser == mock_parser

    def test_single_sql_generation(self, temp_model_dir, mock_environment, mock_parser, mock_torch_and_transformers):
        """Test single SQL generation."""
        from src.inference.inference_engine import SQLInferenceEngine

        # Setup mocks
        mock_tokenizer = mock_torch_and_transformers['tokenizer_instance']
        mock_model = mock_torch_and_transformers['model_instance']

        # Create a mock object that has .to() method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_tokenizer.decode.return_value = "Test prompt SELECT * FROM users"

        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]

        engine = SQLInferenceEngine(
            model_path=temp_model_dir,
            base_model_name="test-model",
            environment=mock_environment,
            parser=mock_parser
        )

        result = engine.generate_sql(
            question="Show all users",
            schema="CREATE TABLE users (id INT, name VARCHAR)"
        )

        assert 'sql' in result
        assert 'raw_output' in result
        assert 'valid' in result
        assert 'metadata' in result

    def test_batch_sql_generation(self, temp_model_dir, mock_environment, mock_parser, mock_torch_and_transformers):
        """Test batch SQL generation."""
        from src.inference.inference_engine import SQLInferenceEngine

        # Setup mocks
        mock_tokenizer = mock_torch_and_transformers['tokenizer_instance']
        mock_model = mock_torch_and_transformers['model_instance']

        # Create a mock object that has .to() method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_tokenizer.decode.side_effect = [
            "Test prompt SELECT * FROM users",
            "Test prompt SELECT * FROM products"
        ]

        mock_model.generate.return_value = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]]

        engine = SQLInferenceEngine(
            model_path=temp_model_dir,
            base_model_name="test-model",
            environment=mock_environment,
            parser=mock_parser
        )

        questions = ["Show all users", "Show all products"]
        results = engine.batch_generate_sql(questions, batch_size=2)

        assert len(results) == 2
        assert all('sql' in r for r in results)
        assert all('question' in r for r in results)

    def test_model_loading_peft(self, temp_model_dir, mock_torch_and_transformers):
        """Test loading PEFT model."""
        from src.inference.inference_engine import SQLInferenceEngine

        # Mock environment and parser imports - they're imported inside __init__
        with patch('src.rubrics.sql_rubric.SQLValidationRubric'), \
             patch('src.environments.sql_env.TextToSQLEnvironment'):

            _engine = SQLInferenceEngine(  # noqa: F841
                model_path=temp_model_dir,
                base_model_name="test-model"
            )

            # Verify PEFT loading was called
            mock_torch_and_transformers['peft'].from_pretrained.assert_called_once()

    def test_model_loading_full(self, mock_torch_and_transformers):
        """Test loading full fine-tuned model."""
        from src.inference.inference_engine import SQLInferenceEngine

        # Create temp dir without adapter config (full model)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.rubrics.sql_rubric.SQLValidationRubric'), \
                 patch('src.environments.sql_env.TextToSQLEnvironment'):

                _engine = SQLInferenceEngine(  # noqa: F841
                    model_path=tmpdir,
                    base_model_name="test-model"
                )

                # Verify full model loading was called
                mock_torch_and_transformers['model'].from_pretrained.assert_called()

    def test_sql_parsing(self, temp_model_dir, mock_environment, mock_torch_and_transformers):
        """Test SQL parsing in results."""
        from src.inference.inference_engine import SQLInferenceEngine
        from src.utils.sql_parser import SQLParser

        parser = SQLParser()

        # Mock the parse to return specific SQL
        mock_environment.parse_response.return_value = {
            'sql': 'SELECT id, name FROM users WHERE active = 1',
            'valid': True,
            'metadata': {'tables': ['users']}
        }

        mock_tokenizer = mock_torch_and_transformers['tokenizer_instance']
        mock_model = mock_torch_and_transformers['model_instance']

        # Create a mock object that has .to() method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_tokenizer.decode.return_value = "Test prompt SELECT id, name FROM users WHERE active = 1"
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]

        engine = SQLInferenceEngine(
            model_path=temp_model_dir,
            base_model_name="test-model",
            environment=mock_environment,
            parser=parser
        )

        result = engine.generate_sql("Show active users")

        assert result['sql'] == 'SELECT id, name FROM users WHERE active = 1'
        assert result['valid'] is True
        assert 'tables' in result['metadata']

    def test_evaluation_on_dataset(self, temp_model_dir, mock_environment, mock_parser, mock_torch_and_transformers):
        """Test evaluation on dataset."""
        from src.inference.inference_engine import SQLInferenceEngine

        # Setup mocks
        mock_tokenizer = mock_torch_and_transformers['tokenizer_instance']
        mock_model = mock_torch_and_transformers['model_instance']

        # Create a mock object that has .to() method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_tokenizer.decode.return_value = "Test prompt SELECT * FROM users"
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]

        engine = SQLInferenceEngine(
            model_path=temp_model_dir,
            base_model_name="test-model",
            environment=mock_environment,
            parser=mock_parser
        )

        dataset = [
            {
                'question': 'Show all users',
                'schema': 'CREATE TABLE users (id INT)',
                'sql': 'SELECT * FROM users'
            }
        ]

        metrics = engine.evaluate_on_dataset(dataset)

        assert 'total_samples' in metrics
        assert 'valid_sql_pct' in metrics
        assert 'avg_reward' in metrics
        assert metrics['total_samples'] == 1


class TestAPI:
    """Tests for REST API."""

    def test_api_health_endpoint(self):
        """Test health endpoint."""
        from src.inference.api import create_app
        from fastapi.testclient import TestClient

        mock_engine = MagicMock()
        app = create_app(mock_engine)
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_api_generate_endpoint(self, mock_environment, mock_parser, temp_model_dir):
        """Test generate endpoint."""
        from src.inference.api import create_app
        from fastapi.testclient import TestClient

        mock_engine = MagicMock()
        mock_engine.generate_sql.return_value = {
            'sql': 'SELECT * FROM users',
            'raw_output': 'SELECT * FROM users',
            'valid': True,
            'metadata': {}
        }

        app = create_app(mock_engine)
        client = TestClient(app)

        response = client.post("/generate", json={
            "question": "Show all users",
            "schema": "CREATE TABLE users (id INT)"
        })

        assert response.status_code == 200
        data = response.json()
        assert 'sql' in data
        assert 'valid' in data
        assert data['sql'] == 'SELECT * FROM users'

    def test_api_batch_generate_endpoint(self):
        """Test batch generate endpoint."""
        from src.inference.api import create_app
        from fastapi.testclient import TestClient

        mock_engine = MagicMock()
        mock_engine.batch_generate_sql.return_value = [
            {
                'sql': 'SELECT * FROM users',
                'raw_output': 'SELECT * FROM users',
                'valid': True,
                'metadata': {},
                'question': 'Show all users',
                'schema': None
            },
            {
                'sql': 'SELECT * FROM products',
                'raw_output': 'SELECT * FROM products',
                'valid': True,
                'metadata': {},
                'question': 'Show all products',
                'schema': None
            }
        ]

        app = create_app(mock_engine)
        client = TestClient(app)

        response = client.post("/batch_generate", json={
            "questions": ["Show all users", "Show all products"]
        })

        assert response.status_code == 200
        data = response.json()
        assert 'results' in data
        assert 'total_count' in data
        assert data['total_count'] == 2
        assert len(data['results']) == 2


class TestCLI:
    """Tests for interactive CLI."""

    def test_cli_initialization(self):
        """Test CLI initialization."""
        from src.inference.cli import SQLInteractiveCLI

        mock_engine = MagicMock()
        cli = SQLInteractiveCLI(mock_engine)

        assert cli.engine == mock_engine
        assert cli.schema is None

    def test_cli_set_schema(self):
        """Test setting schema."""
        from src.inference.cli import SQLInteractiveCLI

        mock_engine = MagicMock()
        cli = SQLInteractiveCLI(mock_engine)

        schema = "CREATE TABLE users (id INT)"
        cli.set_schema(schema)

        assert cli.schema == schema

    def test_cli_load_schema_from_file(self):
        """Test loading schema from file."""
        from src.inference.cli import SQLInteractiveCLI

        mock_engine = MagicMock()
        cli = SQLInteractiveCLI(mock_engine)

        # Create temp file with schema
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sql') as f:
            f.write("CREATE TABLE users (id INT)")
            temp_path = f.name

        try:
            cli.load_schema_from_file(temp_path)
            assert cli.schema == "CREATE TABLE users (id INT)"
        finally:
            os.unlink(temp_path)

    def test_cli_generate(self):
        """Test CLI generate method."""
        from src.inference.cli import SQLInteractiveCLI

        mock_engine = MagicMock()
        mock_engine.generate_sql.return_value = {
            'sql': 'SELECT * FROM users',
            'raw_output': 'SELECT * FROM users',
            'valid': True,
            'metadata': {}
        }

        cli = SQLInteractiveCLI(mock_engine)
        result = cli.generate("Show all users")

        assert result['sql'] == 'SELECT * FROM users'
        mock_engine.generate_sql.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
