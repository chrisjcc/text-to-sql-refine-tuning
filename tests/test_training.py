"""Unit tests for training module."""

from unittest.mock import Mock, patch

import pytest
from datasets import Dataset
from omegaconf import DictConfig

from src.environments.sql_env.environment import TextToSQLEnvironment
from src.rubrics.sql_rubric import SQLValidationRubric
from src.training.callbacks import SQLEvaluationCallback, WandbLoggingCallback
from src.training.config_builder import GRPOConfigBuilder
from src.training.grpo_trainer import SQLGRPOTrainer
from src.utils.sql_parser import SQLParser


@pytest.fixture
def mock_model():
    """Create mock model."""
    model = Mock()
    model.device = "cpu"
    model.config = Mock()
    return model


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    tokenizer.encode = Mock(return_value=[1, 2, 3])
    tokenizer.decode = Mock(return_value="SELECT * FROM users")
    return tokenizer


@pytest.fixture
def sql_parser():
    """Create SQL parser."""
    return SQLParser()


@pytest.fixture
def sql_rubric(sql_parser):
    """Create SQL rubric."""
    return SQLValidationRubric(parser=sql_parser)


@pytest.fixture
def text_to_sql_env(sql_rubric, sql_parser, sample_dataset):
    """Create text-to-SQL environment."""
    return TextToSQLEnvironment(
        rubric=sql_rubric,
        parser=sql_parser,
        prompt_template="default",
        dataset=sample_dataset
    )


@pytest.fixture
def sample_dataset():
    """Create sample dataset."""
    return Dataset.from_dict({
        'prompt': ['Generate SQL for: Get all users', 'Generate SQL for: Count products'],
        'question': ['Get all users', 'Count products'],
        'schema': ['CREATE TABLE users (id INT)', 'CREATE TABLE products (id INT)'],
        'reference': ['SELECT * FROM users', 'SELECT COUNT(*) FROM products']
    })


@pytest.fixture
def hydra_config():
    """Create mock Hydra configuration."""
    config = {
        'project': {
            'seed': 42
        },
        'dataset': {
            'num_workers': 4
        },
        'training': {
            'output_dir': './outputs',
            'run_name': 'test-run',
            'logging_steps': 10,
            'num_train_epochs': 3,
            'max_steps': -1,
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,
            'learning_rate': 5e-6,
            'lr_scheduler_type': 'cosine',
            'warmup_steps': 100,
            'warmup_ratio': 0.03,
            'optim': 'adamw_torch',
            'bf16': True,
            'gradient_checkpointing': True,
            'max_new_tokens': 256,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True,
            'num_generations': 4,
            'kl_coef': 0.05,
            'gamma': 1.0,
            'save_strategy': 'steps',
            'save_steps': 500,
            'save_total_limit': 3,
            'evaluation_strategy': 'steps',
            'eval_steps': 500,
        },
        'wandb': {
            'enabled': False
        }
    }
    return DictConfig(config)


def test_grpo_config_creation():
    """Test GRPO config creation with default parameters."""
    # Mock GRPOTrainer to avoid initialization issues
    with patch('src.training.grpo_trainer.GRPOTrainer'):
        # Create properly mocked model with valid config._name_or_path
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config._name_or_path = "meta-llama/Llama-3-8B"

        trainer = SQLGRPOTrainer(
            model=mock_model,
            tokenizer=Mock(),
            environment=Mock(),
            rubric=Mock(),
            train_dataset=Mock()
        )

        config = trainer.create_default_config()

        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps == 8
        assert config.learning_rate == 5e-6
        assert config.num_generations == 4


def test_grpo_config_from_hydra(hydra_config):
    """Test GRPO config building from Hydra configuration."""
    config = GRPOConfigBuilder.build_from_hydra(hydra_config)

    assert config.output_dir == './outputs'
    assert config.run_name == 'test-run'
    assert config.num_train_epochs == 3
    assert config.learning_rate == 5e-6
    assert config.num_generations == 4
    assert config.seed == 42
    assert config.report_to == []


def test_reward_computation(sql_rubric):
    """Test reward computation."""
    responses = [
        "SELECT * FROM users WHERE id = 1",
        "This is not SQL",
        "SELECT COUNT(*) FROM products"
    ]

    rewards = sql_rubric.score_batch(responses)

    assert len(rewards) == 3
    assert all(0.0 <= r <= 1.0 for r in rewards)
    assert rewards[0] > rewards[1]  # Valid SQL should score higher
    assert rewards[2] > rewards[1]


def test_trainer_initialization(
    mock_model,
    mock_tokenizer,
    text_to_sql_env,
    sql_rubric,
    sample_dataset
):
    """Test GRPO trainer initialization."""
    with patch('src.training.grpo_trainer.GRPOTrainer'):
        trainer = SQLGRPOTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            environment=text_to_sql_env,
            rubric=sql_rubric,
            train_dataset=sample_dataset,
            eval_dataset=None
        )

        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.environment == text_to_sql_env
        assert trainer.rubric == sql_rubric
        assert trainer.train_dataset == sample_dataset
        assert trainer.config is not None


def test_evaluation_callback(
    text_to_sql_env,
    sql_rubric,
    sample_dataset,
    mock_tokenizer
):
    """Test SQL evaluation callback."""
    callback = SQLEvaluationCallback(
        environment=text_to_sql_env,
        rubric=sql_rubric,
        eval_dataset=sample_dataset,
        tokenizer=mock_tokenizer,
        eval_frequency=100,
        num_samples=2
    )

    assert callback.eval_frequency == 100
    assert callback.num_samples == 2
    assert callback.environment == text_to_sql_env


def test_wandb_callback():
    """Test WandB logging callback."""
    config = {'project': 'test', 'name': 'test-run'}

    callback = WandbLoggingCallback(config=config)

    assert callback.config == config


def test_training_step(
    mock_model,
    mock_tokenizer,
    text_to_sql_env,
    sql_rubric,
    sample_dataset
):
    """Test training step (mock test)."""
    with patch('src.training.grpo_trainer.GRPOTrainer') as MockGRPOTrainer:
        mock_trainer = Mock()
        MockGRPOTrainer.return_value = mock_trainer

        trainer = SQLGRPOTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            environment=text_to_sql_env,
            rubric=sql_rubric,
            train_dataset=sample_dataset
        )

        # Test train method
        trainer.train()
        mock_trainer.train.assert_called_once()


def test_checkpoint_saving(
    mock_model,
    mock_tokenizer,
    text_to_sql_env,
    sql_rubric,
    sample_dataset,
    tmp_path
):
    """Test checkpoint saving."""
    with patch('src.training.grpo_trainer.GRPOTrainer') as MockGRPOTrainer:
        mock_trainer = Mock()
        MockGRPOTrainer.return_value = mock_trainer

        trainer = SQLGRPOTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            environment=text_to_sql_env,
            rubric=sql_rubric,
            train_dataset=sample_dataset
        )

        output_dir = str(tmp_path / "checkpoint")
        trainer.save_model(output_dir)

        mock_trainer.save_model.assert_called_once_with(output_dir)
        mock_tokenizer.save_pretrained.assert_called_once_with(output_dir)


def test_model_generation(text_to_sql_env):
    """Test model generation parameters."""
    prompt = text_to_sql_env.format_prompt(
        question="Get all users",
        context={'schema': 'CREATE TABLE users (id INT, name VARCHAR)'}
    )

    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "users" in prompt.lower()


def test_compute_rewards_integration(
    mock_model,
    mock_tokenizer,
    text_to_sql_env,
    sql_rubric,
    sample_dataset
):
    """Test compute_rewards method integration."""
    with patch('src.training.grpo_trainer.GRPOTrainer'):
        trainer = SQLGRPOTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            environment=text_to_sql_env,
            rubric=sql_rubric,
            train_dataset=sample_dataset
        )

        prompts = ["Generate SQL", "Another prompt"]
        responses = ["SELECT * FROM users", "SELECT COUNT(*) FROM products"]

        rewards = trainer.compute_rewards(prompts, responses)

        assert len(rewards) == len(responses)
        assert all(0.0 <= r <= 1.0 for r in rewards)


def test_evaluation_callback_step_end(
    text_to_sql_env,
    sql_rubric,
    sample_dataset,
    mock_tokenizer
):
    """Test evaluation callback on_step_end."""
    callback = SQLEvaluationCallback(
        environment=text_to_sql_env,
        rubric=sql_rubric,
        eval_dataset=sample_dataset,
        tokenizer=mock_tokenizer,
        eval_frequency=100,
        num_samples=2
    )

    from transformers import TrainerState, TrainerControl

    state = TrainerState()
    state.global_step = 100
    control = TrainerControl()

    with patch.object(callback, 'run_evaluation') as mock_eval:
        callback.on_step_end(None, state, control)
        mock_eval.assert_called_once_with(100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
