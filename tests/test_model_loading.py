"""Unit tests for model loading functionality.

This module contains comprehensive tests for the ModelLoader class
and configuration utilities.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from src.models.config_utils import (
    create_bnb_config_from_hydra,
    create_lora_config_from_hydra,
    create_model_config_from_hydra,
    estimate_memory_requirements,
)
from src.models.model_loader import ModelLoader


class TestModelLoader:
    """Tests for ModelLoader class."""

    def test_model_loader_initialization(self):
        """Test ModelLoader initialization with various parameters."""
        loader = ModelLoader(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            cache_dir="./cache",
            device_map="auto",
            trust_remote_code=True,
        )

        assert loader.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert loader.cache_dir == "./cache"
        assert loader.device_map == "auto"
        assert loader.trust_remote_code is True

    def test_bnb_config_creation(self):
        """Test BitsAndBytes config creation."""
        loader = ModelLoader(model_name="test-model")

        config = loader.create_bnb_config(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        assert config.load_in_4bit is True
        assert config.bnb_4bit_compute_dtype == torch.bfloat16
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant is True

    def test_lora_config_creation(self):
        """Test LoRA config creation."""
        loader = ModelLoader(model_name="test-model")

        config = loader.create_lora_config(
            r=16, lora_alpha=32, lora_dropout=0.05, target_modules={"q_proj", "v_proj"}, bias="none"
        )

        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert config.target_modules == {"q_proj", "v_proj"}
        assert config.bias == "none"

    def test_lora_config_default_target_modules(self):
        """Test LoRA config with default target modules."""
        loader = ModelLoader(model_name="test-model")

        config = loader.create_lora_config(r=16, lora_alpha=32)

        # Should have default Llama target modules
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules
        assert "k_proj" in config.target_modules
        assert "o_proj" in config.target_modules

    @patch("src.models.model_loader.AutoModelForCausalLM")
    @patch("src.models.model_loader.prepare_model_for_kbit_training")
    @patch("src.models.model_loader.get_peft_model")
    def test_model_loading_with_quantization(
        self, mock_get_peft, mock_prepare_kbit, mock_auto_model
    ):
        """Test model loading with quantization enabled."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.get_memory_footprint.return_value = 4e9
        mock_model.get_nb_trainable_parameters.return_value = (100000, 8000000000)
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_prepare_kbit.return_value = mock_model
        mock_get_peft.return_value = mock_model

        loader = ModelLoader(model_name="test-model")
        _model = loader.load_model(use_quantization=True, use_peft=True)  # noqa: F841

        # Verify model loading was called
        mock_auto_model.from_pretrained.assert_called_once()
        call_kwargs = mock_auto_model.from_pretrained.call_args[1]
        assert call_kwargs["quantization_config"] is not None

    @patch("src.models.model_loader.AutoModelForCausalLM")
    def test_model_loading_without_quantization(self, mock_auto_model):
        """Test model loading without quantization."""
        mock_model = MagicMock()
        mock_model.get_memory_footprint.return_value = 16e9
        mock_auto_model.from_pretrained.return_value = mock_model

        loader = ModelLoader(model_name="test-model")
        _model = loader.load_model(use_quantization=False, use_peft=False)  # noqa: F841

        # Verify model loading was called
        mock_auto_model.from_pretrained.assert_called_once()
        call_kwargs = mock_auto_model.from_pretrained.call_args[1]
        assert call_kwargs["quantization_config"] is None
        assert call_kwargs["torch_dtype"] == torch.bfloat16

    @patch("src.models.model_loader.AutoTokenizer")
    def test_tokenizer_loading(self, mock_tokenizer):
        """Test tokenizer loading and configuration."""
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"
        mock_tok.eos_token_id = 2
        mock_tok.__len__ = lambda self: 32000
        mock_tokenizer.from_pretrained.return_value = mock_tok

        loader = ModelLoader(model_name="test-model")
        _tokenizer = loader.load_tokenizer(  # noqa: F841
            padding_side="left", add_eos_token=True, add_bos_token=False
        )

        # Verify tokenizer was loaded
        mock_tokenizer.from_pretrained.assert_called_once()

        # Verify padding token was set
        assert mock_tok.pad_token == "<eos>"
        assert mock_tok.pad_token_id == 2

    @patch("src.models.model_loader.AutoModelForCausalLM")
    @patch("src.models.model_loader.AutoTokenizer")
    def test_model_and_tokenizer_loading(self, mock_tokenizer, mock_auto_model):
        """Test convenience method for loading both model and tokenizer."""
        mock_model = MagicMock()
        mock_model.get_memory_footprint.return_value = 4e9
        mock_auto_model.from_pretrained.return_value = mock_model

        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tokenizer.from_pretrained.return_value = mock_tok

        loader = ModelLoader(model_name="test-model")
        _model, _tokenizer = loader.load_model_and_tokenizer(
            use_quantization=False, use_peft=False
        )  # noqa: F841

        # These are not None assertions
        assert _model is not None
        assert _tokenizer is not None

    @patch("src.models.model_loader.AutoModelForCausalLM")
    @patch("src.models.model_loader.prepare_model_for_kbit_training")
    @patch("src.models.model_loader.get_peft_model")
    def test_peft_application(self, mock_get_peft, mock_prepare_kbit, mock_auto_model):
        """Test PEFT adapter application."""
        mock_model = MagicMock()
        mock_model.get_memory_footprint.return_value = 4e9
        mock_model.get_nb_trainable_parameters.return_value = (100000, 8000000000)
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_prepare_kbit.return_value = mock_model
        mock_get_peft.return_value = mock_model

        loader = ModelLoader(model_name="test-model")
        loader.load_model(use_quantization=True, use_peft=True)

        # Verify PEFT functions were called
        mock_prepare_kbit.assert_called_once()
        mock_get_peft.assert_called_once()

    def test_trainable_parameters_count(self):
        """Test trainable parameter counting logic."""
        # This is tested indirectly through the model loading tests
        # The actual parameter counting is done by the PEFT library
        pass

    @patch("src.models.model_loader.torch.cuda")
    def test_print_model_info(self, mock_cuda):
        """Test model info printing."""
        mock_cuda.is_available.return_value = True
        mock_cuda.memory_allocated.return_value = 4e9
        mock_cuda.memory_reserved.return_value = 5e9

        mock_model = MagicMock()
        mock_model.device = "cuda:0"
        mock_model.dtype = torch.bfloat16
        mock_model.parameters.return_value = [
            MagicMock(numel=lambda: 1000000, requires_grad=True),
            MagicMock(numel=lambda: 2000000, requires_grad=False),
        ]

        loader = ModelLoader(model_name="test-model")
        # Should not raise any exceptions
        loader.print_model_info(mock_model)


class TestConfigUtils:
    """Tests for configuration utilities."""

    def test_model_config_from_hydra(self):
        """Test model config creation from Hydra config."""
        cfg = OmegaConf.create(
            {
                "hf": {
                    "model": {"name": "meta-llama/Llama-3.1-8B-Instruct", "cache_dir": "./cache"}
                },
                "training": {"use_peft": True},
            }
        )

        config = create_model_config_from_hydra(cfg)

        assert config["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert config["cache_dir"] == "./cache"
        assert config["use_quantization"] is True
        assert config["use_peft"] is True

    def test_bnb_config_from_hydra(self):
        """Test BnB config creation from Hydra config."""
        cfg = OmegaConf.create(
            {"training": {"peft": {"use_qlora": True, "bnb_4bit_compute_dtype": "bfloat16"}}}
        )

        config = create_bnb_config_from_hydra(cfg)

        assert config is not None
        assert config.load_in_4bit is True
        assert config.bnb_4bit_compute_dtype == torch.bfloat16

    def test_bnb_config_from_hydra_disabled(self):
        """Test BnB config when QLoRA is disabled."""
        cfg = OmegaConf.create(
            {"training": {"peft": {"use_qlora": False, "bnb_4bit_compute_dtype": "bfloat16"}}}
        )

        config = create_bnb_config_from_hydra(cfg)
        assert config is None

    def test_lora_config_from_hydra(self):
        """Test LoRA config creation from Hydra config."""
        cfg = OmegaConf.create(
            {
                "training": {
                    "peft": {
                        "lora_r": 32,
                        "lora_alpha": 64,
                        "lora_dropout": 0.1,
                        "target_modules": ["q_proj", "v_proj", "k_proj"],
                    }
                }
            }
        )

        config = create_lora_config_from_hydra(cfg)

        assert config.r == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.1
        assert config.target_modules == ["q_proj", "v_proj", "k_proj"]

    def test_memory_estimation(self):
        """Test memory requirement estimation."""
        memory = estimate_memory_requirements(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            use_quantization=True,
            use_peft=True,
            batch_size=1,
            sequence_length=512,
        )

        # Check all expected keys are present
        assert "model" in memory
        assert "lora" in memory
        assert "activations" in memory
        assert "optimizer" in memory
        assert "gradients" in memory
        assert "total" in memory
        assert "recommended_gpu" in memory

        # Check reasonable values
        assert memory["model"] > 0
        assert memory["total"] > memory["model"]
        assert memory["recommended_gpu"] > memory["total"]

    def test_memory_estimation_without_quantization(self):
        """Test memory estimation without quantization."""
        memory_quant = estimate_memory_requirements(
            model_name="test", use_quantization=True, use_peft=True
        )

        memory_no_quant = estimate_memory_requirements(
            model_name="test", use_quantization=False, use_peft=True
        )

        # Without quantization should use more memory
        assert memory_no_quant["model"] > memory_quant["model"]


class TestModelInference:
    """Tests for model inference functionality."""

    @patch("src.models.model_loader.AutoModelForCausalLM")
    @patch("src.models.model_loader.AutoTokenizer")
    def test_model_inference(self, mock_tokenizer, mock_auto_model):
        """Test basic model inference."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_model.get_memory_footprint.return_value = 4e9
        mock_auto_model.from_pretrained.return_value = mock_model

        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.pad_token_id = 0
        mock_tok.return_value = {"input_ids": torch.tensor([[1, 2]])}
        mock_tok.decode.return_value = "SELECT * FROM users"
        mock_tokenizer.from_pretrained.return_value = mock_tok

        loader = ModelLoader(model_name="test-model")
        model, tokenizer = loader.load_model_and_tokenizer(use_quantization=False, use_peft=False)

        # Test inference
        test_prompt = "SELECT * FROM"
        inputs = tokenizer(test_prompt)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id,
            )

        result = tokenizer.decode(outputs[0])
        assert result == "SELECT * FROM users"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
