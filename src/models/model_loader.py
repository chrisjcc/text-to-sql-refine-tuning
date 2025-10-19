"""Model loader with QLoRA quantization and LoRA adapter support.

This module provides the ModelLoader class for loading and configuring
language models with QLoRA (4-bit quantization) and LoRA adapters for
memory-efficient fine-tuning on A100 GPUs.
"""

import logging
from typing import Optional, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ModelLoader:
    """
    Loads and configures language models for GRPO fine-tuning.
    Supports QLoRA quantization and LoRA adapters.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ):
        """
        Initialize model loader.

        Args:
            model_name: HuggingFace model identifier
            cache_dir: Directory for caching model weights
            device_map: Device mapping strategy
            trust_remote_code: Whether to trust remote code
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.logger = logging.getLogger(__name__)

    def create_bnb_config(
        self,
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "bfloat16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
    ) -> BitsAndBytesConfig:
        """
        Create BitsAndBytes quantization configuration for QLoRA.

        Args:
            load_in_4bit: Use 4-bit quantization
            bnb_4bit_compute_dtype: Compute dtype (bfloat16/float16)
            bnb_4bit_quant_type: Quantization type (nf4/fp4)
            bnb_4bit_use_double_quant: Use nested quantization

        Returns:
            BitsAndBytesConfig for model loading
        """
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        )

        self.logger.info(f"Created BitsAndBytes config: {bnb_4bit_quant_type} quantization")
        return bnb_config

    def create_lora_config(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None,
        bias: str = "none",
        task_type: TaskType = TaskType.CAUSAL_LM,
    ) -> LoraConfig:
        """
        Create LoRA configuration for PEFT.

        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability
            target_modules: Modules to apply LoRA to
            bias: Bias training strategy
            task_type: Type of task

        Returns:
            LoraConfig for PEFT
        """
        if target_modules is None:
            # Default for Llama models
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            task_type=task_type,
        )

        self.logger.info(
            f"Created LoRA config: r={r}, alpha={lora_alpha}, " f"targets={target_modules}"
        )
        return lora_config

    def load_model(
        self,
        use_quantization: bool = True,
        use_peft: bool = True,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        lora_config: Optional[LoraConfig] = None,
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = "auto",
    ) -> AutoModelForCausalLM:
        """
        Load model with optional quantization and LoRA adapters.

        Args:
            use_quantization: Whether to use QLoRA
            use_peft: Whether to apply LoRA adapters
            bnb_config: Custom BnB config (created if None)
            lora_config: Custom LoRA config (created if None)
            torch_dtype: Base dtype for non-quantized loading
            attn_implementation: Attention implementation ("flash_attention_2", "sdpa", "eager", or "auto")
                                "auto" will try flash_attention_2 and fall back to default if unavailable

        Returns:
            Loaded model (with PEFT if enabled)
        """
        self.logger.info(f"Loading model: {self.model_name}")

        # Prepare quantization config
        if use_quantization:
            if bnb_config is None:
                bnb_config = self.create_bnb_config()
            quantization_config = bnb_config
            # CRITICAL: Set torch_dtype even when using quantization
            # This ensures non-quantized layers (like lm_head) use the same dtype as quantized layers' compute dtype
            # Without this, lm_head defaults to float32 while quantized layers compute in bfloat16, causing dtype mismatch
            if torch_dtype is None:
                torch_dtype = bnb_config.bnb_4bit_compute_dtype
                self.logger.info(
                    f"Setting torch_dtype to {torch_dtype} to match quantization compute dtype"
                )
        else:
            quantization_config = None
            if torch_dtype is None:
                torch_dtype = torch.bfloat16

        # Determine attention implementation
        attn_impl = None
        if attn_implementation == "auto":
            # Try flash_attention_2 if CUDA is available
            if torch.cuda.is_available():
                try:
                    # Test if flash_attn can be imported
                    import flash_attn  # noqa: F401

                    attn_impl = "flash_attention_2"
                    self.logger.info("Using Flash Attention 2 for improved performance")
                except (ImportError, ModuleNotFoundError):
                    self.logger.warning(
                        "Flash Attention 2 not available. Install with: pip install flash-attn --no-build-isolation"
                    )
                    attn_impl = None  # Use default attention
        elif attn_implementation is not None:
            attn_impl = attn_implementation

        # Load base model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=self.device_map,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
                torch_dtype=torch_dtype,  # Always set torch_dtype, even with quantization
                attn_implementation=attn_impl,
            )
        except Exception as e:
            # If flash attention fails during model loading, retry without it
            if attn_impl == "flash_attention_2" and "flash_attn" in str(e):
                self.logger.warning(f"Failed to load model with Flash Attention 2: {str(e)}")
                self.logger.info("Retrying with default attention implementation...")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map=self.device_map,
                    trust_remote_code=self.trust_remote_code,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch_dtype,  # Always set torch_dtype, even with quantization
                    attn_implementation=None,
                )
            else:
                raise

        self.logger.info(
            f"Model loaded. Memory footprint: " f"{model.get_memory_footprint() / 1e9:.2f} GB"
        )

        # CRITICAL FIX: Explicitly cast lm_head weight DATA (not just the module) when using quantization
        # The issue: .to(dtype) on modules doesn't always persist with quantized models
        # Solution: Directly cast the weight.data tensor in-place
        if use_quantization and bnb_config is not None:
            compute_dtype = bnb_config.bnb_4bit_compute_dtype
            if hasattr(model, "lm_head") and model.lm_head is not None:
                if hasattr(model.lm_head, "weight") and model.lm_head.weight is not None:
                    # Cast the weight data directly (in-place)
                    original_dtype = model.lm_head.weight.dtype
                    model.lm_head.weight.data = model.lm_head.weight.data.to(compute_dtype)
                    self.logger.info(
                        f"✓ Cast lm_head.weight.data from {original_dtype} to {compute_dtype}"
                    )

                    # Verify the cast succeeded
                    actual_dtype = model.lm_head.weight.dtype
                    if actual_dtype == compute_dtype:
                        self.logger.info(f"✓ Verified: lm_head.weight is {actual_dtype}")
                    else:
                        self.logger.warning(
                            f"⚠ Verification failed: lm_head.weight is {actual_dtype}, expected {compute_dtype}"
                        )

        # Apply PEFT if requested
        if use_peft:
            self.logger.info("Preparing model for k-bit training")
            model = prepare_model_for_kbit_training(model)

            if lora_config is None:
                lora_config = self.create_lora_config()

            self.logger.info("Applying LoRA adapters")
            model = get_peft_model(model, lora_config)

            # CRITICAL FIX PART 2: Re-cast lm_head weight DATA after PEFT wrapping
            # PEFT wrapping may create new model wrappers, need to cast at all levels
            if use_quantization and bnb_config is not None:
                compute_dtype = bnb_config.bnb_4bit_compute_dtype

                # Try multiple access paths through PEFT's nested structure
                models_to_check = []

                # Path 1: Direct access (top-level PEFT wrapper)
                if hasattr(model, "lm_head"):
                    models_to_check.append(("peft_wrapper", model))

                # Path 2: get_base_model() method
                if hasattr(model, "get_base_model"):
                    base_model = model.get_base_model()
                    if hasattr(base_model, "lm_head"):
                        models_to_check.append(("base_model", base_model))

                # Path 3: base_model.model (nested PEFT structure)
                if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
                    underlying_model = model.base_model.model
                    if hasattr(underlying_model, "lm_head"):
                        models_to_check.append(("underlying_model", underlying_model))

                # Cast lm_head at ALL levels found
                for model_name, model_obj in models_to_check:
                    if (
                        hasattr(model_obj.lm_head, "weight")
                        and model_obj.lm_head.weight is not None
                    ):
                        original_dtype = model_obj.lm_head.weight.dtype
                        if original_dtype != compute_dtype:
                            model_obj.lm_head.weight.data = model_obj.lm_head.weight.data.to(
                                compute_dtype
                            )
                            self.logger.info(
                                f"✓ Cast {model_name}.lm_head.weight.data from {original_dtype} to {compute_dtype}"
                            )

                        # Verify
                        actual_dtype = model_obj.lm_head.weight.dtype
                        if actual_dtype == compute_dtype:
                            self.logger.info(
                                f"✓ Verified {model_name}.lm_head.weight is {actual_dtype}"
                            )
                        else:
                            self.logger.warning(
                                f"⚠ {model_name}.lm_head.weight is {actual_dtype}, expected {compute_dtype}"
                            )

            # Print trainable parameters
            trainable_params, all_param = model.get_nb_trainable_parameters()
            self.logger.info(
                f"Trainable params: {trainable_params:,} || "
                f"All params: {all_param:,} || "
                f"Trainable %: {100 * trainable_params / all_param:.2f}"
            )

        return model

    def load_tokenizer(
        self, padding_side: str = "left", add_eos_token: bool = True, add_bos_token: bool = False
    ) -> AutoTokenizer:
        """
        Load and configure tokenizer.

        Args:
            padding_side: Which side to pad on
            add_eos_token: Add EOS token to sequences
            add_bos_token: Add BOS token to sequences

        Returns:
            Configured tokenizer
        """
        self.logger.info(f"Loading tokenizer: {self.model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            cache_dir=self.cache_dir,
            padding_side=padding_side,
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Configure special tokens
        tokenizer.add_eos_token = add_eos_token
        tokenizer.add_bos_token = add_bos_token

        self.logger.info(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
        self.logger.info(
            f"Special tokens: PAD={tokenizer.pad_token}, "
            f"EOS={tokenizer.eos_token}, BOS={tokenizer.bos_token}"
        )

        return tokenizer

    def load_model_and_tokenizer(
        self, use_quantization: bool = True, use_peft: bool = True, **kwargs
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Convenience method to load both model and tokenizer.

        Returns:
            (model, tokenizer)
        """
        model = self.load_model(use_quantization=use_quantization, use_peft=use_peft, **kwargs)
        tokenizer = self.load_tokenizer()

        return model, tokenizer

    def print_model_info(self, model: AutoModelForCausalLM):
        """Print detailed model information for debugging."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Model Information:")
        self.logger.info(f"Model type: {type(model).__name__}")
        self.logger.info(f"Device: {model.device}")
        self.logger.info(f"Dtype: {model.dtype}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

        # Memory info
        if torch.cuda.is_available():
            self.logger.info(
                f"GPU Memory allocated: " f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
            )
            self.logger.info(
                f"GPU Memory reserved: " f"{torch.cuda.memory_reserved() / 1e9:.2f} GB"
            )

        self.logger.info("=" * 80 + "\n")
