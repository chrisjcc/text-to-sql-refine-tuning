"""Inference engine for text-to-SQL generation.

This module provides the main inference engine for generating SQL queries
from natural language questions using fine-tuned models.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)


class SQLInferenceEngine:
    """
    Inference engine for text-to-SQL generation.
    Supports single and batch predictions.
    """

    def __init__(
        self,
        model_path: str,
        base_model_name: Optional[str] = None,
        device: str = "auto",
        load_in_4bit: bool = False,
        environment: Optional[Any] = None,
        parser: Optional[Any] = None,
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to fine-tuned model (with adapters)
            base_model_name: Base model name (if loading adapters separately)
            device: Device to load model on
            load_in_4bit: Whether to use 4-bit quantization for inference
            environment: Text-to-SQL environment for prompt formatting
            parser: SQL parser for post-processing
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.logger = logging.getLogger(__name__)

        # Import parser if not provided
        if parser is None:
            from ..utils.sql_parser import SQLParser

            parser = SQLParser()
        self.parser = parser

        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()

        # Setup environment
        if environment is None:
            from ..environments.sql_env import TextToSQLEnvironment
            from ..rubrics.sql_rubric import SQLValidationRubric

            rubric = SQLValidationRubric(sql_keywords=[])
            environment = TextToSQLEnvironment(
                rubric=rubric, parser=self.parser, prompt_template="instructional"
            )
        self.environment = environment

        self.logger.info("Inference engine initialized")

    def _load_model(self) -> Tuple[Union[AutoModelForCausalLM, PeftModel, PreTrainedModel], AutoTokenizer]:
        """Load model and tokenizer."""
        self.logger.info(f"Loading model from {self.model_path}")

        # Check if this is a local path
        # Convert to absolute path if it's a relative path
        model_path_obj = Path(self.model_path)
        if not model_path_obj.is_absolute():
            # Resolve relative to current working directory
            model_path_obj = Path.cwd() / model_path_obj

        is_local_path = model_path_obj.exists()

        # Use the absolute path string for loading
        resolved_model_path = str(model_path_obj) if is_local_path else self.model_path

        # Check if this is a PEFT model
        peft_config_path = model_path_obj / "adapter_config.json"
        is_peft_model = peft_config_path.exists()

        model: Union[PeftModel, PreTrainedModel]

        if is_peft_model:
            self.logger.info("Detected PEFT model, loading with adapters")

            # Determine base model
            if self.base_model_name is None:
                with open(peft_config_path) as f:
                    config = json.load(f)
                    self.base_model_name = config.get("base_model_name_or_path")

            self.logger.info(f"Loading base model: {self.base_model_name}")

            # Load base model
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    quantization_config=bnb_config,
                    device_map=self.device,
                    trust_remote_code=True,
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    device_map=self.device,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )

            # Load PEFT adapters
            self.logger.info(f"Loading PEFT adapters from {resolved_model_path}")
            model = PeftModel.from_pretrained(
                base_model, resolved_model_path, local_files_only=is_local_path
            )
            model = model.merge_and_unload()  # Merge adapters for faster inference

        else:
            # Load full fine-tuned model
            self.logger.info("Loading full fine-tuned model")
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_path,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=is_local_path,
            )

        # Load tokenizer - determine source path
        tokenizer_path = resolved_model_path if not is_peft_model else (self.base_model_name or resolved_model_path)
        is_tokenizer_local = Path(tokenizer_path).exists() if tokenizer_path else False


        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, local_files_only=is_tokenizer_local
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model.eval()

        self.logger.info(f"Model loaded. Device: {model.device}, Dtype: {model.dtype}")

        return model, tokenizer

    def generate_sql(
        self,
        question: str,
        schema: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.95,
        num_beams: int = 1,
        do_sample: bool = False,
        **generation_kwargs,
    ) -> Dict[str, Any]:
        """
        Generate SQL query from natural language question.

        Args:
            question: Natural language question
            schema: Database schema (CREATE TABLE statements)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            **generation_kwargs: Additional generation parameters

        Returns:
            Dict with:
                - sql: Generated SQL query
                - raw_output: Raw model output
                - valid: Whether SQL is valid
                - confidence: Generation confidence (if available)
                - metadata: Additional information
        """
        # Format prompt
        context = {"schema": schema} if schema else None
        prompt = self.environment.format_prompt(question, context)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(
            self.model.device
        )

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract SQL from output
        sql_output = generated_text[len(prompt) :].strip()

        # Parse and validate
        parsed = self.environment.parse_response(sql_output)

        return {
            "sql": parsed.get("sql", sql_output),
            "raw_output": sql_output,
            "valid": parsed.get("valid", False),
            "metadata": parsed.get("metadata", {}),
            "prompt": prompt,
        }

    def batch_generate_sql(
        self,
        questions: List[str],
        schemas: Optional[List[str]] = None,
        batch_size: int = 4,
        **generation_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate SQL queries for multiple questions in batches.

        Args:
            questions: List of natural language questions
            schemas: List of database schemas (one per question)
            batch_size: Batch size for processing
            **generation_kwargs: Generation parameters

        Returns:
            List of result dictionaries
        """
        if schemas is None:
            schemas = [None] * len(questions)  # type: ignore[list-item]

        if len(questions) != len(schemas):
            raise ValueError("Number of questions and schemas must match")

        results = []

        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i : i + batch_size]
            batch_schemas = schemas[i : i + batch_size]

            # Process batch
            batch_prompts = []
            for q, s in zip(batch_questions, batch_schemas):
                context = {"schema": s} if s else None
                prompt = self.environment.format_prompt(q, context)
                batch_prompts.append(prompt)

            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode and parse
            for j, output in enumerate(outputs):
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                sql_output = generated_text[len(batch_prompts[j]) :].strip()
                parsed = self.environment.parse_response(sql_output)

                results.append(
                    {
                        "sql": parsed.get("sql", sql_output),
                        "raw_output": sql_output,
                        "valid": parsed.get("valid", False),
                        "metadata": parsed.get("metadata", {}),
                        "question": batch_questions[j],
                        "schema": batch_schemas[j],
                    }
                )

        return results

    def evaluate_on_dataset(self, dataset: List[Dict], **generation_kwargs) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.

        Args:
            dataset: List of dicts with 'question', 'schema', 'sql' (reference)
            **generation_kwargs: Generation parameters

        Returns:
            Dict with evaluation metrics
        """
        self.logger.info(f"Evaluating on {len(dataset)} samples")

        questions = [item["question"] for item in dataset]
        schemas: List[Optional[str]] = [item.get("schema") for item in dataset]

        references = [item.get("sql") for item in dataset]

        # Generate predictions
        predictions = self.batch_generate_sql(questions, schemas, **generation_kwargs)  # type: ignore[arg-type]

        # Compute metrics
        from ..rubrics.sql_rubric import SQLValidationRubric

        rubric = SQLValidationRubric(sql_keywords=[])

        valid_count = sum(1 for p in predictions if p["valid"])
        rewards = [rubric.score(p["sql"]) for p in predictions]

        metrics = {
            "total_samples": len(dataset),
            "valid_sql_pct": valid_count / len(dataset) * 100,
            "avg_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
        }

        # Compute exact match if references available
        if all(r is not None for r in references):  # type: ignore[arg-type]
            exact_matches = sum(
                1 if self._normalize_sql(p["sql"]) == self._normalize_sql(r) else 0  # type: ignore[arg-type, misc]
                for p, r in zip(predictions, references)
                if r is not None and self._normalize_sql(p["sql"]) == self._normalize_sql(r)

            )
            metrics["exact_match_pct"] = exact_matches / len(dataset) * 100  # type: ignore[assignment]

        self.logger.info("Evaluation complete")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.2f}")

        return metrics

    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for comparison."""
        import sqlparse

        return str(sqlparse.format(sql, keyword_case="upper", strip_whitespace=True).strip())
