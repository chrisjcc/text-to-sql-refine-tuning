"""
Publish trained model to HuggingFace Hub.
"""
import argparse
import json
import logging
import os
from pathlib import Path

from huggingface_hub import create_repo, upload_folder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_model_card(
    model_path: str, repo_name: str, base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
) -> str:
    """Create model card markdown."""

    # Try to load training metrics if available
    metrics_path = Path(model_path).parent / "trainer_state.json"
    metrics_info = ""

    if metrics_path.exists():
        with open(metrics_path) as f:
            state = json.load(f)
            if "log_history" in state:
                last_metrics = state["log_history"][-1]
                metrics_info = f"""
## Training Metrics

- Final Loss: {last_metrics.get('loss', 'N/A')}
- Final Reward: {last_metrics.get('reward', 'N/A')}
- Training Steps: {last_metrics.get('step', 'N/A')}
"""

    model_card = f"""---
license: apache-2.0
base_model: {base_model}
tags:
- text-to-sql
- grpo
- qlora
- llama
- sql-generation
datasets:
- b-mc2/sql-create-context
language:
- en
pipeline_tag: text-generation
---

# {repo_name}

Fine-tuned Llama-3.1-8B-Instruct model for text-to-SQL generation using GRPO (Group Relative Policy Optimization).

## Model Description

This model converts natural language questions into SQL queries. It was fine-tuned using:
- **Base Model**: {base_model}
- **Training Method**: GRPO with QLoRA
- **Dataset**: b-mc2/sql-create-context
- **Framework**: Verifiers + TRL

{metrics_info}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model}",
    torch_dtype="auto",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")

# Generate SQL
question = "What are the names of all users?"
schema = "CREATE TABLE users (id INT, name VARCHAR(100));"

prompt = f\"\"\"Given the following database schema:

{{schema}}

Generate a SQL query to answer this question:
Question: {{question}}

SQL Query:\"\"\"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(sql)
```

## Training Details

- **Training Data**: ~78k SQL examples from b-mc2/sql-create-context
- **Hardware**: NVIDIA A100 40GB
- **Optimization**: QLoRA (4-bit quantization)
- **Reward Function**: Custom SQL validation rubric (syntax, keywords, format)
- **GRPO Parameters**:
  - Learning rate: 5e-6
  - Num generations: 4
  - KL coefficient: 0.05

## Evaluation

See the [GitHub repository](https://github.com/chrisjcc/text-to-sql-refine-tuning) for detailed evaluation metrics and benchmarks.

## Limitations

- Trained on specific SQL schema patterns
- May not generalize to all SQL dialects
- Requires schema context for best results
- Not optimized for very complex queries

## Citation

```bibtex
@software{{text_to_sql_grpo_2025,
  title={{Text-to-SQL Fine-Tuning with GRPO}},
  author={{chrisjcc}},
  year={{2025}},
  url={{https://github.com/chrisjcc/text-to-sql-refine-tuning}}
}}
```

## License

Apache 2.0
"""
    return model_card


def publish_model(model_path: str, repo_name: str, private: bool = True, token: str = None):
    """
    Publish model to HuggingFace Hub.

    Args:
        model_path: Path to model checkpoint
        repo_name: Repository name on HuggingFace (username/model-name)
        private: Whether to create private repository
        token: HuggingFace API token (or use HF_TOKEN env var)
    """
    logger.info(f"Publishing model to HuggingFace Hub: {repo_name}")

    # Get token
    if token is None:
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise ValueError("HF_TOKEN not found in environment")

    # Create repository
    logger.info(f"Creating repository: {repo_name}")
    try:
        create_repo(repo_id=repo_name, token=token, private=private, exist_ok=True)
    except Exception as e:
        logger.warning(f"Repository creation warning: {e}")

    # Create model card
    logger.info("Creating model card")
    model_card = create_model_card(model_path, repo_name)

    model_card_path = Path(model_path) / "README.md"
    with open(model_card_path, "w") as f:
        f.write(model_card)

    # Upload model
    logger.info("Uploading model files")
    upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        token=token,
        commit_message="Upload fine-tuned model",
    )

    logger.info("✅ Model published successfully!")
    logger.info(f"View at: https://huggingface.co/{repo_name}")


def main():
    parser = argparse.ArgumentParser(description="Publish model to HuggingFace Hub")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="HuggingFace repository name (username/model-name)",
    )
    parser.add_argument(
        "--private",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Make repository private",
    )
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token")

    args = parser.parse_args()

    publish_model(
        model_path=args.model_path, repo_name=args.repo_name, private=args.private, token=args.token
    )


if __name__ == "__main__":
    main()
