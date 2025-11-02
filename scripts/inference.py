"""Batch inference script for evaluation and testing.

This script runs inference on a dataset using the fine-tuned model,
computes evaluation metrics, and saves results.
"""

import json
from pathlib import Path
from typing import Any

import hydra
from datasets import Dataset, DatasetDict, load_from_disk
from omegaconf import DictConfig

from src.inference.inference_engine import SQLInferenceEngine  # noqa: E402
from src.utils.logging_utils import setup_logging_from_config  # noqa: E402

# Add src to path
# sys.path.insert(0, str(Path(__file__).parent.parent))


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run_inference(cfg: DictConfig) -> None:
    """Run inference on dataset using fine-tuned SQL generation model.

    Loads a fine-tuned model, runs inference on a specified dataset or
    sample questions, computes evaluation metrics, and saves detailed
    results to disk.

    Args:
        cfg: Hydra configuration object containing inference parameters
            including model_path, dataset_path, batch_size, generation
            parameters, and output settings.

    Returns:
        None. Saves inference results and metrics to output directory.

    Raises:
        FileNotFoundError: If specified model or dataset paths don't exist.
        ValueError: If dataset doesn't contain expected splits or columns.
    """
    logger = setup_logging_from_config(cfg)
    logger.info("Starting inference")

    # Load model
    model_path = cfg.inference.get("model_path", "./outputs/final_model")
    logger.info(f"Loading model from {model_path}")

    engine = SQLInferenceEngine(
        model_path=model_path,
        base_model_name=cfg.hf.model.get("name"),
        load_in_4bit=cfg.inference.get("load_in_4bit", False),
    )

    # Load dataset
    dataset_path = cfg.inference.get("dataset_path")
    eval_data: list[dict[str, Any]]

    if dataset_path:
        logger.info(f"Loading dataset from {dataset_path}")
        dataset: DatasetDict = load_from_disk(dataset_path)

        # Use test or validation split
        split = cfg.inference.get("split", "test")
        if split not in dataset:
            split = "validation"
            logger.warning(f"Split 'test' not found, using '{split}' instead")

        data: Dataset = dataset[split]
        logger.info(f"Using {split} split: {len(data)} samples")

        # Convert to list of dicts
        eval_data = [
            {
                "question": item["question"],
                "schema": item.get("schema"),
                "sql": item.get("sql"),
            }
            for item in data
        ]
    else:
        # Use sample questions
        logger.info("No dataset path provided, using sample questions")
        eval_data = [
            {
                "question": "What are the names of all users?",
                "schema": (
                    "CREATE TABLE users (id INT, name VARCHAR(100), " "email VARCHAR(100));"
                ),
                "sql": "SELECT name FROM users;",
            }
        ]

    # Run inference
    logger.info(f"Running inference on {len(eval_data)} samples")

    results: list[dict[str, Any]] = engine.batch_generate_sql(
        questions=[item["question"] for item in eval_data],
        schemas=[item.get("schema") for item in eval_data],
        batch_size=cfg.inference.get("batch_size", 4),
        max_new_tokens=cfg.inference.get("max_new_tokens", 256),
        temperature=cfg.inference.get("temperature", 0.1),
        top_p=cfg.inference.get("top_p", 0.95),
        num_beams=cfg.inference.get("num_beams", 1),
        do_sample=cfg.inference.get("do_sample", False),
    )

    # Evaluate
    metrics: dict[str, Any] = engine.evaluate_on_dataset(eval_data)

    # Save results
    output_dir = Path(cfg.inference.get("output_dir", "./inference_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "inference_results.json"
    logger.info(f"Saving results to {results_path}")

    with results_path.open("w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)

    # Also save metrics separately for easy access
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Inference complete!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    run_inference()
