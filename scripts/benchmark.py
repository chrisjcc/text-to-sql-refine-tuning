"""
Benchmark multiple model checkpoints on test dataset.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import hydra
import pandas as pd
from datasets import load_from_disk
from omegaconf import DictConfig

from src.evaluation.evaluator import SQLEvaluator
from src.evaluation.metrics import SQLMetrics
from src.inference.inference_engine import SQLInferenceEngine
from src.utils.logging_utils import setup_logging_from_config


@hydra.main(version_base=None, config_path="../config", config_name="config")
def benchmark(cfg: DictConfig):
    """
    Benchmark multiple model checkpoints.
    """
    logger = setup_logging_from_config(cfg)
    logger.info("Starting benchmark evaluation")

    # Load dataset
    dataset_path = Path(cfg.dataset.cache_dir) / "processed"
    dataset = load_from_disk(str(dataset_path))

    # Use test or validation split
    split = cfg.evaluation.split or "validation"
    data = dataset[split]

    logger.info(f"Loaded {split} split: {len(data)} samples")

    # Convert to evaluation format
    eval_data = [
        {
            "question": item["question"],
            "schema": item.get("schema"),
            "sql": item["sql"],
            "complexity": item.get("complexity", "unknown"),
        }
        for item in data
    ]

    # Get checkpoint paths
    checkpoint_paths = cfg.evaluation.checkpoint_paths

    if not checkpoint_paths:
        # Use default output directory (resolve to absolute path)
        output_dir = Path(cfg.training.output_dir)
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        checkpoint_paths = [str(output_dir / "final_model")]

    # Evaluate each checkpoint
    all_results = {}

    for checkpoint_path in checkpoint_paths:
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating checkpoint: {checkpoint_path}")
        logger.info(f"{'='*80}\n")

        # Load model
        engine = SQLInferenceEngine(
            model_path=checkpoint_path,
            base_model_name=cfg.hf.model.name,
            load_in_4bit=cfg.evaluation.load_in_4bit,
        )

        # Create evaluator
        metrics = SQLMetrics()
        evaluator = SQLEvaluator(inference_engine=engine, metrics=metrics)

        # Run evaluation
        results = evaluator.evaluate_dataset(
            dataset=eval_data,
            batch_size=cfg.evaluation.batch_size,
            compute_execution=False,
            max_new_tokens=cfg.inference.max_new_tokens,
            temperature=cfg.inference.temperature,
        )

        # Save results
        checkpoint_name = Path(checkpoint_path).name
        output_path = Path(cfg.evaluation.output_dir) / checkpoint_name
        evaluator.generate_report(results, str(output_path))

        all_results[checkpoint_name] = results["aggregate"]

        # Log summary
        logger.info(f"\nResults for {checkpoint_name}:")
        for key, value in results["aggregate"].items():
            logger.info(f"  {key}: {value:.2f}")

    # Compare checkpoints
    if len(all_results) > 1:
        logger.info(f"\n{'='*80}")
        logger.info("Checkpoint Comparison")
        logger.info(f"{'='*80}\n")

        comparison_df = pd.DataFrame(all_results).T
        comparison_path = Path(cfg.evaluation.output_dir) / "checkpoint_comparison.csv"
        comparison_df.to_csv(comparison_path)

        logger.info(comparison_df.to_string())
        logger.info(f"\nComparison saved to {comparison_path}")

    logger.info("\nBenchmark complete!")


if __name__ == "__main__":
    benchmark()
