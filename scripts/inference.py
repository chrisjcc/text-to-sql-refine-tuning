"""Batch inference script for evaluation and testing.

This script runs inference on a dataset using the fine-tuned model,
computes evaluation metrics, and saves results.
"""

import hydra
from omegaconf import DictConfig
import json
from pathlib import Path
from datasets import load_from_disk
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.inference.inference_engine import SQLInferenceEngine
from src.utils.logging_utils import setup_logging


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run_inference(cfg: DictConfig):
    """
    Run inference on dataset.
    """
    logger = setup_logging(cfg)
    logger.info("Starting inference")

    # Load model
    model_path = cfg.inference.get('model_path', './outputs/final_model')
    logger.info(f"Loading model from {model_path}")

    engine = SQLInferenceEngine(
        model_path=model_path,
        base_model_name=cfg.hf.model.get('name'),
        load_in_4bit=cfg.inference.get('load_in_4bit', False)
    )

    # Load dataset
    dataset_path = cfg.inference.get('dataset_path')
    if dataset_path:
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = load_from_disk(dataset_path)

        # Use test or validation split
        split = cfg.inference.get('split', 'test')
        if split not in dataset:
            split = 'validation'
            logger.warning(f"Split 'test' not found, using '{split}' instead")

        data = dataset[split]
        logger.info(f"Using {split} split: {len(data)} samples")

        # Convert to list of dicts
        eval_data = [
            {
                'question': item['question'],
                'schema': item.get('schema'),
                'sql': item.get('sql')
            }
            for item in data
        ]
    else:
        # Use sample questions
        logger.info("No dataset path provided, using sample questions")
        eval_data = [
            {
                'question': 'What are the names of all users?',
                'schema': 'CREATE TABLE users (id INT, name VARCHAR(100), email VARCHAR(100));',
                'sql': 'SELECT name FROM users;'
            }
        ]

    # Run inference
    logger.info(f"Running inference on {len(eval_data)} samples")

    results = engine.batch_generate_sql(
        questions=[item['question'] for item in eval_data],
        schemas=[item.get('schema') for item in eval_data],
        batch_size=cfg.inference.get('batch_size', 4),
        max_new_tokens=cfg.inference.get('max_new_tokens', 256),
        temperature=cfg.inference.get('temperature', 0.1),
        top_p=cfg.inference.get('top_p', 0.95),
        num_beams=cfg.inference.get('num_beams', 1),
        do_sample=cfg.inference.get('do_sample', False)
    )

    # Evaluate
    metrics = engine.evaluate_on_dataset(eval_data)

    # Save results
    output_dir = Path(cfg.inference.get('output_dir', './inference_results'))
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "inference_results.json"
    logger.info(f"Saving results to {results_path}")

    with open(results_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'results': results
        }, f, indent=2)

    # Also save metrics separately for easy access
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info("Inference complete!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    run_inference()
