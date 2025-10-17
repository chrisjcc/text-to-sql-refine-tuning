"""Prepare dataset for GRPO training.

Loads, preprocesses, and formats the SQL dataset.
"""

import hydra
from omegaconf import DictConfig
from pathlib import Path
import logging
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_loader import SQLDatasetLoader
from src.data.preprocessor import SQLDataPreprocessor
from src.utils.logging_utils import setup_logging


@hydra.main(version_base=None, config_path="../config", config_name="config")
def prepare_data(cfg: DictConfig):
    """
    Complete data preparation pipeline.

    Steps:
    1. Load dataset from HuggingFace
    2. Create splits if needed
    3. Preprocess and clean samples
    4. Filter invalid samples
    5. Compute statistics
    6. Save processed dataset
    """
    # Setup logging
    logger = setup_logging(
        log_level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        log_file="data_preparation.log"
    )
    logger.info("=" * 80)
    logger.info("Starting data preparation pipeline")
    logger.info("=" * 80)

    # Load dataset
    loader = SQLDatasetLoader(
        dataset_name=cfg.dataset.name,
        cache_dir=cfg.dataset.cache_dir,
        seed=cfg.project.seed
    )

    logger.info(f"Loading dataset: {cfg.dataset.name}")
    try:
        dataset = loader.load()
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Check if we need to create splits
    if isinstance(dataset, dict):
        # Dataset already has splits
        logger.info(f"Dataset has existing splits: {list(dataset.keys())}")
    else:
        # Single dataset, need to create splits
        if cfg.dataset.create_splits.enabled:
            logger.info("Creating train/val/test splits")
            dataset = loader.create_splits(
                dataset,
                train_size=cfg.dataset.create_splits.train_size,
                val_size=cfg.dataset.create_splits.val_size,
                test_size=cfg.dataset.create_splits.test_size,
                stratify=cfg.dataset.create_splits.stratify_by_complexity
            )
        else:
            # Convert to DatasetDict with just train split
            from datasets import DatasetDict
            dataset = DatasetDict({'train': dataset})
            logger.info("Using entire dataset as training set")

    # Log initial statistics
    logger.info("\n" + "=" * 80)
    logger.info("Initial dataset statistics:")
    logger.info("=" * 80)
    for split_name in dataset.keys():
        logger.info(f"\n{split_name.upper()} split:")
        logger.info(f"  Samples: {len(dataset[split_name])}")
        stats = loader.get_statistics(dataset[split_name])
        for key, value in stats.items():
            if key != 'sql_keyword_distribution':
                logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}:")
                for keyword, count in value.items():
                    logger.info(f"    {keyword}: {count}")

    # Initialize preprocessor
    preprocessor = SQLDataPreprocessor(
        max_question_length=cfg.dataset.preprocessing.max_question_length,
        max_schema_length=cfg.dataset.preprocessing.max_schema_length,
        max_sql_length=cfg.dataset.preprocessing.max_sql_length,
        normalize_sql=cfg.dataset.preprocessing.normalize_sql,
        filter_invalid=cfg.dataset.preprocessing.filter_invalid
    )

    # Preprocess each split
    logger.info("\n" + "=" * 80)
    logger.info("Preprocessing dataset")
    logger.info("=" * 80)

    for split_name in dataset.keys():
        logger.info(f"\nProcessing {split_name} split...")

        # Preprocess
        dataset[split_name] = preprocessor.preprocess_dataset(
            dataset[split_name],
            num_proc=cfg.dataset.num_workers
        )

        # Filter invalid samples
        if cfg.dataset.preprocessing.filter_invalid:
            dataset[split_name] = preprocessor.filter_dataset(
                dataset[split_name]
            )

    # Compute final statistics
    logger.info("\n" + "=" * 80)
    logger.info("Final dataset statistics:")
    logger.info("=" * 80)

    for split_name in dataset.keys():
        logger.info(f"\n{split_name.upper()} split:")
        logger.info(f"  Total samples: {len(dataset[split_name])}")

        # Compute complexity distribution
        if 'complexity' in dataset[split_name].column_names:
            complexities = dataset[split_name]['complexity']
            complexity_dist = {}
            for c in complexities:
                complexity_dist[c] = complexity_dist.get(c, 0) + 1

            logger.info("  Complexity distribution:")
            for complexity, count in sorted(complexity_dist.items()):
                pct = (count / len(dataset[split_name])) * 100
                logger.info(f"    {complexity}: {count} ({pct:.1f}%)")

        # Compute validation statistics
        if 'is_valid' in dataset[split_name].column_names:
            valid_count = sum(dataset[split_name]['is_valid'])
            valid_pct = (valid_count / len(dataset[split_name])) * 100
            logger.info(f"  Valid samples: {valid_count} ({valid_pct:.1f}%)")

        # Length statistics
        if all(col in dataset[split_name].column_names for col in ['question_length', 'sql_length', 'schema_length']):
            import numpy as np
            logger.info(f"  Avg question length: {np.mean(dataset[split_name]['question_length']):.1f} words")
            logger.info(f"  Avg SQL length: {np.mean(dataset[split_name]['sql_length']):.1f} words")
            logger.info(f"  Avg schema length: {np.mean(dataset[split_name]['schema_length']):.1f} words")

    # Save processed dataset
    output_path = Path(cfg.dataset.cache_dir) / "processed"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 80)
    logger.info(f"Saving processed dataset to {output_path}")
    logger.info("=" * 80)

    try:
        dataset.save_to_disk(str(output_path))
        logger.info(f"Successfully saved dataset to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        raise

    logger.info("\n" + "=" * 80)
    logger.info("Data preparation complete!")
    logger.info("=" * 80)

    return dataset


if __name__ == "__main__":
    prepare_data()
