"""Prepare dataset for GRPO training.

Loads, preprocesses, and formats the SQL dataset.
"""

import logging
import sys
from pathlib import Path

import hydra
import numpy as np
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_loader import SQLDatasetLoader  # noqa: E402
from data.preprocessor import SQLDataPreprocessor  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402


def _load_dataset_with_split(
    cfg: DictConfig, loader: SQLDatasetLoader, logger: logging.Logger
) -> Dataset | DatasetDict:
    """Load dataset with optional split specifications.

    Args:
        cfg: Hydra configuration object containing dataset split and
            limit specifications.
        loader: SQLDatasetLoader instance for loading the dataset.
        logger: Logger instance for logging status messages.

    Returns:
        Loaded dataset, either as a single Dataset or DatasetDict
        depending on the split configuration.

    Raises:
        Exception: If dataset loading fails, re-raises the original
            exception after logging.
    """
    split_arg = None
    if cfg.dataset.split.train is not None:
        if cfg.dataset.limit.train is not None:
            split_arg = f"{cfg.dataset.split.train}[:{cfg.dataset.limit.train}]"
            logger.info(f"Loading limited train split: {split_arg}")
        else:
            split_arg = cfg.dataset.split.train

    try:
        return loader.load(split=split_arg)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def _ensure_dataset_splits(
    dataset: Dataset | DatasetDict,
    cfg: DictConfig,
    loader: SQLDatasetLoader,
    logger: logging.Logger,
) -> DatasetDict:
    """Ensure dataset has proper train/val/test splits.

    Args:
        dataset: Input dataset, either as single Dataset or DatasetDict.
        cfg: Hydra configuration object containing split creation settings.
        loader: SQLDatasetLoader instance for creating splits.
        logger: Logger instance for logging status messages.

    Returns:
        DatasetDict with train/validation/test splits.
    """
    if isinstance(dataset, dict):
        logger.info(f"Dataset has existing splits: {list(dataset.keys())}")
        return dataset

    if cfg.dataset.create_splits.enabled:
        logger.info("Creating train/val/test splits")
        return loader.create_splits(
            dataset,
            train_size=cfg.dataset.create_splits.train_size,
            val_size=cfg.dataset.create_splits.val_size,
            test_size=cfg.dataset.create_splits.test_size,
            stratify=cfg.dataset.create_splits.stratify_by_complexity,
        )

    logger.info("Using entire dataset as training set")
    return DatasetDict({"train": dataset})


def _log_dataset_statistics(
    dataset: DatasetDict, loader: SQLDatasetLoader, logger: logging.Logger
) -> None:
    """Log statistics for all dataset splits.

    Args:
        dataset: DatasetDict containing all splits to analyze.
        loader: SQLDatasetLoader instance for computing statistics.
        logger: Logger instance for logging statistics.

    Returns:
        None. Logs statistics to the logger.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Initial dataset statistics:")
    logger.info("=" * 80)
    for split_name in dataset:
        logger.info(f"\n{split_name.upper()} split:")
        logger.info(f"  Samples: {len(dataset[split_name])}")
        stats = loader.get_statistics(dataset[split_name])
        for key, value in stats.items():
            if key != "sql_keyword_distribution":
                logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}:")
                for keyword, count in value.items():
                    logger.info(f"    {keyword}: {count}")


def _preprocess_splits(
    dataset: DatasetDict,
    preprocessor: SQLDataPreprocessor,
    cfg: DictConfig,
    logger: logging.Logger,
) -> None:
    """Preprocess all dataset splits.

    Args:
        dataset: DatasetDict containing all splits to preprocess.
        preprocessor: SQLDataPreprocessor instance for preprocessing.
        cfg: Hydra configuration object containing preprocessing settings.
        logger: Logger instance for logging progress.

    Returns:
        None. Modifies dataset in place.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Preprocessing dataset")
    logger.info("=" * 80)

    for split_name in dataset:
        logger.info(f"\nProcessing {split_name} split...")
        dataset[split_name] = preprocessor.preprocess_dataset(
            dataset[split_name], num_proc=cfg.dataset.num_workers
        )
        if cfg.dataset.preprocessing.filter_invalid:
            dataset[split_name] = preprocessor.filter_dataset(dataset[split_name])


def _compute_final_statistics(dataset: DatasetDict, logger: logging.Logger) -> None:
    """Compute and log final statistics for all splits.

    Args:
        dataset: DatasetDict containing all splits to analyze.
        logger: Logger instance for logging statistics.

    Returns:
        None. Logs final statistics including complexity distribution,
        validity rates, and average lengths.
    """
    logger.info("\n" + "=" * 80)
    logger.info("Final dataset statistics:")
    logger.info("=" * 80)

    for split_name in dataset:
        logger.info(f"\n{split_name.upper()} split:")
        logger.info(f"  Total samples: {len(dataset[split_name])}")

        if "complexity" in dataset[split_name].column_names:
            complexities = dataset[split_name]["complexity"]
            complexity_dist: dict[str, int] = {}
            for c in complexities:
                complexity_dist[c] = complexity_dist.get(c, 0) + 1

            logger.info("  Complexity distribution:")
            for complexity, count in sorted(complexity_dist.items()):
                pct = (count / len(dataset[split_name])) * 100
                logger.info(f"    {complexity}: {count} ({pct:.1f}%)")

        if "is_valid" in dataset[split_name].column_names:
            valid_count = sum(dataset[split_name]["is_valid"])
            valid_pct = (valid_count / len(dataset[split_name])) * 100
            logger.info(f"  Valid samples: {valid_count} ({valid_pct:.1f}%)")

        if all(
            col in dataset[split_name].column_names
            for col in ["question_length", "sql_length", "schema_length"]
        ):
            q_len = dataset[split_name]["question_length"]
            sql_len = dataset[split_name]["sql_length"]
            schema_len = dataset[split_name]["schema_length"]

            logger.info(f"  Avg question length: {np.mean(q_len):.1f} words")
            logger.info(f"  Avg SQL length: {np.mean(sql_len):.1f} words")
            logger.info(f"  Avg schema length: {np.mean(schema_len):.1f} words")


def _save_processed_dataset(
    dataset: DatasetDict,
    cfg: DictConfig,
    logger: logging.Logger
) -> None:
    """Save the processed dataset to disk.

    Args:
        dataset: DatasetDict to save.
        cfg: Hydra configuration object containing output path settings.
        logger: Logger instance for logging save status.

    Returns:
        None. Saves dataset to disk.

    Raises:
        Exception: If dataset saving fails, re-raises the original
            exception after logging.
    """
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


@hydra.main(version_base=None, config_path="../config", config_name="config")
def prepare_data(cfg: DictConfig) -> DatasetDict:
    """Complete data preparation pipeline.

    Executes the full data preparation workflow including loading,
    splitting, preprocessing, filtering, and saving the dataset.

    The pipeline performs the following steps:
    1. Load dataset from HuggingFace
    2. Create train/val/test splits if needed
    3. Preprocess and clean samples
    4. Filter invalid samples
    5. Compute and log statistics
    6. Save processed dataset to disk

    Args:
        cfg: Hydra configuration object containing all pipeline settings
            including dataset name, preprocessing parameters, split
            configurations, and output paths.

    Returns:
        DatasetDict containing the fully processed train/validation/test
        splits ready for training.

    Raises:
        Exception: If any step in the pipeline fails (loading, preprocessing,
            or saving).
    """
    logger = setup_logging(
        log_level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        log_file="data_preparation.log",
    )
    logger.info("=" * 80)
    logger.info("Starting data preparation pipeline")
    logger.info("=" * 80)

    loader = SQLDatasetLoader(
        dataset_name=cfg.dataset.name,
        cache_dir=cfg.dataset.cache_dir,
        seed=cfg.project.seed,
    )

    logger.info(f"Loading dataset: {cfg.dataset.name}")
    dataset = _load_dataset_with_split(cfg, loader, logger)
    dataset = _ensure_dataset_splits(dataset, cfg, loader, logger)
    _log_dataset_statistics(dataset, loader, logger)

    preprocessor = SQLDataPreprocessor(
        max_question_length=cfg.dataset.preprocessing.max_question_length,
        max_schema_length=cfg.dataset.preprocessing.max_schema_length,
        max_sql_length=cfg.dataset.preprocessing.max_sql_length,
        normalize_sql=cfg.dataset.preprocessing.normalize_sql,
        filter_invalid=cfg.dataset.preprocessing.filter_invalid,
    )

    _preprocess_splits(dataset, preprocessor, cfg, logger)
    _compute_final_statistics(dataset, logger)
    _save_processed_dataset(dataset, cfg, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Data preparation complete!")
    logger.info("=" * 80)

    return dataset


if __name__ == "__main__":
    prepare_data()
