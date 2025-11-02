"""Test the text-to-SQL environment with sample data.

This script loads the environment and tests it with real dataset samples,
demonstrating prompt formatting, response parsing, and reward computation.

Usage:
    python scripts/test_environment.py
    python scripts/test_environment.py training.environment.prompt_template=chat
"""

import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra  # noqa: E402
from datasets import load_dataset  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from environments.sql_env import TextToSQLEnvironment  # noqa: E402
from rubrics.sql_rubric import SQLValidationRubric  # noqa: E402
from utils.sql_parser import SQLParser  # noqa: E402

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def test_environment(cfg: DictConfig):
    """Test environment with real dataset samples.

    Args:
        cfg: Hydra configuration
    """
    logger.info("=" * 80)
    logger.info("Text-to-SQL Environment Test")
    logger.info("=" * 80)

    # Load sample data first (required for environment initialization)
    logger.info("\n[1/4] Loading dataset samples...")
    try:
        dataset = load_dataset(cfg.dataset.name, split="train[:10]", trust_remote_code=True)
        logger.info(f"  ✓ Loaded {len(dataset)} samples from {cfg.dataset.name}")
    except Exception as e:
        logger.error(f"  ✗ Failed to load dataset: {e}")
        logger.info("  Using mock data instead...")

        # Create mock dataset
        from datasets import Dataset as HFDataset

        mock_data = {
            "question": [
                "How many users are in the database?",
                "List all product names",
                "Get users who joined in 2024",
            ],
            "context": [
                "CREATE TABLE users (id INT, name VARCHAR(100), email VARCHAR(100))",
                ("CREATE TABLE products (id INT, name VARCHAR(200), " "price DECIMAL(10,2))"),
                "CREATE TABLE users (id INT, name VARCHAR(100), joined_date DATE)",
            ],
            "answer": [
                "SELECT COUNT(*) FROM users",
                "SELECT name FROM products",
                "SELECT * FROM users WHERE YEAR(joined_date) = 2024",
            ],
        }
        dataset = HFDataset.from_dict(mock_data)
        logger.info(f"  ✓ Created {len(dataset)} mock samples")

    # Initialize components
    logger.info("\n[2/4] Initializing environment...")
    parser = SQLParser()
    rubric = SQLValidationRubric()

    # Get environment config
    env_cfg = cfg.training.get("environment", {})
    prompt_template = env_cfg.get("prompt_template", "instructional")
    include_schema = env_cfg.get("include_schema", True)
    max_schema_length = env_cfg.get("max_schema_length", 1024)

    logger.info(f"  - Prompt template: {prompt_template}")
    logger.info(f"  - Include schema: {include_schema}")
    logger.info(f"  - Max schema length: {max_schema_length}")

    env = TextToSQLEnvironment(
        rubric=rubric,
        parser=parser,
        prompt_template=prompt_template,
        include_schema=include_schema,
        max_schema_length=max_schema_length,
        dataset=dataset,  # Now passing the dataset
    )
    logger.info("  ✓ Environment initialized")

    # Test environment with samples
    logger.info("\n[3/4] Testing environment...")
    logger.info("-" * 80)

    all_responses = []
    all_rewards = []

    for i, sample in enumerate(dataset):
        logger.info(f"\nSample {i+1}/{len(dataset)}")
        logger.info("-" * 80)

        # Format prompt
        prompt = env.format_prompt(
            question=sample["question"], context={"schema": sample["context"]}
        )

        logger.info(f"\n📝 Question: {sample['question']}")
        schema_preview = (
            f"{sample['context'][:200]}" f"{'...' if len(sample['context']) > 200 else ''}"
        )
        logger.info(f"\n🗃️  Schema:\n{schema_preview}")
        logger.info(f"\n💬 Prompt:\n{prompt[:300]}{'...' if len(prompt) > 300 else ''}")

        # Test with reference answer
        reference_sql = sample.get("answer", "")
        if reference_sql:
            logger.info(f"\n✅ Reference SQL:\n{reference_sql}")

            # Parse response
            parsed = env.parse_response(reference_sql)
            logger.info("\n🔍 Parsed Result:")
            logger.info(f"  - Valid: {parsed['valid']}")
            logger.info(f"  - Extracted SQL: {parsed['sql']}")

            # Compute reward
            reward = env.compute_reward(reference_sql, context={"schema": sample["context"]})
            logger.info(f"\n⭐ Reward Score: {reward:.3f}")

            all_responses.append(reference_sql)
            all_rewards.append(reward)

    # Test batch computation
    logger.info("\n" + "=" * 80)
    logger.info("[4/4] Testing batch reward computation...")
    logger.info("-" * 80)

    if all_responses:
        # Measure batch processing speed
        start_time = time.time()
        batch_rewards = env.batch_compute_rewards(all_responses)
        elapsed = time.time() - start_time

        samples_per_sec = len(all_responses) / elapsed if elapsed > 0 else 0

        logger.info(f"\n  ✓ Processed {len(all_responses)} responses in {elapsed:.3f}s")
        logger.info(f"  ✓ Throughput: {samples_per_sec:.1f} samples/sec")

        # Compare with individual rewards
        logger.info("\n  Reward comparison:")
        for i, (ind_reward, batch_reward) in enumerate(
            zip(all_rewards, batch_rewards, strict=True)
        ):
            match = "✓" if abs(ind_reward - batch_reward) < 0.001 else "✗"
            logger.info(f"    Sample {i+1}: {ind_reward:.3f} vs {batch_reward:.3f} {match}")

    # Compute aggregate metrics
    logger.info("\n" + "=" * 80)
    logger.info("Aggregate Metrics")
    logger.info("-" * 80)

    if all_responses:
        metrics = env.get_metrics(all_responses)
        logger.info(f"\n  Valid SQL: {metrics['valid_sql_pct']:.1f}%")
        logger.info(f"  Syntax Correct: {metrics['syntax_correct_pct']:.1f}%")
        logger.info(f"  Average Reward: {metrics['avg_reward']:.3f}")
        logger.info(f"  Min Reward: {metrics['min_reward']:.3f}")
        logger.info(f"  Max Reward: {metrics['max_reward']:.3f}")
        logger.info(f"  Num Samples: {metrics['num_samples']}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Environment test complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_environment()
