"""Test the text-to-SQL environment with sample data.

This script loads the environment and tests it with real dataset samples,
demonstrating prompt formatting, response parsing, and reward computation.

Usage:
    python scripts/test_environment.py
    python scripts/test_environment.py training.environment.prompt_template=chat
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
from omegaconf import DictConfig
from datasets import load_dataset

from src.environments.sql_env import TextToSQLEnvironment
from src.rubrics.sql_rubric import SQLValidationRubric
from src.utils.sql_parser import SQLParser


@hydra.main(version_base=None, config_path="../config", config_name="config")
def test_environment(cfg: DictConfig):
    """Test environment with real dataset samples.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("Text-to-SQL Environment Test")
    print("=" * 80)

    # Load components
    print("\n[1/4] Initializing components...")
    parser = SQLParser()
    rubric = SQLValidationRubric()

    # Get environment config
    env_cfg = cfg.training.get("environment", {})
    prompt_template = env_cfg.get("prompt_template", "instructional")
    include_schema = env_cfg.get("include_schema", True)
    max_schema_length = env_cfg.get("max_schema_length", 1024)

    print(f"  - Prompt template: {prompt_template}")
    print(f"  - Include schema: {include_schema}")
    print(f"  - Max schema length: {max_schema_length}")

    env = TextToSQLEnvironment(
        rubric=rubric,
        parser=parser,
        prompt_template=prompt_template,
        include_schema=include_schema,
        max_schema_length=max_schema_length,
    )
    print("  ✓ Environment initialized")

    # Load sample data
    print("\n[2/4] Loading dataset samples...")
    try:
        dataset = load_dataset(
            cfg.dataset.name,
            split="train[:10]",
            trust_remote_code=True
        )
        print(f"  ✓ Loaded {len(dataset)} samples from {cfg.dataset.name}")
    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        print("  Using mock data instead...")

        # Create mock dataset
        from datasets import Dataset as HFDataset
        mock_data = {
            "question": [
                "How many users are in the database?",
                "List all product names",
                "Get users who joined in 2024"
            ],
            "context": [
                "CREATE TABLE users (id INT, name VARCHAR(100), email VARCHAR(100))",
                "CREATE TABLE products (id INT, name VARCHAR(200), price DECIMAL(10,2))",
                "CREATE TABLE users (id INT, name VARCHAR(100), joined_date DATE)"
            ],
            "answer": [
                "SELECT COUNT(*) FROM users",
                "SELECT name FROM products",
                "SELECT * FROM users WHERE YEAR(joined_date) = 2024"
            ]
        }
        dataset = HFDataset.from_dict(mock_data)
        print(f"  ✓ Created {len(dataset)} mock samples")

    # Test environment with samples
    print("\n[3/4] Testing environment...")
    print("-" * 80)

    all_responses = []
    all_rewards = []

    for i, sample in enumerate(dataset):
        print(f"\nSample {i+1}/{len(dataset)}")
        print("-" * 80)

        # Format prompt
        prompt = env.format_prompt(
            question=sample["question"],
            context={"schema": sample["context"]}
        )

        print(f"\n📝 Question: {sample['question']}")
        print(f"\n🗃️  Schema:\n{sample['context'][:200]}{'...' if len(sample['context']) > 200 else ''}")
        print(f"\n💬 Prompt:\n{prompt[:300]}{'...' if len(prompt) > 300 else ''}")

        # Test with reference answer
        reference_sql = sample.get("answer", "")
        if reference_sql:
            print(f"\n✅ Reference SQL:\n{reference_sql}")

            # Parse response
            parsed = env.parse_response(reference_sql)
            print(f"\n🔍 Parsed Result:")
            print(f"  - Valid: {parsed['valid']}")
            print(f"  - Extracted SQL: {parsed['sql']}")

            # Compute reward
            reward = env.compute_reward(
                reference_sql,
                context={"schema": sample["context"]}
            )
            print(f"\n⭐ Reward Score: {reward:.3f}")

            all_responses.append(reference_sql)
            all_rewards.append(reward)

    # Test batch computation
    print("\n" + "=" * 80)
    print("[4/4] Testing batch reward computation...")
    print("-" * 80)

    if all_responses:
        import time

        # Measure batch processing speed
        start_time = time.time()
        batch_rewards = env.batch_compute_rewards(all_responses)
        elapsed = time.time() - start_time

        samples_per_sec = len(all_responses) / elapsed if elapsed > 0 else 0

        print(f"\n  ✓ Processed {len(all_responses)} responses in {elapsed:.3f}s")
        print(f"  ✓ Throughput: {samples_per_sec:.1f} samples/sec")

        # Compare with individual rewards
        print("\n  Reward comparison:")
        for i, (ind_reward, batch_reward) in enumerate(zip(all_rewards, batch_rewards)):
            match = "✓" if abs(ind_reward - batch_reward) < 0.001 else "✗"
            print(f"    Sample {i+1}: {ind_reward:.3f} vs {batch_reward:.3f} {match}")

    # Compute aggregate metrics
    print("\n" + "=" * 80)
    print("Aggregate Metrics")
    print("-" * 80)

    if all_responses:
        metrics = env.get_metrics(all_responses)
        print(f"\n  Valid SQL: {metrics['valid_sql_pct']:.1f}%")
        print(f"  Syntax Correct: {metrics['syntax_correct_pct']:.1f}%")
        print(f"  Average Reward: {metrics['avg_reward']:.3f}")
        print(f"  Min Reward: {metrics['min_reward']:.3f}")
        print(f"  Max Reward: {metrics['max_reward']:.3f}")
        print(f"  Num Samples: {metrics['num_samples']}")

    print("\n" + "=" * 80)
    print("✓ Environment test complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_environment()
