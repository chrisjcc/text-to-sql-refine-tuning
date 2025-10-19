"""
Analyze evaluation errors and generate insights.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def analyze_errors(results_path: str, output_path: str, n_samples: int = 20):
    """
    Analyze evaluation errors.

    Args:
        results_path: Path to per_sample_results.csv
        output_path: Path to save analysis
        n_samples: Number of error samples to include
    """
    # Load results
    df = pd.read_csv(results_path)

    # Identify errors
    errors = df[df["exact_match"] is False]

    print(f"Total samples: {len(df)}")
    print(f"Errors: {len(errors)} ({len(errors)/len(df)*100:.1f}%)")

    # Analyze error patterns
    analysis = {
        "total_errors": len(errors),
        "error_rate": len(errors) / len(df) * 100,
    }

    # By complexity
    print("\nErrors by complexity:")
    complexity_errors = errors.groupby("reference_complexity").size()
    for complexity, count in complexity_errors.items():
        total = len(df[df["reference_complexity"] == complexity])
        rate = count / total * 100 if total > 0 else 0
        print(f"  {complexity}: {count}/{total} ({rate:.1f}%)")
        analysis[f"{complexity}_error_rate"] = rate

    # Common error patterns
    print("\nCommon issues:")
    invalid_sql = errors[errors["valid"] is False]
    print(f"  Invalid SQL: {len(invalid_sql)} ({len(invalid_sql)/len(errors)*100:.1f}% of errors)")

    # Save analysis
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "error_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Save error samples
    error_samples = errors.head(n_samples)[
        [
            "question",
            "reference_sql",
            "predicted_sql",
            "valid",
            "token_accuracy",
            "structural_similarity",
            "reference_complexity",
        ]
    ]
    error_samples.to_csv(output_path / "error_samples.csv", index=False)

    print(f"\nAnalysis saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, default="./error_analysis")
    parser.add_argument("--n-samples", type=int, default=20)

    args = parser.parse_args()
    analyze_errors(args.results, args.output, args.n_samples)
