"""Analyze evaluation errors and generate insights.

This script analyzes model evaluation results to identify error patterns,
calculate error rates by complexity level, and generate detailed error reports.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def analyze_errors(
    results_path: str, output_path: str, n_samples: int = 20
) -> None:
    """Analyze evaluation errors and generate detailed error reports.

    Loads evaluation results, identifies errors, analyzes error patterns
    by complexity level, and saves detailed analysis reports including
    error samples.

    Args:
        results_path: Path to per_sample_results.csv file containing
            evaluation results with columns for exact_match, valid,
            reference_complexity, etc.
        output_path: Directory path where analysis results will be saved.
            Creates error_analysis.json and error_samples.csv.
        n_samples: Number of error samples to include in the detailed
            error report. Defaults to 20.

    Returns:
        None. Saves analysis results to files and logs statistics.
    """
    # Load results
    df = pd.read_csv(results_path)

    # Identify errors
    errors = df[df["exact_match"] is False]

    logger.info(f"Total samples: {len(df)}")
    logger.info(
        f"Errors: {len(errors)} ({len(errors)/len(df)*100:.1f}%)"
    )

    # Analyze error patterns
    analysis = {
        "total_errors": int(len(errors)),
        "error_rate": float(len(errors) / len(df) * 100),
    }

    # By complexity
    logger.info("\nErrors by complexity:")
    complexity_errors = errors.groupby("reference_complexity").size()
    for complexity, count in complexity_errors.items():
        total = len(df[df["reference_complexity"] == complexity])
        rate = count / total * 100 if total > 0 else 0
        logger.info(f"  {complexity}: {count}/{total} ({rate:.1f}%)")
        analysis[f"{complexity}_error_rate"] = float(rate)

    # Common error patterns
    logger.info("\nCommon issues:")
    invalid_sql = errors[errors["valid"] is False]
    invalid_pct = len(invalid_sql) / len(errors) * 100
    logger.info(
        f"  Invalid SQL: {len(invalid_sql)} "
        f"({invalid_pct:.1f}% of errors)"
    )

    # Save analysis
    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    analysis_file = output_path_obj / "error_analysis.json"
    with analysis_file.open("w") as f:
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
    samples_file = output_path_obj / "error_samples.csv"
    error_samples.to_csv(samples_file, index=False)

    logger.info(f"\nAnalysis saved to {output_path}")


def main() -> None:
    """Parse command-line arguments and run error analysis.

    Returns:
        None. Executes the error analysis based on CLI arguments.
    """
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Analyze evaluation errors and generate insights"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to per_sample_results.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./error_analysis",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of error samples to include in report",
    )

    args = parser.parse_args()
    analyze_errors(args.results, args.output, args.n_samples)


if __name__ == "__main__":
    main()
