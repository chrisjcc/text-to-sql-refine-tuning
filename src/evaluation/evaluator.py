"""Comprehensive SQL evaluation engine.

This module provides a comprehensive evaluation framework for SQL generation
models, including metrics computation, stratified analysis, and report
generation.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.evaluation.metrics import ExecutionMetrics, SQLMetrics
from src.inference.inference_engine import SQLInferenceEngine


class SQLEvaluator:
    """Comprehensive SQL evaluation engine.

    Provides end-to-end evaluation capabilities including prediction
    generation, metrics computation, and detailed report generation with
    complexity-stratified analysis.

    Attributes:
        engine: Inference engine for generating SQL predictions.
        metrics: SQL metrics calculator for evaluation.
        execution_metrics: Optional execution-based metrics calculator.
        logger: Logger instance for this class.
    """

    def __init__(
        self,
        inference_engine: SQLInferenceEngine,
        metrics: SQLMetrics | None = None,
        execution_metrics: ExecutionMetrics | None = None,
    ) -> None:
        """Initialize evaluator.

        Args:
            inference_engine: Engine for generating SQL predictions from
                natural language questions.
            metrics: SQL metrics calculator. If None, creates default
                SQLMetrics instance.
            execution_metrics: Optional execution-based metrics calculator
                for database execution validation. Defaults to None.
        """
        self.engine = inference_engine
        self.metrics = metrics or SQLMetrics()
        self.execution_metrics = execution_metrics
        self.logger = logging.getLogger(__name__)

    def evaluate_dataset(
        self,
        dataset: list[dict[str, Any]],
        batch_size: int = 8,
        compute_execution: bool = False,
        **generation_kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate model on dataset with comprehensive metrics.

        Generates predictions for all samples, computes metrics, and
        produces aggregate and per-sample results.

        Args:
            dataset: List of dictionaries with 'question', 'schema', and
                'sql' keys.
            batch_size: Batch size for generation. Defaults to 8.
            compute_execution: Whether to compute execution accuracy by
                running queries against database. Defaults to False.
            **generation_kwargs: Additional generation parameters passed to
                inference engine (e.g., temperature, top_p).

        Returns:
            Dictionary containing:
            - aggregate: Overall metrics across all samples
            - by_complexity: Metrics stratified by SQL complexity
            - per_sample: Individual results for each sample
            - total_samples: Total number of evaluated samples
        """
        self.logger.info(f"Evaluating on {len(dataset)} samples")

        # Generate predictions
        questions = [item["question"] for item in dataset]
        schemas = [item["schema"] for item in dataset if item.get("schema") is not None]
        references = [item["sql"] for item in dataset]

        self.logger.info("Generating predictions...")

        predictions = self.engine.batch_generate_sql(
            questions=questions,
            schemas=schemas,
            batch_size=batch_size,
            **generation_kwargs,  # type: ignore[arg-type]
        )

        # Compute metrics for each sample
        self.logger.info("Computing metrics...")
        per_sample_results = []

        for _i, (pred, ref, item) in enumerate(
            tqdm(
                zip(predictions, references, dataset, strict=True),
                total=len(dataset),
                desc="Computing metrics",
            )
        ):
            sample_metrics = self._compute_sample_metrics(
                predicted=pred["sql"],
                reference=ref,
                compute_execution=compute_execution,
            )

            sample_metrics.update(
                {
                    "question": item["question"],
                    "predicted_sql": pred["sql"],
                    "reference_sql": ref,
                    "valid": pred["valid"],
                    "complexity": item.get("complexity", "unknown"),
                }
            )

            per_sample_results.append(sample_metrics)

        # Compute aggregate metrics
        self.logger.info("Computing aggregate metrics...")
        aggregate_metrics = self._compute_aggregate_metrics(per_sample_results)

        # Compute complexity-stratified metrics
        complexity_metrics = self._compute_complexity_metrics(per_sample_results)

        return {
            "aggregate": aggregate_metrics,
            "by_complexity": complexity_metrics,
            "per_sample": per_sample_results,
            "total_samples": len(dataset),
        }

    def _compute_sample_metrics(
        self,
        predicted: str,
        reference: str,
        compute_execution: bool = False,
    ) -> dict[str, Any]:
        """Compute all metrics for a single sample.

        Args:
            predicted: Model-generated SQL query.
            reference: Ground truth SQL query.
            compute_execution: Whether to compute execution-based metrics.

        Returns:
            Dictionary containing all computed metrics for this sample.
        """
        metrics_dict: dict[str, Any] = {}

        # Basic metrics
        exact_match = self.metrics.exact_match(predicted, reference)
        metrics_dict["exact_match"] = float(exact_match)  # type: ignore[assignment]

        token_acc = self.metrics.token_level_accuracy(predicted, reference)
        metrics_dict["token_accuracy"] = float(token_acc)  # type: ignore[assignment]

        struct_sim = self.metrics.structural_similarity(predicted, reference)
        metrics_dict["structural_similarity"] = float(struct_sim)  # type: ignore[assignment]

        # Keyword F1
        keyword_scores = self.metrics.keyword_f1(predicted, reference)
        metrics_dict.update(
            {
                "keyword_precision": float(keyword_scores["precision"]),
                "keyword_recall": float(keyword_scores["recall"]),
                "keyword_f1": float(keyword_scores["f1"]),
            }
        )

        # Complexity
        pred_complexity = self.metrics.complexity_score(predicted)
        ref_complexity = self.metrics.complexity_score(reference)

        metrics_dict["predicted_complexity"] = pred_complexity["complexity_level"]
        metrics_dict["reference_complexity"] = ref_complexity["complexity_level"]
        pred_level = pred_complexity["complexity_level"]
        ref_level = ref_complexity["complexity_level"]
        metrics_dict["complexity_match"] = int(pred_level == ref_level)  # type: ignore[assignment]

        # Edit distance
        edit_dist = self.metrics.edit_distance(predicted, reference)
        metrics_dict["edit_distance"] = int(edit_dist)

        # Execution metrics (if enabled)
        if compute_execution and self.execution_metrics:
            exec_metrics = self.execution_metrics.execution_accuracy(predicted, reference)
            metrics_dict["execution_match"] = exec_metrics["execution_match"]
            metrics_dict["predicted_executable"] = exec_metrics["predicted_executable"]

        return metrics_dict

    def _compute_aggregate_metrics(self, results: list[dict[str, Any]]) -> dict[str, float]:
        """Compute aggregate metrics across all samples.

        Args:
            results: List of per-sample metric dictionaries.

        Returns:
            Dictionary of aggregated metrics including means and rates.
        """
        results_df = pd.DataFrame(results)

        aggregate = {
            "exact_match_rate": results_df["exact_match"].mean() * 100,
            "avg_token_accuracy": results_df["token_accuracy"].mean() * 100,
            "avg_structural_similarity": (results_df["structural_similarity"].mean() * 100),
            "avg_keyword_f1": results_df["keyword_f1"].mean() * 100,
            "avg_keyword_precision": results_df["keyword_precision"].mean() * 100,
            "avg_keyword_recall": results_df["keyword_recall"].mean() * 100,
            "valid_sql_rate": results_df["valid"].mean() * 100,
            "complexity_match_rate": results_df["complexity_match"].mean() * 100,
            "avg_edit_distance": results_df["edit_distance"].mean(),
        }

        if "execution_match" in results_df.columns:
            aggregate["execution_accuracy"] = results_df["execution_match"].mean() * 100
            executable_rate = results_df["predicted_executable"].mean() * 100
            aggregate["predicted_executable_rate"] = executable_rate

        return aggregate

    def _compute_complexity_metrics(
        self, results: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """Compute metrics stratified by SQL complexity.

        Args:
            results: List of per-sample metric dictionaries.

        Returns:
            Dictionary mapping complexity levels to their metrics.
        """
        results_df = pd.DataFrame(results)

        complexity_metrics: dict[str, dict[str, float]] = {}

        for complexity_level in ["simple", "medium", "complex"]:
            subset = results_df[results_df["reference_complexity"] == complexity_level]

            if len(subset) == 0:
                continue

            struct_sim = subset["structural_similarity"].mean() * 100
            complexity_metrics[complexity_level] = {
                "count": len(subset),
                "exact_match_rate": subset["exact_match"].mean() * 100,
                "avg_token_accuracy": subset["token_accuracy"].mean() * 100,
                "avg_structural_similarity": struct_sim,
                "valid_sql_rate": subset["valid"].mean() * 100,
            }

        return complexity_metrics

    def generate_report(self, evaluation_results: dict[str, Any], output_path: str) -> None:
        """Generate detailed evaluation report.

        Creates multiple output files including JSON summary, CSV with
        per-sample results, and markdown report.

        Args:
            evaluation_results: Results dictionary from evaluate_dataset
                method.
            output_path: Directory path where report files will be saved.

        Returns:
            None. Saves report files to disk.
        """
        output_path_obj = Path(output_path)
        output_path_obj.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        json_path = output_path_obj / "evaluation_results.json"

        with json_path.open("w") as f:
            # Remove per-sample results for cleaner summary
            summary = {
                "aggregate": evaluation_results["aggregate"],
                "by_complexity": evaluation_results["by_complexity"],
                "total_samples": evaluation_results["total_samples"],
            }
            json.dump(summary, f, indent=2)

        # Save per-sample results as CSV
        results_df = pd.DataFrame(evaluation_results["per_sample"])
        csv_path = output_path_obj / "per_sample_results.csv"
        results_df.to_csv(csv_path, index=False)

        # Generate markdown report
        self._generate_markdown_report(evaluation_results, output_path_obj)

        self.logger.info(f"Report saved to {output_path_obj}")

    def _generate_markdown_report(self, results: dict[str, Any], output_path: Path) -> None:
        """Generate markdown evaluation report.

        Args:
            results: Evaluation results dictionary.
            output_path: Path object for output directory.

        Returns:
            None. Writes markdown report to disk.
        """
        report_path = output_path / "evaluation_report.md"

        with report_path.open("w") as f:
            f.write("# SQL Evaluation Report\n\n")

            # Aggregate metrics
            f.write("## Aggregate Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for key, value in results["aggregate"].items():
                f.write(f"| {key.replace('_', ' ').title()} | {value:.2f} |\n")

            # Complexity metrics
            f.write("\n## Metrics by Complexity\n\n")
            for complexity, metrics in results["by_complexity"].items():
                f.write(f"### {complexity.title()}\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for key, value in metrics.items():
                    if key != "count":
                        metric_name = key.replace("_", " ").title()
                        f.write(f"| {metric_name} | {value:.2f} |\n")
                f.write(f"\n**Sample Count:** {metrics['count']}\n\n")

            # Sample count
            f.write(f"\n## Total Samples: {results['total_samples']}\n")
