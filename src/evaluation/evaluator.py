"""
Comprehensive SQL evaluation engine.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.evaluation.metrics import ExecutionMetrics, SQLMetrics
from src.inference.inference_engine import SQLInferenceEngine


class SQLEvaluator:
    """
    Comprehensive SQL evaluation engine.
    """

    def __init__(
        self,
        inference_engine: SQLInferenceEngine,
        metrics: Optional[SQLMetrics] = None,
        execution_metrics: Optional[ExecutionMetrics] = None,
    ):
        """
        Initialize evaluator.

        Args:
            inference_engine: Engine for generating predictions
            metrics: SQL metrics calculator
            execution_metrics: Execution-based metrics (optional)
        """
        self.engine = inference_engine
        self.metrics = metrics or SQLMetrics()
        self.execution_metrics = execution_metrics
        self.logger = logging.getLogger(__name__)

    def evaluate_dataset(
        self,
        dataset: List[Dict],
        batch_size: int = 8,
        compute_execution: bool = False,
        **generation_kwargs,
    ) -> Dict[str, any]:
        """
        Evaluate model on dataset with comprehensive metrics.

        Args:
            dataset: List of dicts with 'question', 'schema', 'sql'
            batch_size: Batch size for generation
            compute_execution: Whether to compute execution accuracy
            **generation_kwargs: Generation parameters

        Returns:
            Dict with aggregate metrics and per-sample results
        """
        self.logger.info(f"Evaluating on {len(dataset)} samples")

        # Generate predictions
        questions = [item["question"] for item in dataset]
        schemas = [item.get("schema") for item in dataset]
        references = [item["sql"] for item in dataset]

        self.logger.info("Generating predictions...")
        predictions = self.engine.batch_generate_sql(
            questions=questions, schemas=schemas, batch_size=batch_size, **generation_kwargs
        )

        # Compute metrics for each sample
        self.logger.info("Computing metrics...")
        per_sample_results = []

        for i, (pred, ref, item) in enumerate(
            tqdm(
                zip(predictions, references, dataset), total=len(dataset), desc="Computing metrics"
            )
        ):
            sample_metrics = self._compute_sample_metrics(
                predicted=pred["sql"], reference=ref, compute_execution=compute_execution
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
        self, predicted: str, reference: str, compute_execution: bool = False
    ) -> Dict[str, any]:
        """Compute all metrics for a single sample."""
        metrics_dict = {}

        # Basic metrics
        metrics_dict["exact_match"] = self.metrics.exact_match(predicted, reference)
        metrics_dict["token_accuracy"] = self.metrics.token_level_accuracy(predicted, reference)
        metrics_dict["structural_similarity"] = self.metrics.structural_similarity(
            predicted, reference
        )

        # Keyword F1
        keyword_scores = self.metrics.keyword_f1(predicted, reference)
        metrics_dict.update(
            {
                "keyword_precision": keyword_scores["precision"],
                "keyword_recall": keyword_scores["recall"],
                "keyword_f1": keyword_scores["f1"],
            }
        )

        # Complexity
        pred_complexity = self.metrics.complexity_score(predicted)
        ref_complexity = self.metrics.complexity_score(reference)

        metrics_dict["predicted_complexity"] = pred_complexity["complexity_level"]
        metrics_dict["reference_complexity"] = ref_complexity["complexity_level"]
        metrics_dict["complexity_match"] = (
            pred_complexity["complexity_level"] == ref_complexity["complexity_level"]
        )

        # Edit distance
        metrics_dict["edit_distance"] = self.metrics.edit_distance(predicted, reference)

        # Execution metrics (if enabled)
        if compute_execution and self.execution_metrics:
            exec_metrics = self.execution_metrics.execution_accuracy(predicted, reference)
            metrics_dict["execution_match"] = exec_metrics["execution_match"]
            metrics_dict["predicted_executable"] = exec_metrics["predicted_executable"]

        return metrics_dict

    def _compute_aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute aggregate metrics across all samples."""
        df = pd.DataFrame(results)

        aggregate = {
            "exact_match_rate": df["exact_match"].mean() * 100,
            "avg_token_accuracy": df["token_accuracy"].mean() * 100,
            "avg_structural_similarity": df["structural_similarity"].mean() * 100,
            "avg_keyword_f1": df["keyword_f1"].mean() * 100,
            "avg_keyword_precision": df["keyword_precision"].mean() * 100,
            "avg_keyword_recall": df["keyword_recall"].mean() * 100,
            "valid_sql_rate": df["valid"].mean() * 100,
            "complexity_match_rate": df["complexity_match"].mean() * 100,
            "avg_edit_distance": df["edit_distance"].mean(),
        }

        if "execution_match" in df.columns:
            aggregate["execution_accuracy"] = df["execution_match"].mean() * 100
            aggregate["predicted_executable_rate"] = df["predicted_executable"].mean() * 100

        return aggregate

    def _compute_complexity_metrics(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Compute metrics stratified by SQL complexity."""
        df = pd.DataFrame(results)

        complexity_metrics = {}

        for complexity_level in ["simple", "medium", "complex"]:
            subset = df[df["reference_complexity"] == complexity_level]

            if len(subset) == 0:
                continue

            complexity_metrics[complexity_level] = {
                "count": len(subset),
                "exact_match_rate": subset["exact_match"].mean() * 100,
                "avg_token_accuracy": subset["token_accuracy"].mean() * 100,
                "avg_structural_similarity": subset["structural_similarity"].mean() * 100,
                "valid_sql_rate": subset["valid"].mean() * 100,
            }

        return complexity_metrics

    def generate_report(self, evaluation_results: Dict, output_path: str):
        """
        Generate detailed evaluation report.

        Args:
            evaluation_results: Results from evaluate_dataset
            output_path: Path to save report
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        json_path = output_path / "evaluation_results.json"
        with open(json_path, "w") as f:
            # Remove per-sample results for cleaner summary
            summary = {
                "aggregate": evaluation_results["aggregate"],
                "by_complexity": evaluation_results["by_complexity"],
                "total_samples": evaluation_results["total_samples"],
            }
            json.dump(summary, f, indent=2)

        # Save per-sample results as CSV
        df = pd.DataFrame(evaluation_results["per_sample"])
        csv_path = output_path / "per_sample_results.csv"
        df.to_csv(csv_path, index=False)

        # Generate markdown report
        self._generate_markdown_report(evaluation_results, output_path)

        self.logger.info(f"Report saved to {output_path}")

    def _generate_markdown_report(self, results: Dict, output_path: Path):
        """Generate markdown evaluation report."""
        report_path = output_path / "evaluation_report.md"

        with open(report_path, "w") as f:
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
                        f.write(f"| {key.replace('_', ' ').title()} | {value:.2f} |\n")
                f.write(f"\n**Sample Count:** {metrics['count']}\n\n")

            # Sample count
            f.write(f"\n## Total Samples: {results['total_samples']}\n")
