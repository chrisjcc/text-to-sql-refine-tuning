"""Batch SQL Scorer for efficient scoring during GRPO training.

This module provides utilities for efficiently scoring batches of SQL outputs,
with support for parallel processing and caching.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .sql_rubric import SQLValidationRubric

logger = logging.getLogger(__name__)


class BatchSQLScorer:
    """Efficiently scores batches of SQL outputs for GRPO.

    Supports parallel processing for faster scoring of large batches
    during training. Provides both simple scores and detailed metadata.

    Examples:
        >>> rubric = SQLValidationRubric()
        >>> scorer = BatchSQLScorer(rubric)
        >>> outputs = ["SELECT * FROM users", "SELECT name FROM products"]
        >>> scores = scorer.score_batch(outputs)
        >>> print(scores)
        [1.0, 1.0]

        >>> metadata = scorer.score_with_metadata(outputs)
        >>> print(metadata[0]['total'])
        1.0
    """

    def __init__(
        self,
        rubric: SQLValidationRubric,
        max_workers: Optional[int] = None,
        use_parallel: bool = True,
    ):
        """Initialize the batch scorer.

        Args:
            rubric: SQLValidationRubric instance to use for scoring
            max_workers: Maximum number of parallel workers (None = auto)
            use_parallel: Whether to use parallel processing for large batches
        """
        self.rubric = rubric
        self.max_workers = max_workers
        self.use_parallel = use_parallel

        # Cache for repeated queries (useful during evaluation)
        self._cache: Dict[str, float] = {}
        self._cache_enabled = False

    def enable_cache(self, enabled: bool = True):
        """Enable or disable result caching.

        Args:
            enabled: Whether to enable caching
        """
        self._cache_enabled = enabled
        if not enabled:
            self._cache.clear()

    def clear_cache(self):
        """Clear the scoring cache."""
        self._cache.clear()

    def score_batch(
        self,
        outputs: List[str],
        references: Optional[List[str]] = None,
        use_cache: bool = False,
    ) -> List[float]:
        """Score multiple outputs efficiently.

        Args:
            outputs: List of generated SQL outputs
            references: Optional list of reference SQLs (not currently used)
            use_cache: Whether to use caching for this batch

        Returns:
            List of scores (one per output)
        """
        if not outputs:
            return []

        # Determine if we should use parallel processing
        # Use parallel for batches larger than 10 items
        should_parallelize = self.use_parallel and len(outputs) > 10

        if should_parallelize:
            return self._score_parallel(outputs, references, use_cache)
        else:
            return self._score_sequential(outputs, references, use_cache)

    def _score_sequential(
        self,
        outputs: List[str],
        references: Optional[List[str]],
        use_cache: bool,
    ) -> List[float]:
        """Score outputs sequentially.

        Args:
            outputs: List of outputs
            references: Optional references
            use_cache: Whether to use cache

        Returns:
            List of scores
        """
        scores = []

        for i, output in enumerate(outputs):
            # Check cache
            if use_cache and self._cache_enabled and output in self._cache:
                scores.append(self._cache[output])
                continue

            # Compute score
            ref = references[i] if references else None
            score = self.rubric.score(output, ref)

            # Cache result
            if use_cache and self._cache_enabled:
                self._cache[output] = score

            scores.append(score)

        return scores

    def _score_parallel(
        self,
        outputs: List[str],
        references: Optional[List[str]],
        use_cache: bool,
    ) -> List[float]:
        """Score outputs in parallel.

        Args:
            outputs: List of outputs
            references: Optional references
            use_cache: Whether to use cache

        Returns:
            List of scores
        """
        scores = [None] * len(outputs)  # Pre-allocate list
        tasks = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            for i, output in enumerate(outputs):
                # Check cache first
                if use_cache and self._cache_enabled and output in self._cache:
                    scores[i] = self._cache[output]  # type: ignore[call-overload]
                    continue

                # Submit scoring task
                ref = references[i] if references else None
                future = executor.submit(self.rubric.score, output, ref)
                tasks.append((i, output, future))

            # Collect results
            for i, output, future in tasks:
                try:
                    score = future.result()
                    scores[i] = score  # type: ignore[call-overload]

                    # Cache result
                    if use_cache and self._cache_enabled:
                        self._cache[output] = score

                except Exception as e:
                    logger.error(f"Error scoring output {i}: {e}")
                    scores[i] = 0.0  # type: ignore[call-overload]

        return scores  # type: ignore[return-value]

    def score_with_metadata(
        self,
        outputs: List[str],
        references: Optional[List[str]] = None,
        include_extracted_sql: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return scores with detailed metadata for logging.

        Args:
            outputs: List of generated SQL outputs
            references: Optional list of reference SQLs
            include_extracted_sql: Whether to include extracted SQL in metadata

        Returns:
            List of dictionaries containing detailed scores and metadata
        """
        if not outputs:
            return []

        results = []

        for i, output in enumerate(outputs):
            try:
                # Get detailed scores
                detailed = self.rubric.get_detailed_scores(output)

                # Build metadata
                metadata = {
                    "index": i,
                    "total": detailed["total"],
                    "syntax": detailed["syntax"],
                    "syntax_valid": detailed.get("syntax_valid", False),
                    "keywords": detailed["keywords"],
                    "format": detailed["format"],
                    "weights": detailed["weights"],
                }

                # Add extracted SQL if requested
                if include_extracted_sql:
                    metadata["extracted_sql"] = detailed.get("extracted_sql")  # type: ignore[dict-item]

                # Add reference if provided
                if references and i < len(references):
                    metadata["reference"] = references[i]  # type: ignore[dict-item]

                results.append(metadata)

            except Exception as e:
                logger.error(f"Error getting metadata for output {i}: {e}")
                results.append(  # type: ignore[dict-item]
                    {
                        "index": i,
                        "total": 0.0,
                        "syntax": 0.0,
                        "syntax_valid": False,
                        "keywords": 0.0,
                        "format": 0.0,
                        "extracted_sql": None,  # type: ignore[dict-item]
                        "weights": {  # type: ignore[dict-item]
                            "syntax": self.rubric.syntax_weight,
                            "keywords": self.rubric.keyword_weight,
                            "format": self.rubric.format_weight,
                        },
                        "error": str(e),  # type: ignore[dict-item]
                    }
                )

        return results

    def compute_batch_statistics(
        self,
        outputs: List[str],
        references: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute aggregate statistics for a batch.

        Useful for logging to WandB or other monitoring systems.

        Args:
            outputs: List of generated SQL outputs
            references: Optional list of reference SQLs

        Returns:
            Dictionary with aggregate statistics
        """
        if not outputs:
            return {
                "count": 0,
                "mean_score": 0.0,
            }

        # Score all outputs with metadata
        metadata_list = self.score_with_metadata(outputs, references)

        # Extract scores and components
        scores = [m["total"] for m in metadata_list]
        syntax_scores = [m["syntax"] for m in metadata_list]
        keyword_scores = [m["keywords"] for m in metadata_list]
        format_scores = [m["format"] for m in metadata_list]

        # Count valid SQL
        valid_count = sum(1 for m in metadata_list if m.get("syntax_valid", False))

        # Compute statistics
        import numpy as np

        return {
            "count": len(outputs),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "median_score": float(np.median(scores)),
            "mean_syntax": float(np.mean(syntax_scores)),
            "mean_keywords": float(np.mean(keyword_scores)),
            "mean_format": float(np.mean(format_scores)),
            "valid_sql_count": valid_count,
            "valid_sql_percentage": valid_count / len(outputs) * 100,
        }

    def score_and_log(
        self,
        outputs: List[str],
        references: Optional[List[str]] = None,
        log_to_wandb: bool = False,
        wandb_prefix: str = "eval",
    ) -> Dict[str, Any]:
        """Score batch and optionally log to WandB.

        Args:
            outputs: List of outputs
            references: Optional references
            log_to_wandb: Whether to log to WandB
            wandb_prefix: Prefix for WandB metrics

        Returns:
            Dictionary with statistics
        """
        stats = self.compute_batch_statistics(outputs, references)

        if log_to_wandb:
            try:
                import wandb

                # Create metrics dict with prefix
                metrics = {
                    f"{wandb_prefix}/{k}": v
                    for k, v in stats.items()
                    if isinstance(v, (int, float))
                }

                wandb.log(metrics)
                logger.info(f"Logged batch statistics to WandB: {wandb_prefix}")

            except ImportError:
                logger.warning("wandb not available, skipping logging")
            except Exception as e:
                logger.error(f"Error logging to WandB: {e}")

        return stats
