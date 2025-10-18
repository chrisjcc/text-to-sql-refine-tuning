"""Inference module for text-to-SQL generation.

This module provides tools for running inference with fine-tuned models,
including batch processing, interactive CLI, and REST API endpoints.
"""

from .inference_engine import SQLInferenceEngine

__all__ = [
    "SQLInferenceEngine",
]
