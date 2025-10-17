"""Logging utilities for text-to-sql fine-tuning.

This module provides utilities for setting up structured logging
that integrates with Hydra's logging system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = "logs",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up structured logging with file and console handlers.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (created if doesn't exist)
        log_file: Name of log file (if None, uses 'training.log')
        format_string: Custom format string for log messages

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> from omegaconf import DictConfig
        >>> logger = setup_logging(
        ...     log_level="INFO",
        ...     log_dir="logs",
        ...     format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ... )
        >>> logger.info("Starting training...")
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Default format if not provided
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_dir is provided)
    if log_dir is not None:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            log_file = "training.log"

        log_file_path = log_dir_path / log_file

        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file_path}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Name for the logger (typically __name__)

    Returns:
        logging.Logger: Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing batch...")
    """
    return logging.getLogger(name)


def setup_logging_from_config(cfg) -> logging.Logger:
    """Set up logging from Hydra config.

    Args:
        cfg: Hydra DictConfig object with logging configuration

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> from config.config import load_config
        >>> cfg = load_config()
        >>> logger = setup_logging_from_config(cfg)
    """
    return setup_logging(
        log_level=cfg.logging.level,
        log_dir=cfg.logging.log_dir,
        format_string=cfg.logging.format,
    )


# Example usage
if __name__ == "__main__":
    # Test basic logging setup
    logger = setup_logging(
        log_level="INFO",
        log_dir="logs",
        format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test module-specific logger
    module_logger = get_logger(__name__)
    module_logger.info("Module-specific log message")

    print("\nLogging test completed. Check the logs/ directory for output.")
