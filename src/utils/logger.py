"""
Logging configuration for Code-Morph

Provides structured logging with rich formatting for CLI
and file logging for persistence.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from src.utils.config import settings


def setup_logger(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    enable_console: bool = True,
) -> None:
    """
    Setup application logger

    Args:
        log_file: Path to log file (None for default)
        log_level: Logging level
        enable_console: Whether to log to console
    """
    # Remove default logger
    logger.remove()

    # Console logging with rich formatting
    if enable_console:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level=log_level,
            colorize=True,
        )

    # File logging with detailed information
    if log_file is None:
        log_file = settings.log_file

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",  # File gets more detail
        rotation="10 MB",
        retention="1 week",
        compression="zip",
    )

    logger.info(f"Logger initialized - Level: {log_level}, File: {log_file}")


def get_logger(name: str):
    """
    Get a logger instance

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Initialize logger on import
setup_logger(log_level=settings.log_level)
