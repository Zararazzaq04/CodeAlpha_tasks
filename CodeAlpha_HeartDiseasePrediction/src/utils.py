"""
Utility functions for Heart Disease Prediction System.

This module contains helper functions used across the project.
"""

import os
import logging
from datetime import datetime


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger with console and optional file handler.

    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def ensure_directory_exists(directory: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)


def get_timestamp() -> str:
    """
    Get current timestamp as string.

    Returns:
        Timestamp in YYYYMMDD_HHMMSS format
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def format_metric(value: float, decimal_places: int = 4) -> str:
    """
    Format a metric value with specified decimal places.

    Args:
        value: Metric value to format
        decimal_places: Number of decimal places

    Returns:
        Formatted string
    """
    return f"{value:.{decimal_places}f}"


def print_separator(char: str = '-', length: int = 60) -> None:
    """
    Print a separator line.

    Args:
        char: Character to use for separator
        length: Length of separator
    """
    print(char * length)


def print_header(text: str, char: str = '=', length: int = 60) -> None:
    """
    Print a formatted header.

    Args:
        text: Header text
        char: Character for border
        length: Total length
    """
    print()
    print(char * length)
    print(f" {text}")
    print(char * length)
    print()


def validate_input_data(data: dict, required_keys: list) -> tuple:
    """
    Validate input data contains all required keys.

    Args:
        data: Input data dictionary
        required_keys: List of required key names

    Returns:
        Tuple of (is_valid, missing_keys)
    """
    missing_keys = [key for key in required_keys if key not in data]
    return len(missing_keys) == 0, missing_keys


def calculate_risk_category(probability: float) -> str:
    """
    Convert probability to risk category.

    Args:
        probability: Disease probability (0-1)

    Returns:
        Risk category string
    """
    if probability < 0.20:
        return "Very Low"
    elif probability < 0.40:
        return "Low"
    elif probability < 0.60:
        return "Moderate"
    elif probability < 0.80:
        return "High"
    else:
        return "Very High"