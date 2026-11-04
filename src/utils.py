"""Shared helpers: configuration loading, logging setup and writing outputs to disk."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

REQUIRED_CONFIG_SECTIONS = ("paths", "logging", "data", "split", "model", "evaluation", "inference")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Read the YAML configuration file and check that every required section is present.

    Args:
        config_path: Path to the YAML file.

    Returns:
        The configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a mapping or a required section is missing.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")
    missing = [section for section in REQUIRED_CONFIG_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Config file {path} is missing sections: {missing}")
    logger.info("Loaded config from %s", path)
    return config


def setup_logging(log_file: str | Path, level: str) -> None:
    """Send log records from every module to the console and to a log file.

    Args:
        log_file: File that receives the audit trail of each run.
        level: Logging level name, for example "INFO".
    """
    ensure_parent_dir(log_file)
    logging.basicConfig(
        level=level.upper(),
        format=LOG_FORMAT,
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def ensure_parent_dir(file_path: str | Path) -> Path:
    """Create the parent folder of a file if it does not exist yet.

    Args:
        file_path: File whose folder must exist.

    Returns:
        The file path as a Path object.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, file_path: str | Path, index: bool = False) -> Path:
    """Write a DataFrame to CSV, creating the folder first.

    Args:
        df: Data to write.
        file_path: Destination CSV file.
        index: Whether to write the row index as a column.

    Returns:
        The path that was written.
    """
    path = ensure_parent_dir(file_path)
    df.to_csv(path, index=index)
    logger.info("Wrote %d rows to %s", len(df), path)
    return path


def save_json(data: dict[str, Any], file_path: str | Path) -> Path:
    """Write a dictionary to a JSON file, creating the folder first.

    Args:
        data: JSON-serialisable dictionary.
        file_path: Destination JSON file.

    Returns:
        The path that was written.
    """
    path = ensure_parent_dir(file_path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    logger.info("Wrote %s", path)
    return path
