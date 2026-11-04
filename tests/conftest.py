"""Shared fixtures: a small synthetic dataset with the real schema and a matching config.

Tests never read the real CSV, so they run anywhere, including CI, without data in Git.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REAL_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
N_ROWS = 300


def make_raw_df(n_rows: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """Build raw sessions with the same columns and value types as the real export."""
    rng = np.random.default_rng(seed)
    page_values = rng.exponential(5, n_rows) * rng.integers(0, 2, n_rows)
    df = pd.DataFrame(
        {
            "Administrative": rng.integers(0, 10, n_rows),
            "Administrative_Duration": rng.uniform(0, 300, n_rows),
            "Informational": rng.integers(0, 5, n_rows),
            "Informational_Duration": rng.uniform(0, 100, n_rows),
            "ProductRelated": rng.integers(1, 60, n_rows),
            "ProductRelated_Duration": rng.uniform(0, 2000, n_rows),
            "BounceRates": rng.uniform(0, 0.2, n_rows),
            "ExitRates": rng.uniform(0, 0.2, n_rows),
            "PageValues": page_values,
            "SpecialDay": rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], n_rows),
            "Month": rng.choice(["Feb", "Mar", "May", "June", "Jul", "Nov", "Dec"], n_rows),
            "OperatingSystems": rng.integers(1, 5, n_rows),
            "Browser": rng.integers(1, 6, n_rows),
            "Region": rng.integers(1, 9, n_rows),
            "TrafficType": rng.integers(1, 10, n_rows),
            "VisitorType": rng.choice(["Returning_Visitor", "New_Visitor", "Other"], n_rows),
            "Weekend": rng.choice([True, False], n_rows),
        }
    )
    df["Revenue"] = (page_values + rng.normal(0, 1, n_rows)) > 4
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Synthetic raw data."""
    return make_raw_df()


@pytest.fixture
def config() -> dict:
    """The project's real config.yaml, so tests use the same columns and settings as production."""
    with REAL_CONFIG.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture
def clean_df(raw_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Synthetic data after cleaning."""
    from src.clean_data import clean_dataframe

    data = config["data"]
    return clean_dataframe(
        raw_df, data["target_column"], data["categorical_columns"], data["value_aliases"]
    )


@pytest.fixture
def tmp_config(tmp_path: Path, raw_df: pd.DataFrame, config: dict) -> Path:
    """A config.yaml whose paths all point into a temporary folder holding the synthetic CSV."""
    raw_path = tmp_path / "data" / "raw" / "sessions.csv"
    raw_path.parent.mkdir(parents=True)
    raw_df.to_csv(raw_path, index=False)
    config["paths"] = {
        "raw_data": str(raw_path),
        "clean_data": str(tmp_path / "data" / "processed" / "clean.csv"),
        "model": str(tmp_path / "models" / "model.joblib"),
        "predictions": str(tmp_path / "reports" / "predictions.csv"),
        "metrics": str(tmp_path / "reports" / "metrics.json"),
        "log_file": str(tmp_path / "logs" / "pipeline.log"),
    }
    config["model"]["n_estimators"] = 20
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path
