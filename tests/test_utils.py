"""Tests for src/utils.py."""

import json
import logging

import pandas as pd
import pytest
import yaml

from src.utils import ensure_parent_dir, load_config, save_dataframe, save_json, setup_logging


def test_unit_load_config_returns_all_sections(tmp_config):
    config = load_config(tmp_config)
    assert config["evaluation"]["metric"] == "pr_auc"


def test_unit_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(tmp_path / "nope.yaml")


def test_unit_load_config_not_a_mapping_raises(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_config(path)


def test_unit_load_config_missing_section_raises(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"paths": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing sections"):
        load_config(path)


def test_unit_ensure_parent_dir_creates_folder(tmp_path):
    target = tmp_path / "a" / "b" / "file.csv"
    assert ensure_parent_dir(target) == target
    assert target.parent.is_dir()


def test_unit_save_dataframe_and_json_round_trip(tmp_path):
    df = pd.DataFrame({"x": [1, 2]})
    csv_path = save_dataframe(df, tmp_path / "out" / "x.csv")
    json_path = save_json({"score": 0.7}, tmp_path / "out" / "m.json")
    pd.testing.assert_frame_equal(pd.read_csv(csv_path), df)
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"score": 0.7}


def test_integration_setup_logging_writes_to_file(tmp_path):
    log_file = tmp_path / "logs" / "run.log"
    setup_logging(log_file, "info")
    logging.getLogger("test").info("hello %s", "world")
    logging.shutdown()
    assert "hello world" in log_file.read_text(encoding="utf-8")
