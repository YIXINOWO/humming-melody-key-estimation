from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_main_experiment_config_loads() -> None:
    config_path = ROOT / "revision" / "config" / "main_experiment.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["project"]["environment"] == "jasm-revision"
    assert config["evaluation"]["outer_splits"] == 5
    assert len(config["random_forest"]["candidates"]) == 3
    assert config["random_forest"]["common"]["class_weight"] == "balanced_subsample"
