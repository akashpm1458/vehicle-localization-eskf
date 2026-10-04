import copy

import pytest

from vehicle_localization.config import DEFAULT_CONFIG, deep_merge, validate_config


def make_cfg(**groups) -> dict:
    """Default config with per-group overrides, validated."""
    cfg = deep_merge(copy.deepcopy(DEFAULT_CONFIG), groups)
    validate_config(cfg)
    return cfg


@pytest.fixture
def smoke_cfg():
    return make_cfg(
        dataset={"name": "smoke", "duration_s": 10.0},
        experiment={"dropout_interval_s": [4.0, 6.0], "outliers": {"after_s": 2.0}},
        report={"burn_in_s": 2.0},
    )


@pytest.fixture(scope="session")
def smoke_dataset(tmp_path_factory):
    """A 10 s noisy, biased dataset generated once per test session."""
    from vehicle_localization.sensors import generate_dataset

    cfg = make_cfg(dataset={"name": "smoke", "duration_s": 10.0})
    out = tmp_path_factory.mktemp("smoke") / "data"
    generate_dataset(cfg, out)
    return out, cfg
