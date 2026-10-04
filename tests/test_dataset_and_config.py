"""Data contract validation, reproducible generation and config handling."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from vehicle_localization.config import DEFAULT_CONFIG, ConfigError, load_config
from vehicle_localization.dataset import load_dataset, load_ground_truth, validate_dataset
from vehicle_localization.sensors import derive_outlier_dataset, generate_dataset

from .conftest import make_cfg

ROOT = Path(__file__).resolve().parents[1]


def _copy(src: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "copy"
    shutil.copytree(src, dst)
    return dst


def _edit_line(path: Path, line_no: int, fn) -> None:
    lines = path.read_text().splitlines()
    lines[line_no] = fn(lines[line_no])
    path.write_text("\n".join(lines) + "\n")


def test_default_yaml_matches_builtin_defaults():
    with (ROOT / "configs" / "default.yaml").open() as fh:
        assert yaml.safe_load(fh) == DEFAULT_CONFIG
    load_config(ROOT / "configs" / "smoke.yaml")


def test_invalid_config_is_rejected():
    with pytest.raises(ConfigError):
        make_cfg(truth={"gnss": {"sigma_m": [1.0, 0.0, 2.0]}})
    with pytest.raises(ConfigError):
        make_cfg(estimator={"max_substep_s": 0.05})
    with pytest.raises(ConfigError):
        make_cfg(dataset={"imu_rate_hz": 3})  # non-integer ns period
    with pytest.raises(ConfigError):
        make_cfg(estimator={"mode": "eskf99"})


def test_generated_dataset_validates(smoke_dataset):
    path, _ = smoke_dataset
    res = validate_dataset(path)
    assert res.ok, res.report()
    ds = load_dataset(path)
    assert ds.imu_t_ns.dtype == np.int64
    assert set(ds.streams) == {"gnss", "lidar"}


def test_generation_is_reproducible(tmp_path):
    cfg = make_cfg(dataset={"duration_s": 3.0})
    generate_dataset(cfg, tmp_path / "a")
    generate_dataset(cfg, tmp_path / "b")
    for f in ("imu.csv", "gnss.csv", "lidar_position.csv", "ground_truth.csv", "initial_prior.json"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes(), f


def test_generate_refuses_to_overwrite(tmp_path):
    cfg = make_cfg(dataset={"duration_s": 1.0})
    generate_dataset(cfg, tmp_path / "d")
    with pytest.raises(FileExistsError):
        generate_dataset(cfg, tmp_path / "d")
    generate_dataset(cfg, tmp_path / "d", overwrite=True)


@pytest.mark.parametrize("case", [
    "nan", "wrong_columns", "bad_cov", "wxyz_truth", "dup_imu", "neg_imu", "missing_meta", "prior_wxyz", "missing_imu",
])
def test_validator_rejects_bad_data(smoke_dataset, tmp_path, case):
    d = _copy(smoke_dataset[0], tmp_path)
    if case == "nan":
        _edit_line(d / "imu.csv", 5, lambda s: s.rsplit(",", 1)[0] + ",nan")
    elif case == "wrong_columns":
        _edit_line(d / "gnss.csv", 3, lambda s: s + ",1.0")
    elif case == "bad_cov":
        _edit_line(d / "gnss.csv", 2, lambda s: ",".join(s.split(",")[:4] + ["1", "0", "0", "-1", "0", "4"]))
    elif case == "wxyz_truth":
        _edit_line(d / "ground_truth.csv", 0, lambda s: s.replace("qx,qy,qz,qw", "qw,qx,qy,qz"))
    elif case == "dup_imu":
        lines = (d / "imu.csv").read_text().splitlines()
        lines.insert(4, lines[3])
        (d / "imu.csv").write_text("\n".join(lines) + "\n")
    elif case == "neg_imu":
        lines = (d / "imu.csv").read_text().splitlines()
        lines[3], lines[4] = lines[4], lines[3]
        (d / "imu.csv").write_text("\n".join(lines) + "\n")
    elif case == "missing_meta":
        meta = json.loads((d / "metadata.json").read_text())
        del meta["frames"]
        (d / "metadata.json").write_text(json.dumps(meta))
    elif case == "prior_wxyz":
        prior = json.loads((d / "initial_prior.json").read_text())
        prior["quaternion"]["order"] = "wxyz"
        (d / "initial_prior.json").write_text(json.dumps(prior))
    elif case == "missing_imu":
        (d / "imu.csv").unlink()
    res = validate_dataset(d)
    assert not res.ok, case
    if case == "wxyz_truth":
        # Truth is evaluation-only: a bad truth file must not block loading estimator inputs.
        assert all(e.startswith("[evaluation]") for e in res.errors)
        load_dataset(d)
        with pytest.raises(ValueError, match="wxyz"):
            load_ground_truth(d)
        return
    with pytest.raises(ValueError):
        load_dataset(d)


def test_shuffled_imu_accepted_only_with_normalization(smoke_dataset, tmp_path):
    d = _copy(smoke_dataset[0], tmp_path)
    lines = (d / "imu.csv").read_text().splitlines()
    lines[3], lines[4] = lines[4], lines[3]
    (d / "imu.csv").write_text("\n".join(lines) + "\n")
    assert not validate_dataset(d).ok
    res = validate_dataset(d, normalize_imu=True)
    assert res.ok and any("sorted" in w for w in res.warnings)


def test_shared_timestamps_allowed_and_out_of_range_reported(smoke_dataset, tmp_path):
    d = _copy(smoke_dataset[0], tmp_path)
    # put a lidar measurement at the same time as a gnss measurement
    g = (d / "gnss.csv").read_text().splitlines()
    t_shared = g[1].split(",")[0]
    _edit_line(d / "lidar_position.csv", 1, lambda s: t_shared + "," + s.split(",", 1)[1])
    # and one gnss measurement after the end of IMU coverage
    _edit_line(d / "gnss.csv", len(g) - 1, lambda s: "99000000000," + s.split(",", 1)[1])
    res = validate_dataset(d)
    assert res.ok, res.report()
    assert any("outside IMU coverage" in w for w in res.warnings)


def test_missing_ground_truth_is_allowed(smoke_dataset, tmp_path):
    d = _copy(smoke_dataset[0], tmp_path)
    (d / "ground_truth.csv").unlink()
    (d / "truth_metadata.json").unlink()
    assert validate_dataset(d).ok


def test_outlier_variant_changes_only_selected_rows(smoke_dataset, tmp_path):
    base, cfg = smoke_dataset
    info = derive_outlier_dataset(base, tmp_path / "out", cfg["experiment"]["outliers"] | {"after_s": 2.0})
    a = np.loadtxt(base / "gnss.csv", delimiter=",", skiprows=1)
    b = np.loadtxt(tmp_path / "out" / "gnss.csv", delimiter=",", skiprows=1)
    changed = np.any(a != b, axis=1)
    np.testing.assert_array_equal(changed, info["labels"])
    np.testing.assert_allclose((b - a)[changed][:, 1:4], np.tile([20.0, -15.0, 8.0], (changed.sum(), 1)))
    assert (base / "imu.csv").read_bytes() == (tmp_path / "out" / "imu.csv").read_bytes()
    tm = json.loads((tmp_path / "out" / "truth_metadata.json").read_text())
    assert len(tm["corruptions"]["outliers"]["indices"]) == info["n_outliers"] > 0
    assert validate_dataset(tmp_path / "out").ok
