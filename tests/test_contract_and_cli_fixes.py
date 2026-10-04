"""Data-contract enforcement, empty streams, truth isolation and CLI input validation."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.stats import chi2

import vehicle_localization.dataset as dataset_mod
from vehicle_localization.dataset import (
    POSITION_COLUMNS,
    load_dataset,
    load_ground_truth,
    validate_dataset,
)
from vehicle_localization.experiments import execute_run, run_monte_carlo
from vehicle_localization.plotting import gate_label
from vehicle_localization.runner import Dropout, run_estimator
from vehicle_localization.sensors import generate_dataset

from .conftest import make_cfg
from .helpers import est_config

ROOT = Path(__file__).resolve().parents[1]
PY = [sys.executable, "-m", "vehicle_localization"]


@pytest.fixture
def ds_copy(smoke_dataset, tmp_path):
    d = tmp_path / "copy"
    shutil.copytree(smoke_dataset[0], d)
    return d


def _edit_json(path: Path, fn) -> None:
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj))


# ----------------------------------------------------------------------------- metadata conventions


@pytest.mark.parametrize("group,key,value", [
    ("schema_version", None, "2.0"),
    ("frames", "world", "ned"),
    ("frames", "body", "x_forward_y_right_z_up"),
    ("frames", "rotation", "R_BW_world_to_body"),
    ("units", "time", "ms"),
    ("units", "length", "ft"),
    ("units", "angle", "deg"),
    ("units", "angular_rate", "deg/s"),
    ("units", "specific_force", "g"),
    ("timing", "imu", "sample_at_interval_end"),
    ("timing", "external", "arrival_time"),
    ("frames", None, "W is right-handed, z up"),  # free text instead of the machine-readable object
])
def test_unsupported_conventions_are_rejected(ds_copy, group, key, value):
    def change(meta):
        if key is None:
            meta[group] = value
        else:
            meta[group][key] = value

    _edit_json(ds_copy / "metadata.json", change)
    res = validate_dataset(ds_copy)
    assert not res.ok
    assert any(group in e for e in res.errors), res.errors
    with pytest.raises(ValueError):
        load_dataset(ds_copy)


def test_generated_metadata_declares_supported_conventions(smoke_dataset):
    meta = json.loads((smoke_dataset[0] / "metadata.json").read_text())
    for group, required in dataset_mod.SUPPORTED_CONVENTIONS.items():
        for k, v in required.items():
            assert meta[group][k] == v


# ----------------------------------------------------------------------------- prior covariance layout


@pytest.mark.parametrize("change", ["order_swapped", "order_missing", "units_deg", "units_old_string"])
def test_prior_covariance_layout_is_checked_before_use(ds_copy, change):
    def edit(prior):
        cov = prior["covariance"]
        if change == "order_swapped":
            o = cov["error_state_order"]
            o[0:3], o[3:6] = o[3:6], o[0:3]
        elif change == "order_missing":
            del cov["error_state_order"]
        elif change == "units_deg":
            cov["units"]["attitude"] = "deg"
        else:
            cov["units"] = "m, m/s, rad (body-local), m/s^2, rad/s"

    _edit_json(ds_copy / "initial_prior.json", edit)
    res = validate_dataset(ds_copy)
    assert not res.ok
    assert any("initial_prior" in e and ("error_state_order" in e or "units" in e) for e in res.errors), res.errors


# ----------------------------------------------------------------------------- empty sensor files


def test_header_only_gnss_is_valid_with_warning_and_runs(ds_copy, smoke_dataset):
    (ds_copy / "gnss.csv").write_text(",".join(POSITION_COLUMNS) + "\n")
    res = validate_dataset(ds_copy)  # used to raise IndexError
    assert res.ok, res.report()
    assert any("gnss.csv" in w and "no measurements" in w for w in res.warnings)
    ds = load_dataset(ds_copy)
    assert ds.streams["gnss"].z.shape == (0, 3) and ds.streams["gnss"].R.shape == (0, 3, 3)
    r = run_estimator(ds, est_config(smoke_dataset[1], "eskf15_all"))
    assert r.counts["gnss"]["processed"] == 0 and r.counts["lidar"]["processed"] > 0


def test_header_only_imu_gives_actionable_error(ds_copy):
    (ds_copy / "imu.csv").write_text("t_ns,fx,fy,fz,wx,wy,wz\n")
    res = validate_dataset(ds_copy)
    assert not res.ok
    assert any("imu.csv needs at least 2 rows" in e for e in res.errors)


def test_generator_stream_starting_after_end_produces_valid_empty_file(tmp_path):
    cfg = make_cfg(dataset={"duration_s": 2.0, "lidar": {"first_time_s": 5.0}})
    generate_dataset(cfg, tmp_path / "d")
    lines = (tmp_path / "d" / "lidar_position.csv").read_text().splitlines()
    assert lines == [",".join(POSITION_COLUMNS)]
    res = validate_dataset(tmp_path / "d")
    assert res.ok and any("lidar_position.csv" in w for w in res.warnings)
    r = run_estimator(load_dataset(tmp_path / "d"), est_config(cfg, "eskf15_all"))
    assert r.counts["lidar"]["available"] == 0


# ----------------------------------------------------------------------------- truth separation


def test_load_dataset_never_reads_ground_truth(ds_copy, monkeypatch):
    opened = []
    real = dataset_mod._read_csv

    def spy(path, columns, errors):
        opened.append(Path(path).name)
        return real(path, columns, errors)

    monkeypatch.setattr(dataset_mod, "_read_csv", spy)
    monkeypatch.setattr(dataset_mod, "_inspect_truth", lambda p: pytest.fail("truth inspected"))
    load_dataset(ds_copy)
    assert "ground_truth.csv" not in opened


def test_malformed_truth_does_not_block_estimation(ds_copy, smoke_dataset, tmp_path):
    lines = (ds_copy / "ground_truth.csv").read_text().splitlines()
    lines[5] = lines[5].rsplit(",", 1)[0] + ",nan"
    (ds_copy / "ground_truth.csv").write_text("\n".join(lines) + "\n")
    res = validate_dataset(ds_copy)
    assert not res.ok and all(e.startswith("[evaluation]") for e in res.errors)
    out = execute_run(ds_copy, smoke_dataset[1], "eskf15_all", tmp_path / "run")
    assert str(out["metrics"]["accuracy"]).startswith("unavailable")
    assert any("ground truth unusable" in w for w in out["metrics"]["warnings"])
    assert (tmp_path / "run" / "estimates.csv").exists()


@pytest.mark.parametrize("kind", ["duplicate", "decreasing"])
def test_truth_timestamps_must_strictly_increase(ds_copy, kind):
    lines = (ds_copy / "ground_truth.csv").read_text().splitlines()
    if kind == "duplicate":
        lines[4] = lines[3]
    else:
        lines[3], lines[4] = lines[4], lines[3]
    (ds_copy / "ground_truth.csv").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=kind):
        load_ground_truth(ds_copy)
    assert any(kind in e for e in validate_dataset(ds_copy).errors)


# ----------------------------------------------------------------------------- dropouts


@pytest.mark.parametrize("spec", ["radar:1:2", "gnss:5:3", "gnss:4:4", "gnss:nan:4", "gnss:1:inf", "gnss:1",
                                  "gnss:a:b", "gnss:1:2:3"])
def test_invalid_dropout_requests_are_rejected(spec):
    with pytest.raises(ValueError):
        Dropout.parse(spec)


def test_dropout_effects_are_recorded_and_no_effect_is_warned(smoke_dataset):
    path, cfg = smoke_dataset
    ds = load_dataset(path)
    drops = [Dropout.parse("gnss:4:6"), Dropout.parse("gnss:500:600"), Dropout.parse("lidar:4:6")]
    r = run_estimator(ds, est_config(cfg, "eskf15_gnss"), drops)
    eff = r.info["dropouts"]
    assert eff[0]["removed"] == 10  # 5 Hz GNSS in [4, 6)
    assert eff[1]["removed"] == 0 and "no measurements" in eff[1]["note"]
    assert eff[2]["removed"] == 0 and "not used by mode" in eff[2]["note"]
    assert len(r.info["warnings"]) == 2


def test_cli_rejects_bad_dropout_before_running(smoke_dataset, tmp_path):
    out = tmp_path / "run"
    r = subprocess.run(PY + ["run", "--data", str(smoke_dataset[0]), "--output", str(out), "--config",
                             str(ROOT / "configs" / "smoke.yaml"), "--dropout", "gnss:6:4"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1 and "start < end" in r.stderr
    assert not out.exists()


# ----------------------------------------------------------------------------- mode resolution


def test_run_uses_config_mode_unless_overridden(smoke_dataset, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({"dataset": {"duration_s": 10.0}, "estimator": {"mode": "eskf9_gnss"},
                                        "report": {"burn_in_s": 2.0}}))
    for args, expected, source in (([], "eskf9_gnss", "config"), (["--mode", "imu_only"], "imu_only", "cli")):
        out = tmp_path / f"run_{expected}"
        r = subprocess.run(PY + ["run", "--data", str(smoke_dataset[0]), "--output", str(out), "--config",
                                 str(cfg_path), *args], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        meta = json.loads((out / "run_metadata.json").read_text())
        assert meta["mode"] == expected and meta["mode_source"] == source
        assert f"{expected}: position RMSE" in r.stdout
        assert yaml.safe_load((out / "config_resolved.yaml").read_text())["estimator"]["mode"] == expected


# ----------------------------------------------------------------------------- monte-carlo, gate label


def test_monte_carlo_supports_imu_only_and_rejects_nonpositive_runs(tmp_path):
    cfg = make_cfg(dataset={"duration_s": 3.0}, report={"burn_in_s": 1.0})
    s = run_monte_carlo(cfg, tmp_path / "mc", runs=1, mode="imu_only")
    assert s["mean_gnss_nis"] is None and s["mean_lidar_nis"] is None
    for bad in (0, -3):
        with pytest.raises(ValueError, match="positive"):
            run_monte_carlo(cfg, tmp_path / "mc2", runs=bad)
    r = subprocess.run(PY + ["monte-carlo", "--runs", "0", "--output", str(tmp_path / "mc3")], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 1 and "positive" in r.stderr


def test_gate_label_follows_configured_probability():
    for p in (0.95, 0.99, 0.9973):
        thr = chi2.ppf(p, df=3)
        assert f"{100 * p:.4g}%" in gate_label(thr, p)
        assert f"{100 * p:.4g}%" in gate_label(thr)  # recovered from the logged threshold
    assert "disabled" in gate_label(chi2.ppf(0.99, 3), 0.99, enabled=False)
