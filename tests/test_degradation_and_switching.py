"""Degraded-GNSS data, the switching fusion policy and the relative position error metric."""

import copy

import numpy as np
import pytest

from vehicle_localization.config import ConfigError
from vehicle_localization.dataset import load_dataset, validate_dataset
from vehicle_localization.eskf import ErrorStateEKF
from vehicle_localization.evaluation import relative_position_error
from vehicle_localization.runner import Dropout, SwitchPolicy, run_estimator
from vehicle_localization.sensors import degradation_bias, derive_degraded_gnss_dataset, generate_dataset

from .conftest import make_cfg
from .helpers import est_config

DEG = {"interval_s": [4.0, 7.0], "noise_scale": 5.0, "bias_m": [4.0, -3.0, 1.5]}


@pytest.fixture(scope="module")
def degraded(tmp_path_factory, smoke_dataset):
    base, cfg = smoke_dataset
    out = tmp_path_factory.mktemp("deg") / "data"
    info = derive_degraded_gnss_dataset(base, out, DEG)
    return base, out, cfg, info


# ----------------------------------------------------------------------------- degraded GNSS data


def test_degradation_changes_only_gnss_inside_the_interval(degraded):
    base, out, _, info = degraded
    a = np.loadtxt(base / "gnss.csv", delimiter=",", skiprows=1)
    b = np.loadtxt(out / "gnss.csv", delimiter=",", skiprows=1)
    t = a[:, 0] * 1e-9
    inside = (t >= 4.0) & (t < 7.0)
    assert info["n_degraded"] == inside.sum() > 0
    assert np.all(np.any(a[inside, 1:4] != b[inside, 1:4], axis=1))
    np.testing.assert_array_equal(a[~inside], b[~inside])
    np.testing.assert_array_equal(a[:, 4:], b[:, 4:])  # reported covariance unchanged (receiver unaware)
    for f in ("imu.csv", "lidar_position.csv", "initial_prior.json", "ground_truth.csv"):
        assert (base / f).read_bytes() == (out / f).read_bytes(), f
    assert validate_dataset(out).ok


def test_degradation_bias_profile():
    t = np.array([3.9, 4.0, 5.5, 6.999, 7.0])
    b = degradation_bias(t, 4.0, 7.0, [4.0, -3.0, 1.5])
    np.testing.assert_allclose(b[[0, 1, 4]], 0.0, atol=1e-12)
    np.testing.assert_allclose(b[2], [4.0, -3.0, 1.5])  # peak mid-interval


def test_degradation_noise_scale(tmp_path):
    cfg = make_cfg(dataset={"duration_s": 200.0, "lidar": {"enabled": False}}, truth={"bias_mode": "zero"})
    generate_dataset(cfg, tmp_path / "base")
    derive_degraded_gnss_dataset(tmp_path / "base", tmp_path / "deg",
                                 {"interval_s": [0.0, 200.0], "noise_scale": 3.0, "bias_m": [0.0, 0.0, 0.0]})
    a = np.loadtxt(tmp_path / "base" / "gnss.csv", delimiter=",", skiprows=1)
    b = np.loadtxt(tmp_path / "deg" / "gnss.csv", delimiter=",", skiprows=1)
    extra = (b - a)[:, 1:4] / np.sqrt(a[:, [4, 7, 9]])
    assert np.std(extra) == pytest.approx(np.sqrt(3.0**2 - 1), rel=0.06)


# ----------------------------------------------------------------------------- switching policy


def _switch_cfg(cfg, **kw):
    c = est_config(cfg, "eskf15_all")
    c.fusion_policy = "switch"
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_switch_on_nominal_data_uses_gnss_and_skips_lidar(smoke_dataset):
    path, cfg = smoke_dataset
    r = run_estimator(load_dataset(path), _switch_cfg(cfg))
    pol = r.info["fusion_policy"]
    assert pol["name"] == "switch"
    assert r.counts["lidar"]["skipped_by_policy"] > 0.8 * r.counts["lidar"]["available"]
    assert r.counts["gnss"]["accepted"] > 0
    assert all(x["sensor"] == "gnss" for x in r.innovations) or pol["switches"]


def test_switch_falls_back_to_lidar_on_gnss_timeout(smoke_dataset):
    path, cfg = smoke_dataset
    r = run_estimator(load_dataset(path), _switch_cfg(cfg), dropouts=[Dropout.parse("gnss:4:6")])
    to_lidar = [s for s in r.info["fusion_policy"]["switches"] if s["to"] == "lidar" and s["t_s"] >= 4.0]
    assert to_lidar and to_lidar[0]["reason"] == "GNSS timeout"
    assert to_lidar[0]["t_s"] <= 3.83 + 0.5 + 0.11  # last GNSS before 4 s is 3.83 s; timeout 0.5 s; LiDAR 10 Hz
    assert r.counts["lidar"]["accepted"] > 0


def test_switch_detects_degraded_gnss_and_recovers(degraded):
    _, out, cfg, _ = degraded
    r = run_estimator(load_dataset(out), _switch_cfg(cfg, switch_recover_after=5))
    sw = r.info["fusion_policy"]["switches"]
    first_bad = [s for s in sw if s["to"] == "lidar" and s["t_s"] >= 4.0][0]
    assert first_bad["t_s"] < 5.0 and "NIS" in first_bad["reason"]
    back = [s for s in sw if s["to"] == "gnss" and s["t_s"] > first_bad["t_s"]]
    assert back and back[0]["t_s"] >= 7.0  # only after the degraded section ends


def test_switch_judging_does_not_change_the_filter_state(smoke_dataset):
    path, cfg = smoke_dataset
    ds = load_dataset(path)
    c = _switch_cfg(cfg)
    f = ErrorStateEKF(ds.prior, c)
    pol = SwitchPolicy(c, 0)
    before = (f.p.copy(), f.P.copy())
    pol.judge_gnss(f, f.predict_measurement(c.lever_arms["gnss"]) + 50.0, np.eye(3), c.lever_arms["gnss"], 1)
    assert not pol.gnss_ok
    np.testing.assert_array_equal(before[0], f.p)
    np.testing.assert_array_equal(before[1], f.P)


def test_switch_with_single_sensor_mode_warns_and_fuses(smoke_dataset):
    path, cfg = smoke_dataset
    ds = load_dataset(path)
    c = copy.copy(est_config(cfg, "eskf15_gnss"))
    c.fusion_policy = "switch"
    a = run_estimator(ds, c)
    b = run_estimator(ds, est_config(cfg, "eskf15_gnss"))
    assert any("switch" in w for w in a.info["warnings"])
    np.testing.assert_array_equal(a.p, b.p)


def test_switching_config_is_validated():
    with pytest.raises(ConfigError):
        make_cfg(estimator={"fusion_policy": "vote"})
    with pytest.raises(ConfigError):
        make_cfg(estimator={"switching": {"recover_after": 0}})
    with pytest.raises(ConfigError):
        make_cfg(experiment={"gnss_degradation": {"noise_scale": 0.5}})


# ----------------------------------------------------------------------------- relative error


def test_relative_position_error_ignores_offset_and_measures_drift():
    t = np.arange(0, 30.01, 0.1)
    const = {"t_s": t, "dp": np.tile([1.0, -2.0, 0.5], (t.size, 1))}
    r = relative_position_error(const, 10.0)
    assert r["windows"] == 3 and r["max_m"] == pytest.approx(0.0, abs=1e-12)
    drift = {"t_s": t, "dp": np.column_stack([0.05 * t, 0 * t, 0 * t])}  # 5 cm/s drift
    r = relative_position_error(drift, 10.0)
    assert r["mean_m"] == pytest.approx(0.5, rel=1e-6)
    short = {"t_s": t[:20], "dp": np.zeros((20, 3))}
    assert relative_position_error(short, 10.0)["windows"] == 0
