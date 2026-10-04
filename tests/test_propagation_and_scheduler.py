"""Inertial propagation sanity checks and the timestamp scheduler."""

import copy

import numpy as np
import pytest

from vehicle_localization.dataset import PositionStream
from vehicle_localization.runner import Dropout, RunError, run_estimator

from .helpers import SpyEKF, est_config, fixture_dataset

# ----------------------------------------------------------------------------- zero-noise


@pytest.mark.parametrize("trajectory", ["stationary", "stationary_tilted", "constant_velocity"])
def test_ten_second_fixtures_do_not_drift(tmp_path, trajectory):
    ds, gt, cfg = fixture_dataset(tmp_path, trajectory)
    res = run_estimator(ds, est_config(cfg, "imu_only"))
    drift = np.linalg.norm(res.p - gt.p, axis=1).max()
    assert drift < 1e-6, f"{trajectory}: position drift {drift:.3e} m"
    assert np.abs(res.v - gt.v).max() < 1e-7
    assert res.info["max_logged_quaternion_norm_error"] < 1e-10


@pytest.mark.parametrize("trajectory", ["constant_yaw_rate", "figure_eight", "smooth_rpy"])
def test_curved_motion_error_shrinks_when_step_is_halved(tmp_path, trajectory):
    errs = []
    for rate in (100, 200, 400):  # integration substep = 1/rate (<= 0.01 s)
        ds, gt, cfg = fixture_dataset(tmp_path, trajectory, imu_rate=rate)
        res = run_estimator(ds, est_config(cfg, "imu_only"))
        errs.append(np.linalg.norm(res.p[-1] - gt.p[-1]))
        assert res.info["max_logged_quaternion_norm_error"] < 1e-10
    assert errs[1] < errs[0] / 1.8 and errs[2] < errs[1] / 1.8, errs


# ----------------------------------------------------------------------------- scheduler


def _with_streams(ds, **streams):
    ds = copy.copy(ds)
    ds.streams = {}
    for name, times in streams.items():
        t = np.asarray(times, dtype=np.int64)
        ds.streams[name] = PositionStream(name, t, np.zeros((t.size, 3)), np.tile(np.eye(3), (t.size, 1, 1)))
    return ds


@pytest.fixture
def short_ds(tmp_path):
    return fixture_dataset(tmp_path, "figure_eight", duration=0.1)


def test_on_grid_and_same_time_ordering(short_ds):
    ds, _, cfg = short_ds
    ds = _with_streams(ds, gnss=[30_000_000], lidar=[30_000_000, 70_000_000])
    run_estimator(ds, est_config(cfg, "eskf15_all"), filter_cls=SpyEKF)
    calls = SpyEKF.last.calls
    preds = [c for c in calls if c[0] == "predict"]
    assert all(c[1] == 0.01 for c in preds) and len(preds) == 10
    ups = [(c[1], c[2]) for c in calls if c[0] == "update"]
    assert ups == [("gnss", 30_000_000), ("lidar", 30_000_000), ("lidar", 70_000_000)]
    # updates at a boundary happen after the previous interval is complete
    i = calls.index(("update", "gnss", 30_000_000))
    assert calls[i - 1][0] == "predict" and calls[i - 1][2] == 20_000_000


def test_off_grid_measurement_splits_the_interval(short_ds):
    ds, _, cfg = short_ds
    ds = _with_streams(ds, gnss=[35_000_000])
    run_estimator(ds, est_config(cfg, "eskf15_gnss"), filter_cls=SpyEKF)
    calls = SpyEKF.last.calls
    i = calls.index(("update", "gnss", 35_000_000))
    before, after = calls[i - 1], calls[i + 1]
    assert before[:3] == ("predict", 0.005, 30_000_000)
    assert after[:3] == ("predict", 0.005, 35_000_000)
    assert before[3] == after[3]  # the same held raw reading of interval [0.03, 0.04)


def test_multiple_measurements_inside_one_interval(short_ds):
    ds, _, cfg = short_ds
    ds = _with_streams(ds, gnss=[32_000_000], lidar=[36_000_000, 38_500_000])
    run_estimator(ds, est_config(cfg, "eskf15_all"), filter_cls=SpyEKF)
    calls = SpyEKF.last.calls
    seq = [c[:3] if c[0] == "predict" else c for c in calls if (c[2] >= 30_000_000 and c[2] < 40_000_000)]
    assert seq == [
        ("predict", 0.002, 30_000_000), ("update", "gnss", 32_000_000),
        ("predict", 0.004, 32_000_000), ("update", "lidar", 36_000_000),
        ("predict", 0.0025, 36_000_000), ("update", "lidar", 38_500_000),
        ("predict", 0.0015, 38_500_000),
    ]


def test_dropout_removes_only_that_stream_and_propagation_continues(short_ds):
    ds, _, cfg = short_ds
    ds = _with_streams(ds, gnss=[20_000_000, 40_000_000, 60_000_000, 80_000_000], lidar=[50_000_000])
    res = run_estimator(ds, est_config(cfg, "eskf15_all"), dropouts=[Dropout("gnss", 40_000_000, 60_000_000)],
                        filter_cls=SpyEKF)
    ups = [(c[1], c[2]) for c in SpyEKF.last.calls if c[0] == "update"]
    assert ups == [("gnss", 20_000_000), ("lidar", 50_000_000), ("gnss", 60_000_000), ("gnss", 80_000_000)]
    assert res.counts["gnss"]["removed_by_dropout"] == 1  # half-open: 60 ms is kept
    total = sum(c[1] for c in SpyEKF.last.calls if c[0] == "predict")
    assert total == pytest.approx(0.1, abs=1e-12)


def test_measurements_outside_imu_coverage_are_skipped(short_ds):
    ds, _, cfg = short_ds
    ds = _with_streams(ds, gnss=[50_000_000, 100_000_000, 150_000_000])
    res = run_estimator(ds, est_config(cfg, "eskf15_gnss"), filter_cls=SpyEKF)
    ups = [c[2] for c in SpyEKF.last.calls if c[0] == "update"]
    assert ups == [50_000_000, 100_000_000]  # the end boundary is inclusive, beyond it is skipped
    assert res.counts["gnss"]["out_of_range"] == 1


def test_measurement_at_start_time_is_applied_before_propagation(short_ds):
    ds, _, cfg = short_ds
    ds = _with_streams(ds, gnss=[0])
    run_estimator(ds, est_config(cfg, "eskf15_gnss"), filter_cls=SpyEKF)
    assert SpyEKF.last.calls[0] == ("update", "gnss", 0)


def test_large_imu_gap_fails_clearly(short_ds):
    ds, _, cfg = short_ds
    ds = copy.copy(ds)
    keep = np.r_[0:4, 9:ds.imu_t_ns.size]  # 60 ms gap > max_imu_gap_s (50 ms)
    ds.imu_t_ns, ds.imu_f, ds.imu_w = ds.imu_t_ns[keep], ds.imu_f[keep], ds.imu_w[keep]
    with pytest.raises(RunError, match="IMU gap"):
        run_estimator(ds, est_config(cfg, "imu_only"))


def test_final_interval_processed_exactly_once_and_last_row_never_used(short_ds):
    ds, _, cfg = short_ds
    run_estimator(ds, est_config(cfg, "imu_only"), filter_cls=SpyEKF)
    preds = [c for c in SpyEKF.last.calls if c[0] == "predict"]
    assert len(preds) == ds.imu_t_ns.size - 1
    assert preds[-1][2] == ds.imu_t_ns[-2]
    assert sum(c[1] for c in preds) == pytest.approx(0.1, abs=1e-12)
    a = run_estimator(ds, est_config(cfg, "imu_only"))
    ds2 = copy.copy(ds)
    ds2.imu_f, ds2.imu_w = ds.imu_f.copy(), ds.imu_w.copy()
    ds2.imu_f[-1] = [1e3, -1e3, 5e2]  # garbage placeholder in the end-of-coverage row
    ds2.imu_w[-1] = [9.0, 9.0, 9.0]
    b = run_estimator(ds2, est_config(cfg, "imu_only"))
    np.testing.assert_array_equal(a.p, b.p)
    np.testing.assert_array_equal(a.q, b.q)
