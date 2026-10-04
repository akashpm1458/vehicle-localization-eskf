"""Independent truth, specific force, noise convention and random-stream independence."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vehicle_localization.rotations import rotate_batch
from vehicle_localization.sensors import TRUE_GRAVITY_W, simulate, specific_force
from vehicle_localization.trajectory import (
    ConstantVelocity,
    ConstantYawRate,
    FigureEight,
    SmoothRPY,
    Stationary,
)

from .conftest import make_cfg

TRAJECTORIES = [FigureEight(), Stationary(), Stationary(roll=0.2, pitch=-0.3, yaw=0.7),
                ConstantVelocity(), ConstantYawRate(), SmoothRPY()]


@pytest.mark.parametrize("traj", TRAJECTORIES, ids=lambda t: t.name)
def test_analytical_derivatives_match_numerical_differences(traj):
    t = np.linspace(1.0, 9.0, 41)
    h = 1e-5
    s, sp, sm = traj.sample(t), traj.sample(t + h), traj.sample(t - h)
    np.testing.assert_allclose((sp.p - sm.p) / (2 * h), s.v, atol=1e-6)
    np.testing.assert_allclose((sp.v - sm.v) / (2 * h), s.a, atol=1e-6)
    # Body angular rate: R^T dR/dt = skew(omega_B)
    R = Rotation.from_quat(s.q).as_matrix()
    dR = (Rotation.from_quat(sp.q).as_matrix() - Rotation.from_quat(sm.q).as_matrix()) / (2 * h)
    W = np.einsum("nji,njk->nik", R, dR)
    omega_num = np.column_stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]])
    np.testing.assert_allclose(omega_num, s.omega_b, atol=1e-6)


def test_figure_eight_matches_brief_formulas():
    t = np.array([7.3])
    s = FigureEight().sample(t)
    u, ud = 2 * np.pi * t / 60, 2 * np.pi / 60
    np.testing.assert_allclose(s.p[0], [40 * np.sin(u[0]), 20 * np.sin(2 * u[0]), 0])
    np.testing.assert_allclose(s.v[0], [40 * ud * np.cos(u[0]), 40 * ud * np.cos(2 * u[0]), 0])
    np.testing.assert_allclose(s.a[0], [-40 * ud**2 * np.sin(u[0]), -80 * ud**2 * np.sin(2 * u[0]), 0])
    speed = np.linalg.norm(FigureEight().sample(np.linspace(0, 120, 12001)).v, axis=1)
    assert speed.min() > 1.0


def test_stationary_level_specific_force_is_plus_g():
    s = Stationary().sample(np.array([0.0, 1.0]))
    np.testing.assert_allclose(specific_force(s, TRUE_GRAVITY_W), [[0, 0, 9.80665]] * 2, atol=1e-14)


@pytest.mark.parametrize("traj", TRAJECTORIES, ids=lambda t: t.name)
def test_specific_force_rotated_plus_gravity_gives_true_acceleration(traj):
    s = traj.sample(np.linspace(0, 5, 11))
    f = specific_force(s, TRUE_GRAVITY_W)
    a_rec = rotate_batch(s.q, f) + TRUE_GRAVITY_W
    np.testing.assert_allclose(a_rec, s.a, atol=1e-12)
    if traj.name == "stationary":
        np.testing.assert_allclose(a_rec, 0, atol=1e-12)  # level and tilted: zero world acceleration


def test_noise_convention_interval_average_std():
    """White noise std per 100 Hz interval is density / sqrt(dt); bias steps sigma*sqrt(dt)."""
    cfg = make_cfg(dataset={"trajectory": "stationary", "duration_s": 600.0},
                   truth={"bias_mode": "zero"})
    sim = simulate(cfg)
    f_true = np.array([0, 0, 9.80665])
    n_a = sim["f_m"][:-1] - f_true
    n_g = sim["w_m"][:-1]
    dt = 0.01
    assert np.std(n_a) == pytest.approx(0.02 / np.sqrt(dt), rel=0.02)
    assert np.std(n_g) == pytest.approx(0.001 / np.sqrt(dt), rel=0.02)
    # zero-bias mode: truth biases identically zero
    assert np.all(sim["truth"].ba == 0) and np.all(sim["truth"].bg == 0)

    cfg_b = make_cfg(dataset={"trajectory": "stationary", "duration_s": 600.0})
    tb = simulate(cfg_b)["truth"]
    steps = np.diff(tb.ba, axis=0)
    assert np.std(steps) == pytest.approx(0.0002 * np.sqrt(dt), rel=0.02)
    np.testing.assert_allclose(tb.ba[0], [0.03, -0.02, 0.04])
    np.testing.assert_allclose(tb.bg[0], [0.001, -0.0005, 0.0008])


def test_removing_lidar_does_not_change_other_streams():
    a = simulate(make_cfg(dataset={"duration_s": 5.0}))
    b = simulate(make_cfg(dataset={"duration_s": 5.0, "lidar": {"enabled": False}}))
    assert "lidar" not in b["streams"]
    np.testing.assert_array_equal(a["f_m"], b["f_m"])
    np.testing.assert_array_equal(a["w_m"], b["w_m"])
    np.testing.assert_array_equal(a["streams"]["gnss"]["z"], b["streams"]["gnss"]["z"])
    np.testing.assert_array_equal(a["prior"].p, b["prior"].p)


def test_sensor_positions_include_rotated_lever_arm():
    sim = simulate(make_cfg(dataset={"duration_s": 30.0}, truth={"noise_free": True}))
    traj = FigureEight()
    for name, lever in (("gnss", [0.2, 0.0, 1.0]), ("lidar", [0.5, 0.0, 1.2])):
        s = sim["streams"][name]
        smp = traj.sample(s["t_ns"] * 1e-9)
        expected = smp.p + Rotation.from_quat(smp.q).apply(lever)
        np.testing.assert_allclose(s["z"], expected, atol=1e-12)
        # lever arm direction turns with the vehicle (not a fixed world offset)
        offs = s["z"] - smp.p
        assert np.ptp(offs[:, 0]) > 0.1


def test_sensor_timing_offsets():
    sim = simulate(make_cfg(dataset={"duration_s": 2.0}))
    np.testing.assert_array_equal(sim["streams"]["gnss"]["t_ns"][:3], [30_000_000, 230_000_000, 430_000_000])
    np.testing.assert_array_equal(sim["streams"]["lidar"]["t_ns"][:3], [70_000_000, 170_000_000, 270_000_000])
    assert sim["imu_t"].size == 201  # N + 1 rows for N = 200 intervals


def test_prior_is_truth_minus_sampled_error():
    sim = simulate(make_cfg(dataset={"duration_s": 1.0}))
    gt, prior, err = sim["truth"], sim["prior"], sim["prior_error"]
    np.testing.assert_allclose(gt.p[0] - prior.p, err["position_m"], atol=1e-12)
    np.testing.assert_allclose(gt.v[0] - prior.v, err["velocity_mps"], atol=1e-12)
    from vehicle_localization.rotations import attitude_error_batch

    np.testing.assert_allclose(attitude_error_batch(gt.q[:1], prior.q[None])[0], err["attitude_rad"], atol=1e-12)
    np.testing.assert_array_equal(prior.ba, 0)
    assert prior.P[6, 6] == pytest.approx(np.deg2rad(2.0) ** 2)
