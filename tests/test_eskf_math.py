"""Measurement/dynamics/reset Jacobians, discretization, gating and covariance health."""

import copy

import numpy as np
import pytest

from vehicle_localization.discretization import (
    continuous_noise,
    error_dynamics,
    fast_discretize,
    noise_mapping,
    van_loan_discretize,
)
from vehicle_localization.eskf import ErrorStateEKF
from vehicle_localization.rotations import (
    quat_from_rotvec,
    quat_multiply,
    quat_normalize,
    quat_to_matrix,
    right_jacobian,
    so3_exp,
    so3_log,
)
from vehicle_localization.state import ATT, BA, BG, POS, VEL, InitialPrior

from .conftest import make_cfg
from .helpers import est_config

RNG = np.random.default_rng(42)
G = np.array([0.0, 0.0, -9.80665])


def _prior(P_scale=1.0, q=None) -> InitialPrior:
    q = quat_normalize(RNG.normal(size=4)) if q is None else q
    P = np.diag(np.r_[np.full(3, 1.0), np.full(3, 0.09), np.full(3, 1e-3), np.full(3, 1e-2), np.full(3, 2.5e-5)])
    return InitialPrior(0, RNG.normal(size=3) * 10, RNG.normal(size=3), q, RNG.normal(size=3) * 0.05,
                        RNG.normal(size=3) * 0.002, P * P_scale, {})


def _boxplus(f: ErrorStateEKF, dx: np.ndarray) -> ErrorStateEKF:
    g = copy.deepcopy(f)
    g.p = g.p + dx[POS]
    g.v = g.v + dx[VEL]
    g.q = quat_normalize(quat_multiply(g.q, quat_from_rotvec(dx[ATT])))
    g.ba = g.ba + dx[BA]
    g.bg = g.bg + dx[BG]
    return g


def _boxminus(a: ErrorStateEKF, b: ErrorStateEKF) -> np.ndarray:
    """dx such that a = b boxplus dx (right-local attitude)."""
    return np.r_[a.p - b.p, a.v - b.v, so3_log(quat_to_matrix(b.q).T @ quat_to_matrix(a.q)), a.ba - b.ba, a.bg - b.bg]


@pytest.fixture
def cfg15():
    return est_config(make_cfg(), "eskf15_all")


# ----------------------------------------------------------------------------- measurement Jacobian


@pytest.mark.parametrize("lever", [np.zeros(3), np.array([0.2, 0.0, 1.0]), np.array([0.5, -0.3, 1.2])])
def test_lever_arm_jacobian_matches_central_differences(cfg15, lever):
    f = ErrorStateEKF(_prior(), cfg15)
    H = f.measurement_jacobian(lever)
    h = 1e-6
    H_num = np.zeros((3, 15))
    for i in range(15):
        dx = np.zeros(15)
        dx[i] = h
        H_num[:, i] = (_boxplus(f, dx).predict_measurement(lever) - _boxplus(f, -dx).predict_measurement(lever)) / (2 * h)
    np.testing.assert_allclose(H, H_num, atol=1e-6, rtol=1e-6)
    if not lever.any():
        np.testing.assert_allclose(H[:, ATT], 0)  # body-origin sensor observes position directly


# ----------------------------------------------------------------------------- discretization

DISC_CASES = {
    "tilted_stationary": (so3_exp([0.2, -0.3, 0.7]), so3_exp([0.2, -0.3, 0.7]).T @ (-G), np.zeros(3)),
    "turning": (so3_exp([0.0, 0.0, 1.1]), np.array([0.0, 1.5, 9.80665]), np.array([0.0, 0.0, 0.3])),
    "rotating_3d": (so3_exp([0.3, 0.2, -0.4]), np.array([0.5, -0.3, 9.9]), np.array([0.2, -0.1, 0.4])),
}


@pytest.mark.parametrize("case", DISC_CASES)
@pytest.mark.parametrize("n", [9, 15])
def test_fast_discretization_matches_van_loan_reference(case, n):
    """Tolerances chosen from the measured error at h = 0.01 s (~7e-7 for Phi, ~6e-10 for
    Qd, relative Frobenius) with margin; halving h must reduce the error ~8x (3rd order)."""
    R, f, w = DISC_CASES[case]
    A = error_dynamics(R, f, w, n)
    L = noise_mapping(R, n)
    W = L @ continuous_noise(0.02, 0.001, 0.0002, 0.00002, n) @ L.T
    errs = []
    for h in (0.01, 0.005, 0.0025):
        pf, qf = fast_discretize(A, W, h)
        pr, qr = van_loan_discretize(A, W, h)
        errs.append((np.linalg.norm(pf - pr) / np.linalg.norm(pr), np.linalg.norm(qf - qr) / np.linalg.norm(qr)))
        assert np.linalg.eigvalsh(qf).min() > 0  # Simpson sum of PSD terms stays PSD
    assert errs[0][0] < 2e-6 and errs[0][1] < 2e-9, errs[0]
    for (p1, q1), (p2, q2) in zip(errs, errs[1:]):
        # Errors already at round-off (e.g. A^3 = 0 makes the 2nd-order Phi exact) cannot shrink.
        assert (p1 / p2 > 6 or p1 < 1e-14) and (q1 / q2 > 6 or q2 < 1e-13), errs


def test_van_loan_matches_closed_form_for_double_integrator():
    """Position/velocity driven by white acceleration noise: Qd = q [[h^3/3, h^2/2], [h^2/2, h]]."""
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    W = np.diag([0.0, 4.0])
    h = 0.01
    phi, Qd = van_loan_discretize(A, W, h)
    np.testing.assert_allclose(phi, [[1, h], [0, 1]], atol=1e-15)
    np.testing.assert_allclose(Qd, 4.0 * np.array([[h**3 / 3, h**2 / 2], [h**2 / 2, h]]), rtol=1e-10)


def test_full_run_fast_vs_van_loan(smoke_dataset):
    from vehicle_localization.dataset import load_dataset
    from vehicle_localization.runner import run_estimator

    path, cfg = smoke_dataset
    ds = load_dataset(path)
    a = run_estimator(ds, est_config(cfg, "eskf15_all"))
    b = run_estimator(ds, est_config(cfg, "eskf15_all", discretization="van_loan"))
    assert np.abs(a.p - b.p).max() < 1e-6
    rel = np.abs(a.P_diag - b.P_diag) / b.P_diag
    assert rel.max() < 1e-4, rel.max()


# ----------------------------------------------------------------------------- dynamics Jacobian


def _phi_numeric(f0: ErrorStateEKF, f_m, w_m, T, h_step, eps=1e-6):
    """Central-difference Jacobian of nominal propagation over T, in right-local error coordinates."""
    def prop(f):
        g = copy.deepcopy(f)
        g.cfg = copy.copy(g.cfg)
        g.cfg.max_substep_s = h_step
        g.predict(f_m, w_m, T)
        return g

    base = prop(f0)
    J = np.zeros((15, 15))
    for i in range(15):
        dx = np.zeros(15)
        dx[i] = eps
        J[:, i] = (_boxminus(prop(_boxplus(f0, dx)), base) - _boxminus(prop(_boxplus(f0, -dx)), base)) / (2 * eps)
    return J, base


def test_dynamics_jacobian_converges_to_propagation_derivative(cfg15):
    """Phi (frozen continuous dynamics) vs the finite-difference derivative of the nominal
    integrator. They are different discretizations, so we check convergence as h shrinks."""
    f0 = ErrorStateEKF(_prior(), cfg15)
    f_m, w_m, T = np.array([0.8, -0.4, 9.9]), np.array([0.15, -0.1, 0.35]), 0.2
    ref, _ = _phi_numeric(f0, f_m, w_m, T, h_step=T / 400)
    errs = []
    for h in (0.01, 0.005, 0.0025):
        f = copy.deepcopy(f0)
        f.cfg = copy.copy(f.cfg)
        f.cfg.max_substep_s = h
        f.P = np.zeros((15, 15))
        f.Qc = np.zeros_like(f.Qc)
        # Product of the per-step Phi matrices: propagate P = I-columns through phi.
        phi_prod = np.eye(15)
        nsteps = int(round(T / h))
        for _ in range(nsteps):
            R = quat_to_matrix(f.q)
            fh, wh = f_m - f.ba, w_m - f.bg
            R_mid = R @ so3_exp(wh * h / 2)
            phi, _ = fast_discretize(error_dynamics(R_mid, fh, wh, 15), np.zeros((15, 15)), h)
            phi_prod = phi @ phi_prod
            f._predict_step(f_m, w_m, h)
        errs.append(np.abs(phi_prod - ref).max())
    assert errs[0] < 5e-3, errs
    assert errs[1] < errs[0] / 1.8 and errs[2] < errs[1] / 1.8, errs


# ----------------------------------------------------------------------------- update, reset, gating


def test_update_reduces_uncertainty_and_keeps_covariance_valid(cfg15):
    f = ErrorStateEKF(_prior(), cfg15)
    P0 = f.P.copy()
    lever = np.array([0.5, 0.0, 1.2])
    z = f.predict_measurement(lever) + np.array([0.1, -0.2, 0.05])
    rec = f.update_position(z, np.diag([0.0225, 0.0225, 0.04]), lever, "lidar")
    assert rec["accepted"]
    assert np.trace(f.P[POS, POS]) < np.trace(P0[POS, POS])
    np.testing.assert_array_equal(f.P, f.P.T)
    assert np.linalg.eigvalsh(f.P).min() > 0


def test_reset_jacobian_matches_monte_carlo_coordinate_change(cfg15):
    """After injecting a large attitude correction, sampled errors re-expressed in the new
    tangent coordinates have covariance G P_J G^T (not P_J)."""
    q = quat_normalize(RNG.normal(size=4))
    R_old = quat_to_matrix(q)
    dth_hat = np.array([0.25, -0.15, 0.3])
    Pj = np.diag([4e-4, 1e-4, 9e-4])
    eps = RNG.multivariate_normal(dth_hat, Pj, size=40000)  # true error in OLD coordinates
    R_new = R_old @ so3_exp(dth_hat)
    new = np.array([so3_log(R_new.T @ R_old @ so3_exp(e)) for e in eps])
    cov_new = np.cov(new.T)
    Jr = right_jacobian(dth_hat)
    expected = Jr @ Pj @ Jr.T
    assert np.abs(new.mean(axis=0)).max() < 1e-3  # error mean is reset to ~zero
    np.testing.assert_allclose(cov_new, expected, atol=4e-5)
    assert np.abs(cov_new - Pj).max() > 4 * np.abs(cov_new - expected).max()


def test_gating_rejection_leaves_state_and_covariance_unchanged(cfg15):
    f = ErrorStateEKF(_prior(), cfg15)
    before = (f.p.copy(), f.v.copy(), f.q.copy(), f.ba.copy(), f.bg.copy(), f.P.copy())
    lever = np.array([0.2, 0.0, 1.0])
    z = f.predict_measurement(lever) + np.array([20.0, -15.0, 8.0])
    rec = f.update_position(z, np.diag([1.0, 1.0, 4.0]), lever, "gnss")
    assert not rec["accepted"]
    assert rec["nis"] > rec["threshold"]
    np.testing.assert_allclose(rec["r"], [20.0, -15.0, 8.0], atol=1e-9)
    for a, b in zip(before, (f.p, f.v, f.q, f.ba, f.bg, f.P)):
        np.testing.assert_array_equal(a, b)
    # with gating disabled the same measurement is applied
    cfg_off = copy.copy(cfg15)
    cfg_off.gating_enabled = False
    g = ErrorStateEKF(_prior(), cfg_off)
    assert g.update_position(g.predict_measurement(lever) + [20.0, -15.0, 8.0], np.diag([1.0, 1.0, 4.0]), lever,
                             "gnss")["accepted"]


def test_gate_threshold_is_chi_square_not_nine(cfg15):
    f = ErrorStateEKF(_prior(), cfg15)
    assert f.gate_threshold == pytest.approx(14.156, abs=1e-3)


def test_nine_state_filter_ignores_biases(cfg15):
    cfg9 = est_config(make_cfg(), "eskf9_all")
    prior = _prior()
    f = ErrorStateEKF(prior, cfg9)
    assert f.P.shape == (9, 9)
    np.testing.assert_array_equal(f.ba, 0)
    np.testing.assert_array_equal(f.P, prior.P[:9, :9])
    lever = np.array([0.5, 0.0, 1.2])
    f.update_position(f.predict_measurement(lever) + 0.3, np.eye(3) * 0.02, lever, "lidar")
    np.testing.assert_array_equal(f.ba, 0)
    np.testing.assert_array_equal(f.bg, 0)
