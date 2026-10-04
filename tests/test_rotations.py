"""Quaternion / frame conventions and the SO(3) right Jacobian."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vehicle_localization.rotations import (
    attitude_error_batch,
    quat_conjugate,
    quat_from_rotvec,
    quat_multiply,
    quat_normalize,
    quat_to_matrix,
    quat_to_rotvec,
    right_jacobian,
    skew,
    so3_exp,
    so3_log,
)

RNG = np.random.default_rng(0)


def test_skew_matches_cross_product():
    for _ in range(10):
        u, v = RNG.normal(size=3), RNG.normal(size=3)
        np.testing.assert_allclose(skew(u) @ v, np.cross(u, v), atol=1e-14)


def test_positive_90_deg_about_z_maps_x_to_y():
    q = quat_from_rotvec([0, 0, np.pi / 2])
    np.testing.assert_allclose(quat_to_matrix(q) @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    # also about x: y -> z, and about y: z -> x (right-handed)
    np.testing.assert_allclose(quat_to_matrix(quat_from_rotvec([np.pi / 2, 0, 0])) @ [0, 1, 0], [0, 0, 1], atol=1e-15)
    np.testing.assert_allclose(quat_to_matrix(quat_from_rotvec([0, np.pi / 2, 0])) @ [0, 0, 1], [1, 0, 0], atol=1e-15)


def test_storage_order_is_scipy_xyzw():
    rv = RNG.normal(size=3)
    np.testing.assert_allclose(quat_from_rotvec(rv), Rotation.from_rotvec(rv).as_quat(), atol=1e-14)
    q = quat_normalize(RNG.normal(size=4))
    np.testing.assert_allclose(quat_to_matrix(q), Rotation.from_quat(q).as_matrix(), atol=1e-14)


def test_hamilton_composition_matches_matrix_product():
    for _ in range(10):
        p, q = quat_normalize(RNG.normal(size=4)), quat_normalize(RNG.normal(size=4))
        np.testing.assert_allclose(quat_to_matrix(quat_multiply(p, q)), quat_to_matrix(p) @ quat_to_matrix(q),
                                   atol=1e-14)
        expected = (Rotation.from_quat(p) * Rotation.from_quat(q)).as_quat()
        got = quat_multiply(p, q)
        assert np.allclose(got, expected, atol=1e-14) or np.allclose(got, -expected, atol=1e-14)


def test_inverse_and_normalization():
    q = quat_normalize(RNG.normal(size=4))
    np.testing.assert_allclose(quat_multiply(q, quat_conjugate(q)), [0, 0, 0, 1], atol=1e-15)
    np.testing.assert_allclose(quat_to_matrix(quat_conjugate(q)), quat_to_matrix(q).T, atol=1e-14)
    assert abs(np.linalg.norm(quat_normalize(3.7 * q)) - 1) < 1e-15
    with pytest.raises(ValueError):
        quat_normalize(np.zeros(4))


def test_q_and_minus_q_are_the_same_rotation():
    q = quat_normalize(RNG.normal(size=4))
    np.testing.assert_allclose(quat_to_matrix(q), quat_to_matrix(-q), atol=1e-15)
    np.testing.assert_allclose(quat_to_rotvec(q), quat_to_rotvec(-q), atol=1e-14)


@pytest.mark.parametrize("angle", [0.0, 1e-9, 1e-5, 1e-3, 0.5, 2.0, 3.0])
def test_rotvec_round_trip_and_exp(angle):
    axis = quat_normalize(np.append(RNG.normal(size=3), 0))[:3]
    axis /= np.linalg.norm(axis)
    phi = angle * axis
    np.testing.assert_allclose(quat_to_rotvec(quat_from_rotvec(phi)), phi, atol=1e-13)
    np.testing.assert_allclose(so3_exp(phi), Rotation.from_rotvec(phi).as_matrix(), atol=1e-14)
    np.testing.assert_allclose(quat_to_matrix(quat_from_rotvec(phi)), so3_exp(phi), atol=1e-14)


@pytest.mark.parametrize("phi", [
    np.array([0.0, 0.0, 0.0]),
    np.array([1e-7, -2e-7, 3e-7]),
    np.array([2e-4, 1e-4, -3e-4]),
    np.array([0.01, -0.02, 0.015]),
    np.array([0.3, -0.2, 0.5]),
    np.array([1.0, 0.7, -1.2]),
])
def test_right_jacobian_matches_finite_difference(phi):
    """J_r(phi) = d/d eps Log(Exp(-phi) Exp(phi + eps)) at eps = 0 (central differences)."""
    h = 1e-6
    R_inv = so3_exp(-phi)
    J = np.zeros((3, 3))
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        J[:, i] = (so3_log(R_inv @ so3_exp(phi + e)) - so3_log(R_inv @ so3_exp(phi - e))) / (2 * h)
    np.testing.assert_allclose(right_jacobian(phi), J, atol=1e-6)


def test_right_jacobian_series_is_continuous_across_threshold():
    axis = np.array([0.6, -0.48, 0.64])
    below = right_jacobian(axis * (1e-3 - 1e-12))
    above = right_jacobian(axis * (1e-3 + 1e-12))
    np.testing.assert_allclose(below, above, atol=1e-14)


def test_attitude_error_definition_is_right_local():
    q_est = quat_normalize(RNG.normal(size=4))
    dtheta = np.array([0.01, -0.02, 0.03])
    q_true = quat_multiply(q_est, quat_from_rotvec(dtheta))  # R_true = R_est Exp(dtheta)
    np.testing.assert_allclose(attitude_error_batch(q_true[None], q_est[None])[0], dtheta, atol=1e-14)
