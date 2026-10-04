"""Rotation utilities.

Conventions (used everywhere in this project):

* Quaternions are Hamilton quaternions stored as ``[qx, qy, qz, qw]`` (scalar last),
  the same order SciPy uses.
* A quaternion ``q`` represents ``R_WB``: it maps body vectors into the world frame.
* Orientation errors are right-multiplicative and body-local:
  ``R_true = R_est @ Exp(dtheta)``.
* ``Exp(phi)`` is the SO(3) exponential of a rotation vector ``phi`` (radians) and
  ``quat(phi)`` is the matching unit quaternion.

The single-sample functions are written with plain NumPy because the filter calls them
thousands of times per run; SciPy's ``Rotation`` is used for vectorised work and as an
independent reference in the tests.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# Below this angle (rad) the closed-form expressions lose precision, so we switch to
# Taylor series. The truncation error of the series used is O(angle^6).
_SERIES_ANGLE = 1e-3


def skew(u: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix with ``skew(u) @ v == np.cross(u, v)``."""
    x, y, z = u
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return ``q`` scaled to unit norm."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n == 0.0:
        raise ValueError(f"cannot normalize quaternion {q}")
    return q / n


def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product ``p ⊗ q`` for ``xyzw`` quaternions."""
    px, py, pz, pw = p
    qx, qy, qz, qw = q
    return np.array(
        [
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
            pw * qw - px * qx - py * qy - pz * qz,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (the inverse, for a unit quaternion)."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_from_rotvec(phi: np.ndarray) -> np.ndarray:
    """Unit quaternion ``quat(phi)`` for a rotation vector ``phi`` in radians."""
    phi = np.asarray(phi, dtype=float)
    angle = np.linalg.norm(phi)
    if angle < _SERIES_ANGLE:
        # sin(a/2)/a = 1/2 - a^2/48 + a^4/3840 - ...
        k = 0.5 - angle**2 / 48.0 + angle**4 / 3840.0
    else:
        k = np.sin(0.5 * angle) / angle
    return np.array([k * phi[0], k * phi[1], k * phi[2], np.cos(0.5 * angle)])


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Rotation vector of a unit quaternion (shortest rotation, angle in [0, pi])."""
    q = quat_normalize(q)
    if q[3] < 0.0:  # q and -q are the same rotation; pick the short way round
        q = -q
    v = q[:3]
    s = np.linalg.norm(v)
    w = q[3]
    if s < 1e-12:
        return 2.0 * v / w
    angle = 2.0 * np.arctan2(s, w)
    return (angle / s) * v


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a unit ``xyzw`` quaternion."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """``xyzw`` quaternion of a rotation matrix (via SciPy), with ``qw >= 0``."""
    q = Rotation.from_matrix(R).as_quat()
    return -q if q[3] < 0 else q


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """``Exp(phi)``: rotation matrix of a rotation vector (Rodrigues' formula)."""
    phi = np.asarray(phi, dtype=float)
    angle = np.linalg.norm(phi)
    K = skew(phi)
    if angle < _SERIES_ANGLE:
        a = 1.0 - angle**2 / 6.0 + angle**4 / 120.0
        b = 0.5 - angle**2 / 24.0 + angle**4 / 720.0
    else:
        a = np.sin(angle) / angle
        b = (1.0 - np.cos(angle)) / angle**2
    return np.eye(3) + a * K + b * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """``Log(R)``: rotation vector of a rotation matrix (via SciPy)."""
    return Rotation.from_matrix(R).as_rotvec()


def right_jacobian(phi: np.ndarray) -> np.ndarray:
    """SO(3) right Jacobian ``J_r(phi)``.

    ``Exp(phi + eps) ≈ Exp(phi) @ Exp(J_r(phi) @ eps)`` for small ``eps``, so
    ``J_r(phi)`` is the derivative of ``Log(Exp(-phi) @ Exp(phi + eps))`` at ``eps = 0``.
    It is used for the covariance reset after an attitude correction.
    """
    phi = np.asarray(phi, dtype=float)
    angle = np.linalg.norm(phi)
    K = skew(phi)
    if angle < _SERIES_ANGLE:
        c1 = 0.5 - angle**2 / 24.0 + angle**4 / 720.0
        c2 = 1.0 / 6.0 - angle**2 / 120.0 + angle**4 / 5040.0
    else:
        c1 = (1.0 - np.cos(angle)) / angle**2
        c2 = (angle - np.sin(angle)) / angle**3
    return np.eye(3) - c1 * K + c2 * (K @ K)


def quat_from_yaw(yaw: np.ndarray) -> np.ndarray:
    """Quaternions (N, 4) for pure rotations about world ``z`` by ``yaw`` (rad)."""
    yaw = np.atleast_1d(np.asarray(yaw, dtype=float))
    q = np.zeros((yaw.size, 4))
    q[:, 2] = np.sin(0.5 * yaw)
    q[:, 3] = np.cos(0.5 * yaw)
    return q


def rotate_batch(q: np.ndarray, v: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Rotate vectors ``v`` (N, 3) by quaternions ``q`` (N, 4): ``R @ v`` or ``R.T @ v``."""
    rot = Rotation.from_quat(q)
    return rot.inv().apply(v) if inverse else rot.apply(v)


def attitude_error_batch(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    """Local attitude error ``Log(R_est.T @ R_true)`` for (N, 4) quaternion arrays.

    This matches the filter's error definition ``R_true = R_est @ Exp(dtheta)``.
    """
    return (Rotation.from_quat(q_est).inv() * Rotation.from_quat(q_true)).as_rotvec()
