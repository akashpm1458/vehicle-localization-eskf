"""Error dynamics and continuous-to-discrete covariance models.

Continuous error-state model (right-local attitude error), frozen over a substep h:

    d(dx)/dt = A dx + L n,      E[n n^T] = Qc delta(t - t'),     W = L Qc L^T

Non-zero blocks of A (15-state; the 9-state model uses the top-left 9x9 block):

    A_pv = I,  A_v,theta = -R [f]x,  A_v,ba = -R,  A_theta,theta = -[w]x,  A_theta,bg = -I

Noise order: accelerometer white, gyro white, accel-bias drive, gyro-bias drive.

Two discretizations are provided:

* ``fast``: Phi = I + A h + A^2 h^2 / 2 and a Simpson-rule approximation of
  Qd = int_0^h Phi(s) W Phi(s)^T ds. An approximation, not an exact discretization.
* ``van_loan``: exact for frozen A, W via one matrix exponential (reference only).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .rotations import skew
from .state import ATT, BA, BG, POS, VEL

_I3 = np.eye(3)


def error_dynamics(R: np.ndarray, f_hat: np.ndarray, w_hat: np.ndarray, n: int) -> np.ndarray:
    """Continuous error Jacobian A (n x n, n = 9 or 15)."""
    A = np.zeros((n, n))
    A[POS, VEL] = _I3
    A[VEL, ATT] = -R @ skew(f_hat)
    A[ATT, ATT] = -skew(w_hat)
    if n == 15:
        A[VEL, BA] = -R
        A[ATT, BG] = -_I3
    return A


def noise_mapping(R: np.ndarray, n: int) -> np.ndarray:
    """Noise input matrix L: n x 12 (15-state) or n x 6 (9-state)."""
    m = 12 if n == 15 else 6
    L = np.zeros((n, m))
    L[VEL, 0:3] = -R
    L[ATT, 3:6] = -_I3
    if n == 15:
        L[BA, 6:9] = _I3
        L[BG, 9:12] = _I3
    return L


def continuous_noise(accel: float, gyro: float, accel_bias_rw: float, gyro_bias_rw: float, n: int) -> np.ndarray:
    """Qc = diag(sa^2 I, sg^2 I, sba^2 I, sbg^2 I) (bias terms only for n = 15)."""
    d = [accel**2] * 3 + [gyro**2] * 3
    if n == 15:
        d += [accel_bias_rw**2] * 3 + [gyro_bias_rw**2] * 3
    return np.diag(d)


def fast_discretize(A: np.ndarray, W: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Second-order Phi and Simpson-rule Qd over a substep h (h <= 0.01 s)."""
    n = A.shape[0]
    I = np.eye(n)
    Ah = A * h
    A2h2 = Ah @ Ah
    phi = I + Ah + 0.5 * A2h2
    phi_half = I + 0.5 * Ah + 0.125 * A2h2
    Qd = (h / 6.0) * (W + 4.0 * phi_half @ W @ phi_half.T + phi @ W @ phi.T)
    return phi, Qd


def van_loan_discretize(A: np.ndarray, W: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Reference: E = expm([[A, W], [0, -A^T]] h); Phi = E11; Qd = E12 Phi^T."""
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = A
    M[:n, n:] = W
    M[n:, n:] = -A.T
    E = expm(M * h)
    phi = E[:n, :n]
    Qd = E[:n, n:] @ phi.T
    return phi, 0.5 * (Qd + Qd.T)


DISCRETIZERS = {"fast": fast_discretize, "van_loan": van_loan_discretize}
