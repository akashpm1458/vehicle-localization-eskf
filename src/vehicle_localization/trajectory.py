"""Independent ground-truth trajectories with analytical derivatives.

Truth is defined in closed form here and never by calling the estimator's integrator,
so a bug in the filter's propagation cannot silently cancel out against the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .rotations import quat_from_yaw


@dataclass
class TrajectorySample:
    """Truth at times ``t`` (seconds). Arrays are (N, 3) except ``q`` (N, 4, xyzw)."""

    t: np.ndarray
    p: np.ndarray  # body-origin position in W, m
    v: np.ndarray  # body-origin velocity in W, m/s
    a: np.ndarray  # body-origin acceleration in W, m/s^2 (gravity NOT included)
    q: np.ndarray  # R_WB as xyzw quaternion
    omega_b: np.ndarray  # body angular rate in B, rad/s


def _yaw_aligned(t: np.ndarray, p: np.ndarray, v: np.ndarray, a: np.ndarray) -> TrajectorySample:
    """Planar vehicle: zero roll/pitch, yaw along the horizontal velocity.

    yaw = atan2(vy, vx) and yaw_rate = (vx*ay - vy*ax) / (vx^2 + vy^2), computed
    directly so we never differentiate a wrapped angle.
    """
    vx, vy = v[:, 0], v[:, 1]
    ax, ay = a[:, 0], a[:, 1]
    speed2 = vx**2 + vy**2
    if np.any(speed2 <= 0):
        raise ValueError("yaw-aligned trajectory needs non-zero horizontal speed")
    yaw = np.arctan2(vy, vx)
    yaw_rate = (vx * ay - vy * ax) / speed2
    omega = np.zeros_like(p)
    omega[:, 2] = yaw_rate
    return TrajectorySample(t, p, v, a, quat_from_yaw(yaw), omega)


class Trajectory:
    name = "base"

    def sample(self, t: np.ndarray) -> TrajectorySample:  # pragma: no cover - interface
        raise NotImplementedError


class FigureEight(Trajectory):
    """Horizontal figure-eight: p = [Ax sin u, Ay sin 2u, 0], u = 2*pi*t/T."""

    name = "figure_eight"

    def __init__(self, period_s: float = 60.0, amp_x_m: float = 40.0, amp_y_m: float = 20.0):
        self.T, self.ax, self.ay = period_s, amp_x_m, amp_y_m

    def sample(self, t: np.ndarray) -> TrajectorySample:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        w = 2.0 * np.pi / self.T  # du/dt
        u = w * t
        z = np.zeros_like(t)
        p = np.column_stack([self.ax * np.sin(u), self.ay * np.sin(2 * u), z])
        v = np.column_stack([self.ax * w * np.cos(u), 2 * self.ay * w * np.cos(2 * u), z])
        a = np.column_stack([-self.ax * w**2 * np.sin(u), -4 * self.ay * w**2 * np.sin(2 * u), z])
        return _yaw_aligned(t, p, v, a)


class Stationary(Trajectory):
    """Body at rest with a fixed attitude (level when all angles are zero)."""

    name = "stationary"

    def __init__(self, position=(0.0, 0.0, 0.0), roll=0.0, pitch=0.0, yaw=0.0):
        self.p0 = np.asarray(position, dtype=float)
        self.q0 = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()

    def sample(self, t: np.ndarray) -> TrajectorySample:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        n = t.size
        z = np.zeros((n, 3))
        return TrajectorySample(t, np.tile(self.p0, (n, 1)), z.copy(), z.copy(), np.tile(self.q0, (n, 1)), z.copy())


class ConstantVelocity(Trajectory):
    """Straight horizontal line at constant velocity, heading along the velocity."""

    name = "constant_velocity"

    def __init__(self, position=(0.0, 0.0, 0.0), velocity=(5.0, 2.0, 0.0)):
        self.p0 = np.asarray(position, dtype=float)
        self.v0 = np.asarray(velocity, dtype=float)
        if self.v0[2] != 0.0:
            raise ValueError("constant_velocity fixture is horizontal; vz must be 0")

    def sample(self, t: np.ndarray) -> TrajectorySample:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        p = self.p0 + t[:, None] * self.v0
        v = np.tile(self.v0, (t.size, 1))
        return _yaw_aligned(t, p, v, np.zeros_like(p))


class ConstantYawRate(Trajectory):
    """Circle at constant speed and yaw rate, starting at the origin heading +x."""

    name = "constant_yaw_rate"

    def __init__(self, speed_mps: float = 5.0, yaw_rate_radps: float = 0.2):
        self.s, self.w = speed_mps, yaw_rate_radps

    def sample(self, t: np.ndarray) -> TrajectorySample:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        s, w = self.s, self.w
        r = s / w
        wt = w * t
        z = np.zeros_like(t)
        p = np.column_stack([r * np.sin(wt), r * (1 - np.cos(wt)), z])
        v = np.column_stack([s * np.cos(wt), s * np.sin(wt), z])
        a = np.column_stack([-s * w * np.sin(wt), s * w * np.cos(wt), z])
        return _yaw_aligned(t, p, v, a)


class SmoothRPY(Trajectory):
    """Short smooth roll/pitch/yaw motion while translating at constant velocity.

    Euler angles (ZYX, R = Rz(yaw) Ry(pitch) Rx(roll)) follow sinusoids; the body
    rate is the analytical ZYX kinematic map. Used to test frame conventions in 3-D.
    """

    name = "smooth_rpy"

    def __init__(self, amplitudes=(0.3, 0.2, 0.5), freqs_hz=(0.3, 0.2, 0.1), velocity=(1.0, 0.5, 0.2)):
        self.A = np.asarray(amplitudes, dtype=float)
        self.w = 2 * np.pi * np.asarray(freqs_hz, dtype=float)
        self.v0 = np.asarray(velocity, dtype=float)

    def sample(self, t: np.ndarray) -> TrajectorySample:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        (Ar, Ap, Ay), (wr, wp, wy) = self.A, self.w
        roll, pitch, yaw = Ar * np.sin(wr * t), Ap * np.sin(wp * t), Ay * np.sin(wy * t)
        droll, dpitch, dyaw = Ar * wr * np.cos(wr * t), Ap * wp * np.cos(wp * t), Ay * wy * np.cos(wy * t)
        sr, cr, sp, cp = np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch)
        omega = np.column_stack(
            [
                droll - dyaw * sp,
                dpitch * cr + dyaw * cp * sr,
                -dpitch * sr + dyaw * cp * cr,
            ]
        )
        q = Rotation.from_euler("ZYX", np.column_stack([yaw, pitch, roll])).as_quat()
        p = t[:, None] * self.v0
        v = np.tile(self.v0, (t.size, 1))
        return TrajectorySample(t, p, v, np.zeros_like(p), q, omega)


def make_trajectory(dataset_cfg: dict) -> Trajectory:
    """Build the trajectory named in ``dataset.trajectory``."""
    name = dataset_cfg["trajectory"]
    if name == "figure_eight":
        f8 = dataset_cfg["figure_eight"]
        return FigureEight(f8["period_s"], f8["amp_x_m"], f8["amp_y_m"])
    if name == "stationary":
        return Stationary()
    if name == "stationary_tilted":
        return Stationary(roll=0.2, pitch=-0.3, yaw=0.7)
    if name == "constant_velocity":
        return ConstantVelocity()
    if name == "constant_yaw_rate":
        return ConstantYawRate()
    if name == "smooth_rpy":
        return SmoothRPY()
    raise ValueError(f"unknown trajectory {name!r}")
