"""Nominal state, error-state indexing and the initial prior.

Nominal state (16 stored numbers):  p_W (3), v_W (3), q_WB (4, xyzw), b_a (3), b_g (3)
Error state   (15 numbers):         dp (3), dv (3), dtheta (3, body-local), dba (3), dbg (3)

The 9-state filter uses only the first three error blocks and treats both biases as
known zeros (10 nominal numbers, 9x9 covariance).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .rotations import quat_normalize

# Error-state slices. Use these everywhere instead of literal indices.
POS = slice(0, 3)
VEL = slice(3, 6)
ATT = slice(6, 9)
BA = slice(9, 12)
BG = slice(12, 15)
ERROR_BLOCKS = {"position": POS, "velocity": VEL, "attitude": ATT, "accel_bias": BA, "gyro_bias": BG}
ERROR_STATE_LABELS = [
    "px", "py", "pz", "vx", "vy", "vz", "thx", "thy", "thz",
    "bax", "bay", "baz", "bgx", "bgy", "bgz",
]


@dataclass
class NominalState:
    """Full nominal state. Biases are zero (and stay zero) in the 9-state filter."""

    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bg: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "NominalState":
        return NominalState(self.p.copy(), self.v.copy(), self.q.copy(), self.ba.copy(), self.bg.copy())


@dataclass
class InitialPrior:
    """The one externally supplied starting estimate the filter may read."""

    t_ns: int
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray  # xyzw
    ba: np.ndarray
    bg: np.ndarray
    P: np.ndarray  # 15x15 covariance of the error state
    provenance: dict

    def nominal(self) -> NominalState:
        return NominalState(self.p.copy(), self.v.copy(), quat_normalize(self.q), self.ba.copy(), self.bg.copy())

    def to_json_dict(self) -> dict:
        return {
            "t_ns": int(self.t_ns),
            "position_m": self.p.tolist(),
            "velocity_mps": self.v.tolist(),
            "quaternion": {"order": "xyzw", "value": self.q.tolist()},
            "accel_bias_mps2": self.ba.tolist(),
            "gyro_bias_radps": self.bg.tolist(),
            "covariance": {
                "error_state_order": ERROR_STATE_LABELS,
                "units": "m, m/s, rad (body-local), m/s^2, rad/s",
                "matrix": self.P.tolist(),
            },
            "provenance": self.provenance,
        }

    @staticmethod
    def from_json_dict(d: dict) -> "InitialPrior":
        quat = d["quaternion"]
        if not isinstance(quat, dict) or quat.get("order") != "xyzw":
            raise ValueError("initial_prior.quaternion must be {'order': 'xyzw', 'value': [...]}")
        return InitialPrior(
            t_ns=int(d["t_ns"]),
            p=np.asarray(d["position_m"], dtype=float),
            v=np.asarray(d["velocity_mps"], dtype=float),
            q=np.asarray(quat["value"], dtype=float),
            ba=np.asarray(d["accel_bias_mps2"], dtype=float),
            bg=np.asarray(d["gyro_bias_radps"], dtype=float),
            P=np.asarray(d["covariance"]["matrix"], dtype=float),
            provenance=d.get("provenance", {}),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "InitialPrior":
        return InitialPrior.from_json_dict(json.loads(path.read_text(encoding="utf-8")))


def validate_prior(prior: InitialPrior) -> list[str]:
    """Return a list of problems with a prior (empty when valid)."""
    errors: list[str] = []
    for name, arr, n in (("position", prior.p, 3), ("velocity", prior.v, 3), ("quaternion", prior.q, 4),
                         ("accel_bias", prior.ba, 3), ("gyro_bias", prior.bg, 3)):
        if arr.shape != (n,):
            errors.append(f"prior {name} must have {n} elements, got shape {arr.shape}")
        elif not np.all(np.isfinite(arr)):
            errors.append(f"prior {name} has non-finite values")
    if prior.q.shape == (4,) and np.all(np.isfinite(prior.q)) and abs(np.linalg.norm(prior.q) - 1.0) > 1e-6:
        errors.append(f"prior quaternion is not unit length (norm={np.linalg.norm(prior.q):.9f})")
    if prior.P.shape != (15, 15):
        errors.append(f"prior covariance must be 15x15, got {prior.P.shape}")
    elif not np.all(np.isfinite(prior.P)):
        errors.append("prior covariance has non-finite values")
    else:
        if not np.allclose(prior.P, prior.P.T, rtol=0, atol=1e-12 * max(1.0, np.abs(prior.P).max())):
            errors.append("prior covariance is not symmetric")
        if np.linalg.eigvalsh(0.5 * (prior.P + prior.P.T)).min() <= 0:
            errors.append("prior covariance is not positive definite")
    return errors
