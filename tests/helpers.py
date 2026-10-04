"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vehicle_localization.dataset import load_dataset, load_ground_truth
from vehicle_localization.eskf import ErrorStateEKF, EstimatorConfig
from vehicle_localization.sensors import generate_dataset

from .conftest import make_cfg


def fixture_dataset(tmp: Path, trajectory: str, duration: float = 10.0, imu_rate: int = 100, **truth):
    """Noise-free, zero-bias dataset with exact initialization."""
    cfg = make_cfg(
        dataset={"trajectory": trajectory, "duration_s": duration, "imu_rate_hz": imu_rate},
        truth={"noise_free": True, "bias_mode": "zero", **truth},
        initialization={"source": "exact_truth"},
    )
    out = tmp / f"{trajectory}_{imu_rate}"
    generate_dataset(cfg, out)
    return load_dataset(out), load_ground_truth(out), cfg


def est_config(cfg: dict, mode: str, **overrides) -> EstimatorConfig:
    est = EstimatorConfig.from_dict(cfg["estimator"], mode)
    for k, v in overrides.items():
        setattr(est, k, v)
    return est


class SpyEKF(ErrorStateEKF):
    """Records every propagation interval and update time; does not change the state."""

    def __init__(self, prior, config):
        super().__init__(prior, config)
        self.calls: list[tuple] = []
        SpyEKF.last = self

    def predict(self, specific_force, angular_rate, dt):
        assert dt > 0, "scheduler must never request a zero or negative step"
        self.calls.append(("predict", round(dt, 12), self.t_ns, tuple(np.round(specific_force, 12))))
        super().predict(specific_force, angular_rate, dt)

    def update_position(self, z_world, covariance, lever_arm_body, sensor):
        self.calls.append(("update", sensor, self.t_ns))
        return {"sensor": sensor, "accepted": True, "nis": 0.0, "threshold": 1.0, "gating_enabled": True,
                "z": z_world, "zhat": z_world, "r": np.zeros(3), "S": np.eye(3)}
