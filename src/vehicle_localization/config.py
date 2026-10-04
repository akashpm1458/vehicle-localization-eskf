"""Configuration loading, defaults and validation.

All tunable numbers live here or in ``configs/*.yaml``; none are hard-coded in the
estimator. The configuration is split into groups so that the runner can hand the
filter *only* the ``estimator`` group (what the filter is allowed to know) and keep
the ``truth`` group (what the simulator used) away from it.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

# The built-in defaults. ``configs/default.yaml`` is an explicit copy of this
# dictionary; a test keeps the two in sync.
DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "name": "default",
        "duration_s": 120.0,
        "trajectory": "figure_eight",
        "figure_eight": {"period_s": 60.0, "amp_x_m": 40.0, "amp_y_m": 20.0},
        "imu_rate_hz": 100,
        "gnss": {"rate_hz": 5, "first_time_s": 0.03},
        "lidar": {"enabled": True, "rate_hz": 10, "first_time_s": 0.07},
        "seed": 7,
        "output": "data/generated/default",
    },
    "truth": {
        "noise_free": False,
        "bias_mode": "biased",  # "biased" or "zero"
        "accel_noise_density": 0.02,  # m/s^(3/2)
        "gyro_noise_density": 0.001,  # rad/s^(1/2)
        "accel_bias_rw": 0.0002,  # (m/s^2)/sqrt(s)
        "gyro_bias_rw": 0.00002,  # (rad/s)/sqrt(s)
        "accel_bias_initial": [0.03, -0.02, 0.04],  # m/s^2
        "gyro_bias_initial": [0.001, -0.0005, 0.0008],  # rad/s
        "gnss": {"sigma_m": [1.0, 1.0, 2.0], "lever_arm_m": [0.2, 0.0, 1.0]},
        "lidar": {"sigma_m": [0.15, 0.15, 0.20], "lever_arm_m": [0.5, 0.0, 1.2]},
    },
    "estimator": {
        "mode": "eskf15_all",
        "gravity_mps2": [0.0, 0.0, -9.80665],
        "accel_noise_density": 0.02,
        "gyro_noise_density": 0.001,
        "accel_bias_rw": 0.0002,
        "gyro_bias_rw": 0.00002,
        "lever_arms_m": {"gnss": [0.2, 0.0, 1.0], "lidar": [0.5, 0.0, 1.2]},
        "covariance_scale": {"gnss": 1.0, "lidar": 1.0},
        "gating": {"enabled": True, "probability": 0.9973},
        "discretization": "fast",  # "fast" or "van_loan"
        "max_substep_s": 0.01,
        "max_imu_gap_s": 0.05,
        "covariance_check_every": 100,
        "full_covariance_log_every": 10,
        # How GNSS and LiDAR positions are combined when a mode uses both:
        #   "fuse"   - every measurement updates the filter, weighted by its covariance (default)
        #   "switch" - one source at a time: GNSS while it looks consistent, LiDAR otherwise
        #              (an innovation-based version of the switching idea in Wang et al. 2024)
        "fusion_policy": "fuse",
        "switching": {"judge_probability": 0.9973, "recover_after": 10, "gnss_timeout_s": 0.5},
    },
    "initialization": {
        "source": "generated_perturbed_truth",  # or "exact_truth" for noise-free fixtures
        "sigma": {
            "position_m": 1.0,
            "velocity_mps": 0.3,
            "attitude_deg": 2.0,
            "accel_bias_mps2": 0.1,
            "gyro_bias_radps": 0.005,
        },
    },
    "experiment": {
        "dropout_interval_s": [40.0, 60.0],
        "outliers": {
            "stream": "gnss",
            "fraction": 0.05,
            "offset_m": [20.0, -15.0, 8.0],
            "after_s": 10.0,
        },
        "gnss_covariance_scales": [0.1, 1.0, 10.0],
        "lidar_wrong_lever_arm_m": [0.7, 0.1, 1.2],
        # E9: a "viaduct" section where GNSS degrades without the receiver reporting it:
        # noise std multiplied by noise_scale plus a smooth multipath bias peaking at bias_m.
        "gnss_degradation": {"interval_s": [40.0, 60.0], "noise_scale": 5.0, "bias_m": [4.0, -3.0, 1.5]},
        "verification_seeds": [7, 23, 42],
    },
    "report": {
        "burn_in_s": 10.0,
        "dpi": 130,
        "relative_window_s": 10.0,
        "output": "results/suite",
    },
}

MODES: dict[str, tuple[int, tuple[str, ...]]] = {
    "imu_only": (9, ()),
    "eskf9_gnss": (9, ("gnss",)),
    "eskf9_all": (9, ("gnss", "lidar")),
    "eskf15_gnss": (15, ("gnss",)),
    "eskf15_all": (15, ("gnss", "lidar")),
}

TRAJECTORIES = (
    "figure_eight",
    "stationary",
    "stationary_tilted",
    "constant_velocity",
    "constant_yaw_rate",
    "smooth_rpy",
)


class ConfigError(ValueError):
    """Raised when a configuration fails validation."""


def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of ``base`` with ``override`` merged in recursively."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load a YAML config on top of the defaults, apply overrides, and validate."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        if not isinstance(user, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
        unknown = set(user) - set(DEFAULT_CONFIG)
        if unknown:
            raise ConfigError(f"{path}: unknown config groups {sorted(unknown)}")
        cfg = deep_merge(cfg, user)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def _vec3(errors: list[str], name: str, value: Any) -> None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in value)
    ):
        errors.append(f"{name} must be a list of 3 finite numbers, got {value!r}")


def _positive(errors: list[str], name: str, value: Any) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        errors.append(f"{name} must be a positive number, got {value!r}")


def _nonnegative(errors: list[str], name: str, value: Any) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        errors.append(f"{name} must be a non-negative number, got {value!r}")


def _rate_to_period_ns(errors: list[str], name: str, rate: Any) -> None:
    _positive(errors, name, rate)
    if isinstance(rate, (int, float)) and rate > 0:
        period = 1e9 / rate
        if abs(period - round(period)) > 1e-6:
            errors.append(f"{name}={rate} Hz does not give an integer-nanosecond period")


def validate_config(cfg: dict) -> None:
    """Check rates, densities, variances, lever arms and choices. Raise ConfigError."""
    errors: list[str] = []
    ds, tr, est, ini, exp, rep = (
        cfg["dataset"],
        cfg["truth"],
        cfg["estimator"],
        cfg["initialization"],
        cfg["experiment"],
        cfg["report"],
    )

    _positive(errors, "dataset.duration_s", ds["duration_s"])
    if ds["trajectory"] not in TRAJECTORIES:
        errors.append(f"dataset.trajectory must be one of {TRAJECTORIES}")
    _rate_to_period_ns(errors, "dataset.imu_rate_hz", ds["imu_rate_hz"])
    for s in ("gnss", "lidar"):
        _rate_to_period_ns(errors, f"dataset.{s}.rate_hz", ds[s]["rate_hz"])
        _nonnegative(errors, f"dataset.{s}.first_time_s", ds[s]["first_time_s"])
    if isinstance(ds["seed"], bool) or not isinstance(ds["seed"], int) or ds["seed"] < 0:
        errors.append("dataset.seed must be a non-negative integer")
    if isinstance(ds["imu_rate_hz"], (int, float)) and isinstance(ds["duration_s"], (int, float)):
        n = ds["duration_s"] * ds["imu_rate_hz"]
        if abs(n - round(n)) > 1e-9:
            errors.append("dataset.duration_s * imu_rate_hz must be an integer number of IMU intervals")

    if tr["bias_mode"] not in ("biased", "zero"):
        errors.append("truth.bias_mode must be 'biased' or 'zero'")
    for key in ("accel_noise_density", "gyro_noise_density", "accel_bias_rw", "gyro_bias_rw"):
        _nonnegative(errors, f"truth.{key}", tr[key])
        _nonnegative(errors, f"estimator.{key}", est[key])
    _vec3(errors, "truth.accel_bias_initial", tr["accel_bias_initial"])
    _vec3(errors, "truth.gyro_bias_initial", tr["gyro_bias_initial"])
    for s in ("gnss", "lidar"):
        _vec3(errors, f"truth.{s}.sigma_m", tr[s]["sigma_m"])
        if all(isinstance(x, (int, float)) for x in tr[s]["sigma_m"]) and min(tr[s]["sigma_m"]) <= 0:
            errors.append(f"truth.{s}.sigma_m must be strictly positive (measurement variances > 0)")
        _vec3(errors, f"truth.{s}.lever_arm_m", tr[s]["lever_arm_m"])
        _vec3(errors, f"estimator.lever_arms_m.{s}", est["lever_arms_m"][s])
        _positive(errors, f"estimator.covariance_scale.{s}", est["covariance_scale"][s])

    if est["mode"] not in MODES:
        errors.append(f"estimator.mode must be one of {sorted(MODES)}")
    _vec3(errors, "estimator.gravity_mps2", est["gravity_mps2"])
    p = est["gating"]["probability"]
    if not isinstance(p, (int, float)) or not 0.0 < p < 1.0:
        errors.append("estimator.gating.probability must be in (0, 1)")
    if est["discretization"] not in ("fast", "van_loan"):
        errors.append("estimator.discretization must be 'fast' or 'van_loan'")
    _positive(errors, "estimator.max_substep_s", est["max_substep_s"])
    if isinstance(est["max_substep_s"], (int, float)) and est["max_substep_s"] > 0.01 + 1e-12:
        errors.append("estimator.max_substep_s must be at most 0.01 s")
    _positive(errors, "estimator.max_imu_gap_s", est["max_imu_gap_s"])
    if est["fusion_policy"] not in ("fuse", "switch"):
        errors.append("estimator.fusion_policy must be 'fuse' or 'switch'")
    sw = est["switching"]
    if not isinstance(sw["judge_probability"], (int, float)) or not 0.0 < sw["judge_probability"] < 1.0:
        errors.append("estimator.switching.judge_probability must be in (0, 1)")
    if isinstance(sw["recover_after"], bool) or not isinstance(sw["recover_after"], int) or sw["recover_after"] < 1:
        errors.append("estimator.switching.recover_after must be a positive integer")
    _positive(errors, "estimator.switching.gnss_timeout_s", sw["gnss_timeout_s"])
    for key in ("covariance_check_every", "full_covariance_log_every"):
        if not isinstance(est[key], int) or est[key] < 1:
            errors.append(f"estimator.{key} must be a positive integer")

    if ini["source"] not in ("generated_perturbed_truth", "exact_truth"):
        errors.append("initialization.source must be 'generated_perturbed_truth' or 'exact_truth'")
    for key, value in ini["sigma"].items():
        _positive(errors, f"initialization.sigma.{key}", value)

    d = exp["dropout_interval_s"]
    if not (isinstance(d, (list, tuple)) and len(d) == 2 and 0 <= d[0] < d[1]):
        errors.append("experiment.dropout_interval_s must be [start, end) with 0 <= start < end")
    o = exp["outliers"]
    if o["stream"] not in ("gnss", "lidar"):
        errors.append("experiment.outliers.stream must be 'gnss' or 'lidar'")
    if not isinstance(o["fraction"], (int, float)) or not 0 <= o["fraction"] <= 1:
        errors.append("experiment.outliers.fraction must be in [0, 1]")
    _vec3(errors, "experiment.outliers.offset_m", o["offset_m"])
    _nonnegative(errors, "experiment.outliers.after_s", o["after_s"])
    for s in exp["gnss_covariance_scales"]:
        _positive(errors, "experiment.gnss_covariance_scales[]", s)
    _vec3(errors, "experiment.lidar_wrong_lever_arm_m", exp["lidar_wrong_lever_arm_m"])
    deg = exp["gnss_degradation"]
    di = deg["interval_s"]
    if not (isinstance(di, (list, tuple)) and len(di) == 2 and 0 <= di[0] < di[1]):
        errors.append("experiment.gnss_degradation.interval_s must be [start, end) with 0 <= start < end")
    if not isinstance(deg["noise_scale"], (int, float)) or deg["noise_scale"] < 1:
        errors.append("experiment.gnss_degradation.noise_scale must be >= 1")
    _vec3(errors, "experiment.gnss_degradation.bias_m", deg["bias_m"])

    _nonnegative(errors, "report.burn_in_s", rep["burn_in_s"])
    _positive(errors, "report.dpi", rep["dpi"])
    _positive(errors, "report.relative_window_s", rep["relative_window_s"])

    if errors:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))


def dump_config(cfg: dict, path: str | Path) -> None:
    """Write a fully resolved config as YAML."""
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
