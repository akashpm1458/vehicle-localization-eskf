"""Synthetic sensor generation from an independent truth trajectory.

Models (frames and symbols as in docs/math.md):

* Accelerometer (specific force):  f_m = R_WB^T (a_W - g_W) + b_a + n_a
* Gyroscope:                       w_m = w_B + b_g + n_g
* Bias random walk:                b[k+1] = b[k] + sigma_rw * sqrt(dt) * N(0, I)
* External position sensor s:      z_s = p_WB + R_WB @ l_Bs + n_s,   n_s ~ N(0, diag(sigma_s^2))

Noise convention: continuous white noise with E[n(t) n(t')^T] = sigma^2 I delta(t - t').
Over one IMU interval of length dt the interval-average noise has standard deviation
sigma / sqrt(dt). The filter uses exactly the same convention.

Every random quantity comes from its own seeded stream, so removing one sensor never
changes another sensor's samples.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .dataset import (
    SCHEMA_VERSION,
    STREAM_FILES,
    GroundTruth,
    read_json,
    write_imu_csv,
    write_json,
    write_position_csv,
    write_truth_csv,
)
from .rotations import quat_from_rotvec, quat_multiply, rotate_batch
from .state import InitialPrior
from .trajectory import TrajectorySample, make_trajectory

# Fixed identifiers so each stream's seed depends only on (dataset seed, stream name).
TRUE_GRAVITY_W = np.array([0.0, 0.0, -9.80665])

RANDOM_STREAMS = {
    "imu_noise": 101,
    "bias_walk": 102,
    "gnss_noise": 103,
    "lidar_noise": 104,
    "prior": 105,
    "outliers": 106,
}


def stream_rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, RANDOM_STREAMS[name]]))


@dataclass
class DatasetManifest:
    path: Path
    n_imu_rows: int
    counts: dict
    seed: int
    duration_s: float


def specific_force(sample: TrajectorySample, gravity_w: np.ndarray) -> np.ndarray:
    """f_B = R_WB^T (a_W - g_W). A level body at rest gives [0, 0, +9.80665]."""
    return rotate_batch(sample.q, sample.a - gravity_w, inverse=True)


def _period_ns(rate_hz: float) -> int:
    return int(round(1e9 / rate_hz))


def _sensor_times(first_s: float, rate_hz: float, t_end_ns: int) -> np.ndarray:
    first_ns = int(round(first_s * 1e9))
    if first_ns > t_end_ns:
        return np.zeros(0, dtype=np.int64)
    n = (t_end_ns - first_ns) // _period_ns(rate_hz) + 1
    return first_ns + np.arange(n, dtype=np.int64) * _period_ns(rate_hz)


def simulate(cfg: dict) -> dict:
    """Generate all streams in memory. Returns a dict of arrays and truth."""
    ds, tr, ini = cfg["dataset"], cfg["truth"], cfg["initialization"]
    seed = ds["seed"]
    g_w = TRUE_GRAVITY_W  # the simulator's own gravity; never read from estimator config
    traj = make_trajectory(ds)
    noise_free = bool(tr["noise_free"])

    # --- IMU timestamps: N intervals, N+1 rows (the last row only marks the end).
    dt_ns = _period_ns(ds["imu_rate_hz"])
    n_int = int(round(ds["duration_s"] * ds["imu_rate_hz"]))
    imu_t = np.arange(n_int + 1, dtype=np.int64) * dt_ns
    dt = dt_ns * 1e-9

    # Deterministic force/rate at interval midpoints (interval measurement convention).
    mid = traj.sample((imu_t[:-1] + dt_ns / 2) * 1e-9)
    f_true = specific_force(mid, g_w)
    w_true = mid.omega_b

    # --- Biases: random walk evaluated at IMU timestamps, held over each interval.
    zero_bias = tr["bias_mode"] == "zero"
    ba0 = np.zeros(3) if zero_bias else np.asarray(tr["accel_bias_initial"], dtype=float)
    bg0 = np.zeros(3) if zero_bias else np.asarray(tr["gyro_bias_initial"], dtype=float)
    sba = 0.0 if zero_bias else tr["accel_bias_rw"]
    sbg = 0.0 if zero_bias else tr["gyro_bias_rw"]
    rng_b = stream_rng(seed, "bias_walk")
    inc_a = rng_b.standard_normal((n_int, 3)) * sba * np.sqrt(dt)
    inc_g = rng_b.standard_normal((n_int, 3)) * sbg * np.sqrt(dt)
    ba = np.vstack([ba0, ba0 + np.cumsum(inc_a, axis=0)])  # (N+1, 3)
    bg = np.vstack([bg0, bg0 + np.cumsum(inc_g, axis=0)])

    # --- White noise: interval-average samples with std sigma / sqrt(dt).
    rng_i = stream_rng(seed, "imu_noise")
    n_a = rng_i.standard_normal((n_int, 3)) * tr["accel_noise_density"] / np.sqrt(dt)
    n_g = rng_i.standard_normal((n_int, 3)) * tr["gyro_noise_density"] / np.sqrt(dt)
    if noise_free:
        n_a[:] = 0.0
        n_g[:] = 0.0
    f_m = np.vstack([f_true + ba[:-1] + n_a, np.zeros((1, 3))])
    w_m = np.vstack([w_true + bg[:-1] + n_g, np.zeros((1, 3))])
    # Final row: finite placeholder (copy of the last reading). It is never propagated.
    f_m[-1], w_m[-1] = f_m[-2], w_m[-2]

    # --- External position sensors.
    streams = {}
    for name in ("gnss", "lidar"):
        if name == "lidar" and not ds["lidar"]["enabled"]:
            continue
        t_s = _sensor_times(ds[name]["first_time_s"], ds[name]["rate_hz"], int(imu_t[-1]))
        smp = traj.sample(t_s * 1e-9)
        lever = np.asarray(tr[name]["lever_arm_m"], dtype=float)
        sigma = np.asarray(tr[name]["sigma_m"], dtype=float)
        z_ref = smp.p + rotate_batch(smp.q, np.tile(lever, (t_s.size, 1)))
        noise = stream_rng(seed, f"{name}_noise").standard_normal((t_s.size, 3)) * sigma
        if noise_free:
            noise[:] = 0.0
        R = np.tile(np.diag(sigma**2), (t_s.size, 1, 1))
        streams[name] = {"t_ns": t_s, "z": z_ref + noise, "R": R, "z_true_ref": z_ref}

    # --- Ground truth at IMU timestamps (evaluation only).
    truth_s = traj.sample(imu_t * 1e-9)
    gt = GroundTruth(imu_t, truth_s.p, truth_s.v, truth_s.q, ba, bg, {})

    # --- Initial prior: truth at t=0 plus a sampled perturbation (ground-truth assisted).
    sig = ini["sigma"]
    att_sigma = np.deg2rad(sig["attitude_deg"])
    rng_p = stream_rng(seed, "prior")
    e_p = rng_p.standard_normal(3) * sig["position_m"]
    e_v = rng_p.standard_normal(3) * sig["velocity_mps"]
    e_th = rng_p.standard_normal(3) * att_sigma
    if ini["source"] == "exact_truth":
        e_p, e_v, e_th = np.zeros(3), np.zeros(3), np.zeros(3)
    p0, v0, q0 = truth_s.p[0], truth_s.v[0], truth_s.q[0]
    # Error is truth - estimate; attitude error eps satisfies R_true = R_est Exp(eps).
    prior_q = quat_multiply(q0, quat_from_rotvec(-e_th))
    P0 = np.diag(
        np.concatenate(
            [
                np.full(3, sig["position_m"] ** 2),
                np.full(3, sig["velocity_mps"] ** 2),
                np.full(3, att_sigma**2),
                np.full(3, sig["accel_bias_mps2"] ** 2),
                np.full(3, sig["gyro_bias_radps"] ** 2),
            ]
        )
    )
    prior = InitialPrior(
        t_ns=int(imu_t[0]),
        p=p0 - e_p,
        v=v0 - e_v,
        q=prior_q / np.linalg.norm(prior_q),
        ba=np.zeros(3),
        bg=np.zeros(3),
        P=P0,
        provenance={
            "method": "ground_truth_assisted" if ini["source"] != "exact_truth" else "exact_truth",
            "description": (
                "Generated from the true state at t=0 minus a perturbation sampled from the "
                "prior covariance (seeded 'prior' stream). Bias prior mean is zero."
                if ini["source"] != "exact_truth"
                else "Exact true state at t=0 (noise-free correctness fixture)."
            ),
        },
    )
    return {
        "imu_t": imu_t,
        "f_m": f_m,
        "w_m": w_m,
        "streams": streams,
        "truth": gt,
        "prior": prior,
        "prior_error": {"position_m": e_p.tolist(), "velocity_mps": e_v.tolist(), "attitude_rad": e_th.tolist()},
    }


def _metadata(cfg: dict, counts: dict) -> dict:
    ds = cfg["dataset"]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_kind": "synthetic",
        "name": ds["name"],
        "trajectory": ds["trajectory"],
        "frames": {
            "world": "W: right-handed local frame, z up (synthetic x/y are local axes)",
            "body": "B: IMU/body frame, x forward, y left, z up; IMU at the body origin",
            "rotation": "quaternion represents R_WB (body to world)",
        },
        "units": {"time": "integer nanoseconds from dataset start", "length": "m", "angle": "rad",
                  "specific_force": "m/s^2", "angular_rate": "rad/s", "covariance": "m^2"},
        "quaternion_order": "xyzw",
        "timing": {
            "imu": "row k is an interval measurement held over [t_k, t_k+1); the final row only marks the "
                   "end of coverage and is never propagated",
            "external": "position measurements are stamped at measurement time; no delivery delay",
        },
        "seed": ds["seed"],
        "duration_s": ds["duration_s"],
        "imu_rate_hz": ds["imu_rate_hz"],
        "sensors": {
            "imu": {"rate_hz": ds["imu_rate_hz"], "reference_point": "body origin"},
            "gnss": {"file": "gnss.csv", "rate_hz": ds["gnss"]["rate_hz"],
                     "reference_point": "GNSS antenna reference point", "count": counts.get("gnss", 0)},
            "lidar": {"file": "lidar_position.csv", "rate_hz": ds["lidar"]["rate_hz"],
                      "reference_point": "LiDAR-localizer reference point",
                      "note": "SIMULATED LiDAR-localizer positions (truth + noise), not real scan matching",
                      "count": counts.get("lidar", 0)},
        },
        "generator_version": __version__,
    }


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists and is not empty (use --overwrite)")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def generate_dataset(cfg: dict, output_dir: str | Path, overwrite: bool = False) -> DatasetManifest:
    """Simulate and save one dataset. Data are saved before any estimator runs."""
    out = Path(output_dir)
    _prepare_output(out, overwrite)
    sim = simulate(cfg)
    write_imu_csv(out / "imu.csv", sim["imu_t"], sim["f_m"], sim["w_m"])
    counts = {}
    for name, s in sim["streams"].items():
        write_position_csv(out / STREAM_FILES[name], s["t_ns"], s["z"], s["R"])
        counts[name] = int(s["t_ns"].size)
    sim["prior"].save(out / "initial_prior.json")
    write_json(out / "metadata.json", _metadata(cfg, counts))
    write_truth_csv(out / "ground_truth.csv", sim["truth"])
    write_json(
        out / "truth_metadata.json",
        {
            "warning": "evaluation only - the estimator must never read this file",
            "truth_config": cfg["truth"],
            "dataset_config": cfg["dataset"],
            "random_streams": {k: [cfg["dataset"]["seed"], v] for k, v in RANDOM_STREAMS.items()},
            "prior_error_truth_minus_estimate": sim["prior_error"],
            "corruptions": {},
            "generator_version": __version__,
        },
    )
    return DatasetManifest(out, int(sim["imu_t"].size), counts, cfg["dataset"]["seed"], cfg["dataset"]["duration_s"])


def derive_outlier_dataset(
    base_dir: str | Path, output_dir: str | Path, outlier_cfg: dict, overwrite: bool = False
) -> dict:
    """Copy a base dataset and add gross errors to a fraction of one stream.

    Indices are chosen with the dataset's 'outliers' random stream. All other data stay
    identical. Labels are written only to truth_metadata.json (evaluator access).
    """
    base, out = Path(base_dir), Path(output_dir)
    _prepare_output(out, overwrite)
    for f in base.iterdir():
        if f.is_file():
            shutil.copy2(f, out / f.name)
    meta = read_json(base / "metadata.json")
    tmeta = read_json(base / "truth_metadata.json")
    stream = outlier_cfg["stream"]
    fname = STREAM_FILES[stream]
    from .dataset import POSITION_COLUMNS  # local import keeps the public surface small

    raw = np.loadtxt(base / fname, delimiter=",", skiprows=1, ndmin=2)
    t_ns = np.loadtxt(base / fname, delimiter=",", skiprows=1, usecols=0, dtype=np.int64, ndmin=1)
    eligible = np.where(t_ns >= int(round(outlier_cfg["after_s"] * 1e9)))[0]
    n_out = int(round(outlier_cfg["fraction"] * eligible.size))
    rng = stream_rng(meta["seed"], "outliers")
    idx = np.sort(rng.choice(eligible, size=n_out, replace=False)) if n_out else np.zeros(0, dtype=int)
    raw[idx, 1:4] += np.asarray(outlier_cfg["offset_m"], dtype=float)
    data = raw.copy()
    data[:, 0] = t_ns
    np.savetxt(out / fname, data, delimiter=",", header=",".join(POSITION_COLUMNS), comments="",
               fmt=["%d"] + ["%.17g"] * 9)
    labels = np.zeros(t_ns.size, dtype=bool)
    labels[idx] = True
    tmeta["corruptions"] = {
        "outliers": {
            "stream": stream,
            "offset_m": outlier_cfg["offset_m"],
            "after_s": outlier_cfg["after_s"],
            "fraction": outlier_cfg["fraction"],
            "indices": idx.tolist(),
            "t_ns": t_ns[idx].tolist(),
        }
    }
    write_json(out / "truth_metadata.json", tmeta)
    meta["name"] = f"{meta['name']}_outliers"
    meta["derived_from"] = str(base)
    write_json(out / "metadata.json", meta)
    return {"stream": stream, "n_outliers": int(n_out), "n_measurements": int(t_ns.size), "labels": labels}
