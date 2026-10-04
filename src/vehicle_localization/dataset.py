"""Dataset files: writing, loading and strict validation.

A dataset directory contains::

    imu.csv               t_ns,fx,fy,fz,wx,wy,wz           (estimator input)
    gnss.csv              t_ns,x,y,z,r_xx,...,r_zz          (estimator input)
    lidar_position.csv    same schema as gnss.csv, optional (estimator input)
    initial_prior.json    the one prior the estimator may read
    metadata.json         schema, frames, units, timing semantics
    ground_truth.csv      evaluation only - the estimator never opens it
    truth_metadata.json   evaluation only - true parameters and corruption labels

``load_dataset`` reads only the estimator inputs. ``load_ground_truth`` is used by the
evaluator alone. See docs/data_format.md for the full contract.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .state import InitialPrior, validate_prior

SCHEMA_VERSION = "1.0"
IMU_COLUMNS = ["t_ns", "fx", "fy", "fz", "wx", "wy", "wz"]
POSITION_COLUMNS = ["t_ns", "x", "y", "z", "r_xx", "r_xy", "r_xz", "r_yy", "r_yz", "r_zz"]
TRUTH_COLUMNS = [
    "t_ns", "px", "py", "pz", "vx", "vy", "vz", "qx", "qy", "qz", "qw",
    "bax", "bay", "baz", "bgx", "bgy", "bgz",
]
STREAM_FILES = {"gnss": "gnss.csv", "lidar": "lidar_position.csv"}
REQUIRED_METADATA = [
    "schema_version", "frames", "units", "quaternion_order", "timing",
    "seed", "duration_s", "sensors", "generator_version",
]
# Same-time measurements are processed in this deterministic order.
STREAM_ORDER = ("gnss", "lidar")

# Machine-readable physical conventions this code implements. metadata.json must state
# exactly these values; anything else (another frame, unit or timing model) is rejected
# rather than silently misinterpreted.
SUPPORTED_SCHEMA_VERSIONS = ("1.0",)
SUPPORTED_CONVENTIONS = {
    "frames": {
        "world": "right_handed_z_up",
        "body": "x_forward_y_left_z_up",
        "rotation": "R_WB_body_to_world",
    },
    "units": {
        "time": "ns",
        "length": "m",
        "angle": "rad",
        "specific_force": "m/s^2",
        "angular_rate": "rad/s",
        "covariance": "m^2",
    },
    "timing": {
        "imu": "interval_held_forward_last_row_end_marker",
        "external": "measurement_time_no_delivery_delay",
    },
}


class DatasetError(ValueError):
    """Raised when a dataset fails validation."""


@dataclass
class PositionStream:
    name: str
    t_ns: np.ndarray  # int64, sorted by measurement time
    z: np.ndarray  # (M, 3) world position of the sensor reference point, m
    R: np.ndarray  # (M, 3, 3) reported measurement covariance, m^2


@dataclass
class Dataset:
    """Estimator-visible data only (no ground truth)."""

    path: Path
    imu_t_ns: np.ndarray  # (N+1,) int64; row k is held over [t_k, t_k+1)
    imu_f: np.ndarray  # (N+1, 3) specific force, m/s^2
    imu_w: np.ndarray  # (N+1, 3) angular rate, rad/s
    streams: dict[str, PositionStream]
    prior: InitialPrior
    metadata: dict
    warnings: list[str] = field(default_factory=list)


@dataclass
class GroundTruth:
    t_ns: np.ndarray
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    ba: np.ndarray
    bg: np.ndarray
    metadata: dict


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    summary: dict

    def report(self) -> str:
        lines = [f"Dataset validation: {'PASSED' if self.ok else 'FAILED'}"]
        for k, v in self.summary.items():
            lines.append(f"  {k}: {v}")
        lines += [f"  ERROR: {e}" for e in self.errors]
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


# ----------------------------------------------------------------------------- writing

def _fmt(n_float: int) -> list[str]:
    # 17 significant digits round-trip float64 exactly.
    return ["%d"] + ["%.17g"] * n_float


def write_imu_csv(path: Path, t_ns: np.ndarray, f: np.ndarray, w: np.ndarray) -> None:
    data = np.column_stack([t_ns.astype(np.float64), f, w])
    np.savetxt(path, data, delimiter=",", header=",".join(IMU_COLUMNS), comments="", fmt=_fmt(6))


def cov_to_columns(R: np.ndarray) -> np.ndarray:
    return np.column_stack([R[:, 0, 0], R[:, 0, 1], R[:, 0, 2], R[:, 1, 1], R[:, 1, 2], R[:, 2, 2]])


def columns_to_cov(c: np.ndarray) -> np.ndarray:
    R = np.empty((c.shape[0], 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = c[:, 0], c[:, 1], c[:, 2]
    R[:, 1, 0], R[:, 1, 1], R[:, 1, 2] = c[:, 1], c[:, 3], c[:, 4]
    R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = c[:, 2], c[:, 4], c[:, 5]
    return R


def write_position_csv(path: Path, t_ns: np.ndarray, z: np.ndarray, R: np.ndarray) -> None:
    data = np.column_stack([t_ns.astype(np.float64), z, cov_to_columns(R)])
    np.savetxt(path, data, delimiter=",", header=",".join(POSITION_COLUMNS), comments="", fmt=_fmt(9))


def write_truth_csv(path: Path, gt: GroundTruth) -> None:
    data = np.column_stack([gt.t_ns.astype(np.float64), gt.p, gt.v, gt.q, gt.ba, gt.bg])
    np.savetxt(path, data, delimiter=",", header=",".join(TRUTH_COLUMNS), comments="", fmt=_fmt(16))


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- reading

def _read_csv(path: Path, columns: list[str], errors: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """Read a CSV with an exact header. Returns (t_ns int64, float data) or None."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
    except OSError as exc:
        errors.append(f"{path.name}: cannot read ({exc})")
        return None
    if header != columns:
        hint = ""
        if "qw" in header and "qx" in header and header.index("qw") < header.index("qx"):
            hint = " (quaternion columns look like wxyz order; this project requires qx,qy,qz,qw)"
        errors.append(f"{path.name}: header must be {','.join(columns)}, got {','.join(header)}{hint}")
        return None
    with path.open("r", encoding="utf-8") as fh:
        fh.readline()
        has_rows = any(line.strip() for line in fh)
    if not has_rows:
        # Header-only file: keep the documented column dimensions for downstream code.
        return np.zeros(0, dtype=np.int64), np.zeros((0, len(columns) - 1))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = np.loadtxt(path, delimiter=",", skiprows=1, usecols=0, dtype=np.int64, ndmin=1)
            # Parse every column (no usecols) so rows with extra/missing fields are rejected.
            full = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except ValueError as exc:
        errors.append(f"{path.name}: malformed rows or wrong number of columns / non-integer t_ns ({exc})")
        return None
    if full.shape[0] != t.shape[0] or (full.size and full.shape[1] != len(columns)):
        errors.append(f"{path.name}: wrong dimensions {full.shape}, expected {len(columns)} columns")
        return None
    data = full[:, 1:]
    if not np.all(np.isfinite(data)):
        bad = int(np.sum(~np.all(np.isfinite(data), axis=1)))
        errors.append(f"{path.name}: {bad} row(s) contain non-finite values (NaN/inf)")
        return None
    return t, data


def _check_covariances(name: str, R: np.ndarray, errors: list[str]) -> None:
    if R.shape[0] == 0:
        return
    eig = np.linalg.eigvalsh(R)
    bad = np.where(eig[:, 0] <= 0)[0]
    if bad.size:
        errors.append(f"{name}: {bad.size} measurement covariance(s) are not positive definite (first row {bad[0]})")


def _check_metadata(meta: dict, errors: list[str]) -> None:
    """Reject metadata whose declared conventions differ from what the estimator implements."""
    missing = [k for k in REQUIRED_METADATA if k not in meta]
    if missing:
        errors.append(f"metadata.json: missing required keys {missing}")
    if "schema_version" in meta and meta["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(f"metadata.json: unsupported schema_version {meta['schema_version']!r}; "
                      f"supported: {list(SUPPORTED_SCHEMA_VERSIONS)}")
    if meta.get("quaternion_order") != "xyzw":
        errors.append(f"metadata.json: quaternion_order must be 'xyzw', got {meta.get('quaternion_order')!r}")
    for group, required in SUPPORTED_CONVENTIONS.items():
        if group not in meta:
            continue  # already reported as missing
        declared = meta[group]
        if not isinstance(declared, dict):
            errors.append(f"metadata.json: '{group}' must be an object with keys {sorted(required)}")
            continue
        for key, expected in required.items():
            got = declared.get(key)
            if got != expected:
                errors.append(f"metadata.json: {group}.{key} must be {expected!r} (the only supported "
                              f"convention), got {got!r}; convert the data before using it")


def _inspect(path: Path, normalize_imu: bool = False) -> tuple[ValidationResult, Dataset | None]:
    path = Path(path)
    errors: list[str] = []
    warns: list[str] = []
    summary: dict = {"path": str(path)}
    if not path.is_dir():
        return ValidationResult(False, [f"{path} is not a directory"], [], summary), None

    # metadata
    meta: dict = {}
    if (path / "metadata.json").exists():
        try:
            meta = read_json(path / "metadata.json")
            _check_metadata(meta, errors)
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"metadata.json: unreadable ({exc})")
    else:
        errors.append("metadata.json is missing")

    # IMU
    imu = None
    if (path / "imu.csv").exists():
        imu = _read_csv(path / "imu.csv", IMU_COLUMNS, errors)
    else:
        errors.append("imu.csv is missing")
    imu_t = imu_f = imu_w = None
    if imu is not None:
        imu_t, d = imu
        if imu_t.size < 2:
            errors.append("imu.csv needs at least 2 rows (one supported interval)")
        else:
            dt = np.diff(imu_t)
            if np.any(dt < 0):
                if normalize_imu:
                    order = np.argsort(imu_t, kind="stable")
                    imu_t, d = imu_t[order], d[order]
                    dt = np.diff(imu_t)
                    warns.append("imu.csv was not time-ordered; sorted because import normalization was requested")
                else:
                    errors.append(f"imu.csv: {int(np.sum(dt < 0))} negative time step(s); IMU must be time-ordered")
            if np.any(dt == 0):
                errors.append(f"imu.csv: {int(np.sum(dt == 0))} duplicate timestamp(s)")
            pos = dt[dt > 0]
            if pos.size:
                med = np.median(pos)
                if np.any(pos > 5 * med):
                    warns.append(f"imu.csv: {int(np.sum(pos > 5 * med))} gap(s) longer than 5x the median step")
                summary["imu_rows"] = int(imu_t.size)
                summary["imu_median_dt_s"] = float(med * 1e-9)
        imu_f, imu_w = d[:, 0:3], d[:, 3:6]

    # external position streams
    streams: dict[str, PositionStream] = {}
    for name, fname in STREAM_FILES.items():
        fpath = path / fname
        if not fpath.exists():
            if name == "gnss":
                errors.append("gnss.csv is missing")
            continue
        res = _read_csv(fpath, POSITION_COLUMNS, errors)
        if res is None:
            continue
        t, d = res
        if t.size and np.any(np.diff(t) < 0):
            warns.append(f"{fname}: not time-ordered; sorted by measurement time (assumes no delivery delay)")
            order = np.argsort(t, kind="stable")
            t, d = t[order], d[order]
        if t.size and np.any(np.diff(t) == 0):
            errors.append(f"{fname}: duplicate timestamps within one stream")
        R = columns_to_cov(d[:, 3:9])
        _check_covariances(fname, R, errors)
        streams[name] = PositionStream(name, t, d[:, 0:3].copy(), R)
        summary[f"{name}_rows"] = int(t.size)
        if t.size == 0:
            warns.append(f"{fname}: contains no measurements (header only); modes that use '{name}' will "
                         f"receive no {name} aiding from this dataset")
        if imu_t is not None and imu_t.size >= 2 and t.size:
            out = int(np.sum((t < imu_t[0]) | (t > imu_t[-1])))
            if out:
                warns.append(f"{fname}: {out} measurement(s) outside IMU coverage will be skipped, not extrapolated")

    # prior
    prior = None
    if (path / "initial_prior.json").exists():
        try:
            prior = InitialPrior.load(path / "initial_prior.json")
            errors += validate_prior(prior)
            if imu_t is not None and imu_t.size and prior.t_ns != imu_t[0]:
                errors.append(f"initial_prior.t_ns ({prior.t_ns}) must equal the first IMU timestamp ({imu_t[0]})")
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"initial_prior.json: invalid ({exc})")
    else:
        errors.append("initial_prior.json is missing")

    ok = not errors
    ds = None
    if ok:
        ds = Dataset(path, imu_t, imu_f, imu_w, streams, prior, meta, warns)
    return ValidationResult(ok, errors, warns, summary), ds


def _inspect_truth(path: Path) -> tuple[list[str], GroundTruth | None]:
    """Evaluation-side validation of ground_truth.csv (never used for estimation)."""
    errors: list[str] = []
    res = _read_csv(path / "ground_truth.csv", TRUTH_COLUMNS, errors)
    if res is None:
        return errors, None
    t, d = res
    if t.size == 0:
        errors.append("ground_truth.csv: contains no rows")
        return errors, None
    dt = np.diff(t)
    if np.any(dt == 0):
        errors.append(f"ground_truth.csv: {int(np.sum(dt == 0))} duplicate timestamp(s); truth must be strictly "
                      "increasing before it can be interpolated")
    if np.any(dt < 0):
        errors.append(f"ground_truth.csv: {int(np.sum(dt < 0))} decreasing timestamp(s); truth must be strictly "
                      "increasing before it can be interpolated")
    norms = np.linalg.norm(d[:, 6:10], axis=1)
    if np.any(np.abs(norms - 1) > 1e-6):
        errors.append("ground_truth.csv: quaternions are not unit length")
    if errors:
        return errors, None
    meta: dict = {}
    if (path / "truth_metadata.json").exists():
        try:
            meta = read_json(path / "truth_metadata.json")
        except (json.JSONDecodeError, OSError) as exc:
            return [f"truth_metadata.json: unreadable ({exc})"], None
    return [], GroundTruth(t, d[:, 0:3], d[:, 3:6], d[:, 6:10], d[:, 10:13], d[:, 13:16], meta)


def validate_dataset(dataset_dir: str | Path, normalize_imu: bool = False) -> ValidationResult:
    """Validate a dataset directory without raising.

    Estimator inputs and evaluation-only truth are checked separately. ``ok`` is False
    if either is invalid; truth problems are prefixed with "[evaluation]" so it is clear
    that estimation itself is still possible.
    """
    path = Path(dataset_dir)
    result, _ = _inspect(path, normalize_imu)
    if path.is_dir() and (path / "ground_truth.csv").exists():
        truth_errors, _ = _inspect_truth(path)
        result.errors += [f"[evaluation] {e}" for e in truth_errors]
        result.summary["ground_truth"] = "present, invalid" if truth_errors else "present, valid"
        result.ok = result.ok and not truth_errors
    else:
        result.summary["ground_truth"] = "absent (accuracy metrics will be disabled)"
    return result


def load_dataset(dataset_dir: str | Path, normalize_imu: bool = False) -> Dataset:
    """Load estimator inputs only. Never opens ground_truth.csv or truth_metadata.json.

    Raises DatasetError if the inputs are invalid.
    """
    result, ds = _inspect(Path(dataset_dir), normalize_imu)
    if not result.ok:
        raise DatasetError(result.report())
    return ds


def load_ground_truth(dataset_dir: str | Path) -> GroundTruth | None:
    """Evaluator-only access to truth. Returns None when no truth file exists.

    Raises DatasetError if the truth file is malformed (wrong columns, non-finite values,
    non-unit quaternions, or duplicate/decreasing timestamps).
    """
    path = Path(dataset_dir)
    if not (path / "ground_truth.csv").exists():
        return None
    errors, gt = _inspect_truth(path)
    if errors:
        raise DatasetError("invalid ground truth: " + "; ".join(errors))
    return gt
