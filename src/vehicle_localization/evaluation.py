"""Errors, RMSE, uncertainty coverage and innovation statistics.

Conventions:
* Additive errors are truth minus estimate.
* Attitude error is the body-local rotation vector Log(R_est^T R_true), matching the
  filter's error definition, so it can be compared with the attitude covariance block.
* Quaternion components are never subtracted.
* Truth is matched at identical timestamps when available; otherwise interpolated
  (linear for vectors, SLERP for attitude) inside the overlap only - never extrapolated
  and never nearest-neighbour matched.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .dataset import GroundTruth
from .rotations import attitude_error_batch, rotate_batch


def align_truth(truth: GroundTruth, t_ns: np.ndarray) -> tuple[dict, np.ndarray]:
    """Truth at times ``t_ns``. Returns (fields, mask of times inside the truth span)."""
    t_ns = np.asarray(t_ns, dtype=np.int64)
    idx = np.searchsorted(truth.t_ns, t_ns)
    idx_c = np.clip(idx, 0, truth.t_ns.size - 1)
    if np.all(truth.t_ns[idx_c] == t_ns):  # exact timestamps (synthetic data)
        return {k: getattr(truth, k)[idx_c] for k in ("p", "v", "q", "ba", "bg")}, np.ones(t_ns.size, bool)
    inside = (t_ns >= truth.t_ns[0]) & (t_ns <= truth.t_ns[-1])
    tq = t_ns[inside].astype(float)
    tt = truth.t_ns.astype(float)
    out = {k: np.column_stack([np.interp(tq, tt, getattr(truth, k)[:, j]) for j in range(3)])
           for k in ("p", "v", "ba", "bg")}
    out["q"] = Slerp(tt, Rotation.from_quat(truth.q))(tq).as_quat()
    return out, inside


def _rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(e**2, axis=1)))) if e.size else float("nan")


def _axis_rmse(e: np.ndarray) -> list[float]:
    return np.sqrt(np.mean(e**2, axis=0)).tolist() if e.size else [float("nan")] * 3


def _coverage(err: np.ndarray, var: np.ndarray) -> list[float] | None:
    if var.size == 0 or np.all(np.isnan(var)):
        return None
    sd = np.sqrt(np.maximum(var, 0))
    return np.mean(np.abs(err) <= 3 * sd, axis=0).tolist()


def error_series(run, truth: GroundTruth) -> dict:
    """Per-sample errors at the run's logged timestamps (inside the truth span)."""
    tr, mask = align_truth(truth, run.t_ns)
    q_est = run.q[mask]
    dth = attitude_error_batch(tr["q"], q_est)
    return {
        "t_ns": run.t_ns[mask],
        "t_s": run.t_ns[mask] * 1e-9,
        "dp": tr["p"] - run.p[mask],
        "dv": tr["v"] - run.v[mask],
        "dth": dth,
        "dba": tr["ba"] - run.ba[mask],
        "dbg": tr["bg"] - run.bg[mask],
        "P_diag": run.P_diag[mask],
        "truth_ba": tr["ba"],
        "truth_bg": tr["bg"],
    }


def _block_metrics(es: dict, sel: np.ndarray, n_states: int) -> dict:
    dp, dv, dth = es["dp"][sel], es["dv"][sel], es["dth"][sel]
    Pd = es["P_diag"][sel]
    ang = np.rad2deg(np.linalg.norm(dth, axis=1))
    pos_norm = np.linalg.norm(dp, axis=1)
    m = {
        "samples": int(sel.sum()),
        "position_rmse_3d_m": _rmse(dp),
        "position_rmse_horizontal_m": _rmse(dp[:, :2]),
        "position_rmse_axis_m": _axis_rmse(dp),
        "position_mean_error_axis_m": dp.mean(axis=0).tolist() if dp.size else None,
        "position_final_error_m": float(pos_norm[-1]) if pos_norm.size else float("nan"),
        "position_max_error_m": float(pos_norm.max()) if pos_norm.size else float("nan"),
        "velocity_rmse_3d_mps": _rmse(dv),
        "velocity_rmse_axis_mps": _axis_rmse(dv),
        "attitude_rmse_deg": float(np.sqrt(np.mean(ang**2))) if ang.size else float("nan"),
        "attitude_max_error_deg": float(ang.max()) if ang.size else float("nan"),
        "coverage_3sigma": {
            "position_axis": _coverage(dp, Pd[:, 0:3]),
            "velocity_axis": _coverage(dv, Pd[:, 3:6]),
            "attitude_axis": _coverage(dth, Pd[:, 6:9]),
        },
    }
    dba, dbg = es["dba"][sel], es["dbg"][sel]
    m["accel_bias_rmse_mps2"] = _rmse(dba)
    m["accel_bias_rmse_axis_mps2"] = _axis_rmse(dba)
    m["gyro_bias_rmse_radps"] = _rmse(dbg)
    m["gyro_bias_rmse_axis_radps"] = _axis_rmse(dbg)
    m["biases_estimated"] = n_states == 15
    if n_states == 15:
        m["coverage_3sigma"]["accel_bias_axis"] = _coverage(dba, Pd[:, 9:12])
        m["coverage_3sigma"]["gyro_bias_axis"] = _coverage(dbg, Pd[:, 12:15])
    return m


def innovation_stats(inn: dict, threshold: float | None = None) -> dict:
    """Per-sensor counts and PRE-GATE NIS statistics (rejected measurements included)."""
    out = {}
    sensors = np.unique(inn["sensor"]) if len(inn["sensor"]) else []
    for s in sensors:
        m = inn["sensor"] == s
        nis = inn["nis"][m]
        acc = inn["accepted"][m].astype(bool)
        thr = float(inn["threshold"][m][0])
        out[str(s)] = {
            "processed": int(m.sum()),
            "accepted": int(acc.sum()),
            "rejected": int((~acc).sum()),
            "nis_threshold": thr,
            "nis_pre_gate_mean": float(nis.mean()),
            "nis_pre_gate_median": float(np.median(nis)),
            "nis_pre_gate_p95": float(np.percentile(nis, 95)),
            "nis_fraction_above_threshold": float(np.mean(nis > thr)),
            "nis_accepted_only_mean_truncated": float(nis[acc].mean()) if acc.any() else float("nan"),
            "measurement_dim": 3,
        }
    return out


def outlier_detection(inn: dict, labels_t_ns: list[int], stream: str) -> dict:
    """Recall of injected outliers and false-rejection rate of clean measurements."""
    m = inn["sensor"] == stream
    t = inn["t_ns"][m]
    rejected = ~inn["accepted"][m].astype(bool)
    is_out = np.isin(t, np.asarray(labels_t_ns, dtype=np.int64))
    n_out, n_clean = int(is_out.sum()), int((~is_out).sum())
    return {
        "stream": stream,
        "outliers_processed": n_out,
        "outliers_rejected": int((rejected & is_out).sum()),
        "recall": float((rejected & is_out).sum() / n_out) if n_out else float("nan"),
        "clean_processed": n_clean,
        "clean_rejected": int((rejected & ~is_out).sum()),
        "false_rejection_rate": float((rejected & ~is_out).sum() / n_clean) if n_clean else float("nan"),
    }


def dropout_metrics(es: dict, inn: dict, start_s: float, end_s: float, after_s: float = 5.0) -> dict:
    """Error and position-covariance trace before, during and after an outage [start, end)."""
    t = es["t_s"]
    err = np.linalg.norm(es["dp"], axis=1)
    tr = np.sum(es["P_diag"][:, 0:3], axis=1)

    def at(time_s: float, before: bool = False) -> dict:
        i = (np.searchsorted(t, time_s, side="left") - 1) if before else np.searchsorted(t, time_s, side="left")
        i = int(np.clip(i, 0, t.size - 1))
        return {"t_s": float(t[i]), "position_error_m": float(err[i]), "position_cov_trace_m2": float(tr[i])}

    during = (t >= start_s) & (t < end_s)
    tin = inn["t_ns"] * 1e-9
    window = (tin >= end_s) & (tin < end_s + 10.0)
    return {
        "interval_s": [start_s, end_s],
        "before_start": at(start_s, before=True),
        "end_of_outage": at(end_s, before=True),
        f"after_{after_s:g}s": at(end_s + after_s),
        "max_position_error_during_m": float(err[during].max()) if during.any() else float("nan"),
        "max_position_cov_trace_during_m2": float(tr[during].max()) if during.any() else float("nan"),
        "updates_in_10s_after": int(window.sum()),
        "rejections_in_10s_after": int((window & ~inn["accepted"].astype(bool)).sum()),
        "first_update_after_accepted": bool(inn["accepted"][window][0]) if window.any() else None,
    }


def external_reference_errors(streams: dict, truth: GroundTruth) -> dict:
    """Error of each raw position measurement against the truth of ITS OWN reference point.

    Uses the true lever arms from truth metadata (evaluator-only). These are measurement
    errors at the measurement timestamps, not a causal estimator.
    """
    arms = truth.metadata.get("truth_config", {})
    out = {}
    for name, s in streams.items():
        lever = arms.get(name, {}).get("lever_arm_m")
        if lever is None or s.t_ns.size == 0:
            continue
        tr, mask = align_truth(truth, s.t_ns)
        ref = tr["p"] + rotate_batch(tr["q"], np.tile(np.asarray(lever, float), (int(mask.sum()), 1)))
        e = ref - s.z[mask]
        out[name] = {"count": int(mask.sum()), "rmse_3d_m": _rmse(e), "rmse_horizontal_m": _rmse(e[:, :2]),
                     "rmse_axis_m": _axis_rmse(e)}
    return out


def evaluate_run(run, truth: GroundTruth | None, n_states: int, burn_in_s: float = 10.0,
                 dropouts: list[tuple[float, float]] | None = None, outlier_labels: dict | None = None) -> dict:
    """Metrics for one run. Without truth, only innovation statistics are reported."""
    inn = run.innovations if isinstance(run.innovations, dict) else _inn_to_arrays(run.innovations)
    metrics: dict = {"innovations": innovation_stats(inn)}
    if truth is None:
        metrics["accuracy"] = "unavailable (no ground truth)"
        return metrics
    es = error_series(run, truth)
    t = es["t_s"]
    duration = float(t[-1] - t[0]) if t.size else 0.0
    burn = burn_in_s
    note = None
    if burn >= duration:
        burn = 0.2 * duration
        note = f"burn-in {burn_in_s} s >= run length; used {burn:.2f} s instead"
    metrics["whole_run"] = _block_metrics(es, np.ones(t.size, bool), n_states)
    metrics["after_burn_in"] = _block_metrics(es, t >= t[0] + burn, n_states)
    metrics["burn_in_s"] = burn
    if note:
        metrics["burn_in_note"] = note
    if dropouts:
        metrics["dropouts"] = [dropout_metrics(es, inn, s, e) for s, e in dropouts]
    if outlier_labels:
        metrics["outlier_detection"] = outlier_detection(inn, outlier_labels["t_ns"], outlier_labels["stream"])
    return metrics


def _inn_to_arrays(records: list[dict]) -> dict:
    if not records:
        return {"sensor": np.array([], dtype=str), "nis": np.array([]), "accepted": np.array([], dtype=np.int64),
                "threshold": np.array([]), "t_ns": np.array([], dtype=np.int64)}
    return {
        "sensor": np.array([r["sensor"] for r in records], dtype=str),
        "nis": np.array([r["nis"] for r in records], dtype=float),
        "accepted": np.array([int(r["accepted"]) for r in records], dtype=np.int64),
        "threshold": np.array([r["threshold"] for r in records], dtype=float),
        "t_ns": np.array([r["t_ns"] for r in records], dtype=np.int64),
    }
