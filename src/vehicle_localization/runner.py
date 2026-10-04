"""Timestamp scheduling, logging and saving of one estimator run.

IMU convention: row k (time t_k) is held over [t_k, t_k+1). The final IMU row only
marks the end of coverage and is never propagated.

For every external measurement at time t (t_k <= t <= t_k+1):
  1. propagate to t with the held reading of interval k,
  2. update at t,
  3. continue to t_k+1 with the same raw reading (and the *current* bias estimate).
At an IMU boundary, propagation of the previous interval finishes first; then all
measurements stamped exactly at that boundary are applied in a fixed order (GNSS, then
LiDAR); then the posterior is logged; then the next interval starts.

Integer nanoseconds are used for all scheduling; dt = (t2 - t1) * 1e-9 is derived only
by subtracting timestamps.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dataset import STREAM_ORDER, Dataset
from .eskf import ErrorStateEKF, EstimatorConfig


class RunError(RuntimeError):
    """The run could not be completed (e.g. missing IMU coverage)."""


@dataclass
class Dropout:
    stream: str
    start_ns: int
    end_ns: int  # half-open [start, end)

    @staticmethod
    def from_seconds(stream: str, start_s: float, end_s: float) -> "Dropout":
        """Build a validated dropout. Raises ValueError for an unknown stream, a non-finite
        time or an empty/reversed interval (which would otherwise silently remove nothing)."""
        if stream not in STREAM_ORDER:
            raise ValueError(f"dropout stream must be one of {list(STREAM_ORDER)}, got {stream!r}")
        try:
            a, b = float(start_s), float(end_s)
        except (TypeError, ValueError):
            raise ValueError(f"dropout times must be numbers, got {start_s!r}, {end_s!r}") from None
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError(f"dropout times must be finite, got [{start_s}, {end_s})")
        if not a < b:
            raise ValueError(f"dropout interval must have start < end, got [{a:g}, {b:g})")
        return Dropout(stream, int(round(a * 1e9)), int(round(b * 1e9)))

    @staticmethod
    def parse(spec: str) -> "Dropout":
        """Parse 'STREAM:START_S:END_S' (half-open interval, seconds from dataset start)."""
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"dropout must look like STREAM:START_S:END_S (e.g. gnss:40:60), got {spec!r}")
        return Dropout.from_seconds(parts[0].strip(), parts[1], parts[2])


@dataclass
class RunResult:
    mode: str
    n_states: int
    t_ns: np.ndarray
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    ba: np.ndarray
    bg: np.ndarray
    P_diag: np.ndarray  # (rows, n) variances
    cov_t_ns: np.ndarray
    cov: np.ndarray  # (M, n, n) full covariance at a reduced rate
    innovations: list[dict]
    counts: dict
    info: dict = field(default_factory=dict)


def _build_events(ds: Dataset, cfg: EstimatorConfig, dropouts: list[Dropout]) -> tuple[np.ndarray, list, dict,
                                                                                     list[dict]]:
    t0, t_end = int(ds.imu_t_ns[0]), int(ds.imu_t_ns[-1])
    times, order, idx, names = [], [], [], []
    counts: dict = {}
    effects = [{"stream": d.stream, "start_s": d.start_ns * 1e-9, "end_s": d.end_ns * 1e-9, "removed": 0,
                "note": ""} for d in dropouts]
    for e in effects:
        if e["stream"] not in cfg.sensors:
            e["note"] = f"stream '{e['stream']}' is not used by mode {cfg.mode}"
    for name in STREAM_ORDER:
        if name not in cfg.sensors:
            continue
        s = ds.streams.get(name)
        c = {"available": 0, "removed_by_dropout": 0, "out_of_range": 0, "processed": 0,
             "accepted": 0, "rejected": 0}
        counts[name] = c
        if s is None:
            for e in effects:
                if e["stream"] == name:
                    e["note"] = f"stream '{name}' is absent from this dataset"
            continue
        t = s.t_ns
        c["available"] = int(t.size)
        keep = np.ones(t.size, dtype=bool)
        for d, e in zip(dropouts, effects):
            if d.stream == name:
                drop = (t >= d.start_ns) & (t < d.end_ns)
                e["removed"] = int(np.sum(drop & keep))
                c["removed_by_dropout"] += e["removed"]
                keep &= ~drop
        oor = (t < t0) | (t > t_end)
        c["out_of_range"] = int(np.sum(oor & keep))
        keep &= ~oor
        sel = np.where(keep)[0]
        times.append(t[sel])
        order.append(np.full(sel.size, STREAM_ORDER.index(name)))
        idx.append(sel)
        names.append(name)
    for e in effects:
        if e["removed"] == 0 and not e["note"]:
            e["note"] = "no measurements fall in this interval"
    if not times:
        return np.zeros(0, dtype=np.int64), [], counts, effects
    t_all = np.concatenate(times)
    o_all = np.concatenate(order)
    i_all = np.concatenate(idx)
    srt = np.lexsort((i_all, o_all, t_all))  # by time, then GNSS before LiDAR, then index
    events = [(int(t_all[k]), STREAM_ORDER[o_all[k]], int(i_all[k])) for k in srt]
    return t_all[srt], events, counts, effects


def run_estimator(
    dataset: Dataset,
    config: EstimatorConfig,
    dropouts: list[Dropout] | None = None,
    filter_cls=ErrorStateEKF,
) -> RunResult:
    """Run one estimator configuration over a dataset. Never reads ground truth."""
    wall0 = time.perf_counter()
    dropouts = dropouts or []
    ekf = filter_cls(dataset.prior, config)
    imu_t = dataset.imu_t_ns
    n_rows = imu_t.size
    n = config.n_states
    ev_t, events, counts, dropout_effects = _build_events(dataset, config, dropouts)
    n_ev = len(events)

    log_p = np.empty((n_rows, 3))
    log_v = np.empty((n_rows, 3))
    log_q = np.empty((n_rows, 4))
    log_ba = np.empty((n_rows, 3))
    log_bg = np.empty((n_rows, 3))
    log_Pd = np.empty((n_rows, n))
    every = config.full_covariance_log_every
    cov_rows = list(range(0, n_rows, every))
    if cov_rows[-1] != n_rows - 1:
        cov_rows.append(n_rows - 1)
    cov_t = imu_t[cov_rows]
    cov = np.empty((len(cov_rows), n, n))
    cov_slot = {r: j for j, r in enumerate(cov_rows)}
    innovations: list[dict] = []

    def log(row: int) -> None:
        log_p[row], log_v[row], log_q[row] = ekf.p, ekf.v, ekf.q
        log_ba[row], log_bg[row] = ekf.ba, ekf.bg
        log_Pd[row] = np.diag(ekf.P)
        j = cov_slot.get(row)
        if j is not None:
            cov[j] = ekf.P

    def apply(event) -> None:
        t, name, i = event
        s = dataset.streams[name]
        R = s.R[i] * config.covariance_scale.get(name, 1.0)
        rec = ekf.update_position(s.z[i], R, config.lever_arms[name], name)
        rec["t_ns"] = t
        rec["index"] = i
        innovations.append(rec)
        counts[name]["processed"] += 1
        counts[name]["accepted" if rec["accepted"] else "rejected"] += 1

    ek = 0
    t_cur = int(imu_t[0])
    ekf.t_ns = t_cur
    # Measurements exactly at the prior time: update before any propagation.
    while ek < n_ev and events[ek][0] == t_cur:
        apply(events[ek])
        ek += 1
    log(0)

    for k in range(n_rows - 1):
        t_a, t_b = int(imu_t[k]), int(imu_t[k + 1])
        if (t_b - t_a) * 1e-9 > config.max_imu_gap_s:
            raise RunError(
                f"IMU gap of {(t_b - t_a) * 1e-9:.3f} s at t={t_a * 1e-9:.3f} s exceeds "
                f"max_imu_gap_s={config.max_imu_gap_s}; refusing to hold an old reading across it"
            )
        f, w = dataset.imu_f[k], dataset.imu_w[k]
        # Off-grid measurements strictly inside (t_a, t_b).
        while ek < n_ev and events[ek][0] < t_b:
            t_e = events[ek][0]
            ekf.predict(f, w, (t_e - t_cur) * 1e-9)
            t_cur = t_e
            ekf.t_ns = t_cur
            while ek < n_ev and events[ek][0] == t_e:
                apply(events[ek])
                ek += 1
        # Finish the interval, then apply measurements stamped at the boundary.
        ekf.predict(f, w, (t_b - t_cur) * 1e-9)
        t_cur = t_b
        ekf.t_ns = t_cur
        while ek < n_ev and events[ek][0] == t_b:
            apply(events[ek])
            ek += 1
        log(k + 1)

    if ek != n_ev:  # pragma: no cover - guarded by the out-of-range filter
        raise RunError(f"{n_ev - ek} measurement events were not processed")
    ekf.check_covariance("end of run")
    info = {
        "wall_time_s": time.perf_counter() - wall0,
        "predict_substeps": ekf.n_predict_steps,
        "imu_intervals_processed": n_rows - 1,
        "covariance_checks": ekf.cov_stats,
        "discretization": config.discretization,
        "dropouts": dropout_effects,
        "warnings": [f"dropout {e['stream']} [{e['start_s']:g}, {e['end_s']:g}) s removed no measurements: "
                     f"{e['note']}" for e in dropout_effects if e["removed"] == 0],
        "final_quaternion_norm_error": float(abs(np.linalg.norm(ekf.q) - 1.0)),
        "max_logged_quaternion_norm_error": float(np.max(np.abs(np.linalg.norm(log_q, axis=1) - 1.0))),
    }
    return RunResult(config.mode, n, imu_t.copy(), log_p, log_v, log_q, log_ba, log_bg, log_Pd,
                     cov_t, cov, innovations, counts, info)


# ----------------------------------------------------------------------------- saving

ESTIMATE_COLUMNS = ["t_ns", "px", "py", "pz", "vx", "vy", "vz", "qx", "qy", "qz", "qw",
                    "bax", "bay", "baz", "bgx", "bgy", "bgz"]
VAR_LABELS = ["P_px", "P_py", "P_pz", "P_vx", "P_vy", "P_vz", "P_thx", "P_thy", "P_thz",
              "P_bax", "P_bay", "P_baz", "P_bgx", "P_bgy", "P_bgz"]
INNOVATION_COLUMNS = ["t_ns", "sensor", "index", "z_x", "z_y", "z_z", "zhat_x", "zhat_y", "zhat_z",
                      "r_x", "r_y", "r_z", "S_xx", "S_xy", "S_xz", "S_yy", "S_yz", "S_zz",
                      "nis", "threshold", "gating_enabled", "accepted"]


def save_run(result: RunResult, out_dir: Path) -> None:
    """Write estimates.csv, covariance.npz and innovations.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    Pd = np.full((result.t_ns.size, 15), np.nan)
    Pd[:, : result.n_states] = result.P_diag
    data = np.column_stack([result.t_ns.astype(float), result.p, result.v, result.q, result.ba, result.bg, Pd])
    fmt = ["%d"] + ["%.12g"] * 16 + ["%.6g"] * 15
    np.savetxt(out_dir / "estimates.csv", data, delimiter=",", comments="",
               header=",".join(ESTIMATE_COLUMNS + VAR_LABELS), fmt=fmt)
    np.savez_compressed(out_dir / "covariance.npz", t_ns=result.cov_t_ns, P=result.cov,
                        n_states=np.array(result.n_states))
    with (out_dir / "innovations.csv").open("w", encoding="utf-8") as fh:
        fh.write(",".join(INNOVATION_COLUMNS) + "\n")
        for r in result.innovations:
            S = r["S"]
            vals = [str(r["t_ns"]), r["sensor"], str(r["index"])]
            vals += [f"{x:.10g}" for x in (*r["z"], *r["zhat"], *r["r"])]
            vals += [f"{x:.8g}" for x in (S[0, 0], S[0, 1], S[0, 2], S[1, 1], S[1, 2], S[2, 2])]
            vals += [f"{r['nis']:.8g}", f"{r['threshold']:.8g}", str(int(r["gating_enabled"])), str(int(r["accepted"]))]
            fh.write(",".join(vals) + "\n")


@dataclass
class LoadedRun:
    t_ns: np.ndarray
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    ba: np.ndarray
    bg: np.ndarray
    P_diag: np.ndarray  # (rows, 15) with NaN for unused states
    innovations: dict  # column -> array


def load_run(out_dir: Path) -> LoadedRun:
    d = np.loadtxt(out_dir / "estimates.csv", delimiter=",", skiprows=1, ndmin=2)
    t = np.loadtxt(out_dir / "estimates.csv", delimiter=",", skiprows=1, usecols=0, dtype=np.int64, ndmin=1)
    inn: dict = {c: [] for c in INNOVATION_COLUMNS}
    with (out_dir / "innovations.csv").open("r", encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            for c, v in zip(INNOVATION_COLUMNS, line.strip().split(",")):
                inn[c].append(v)
    conv = {}
    for c, vals in inn.items():
        if c == "sensor":
            conv[c] = np.array(vals, dtype=str)
        elif c in ("t_ns", "index", "gating_enabled", "accepted"):
            conv[c] = np.array(vals, dtype=np.int64)
        else:
            conv[c] = np.array(vals, dtype=float)
    return LoadedRun(t, d[:, 1:4], d[:, 4:7], d[:, 7:11], d[:, 11:14], d[:, 14:17], d[:, 17:32], conv)
