"""Figures. Every figure is saved as a PNG plus a CSV of the numbers it shows.

Methods are distinguished by line style and marker as well as colour, so the plots
remain readable in greyscale and for colour-blind readers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CLI runs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STYLE = {
    "truth": {"color": "#222222", "ls": "-", "lw": 2.2, "marker": None},
    "imu_only": {"color": "#d62728", "ls": ":", "lw": 1.6, "marker": "x"},
    "eskf9_gnss": {"color": "#ff7f0e", "ls": "--", "lw": 1.4, "marker": "^"},
    "eskf9_all": {"color": "#9467bd", "ls": "-.", "lw": 1.4, "marker": "s"},
    "eskf15_gnss": {"color": "#2ca02c", "ls": "--", "lw": 1.4, "marker": "v"},
    "eskf15_all": {"color": "#1f77b4", "ls": "-", "lw": 1.4, "marker": "o"},
}
_CYCLE = [("-", "o"), ("--", "s"), ("-.", "^"), (":", "x"), ((0, (5, 1, 1, 1)), "D"), ((0, (1, 1)), "v")]


def style_for(label: str, i: int = 0) -> dict:
    """Plot kwargs for a method label.

    Exact method names get their fixed style; any other label (e.g. two variants of the
    same method in one plot) gets a distinct dash + marker combination from its index,
    so curves never differ by colour alone.
    """
    if label in STYLE:
        s = STYLE[label]
    else:
        ls, mk = _CYCLE[i % len(_CYCLE)]
        s = {"color": f"C{i % 10}", "ls": ls, "lw": 1.4, "marker": mk}
    out = {"color": s["color"], "linestyle": s["ls"], "linewidth": s["lw"]}
    if s["marker"]:
        out.update(marker=s["marker"], markevery=0.08, markersize=5)
    return out


def save_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    """Save the numbers behind a plot. Columns may differ in length (padded with NaN)."""
    n = max(len(v) for v in columns.values())
    data = np.full((n, len(columns)), np.nan)
    for j, v in enumerate(columns.values()):
        data[: len(v), j] = v
    np.savetxt(path, data, delimiter=",", header=",".join(columns), comments="", fmt="%.10g")


def _finish(fig, path: Path, dpi: int) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_ground_truth(truth, streams: dict, path: Path, dpi: int = 130) -> Path:
    """Milestone A check: the truth path and its derived quantities."""
    t = truth.t_ns * 1e-9
    yaw = np.unwrap(2 * np.arctan2(truth.q[:, 2], truth.q[:, 3]))
    speed = np.linalg.norm(truth.v, axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.plot(truth.p[:, 0], truth.p[:, 1], label="truth (body origin)", **style_for("truth"))
    for i, (name, s) in enumerate(streams.items()):
        ax.plot(s.z[:, 0], s.z[:, 1], linestyle="none", marker=".x"[i % 2], markersize=3, alpha=0.6,
                label=f"{name} measurements (simulated)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Ground-truth path and simulated position measurements")
    ax.axis("equal")
    ax.legend(fontsize=8)
    axes[1].plot(t, speed, "k-")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("speed [m/s]")
    axes[1].set_title("Speed (never zero on this path)")
    axes[2].plot(t, np.rad2deg(yaw), "k-")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("yaw [deg] (unwrapped)")
    axes[2].set_title("Heading follows the velocity")
    for a in axes:
        a.grid(alpha=0.3)
    save_csv(path.with_suffix(".csv"), {"t_s": t, "px": truth.p[:, 0], "py": truth.p[:, 1],
                                         "speed_mps": speed, "yaw_deg": np.rad2deg(yaw)})
    return _finish(fig, path, dpi)


def plot_position_error(series: dict, path: Path, title: str, dropouts=None, logy: bool = False,
                        dpi: int = 130) -> Path:
    """Position error norm vs time for several runs; dropout intervals shaded."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cols = {}
    for i, (label, es) in enumerate(series.items()):
        err = np.linalg.norm(es["dp"], axis=1)
        ax.plot(es["t_s"], err, label=label, **style_for(label, i))
        cols[f"{label} t_s"] = es["t_s"][::10]
        cols[f"{label} pos_err_m"] = err[::10]
    for j, (s, e, lab) in enumerate(dropouts or []):
        ax.axvspan(s, e, color="0.85", hatch="//", alpha=0.6, label=lab if j == 0 or lab else None)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("3-D position error |truth - estimate| [m]")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    save_csv(path.with_suffix(".csv"), cols)
    return _finish(fig, path, dpi)


def plot_trajectory_xy(truth, runs: dict, path: Path, title: str, imu_only=None, dpi: int = 130) -> Path:
    """XY tracks. A diverged IMU-only track gets its own panel so it cannot hide the fused ones."""
    ncols = 2 if imu_only is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5.5), squeeze=False)
    ax = axes[0, 0]
    ax.plot(truth.p[:, 0], truth.p[:, 1], label="truth", **style_for("truth"))
    cols = {"truth_x": truth.p[::10, 0], "truth_y": truth.p[::10, 1]}
    for i, (label, r) in enumerate(runs.items()):
        ax.plot(r.p[:, 0], r.p[:, 1], label=label, **style_for(label, i))
        cols[f"{label} x"], cols[f"{label} y"] = r.p[::10, 0], r.p[::10, 1]
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if imu_only is not None:
        ax2 = axes[0, 1]
        ax2.plot(truth.p[:, 0], truth.p[:, 1], label="truth", **style_for("truth"))
        ax2.plot(imu_only.p[:, 0], imu_only.p[:, 1], label="imu_only", **style_for("imu_only"))
        ax2.set_title("IMU-only dead reckoning (separate scale)")
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("y [m]")
        ax2.axis("equal")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)
        cols["imu_only x"], cols["imu_only y"] = imu_only.p[::10, 0], imu_only.p[::10, 1]
    save_csv(path.with_suffix(".csv"), cols)
    return _finish(fig, path, dpi)


def plot_axis_errors_with_bounds(es: dict, path: Path, title: str, block: str = "position", dropouts=None,
                                 dpi: int = 130) -> Path:
    """Per-axis error (truth - estimate) against +/-3 sqrt(P_ii) in matching coordinates."""
    key, sl, unit, scale = {
        "position": ("dp", slice(0, 3), "m", 1.0),
        "velocity": ("dv", slice(3, 6), "m/s", 1.0),
        "attitude": ("dth", slice(6, 9), "deg (local)", 180 / np.pi),
        "accel_bias": ("dba", slice(9, 12), "m/s^2", 1.0),
        "gyro_bias": ("dbg", slice(12, 15), "deg/s", 180 / np.pi),
    }[block]
    t = es["t_s"]
    err = es[key] * scale
    sd = np.sqrt(np.maximum(es["P_diag"][:, sl], 0)) * scale
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    cols = {"t_s": t[::10]}
    for j, ax in enumerate(axes):
        ax.plot(t, err[:, j], "k-", lw=1.0, label="error (truth - estimate)")
        ax.plot(t, 3 * sd[:, j], color="#1f77b4", ls="--", lw=1.2, label="+/- 3 sigma")
        ax.plot(t, -3 * sd[:, j], color="#1f77b4", ls="--", lw=1.2)
        for s, e, lab in dropouts or []:
            ax.axvspan(s, e, color="0.85", hatch="//", alpha=0.5)
        ax.set_ylabel(f"{'xyz'[j]} [{unit}]")
        ax.grid(alpha=0.3)
        inside = np.mean(np.abs(err[:, j]) <= 3 * sd[:, j]) * 100
        ax.text(0.01, 0.92, f"inside 3 sigma: {inside:.1f}% of samples (diagnostic only)",
                transform=ax.transAxes, fontsize=8, va="top")
        cols[f"err_{'xyz'[j]}"] = err[::10, j]
        cols[f"3sigma_{'xyz'[j]}"] = 3 * sd[::10, j]
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title(title)
    axes[-1].set_xlabel("time [s]")
    save_csv(path.with_suffix(".csv"), cols)
    return _finish(fig, path, dpi)


def plot_velocity_attitude_error(series: dict, path: Path, title: str, dpi: int = 130) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    cols = {}
    for i, (label, es) in enumerate(series.items()):
        ve = np.linalg.norm(es["dv"], axis=1)
        ae = np.rad2deg(np.linalg.norm(es["dth"], axis=1))
        axes[0].plot(es["t_s"], ve, label=label, **style_for(label, i))
        axes[1].plot(es["t_s"], ae, label=label, **style_for(label, i))
        cols[f"{label} t_s"] = es["t_s"][::10]
        cols[f"{label} vel_err_mps"] = ve[::10]
        cols[f"{label} att_err_deg"] = ae[::10]
    axes[0].set_ylabel("3-D velocity error [m/s]")
    axes[1].set_ylabel("attitude error angle [deg]")
    axes[1].set_xlabel("time [s]")
    axes[0].set_title(title)
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    save_csv(path.with_suffix(".csv"), cols)
    return _finish(fig, path, dpi)


def plot_biases(es: dict, run, path: Path, title: str, dpi: int = 130) -> Path:
    """Estimated vs true accelerometer and gyro biases (15-state), with 3-sigma bands."""
    t = es["t_s"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), sharex=True)
    cols = {"t_s": t[::10]}
    for row, (name, est_key, tru_key, sl, scale, unit) in enumerate([
        ("accel bias", "ba", "truth_ba", slice(9, 12), 1.0, "m/s^2"),
        ("gyro bias", "bg", "truth_bg", slice(12, 15), 180 / np.pi, "deg/s"),
    ]):
        est = getattr(run, est_key)[: t.size] * scale
        tru = es[tru_key] * scale
        sd = np.sqrt(np.maximum(es["P_diag"][:, sl], 0)) * scale
        for j in range(3):
            ax = axes[row, j]
            ax.plot(t, tru[:, j], "k-", lw=2, label="truth")
            ax.plot(t, est[:, j], color="#1f77b4", ls="--", lw=1.3, label="estimate")
            ax.fill_between(t, est[:, j] - 3 * sd[:, j], est[:, j] + 3 * sd[:, j], color="#1f77b4", alpha=0.15,
                            label="estimate +/- 3 sigma")
            ax.set_title(f"{name} {'xyz'[j]} [{unit}]", fontsize=10)
            ax.grid(alpha=0.3)
            cols[f"{est_key}_{'xyz'[j]}_true"] = tru[::10, j]
            cols[f"{est_key}_{'xyz'[j]}_est"] = est[::10, j]
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("time [s]")
    fig.suptitle(title)
    save_csv(path.with_suffix(".csv"), cols)
    return _finish(fig, path, dpi)


def gate_label(threshold: float, probability: float | None = None, enabled: bool = True) -> str:
    """Legend text for the NIS gate, derived from the configured probability.

    If no probability is given it is recovered from the logged threshold
    (threshold = chi2.ppf(p, 3)), so the label always matches what was applied.
    """
    from scipy.stats import chi2

    p = float(chi2.cdf(threshold, df=3)) if probability is None else float(probability)
    text = f"gate chi2(3), p = {100 * p:.4g}%: NIS = {threshold:.2f}"
    return text if enabled else text + " (gating disabled; not applied)"


def plot_nis(innovations: dict, path: Path, title: str, outlier_t_ns=None, dpi: int = 130,
             gate_probability: float | None = None) -> Path:
    """Pre-gate NIS per sensor with the gate threshold; rejected measurements marked."""
    sensors = [s for s in ("gnss", "lidar") if np.any(innovations["sensor"] == s)]
    fig, axes = plt.subplots(len(sensors) or 1, 1, figsize=(13, 3.2 * max(1, len(sensors))), sharex=True,
                             squeeze=False)
    cols = {}
    for ax, s in zip(axes[:, 0], sensors):
        m = innovations["sensor"] == s
        t = innovations["t_ns"][m] * 1e-9
        nis = innovations["nis"][m]
        acc = innovations["accepted"][m].astype(bool)
        thr = innovations["threshold"][m][0]
        ax.semilogy(t[acc], nis[acc], linestyle="none", marker=".", color="#1f77b4", markersize=3,
                    label="accepted (pre-gate NIS)")
        ax.semilogy(t[~acc], nis[~acc], linestyle="none", marker="x", color="#d62728", markersize=6,
                    label=f"rejected ({int((~acc).sum())})")
        if outlier_t_ns is not None and s == outlier_t_ns[0]:
            lab = np.isin(innovations["t_ns"][m], outlier_t_ns[1])
            ax.semilogy(t[lab], nis[lab], linestyle="none", marker="o", mfc="none", color="k", markersize=8,
                        label="injected outlier (evaluator label)")
        enabled = bool(np.any(innovations["gating_enabled"][m])) if "gating_enabled" in innovations else True
        ax.axhline(thr, color="k", ls="--", lw=1, label=gate_label(thr, gate_probability, enabled))
        ax.axhline(3.0, color="0.5", ls=":", lw=1, label="E[NIS] = 3 if consistent")
        ax.set_ylabel(f"{s} NIS")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))  # outside: never hides data
        cols[f"{s} t_s"], cols[f"{s} nis"], cols[f"{s} accepted"] = t, nis, acc.astype(float)
    axes[0, 0].set_title(title)
    axes[-1, 0].set_xlabel("time [s]")
    if cols:
        save_csv(path.with_suffix(".csv"), cols)
    return _finish(fig, path, dpi)
