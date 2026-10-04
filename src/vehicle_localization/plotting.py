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
    """Plot kwargs for a method label; unknown labels get a distinct dash/marker combo."""
    for key, st in STYLE.items():
        if label == key or label.startswith(key + " "):
            s = st
            break
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
