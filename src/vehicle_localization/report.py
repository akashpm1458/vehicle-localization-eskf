"""Suite figures and the generated Markdown report.

Every number in the report is read from saved files (comparison.csv, metrics.json,
e0_fixtures.json, suite_status.json). Sentences that compare results are produced by
code from those numbers; nothing is typed in by hand.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import __version__
from .dataset import load_ground_truth, read_json
from .evaluation import error_series
from .experiments import read_comparison
from .plotting import (
    _finish,
    plot_axis_errors_with_bounds,
    plot_biases,
    plot_nis,
    plot_position_error,
    plot_trajectory_xy,
    plot_velocity_attitude_error,
    save_csv,
    style_for,
)
from .runner import load_run

FIGURES = [
    ("fig01_trajectory_xy.png", "XY ground truth, fused trajectories (biased data) and IMU-only on its own scale"),
    ("fig02_position_error_dropouts.png", "Position error over time for the dropout experiments E3-E5 "
                                          "(shaded: outage interval)"),
    ("fig03_position_axes_bounds.png", "Per-axis position error of eskf15_all with +/-3 sigma bounds (E2)"),
    ("fig03b_attitude_axes_bounds.png", "Per-axis local attitude error of eskf15_all with +/-3 sigma bounds (E2)"),
    ("fig04_velocity_attitude_error.png", "Velocity and attitude error: 9-state vs 15-state on biased data (E2)"),
    ("fig05_biases.png", "Estimated vs true IMU biases, eskf15_all (E2)"),
    ("fig06_nis_outliers.png", "Pre-gate NIS per sensor with the configured chi-square gate; rejected and "
                               "injected outliers marked (E8, eskf15_all, gating on)"),
    ("fig07_calibration_and_dropout.png", "Left: lever-arm calibration error (E7), signed position error in "
                                          "the body frame. Right: total-dropout close-up (E5) with 3-sigma bound"),
    ("fig08_summary.png", "Compact experiment summary (metric values with units)"),
    ("fig09_e1_position_error.png", "E1: IMU-only vs 9-state fusion on zero-bias data (log scale)"),
    ("fig10_e6_gnss_covariance_scaling.png", "E6: effect of the assumed GNSS covariance scale"),
    ("fig11_e9_e10_degraded_gnss_and_switching.png", "E9/E10: position error with degraded GNSS (shaded) for "
                                                     "GNSS-only, fused and switching filters, and which source "
                                                     "the switching filter used"),
]


# ----------------------------------------------------------------------------- figures


def make_suite_figures(suite_dir: Path, cfg: dict, seed: int) -> list[Path]:
    fig_dir = suite_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    dpi = cfg["report"]["dpi"]
    runs_dir = suite_dir / "runs" / f"seed_{seed}"
    data_dir = suite_dir / "data" / f"seed_{seed}"
    truth = {k: load_ground_truth(data_dir / k) for k in ("zero_bias", "biased", "biased_outliers",
                                                           "biased_gnss_degraded") if (data_dir / k).exists()}
    dataset_of = {"E1": "zero_bias", "E8": "biased_outliers", "E9": "biased_gnss_degraded"}
    d0, d1 = cfg["experiment"]["dropout_interval_s"]
    cache: dict = {}

    def run(rid):
        if rid not in cache:
            cache[rid] = load_run(runs_dir / rid)
        return cache[rid]

    def es(rid):
        return error_series(run(rid), truth[dataset_of.get(rid.split("_")[0], "biased")])

    def have(*rids):
        return all((runs_dir / r / "estimates.csv").exists() for r in rids)

    out = []
    if have("E2_imu_only", "E2_eskf9_all", "E2_eskf15_all"):
        out.append(plot_trajectory_xy(truth["biased"], {"eskf9_all": run("E2_eskf9_all"),
                                                        "eskf15_all": run("E2_eskf15_all")},
                                      fig_dir / FIGURES[0][0], "Biased data: truth and fused estimates",
                                      imu_only=run("E2_imu_only"), dpi=dpi))
    ids = ["E3_eskf15_gnss_reference", "E3_eskf15_gnss_gnss_dropout", "E4_eskf15_all_gnss_dropout",
           "E5_eskf15_all_total_dropout"]
    if have(*ids):
        labels = ["eskf15_gnss (no dropout)", "eskf15_gnss E3: GNSS lost", "eskf15_all E4: GNSS lost, LiDAR on",
                  "eskf15_all E5: both lost"]
        out.append(plot_position_error({lab: es(r) for lab, r in zip(labels, ids)}, fig_dir / FIGURES[1][0],
                                       "Dropout experiments", dropouts=[(d0, d1, "outage")], logy=True, dpi=dpi))
    if have("E2_eskf15_all"):
        e = es("E2_eskf15_all")
        out.append(plot_axis_errors_with_bounds(e, fig_dir / FIGURES[2][0], "eskf15_all position error (E2)",
                                                "position", dpi=dpi))
        out.append(plot_axis_errors_with_bounds(e, fig_dir / FIGURES[3][0], "eskf15_all attitude error (E2)",
                                                "attitude", dpi=dpi))
        out.append(plot_biases(e, run("E2_eskf15_all"), fig_dir / FIGURES[5][0],
                               "eskf15_all: IMU bias estimates (E2)", dpi=dpi))
    if have("E2_eskf9_all", "E2_eskf15_all"):
        out.append(plot_velocity_attitude_error({"eskf9_all": es("E2_eskf9_all"), "eskf15_all": es("E2_eskf15_all")},
                                                fig_dir / FIGURES[4][0], "Biased data (E2)", dpi=dpi))
    if have("E8_eskf15_all_gating_on"):
        o = truth["biased_outliers"].metadata.get("corruptions", {}).get("outliers") or {"stream": "gnss",
                                                                                        "t_ns": []}
        out.append(plot_nis(run("E8_eskf15_all_gating_on").innovations, fig_dir / FIGURES[6][0],
                            "E8: eskf15_all with injected GNSS outliers, gating on",
                            outlier_t_ns=(o["stream"], np.asarray(o["t_ns"])), dpi=dpi,
                            gate_probability=cfg["estimator"]["gating"]["probability"]))
    if have("E7_eskf15_all_lidar_arm_correct", "E7_eskf15_all_lidar_arm_wrong", "E5_eskf15_all_total_dropout"):
        out.append(_calibration_dropout_figure(es("E7_eskf15_all_lidar_arm_correct"),
                                               es("E7_eskf15_all_lidar_arm_wrong"),
                                               es("E5_eskf15_all_total_dropout"), d0, d1,
                                               fig_dir / FIGURES[7][0], dpi))
    rows = [r for r in read_comparison(suite_dir / "comparison.csv") if r["seed"] == seed]
    out.append(_summary_figure(rows, fig_dir / FIGURES[8][0], dpi))
    if have("E1_imu_only", "E1_eskf9_gnss", "E1_eskf9_all"):
        out.append(plot_position_error({m: es(f"E1_{m}") for m in ("imu_only", "eskf9_gnss", "eskf9_all")},
                                       fig_dir / FIGURES[9][0], "E1: zero-bias noisy motion", logy=True, dpi=dpi))
    e6 = [r for r in rows if r["experiment"] == "E6"]
    if e6:
        out.append(_e6_figure(e6, fig_dir / FIGURES[10][0], dpi))
    e9 = ["E9_eskf15_gnss_degraded", "E9_eskf15_all_fuse_degraded", "E9_eskf15_all_switch_degraded",
          "E10_eskf15_all_switch"]
    if have(*e9):
        g0, g1 = cfg["experiment"]["gnss_degradation"]["interval_s"]
        pol = json.loads((runs_dir / "E9_eskf15_all_switch_degraded" / "metrics.json").read_text())["fusion_policy"]
        out.append(_e9_figure({"GNSS only (E9)": es(e9[0]), "fuse GNSS + LiDAR (E9)": es(e9[1]),
                               "switch GNSS / LiDAR (E9)": es(e9[2]), "switch, nominal data (E10)": es(e9[3])},
                              pol, g0, g1, fig_dir / FIGURES[11][0], dpi))
    return out


def _calibration_dropout_figure(es_ok, es_bad, es_drop, d0, d1, path: Path, dpi: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    cols = {"t_s": es_ok["t_s"][::10]}
    for j, ls in zip(range(2), ("-", "--")):
        ax.plot(es_ok["t_s"], es_ok["dp_body"][:, j], color="#1f77b4", ls=ls, lw=1.0,
                label=f"correct arm, body {'xyz'[j]}")
        ax.plot(es_bad["t_s"], es_bad["dp_body"][:, j], color="#d62728", ls=ls, lw=1.0, marker="ox"[j],
                markevery=0.05, markersize=4, label=f"wrong arm, body {'xyz'[j]}")
        cols[f"correct_body_{'xyz'[j]}"] = es_ok["dp_body"][::10, j]
        cols[f"wrong_body_{'xyz'[j]}"] = es_bad["dp_body"][::10, j]
    ax.set_title("E7: signed position error in the BODY frame (truth - estimate)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_ylim(-0.6, 0.6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1]
    sel = (es_drop["t_s"] >= d0 - 5) & (es_drop["t_s"] <= d1 + 20)
    t = es_drop["t_s"][sel]
    err = np.linalg.norm(es_drop["dp"][sel], axis=1)
    bound = 3 * np.sqrt(np.sum(es_drop["P_diag"][sel, 0:3], axis=1))
    ax.plot(t, err, "k-", lw=1.4, label="|position error|")
    ax.plot(t, bound, color="#1f77b4", ls="--", lw=1.4, label="3 sqrt(trace P_pos)")
    ax.axvspan(d0, d1, color="0.85", hatch="//", alpha=0.6, label="GNSS + LiDAR outage")
    ax.set_yscale("log")
    ax.set_title("E5: total dropout close-up")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("[m]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    cols["e5_t_s"], cols["e5_err_m"], cols["e5_3sigma_m"] = t, err, bound
    save_csv(path.with_suffix(".csv"), cols)
    return _finish(fig, path, dpi)


def _e9_figure(series: dict, policy: dict, g0: float, g1: float, path: Path, dpi: int) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    cols = {}
    for i, (label, e) in enumerate(series.items()):
        err = np.linalg.norm(e["dp"], axis=1)
        ax.plot(e["t_s"], err, label=label, **style_for(label, i))
        cols[f"{label} t_s"], cols[f"{label} pos_err_m"] = e["t_s"][::10], err[::10]
    for a in axes:
        a.axvspan(g0, g1, color="0.85", hatch="//", alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("3-D position error [m]")
    ax.set_title(f"E9/E10: GNSS degraded in [{g0:g}, {g1:g}) s (shaded), receiver unaware")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    # source used by the switching filter (E9)
    t_end = series[next(iter(series))]["t_s"][-1]
    times, src = [0.0], [1]
    for sw in policy.get("switches", []):
        times.append(sw["t_s"])
        src.append(1 if sw["to"] == "gnss" else 0)
    times.append(t_end)
    src.append(src[-1])
    axes[1].step(times, src, where="post", color="k", lw=1.5)
    axes[1].set_yticks([0, 1], ["LiDAR", "GNSS"])
    axes[1].set_ylim(-0.3, 1.3)
    axes[1].set_ylabel("switch (E9)\nsource")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(alpha=0.3)
    cols["switch_t_s"], cols["switch_source_gnss1_lidar0"] = np.array(times), np.array(src, float)
    save_csv(path.with_suffix(".csv"), cols)
    return _finish(fig, path, dpi)


def _summary_figure(rows: list[dict], path: Path, dpi: int) -> Path:
    head = ["run", "pos RMSE [m]", "vel RMSE [m/s]", "att RMSE [deg]", "GNSS acc/rej", "LiDAR acc/rej"]
    cells = []
    for r in rows:
        cells.append([r["run_id"], _f(r["pos_rmse_3d_m"]), _f(r["vel_rmse_3d_mps"]), _f(r["att_rmse_deg"]),
                      f"{int(r['gnss_accepted'])}/{int(r['gnss_rejected'])}",
                      f"{int(r['lidar_accepted'])}/{int(r['lidar_rejected'])}"])
    fig, ax = plt.subplots(figsize=(13, 0.32 * len(cells) + 1.2))
    ax.axis("off")
    tab = ax.table(cellText=cells, colLabels=head, loc="center", cellLoc="left", colLoc="left")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.auto_set_column_width(list(range(len(head))))
    ax.set_title("Experiment summary (after burn-in; seed shown in report)", fontsize=10)
    save_csv(path.with_suffix(".csv"), {h: np.arange(0) for h in ["see comparison.csv"]})
    return _finish(fig, path, dpi)


def _e6_figure(rows: list[dict], path: Path, dpi: int) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = {}
    for i, mode in enumerate(("eskf15_gnss", "eskf15_all")):
        rr = sorted([r for r in rows if r["mode"] == mode], key=lambda r: float(r["run_id"].split("_x")[-1]))
        sc = [float(r["run_id"].split("_x")[-1]) for r in rr]
        kw = style_for(mode, i)
        kw.pop("markevery", None)
        axes[0].plot(sc, [r["pos_rmse_3d_m"] for r in rr], label=mode, **kw)
        axes[1].plot(sc, [r["gnss_nis_mean_pre_gate"] for r in rr], label=mode, **kw)
        axes[2].plot(sc, [r["gnss_rejected"] for r in rr], label=mode, **kw)
        cols[f"{mode} scale"] = np.array(sc)
        cols[f"{mode} pos_rmse_m"] = np.array([r["pos_rmse_3d_m"] for r in rr])
        cols[f"{mode} gnss_nis_mean"] = np.array([r["gnss_nis_mean_pre_gate"] for r in rr])
        cols[f"{mode} gnss_rejected"] = np.array([r["gnss_rejected"] for r in rr])
    axes[1].axhline(3, color="0.5", ls=":", label="3 (consistent)")
    for ax, yl in zip(axes, ("position RMSE [m]", "mean pre-gate GNSS NIS", "GNSS measurements rejected")):
        ax.set_xscale("log")
        ax.set_xlabel("assumed GNSS covariance scale")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[1].set_yscale("log")
    fig.suptitle("E6: same measurements, different assumed GNSS covariance")
    save_csv(path.with_suffix(".csv"), cols)
    return _finish(fig, path, dpi)


# ----------------------------------------------------------------------------- report


def _f(v, nd=3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_f(x, nd) for x in v) + "]"
    if isinstance(v, (int, np.integer)):
        return str(v)
    av = abs(v)
    if av != 0 and (av < 10 ** (-nd) or av >= 1e5):
        return f"{v:.{nd}e}"
    return f"{v:.{nd}f}"


def _table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _pct(a: float, b: float) -> str:
    return f"{100 * (a - b) / b:+.1f}%"


def write_report(suite_dir: Path, cfg: dict, status: dict, host: str, title: str = "Experiment report",
                 verification: dict | None = None) -> Path:
    rows_all = read_comparison(suite_dir / "comparison.csv") if (suite_dir / "comparison.csv").exists() else []
    seed = status["seeds"][0]
    rows = [r for r in rows_all if r["seed"] == seed]
    by = {r["run_id"]: r for r in rows}
    runs_dir = suite_dir / "runs" / f"seed_{seed}"
    data_dir = suite_dir / "data" / f"seed_{seed}"

    def metrics(rid):
        p = runs_dir / rid / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else None

    L: list[str] = []
    L.append(f"# {title}")
    L.append("")
    L.append(f"Generated {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by vehicle_localization {__version__} "
             f"from saved metrics in this directory. No numbers in this report were entered by hand.")
    L.append("")
    L.append("> All sensor data are **synthetic**. \"LiDAR\" means **simulated LiDAR-localizer positions** "
             "(truth plus noise), not real scan matching. Initialization is **ground-truth-assisted**: the prior "
             "is the true state at t=0 minus a sampled perturbation.")
    L.append("")

    # ---------------- status
    L.append("## 1. Status")
    L.append("")
    n_chk = len(status["checks"])
    n_ok = sum(c["passed"] for c in status["checks"])
    L.append(f"- Deterministic suite checks: **{n_ok}/{n_chk} passed**"
             + ("" if n_ok == n_chk else " - see the failed checks below"))
    if status.get("failed_runs"):
        L.append(f"- **Failed runs: {len(status['failed_runs'])}** (artifacts kept; see FAILED.txt in each run dir)")
        for f in status["failed_runs"]:
            L.append(f"  - seed {f['seed']} {f['run_id']}: {f['error']}")
    if verification and verification.get("pytest"):
        pt = verification["pytest"]
        L.append(f"- pytest: **{'PASSED' if pt['passed'] else 'FAILED'}** ({pt['summary']})")
    met = [t for t in status["targets"] if t["met"]]
    L.append(f"- Engineering targets: **{len(met)}/{len(status['targets'])} met**")
    L.append(f"- Runtime: {_f(status['wall_time_s'], 1)} s wall time, process peak memory "
             f"{_f(status.get('process_peak_memory_mb'), 0)} MB, output size {_f(status.get('output_size_mb'), 1)} MB")
    L.append(f"- Host: {host}")
    L.append("")
    L.append("### Engineering targets")
    L.append("")
    L.append(_table(["target", "seed", "value", "threshold", "met", "note"],
                    [[t["target"], t["seed"], _f(t["value"]), t["threshold"], "yes" if t["met"] else "**NO**",
                      t.get("note", "")] for t in status["targets"]]))
    L.append("")
    failed = [c for c in status["checks"] if not c["passed"]]
    if failed:
        L.append("### Failed checks")
        L.append("")
        for c in failed:
            L.append(f"- {c['check']}: {_f(c['value'])}")
        L.append("")

    # ---------------- what was generated
    L.append("## 2. What was generated")
    L.append("")
    gen_rows = []
    for name in ("zero_bias", "biased", "biased_outliers"):
        p = data_dir / name / "metadata.json"
        if p.exists():
            m = read_json(p)
            gt = load_ground_truth(data_dir / name)  # validated truth metadata
            tm = gt.metadata if gt is not None else {}
            o = tm.get("corruptions", {}).get("outliers")
            sensors = m.get("sensors", {})
            gen_rows.append([name, m.get("seed"), m.get("duration_s"), m.get("imu_rate_hz"),
                             sensors.get("gnss", {}).get("count"), sensors.get("lidar", {}).get("count"),
                             tm.get("truth_config", {}).get("bias_mode", "unknown"),
                             f"{len(o['t_ns'])} {o['stream'].upper()} outliers" if o else "none"])
    L.append(_table(["dataset", "seed", "duration [s]", "IMU [Hz]", "GNSS meas.", "LiDAR meas.", "true biases",
                     "corruption"], gen_rows))
    L.append("")
    L.append(f"Trajectory: {cfg['dataset']['trajectory']} (planar path in a 3-D estimator; not 6-DOF driving "
             "validation). Sensor rates and noise: see `config_resolved.yaml`.")
    L.append("")

    # ---------------- E0
    e0p = suite_dir / "e0_fixtures.json"
    if e0p.exists():
        e0 = json.loads(e0p.read_text())
        L.append("## 3. E0 - noise-free correctness fixtures (IMU-only, exact initialization)")
        L.append("")
        L.append(_table(["fixture (10 s)", "max position error [m]", "max velocity error [m/s]",
                         "max attitude error [deg]", "max quaternion norm error"],
                        [[k, _f(v["max_position_error_m"]), _f(v["max_velocity_error_mps"]),
                          _f(v["max_attitude_error_deg"]), _f(v["max_quaternion_norm_error"])]
                         for k, v in e0["fixtures"].items()]))
        L.append("")
        L.append("Step-size convergence on curved paths (final position error after 10 s):")
        L.append("")
        L.append(_table(["trajectory", "100 Hz [m]", "200 Hz [m]", "400 Hz [m]", "ratio 100/200", "ratio 200/400"],
                        [[k, *[_f(v["final_position_error_m_by_imu_rate_hz"][r]) for r in ("100", "200", "400")],
                          _f(v["ratio_100_to_200"], 2), _f(v["ratio_200_to_400"], 2)]
                         for k, v in e0["convergence"].items()]))
        L.append("")
        L.append("IMU-only drift decomposition on the default trajectory (one effect at a time):")
        L.append("")
        L.append(_table(["case", "error at 10 s [m]", f"error at {cfg['dataset']['duration_s']:g} s [m]"],
                        [[k, _f(v["position_error_at_10s_m"]), _f(v["final_position_error_m"])]
                         for k, v in e0["drift_decomposition"].items()]))
        dd = e0["drift_decomposition"]
        biggest = max(dd, key=lambda k: dd[k]["final_position_error_m"] if not k.startswith("all") else -1)
        L.append("")
        L.append(f"Observation (computed): among the single effects, **{biggest}** produces the largest IMU-only "
                 "drift. A small attitude error tilts the gravity vector into the horizontal axes, and a constant "
                 "acceleration error integrates twice, so position error grows roughly with time squared.")
        L.append("")

    # ---------------- experiment tables
    def exp_table(ids, extra=None):
        head = ["run", "mode", "pos RMSE [m]", "pos RMSE whole [m]", "horiz. RMSE [m]", "vel RMSE [m/s]",
                "att RMSE [deg]", "max pos err [m]", "GNSS acc/rej", "LiDAR acc/rej", "GNSS NIS", "LiDAR NIS"]
        if extra:
            head += [e[0] for e in extra]
        out = []
        for rid in ids:
            r = by.get(rid)
            if r is None:
                continue
            row = [rid, r["mode"], _f(r["pos_rmse_3d_m"]), _f(r["pos_rmse_3d_whole_m"]),
                   _f(r["pos_rmse_horizontal_m"]), _f(r["vel_rmse_3d_mps"]), _f(r["att_rmse_deg"]),
                   _f(r["pos_max_error_m"]), f"{int(r['gnss_accepted'])}/{int(r['gnss_rejected'])}",
                   f"{int(r['lidar_accepted'])}/{int(r['lidar_rejected'])}",
                   _f(r["gnss_nis_mean_pre_gate"], 2), _f(r["lidar_nis_mean_pre_gate"], 2)]
            if extra:
                row += [e[1](r) for e in extra]
            out.append(row)
        return _table(head, out)

    burn = cfg["report"]["burn_in_s"]
    L.append(f"## 4. Results (seed {seed})")
    L.append("")
    L.append(f"\"pos RMSE\" is computed after a {burn:g} s burn-in; \"whole\" includes the initialization period. "
             "NIS columns are mean **pre-gate** NIS. About 3 is expected when the filter's models and uncertainties "
             "are consistent with the data; a departure signals some inconsistency (covariance, process model, "
             "calibration, timing or outliers) without identifying which. Diagnostic only.")
    L.append("")
    L.append("### E1 - zero-bias noisy motion")
    L.append("")
    L.append(exp_table(["E1_imu_only", "E1_eskf9_gnss", "E1_eskf9_all"]))
    L.append("")
    if all(k in by for k in ("E1_imu_only", "E1_eskf9_gnss", "E1_eskf9_all")):
        a, b, c = by["E1_imu_only"], by["E1_eskf9_gnss"], by["E1_eskf9_all"]
        L.append(f"Observation (computed): GNSS updates change the whole-run position RMSE from "
                 f"{_f(a['pos_rmse_3d_whole_m'])} m (IMU only) to {_f(b['pos_rmse_3d_whole_m'])} m; adding LiDAR "
                 f"positions gives {_f(c['pos_rmse_3d_whole_m'])} m.")
        L.append("")
    L.append("### E2 - biased motion: bias estimation")
    L.append("")
    L.append(exp_table(["E2_imu_only", "E2_eskf9_all", "E2_eskf15_all"],
                       [("accel bias RMSE [m/s^2]", lambda r: _f(r["accel_bias_rmse_mps2"], 4)),
                        ("gyro bias RMSE [rad/s]", lambda r: _f(r["gyro_bias_rmse_radps"], 5)),
                        ("min pos 3-sigma coverage", lambda r: _f(r["pos_3sigma_coverage_min"], 3))]))
    L.append("")
    if "E2_eskf9_all" in by and "E2_eskf15_all" in by:
        a, b = by["E2_eskf9_all"], by["E2_eskf15_all"]
        word = "lower" if b["pos_rmse_3d_m"] < a["pos_rmse_3d_m"] else "higher"
        L.append(f"Observation (computed): with biased IMU data the 15-state filter's post-burn-in position RMSE "
                 f"is {_f(b['pos_rmse_3d_m'])} m vs {_f(a['pos_rmse_3d_m'])} m for the 9-state filter "
                 f"({word}, {_pct(b['pos_rmse_3d_m'], a['pos_rmse_3d_m'])}). Mean pre-gate LiDAR NIS: "
                 f"{_f(a['lidar_nis_mean_pre_gate'], 2)} (9-state) vs {_f(b['lidar_nis_mean_pre_gate'], 2)} "
                 f"(15-state); minimum per-axis 3-sigma position coverage {_f(a['pos_3sigma_coverage_min'])} vs "
                 f"{_f(b['pos_3sigma_coverage_min'])}. The 9-state filter assumes zero bias, so its covariance "
                 "does not account for the bias-induced error.")
        m15 = metrics("E2_eskf15_all")
        if m15:
            ab = m15["after_burn_in"]
            L.append("")
            L.append(f"Per-axis bias RMSE after burn-in (15-state): accelerometer "
                     f"{_f(ab['accel_bias_rmse_axis_mps2'], 4)} m/s^2, gyro {_f(ab['gyro_bias_rmse_axis_radps'], 5)} "
                     "rad/s. Position-only aiding leaves some directions weakly observable (for example yaw and "
                     "the vertical gyro bias are only excited by turning), so not every component is expected to "
                     "converge equally.")
        L.append("")

    L.append("### E3 / E4 / E5 - dropouts")
    L.append("")
    L.append(exp_table(["E3_eskf15_gnss_reference", "E3_eskf15_gnss_gnss_dropout", "E4_eskf15_all_gnss_dropout",
                        "E5_eskf15_all_total_dropout", "E5_eskf15_all_total_dropout_gating_off"]))
    L.append("")
    drows = []
    for rid in ("E3_eskf15_gnss_gnss_dropout", "E4_eskf15_all_gnss_dropout", "E5_eskf15_all_total_dropout",
                "E5_eskf15_all_total_dropout_gating_off"):
        m = metrics(rid)
        if not m or "dropouts" not in m:
            continue
        d = m["dropouts"][0]
        after = [k for k in d if k.startswith("after_")][0]
        drows.append([rid, _f(d["before_start"]["position_error_m"]), _f(d["before_start"]["position_cov_trace_m2"]),
                      _f(d["max_position_error_during_m"]), _f(d["end_of_outage"]["position_error_m"]),
                      _f(d["end_of_outage"]["position_cov_trace_m2"]), _f(d[after]["position_error_m"]),
                      d["updates_in_10s_after"], d["rejections_in_10s_after"]])
    L.append(_table(["run", "err before [m]", "trace P_pos before [m^2]", "max err during [m]", "err at end [m]",
                     "trace P_pos at end [m^2]", "err 5 s after [m]", "updates in 10 s after",
                     "rejected in 10 s after"], drows))
    L.append("")
    if all(k in by for k in ("E3_eskf15_gnss_gnss_dropout", "E4_eskf15_all_gnss_dropout",
                             "E5_eskf15_all_total_dropout")):
        m3, m4, m5 = (metrics(k)["dropouts"][0] for k in ("E3_eskf15_gnss_gnss_dropout",
                                                         "E4_eskf15_all_gnss_dropout", "E5_eskf15_all_total_dropout"))
        L.append(f"Observation (computed): losing GNSS when it is the only aid (E3) lets the error reach "
                 f"{_f(m3['max_position_error_during_m'])} m during the outage; with LiDAR positions still arriving "
                 f"(E4) the maximum is {_f(m4['max_position_error_during_m'])} m. With both streams lost (E5) the "
                 f"maximum is {_f(m5['max_position_error_during_m'])} m and the position covariance trace grows from "
                 f"{_f(m5['before_start']['position_cov_trace_m2'])} to "
                 f"{_f(m5['end_of_outage']['position_cov_trace_m2'])} m^2. After E5's outage, "
                 f"{m5['rejections_in_10s_after']} of {m5['updates_in_10s_after']} returning measurements in the "
                 f"first 10 s were rejected by the gate.")
        L.append("")

    L.append("### E6 - assumed GNSS covariance (same measurements)")
    L.append("")
    e6 = [r["run_id"] for r in rows if r["experiment"] == "E6"]
    L.append(exp_table(e6, [("min pos 3-sigma coverage", lambda r: _f(r["pos_3sigma_coverage_min"], 3))]))
    L.append("")
    for r in sorted((r for r in rows if r["experiment"] == "E6"), key=lambda r: (r["mode"], r["run_id"])):
        tot = r["gnss_accepted"] + r["gnss_rejected"]
        L.append(f"- {r['run_id']}: mean pre-gate GNSS NIS {_f(r['gnss_nis_mean_pre_gate'], 2)}, "
                 f"{int(r['gnss_rejected'])}/{int(tot)} GNSS measurements rejected, position RMSE "
                 f"{_f(r['pos_rmse_3d_m'])} m.")
    lock = [r for r in rows if r["experiment"] == "E6" and r["gnss_rejected"] > 0.5 * (r["gnss_accepted"] +
                                                                                       r["gnss_rejected"])]
    L.append("")
    L.append("Interpretation: a mean pre-gate NIS well above 3 means the innovations are larger than the filter's "
             "own predicted innovation covariance S; well below 3 means they are smaller. An elevated NIS is a "
             "symptom, not a diagnosis: it can come from an assumed measurement covariance that is too small, but "
             "equally from process noise that is too small, unmodelled biases or calibration errors, timing "
             "errors, outliers or a diverging state. In E6 the measurements and every other setting are identical "
             "across runs and only the assumed GNSS covariance scale changes, so the *differences* between these "
             "runs can be attributed to that scale; the NIS value alone would not prove it.")
    for r in lock:
        L.append(f"**Gate lock-out (computed):** in {r['run_id']} the gate rejected most GNSS measurements, "
                 "because the scaled-down assumed covariance made their innovations look improbable. "
                 + ("With no other aid it then coasted on the IMU and diverged - over-confidence plus gating can "
                    "remove the very corrections the filter needs." if r["mode"] == "eskf15_gnss" else
                    "LiDAR positions kept the estimate accurate despite the rejections."))
    L.append("")

    L.append("### E7 - LiDAR lever-arm calibration error")
    L.append("")
    L.append(exp_table(["E7_eskf15_all_lidar_arm_correct", "E7_eskf15_all_lidar_arm_wrong"],
                       [("mean err world x/y/z [m]", lambda r: _f([r["pos_mean_err_x_m"], r["pos_mean_err_y_m"],
                                                                   r["pos_mean_err_z_m"]], 3)),
                        ("mean err body x/y/z [m]", lambda r: _f([r["pos_mean_err_body_x_m"],
                                                                  r["pos_mean_err_body_y_m"],
                                                                  r["pos_mean_err_body_z_m"]], 3))]))
    L.append("")
    if "E7_eskf15_all_lidar_arm_correct" in by and "E7_eskf15_all_lidar_arm_wrong" in by:
        a, b = by["E7_eskf15_all_lidar_arm_correct"], by["E7_eskf15_all_lidar_arm_wrong"]
        d_arm = np.asarray(cfg["experiment"]["lidar_wrong_lever_arm_m"]) - np.asarray(
            cfg["estimator"]["lever_arms_m"]["lidar"])
        L.append(f"Observation (computed): the wrong body-fixed lever arm (error {_f(d_arm.tolist(), 2)} m, "
                 f"magnitude {_f(float(np.linalg.norm(d_arm)))} m) raises post-burn-in position RMSE from "
                 f"{_f(a['pos_rmse_3d_m'])} m to {_f(b['pos_rmse_3d_m'])} m, while LiDAR mean NIS only moves from "
                 f"{_f(a['lidar_nis_mean_pre_gate'], 2)} to {_f(b['lidar_nis_mean_pre_gate'], 2)}. The filter "
                 "explains the LiDAR data by shifting its position estimate by R * (lever-arm error), so the "
                 "innovations hardly reveal the fault; only the coarse GNSS disagrees. In world axes the signed "
                 "mean error largely cancels as the heading turns through the figure-eight, but in the body frame "
                 f"it is {_f([b['pos_mean_err_body_x_m'], b['pos_mean_err_body_y_m'], b['pos_mean_err_body_z_m']])} "
                 "m - close to the lever-arm error itself. Only a translational lever-arm error was tested; "
                 "rotational (boresight) calibration is not represented by a position-only measurement model.")
        L.append("")

    L.append("### E8 - GNSS outliers, gating off vs on")
    L.append("")
    L.append(exp_table([r["run_id"] for r in rows if r["experiment"] == "E8"],
                       [("outlier recall", lambda r: _f(r["outlier_recall"], 3)),
                        ("false rejection rate", lambda r: _f(r["false_rejection_rate"], 4))]))
    L.append("")
    for mode in ("eskf15_gnss", "eskf15_all"):
        off, on = by.get(f"E8_{mode}_gating_off"), by.get(f"E8_{mode}_gating_on")
        if off and on:
            word = "reduced" if on["pos_rmse_3d_m"] < off["pos_rmse_3d_m"] else "did not reduce"
            L.append(f"- {mode}: gating {word} post-burn-in position RMSE ({_f(off['pos_rmse_3d_m'])} m off -> "
                     f"{_f(on['pos_rmse_3d_m'])} m on); recall {_f(on['outlier_recall'])}, clean-measurement false "
                     f"rejection rate {_f(on['false_rejection_rate'], 4)}.")
    L.append("")

    L.append("### E9 - degraded GNSS (viaduct / urban canyon stand-in)")
    L.append("")
    g = cfg["experiment"]["gnss_degradation"]
    L.append(f"Inside [{g['interval_s'][0]:g}, {g['interval_s'][1]:g}) s the GNSS noise standard deviation is "
             f"multiplied by {g['noise_scale']:g} and a smooth multipath-like bias peaking at {g['bias_m']} m is "
             "added. The receiver keeps reporting its nominal covariance, so the filter is not told. The "
             "'switch' policy uses one source at a time (GNSS while its innovations are consistent, LiDAR "
             "otherwise) - an innovation-test version of the sensor switching in Wang et al. (2024). 'fuse' "
             "uses both sources, each weighted by its covariance, behind the chi-square gate.")
    L.append("")
    L.append(exp_table([r["run_id"] for r in rows if r["experiment"] == "E9"],
                       [("max err in degraded section [m]", lambda r: _f(r["max_pos_err_in_interval_m"])),
                        ("RPE mean/max [m]", lambda r: f"{_f(r['rpe_mean_m'])}/{_f(r['rpe_max_m'])}"),
                        ("time on LiDAR [s]", lambda r: _f(r["time_on_lidar_s"], 1))]))
    L.append("")
    for rid in ("E9_eskf15_gnss_degraded", "E9_eskf15_all_fuse_degraded", "E9_eskf15_all_switch_degraded"):
        m = metrics(rid)
        if not m or not m.get("analysis_intervals"):
            continue
        d = m["analysis_intervals"][0]
        after = [k for k in d if k.startswith("after_")][0]
        L.append(f"- {rid}: max error {_f(d['max_position_error_during_m'])} m inside the section, "
                 f"{_f(d['end_of_outage']['position_error_m'])} m at its end, {_f(d[after]['position_error_m'])} m "
                 f"{after.split('_')[1]} after; {d['rejections_in_10s_after']} of {d['updates_in_10s_after']} updates "
                 "rejected in the 10 s after it.")
        if rid == "E9_eskf15_gnss_degraded" and d["rejections_in_10s_after"] > 0.5 * max(d["updates_in_10s_after"], 1):
            L.append("  **Gate lock-out (computed):** the gate rejected most degraded GNSS fixes, the filter coasted "
                     "on the IMU, and after the section ended the now-correct GNSS disagreed so strongly with the "
                     "drifted estimate that it kept being rejected. Gating protects against short outliers but "
                     "can turn a long degradation into a much larger error when there is no second source.")
    L.append("")

    L.append("### E10 - fuse vs switch")
    L.append("")
    L.append(exp_table(["E2_eskf15_all", "E10_eskf15_all_switch", "E4_eskf15_all_gnss_dropout",
                        "E10_eskf15_all_switch_gnss_dropout", "E9_eskf15_all_fuse_degraded",
                        "E9_eskf15_all_switch_degraded"],
                       [("policy", lambda r: r.get("fusion_policy") or "fuse"),
                        ("RPE mean/max [m]", lambda r: f"{_f(r['rpe_mean_m'])}/{_f(r['rpe_max_m'])}"),
                        ("time on LiDAR [s]", lambda r: _f(r["time_on_lidar_s"], 1)),
                        ("switches", lambda r: _f(r["policy_switches"]))]))
    L.append("")
    pairs = [("nominal data", "E2_eskf15_all", "E10_eskf15_all_switch"),
             ("GNSS dropout", "E4_eskf15_all_gnss_dropout", "E10_eskf15_all_switch_gnss_dropout"),
             ("degraded GNSS", "E9_eskf15_all_fuse_degraded", "E9_eskf15_all_switch_degraded")]
    compared, switch_better = 0, 0
    for label, f_id, s_id in pairs:
        f, sw_ = by.get(f_id), by.get(s_id)
        if f and sw_ and f["pos_rmse_3d_m"] is not None and sw_["pos_rmse_3d_m"] is not None:
            ratio = sw_["pos_rmse_3d_m"] / f["pos_rmse_3d_m"] if f["pos_rmse_3d_m"] else float("nan")
            compared += 1
            switch_better += sw_["pos_rmse_3d_m"] < f["pos_rmse_3d_m"]
            L.append(f"- {label}: fused {_f(f['pos_rmse_3d_m'])} m vs switching {_f(sw_['pos_rmse_3d_m'])} m "
                     f"position RMSE ({_f(ratio, 1)}x).")
    L.append("")
    if compared:
        L.append(f"Observation (computed): switching had the lower RMSE in {switch_better} of {compared} "
                 "comparisons. Switching discards the source it is not currently using, while the Kalman update "
                 "already weights both sources by their covariances and the gate removes inconsistent fixes. "
                 "These sensors are synthetic with independent errors; real localizers with correlated or "
                 "drifting errors could change the balance, so this is a result for this setup, not a general law.")
    L.append("")

    m_ref = metrics("E2_eskf15_all")
    if m_ref and "external_position_reference_errors" in m_ref:
        L.append("### Raw external position measurements (reference)")
        L.append("")
        L.append("Each measurement compared with the truth of **its own reference point** at its own timestamp "
                 "(not a causal estimator):")
        L.append("")
        L.append(_table(["stream", "count", "3-D RMSE [m]", "horizontal RMSE [m]"],
                        [[k, v["count"], _f(v["rmse_3d_m"]), _f(v["rmse_horizontal_m"])]
                         for k, v in m_ref["external_position_reference_errors"].items()]))
        L.append("")

    if verification and verification.get("per_seed"):
        L.append("## 5. Multi-seed verification")
        L.append("")
        L.append(_table(["seed", "eskf15_all pos RMSE [m]", "eskf9_all pos RMSE [m]", "imu_only whole RMSE [m]",
                         "E8 recall", "E8 false rejection"],
                        [[s["seed"], _f(s.get("eskf15_all")), _f(s.get("eskf9_all")), _f(s.get("imu_only")),
                          _f(s.get("recall")), _f(s.get("false_rejection"), 4)] for s in verification["per_seed"]]))
        L.append("")

    L.append("## Figures")
    L.append("")
    for name, caption in FIGURES:
        if (suite_dir / "figures" / name).exists():
            L.append(f"- [{name}](figures/{name}) - {caption}")
    L.append("")
    if (suite_dir / "figures" / "fig01_trajectory_xy.png").exists():
        L.append("![trajectory](figures/fig01_trajectory_xy.png)")
        L.append("")
    L.append("## Limitations")
    L.append("")
    L.append("- Synthetic data only; conclusions need validation on recorded data. Not tested on a real vehicle.")
    L.append("- The two position streams are simulated independently; real localizers can have correlated errors.")
    L.append("- Missing effects: multipath, real point-cloud registration, time-correlated errors, imperfect "
             "synchronization, temperature effects, complex 3-D motion.")
    L.append("- Coverage percentages and mean NIS from one trajectory are diagnostics, not proof of consistency.")
    L.append("- Offline processing; delayed or out-of-order measurements are not supported.")
    L.append("- Timings were measured on the host stated above. They are not laptop benchmarks unless the host "
             "line says so.")
    path = suite_dir / "report.md"
    path.write_text("\n".join(_drop_empty_experiment_sections(L)) + "\n", encoding="utf-8")
    return path


def _drop_empty_experiment_sections(lines: list[str]) -> list[str]:
    """Remove '### E..' sections whose tables have no data rows (e.g. runs not in this suite)."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        if re.match(r"### E\d", lines[i]):
            j = i + 1
            while j < len(lines) and not lines[j].startswith(("### ", "## ")):
                j += 1
            if any(re.search(r"(^|\n)\| E\d", ln) for ln in lines[i:j]):
                out.extend(lines[i:j])
            i = j
        else:
            out.append(lines[i])
            i += 1
    return out
