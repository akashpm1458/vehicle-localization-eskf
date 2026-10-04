"""Reproducible experiment definitions (E0-E8), suite and verification runs.

Fair-comparison rules enforced here:
* Each base dataset is generated once per seed and saved before any estimator runs.
* Dropout, outlier and calibration variants are derived from that same base data, so
  every compared run sees identical sensor noise.
* The filter receives only the ``estimator`` config (plus per-experiment assumptions such
  as a wrong lever arm or a covariance multiplier). Truth is loaded only for evaluation,
  after the run.
* Runs execute one after another; only small summaries are kept in memory.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import __version__
from .config import deep_merge, dump_config
from .dataset import load_dataset, load_ground_truth
from .eskf import EstimatorConfig
from .evaluation import error_series, evaluate_run, external_reference_errors
from .runner import Dropout, load_run, run_estimator, save_run
from .sensors import derive_outlier_dataset, generate_dataset
from .sysinfo import dir_size_mb, package_versions, peak_memory_mb


@dataclass
class RunSpec:
    experiment: str
    run_id: str
    dataset: str  # key into the suite's datasets
    mode: str
    description: str
    estimator_overrides: dict = field(default_factory=dict)
    dropouts: list[tuple[str, float, float]] = field(default_factory=list)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


# ----------------------------------------------------------------------------- one run


def execute_run(dataset_dir: Path, cfg: dict, mode: str, out_dir: Path, *, estimator_overrides: dict | None = None,
                dropouts: list[tuple[str, float, float]] | None = None, meta: dict | None = None) -> dict:
    """Run, save and evaluate one estimator configuration. Returns a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    est_dict = deep_merge(cfg["estimator"], estimator_overrides or {})
    est_dict["mode"] = mode
    est = EstimatorConfig.from_dict(est_dict, mode)
    ds = load_dataset(dataset_dir)  # estimator inputs only; no ground truth
    drops = [Dropout.from_seconds(s, a, b) for s, a, b in (dropouts or [])]
    result = run_estimator(ds, est, drops)
    save_run(result, out_dir)

    truth = load_ground_truth(dataset_dir)  # evaluator access, after the run
    labels = None
    if truth is not None:
        o = truth.metadata.get("corruptions", {}).get("outliers")
        if o:
            labels = {"stream": o["stream"], "t_ns": o["t_ns"]}
    intervals = sorted({(a, b) for _, a, b in (dropouts or [])})
    metrics = evaluate_run(result, truth, result.n_states, cfg["report"]["burn_in_s"], dropouts=intervals,
                           outlier_labels=labels)
    if truth is not None:
        metrics["external_position_reference_errors"] = external_reference_errors(ds.streams, truth)
    metrics["counts"] = result.counts
    metrics["numerics"] = {
        "covariance_checks": result.info["covariance_checks"],
        "max_logged_quaternion_norm_error": result.info["max_logged_quaternion_norm_error"],
    }
    metrics["runtime"] = {
        "wall_time_s": result.info["wall_time_s"],
        "process_peak_memory_mb": peak_memory_mb(),
    }
    write_json(out_dir / "metrics.json", metrics)
    resolved = copy.deepcopy(cfg)
    resolved["estimator"] = est_dict
    resolved["experiment"] = deep_merge(cfg["experiment"], {"applied_dropouts": [list(d) for d in (dropouts or [])]})
    dump_config(resolved, out_dir / "config_resolved.yaml")
    write_json(out_dir / "run_metadata.json", {
        "mode": mode,
        "dataset": str(dataset_dir),
        "dataset_name": ds.metadata.get("name"),
        "seed": ds.metadata.get("seed"),
        "initialization": ds.prior.provenance,
        "package_version": __version__,
        "environment": package_versions(),
        "logged_rows": int(result.t_ns.size),
        "full_covariance_rows": int(result.cov_t_ns.size),
        "warnings": ds.warnings,
        **(meta or {}),
    })
    return {"metrics": metrics, "out_dir": out_dir, "n_states": result.n_states}


# ----------------------------------------------------------------------------- E0 fixtures


def run_e0(cfg: dict, work: Path) -> dict:
    """Noise-free fixtures with exact initialization (IMU-only propagation)."""
    out: dict = {"fixtures": {}, "convergence": {}, "drift_decomposition": {}}
    est = EstimatorConfig.from_dict(cfg["estimator"], "imu_only")

    def make(name, trajectory, duration, rate=100, truth=None, init="exact_truth"):
        c = deep_merge(cfg, {
            "dataset": {"trajectory": trajectory, "duration_s": duration, "imu_rate_hz": rate, "name": name},
            "truth": {"noise_free": True, "bias_mode": "zero", **(truth or {})},
            "initialization": {"source": init},
        })
        d = work / name
        generate_dataset(c, d, overwrite=True)
        return d

    for trajectory in ("stationary", "stationary_tilted", "constant_velocity", "constant_yaw_rate", "smooth_rpy"):
        d = make(f"e0_{trajectory}", trajectory, 10.0)
        r = run_estimator(load_dataset(d), est)
        es = error_series(r, load_ground_truth(d))
        out["fixtures"][trajectory] = {
            "max_position_error_m": float(np.linalg.norm(es["dp"], axis=1).max()),
            "max_velocity_error_mps": float(np.linalg.norm(es["dv"], axis=1).max()),
            "max_attitude_error_deg": float(np.rad2deg(np.linalg.norm(es["dth"], axis=1)).max()),
            "max_quaternion_norm_error": r.info["max_logged_quaternion_norm_error"],
        }
    for trajectory in ("constant_yaw_rate", "figure_eight", "smooth_rpy"):
        errs = {}
        for rate in (100, 200, 400):
            d = make(f"e0_conv_{trajectory}_{rate}", trajectory, 10.0, rate)
            r = run_estimator(load_dataset(d), est)
            errs[rate] = float(np.linalg.norm(error_series(r, load_ground_truth(d))["dp"][-1]))
        out["convergence"][trajectory] = {
            "final_position_error_m_by_imu_rate_hz": errs,
            "ratio_100_to_200": errs[100] / errs[200] if errs[200] > 0 else float("inf"),
            "ratio_200_to_400": errs[200] / errs[400] if errs[400] > 0 else float("inf"),
        }
    dur = cfg["dataset"]["duration_s"]
    for label, truth, init in (
        ("exact init, no noise, no bias", {"noise_free": True, "bias_mode": "zero"}, "exact_truth"),
        ("exact init, IMU white noise only", {"noise_free": False, "bias_mode": "zero"}, "exact_truth"),
        ("exact init, IMU biases only", {"noise_free": True, "bias_mode": "biased"}, "exact_truth"),
        ("perturbed initial prior only", {"noise_free": True, "bias_mode": "zero"}, "generated_perturbed_truth"),
        ("all effects (default biased data)", {"noise_free": False, "bias_mode": "biased"}, "generated_perturbed_truth"),
    ):
        d = make("e0_drift", cfg["dataset"]["trajectory"], dur, truth=truth, init=init)
        r = run_estimator(load_dataset(d), est)
        e = np.linalg.norm(error_series(r, load_ground_truth(d))["dp"], axis=1)
        out["drift_decomposition"][label] = {"final_position_error_m": float(e[-1]),
                                             "position_error_at_10s_m": float(e[min(1000, e.size - 1)])}
        shutil.rmtree(d)
    checks = []
    for name in ("stationary", "stationary_tilted", "constant_velocity"):
        v = out["fixtures"][name]["max_position_error_m"]
        checks.append({"check": f"E0 {name}: 10 s position drift < 1e-6 m", "value": v, "passed": v < 1e-6})
    for name, f in out["fixtures"].items():
        v = f["max_quaternion_norm_error"]
        checks.append({"check": f"E0 {name}: quaternion norm error < 1e-10", "value": v, "passed": v < 1e-10})
    for name, c in out["convergence"].items():
        ok = c["ratio_100_to_200"] > 1.8 and c["ratio_200_to_400"] > 1.8
        checks.append({"check": f"E0 {name}: error shrinks when the step is halved",
                       "value": [c["ratio_100_to_200"], c["ratio_200_to_400"]], "passed": ok})
    out["checks"] = checks
    return out


# ----------------------------------------------------------------------------- suite definition


def suite_specs(cfg: dict) -> list[RunSpec]:
    exp = cfg["experiment"]
    d0, d1 = exp["dropout_interval_s"]
    wrong_arm = exp["lidar_wrong_lever_arm_m"]
    true_arm = cfg["estimator"]["lever_arms_m"]["lidar"]
    specs = [
        RunSpec("E1", "E1_imu_only", "zero_bias", "imu_only", "Zero-bias noisy motion, IMU only"),
        RunSpec("E1", "E1_eskf9_gnss", "zero_bias", "eskf9_gnss", "Zero-bias noisy motion, 9-state + GNSS"),
        RunSpec("E1", "E1_eskf9_all", "zero_bias", "eskf9_all", "Zero-bias noisy motion, 9-state + GNSS + LiDAR"),
        RunSpec("E2", "E2_imu_only", "biased", "imu_only", "Biased motion, IMU only"),
        RunSpec("E2", "E2_eskf9_all", "biased", "eskf9_all", "Biased motion, 9-state (assumes zero bias)"),
        RunSpec("E2", "E2_eskf15_all", "biased", "eskf15_all", "Biased motion, 15-state (estimates biases)"),
        RunSpec("E3", "E3_eskf15_gnss_reference", "biased", "eskf15_gnss", "15-state GNSS-only, no dropout (reference)"),
        RunSpec("E3", "E3_eskf15_gnss_gnss_dropout", "biased", "eskf15_gnss",
                f"15-state GNSS-only, GNSS absent [{d0}, {d1}) s", dropouts=[("gnss", d0, d1)]),
        RunSpec("E4", "E4_eskf15_all_gnss_dropout", "biased", "eskf15_all",
                f"15-state GNSS+LiDAR, GNSS absent [{d0}, {d1}) s (LiDAR continues)", dropouts=[("gnss", d0, d1)]),
        RunSpec("E5", "E5_eskf15_all_total_dropout", "biased", "eskf15_all",
                f"15-state, GNSS and LiDAR absent [{d0}, {d1}) s", dropouts=[("gnss", d0, d1), ("lidar", d0, d1)]),
        RunSpec("E5", "E5_eskf15_all_total_dropout_gating_off", "biased", "eskf15_all",
                f"Same as E5, gating disabled (diagnostic)", {"gating": {"enabled": False}},
                [("gnss", d0, d1), ("lidar", d0, d1)]),
    ]
    for mode in ("eskf15_gnss", "eskf15_all"):
        for s in exp["gnss_covariance_scales"]:
            specs.append(RunSpec("E6", f"E6_{mode}_gnss_cov_x{s:g}", "biased", mode,
                                 f"{mode}, assumed GNSS covariance x{s:g} (same measurements)",
                                 {"covariance_scale": {"gnss": float(s)}}))
    specs += [
        RunSpec("E7", "E7_eskf15_all_lidar_arm_correct", "biased", "eskf15_all",
                f"Assumed LiDAR lever arm {true_arm} (correct)"),
        RunSpec("E7", "E7_eskf15_all_lidar_arm_wrong", "biased", "eskf15_all",
                f"Assumed LiDAR lever arm {wrong_arm} (true {true_arm})", {"lever_arms_m": {"lidar": wrong_arm}}),
    ]
    for mode in ("eskf15_gnss", "eskf15_all"):
        for gate in (False, True):
            specs.append(RunSpec("E8", f"E8_{mode}_gating_{'on' if gate else 'off'}", "biased_outliers", mode,
                                 f"{mode} with GNSS outliers, gating {'on' if gate else 'off'}",
                                 {"gating": {"enabled": gate}}))
    return specs


def prepare_datasets(cfg: dict, data_dir: Path, seed: int) -> dict[str, Path]:
    """Generate the base datasets for one seed (each saved once) and derived variants."""
    c = deep_merge(cfg, {"dataset": {"seed": seed}})
    paths = {"zero_bias": data_dir / "zero_bias", "biased": data_dir / "biased",
             "biased_outliers": data_dir / "biased_outliers"}
    generate_dataset(deep_merge(c, {"truth": {"bias_mode": "zero"}, "dataset": {"name": "zero_bias"}}),
                     paths["zero_bias"], overwrite=True)
    generate_dataset(deep_merge(c, {"truth": {"bias_mode": "biased"}, "dataset": {"name": "biased"}}),
                     paths["biased"], overwrite=True)
    derive_outlier_dataset(paths["biased"], paths["biased_outliers"], c["experiment"]["outliers"], overwrite=True)
    return paths


SUMMARY_FIELDS = [
    ("pos_rmse_3d_whole_m", ("whole_run", "position_rmse_3d_m")),
    ("pos_rmse_3d_m", ("after_burn_in", "position_rmse_3d_m")),
    ("pos_rmse_horizontal_m", ("after_burn_in", "position_rmse_horizontal_m")),
    ("vel_rmse_3d_mps", ("after_burn_in", "velocity_rmse_3d_mps")),
    ("att_rmse_deg", ("after_burn_in", "attitude_rmse_deg")),
    ("att_max_deg", ("after_burn_in", "attitude_max_error_deg")),
    ("pos_final_error_m", ("whole_run", "position_final_error_m")),
    ("pos_max_error_m", ("whole_run", "position_max_error_m")),
    ("accel_bias_rmse_mps2", ("after_burn_in", "accel_bias_rmse_mps2")),
    ("gyro_bias_rmse_radps", ("after_burn_in", "gyro_bias_rmse_radps")),
]


def summarize(spec: RunSpec, metrics: dict, seed: int) -> dict:
    row = {"experiment": spec.experiment, "run_id": spec.run_id, "seed": seed, "mode": spec.mode,
           "dataset": spec.dataset, "description": spec.description}
    for name, (block, key) in SUMMARY_FIELDS:
        row[name] = metrics[block][key]
    me = metrics["after_burn_in"]["position_mean_error_axis_m"]
    row["pos_mean_err_x_m"], row["pos_mean_err_y_m"], row["pos_mean_err_z_m"] = me
    cov = metrics["after_burn_in"]["coverage_3sigma"]["position_axis"]
    row["pos_3sigma_coverage_min"] = min(cov) if cov else None
    for s in ("gnss", "lidar"):
        st = metrics["innovations"].get(s)
        row[f"{s}_accepted"] = st["accepted"] if st else 0
        row[f"{s}_rejected"] = st["rejected"] if st else 0
        row[f"{s}_nis_mean_pre_gate"] = st["nis_pre_gate_mean"] if st else None
    od = metrics.get("outlier_detection")
    row["outlier_recall"] = od["recall"] if od else None
    row["false_rejection_rate"] = od["false_rejection_rate"] if od else None
    row["wall_time_s"] = metrics["runtime"]["wall_time_s"]
    return row


def write_comparison(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r[c]
                if v is None:
                    vals.append("")
                elif isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v).replace(",", ";"))
            fh.write(",".join(vals) + "\n")


def read_comparison(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    cols = lines[0].split(",")
    out = []
    for line in lines[1:]:
        r = {}
        for c, v in zip(cols, line.split(",")):
            try:
                r[c] = float(v) if v != "" else None
            except ValueError:
                r[c] = v
        out.append(r)
    return out


# ----------------------------------------------------------------------------- targets


def evaluate_targets(rows: list[dict], seed: int) -> list[dict]:
    by = {r["run_id"]: r for r in rows if r["seed"] == seed}
    t = []

    def add(name, value, threshold, met, note=""):
        t.append({"target": name, "seed": seed, "value": value, "threshold": threshold, "met": bool(met), "note": note})

    if "E2_eskf15_all" in by:
        v = by["E2_eskf15_all"]["pos_rmse_3d_m"]
        add("eskf15_all post-initialization 3-D position RMSE (biased case)", v, "< 1.0 m", v < 1.0)
    if "E2_eskf15_all" in by and "E2_imu_only" in by:
        a, b = by["E2_eskf15_all"]["pos_rmse_3d_whole_m"], by["E2_imu_only"]["pos_rmse_3d_whole_m"]
        add("full aided estimator beats IMU-only over the complete run (position RMSE)", [a, b], "eskf15_all < imu_only",
            a < b)
    if "E8_eskf15_all_gating_on" in by:
        v = by["E8_eskf15_all_gating_on"]["outlier_recall"]
        add("injected GNSS outlier recall with gating (eskf15_all)", v, ">= 0.90", v is not None and v >= 0.9,
            f"false rejection rate of clean GNSS: {by['E8_eskf15_all_gating_on']['false_rejection_rate']:.4f}")
    if "E2_eskf15_all" in by and "E2_eskf9_all" in by:
        a, b = by["E2_eskf15_all"]["pos_rmse_3d_m"], by["E2_eskf9_all"]["pos_rmse_3d_m"]
        add("bias estimation improves position RMSE in E2 (investigate if not)", [a, b], "eskf15_all < eskf9_all",
            a < b, "comparison target, not a guaranteed result")
    return t


# ----------------------------------------------------------------------------- suite


VERIFY_RUN_IDS = ("E2_imu_only", "E2_eskf9_all", "E2_eskf15_all", "E8_eskf15_all_gating_off",
                  "E8_eskf15_all_gating_on")


def run_suite(cfg: dict, output_dir: Path, seeds: list[int], include_e0: bool = True,
              make_figures: bool = True, quiet: bool = False, run_ids: tuple[str, ...] | None = None) -> dict:
    """Run the experiment matrix (or the subset ``run_ids``) for each seed sequentially."""
    t_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, output_dir / "config_resolved.yaml")
    rows: list[dict] = []
    failures: list[dict] = []
    status: dict = {"seeds": seeds, "checks": [], "targets": [], "failed_runs": failures}
    if include_e0:
        if not quiet:
            print("E0: noise-free fixtures ...", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            e0 = run_e0(cfg, Path(tmp))
        write_json(output_dir / "e0_fixtures.json", e0)
        status["checks"] += e0["checks"]
    specs = [s for s in suite_specs(cfg) if run_ids is None or s.run_id in run_ids]
    for seed in seeds:
        data_dir = output_dir / "data" / f"seed_{seed}"
        paths = prepare_datasets(cfg, data_dir, seed)
        for spec in specs:
            run_dir = output_dir / "runs" / f"seed_{seed}" / spec.run_id
            if not quiet:
                print(f"seed {seed}: {spec.run_id} ...", flush=True)
            try:
                res = execute_run(paths[spec.dataset], cfg, spec.mode, run_dir,
                                  estimator_overrides=spec.estimator_overrides, dropouts=spec.dropouts,
                                  meta={"experiment": spec.experiment, "run_id": spec.run_id,
                                        "description": spec.description})
            except Exception as exc:  # keep going; failed runs are reported and make the suite fail
                failures.append({"seed": seed, "run_id": spec.run_id, "error": f"{type(exc).__name__}: {exc}"})
                (run_dir / "FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
                continue
            rows.append(summarize(spec, res["metrics"], seed))
            nm = res["metrics"]["numerics"]
            status["checks"].append({
                "check": f"seed {seed} {spec.run_id}: covariance finite/symmetric/PSD, quaternion norm < 1e-10",
                "value": {"roundoff_negative_eigs": nm["covariance_checks"]["roundoff_negative_eigs"],
                          "q_norm_err": nm["max_logged_quaternion_norm_error"]},
                "passed": nm["max_logged_quaternion_norm_error"] < 1e-10,
            })
        status["targets"] += evaluate_targets(rows, seed)
    write_comparison(rows, output_dir / "comparison.csv")
    if make_figures and rows:
        from .report import make_suite_figures

        make_suite_figures(output_dir, cfg, seeds[0])
    status["all_checks_passed"] = all(c["passed"] for c in status["checks"]) and not failures
    status["all_targets_met"] = all(t["met"] for t in status["targets"])
    status["wall_time_s"] = time.perf_counter() - t_start
    status["process_peak_memory_mb"] = peak_memory_mb()
    status["output_size_mb"] = dir_size_mb(output_dir)
    write_json(output_dir / "suite_status.json", status)
    return status


def run_pytest(output_dir: Path) -> dict:
    """Run the deterministic test suite and save its output."""
    root = Path(__file__).resolve().parents[2]
    tests = root / "tests"
    if not tests.is_dir():
        return {"ran": False, "passed": False, "summary": "tests/ directory not found next to the package"}
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", str(tests)], capture_output=True, text=True,
                          cwd=root)
    (output_dir / "pytest_output.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    last = (proc.stdout.strip().splitlines() or [""])[-1]
    return {"ran": True, "passed": proc.returncode == 0, "returncode": proc.returncode, "summary": last}


def run_monte_carlo(cfg: dict, output_dir: Path, runs: int, first_seed: int = 1000, mode: str = "eskf15_all") -> dict:
    """Optional statistical study: many seeds of the nominal biased case, sequentially."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(runs):
        seed = first_seed + i
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "biased"
            generate_dataset(deep_merge(cfg, {"dataset": {"seed": seed}}), d)
            res = execute_run(d, cfg, mode, Path(tmp) / "run")
            m = res["metrics"]
            rows.append({
                "seed": seed,
                "pos_rmse_3d_m": m["after_burn_in"]["position_rmse_3d_m"],
                "att_rmse_deg": m["after_burn_in"]["attitude_rmse_deg"],
                "gnss_nis_mean": m["innovations"]["gnss"]["nis_pre_gate_mean"],
                "lidar_nis_mean": m["innovations"].get("lidar", {}).get("nis_pre_gate_mean"),
                "pos_3sigma_coverage_min": min(m["after_burn_in"]["coverage_3sigma"]["position_axis"]),
            })
        print(f"monte-carlo {i + 1}/{runs}: seed {seed} pos RMSE {rows[-1]['pos_rmse_3d_m']:.3f} m", flush=True)
    write_comparison(rows, output_dir / "monte_carlo.csv")
    arr = np.array([r["pos_rmse_3d_m"] for r in rows])
    summary = {"mode": mode, "runs": runs, "seeds": [r["seed"] for r in rows],
               "pos_rmse_3d_m": {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if runs > 1 else 0.0,
                                 "max": float(arr.max())},
               "mean_gnss_nis": float(np.mean([r["gnss_nis_mean"] for r in rows])),
               "mean_lidar_nis": float(np.mean([r["lidar_nis_mean"] for r in rows if r["lidar_nis_mean"] is not None])),
               "fraction_below_1m": float(np.mean(arr < 1.0))}
    write_json(output_dir / "monte_carlo_summary.json", summary)
    return summary


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
