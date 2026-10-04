"""Command-line interface: ``python -m vehicle_localization <command>``.

Exit codes: 0 = success; 1 = failure (invalid input, failed deterministic check or
incomplete run); 2 = runs completed but one or more engineering targets were not met.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

from . import __version__


def resolve_output(path: str | Path, overwrite: bool) -> Path:
    """Never overwrite silently: reuse ``path`` if empty/overwrite, else add a run ID."""
    p = Path(path)
    if p.exists() and any(p.iterdir()) and not overwrite:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        new = p.with_name(f"{p.name}_{stamp}")
        print(f"note: {p} already exists; writing to new run directory {new} (use --overwrite to replace)")
        return new
    return p


def cmd_doctor(args) -> int:
    from .doctor import collect, report

    info = collect(".")
    print(report(info))
    missing = [k for k, v in info["dependencies"].items() if v is None and k != "pytest"]
    return 0 if info["python_ok"] and not missing else 1


def cmd_generate(args) -> int:
    from .config import load_config
    from .dataset import load_ground_truth, load_dataset
    from .plotting import plot_ground_truth
    from .sensors import generate_dataset

    overrides = {"dataset": {"seed": args.seed}} if args.seed is not None else None
    cfg = load_config(args.config, overrides)
    out = resolve_output(args.output or cfg["dataset"]["output"], args.overwrite)
    manifest = generate_dataset(cfg, out, overwrite=args.overwrite)
    ds = load_dataset(out)
    plot_ground_truth(load_ground_truth(out), ds.streams, out / "ground_truth_plot.png", cfg["report"]["dpi"])
    print(f"Generated dataset in {manifest.path}: {manifest.n_imu_rows} IMU rows, "
          f"measurements {manifest.counts}, seed {manifest.seed}, {manifest.duration_s} s")
    return 0


def cmd_validate(args) -> int:
    from .dataset import validate_dataset

    result = validate_dataset(args.data, normalize_imu=args.normalize_imu)
    print(result.report())
    return 0 if result.ok else 1


def _host() -> str:
    from .doctor import collect, host_description

    return host_description(collect("."))


def cmd_run(args) -> int:
    from .config import load_config
    from .dataset import load_ground_truth
    from .evaluation import error_series
    from .experiments import execute_run
    from .plotting import plot_axis_errors_with_bounds, plot_nis, plot_trajectory_xy
    from .runner import load_run

    cfg = load_config(args.config)
    overrides: dict = {}
    if args.gating is not None:
        overrides["gating"] = {"enabled": args.gating == "on"}
    if args.discretization:
        overrides["discretization"] = args.discretization
    out = resolve_output(args.output, args.overwrite)
    dropouts = [(s, float(a), float(b)) for s, a, b in (d.split(":") for d in args.dropout or [])]
    res = execute_run(Path(args.data), cfg, args.mode, out, estimator_overrides=overrides, dropouts=dropouts,
                      meta={"command": "run", "host": _host()})
    m = res["metrics"]
    run = load_run(out)
    truth = load_ground_truth(args.data)
    dpi = cfg["report"]["dpi"]
    plot_nis(run.innovations, out / "nis.png", f"{args.mode}: pre-gate NIS", dpi=dpi)
    if truth is not None:
        es = error_series(run, truth)
        plot_trajectory_xy(truth, {args.mode: run}, out / "trajectory_xy.png", f"{args.mode}", dpi=dpi)
        plot_axis_errors_with_bounds(es, out / "position_error_bounds.png", f"{args.mode}: position error", dpi=dpi)
        a = m["after_burn_in"]
        print(f"{args.mode}: position RMSE {a['position_rmse_3d_m']:.3f} m (after {m['burn_in_s']:g} s burn-in), "
              f"whole run {m['whole_run']['position_rmse_3d_m']:.3f} m; velocity RMSE {a['velocity_rmse_3d_mps']:.3f} "
              f"m/s; attitude RMSE {a['attitude_rmse_deg']:.3f} deg")
    else:
        print(f"{args.mode}: no ground truth available; accuracy metrics disabled, innovations logged")
    for s, st in m["innovations"].items():
        print(f"  {s}: {st['accepted']} accepted, {st['rejected']} rejected, mean pre-gate NIS {st['nis_pre_gate_mean']:.2f}")
    print(f"  wall time {m['runtime']['wall_time_s']:.2f} s; outputs in {out}")
    return 0


def _exit_code(status: dict) -> int:
    if not status.get("all_checks_passed", False) or status.get("failed_runs"):
        return 1
    if not status.get("all_targets_met", False):
        return 2
    return 0


def _print_status(status: dict, report: Path) -> None:
    n_ok = sum(c["passed"] for c in status["checks"])
    print(f"checks: {n_ok}/{len(status['checks'])} passed; failed runs: {len(status.get('failed_runs', []))}")
    for t in status["targets"]:
        print(f"target [{'MET' if t['met'] else 'NOT MET'}] seed {t['seed']}: {t['target']} = {t['value']} "
              f"({t['threshold']})")
    print(f"wall time {status['wall_time_s']:.1f} s, peak memory {status.get('process_peak_memory_mb') or 0:.0f} MB, "
          f"output {status.get('output_size_mb', 0):.1f} MB")
    print(f"report: {report}")


def cmd_suite(args, title: str = "Experiment report") -> int:
    from .config import load_config
    from .experiments import run_suite
    from .report import write_report

    cfg = load_config(args.config)
    seeds = args.seeds or [cfg["dataset"]["seed"]]
    out = resolve_output(args.output or cfg["report"]["output"], args.overwrite)
    status = run_suite(cfg, out, seeds)
    report = write_report(out, cfg, status, _host(), title=title)
    _print_status(status, report)
    code = _exit_code(status)
    print("RESULT:", {0: "SUCCESS", 1: "FAILURE (check or run failed)", 2: "COMPLETED, some targets not met"}[code])
    return code


def cmd_demo(args) -> int:
    print("demo: generate data -> run experiment suite E0-E8 -> evaluate -> figures -> report")
    return cmd_suite(args, title="Demo report")


def cmd_verify(args) -> int:
    from .config import load_config
    from .experiments import VERIFY_RUN_IDS, read_comparison, run_pytest, run_suite, write_json
    from .report import write_report

    cfg = load_config(args.config)
    seeds = args.seeds or cfg["experiment"]["verification_seeds"]
    out = resolve_output(args.output, args.overwrite)
    out.mkdir(parents=True, exist_ok=True)
    print("verify: deterministic tests ...", flush=True)
    pt = run_pytest(out)
    print(f"  pytest: {'PASSED' if pt['passed'] else 'FAILED'} ({pt.get('summary')})")
    status = run_suite(cfg, out, seeds, include_e0=True, make_figures=False, run_ids=VERIFY_RUN_IDS)
    if not pt["passed"]:
        status["checks"].append({"check": "pytest deterministic test suite", "value": pt.get("summary"),
                                 "passed": False})
        status["all_checks_passed"] = False
    rows = read_comparison(out / "comparison.csv") if (out / "comparison.csv").exists() else []
    per_seed = []
    for s in seeds:
        by = {r["run_id"]: r for r in rows if r["seed"] == s}
        g = lambda k, f: by[k][f] if k in by else None  # noqa: E731
        per_seed.append({"seed": s, "eskf15_all": g("E2_eskf15_all", "pos_rmse_3d_m"),
                         "eskf9_all": g("E2_eskf9_all", "pos_rmse_3d_m"),
                         "imu_only": g("E2_imu_only", "pos_rmse_3d_whole_m"),
                         "recall": g("E8_eskf15_all_gating_on", "outlier_recall"),
                         "false_rejection": g("E8_eskf15_all_gating_on", "false_rejection_rate")})
    verification = {"pytest": pt, "per_seed": per_seed}
    status["pytest"] = pt
    status["per_seed"] = per_seed
    status["exit_code"] = _exit_code(status)
    write_json(out / "verification_status.json", status)
    report = write_report(out, cfg, status, _host(), title="Verification report", verification=verification)
    _print_status(status, report)
    print("machine-readable status:", out / "verification_status.json")
    code = status["exit_code"]
    print("RESULT:", {0: "SUCCESS", 1: "FAILURE (check or run failed)", 2: "COMPLETED, some targets not met"}[code])
    return code


def cmd_monte_carlo(args) -> int:
    from .config import load_config
    from .experiments import run_monte_carlo

    cfg = load_config(args.config)
    out = resolve_output(args.output, args.overwrite)
    s = run_monte_carlo(cfg, out, args.runs, mode=args.mode)
    print(f"{s['runs']} runs of {s['mode']}: position RMSE mean {s['pos_rmse_3d_m']['mean']:.3f} m, "
          f"std {s['pos_rmse_3d_m']['std']:.4f} m, max {s['pos_rmse_3d_m']['max']:.3f} m; "
          f"fraction below 1 m: {s['fraction_below_1m']:.2f}; mean GNSS NIS {s['mean_gnss_nis']:.2f}, "
          f"mean LiDAR NIS {s['mean_lidar_nis']:.2f}")
    print(f"outputs in {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m vehicle_localization", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="inspect the environment")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("generate", help="generate a synthetic dataset")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("validate", help="validate a dataset directory")
    p.add_argument("--data", required=True)
    p.add_argument("--normalize-imu", action="store_true", help="sort time-shuffled IMU rows instead of rejecting")
    p.set_defaults(func=cmd_validate)

    from .config import MODES

    p = sub.add_parser("run", help="run one estimator on a dataset")
    p.add_argument("--data", required=True)
    p.add_argument("--mode", choices=sorted(MODES), default="eskf15_all")
    p.add_argument("--output", default="results/baseline")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--gating", choices=["on", "off"], default=None)
    p.add_argument("--discretization", choices=["fast", "van_loan"], default=None)
    p.add_argument("--dropout", action="append", metavar="STREAM:START_S:END_S",
                   help="remove a stream's measurements in [start, end) seconds; repeatable")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_run)

    for name, func, helptext, default_out in (
        ("suite", cmd_suite, "run the experiment matrix E0-E8 and write the report", None),
        ("demo", cmd_demo, "standard demonstration: data -> suite -> figures -> report", "results/demo"),
        ("verify", cmd_verify, "deterministic tests plus selected multi-seed comparisons", "results/verification"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--config", default="configs/default.yaml")
        p.add_argument("--seeds", type=int, nargs="+", default=None)
        p.add_argument("--output", default=default_out)
        p.add_argument("--overwrite", action="store_true")
        p.set_defaults(func=func)

    p = sub.add_parser("monte-carlo", help="OPTIONAL statistical study over many seeds (sequential)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--mode", choices=sorted(MODES), default="eskf15_all")
    p.add_argument("--output", default="results/monte_carlo")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_monte_carlo)
    return ap


def main(argv: list[str] | None = None) -> int:
    # Tiny matrices: keep BLAS single-threaded (also covers the console-script entry point,
    # which does not go through __main__.py). NumPy is imported lazily after this point.
    import os

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileExistsError, ValueError, RuntimeError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
