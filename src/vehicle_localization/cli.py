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
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
