"""Small helpers for runtime measurements recorded with every run."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process so far (MB), or None if not measurable."""
    try:
        if sys.platform == "win32":
            import psutil

            return psutil.Process().memory_info().peak_wset / 2**20
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 1024 if sys.platform != "darwin" else kb / 2**20
    except Exception:  # pragma: no cover - platform specific
        return None


def package_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "scipy", "matplotlib", "yaml"):
        try:
            out[name] = importlib.import_module(name).__version__
        except ImportError:
            out[name] = None
    return out


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 2**20
