"""Environment inspection (`doctor` command).

Reports only non-sensitive host facts: OS, Python, CPU model, core count, RAM, free disk
and dependency versions. No serial numbers, user names or network identifiers.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

TARGET = {
    "description": "HP Victus 15 (i7-12650H, RTX 3050 Laptop 4 GB, 16 GB RAM)",
    "cpu_substring": "12650H",
    "ram_gb": 16,
}
DEPENDENCIES = ["numpy", "scipy", "matplotlib", "yaml", "pytest", "psutil", "threadpoolctl"]


def _cpu_model() -> str:
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=10,
            )
            if out.stdout.strip():
                return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def _gpu() -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "not inspected (nvidia-smi not found; GPU is not needed)"
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def collect(path: str | Path = ".") -> dict:
    info: dict = {
        "python": sys.version.split()[0],
        "python_ok": (3, 11) <= sys.version_info[:2] <= (3, 12),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpus": __import__("os").cpu_count(),
    }
    try:
        import psutil

        info["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        info["ram_available_gb"] = round(psutil.virtual_memory().available / 2**30, 1)
    except ImportError:
        info["ram_total_gb"] = info["ram_available_gb"] = None
    du = shutil.disk_usage(Path(path).resolve())
    info["disk_free_gb"] = round(du.free / 2**30, 1)
    deps = {}
    for name in DEPENDENCIES:
        try:
            mod = importlib.import_module(name)
            deps[name] = getattr(mod, "__version__", "installed")
        except ImportError:
            deps[name] = None
    info["dependencies"] = deps
    info["gpu"] = _gpu()
    cpu_match = TARGET["cpu_substring"] in info["cpu_model"]
    ram_match = info["ram_total_gb"] is not None and abs(info["ram_total_gb"] - TARGET["ram_gb"]) < 2
    info["matches_target_laptop"] = bool(cpu_match and ram_match)
    return info


def report(info: dict) -> str:
    lines = ["vehicle_localization doctor", "-" * 40]
    lines.append(f"Python            {info['python']} ({'OK' if info['python_ok'] else 'need 3.11 or 3.12'})")
    lines.append(f"Platform          {info['platform']}")
    lines.append(f"CPU               {info['cpu_model']} ({info['logical_cpus']} logical)")
    lines.append(f"RAM               total {info['ram_total_gb']} GB, available {info['ram_available_gb']} GB")
    lines.append(f"Free disk         {info['disk_free_gb']} GB (project budget: < 0.25 GB of data + results)")
    lines.append(f"GPU               {info['gpu']}")
    lines.append("Dependencies:")
    for k, v in info["dependencies"].items():
        lines.append(f"  {k:<14} {v if v else 'MISSING'}")
    lines.append(f"Target laptop     {TARGET['description']}")
    if info["matches_target_laptop"]:
        lines.append("Host              matches the target laptop specification")
    else:
        lines.append("Host              does NOT match the target laptop; timings measured here are not "
                     "laptop benchmarks. Budgets (1 GB RAM, 250 MB disk, CPU only) still apply.")
    missing = [k for k, v in info["dependencies"].items() if v is None and k != "pytest"]
    lines.append("Status            " + ("OK" if info["python_ok"] and not missing else f"PROBLEM (missing: {missing})"))
    return "\n".join(lines)


def host_description(info: dict) -> str:
    """Short non-sensitive host label for reports."""
    label = "target laptop" if info["matches_target_laptop"] else "remote/other host (not the target laptop)"
    return (f"{label}: {info['cpu_model']}, {info['logical_cpus']} logical CPUs, "
            f"{info['ram_total_gb']} GB RAM, {info['platform']}, Python {info['python']}")
