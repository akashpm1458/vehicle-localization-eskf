"""End-to-end behaviour: truth isolation, reproducibility, covariance health, CLI."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vehicle_localization.dataset import load_dataset, load_ground_truth
from vehicle_localization.evaluation import evaluate_run
from vehicle_localization.experiments import execute_run
from vehicle_localization.runner import run_estimator

from .helpers import est_config

ROOT = Path(__file__).resolve().parents[1]


def _arrays(r):
    return np.hstack([r.p, r.v, r.q, r.ba, r.bg, r.P_diag])


def test_no_truth_leakage(smoke_dataset, tmp_path):
    """Removing or corrupting ground truth must not change the estimates."""
    path, cfg = smoke_dataset
    ref = run_estimator(load_dataset(path), est_config(cfg, "eskf15_all"))
    d = tmp_path / "no_truth"
    shutil.copytree(path, d)
    (d / "ground_truth.csv").unlink()
    (d / "truth_metadata.json").unlink()
    np.testing.assert_array_equal(_arrays(run_estimator(load_dataset(d), est_config(cfg, "eskf15_all"))), _arrays(ref))
    d2 = tmp_path / "bad_truth"
    shutil.copytree(path, d2)
    lines = (d2 / "ground_truth.csv").read_text().splitlines()
    lines[1:] = [",".join([ln.split(",")[0]] + ["0"] * 6 + ["0", "0", "0", "1"] + ["0"] * 6) for ln in lines[1:]]
    (d2 / "ground_truth.csv").write_text("\n".join(lines) + "\n")
    tm = json.loads((d2 / "truth_metadata.json").read_text())
    tm["truth_config"]["lidar"]["lever_arm_m"] = [9.0, 9.0, 9.0]
    (d2 / "truth_metadata.json").write_text(json.dumps(tm))
    np.testing.assert_array_equal(_arrays(run_estimator(load_dataset(d2), est_config(cfg, "eskf15_all"))), _arrays(ref))


def test_missing_truth_disables_accuracy_metrics_but_estimation_runs(smoke_dataset, tmp_path):
    path, cfg = smoke_dataset
    d = tmp_path / "no_truth"
    shutil.copytree(path, d)
    (d / "ground_truth.csv").unlink()
    res = execute_run(d, cfg, "eskf15_all", tmp_path / "run")
    assert res["metrics"]["accuracy"].startswith("unavailable")
    assert res["metrics"]["innovations"]["gnss"]["processed"] > 0
    assert (tmp_path / "run" / "innovations.csv").exists()


def test_reproducible_outputs(smoke_dataset):
    path, cfg = smoke_dataset
    a = run_estimator(load_dataset(path), est_config(cfg, "eskf15_all"))
    b = run_estimator(load_dataset(path), est_config(cfg, "eskf15_all"))
    np.testing.assert_allclose(_arrays(a), _arrays(b), rtol=0, atol=1e-12)
    assert [r["nis"] for r in a.innovations] == pytest.approx([r["nis"] for r in b.innovations], abs=1e-12)


@pytest.mark.parametrize("mode", ["imu_only", "eskf9_gnss", "eskf9_all", "eskf15_gnss", "eskf15_all"])
def test_covariance_finite_symmetric_psd_over_full_run(smoke_dataset, mode):
    path, cfg = smoke_dataset
    r = run_estimator(load_dataset(path), est_config(cfg, mode))
    assert np.all(np.isfinite(r.cov))
    for P in r.cov:
        np.testing.assert_array_equal(P, P.T)
        eig = np.linalg.eigvalsh(P)
        assert eig[0] >= -64 * P.shape[0] * np.finfo(float).eps * eig[-1]
    assert np.all(r.P_diag > 0)
    assert r.info["max_logged_quaternion_norm_error"] < 1e-10


def test_aided_filter_beats_imu_only_on_smoke_data(smoke_dataset):
    path, cfg = smoke_dataset
    ds, gt = load_dataset(path), load_ground_truth(path)
    rm = {m: evaluate_run(run_estimator(ds, est_config(cfg, m)), gt, 15 if "15" in m else 9, burn_in_s=2.0)
          for m in ("imu_only", "eskf15_all")}
    assert rm["eskf15_all"]["whole_run"]["position_rmse_3d_m"] < rm["imu_only"]["whole_run"]["position_rmse_3d_m"]
    assert rm["eskf15_all"]["after_burn_in"]["position_rmse_3d_m"] < 1.0


def test_cli_generate_validate_run(tmp_path):
    cfgp = ROOT / "configs" / "smoke.yaml"
    py = [sys.executable, "-m", "vehicle_localization"]
    data = tmp_path / "data"
    assert subprocess.run(py + ["generate", "--config", str(cfgp), "--output", str(data)], cwd=ROOT).returncode == 0
    assert subprocess.run(py + ["validate", "--data", str(data)], cwd=ROOT).returncode == 0
    out = tmp_path / "run"
    r = subprocess.run(py + ["run", "--data", str(data), "--mode", "eskf15_all", "--output", str(out),
                             "--config", str(cfgp), "--dropout", "gnss:4:6"], cwd=ROOT)
    assert r.returncode == 0
    for f in ("estimates.csv", "covariance.npz", "innovations.csv", "metrics.json", "config_resolved.yaml",
              "run_metadata.json"):
        assert (out / f).exists(), f
    with np.load(out / "covariance.npz", allow_pickle=False) as z:
        assert z["P"].shape[1:] == (15, 15)
    # a second run to the same directory must not overwrite silently
    r2 = subprocess.run(py + ["run", "--data", str(data), "--output", str(out), "--config", str(cfgp)], cwd=ROOT,
                        capture_output=True, text=True)
    assert r2.returncode == 0 and "new run directory" in r2.stdout
    bad = subprocess.run(py + ["validate", "--data", str(tmp_path / "nope")], cwd=ROOT)
    assert bad.returncode == 1


def test_suite_on_smoke_config_produces_report(tmp_path):
    """The whole experiment pipeline (E0-E8, figures, report) on the 10 s smoke config."""
    out = tmp_path / "suite"
    r = subprocess.run([sys.executable, "-m", "vehicle_localization", "suite", "--config",
                        str(ROOT / "configs" / "smoke.yaml"), "--output", str(out)], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode in (0, 2), r.stdout + r.stderr  # 2 = completed but a target was missed
    status = json.loads((out / "suite_status.json").read_text())
    assert status["all_checks_passed"] and not status["failed_runs"]
    report = (out / "report.md").read_text()
    for section in ("E1", "E2", "E3", "E5", "E6", "E7", "E8", "ground-truth-assisted"):
        assert section in report
    assert (out / "comparison.csv").exists()
    assert len(list((out / "figures").glob("*.png"))) >= 8
