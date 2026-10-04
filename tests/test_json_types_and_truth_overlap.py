"""JSON object-type checks and ground truth that does not overlap the estimates."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vehicle_localization.dataset import DatasetError, load_dataset, load_ground_truth, validate_dataset
from vehicle_localization.experiments import RunSpec, evaluate_targets, execute_run, summarize
from vehicle_localization.sensors import derive_outlier_dataset

ROOT = Path(__file__).resolve().parents[1]
PY = [sys.executable, "-m", "vehicle_localization"]


@pytest.fixture
def ds_copy(smoke_dataset, tmp_path):
    d = tmp_path / "copy"
    shutil.copytree(smoke_dataset[0], d)
    return d


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _edit_json(path: Path, fn) -> None:
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj))


def _shift_truth(d: Path, offset_ns: int, keep_rows: slice = slice(None)) -> None:
    lines = (d / "ground_truth.csv").read_text().splitlines()
    rows = lines[1:][keep_rows]
    out = [lines[0]] + [f"{int(r.split(',', 1)[0]) + offset_ns},{r.split(',', 1)[1]}" for r in rows]
    (d / "ground_truth.csv").write_text("\n".join(out) + "\n")


# ----------------------------------------------------------------------------- input metadata types


@pytest.mark.parametrize("text", ["null", "[]", "\"metadata\"", "3", "true"])
def test_non_object_metadata_is_a_validation_error(ds_copy, text):
    _write(ds_copy / "metadata.json", text)
    res = validate_dataset(ds_copy)  # must not raise
    assert not res.ok
    assert any("metadata.json: top level must be a JSON object" in e for e in res.errors), res.errors
    with pytest.raises(DatasetError):
        load_dataset(ds_copy)


@pytest.mark.parametrize("field,value,message", [
    ("sensors", [], "sensors must be an object"),
    ("sensors", None, "sensors must be an object"),
    ("seed", "7", "seed must be a non-negative integer"),
    ("seed", True, "seed must be a non-negative integer"),
    ("duration_s", -1, "duration_s must be a positive number"),
    ("duration_s", None, "duration_s must be a positive number"),
    ("name", 5, "name must be a string"),
    ("units", [], "'units' must be an object"),
    ("timing", None, "'timing' must be an object"),
])
def test_wrong_typed_metadata_fields_are_rejected(ds_copy, field, value, message):
    _edit_json(ds_copy / "metadata.json", lambda m: m.__setitem__(field, value))
    res = validate_dataset(ds_copy)
    assert not res.ok and any(message in e for e in res.errors), res.errors


@pytest.mark.parametrize("text", ["null", "[]", "[1, 2]"])
def test_non_object_prior_is_a_validation_error(ds_copy, text):
    _write(ds_copy / "initial_prior.json", text)
    res = validate_dataset(ds_copy)
    assert not res.ok and any("initial_prior" in e and "JSON object" in e for e in res.errors), res.errors


# ----------------------------------------------------------------------------- optional truth metadata


@pytest.mark.parametrize("text", ["[]", "null", "\"x\"", "{not json"])
def test_bad_truth_metadata_disables_only_dependent_evaluation(ds_copy, smoke_dataset, tmp_path, text):
    _write(ds_copy / "truth_metadata.json", text)
    gt = load_ground_truth(ds_copy)  # truth itself stays usable
    assert gt is not None and gt.metadata == {} and gt.warnings
    res = validate_dataset(ds_copy)
    assert res.ok and any(w.startswith("[evaluation] truth_metadata.json") for w in res.warnings)
    out = execute_run(ds_copy, smoke_dataset[1], "eskf15_all", tmp_path / "run")
    m = out["metrics"]
    assert m["evaluation_available"] and m["after_burn_in"]["position_rmse_3d_m"] < 1.0
    assert m["external_position_reference_errors"] == {}  # needs the true lever arms
    assert any("truth_metadata.json" in w for w in m["warnings"])


def test_malformed_nested_truth_metadata_fields(ds_copy, smoke_dataset, tmp_path):
    base, cfg = smoke_dataset
    d = tmp_path / "outliers"
    derive_outlier_dataset(base, d, cfg["experiment"]["outliers"] | {"after_s": 2.0})

    def corrupt(tm):
        tm["truth_config"]["lidar"] = []  # GNSS entry stays valid
        tm["corruptions"]["outliers"]["stream"] = "radar"

    _edit_json(d / "truth_metadata.json", corrupt)
    gt = load_ground_truth(d)
    assert "lidar" not in gt.metadata["truth_config"] and "gnss" in gt.metadata["truth_config"]
    assert "outliers" not in gt.metadata["corruptions"]
    assert len(gt.warnings) == 2
    m = execute_run(d, cfg, "eskf15_all", tmp_path / "run")["metrics"]
    assert "outlier_detection" not in m
    assert set(m["external_position_reference_errors"]) == {"gnss"}

    _edit_json(d / "truth_metadata.json", lambda tm: tm.__setitem__("truth_config", []))
    assert "truth_config" not in load_ground_truth(d).metadata


# ----------------------------------------------------------------------------- truth overlap


def test_truth_without_overlap_marks_accuracy_unavailable(ds_copy, smoke_dataset, tmp_path):
    _shift_truth(ds_copy, offset_ns=1_000_500_000_000)  # truth 1000.5 s later, off the estimate grid
    out = execute_run(ds_copy, smoke_dataset[1], "eskf15_all", tmp_path / "run")
    m = out["metrics"]
    assert m["evaluation_available"] is False
    assert "does not overlap" in m["accuracy"]
    assert m["truth_overlap"]["samples"] == 0
    assert "after_burn_in" not in m and "external_position_reference_errors" not in m
    assert m["innovations"]["gnss"]["processed"] > 0  # innovation statistics preserved
    assert (tmp_path / "run" / "estimates.csv").exists()
    row = summarize(RunSpec("E2", "E2_eskf15_all", "biased", "eskf15_all", ""), m, 7)
    assert row["pos_rmse_3d_m"] is None and row["gnss_accepted"] > 0
    t = evaluate_targets([row], 7)
    assert t and not t[0]["met"] and "unavailable" in t[0]["note"]


def test_partial_and_single_sample_truth(ds_copy, smoke_dataset, tmp_path):
    _shift_truth(ds_copy, offset_ns=5_000_000, keep_rows=slice(0, 500))  # off-grid, first ~5 s only
    m = execute_run(ds_copy, smoke_dataset[1], "eskf15_all", tmp_path / "a")["metrics"]
    assert m["evaluation_available"] and 0 < m["truth_overlap"]["samples"] < m["truth_overlap"]["of"]
    _shift_truth(ds_copy, offset_ns=3_000_000, keep_rows=slice(0, 1))  # one row, not on any estimate time
    m = execute_run(ds_copy, smoke_dataset[1], "eskf15_all", tmp_path / "b")["metrics"]
    assert m["evaluation_available"] is False


def test_cli_run_skips_truth_plots_without_overlap(ds_copy, tmp_path):
    _shift_truth(ds_copy, offset_ns=-50_000_000_000)  # truth entirely before the estimates
    out = tmp_path / "run"
    r = subprocess.run(PY + ["run", "--data", str(ds_copy), "--output", str(out), "--config",
                             str(ROOT / "configs" / "smoke.yaml")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "does not overlap" in r.stdout and "truth-dependent plots skipped" in r.stdout
    assert (out / "nis.png").exists() and (out / "estimates.csv").exists()
    assert not (out / "trajectory_xy.png").exists() and not (out / "position_error_bounds.png").exists()
    m = json.loads((out / "metrics.json").read_text())
    assert m["evaluation_available"] is False and m["innovations"]
