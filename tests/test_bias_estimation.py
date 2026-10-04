"""15-state bias estimation and observability (no claim of universal convergence)."""

import numpy as np
import pytest

from vehicle_localization.dataset import load_dataset, load_ground_truth
from vehicle_localization.evaluation import error_series
from vehicle_localization.runner import run_estimator
from vehicle_localization.sensors import generate_dataset
from vehicle_localization.state import BA, BG

from .conftest import make_cfg
from .helpers import est_config


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = {}
    for traj in ("figure_eight", "stationary"):
        cfg = make_cfg(dataset={"trajectory": traj, "duration_s": 60.0})
        d = tmp_path_factory.mktemp(traj)
        generate_dataset(cfg, d / "data")
        ds, gt = load_dataset(d / "data"), load_ground_truth(d / "data")
        r = run_estimator(ds, est_config(cfg, "eskf15_all"))
        out[traj] = (r, error_series(r, gt), ds.prior.P)
    return out


def test_accelerometer_bias_converges_on_moving_trajectory(runs):
    r, es, P0 = runs["figure_eight"]
    true_ba0 = np.array([0.03, -0.02, 0.04])
    final_err = np.abs(es["dba"][-1])
    assert np.all(final_err < 0.5 * np.abs(true_ba0)), final_err
    sd_final = np.sqrt(r.P_diag[-1, BA])
    assert np.all(sd_final < 0.2 * np.sqrt(np.diag(P0)[BA]))
    assert np.all(np.abs(es["dba"][-1]) < 3 * sd_final)


def test_gyro_bias_uncertainty_shrinks_and_error_is_covered(runs):
    r, es, P0 = runs["figure_eight"]
    sd0 = np.sqrt(np.diag(P0)[BG])
    sd = np.sqrt(r.P_diag[-1, BG])
    assert np.all(sd < sd0)
    assert np.all(np.abs(es["dbg"][-1]) < 3 * sd)


def test_stationary_motion_leaves_yaw_and_vertical_gyro_bias_weakly_observable(runs):
    """At rest, position-only aiding cannot separate yaw from the vertical gyro bias."""
    r_move, _, P0 = runs["figure_eight"]
    r_stat, es_stat, _ = runs["stationary"]
    yaw_sd_stat = np.sqrt(r_stat.P_diag[-1, 8])
    yaw_sd_move = np.sqrt(r_move.P_diag[-1, 8])
    assert yaw_sd_stat > 2 * yaw_sd_move
    assert yaw_sd_stat > 0.9 * np.sqrt(P0[8, 8])  # yaw uncertainty does not collapse at rest
    # At rest, tilt and horizontal accelerometer bias are confounded: the tilt sigma
    # cannot shrink much below sigma_ba_prior / g (0.1 / 9.81 rad ~ 0.58 deg).
    tilt_sd = np.sqrt(r_stat.P_diag[-1, 6:8])
    ambiguity = np.sqrt(P0[9, 9]) / 9.80665
    assert np.all(tilt_sd < np.sqrt(P0[6, 6]))
    assert np.all(tilt_sd > 0.8 * ambiguity)
    # On the moving path the same directions become much better determined.
    assert np.all(np.sqrt(r_move.P_diag[-1, 6:8]) < 0.5 * ambiguity)
