# vehicle-localization-eskf

Offline vehicle localization with an **error-state Kalman filter (ESKF)** that fuses simulated IMU measurements with simulated GNSS positions and simulated LiDAR-localizer positions.

> **Status: in development.** The code has not been written yet. This README describes the planned project. The commands, results and figures below will be filled in from real runs as the work progresses. No result in this repository is claimed until a run has produced it.

## What this project does

The project reconstructs a vehicle's motion — position, velocity, orientation and, in the final estimator, accelerometer and gyroscope biases — from noisy sensor measurements. All data is generated synthetically, so no external dataset is needed.

It is meant to demonstrate:

1. Why pure IMU integration drifts.
2. How GNSS and external LiDAR-localizer positions correct that drift.
3. How uncertainty propagates through an error-state Kalman filter.
4. What happens during measurement dropout, incorrect calibration and outliers.
5. How estimating IMU biases changes performance.
6. Which conclusions synthetic data supports, and which would still need validation on recorded data.

## Estimators

All estimators run on the same generated sensor data and initial prior, so comparisons between them are fair.

| Name | Description |
|---|---|
| `imu_only` | Dead reckoning: integrates the IMU with zero bias estimates and no corrections. |
| `eskf9_gnss` | 9-state ESKF (position, velocity, attitude) with GNSS updates; assumes zero IMU biases. |
| `eskf9_all` | 9-state ESKF with GNSS and LiDAR-localizer position updates. |
| `eskf15_gnss` | 15-state ESKF that also estimates accelerometer and gyroscope biases; GNSS updates only. |
| `eskf15_all` | Final 15-state ESKF with both position streams. |

Key features:

- Midpoint-attitude IMU propagation, the same integrator for every estimator.
- Covariance discretization with a fast Simpson-rule method, validated against a Van Loan matrix-exponential reference.
- Position updates with body-fixed sensor lever arms.
- Joseph-form covariance correction followed by an SO(3) right-Jacobian covariance reset.
- Exact-timestamp processing of measurements that fall between IMU samples.
- Chi-square innovation gating (NIS) with full pre-gate logging.

## Conventions

| Item | Convention |
|---|---|
| World frame `W` | Right-handed local frame, `z` up |
| Body frame `B` | `x` forward, `y` left, `z` up; IMU at the body origin |
| Rotation | `R_WB` maps body vectors into world: `v_W = R_WB @ v_B` |
| Quaternions | Hamilton product, stored as `[qx, qy, qz, qw]` (SciPy order) |
| Attitude error | Right-multiplicative, body-local: `R_true = R_est · Exp(δθ)` |
| Gravity | `g_W = [0, 0, -9.80665] m/s²` |
| Units | Metres, seconds, radians; CSV timestamps are integer nanoseconds |

The full derivations will be in `docs/math.md`.

## Synthetic scenario

- A 120-second horizontal figure-eight trajectory with analytical derivatives.
- IMU at 100 Hz, GNSS at 5 Hz (σ = 1.0 / 1.0 / 2.0 m), LiDAR-localizer positions at 10 Hz (σ = 0.15 / 0.15 / 0.20 m).
- Optional accelerometer and gyroscope biases that evolve as random walks.
- Seeded, independent random streams for each noise source, so removing one sensor never changes another sensor's data.

"LiDAR-localizer positions" means noisy position outputs that stand in for an external localization system. They are **not** real LiDAR scans or scan-matching results.

## Planned experiments

| ID | Experiment |
|---|---|
| E0 | Noise-free fixtures: stationary, tilted, straight-line and turning motion |
| E1 | Zero-bias motion: IMU-only compared with 9-state fusion |
| E2 | Biased motion: 9-state compared with 15-state |
| E3 | 15-state with GNSS only, GNSS dropout from 40 to 60 s |
| E4 | 15-state with both streams, GNSS dropout from 40 to 60 s |
| E5 | 15-state with both streams, total dropout from 40 to 60 s |
| E6 | Assumed GNSS covariance scaled by 0.1, 1 and 10 |
| E7 | Wrong LiDAR lever arm assumed by the filter |
| E8 | 5% GNSS outliers, with gating off and on |

Results will appear here once the experiments have been run.

## Installation (planned)

Requires Python 3.11 or 3.12. Everything runs on the CPU; no GPU is needed.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m vehicle_localization demo
```

**Linux / WSL**

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m vehicle_localization demo
```

## Commands (planned)

```bash
python -m vehicle_localization doctor      # inspect the environment
python -m vehicle_localization generate --config configs/default.yaml --output data/generated/default
python -m vehicle_localization validate --data data/generated/default
python -m vehicle_localization run --data data/generated/default --mode eskf15_all --output results/baseline
python -m vehicle_localization suite --config configs/default.yaml --seeds 7 --output results/suite
python -m vehicle_localization verify --config configs/default.yaml --seeds 7 23 42 --output results/verification
python -m vehicle_localization demo --config configs/default.yaml --output results/demo
python -m pytest -q
```

## Repository layout (planned)

```
configs/                    default and smoke configurations
src/vehicle_localization/   package: rotations, state, trajectory, sensors, dataset,
                            discretization, eskf, runner, experiments, evaluation,
                            plotting, report, cli
tests/                      mathematical and end-to-end tests
docs/                       math.md, data_format.md, learning_guide.md, recorded_data.md
data/generated/             generated datasets (not committed)
results/                    run outputs and reports (not committed)
```

## Limitations

- This is **localization** only: there is no mapping and no loop closure.
- Classical estimation only; there are no neural networks.
- All sensor data is synthetic. The two position streams are simulated independently; real localizers may have correlated errors.
- The default trajectory is planar motion run through a 3-D estimator. It is not a full validation of 6-DOF driving.
- Processing is offline. Delayed or out-of-sequence measurements in a live system are not supported.
- Initialization is ground-truth-assisted: the initial prior is the true state plus a sampled perturbation.

## Background and references

This is an independent project inspired by the University of Toronto course *State Estimation and Localization for Self-Driving Cars* (Coursera). It is not a reproduction of the course assignment and contains no course solution code.

- J. Solà, [Quaternion kinematics for the error-state Kalman filter](https://arxiv.org/abs/1711.02508)
- SciPy: [`Rotation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html), [`linalg.expm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html), [`stats.chi2`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html)

## License

MIT. See [LICENSE](LICENSE).
