# Verification report

Generated 2026-10-04 21:10 by vehicle_localization 0.1.0 from saved metrics in this directory. No numbers in this report were entered by hand.

> All sensor data are **synthetic**. "LiDAR" means **simulated LiDAR-localizer positions** (truth plus noise), not real scan matching. Initialization is **ground-truth-assisted**: the prior is the true state at t=0 minus a sampled perturbation.

## 1. Status

- Deterministic suite checks: **26/26 passed**
- pytest: **PASSED** (140 passed in 48.73s)
- Engineering targets: **12/12 met**
- Runtime: 47.5 s wall time, process peak memory 143 MB, output size 125.2 MB
- Host: remote/other host (not the target laptop): Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical CPUs, 15.7 GB RAM, Linux-6.18.44-fc-v64-x86_64-with-glibc2.39, Python 3.12.3

### Engineering targets

| target | seed | value | threshold | met | note |
|---|---|---|---|---|---|
| eskf15_all post-initialization 3-D position RMSE (biased case) | 7 | 0.096 | < 1.0 m | yes |  |
| full aided estimator beats IMU-only over the complete run (position RMSE) | 7 | [0.103, 1006.703] | eskf15_all < imu_only | yes |  |
| injected GNSS outlier recall with gating (eskf15_all) | 7 | 1.000 | >= 0.90 | yes | false rejection rate of clean GNSS: 0.0035 |
| bias estimation improves position RMSE in E2 (investigate if not) | 7 | [0.096, 0.162] | eskf15_all < eskf9_all | yes | comparison target, not a guaranteed result |
| eskf15_all post-initialization 3-D position RMSE (biased case) | 23 | 0.099 | < 1.0 m | yes |  |
| full aided estimator beats IMU-only over the complete run (position RMSE) | 23 | [0.111, 1795.459] | eskf15_all < imu_only | yes |  |
| injected GNSS outlier recall with gating (eskf15_all) | 23 | 1.000 | >= 0.90 | yes | false rejection rate of clean GNSS: 0.0017 |
| bias estimation improves position RMSE in E2 (investigate if not) | 23 | [0.099, 0.165] | eskf15_all < eskf9_all | yes | comparison target, not a guaranteed result |
| eskf15_all post-initialization 3-D position RMSE (biased case) | 42 | 0.096 | < 1.0 m | yes |  |
| full aided estimator beats IMU-only over the complete run (position RMSE) | 42 | [0.099, 801.324] | eskf15_all < imu_only | yes |  |
| injected GNSS outlier recall with gating (eskf15_all) | 42 | 1.000 | >= 0.90 | yes | false rejection rate of clean GNSS: 0.0052 |
| bias estimation improves position RMSE in E2 (investigate if not) | 42 | [0.096, 0.166] | eskf15_all < eskf9_all | yes | comparison target, not a guaranteed result |

## 2. What was generated

| dataset | seed | duration [s] | IMU [Hz] | GNSS meas. | LiDAR meas. | true biases | corruption |
|---|---|---|---|---|---|---|---|
| zero_bias | 7 | 120.0 | 100 | 600 | 1200 | zero | none |
| biased | 7 | 120.0 | 100 | 600 | 1200 | biased | none |
| biased_outliers | 7 | 120.0 | 100 | 600 | 1200 | biased | 28 GNSS outliers |

Trajectory: figure_eight (planar path in a 3-D estimator; not 6-DOF driving validation). Sensor rates and noise: see `config_resolved.yaml`.

## 3. E0 - noise-free correctness fixtures (IMU-only, exact initialization)

| fixture (10 s) | max position error [m] | max velocity error [m/s] | max attitude error [deg] | max quaternion norm error |
|---|---|---|---|---|
| stationary | 0.000 | 0.000 | 0.000 | 0.000 |
| stationary_tilted | 3.806e-15 | 7.613e-16 | 0.000 | 0.000 |
| constant_velocity | 7.802e-13 | 0.000 | 0.000 | 0.000 |
| constant_yaw_rate | 1.348e-05 | 1.402e-06 | 3.117e-13 | 2.220e-16 |
| smooth_rpy | 0.001 | 2.038e-04 | 4.002e-04 | 2.220e-16 |

Step-size convergence on curved paths (final position error after 10 s):

| trajectory | 100 Hz [m] | 200 Hz [m] | 400 Hz [m] | ratio 100/200 | ratio 200/400 |
|---|---|---|---|---|---|
| constant_yaw_rate | 1.348e-05 | 3.370e-06 | 8.426e-07 | 4.00 | 4.00 |
| figure_eight | 1.541e-05 | 3.852e-06 | 9.630e-07 | 4.00 | 4.00 |
| smooth_rpy | 0.001 | 2.684e-04 | 6.709e-05 | 4.00 | 4.00 |

IMU-only drift decomposition on the default trajectory (one effect at a time):

| case | error at 10 s [m] | error at 120 s [m] |
|---|---|---|
| exact init, no noise, no bias | 1.541e-05 | 1.546e-04 |
| exact init, IMU white noise only | 0.725 | 435.712 |
| exact init, IMU biases only | 3.408 | 453.190 |
| perturbed initial prior only | 18.061 | 2269.876 |
| all effects (default biased data) | 21.594 | 2178.501 |

Observation (computed): among the single effects, **perturbed initial prior only** produces the largest IMU-only drift. A small attitude error tilts the gravity vector into the horizontal axes, and a constant acceleration error integrates twice, so position error grows roughly with time squared.

## 4. Results (seed 7)

"pos RMSE" is computed after a 10 s burn-in; "whole" includes the initialization period. NIS columns are mean **pre-gate** NIS. About 3 is expected when the filter's models and uncertainties are consistent with the data; a departure signals some inconsistency (covariance, process model, calibration, timing or outliers) without identifying which. Diagnostic only.

### E2 - biased motion: bias estimation

| run | mode | pos RMSE [m] | pos RMSE whole [m] | horiz. RMSE [m] | vel RMSE [m/s] | att RMSE [deg] | max pos err [m] | GNSS acc/rej | LiDAR acc/rej | GNSS NIS | LiDAR NIS | accel bias RMSE [m/s^2] | gyro bias RMSE [rad/s] | min pos 3-sigma coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E2_imu_only | imu_only | 1051.460 | 1006.700 | 1041.890 | 21.598 | 2.970 | 2178.500 | 0/0 | 0/0 | n/a | n/a | 0.0527 | 0.00137 | 1.000 |
| E2_eskf9_all | eskf9_all | 0.162 | 0.161 | 0.082 | 0.137 | 2.929 | 0.697 | 598/2 | 1192/8 | 2.80 | 3.49 | 0.0527 | 0.00137 | 0.780 |
| E2_eskf15_all | eskf15_all | 0.096 | 0.103 | 0.075 | 0.072 | 1.978 | 0.697 | 598/2 | 1196/4 | 2.79 | 2.98 | 0.0212 | 0.00088 | 0.999 |

Observation (computed): with biased IMU data the 15-state filter's post-burn-in position RMSE is 0.096 m vs 0.162 m for the 9-state filter (lower, -40.7%). Mean pre-gate LiDAR NIS: 3.49 (9-state) vs 2.98 (15-state); minimum per-axis 3-sigma position coverage 0.780 vs 0.999. The 9-state filter assumes zero bias, so its covariance does not account for the bias-induced error.

Per-axis bias RMSE after burn-in (15-state): accelerometer [0.0135, 0.0159, 0.0034] m/s^2, gyro [0.00031, 0.00031, 0.00076] rad/s. Position-only aiding leaves some directions weakly observable (for example yaw and the vertical gyro bias are only excited by turning), so not every component is expected to converge equally.

### E8 - GNSS outliers, gating off vs on

| run | mode | pos RMSE [m] | pos RMSE whole [m] | horiz. RMSE [m] | vel RMSE [m/s] | att RMSE [deg] | max pos err [m] | GNSS acc/rej | LiDAR acc/rej | GNSS NIS | LiDAR NIS | outlier recall | false rejection rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E8_eskf15_all_gating_off | eskf15_all | 0.104 | 0.109 | 0.086 | 0.077 | 2.132 | 0.697 | 600/0 | 1200/0 | 33.06 | 3.03 | 0.000 | 0.0000 |
| E8_eskf15_all_gating_on | eskf15_all | 0.096 | 0.103 | 0.075 | 0.072 | 1.962 | 0.697 | 570/30 | 1196/4 | 33.08 | 2.98 | 1.000 | 0.0035 |

- eskf15_all: gating reduced post-burn-in position RMSE (0.104 m off -> 0.096 m on); recall 1.000, clean-measurement false rejection rate 0.0035.

### Raw external position measurements (reference)

Each measurement compared with the truth of **its own reference point** at its own timestamp (not a causal estimator):

| stream | count | 3-D RMSE [m] | horizontal RMSE [m] |
|---|---|---|---|
| gnss | 600 | 2.373 | 1.359 |
| lidar | 1200 | 0.291 | 0.211 |

## 5. Multi-seed verification

| seed | eskf15_all pos RMSE [m] | eskf9_all pos RMSE [m] | imu_only whole RMSE [m] | E8 recall | E8 false rejection |
|---|---|---|---|---|---|
| 7 | 0.096 | 0.162 | 1006.700 | 1.000 | 0.0035 |
| 23 | 0.099 | 0.165 | 1795.460 | 1.000 | 0.0017 |
| 42 | 0.096 | 0.166 | 801.324 | 1.000 | 0.0052 |

## Figures


## Limitations

- Synthetic data only; conclusions need validation on recorded data. Not tested on a real vehicle.
- The two position streams are simulated independently; real localizers can have correlated errors.
- Missing effects: multipath, real point-cloud registration, time-correlated errors, imperfect synchronization, temperature effects, complex 3-D motion.
- Coverage percentages and mean NIS from one trajectory are diagnostics, not proof of consistency.
- Offline processing; delayed or out-of-order measurements are not supported.
- Timings were measured on the host stated above. They are not laptop benchmarks unless the host line says so.
