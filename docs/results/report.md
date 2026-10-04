# Demo report

Generated 2026-10-04 21:09 by vehicle_localization 0.1.0 from saved metrics in this directory. No numbers in this report were entered by hand.

> All sensor data are **synthetic**. "LiDAR" means **simulated LiDAR-localizer positions** (truth plus noise), not real scan matching. Initialization is **ground-truth-assisted**: the prior is the true state at t=0 minus a sampled perturbation.

## 1. Status

- Deterministic suite checks: **34/34 passed**
- Engineering targets: **4/4 met**
- Runtime: 74.6 s wall time, process peak memory 254 MB, output size 154.8 MB
- Host: remote/other host (not the target laptop): Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical CPUs, 15.7 GB RAM, Linux-6.18.44-fc-v64-x86_64-with-glibc2.39, Python 3.12.3

### Engineering targets

| target | seed | value | threshold | met | note |
|---|---|---|---|---|---|
| eskf15_all post-initialization 3-D position RMSE (biased case) | 7 | 0.096 | < 1.0 m | yes |  |
| full aided estimator beats IMU-only over the complete run (position RMSE) | 7 | [0.103, 1006.703] | eskf15_all < imu_only | yes |  |
| injected GNSS outlier recall with gating (eskf15_all) | 7 | 1.000 | >= 0.90 | yes | false rejection rate of clean GNSS: 0.0035 |
| bias estimation improves position RMSE in E2 (investigate if not) | 7 | [0.096, 0.162] | eskf15_all < eskf9_all | yes | comparison target, not a guaranteed result |

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

### E1 - zero-bias noisy motion

| run | mode | pos RMSE [m] | pos RMSE whole [m] | horiz. RMSE [m] | vel RMSE [m/s] | att RMSE [deg] | max pos err [m] | GNSS acc/rej | LiDAR acc/rej | GNSS NIS | LiDAR NIS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1_imu_only | imu_only | 976.406 | 934.844 | 976.173 | 20.368 | 1.650 | 2050.270 | 0/0 | 0/0 | n/a | n/a |
| E1_eskf9_gnss | eskf9_gnss | 0.613 | 0.635 | 0.389 | 0.152 | 0.502 | 1.383 | 599/1 | 0/0 | 2.80 | n/a |
| E1_eskf9_all | eskf9_all | 0.091 | 0.098 | 0.069 | 0.066 | 0.668 | 0.697 | 598/2 | 1195/5 | 2.79 | 2.98 |

Observation (computed): GNSS updates change the whole-run position RMSE from 934.844 m (IMU only) to 0.635 m; adding LiDAR positions gives 0.098 m.

### E2 - biased motion: bias estimation

| run | mode | pos RMSE [m] | pos RMSE whole [m] | horiz. RMSE [m] | vel RMSE [m/s] | att RMSE [deg] | max pos err [m] | GNSS acc/rej | LiDAR acc/rej | GNSS NIS | LiDAR NIS | accel bias RMSE [m/s^2] | gyro bias RMSE [rad/s] | min pos 3-sigma coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E2_imu_only | imu_only | 1051.460 | 1006.700 | 1041.890 | 21.598 | 2.970 | 2178.500 | 0/0 | 0/0 | n/a | n/a | 0.0527 | 0.00137 | 1.000 |
| E2_eskf9_all | eskf9_all | 0.162 | 0.161 | 0.082 | 0.137 | 2.929 | 0.697 | 598/2 | 1192/8 | 2.80 | 3.49 | 0.0527 | 0.00137 | 0.780 |
| E2_eskf15_all | eskf15_all | 0.096 | 0.103 | 0.075 | 0.072 | 1.978 | 0.697 | 598/2 | 1196/4 | 2.79 | 2.98 | 0.0212 | 0.00088 | 0.999 |

Observation (computed): with biased IMU data the 15-state filter's post-burn-in position RMSE is 0.096 m vs 0.162 m for the 9-state filter (lower, -40.7%). Mean pre-gate LiDAR NIS: 3.49 (9-state) vs 2.98 (15-state); minimum per-axis 3-sigma position coverage 0.780 vs 0.999. The 9-state filter assumes zero bias, so its covariance does not account for the bias-induced error.

Per-axis bias RMSE after burn-in (15-state): accelerometer [0.0135, 0.0159, 0.0034] m/s^2, gyro [0.00031, 0.00031, 0.00076] rad/s. Position-only aiding leaves some directions weakly observable (for example yaw and the vertical gyro bias are only excited by turning), so not every component is expected to converge equally.

### E3 / E4 / E5 - dropouts

| run | mode | pos RMSE [m] | pos RMSE whole [m] | horiz. RMSE [m] | vel RMSE [m/s] | att RMSE [deg] | max pos err [m] | GNSS acc/rej | LiDAR acc/rej | GNSS NIS | LiDAR NIS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E3_eskf15_gnss_reference | eskf15_gnss | 0.676 | 0.678 | 0.466 | 0.226 | 1.439 | 1.414 | 599/1 | 0/0 | 2.80 | n/a |
| E3_eskf15_gnss_gnss_dropout | eskf15_gnss | 3.216 | 3.086 | 3.123 | 0.447 | 1.653 | 15.060 | 499/1 | 0/0 | 2.71 | n/a |
| E4_eskf15_all_gnss_dropout | eskf15_all | 0.096 | 0.103 | 0.076 | 0.072 | 2.040 | 0.697 | 498/2 | 1196/4 | 2.71 | 2.98 |
| E5_eskf15_all_total_dropout | eskf15_all | 4.472 | 4.282 | 4.470 | 0.598 | 2.145 | 23.366 | 498/2 | 997/3 | 2.71 | 2.93 |
| E5_eskf15_all_total_dropout_gating_off | eskf15_all | 3.986 | 3.816 | 3.983 | 0.542 | 1.801 | 20.951 | 500/0 | 1000/0 | 2.71 | 2.92 |

| run | err before [m] | trace P_pos before [m^2] | max err during [m] | err at end [m] | trace P_pos at end [m^2] | err 5 s after [m] | updates in 10 s after | rejected in 10 s after |
|---|---|---|---|---|---|---|---|---|
| E3_eskf15_gnss_gnss_dropout | 0.362 | 0.597 | 15.027 | 15.027 | 1609.219 | 0.466 | 50 | 0 |
| E4_eskf15_all_gnss_dropout | 0.100 | 0.011 | 0.191 | 0.071 | 0.009 | 0.101 | 150 | 0 |
| E5_eskf15_all_total_dropout | 0.100 | 0.011 | 23.300 | 23.300 | 388.956 | 0.155 | 150 | 0 |
| E5_eskf15_all_total_dropout_gating_off | 0.093 | 0.011 | 20.890 | 20.890 | 385.498 | 0.153 | 150 | 0 |

Observation (computed): losing GNSS when it is the only aid (E3) lets the error reach 15.027 m during the outage; with LiDAR positions still arriving (E4) the maximum is 0.191 m. With both streams lost (E5) the maximum is 23.300 m and the position covariance trace grows from 0.011 to 388.956 m^2. After E5's outage, 0 of 150 returning measurements in the first 10 s were rejected by the gate.

### E6 - assumed GNSS covariance (same measurements)

| run | mode | pos RMSE [m] | pos RMSE whole [m] | horiz. RMSE [m] | vel RMSE [m/s] | att RMSE [deg] | max pos err [m] | GNSS acc/rej | LiDAR acc/rej | GNSS NIS | LiDAR NIS | min pos 3-sigma coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E6_eskf15_gnss_gnss_cov_x0.1 | eskf15_gnss | 204.840 | 196.194 | 202.982 | 16.210 | 42.995 | 722.219 | 19/581 | 0/0 | 73.73 | n/a | 0.498 |
| E6_eskf15_gnss_gnss_cov_x1 | eskf15_gnss | 0.676 | 0.678 | 0.466 | 0.226 | 1.439 | 1.414 | 599/1 | 0/0 | 2.80 | n/a | 0.995 |
| E6_eskf15_gnss_gnss_cov_x10 | eskf15_gnss | 0.878 | 0.949 | 0.497 | 0.264 | 1.811 | 2.219 | 600/0 | 0/0 | 0.30 | n/a | 1.000 |
| E6_eskf15_all_gnss_cov_x0.1 | eskf15_all | 0.099 | 0.106 | 0.077 | 0.073 | 1.703 | 0.697 | 201/399 | 1196/4 | 27.31 | 3.01 | 0.994 |
| E6_eskf15_all_gnss_cov_x1 | eskf15_all | 0.096 | 0.103 | 0.075 | 0.072 | 1.978 | 0.697 | 598/2 | 1196/4 | 2.79 | 2.98 | 0.999 |
| E6_eskf15_all_gnss_cov_x10 | eskf15_all | 0.096 | 0.104 | 0.075 | 0.072 | 1.912 | 0.697 | 600/0 | 1195/5 | 0.28 | 2.98 | 0.999 |

- E6_eskf15_all_gnss_cov_x0.1: mean pre-gate GNSS NIS 27.31, 399/600 GNSS measurements rejected, position RMSE 0.099 m.
- E6_eskf15_all_gnss_cov_x1: mean pre-gate GNSS NIS 2.79, 2/600 GNSS measurements rejected, position RMSE 0.096 m.
- E6_eskf15_all_gnss_cov_x10: mean pre-gate GNSS NIS 0.28, 0/600 GNSS measurements rejected, position RMSE 0.096 m.
- E6_eskf15_gnss_gnss_cov_x0.1: mean pre-gate GNSS NIS 73.73, 581/600 GNSS measurements rejected, position RMSE 204.840 m.
- E6_eskf15_gnss_gnss_cov_x1: mean pre-gate GNSS NIS 2.80, 1/600 GNSS measurements rejected, position RMSE 0.676 m.
- E6_eskf15_gnss_gnss_cov_x10: mean pre-gate GNSS NIS 0.30, 0/600 GNSS measurements rejected, position RMSE 0.878 m.

Interpretation: a mean pre-gate NIS well above 3 means the innovations are larger than the filter's own predicted innovation covariance S; well below 3 means they are smaller. An elevated NIS is a symptom, not a diagnosis: it can come from an assumed measurement covariance that is too small, but equally from process noise that is too small, unmodelled biases or calibration errors, timing errors, outliers or a diverging state. In E6 the measurements and every other setting are identical across runs and only the assumed GNSS covariance scale changes, so the *differences* between these runs can be attributed to that scale; the NIS value alone would not prove it.
**Gate lock-out (computed):** in E6_eskf15_gnss_gnss_cov_x0.1 the gate rejected most GNSS measurements, because the scaled-down assumed covariance made their innovations look improbable. With no other aid it then coasted on the IMU and diverged - over-confidence plus gating can remove the very corrections the filter needs.
**Gate lock-out (computed):** in E6_eskf15_all_gnss_cov_x0.1 the gate rejected most GNSS measurements, because the scaled-down assumed covariance made their innovations look improbable. LiDAR positions kept the estimate accurate despite the rejections.

### E7 - LiDAR lever-arm calibration error

| run | mode | pos RMSE [m] | pos RMSE whole [m] | horiz. RMSE [m] | vel RMSE [m/s] | att RMSE [deg] | max pos err [m] | GNSS acc/rej | LiDAR acc/rej | GNSS NIS | LiDAR NIS | mean err world x/y/z [m] | mean err body x/y/z [m] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E7_eskf15_all_lidar_arm_correct | eskf15_all | 0.096 | 0.103 | 0.075 | 0.072 | 1.978 | 0.697 | 598/2 | 1196/4 | 2.79 | 2.98 | [0.004, 7.486e-04, -0.009] | [0.009, 0.008, -0.009] |
| E7_eskf15_all_lidar_arm_wrong | eskf15_all | 0.249 | 0.249 | 0.242 | 0.079 | 1.929 | 0.697 | 599/1 | 1194/6 | 2.85 | 2.99 | [8.309e-04, -0.031, -0.008] | [0.205, 0.101, -0.008] |

Observation (computed): the wrong body-fixed lever arm (error [0.20, 0.10, 0.00] m, magnitude 0.224 m) raises post-burn-in position RMSE from 0.096 m to 0.249 m, while LiDAR mean NIS only moves from 2.98 to 2.99. The filter explains the LiDAR data by shifting its position estimate by R * (lever-arm error), so the innovations hardly reveal the fault; only the coarse GNSS disagrees. In world axes the signed mean error largely cancels as the heading turns through the figure-eight, but in the body frame it is [0.205, 0.101, -0.008] m - close to the lever-arm error itself. Only a translational lever-arm error was tested; rotational (boresight) calibration is not represented by a position-only measurement model.

### E8 - GNSS outliers, gating off vs on

| run | mode | pos RMSE [m] | pos RMSE whole [m] | horiz. RMSE [m] | vel RMSE [m/s] | att RMSE [deg] | max pos err [m] | GNSS acc/rej | LiDAR acc/rej | GNSS NIS | LiDAR NIS | outlier recall | false rejection rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E8_eskf15_gnss_gating_off | eskf15_gnss | 2.296 | 2.208 | 2.190 | 0.654 | 5.363 | 8.541 | 600/0 | 0/0 | 30.99 | n/a | 0.000 | 0.0000 |
| E8_eskf15_gnss_gating_on | eskf15_gnss | 0.701 | 0.701 | 0.469 | 0.229 | 1.470 | 1.436 | 571/29 | 0/0 | 29.03 | n/a | 1.000 | 0.0017 |
| E8_eskf15_all_gating_off | eskf15_all | 0.104 | 0.109 | 0.086 | 0.077 | 2.132 | 0.697 | 600/0 | 1200/0 | 33.06 | 3.03 | 0.000 | 0.0000 |
| E8_eskf15_all_gating_on | eskf15_all | 0.096 | 0.103 | 0.075 | 0.072 | 1.962 | 0.697 | 570/30 | 1196/4 | 33.08 | 2.98 | 1.000 | 0.0035 |

- eskf15_gnss: gating reduced post-burn-in position RMSE (2.296 m off -> 0.701 m on); recall 1.000, clean-measurement false rejection rate 0.0017.
- eskf15_all: gating reduced post-burn-in position RMSE (0.104 m off -> 0.096 m on); recall 1.000, clean-measurement false rejection rate 0.0035.

### Raw external position measurements (reference)

Each measurement compared with the truth of **its own reference point** at its own timestamp (not a causal estimator):

| stream | count | 3-D RMSE [m] | horizontal RMSE [m] |
|---|---|---|---|
| gnss | 600 | 2.373 | 1.359 |
| lidar | 1200 | 0.291 | 0.211 |

## Figures

- [fig01_trajectory_xy.png](figures/fig01_trajectory_xy.png) - XY ground truth, fused trajectories (biased data) and IMU-only on its own scale
- [fig02_position_error_dropouts.png](figures/fig02_position_error_dropouts.png) - Position error over time for the dropout experiments E3-E5 (shaded: outage interval)
- [fig03_position_axes_bounds.png](figures/fig03_position_axes_bounds.png) - Per-axis position error of eskf15_all with +/-3 sigma bounds (E2)
- [fig03b_attitude_axes_bounds.png](figures/fig03b_attitude_axes_bounds.png) - Per-axis local attitude error of eskf15_all with +/-3 sigma bounds (E2)
- [fig04_velocity_attitude_error.png](figures/fig04_velocity_attitude_error.png) - Velocity and attitude error: 9-state vs 15-state on biased data (E2)
- [fig05_biases.png](figures/fig05_biases.png) - Estimated vs true IMU biases, eskf15_all (E2)
- [fig06_nis_outliers.png](figures/fig06_nis_outliers.png) - Pre-gate NIS per sensor with the configured chi-square gate; rejected and injected outliers marked (E8, eskf15_all, gating on)
- [fig07_calibration_and_dropout.png](figures/fig07_calibration_and_dropout.png) - Left: lever-arm calibration error (E7), signed position error in the body frame. Right: total-dropout close-up (E5) with 3-sigma bound
- [fig08_summary.png](figures/fig08_summary.png) - Compact experiment summary (metric values with units)
- [fig09_e1_position_error.png](figures/fig09_e1_position_error.png) - E1: IMU-only vs 9-state fusion on zero-bias data (log scale)
- [fig10_e6_gnss_covariance_scaling.png](figures/fig10_e6_gnss_covariance_scaling.png) - E6: effect of the assumed GNSS covariance scale

![trajectory](figures/fig01_trajectory_xy.png)

## Limitations

- Synthetic data only; conclusions need validation on recorded data. Not tested on a real vehicle.
- The two position streams are simulated independently; real localizers can have correlated errors.
- Missing effects: multipath, real point-cloud registration, time-correlated errors, imperfect synchronization, temperature effects, complex 3-D motion.
- Coverage percentages and mean NIS from one trajectory are diagnostics, not proof of consistency.
- Offline processing; delayed or out-of-order measurements are not supported.
- Timings were measured on the host stated above. They are not laptop benchmarks unless the host line says so.
