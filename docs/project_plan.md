# Project Plan — Step by Step

This file breaks the whole project brief into small steps, in build order. Every requirement from the brief is placed in one of the steps below.

- Work through the steps in order.
- Finish a step's **Done when** before starting the next step.
- Tick the boxes as you go.

## Status (implementation complete)

| Phase | State | Evidence |
|---|---|---|
| Part 0 — Ground rules | followed | Budgets checked in the report; remote timings labelled as remote |
| A — Environment and data | done | `doctor`, `generate`, `validate`; tests in `tests/test_rotations.py`, `test_truth_and_sensors.py`, `test_dataset_and_config.py` |
| B — Inertial propagation | done | Zero-noise fixtures drift below 1e-6 m; scheduler tests in `test_propagation_and_scheduler.py` |
| C — Nine-state fusion | done | Jacobian, reset and discretization tests in `test_eskf_math.py`; E1 in the report |
| D — Fifteen-state | done | `test_bias_estimation.py`; E2 in the report |
| E — Robustness | done | E0–E8 in [results/report.md](results/report.md); seeds 7, 23, 42 in [results/verification_report.md](results/verification_report.md) |
| F — Docs and handover | done | `math.md`, `data_format.md`, `learning_guide.md`, `recorded_data.md`, README |

### Deliberate choices

These go beyond the brief or interpret it:

- **E6** runs both `eskf15_gnss` and `eskf15_all`, because the effect of the GNSS covariance is only clearly visible when GNSS is the only aid.
- **E8** runs both modes, for the same reason.
- **E3** adds a no-dropout reference run.
- **E5** adds a gating-off diagnostic run.
- **`verify`** runs E0, the pytest suite, E2 and E8 for each seed, rather than the full matrix, to stay within the disk budget.
- **E9, E10 and RPE** were added after reviewing two papers (Xu, ICMAE 2024; Wang et al., *Mathematics* 2024):
  - E9: degraded GNSS that the receiver doesn't report.
  - E10: an option to switch between sources instead of fusing them (`estimator.fusion_policy`).
  - RPE: relative position error per window.

  The KITTI comparison was **not** done: it needs a registered download that exceeds the disk budget,
  and its OXTS ground truth would need careful separation from the GNSS input.

### Not done or not verified

- Results on the target laptop. All timings so far are from a remote cloud machine.
- Installation on Windows.
- The optional Monte Carlo study: the command exists and works, but the 20-run study has not been run.
- All optional extensions (Steps X1–X4).

---

**Contents**

- [Part 0 — Ground rules](#part-0--ground-rules) (Steps 0.1–0.7)
- [Phase A — Environment, conventions and data](#phase-a--environment-conventions-and-data) (Steps A1–A15)
- [Phase B — Inertial propagation](#phase-b--inertial-propagation) (Steps B1–B6)
- [Phase C — Nine-state fusion](#phase-c--nine-state-fusion) (Steps C1–C10)
- [Phase D — Fifteen-state estimator](#phase-d--fifteen-state-estimator) (Steps D1–D4)
- [Phase E — Robustness experiments](#phase-e--robustness-experiments) (Steps E1–E15)
- [Phase F — Documentation and handover](#phase-f--documentation-and-handover) (Steps F1–F10)
- [Later — Optional extensions](#later--optional-extensions-not-in-the-required-scope) (Steps X1–X4)
- [Appendix — References](#appendix--references)

---

## Part 0 — Ground rules

Read these before writing any code. They apply to every step.

### Step 0.1 — Know the goal

Build a small, offline localization system that estimates a vehicle's motion from noisy sensors.

The finished project must show:

- [ ] Why IMU integration drifts.
- [ ] How GNSS and external LiDAR-localizer positions correct that drift.
- [ ] How uncertainty propagates through an error-state Kalman filter.
- [ ] What changes during measurement dropout, wrong calibration and outliers.
- [ ] How estimating IMU biases changes performance.
- [ ] Which conclusions synthetic data supports, and which need recorded data later.

### Step 0.2 — Know the scope limits

- Localization only: no map building and no loop closure.
- Classical estimation only: no neural networks.
- "LiDAR position" means the output of an external localizer, not raw laser ranges or point clouds.
- Always call it a **simulated LiDAR-localizer position**, never a real LiDAR scan-matching result.
- The GNSS and LiDAR streams are simulated independently. Real localizers can share information with GNSS or the IMU and have correlated errors, so real outputs can't automatically be fused as independent measurements.
- Radar/camera SLAM is a possible later project. Don't call this project learning-based radar-camera fusion.
- No web application and no pile of frameworks.

### Step 0.3 — Know the hardware limits

Target laptop: HP Victus 15, i7-12650H, RTX 3050 4 GB, 16 GB RAM. These specs came from the user and haven't been measured.

| Limit | Rule |
|---|---|
| CPU | NumPy/SciPy on the CPU only. |
| RAM | Keep a normal run under **1 GB** of process memory. No large point clouds or image sequences. |
| GPU | Not used. No CUDA, no PyTorch. |
| OS | Support Windows PowerShell and Linux/WSL. Detect the platform before installing. |
| Disk | Report free space before setup. Keep generated data plus results under **250 MB** (not counting the venv). |
| Style | Explicit equations and small modules, not opaque filtering frameworks. |

- These are budgets, not measured claims.
- Record the actual peak memory, wall time, platform and package versions.
- If something runs on a remote machine, label its timings **remote**. Never attribute them to the laptop.
- If the code runs on a bigger machine, keep the laptop limits anyway.

### Step 0.4 — Know the runtime budget

| Item | Value |
|---|---|
| Default dataset | 120 s; IMU 100 Hz, GNSS 5 Hz, LiDAR 10 Hz |
| Smoke dataset | 10 s |
| Development suite | All experiments, seed 7 |
| Verification suite | Selected comparisons, seeds 7, 23 and 42 |
| Optional study | 20 seeds, only when explicitly asked, run one after another |
| One filter run | Target under 60 s on the laptop |
| Full default suite | Target under 5 min on the laptop, not counting install |

- These are targets to measure, never results to fake.
- Use one process by default.
- Limit BLAS threads before importing NumPy in the CLI launcher (or use `threadpoolctl`).
- No CARLA, Docker, ROS, point-cloud registration, GPU kernels, cloud services or paid APIs.
- After installation, everything (generate, estimate, test, evaluate, report) runs offline.

### Step 0.5 — Know the honesty rules

- Never invent runs, metrics, plots or benchmarks.
- If something can't be run, finish everything that can be done and list exactly what's unverified.
- Don't type desirable results into the report by hand.
- Don't cherry-pick a favourable seed.
- Keep the artifacts of failed runs for diagnosis.
- Make ordinary implementation decisions yourself, and document them.
- Don't stop after scaffolding or a plan.

### Step 0.6 — Know the course relationship

- This is an independent project **inspired by** the University of Toronto course *State Estimation and Localization for Self-Driving Cars*.
- The course's final assignment is a CARLA-based ES-EKF called "Vehicle State Estimation on a Roadway".
- Don't present this project as a copy of that assignment.
- Don't use copied solution code.

### Step 0.7 — Know the required deliverables

1. [ ] A deterministic synthetic trajectory and sensor generator.
2. [ ] A strict CSV/JSON data format and a validator for user-supplied data.
3. [ ] IMU-only dead reckoning.
4. [ ] A 9-state ESKF (position, velocity, orientation) that assumes zero IMU bias.
5. [ ] A 15-state ESKF that also estimates both IMU biases.
6. [ ] GNSS and LiDAR position updates, each with its own covariance and lever arm.
7. [ ] Correct time-ordered processing at measurement timestamps.
8. [ ] Innovation logging and configurable outlier rejection.
9. [ ] The experiment matrix, plots, machine-readable metrics and a generated Markdown report.
10. [ ] Tests of rotations, Jacobians, timing, numerical stability and end-to-end behaviour.
11. [ ] A beginner-readable README and a math explanation.
12. [ ] One command that runs the whole demo, from data generation to report.

---

## Phase A — Environment, conventions and data

**Phase exit check:** the motion can be verified independently, and the sensor units and frames are correct.

### Step A1 — Check the environment

- [ ] Read any repository instructions and look at the existing files.
- [ ] Keep any existing user work.
- [ ] Detect the OS, Python version, RAM and free disk space.
- [ ] Use Python 3.11 or 3.12. If 3.12 is already installed, use it and document that.
- [ ] Create a project-local `.venv`.
- [ ] Don't change system-wide PowerShell execution policies.
- [ ] Don't require a global Python install if a suitable one already exists.

**Done when:** the venv works and the host details are written down.

### Step A2 — Create the project skeleton

Dependencies:
- Required: Python 3.11/3.12, NumPy, SciPy, Matplotlib, PyYAML, pytest.
- Allowed: a lightweight memory-monitoring package.
- Optional (never required): Pandas, Jupyter.

Files and folders:

| Path | Purpose |
|---|---|
| `pyproject.toml` | Metadata, runtime dependencies, `[dev]` extras |
| `requirements-tested.txt` | Exact versions used for the reported results |
| `.gitignore` | Ignore `.venv`, caches, `data/generated/`, `results/` |
| `configs/default.yaml` | All default settings |
| `configs/smoke.yaml` | Small development settings |
| `src/vehicle_localization/__main__.py` | `python -m vehicle_localization` entry |
| `src/vehicle_localization/cli.py` | Argument parsing and commands |
| `src/vehicle_localization/rotations.py` | Quaternions, skew, rotation vectors, Jacobians |
| `src/vehicle_localization/state.py` | Nominal state, error-state slices, priors |
| `src/vehicle_localization/trajectory.py` | Ground-truth trajectories |
| `src/vehicle_localization/sensors.py` | Synthetic sensors, noise and biases |
| `src/vehicle_localization/dataset.py` | Load and validate CSV/JSON |
| `src/vehicle_localization/discretization.py` | Continuous-to-discrete covariance |
| `src/vehicle_localization/eskf.py` | Shared 9-state and 15-state filter code |
| `src/vehicle_localization/runner.py` | Timestamp scheduling and logging |
| `src/vehicle_localization/experiments.py` | Experiment definitions |
| `src/vehicle_localization/evaluation.py` | Errors, RMSE, coverage, NIS stats |
| `src/vehicle_localization/plotting.py` | Figures |
| `src/vehicle_localization/report.py` | Markdown report from saved metrics |
| `tests/` | Tests |
| `docs/` | `math.md`, `data_format.md`, `learning_guide.md`, `recorded_data.md` |

Coding rules:
- Use `pathlib` and UTF-8 files.
- Use a headless Matplotlib backend for CLI runs.
- Use `float64` for all filter maths.
- Don't claim an untested lockfile works on every platform.

Suggested public interfaces:

```python
generate_dataset(config, output_dir) -> DatasetManifest
validate_dataset(dataset_dir) -> ValidationResult
run_estimator(dataset, initial_prior, estimator_config) -> RunResult
evaluate_run(run_result, ground_truth) -> Metrics
run_suite(config, output_dir) -> SuiteResult

class ErrorStateEKF:
    def predict(self, specific_force, angular_rate, dt): ...
    def update_position(self, z_world, covariance, lever_arm_body, sensor): ...
```

**Done when:** `pip install -e ".[dev]"` works and `python -m vehicle_localization` starts.

### Step A3 — Build the `doctor` command

It reports:
- [ ] Python version and platform.
- [ ] Available RAM, when it can be measured.
- [ ] Free disk space.
- [ ] Which dependencies are available.
- [ ] Whether the host matches the target laptop.

Rules:
- No serial numbers or unrelated personal information.
- GPU inspection is optional and never required for success.

**Done when:** `python -m vehicle_localization doctor` prints a clear report.

### Step A4 — Fix the frames and units

- `W` (world): right-handed, `z` up. It can be local ENU for real data; for synthetic data `x/y` are just local axes.
- `B` (body/IMU): `x` forward, `y` left, `z` up.
- The IMU sits at the body origin, and its axes match `B`.
- `R_WB` maps body vectors into world: `v_W = R_WB @ v_B`.
- `p_WB` is the position of the body origin in `W`.
- `v_WB` is the world-frame velocity of that origin.
- Gravity: `g_W = [0, 0, -9.80665] m/s²`.
- Units: metres, seconds (internally), radians.
- CSV timestamps: integer nanoseconds from the start of the dataset.

**Done when:** these are written in `docs/math.md` and used everywhere.

### Step A5 — Write the rotation utilities

Conventions:
- Hamilton quaternion product.
- Always store quaternions as `[qx, qy, qz, qw]` (SciPy order) in JSON, CSV and arrays.
- The quaternion represents `R_WB`.
- Orientation error is right-multiplicative and body-local: `R_true = R_est · Exp(δθ)`.
- `Exp(φ)` is the SO(3) exponential of a rotation vector; `quat(φ)` is its unit quaternion.
- Prediction composes on the right: `q_next = q ⊗ quat(ω·Δt)`.
- Correction composes on the right: `q_corr = q ⊗ quat(δθ)`.
- Every attitude Jacobian uses this same error convention. Check it against Solà's ESKF paper.

Functions to write:
- [ ] `skew(u)`
- [ ] Quaternion normalize, multiply and conjugate.
- [ ] Rotation vector ↔ quaternion.
- [ ] Quaternion → rotation matrix.
- [ ] A numerically stable SO(3) right Jacobian `J_r(φ)` (Step C5).

SciPy may do the work behind documented wrappers; a full hand-written quaternion library isn't needed.

**Done when** these tests pass:
- [ ] `skew(u) @ v == np.cross(u, v)`.
- [ ] A +90° rotation about `z` maps `[1,0,0]` to `[0,1,0]`.
- [ ] `q` and `-q` give the same orientation.
- [ ] Known 90° rotations, composition, inverse and normalization all behave correctly.

### Step A6 — Write the configuration system

Keep every parameter out of the estimator code. The config groups are:

| Group | Contains |
|---|---|
| `dataset` | Duration, sample rates, phase offsets, trajectory, seed, output path |
| `truth` | Noise intensities, true initial biases, bias random walks, true lever arms |
| `estimator` | Mode, assumed gravity, assumed noise, assumed lever arms, covariance multipliers, gating, discretization, maximum substep |
| `initialization` | Prior source and uncertainty settings |
| `experiment` | Dropped streams and intervals, injected outliers, named perturbations |
| `report` | Burn-in duration, plot choices, output directory |

Rules:
- [ ] Validate rates, non-negative noise densities, positive measurement variances, timestamp coverage and units.
- [ ] Save the fully resolved config with every run.
- [ ] Keep filter-visible config separate from truth-only config. The runner passes the filter only what it's allowed to know.

**Done when:** an invalid config fails with a clear message.

### Step A7 — Build the figure-eight ground truth

Path (120 s):

```
u   = 2π t / 60
p_W = [40 sin u,  20 sin 2u,  0]
v_W = [40 u̇ cos u,  40 u̇ cos 2u,  0]
a_W = [-40 u̇² sin u,  -80 u̇² sin 2u,  0]
```

Orientation:

```
roll = pitch = 0
yaw  = atan2(v_y, v_x)
yaw_rate = (v_x a_y − v_y a_x) / (v_x² + v_y²)
omega_B  = [0, 0, yaw_rate]
```

Rules:
- [ ] Differentiate analytically.
- [ ] Build orientation directly from yaw. Never differentiate wrapped Euler angles.
- [ ] The speed is never zero on this path.
- [ ] **Never** generate truth with the estimator's own integrator, or a shared bug could cancel out.
- [ ] Describe it as a planar path in a 3-D estimator, not as 6-DOF driving validation.

**Done when:** a test confirms the analytical derivatives match numerical differences away from the boundaries.

### Step A8 — Build the small test fixtures

Each fixture is independent and kept simple:
- [ ] Stationary.
- [ ] Straight line at constant velocity.
- [ ] Constant yaw rate (turning).
- [ ] Tilted and stationary.
- [ ] A short, smooth roll/pitch/yaw motion for checking frame conventions.

These are for correctness, not realistic vehicle dynamics.

**Done when:** each fixture gives position, velocity, orientation and angular rate.

### Step A9 — Simulate the IMU

The accelerometer measures **specific force**, not world acceleration:

```
f_B = R_WBᵀ (a_W − g_W)
f_m = f_B + b_a + n_a
ω_m = ω_B + b_g + n_g
```

Bias random walk:

```
b_a[k+1] = b_a[k] + σ_ba √Δt ε
b_g[k+1] = b_g[k] + σ_bg √Δt ε
```

Defaults:

| Parameter | Value |
|---|---|
| IMU rate | 100 Hz |
| Accelerometer noise (√intensity) | 0.02 m/s^(3/2) per axis |
| Gyroscope noise (√intensity) | 0.001 rad/s^(1/2) per axis |
| Accelerometer bias random walk | 0.0002 (m/s²)/√s per axis |
| Gyroscope bias random walk | 0.00002 (rad/s)/√s per axis |
| True initial accel bias (biased runs) | [0.03, −0.02, 0.04] m/s² |
| True initial gyro bias (biased runs) | [0.001, −0.0005, 0.0008] rad/s |
| Default seed | 7 |

Rules:
- [ ] A stationary, level body gives exactly `f_B = [0, 0, +9.80665]`.
- [ ] Support both zero biases and time-varying biases.
- [ ] Zero-bias runs set **both** the initial biases **and** the random-walk coefficients to zero.
- [ ] Keep white noise on unless a fixture is explicitly noise-free.

**Done when:** stationary level and tilted bodies give zero predicted world acceleration (with no noise and no bias).

### Step A10 — Apply the noise convention

- Continuous white noise: `E[n(t) n(t')ᵀ] = Qc δ(t − t')`.
- Per IMU interval `dt`, the interval-average noise has standard deviation `σ_density / √dt`.
- Each bias random-walk step has standard deviation `σ_rw · √dt`.
- Use **exactly** the same convention inside the filter. Don't square, or multiply by `dt`, twice.
- Document that real IMU datasheets may use a different (one-sided) convention; these are teaching values, not a real IMU's specification.

**Done when:** the convention is documented and a test checks the sample variance.

### Step A11 — Use separate random streams

Give each of these its own seeded random stream:
- [ ] IMU white noise
- [ ] Bias random walks
- [ ] GNSS noise
- [ ] LiDAR noise
- [ ] Initial-prior errors
- [ ] Injected outliers

Rules:
- [ ] Removing one sensor never changes another sensor's data.
- [ ] Save the seeds and the resolved config.

**Done when:** a test shows that dropping LiDAR leaves the GNSS and IMU data identical.

### Step A12 — Simulate GNSS and LiDAR positions

Each sensor reports the world position of its own reference point:

```
z_s = p_WB + R_WB ℓ_Bs + n_s
```

Here `ℓ_Bs` is the lever arm from the body origin to the sensor, in `B`.

| Stream | Rate | First sample | σ per axis | True lever arm |
|---|---|---|---|---|
| GNSS | 5 Hz | 0.03 s | [1.0, 1.0, 2.0] m | [0.2, 0.0, 1.0] m |
| LiDAR localizer | 10 Hz | 0.07 s | [0.15, 0.15, 0.20] m | [0.5, 0.0, 1.2] m |

Rules:
- [ ] These default offsets fall on the IMU grid, but the two sensors have different rates and phases.
- [ ] Baseline covariance: `R_s = diag(σ²)`.
- [ ] The file format must also support a full symmetric 3×3 covariance.
- [ ] Noisy update runs require a positive-definite covariance.
- [ ] Assume the LiDAR positions are already in the same world frame as GNSS. For real data in a different map frame, transform both the position and its covariance using separately supplied calibration.

**Done when:** sensor positions match the truth plus the rotated lever arm in a noise-free test.

### Step A13 — Write the data files

| File | Contents |
|---|---|
| `imu.csv` | `t_ns,fx,fy,fz,wx,wy,wz` (m/s², rad/s) |
| `gnss.csv` | `t_ns,x,y,z,r_xx,r_xy,r_xz,r_yy,r_yz,r_zz` |
| `lidar_position.csv` | Same as GNSS; may be absent for IMU+GNSS runs |
| `initial_prior.json` | Start time, `p`, `v`, quaternion (`xyzw`), bias estimates, covariance, provenance |
| `metadata.json` | Schema version, frames, units, timing rules, seed, duration, sensor reference points, generator version |
| `ground_truth.csv` | `t_ns,px,py,pz,vx,vy,vz,qx,qy,qz,qw,bax,bay,baz,bgx,bgy,bgz`, for evaluation only |
| `truth_metadata.json` | True parameters and corruption labels, for the generator and evaluator only |

Rules:
- [ ] Save enough numeric precision for repeatable results.
- [ ] Generate data and save it **before** running any estimator.
- [ ] Generated data goes to `data/generated/` and isn't committed.
- [ ] Estimator assumptions (lever arms, covariance multipliers) live in the estimator config.
- [ ] The filter **never** reads the true calibration from the truth files.

**Done when:** `generate` writes all seven files.

### Step A14 — Build the validator

Reject:
- [ ] Non-finite values (NaN, inf).
- [ ] Wrong dimensions.
- [ ] Invalid covariance.
- [ ] Ambiguous quaternion ordering.
- [ ] Negative time steps.
- [ ] Duplicate IMU timestamps.
- [ ] Missing required metadata.

Allow:
- [ ] Different sensors sharing a timestamp.

Report:
- [ ] External measurements outside the IMU time range. Never extrapolate silently.

**Done when:** `validate --data ...` passes good data and rejects each kind of bad case in tests.

### Step A15 — Create the initial prior

- Supply the prior externally. For generated data, it is the state at `t = 0` plus a sampled perturbation.
- This is **ground-truth-assisted initialization**, and the report must say so.

Standard deviations:

| State | σ per axis |
|---|---|
| Position | 1 m |
| Velocity | 0.3 m/s |
| Attitude (local error) | 2° (convert to radians) |
| Accel bias (15-state) | 0.1 m/s² |
| Gyro bias (15-state) | 0.005 rad/s |

Rules:
- [ ] Bias prior mean = 0.
- [ ] Additive states: `estimate = truth − sampled_error`.
- [ ] Attitude: `q_est = q_true ⊗ quat(−ε)`.
- [ ] Use the same position, velocity and attitude prior, with a matching leading covariance block, for every estimator.
- [ ] The estimator reads only this one prior, never later truth samples.
- [ ] For recorded data, accept a supplied prior and explain where it came from.
- [ ] Automatic global heading initialization is out of scope. A stationary accelerometer can't determine yaw, and can't separate tilt from every accelerometer-bias component.

**Done when:** `initial_prior.json` is written, and a ground-truth plot is produced.

---

## Phase B — Inertial propagation

**Phase exit check:** the zero-noise sanity checks pass, and the drift is explained without changing the truth.

### Step B1 — Define the state

Nominal state (16 numbers): `x = (p_W, v_W, q_WB, b_a, b_g)`.

Error state (15 numbers): `δx = [δp, δv, δθ, δb_a, δb_g]`.

| Slice | Meaning |
|---|---|
| `0:3` | Position error in `W` |
| `3:6` | Velocity error in `W` |
| `6:9` | Local rotation error (rad) |
| `9:12` | Accelerometer bias error |
| `12:15` | Gyroscope bias error |

Rules:
- [ ] `P` is 15×15. Never use a 16×16 quaternion covariance.
- [ ] The 9-state version keeps only the first three blocks: 10 nominal numbers, 9×9 `P`, zero biases assumed.
- [ ] In biased experiments the 9-state filter **still** assumes zero bias. That mismatch is the point of the comparison.
- [ ] Use a state dataclass and centralize the slices. No anonymous long vectors.

**Done when:** the state class and slice constants exist and are tested.

### Step B2 — Write nominal prediction

```
f̂ = f_m − b̂_a
ω̂ = ω_m − b̂_g

R_mid = R_k · Exp(ω̂ dt / 2)
a_W   = R_mid f̂ + g_W

p[k+1] = p[k] + v[k] dt + ½ a_W dt²
v[k+1] = v[k] + a_W dt
q[k+1] = q[k] ⊗ quat(ω̂ dt)
```

Rules:
- [ ] Biases stay constant during prediction; only their uncertainty grows.
- [ ] Normalize the quaternion after propagation and after correction.
- [ ] Document that this is numerical integration of held IMU readings, not an exact solution.
- [ ] `predict` never reads ground truth.

**Done when:** a stationary body stays still and a constant-velocity body moves straight.

### Step B3 — Write the timestamp scheduler

IMU convention:
- [ ] The reading at `t_k` is held over `[t_k, t_k+1)`.
- [ ] The generator evaluates force and rate at the interval midpoint and adds interval-average noise. Document that it's an interval measurement, not one available before the interval ends.
- [ ] Write `N+1` IMU timestamps for `N` intervals. The last row only marks the end of coverage and is never propagated.
- [ ] Test that the final interval is processed exactly once.

Time handling:
- [ ] Schedule with integer nanoseconds.
- [ ] Compute `dt = (t2 − t1) × 1e-9` only by subtracting timestamps.
- [ ] Never compare floating-point timestamps for equality.

For each external measurement:
1. [ ] Propagate to its exact timestamp.
2. [ ] Apply the update there.
3. [ ] Finish the rest of the IMU interval with the same raw reading and the current bias estimate.
4. [ ] Process every queued measurement, even when several fall inside one IMU interval.

At an IMU boundary:
1. [ ] Finish propagation with the previous interval.
2. [ ] Apply all measurements at that time: GNSS first, then LiDAR.
3. [ ] Switch to the next IMU interval.
4. [ ] Log the state after all same-time updates.

Hard rules:
- [ ] Never use a future measurement to correct an earlier state.
- [ ] The loader may sort external records by measurement time (this assumes no delivery delay; document that).
- [ ] Reject shuffled IMU input unless an explicit normalization option is used.
- [ ] Don't claim support for delayed or out-of-order arrival, rollback or buffering.
- [ ] Fail clearly on missing IMU coverage or large gaps. Never hold an old reading across an outage.
- [ ] A sensor dropout removes only that sensor's events. IMU propagation never stops.
- [ ] Never reuse the last GNSS/LiDAR reading on every IMU step.
- [ ] Stop at the last IMU-supported interval. Never extrapolate.
- [ ] Disclose that holding one noisy IMU reading across several off-grid updates creates correlations the model ignores.

**Done when:** tests pass for on-grid, off-grid (for example GNSS at 0.035 s between 0.03 and 0.04 s), same-time and several-per-interval updates, plus dropout and missing-coverage cases.

### Step B4 — Build the IMU-only baseline

- [ ] `imu_only` integrates with zero bias estimates and no corrections.
- [ ] It uses **the same integrator** as the filters, so a better integrator can't explain any improvement.

**Done when:** `run --mode imu_only` produces output.

### Step B5 — Check the zero-noise cases

- [ ] Stationary, 10 s: position drift under `1e-6 m`.
- [ ] Tilted stationary: no drift (gravity cancels correctly).
- [ ] Straight constant velocity, 10 s: drift under `1e-6 m`.
- [ ] Curved motion: the error shrinks when the step is halved. (Don't demand straight-line accuracy here.)
- [ ] Quaternion norm error stays under `1e-10`.

**Done when:** all of these pass.

### Step B6 — Show the drift

- [ ] Run IMU-only on the noisy and biased figure-eight.
- [ ] Plot how the error grows.
- [ ] Explain the drift without touching the truth.

**Done when:** the drift plot exists and is explained.

---

## Phase C — Nine-state fusion

**Phase exit check:** the zero-bias case works on the saved dataset and passes the timing checks.

### Step C1 — Build the error dynamics `A`

Non-zero blocks only:

```
A_pv   =  I
A_vθ   = −R [f̂]×
A_vba  = −R
A_θθ   = −[ω̂]×
A_θbg  = −I
```

- [ ] Use the midpoint rotation `R_mid` when freezing `A` over a substep.
- [ ] Bias rows are zero (random walk).
- [ ] The 9-state version uses the top-left 9×9 block.

### Step C2 — Build the noise mapping `L` and `Qc`

Noise order: accelerometer, gyro, accel-bias drive, gyro-bias drive (15×12).

```
L_v,na   = −R
L_θ,ng   = −I
L_ba,nba =  I
L_bg,nbg =  I

Qc = diag(σ_a² I, σ_g² I, σ_ba² I, σ_bg² I)
W  = L Qc Lᵀ
```

- [ ] The 9-state version uses only the accelerometer and gyro channels.

### Step C3 — Write the fast discretization

Substep `h ≤ 0.01 s`:

```
Φ(h) = I + A h + ½ A² h²
Qd  ≈ h/6 · [ W + 4 Φ(h/2) W Φ(h/2)ᵀ + Φ(h) W Φ(h)ᵀ ]
P⁻   = Φ P Φᵀ + Qd
```

- [ ] This is Simpson's rule on the covariance integral. Call it an approximation, never exact.

### Step C4 — Write the Van Loan reference

```
E  = expm( [[A, W], [0, −Aᵀ]] · h )
Φ  = E₁₁
Qd = E₁₂ Φᵀ
```

- [ ] Use `scipy.linalg.expm`, behind a `van_loan` option.
- [ ] Use this one block convention consistently.
- [ ] Compare the fast method with it, including the tilted-stationary and turning cases.
- [ ] If the fast method fails, fix it or shrink the substep. **Never** loosen the tolerance just to pass.
- [ ] Choose and document a scaled matrix-norm tolerance, and show convergence with smaller steps.

**Done when:** the discretization test passes.

### Step C5 — Write the position update

```
ẑ = p̂ + R̂ ℓ
r = z − ẑ
H = [ I   0   −R̂ [ℓ]×   0   0 ]       (9-state: first three blocks)

S  = H P⁻ Hᵀ + R_s
K  = P⁻ Hᵀ S⁻¹                         (use a solve, never an explicit inverse)
δx̂ = K r
```

- [ ] With `ℓ = 0`, `H` reduces to a direct position observation.
- [ ] Never add a body-fixed lever arm as a fixed world-frame offset; its world direction turns with the vehicle.
- [ ] `update_position` never reads hidden generator parameters, future measurements or earlier experiment results.

**Done when:** a finite-difference test of `h(x ⊞ δx)` matches `H` to about `1e-6`, with a non-zero orientation and lever arm.

### Step C6 — Inject the correction

- [ ] Add the correction to position, velocity and both biases.
- [ ] Apply attitude on the right: `q ← q ⊗ quat(δθ)`.
- [ ] Normalize the quaternion.

### Step C7 — Joseph update, then reset

```
P_J = (I − K H) P⁻ (I − K H)ᵀ + K R_s Kᵀ
P⁺  = G P_J Gᵀ
```

`G` is the identity, except its attitude block is `J_r(φ)` with `φ = δθ`:

```
J_r(φ) = I − (1 − cos α)/α² [φ]× + (α − sin α)/α³ [φ]×²,   α = |φ|
```

- [ ] Near zero, use the series `I − ½[φ]× + ⅙[φ]×²`.
- [ ] Reset the error mean to zero, keep the corrected nominal state, and normalize the quaternion.
- [ ] The Joseph form alone does **not** reset the coordinates. Both steps are needed.

**Done when:** a finite-difference test of `Log(Exp(−φ) Exp(φ + ε))` matches `J_r(φ)` to about `1e-6` (step about `1e-6 rad`), for both near-zero and moderate `φ`, staying away from the π branch.

### Step C8 — Apply the numerical safety rules

- [ ] Symmetrize: `P = (P + Pᵀ) / 2` after both prediction and correction.
- [ ] Use linear or Cholesky solves, never explicit inverses.
- [ ] Check for finite values and positive (semi-)definiteness at regular intervals and in tests.
- [ ] Tiny negative eigenvalues at round-off level are diagnosed against a documented scale-aware tolerance.
- [ ] Never hide instability by repeatedly clipping eigenvalues.
- [ ] If factoring `S` fails, report the sensor and timestamp. Any jitter added must be bounded and logged.

### Step C9 — Check the dynamics Jacobian

- [ ] Finite-difference the nominal propagation using the same right-local error.
- [ ] Use a step-size convergence study, not exact agreement (`Φ` freezes the dynamics).

### Step C10 — Run experiment E1

- [ ] Zero-bias noisy motion: `imu_only` vs `eskf9_gnss` vs `eskf9_all`.
- [ ] Plot the innovations, errors and uncertainty.
- [ ] Measure whether the corrections reduce drift.

**Done when:** E1 runs on the saved dataset and the timing tests pass.

---

## Phase D — Fifteen-state estimator

**Phase exit check:** full runs are stable, the covariance behaves correctly, and the bias-estimation effects are reported.

### Step D1 — Add the bias states

- [ ] Extend to 15 states with the bias cross-covariances.
- [ ] Add the bias random-walk noise to `Qc`.
- [ ] Share code with the 9-state filter, but keep their different assumptions clear.

### Step D2 — Define the five setups

All five run on **identical** data and priors:

| Name | Setup |
|---|---|
| `imu_only` | IMU only, zero bias, no corrections |
| `eskf9_gnss` | 9-state + GNSS |
| `eskf9_all` | 9-state + GNSS + LiDAR |
| `eskf15_gnss` | 15-state + GNSS |
| `eskf15_all` | 15-state + GNSS + LiDAR (final) |

### Step D3 — Run the nominal biased case and E2

- [ ] Run E2: `eskf9_all` vs `eskf15_all` on biased motion.
- [ ] Compare the errors **and** the innovations.

### Step D4 — Check bias information

- [ ] Show that turning and changing acceleration help estimate the biases.
- [ ] Don't claim every bias is always observable.

**Done when:** the 15-state runs are stable and the bias plots exist.

---

## Phase E — Robustness experiments

**Phase exit check:** the experiments are reproducible, and failures and trade-offs show up in the saved metrics.

### Step E1 — Add outlier gating

```
NIS = rᵀ S⁻¹ r
threshold = chi2.ppf(0.9973, df=3)
```

- [ ] Compute NIS **before** changing the state.
- [ ] Get the threshold from SciPy. Don't use `9`.
- [ ] Make the threshold configurable, and allow gating to be turned off.
- [ ] Log every innovation, NIS value, threshold and accept/reject decision before gating.
- [ ] A rejected measurement changes neither the state nor the covariance.
- [ ] Repeated rejections are a diagnostic, not proof the sensor is broken; an inaccurate filter can reject good data.

**Done when:** a test shows a rejection leaves the state and `P` unchanged while the statistics are still logged.

### Step E2 — Derive the experiment variants

- [ ] Generate each base dataset **once**.
- [ ] Derive the dropout, outlier and calibration variants from it reproducibly.
- [ ] Keep the true sensor noise separate from the covariance the filter assumes.
- [ ] Dropout intervals are half-open: `[40, 60)` s.

### Step E3 — Run E0 (noise-free fixtures)

- [ ] Stationary, tilted-stationary, straight-line and turning fixtures.
- [ ] Check gravity cancellation, quaternion behaviour, integration convergence and the absence of unexplained drift.

### Step E4 — Run E3 (GNSS-only filter loses GNSS)

- [ ] `eskf15_gnss` with GNSS missing from 40 to 60 s.
- [ ] This is pure inertial behaviour, because the filter has lost its only aid.

### Step E5 — Run E4 (GNSS lost, LiDAR still on)

- [ ] `eskf15_all` with GNSS missing from 40 to 60 s.
- [ ] Measure how LiDAR keeps aiding the filter. Don't assume there will be large drift.

### Step E6 — Run E5 (both sensors lost)

- [ ] `eskf15_all` with both streams missing from 40 to 60 s.
- [ ] Measure error and uncertainty growth, and recovery afterwards.
- [ ] Report whether gating rejects the returning measurements.
- [ ] **Never** secretly reset to truth or inflate the covariance to make it look good.
- [ ] Treat E4 and E5 as different experiments: one still has aiding, the other has none.

### Step E7 — Run E6 (wrong GNSS trust)

- [ ] Same biased data; scale the assumed GNSS covariance by 0.1, 1 and 10.
- [ ] Don't regenerate the noise.
- [ ] Show the effects of over- and under-confidence.

### Step E8 — Run E7 (wrong lever arm)

- [ ] True LiDAR arm `[0.5, 0, 1.2]`; assumed arm correct vs `[0.7, 0.1, 1.2]`.
- [ ] Compare the signed mean errors as well as the RMSE.
- [ ] This tests a lever arm, not a rotation (boresight). Don't claim to have tested rotational calibration.

### Step E9 — Run E8 (outliers)

- [ ] Add `[+20, −15, +8] m` to 5% of GNSS readings after 10 s.
- [ ] Pick them with the outlier random stream. Labels go to the evaluator only.
- [ ] Keep all other data identical.
- [ ] Compare gating off and on, for both trajectory metrics and gate decisions.
- [ ] Don't conclude that gating always helps.

### Step E10 — Build the evaluation

- [ ] Evaluate at common logged timestamps. For synthetic data, prefer the analytical truth.
- [ ] Recorded data: interpolate position and velocity, SLERP quaternions, respect overlap limits, never extrapolate.
- [ ] No nearest-neighbour matching without an explicit tolerance.
- [ ] Error = **truth − estimate** for additive states.
- [ ] Attitude error = `Log(R_estᵀ R_true)`.
- [ ] Angle error: `q_e = q_true⁻¹ ⊗ q̂`, `θ = 2 acos(clip(|q_e,w|, 0, 1))`, or the rotation-vector norm if that's more stable.
- [ ] **Never** subtract quaternion components.
- [ ] Compare each external measurement with the truth at **its own** reference point, not the body origin.
- [ ] Any interpolated position-only curve must be labelled as using future samples, not called causal, and not used with mismatched timestamps.

### Step E11 — Compute the metrics

- [ ] Position RMSE: 3-D, horizontal and per axis (m). `RMSE = √(1/N Σ |p_true − p̂|²)`.
- [ ] Velocity RMSE: 3-D and per axis (m/s).
- [ ] Orientation RMSE and maximum error (degrees).
- [ ] Final and maximum position error.
- [ ] Bias RMSE where truth exists, with the observability limits stated.
- [ ] Per-sensor counts: total, accepted and rejected, plus pre-gate NIS statistics.
- [ ] Outlier recall and false-rejection rate (when labels exist).
- [ ] Error and position-covariance trace before, during and after dropout.
- [ ] Wall time, peak memory (if measurable) and output size.
- [ ] Report both the whole-run figures and the figures after the first 10 s. State the initialization effects.
- [ ] Smoke runs: avoid an empty 10-second burn-in window.

### Step E12 — Interpret uncertainty correctly

- [ ] Plot each axis's error against `±3√P_ii` in matching coordinates.
- [ ] Orientation covariance is in rad² in the local frame. Don't compare it with world Euler differences.
- [ ] Coverage percentages are diagnostics, not proof of consistency.
- [ ] NIS of accepted measurements only is truncated, so it isn't a clean chi-square sample.
- [ ] A mean NIS near 3 is a useful sign, not a strict requirement.
- [ ] Covariance doesn't have to grow monotonically during dropout. Report the trend and the endpoints.
- [ ] Some attitude and bias directions can be weakly observable. Don't promise convergence.

### Step E13 — Make the figures

1. [ ] XY plot: truth, IMU-only and selected fused tracks.
2. [ ] Position error over time, with the dropout intervals marked.
3. [ ] Per-axis position error with covariance bounds.
4. [ ] Velocity and orientation error.
5. [ ] Estimated vs true biases (15-state).
6. [ ] NIS per sensor, with the threshold and rejections marked.
7. [ ] Calibration-error comparison, and a dropout close-up.
8. [ ] Experiment summary with values and units.

Rules:
- [ ] Readable PNGs, plus the numeric data behind each plot.
- [ ] Clear legends, axes, units and captions.
- [ ] Don't rely on colour alone; use line styles or markers too.
- [ ] Don't let the diverged IMU-only track hide the fused tracks; use an inset or a separate plot.

### Step E14 — Save the outputs

Per run:
- [ ] `estimates.csv`
- [ ] Covariance history as compressed `.npz`, loaded with `allow_pickle=False`.
- [ ] `innovations.csv`
- [ ] `metrics.json`
- [ ] `config_resolved.yaml`
- [ ] `run_metadata.json`

Per suite:
- [ ] `comparison.csv`
- [ ] `report.md`, linking the figures.

Storage rules:
- [ ] Log states and covariance diagonals at 100 Hz.
- [ ] Log full `P` at 10 Hz by default; keep the full matrix in memory while running.
- [ ] A diagnostic option may log full `P` at full rate, for short runs only.
- [ ] Record the actual logging timestamps.
- [ ] Don't copy datasets into every run folder.
- [ ] Run experiments one at a time; don't hold every run in memory.
- [ ] Use run-specific folders. Never overwrite silently: add an `--overwrite` flag or a new run ID.

### Step E15 — Run the suite and verification

- [ ] `suite` runs every experiment with seed 7.
- [ ] `verify` runs the selected comparisons for seeds 7, 23 and 42, plus the deterministic tests. It must **not** start a large Monte Carlo study.
- [ ] Optional: `monte-carlo --runs 20`, only when explicitly asked.
- [ ] Diagnose, fix and rerun any failures.

Engineering targets (seeds 7, 23, 42):
- [ ] `eskf15_all` position RMSE after initialization below **1.0 m** (biased case).
- [ ] The full aided filter beats IMU-only on position RMSE over the whole run.
- [ ] Outlier recall of at least **90%** with gating on; false rejections reported separately.
- [ ] If E2 doesn't improve, investigate and report honestly.

If a target fails:
- [ ] Check units, frames, initialization, timing, Jacobians, noise scaling and numerical errors first.
- [ ] Document any change to the simulator, thresholds or evaluation window, and rerun the affected comparisons.
- [ ] Never cherry-pick a seed.

Exit codes:
- [ ] A failed deterministic check or an incomplete run returns non-zero.
- [ ] Unmet targets appear clearly in the output and in a machine-readable status file.

**Done when:** the suite and verify both complete, and the status is saved.

---

## Phase F — Documentation and handover

**Phase exit check:** someone else can run the project and explain its assumptions without filling in missing steps.

### Step F1 — Complete the test list

Every one of these must pass:

1. [ ] Quaternion and frame conventions.
2. [ ] Specific force (level and tilted).
3. [ ] Truth derivatives vs numerical differences.
4. [ ] Lever-arm Jacobian (finite differences).
5. [ ] Dynamics (convergence study).
6. [ ] Reset Jacobian (finite differences).
7. [ ] Fast vs Van Loan discretization.
8. [ ] Timestamp scheduler cases.
9. [ ] Covariance finite, symmetric and PSD over full runs.
10. [ ] Gating leaves state and `P` unchanged on rejection.
11. [ ] **No truth leakage:** changing or removing `ground_truth.csv` doesn't change the estimates.
12. [ ] **Reproducibility:** the same inputs give the same outputs within tolerance (bitwise equality across platforms isn't required).

Write tests that catch real failures, not tests that just repeat a constant.

### Step F2 — Build the `demo` command

- [ ] Generates data, runs the suite, evaluates, makes the figures and writes the report.
- [ ] No manual notebook steps.
- [ ] Prints a short pass/fail summary and the report path.

### Step F3 — Finish the CLI

```bash
python -m vehicle_localization doctor
python -m vehicle_localization generate --config configs/default.yaml --output data/generated/default
python -m vehicle_localization validate --data data/generated/default
python -m vehicle_localization run --data data/generated/default --mode eskf15_all --output results/baseline
python -m vehicle_localization suite --config configs/default.yaml --seeds 7 --output results/suite
python -m vehicle_localization verify --config configs/default.yaml --seeds 7 23 42 --output results/verification
python -m vehicle_localization demo --config configs/default.yaml --output results/demo
python -m vehicle_localization monte-carlo --runs 20 --output results/monte_carlo   # optional
python -m pytest -q
```

### Step F4 — Write the generated report

It must say:
- [ ] What was generated.
- [ ] What was measured.
- [ ] Which checks passed or failed.
- [ ] Which conclusions the results support.
- [ ] That initialization is ground-truth-assisted.

All numbers come from saved metrics, never typed in by hand.

### Step F5 — Write `docs/math.md`

- [ ] Conventions, derivations and references.

### Step F6 — Write `docs/data_format.md`

- [ ] Exact units, timing rules (including the N+1 IMU-row rule), schemas and examples.

### Step F7 — Write `docs/learning_guide.md`

Explain, in implementation order:

1. [ ] What the state is, and why a quaternion has 4 numbers but 3 error coordinates.
2. [ ] Why accelerometer readings need both rotation and gravity handling.
3. [ ] What covariance means here.
4. [ ] How a position measurement improves velocity and attitude through cross-covariance.
5. [ ] Why quaternion correction needs a consistent local error and a covariance reset.
6. [ ] Why constant IMU biases cause growing errors.
7. [ ] Why losing GNSS matters differently with and without LiDAR.
8. [ ] How lever arms, timing and assumed covariance affect the results.
9. [ ] What synthetic data leaves out: multipath, real point-cloud registration, time-correlated errors, imperfect synchronization, temperature effects, complex motion.
10. [ ] How to reproduce one result and trace it to the code and equations.

Also:
- [ ] A vocabulary table: IMU, GNSS, nominal state, error state, innovation, Jacobian, covariance, lever arm, observability.
- [ ] Code comments that explain reasoning and units, not every obvious line.

### Step F8 — Write `docs/recorded_data.md`

- [ ] How to supply a small recorded sequence in the CSV/JSON format.
- [ ] Without ground truth: accuracy metrics switch off cleanly, but estimation, innovation logging and timing checks still run.
- [ ] GNSS latitude/longitude/altitude must be converted to the local metric frame first. Never put degrees into the state.
- [ ] Any future geodetic adapter must state its datum, local origin, altitude convention and covariance conversion.
- [ ] Limits of any future CARLA adapter (see Step X1).

### Step F9 — Update the README with real results

- [ ] Installation and commands for Windows and Linux.
- [ ] The actual measured results and experiments.
- [ ] Never imply real-vehicle deployment, official course completion, radar-camera fusion or real LiDAR processing.

### Step F10 — Run the final checks and hand over

Before declaring success, check for each of these failures:

- [ ] Truth leaking into prediction, updates, calibration or resets.
- [ ] The same sign error in both the simulator and the estimator.
- [ ] Specific force treated as world acceleration, or gravity counted twice.
- [ ] Mixed `xyzw` and `wxyz` quaternions at file boundaries.
- [ ] Local and global attitude errors mixed between Jacobians.
- [ ] A Joseph update without the tangent-space reset.
- [ ] ms/ns treated as seconds, or degrees treated as radians.
- [ ] Noise density confused with per-sample standard deviation.
- [ ] Measurements fused before their timestamps, or reused.
- [ ] A body-fixed lever arm treated as a world-fixed offset.
- [ ] GNSS dropout described as "no aiding" while LiDAR is still on.
- [ ] Different noise realizations in supposedly controlled comparisons.
- [ ] Rejected measurements missing from the NIS statistics.
- [ ] Simulator truth used as LiDAR output without a label.
- [ ] Covariance clipped or inflated to hide errors.
- [ ] Good RMSE achieved by quietly dropping the initialization or outage periods.
- [ ] Performance numbers copied from another machine or guessed from the specs.

Then:
- [ ] Run `demo` from a clean `results/` folder.
- [ ] Record the real runtime, memory and environment (non-sensitive host description).
- [ ] Write `requirements-tested.txt`.

Handover checklist:
- [ ] Working CPU-only package with project-local install instructions.
- [ ] Default and smoke datasets, with provenance, frames and units.
- [ ] IMU-only, 9-state and 15-state estimators.
- [ ] Exact-timestamp updates, lever arms, Joseph correction and reset.
- [ ] Deterministic tests, with their saved status.
- [ ] E0–E8 results and multi-seed verification.
- [ ] Plots, raw outputs, metrics and the report.
- [ ] Actual runtime and memory measurements.
- [ ] Beginner docs and the recorded-data format.
- [ ] A list of failed targets, unsupported cases and unrun checks.
- [ ] One command that reproduces the demo.

Final message: what was built, how it was tested, where the report is, and any unresolved issue. A result with an unresolved maths defect is **not** complete, even if the plot looks good.

---

## Later — Optional extensions (not in the required scope)

Do these only after everything above is finished, and only when separately requested.

### Step X1 — CARLA import

- Don't install or launch CARLA by default. The 4 GB of VRAM is a reason not to make it a prerequisite, but it doesn't prove CARLA can't run.
- A future adapter could read a small, already-recorded CARLA sequence.
- CARLA uses `x` forward, `y` **right**, `z` up, which differs from this project.
- CARLA GNSS gives latitude/longitude; CARLA LiDAR gives detections, not localization.
- Before using CARLA data:
  - [ ] Verify the handedness changes for positions, orientations, accelerations and angular rates. Angular rates are axial vectors, so don't just copy a position-axis sign flip.
  - [ ] Check accelerometer and gravity behaviour with a stationary test in that exact CARLA version.
  - [ ] Convert GNSS into the world frame.
  - [ ] Use genuine localizer outputs, or clearly label surrogates made from simulator truth.
  - [ ] Never call ground truth plus noise "LiDAR odometry".
- Never auto-download course pickles or run untrusted serialized objects. A trusted dataset can be converted separately, with its provenance recorded.

### Step X2 — Radar ego-velocity aiding

- This connects most directly to the user's radar experience.
- It's a velocity measurement, with its own model and covariance.
- Radar at the body origin: `z_v = R_WBᵀ v_W + noise`.
- A displaced radar adds `ω_B × ℓ_BR`, and needs the radar-to-body rotation.
- It needs explicit handling of Doppler signs, static-object selection, outlier rejection and gyro correlations.
- Never feed raw Doppler into the position update.

### Step X3 — Camera/radar odometry and mapping

- Relative odometry links two poses; it isn't an absolute position.
- Accumulating relative estimates creates correlations a naive update ignores.
- This may need state augmentation, pose cloning or a factor graph.
- Camera scale, calibration, time alignment and mapping are separate tasks.

### Step X4 — Learning-based fusion

- Only if a real task and dataset justify it.
- Keep it within 4 GB of VRAM, or reassess the hardware.
- No GPU framework in the required project.

---

## Appendix — References

- [S1] University of Toronto / Coursera — [State Estimation and Localization for Self-Driving Cars](https://www.coursera.org/learn/state-estimation-localization-self-driving-cars)
- [S2] J. Solà — [Quaternion kinematics for the error-state Kalman filter](https://arxiv.org/abs/1711.02508)
- [S3] SciPy — [`Rotation.from_quat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html)
- [S4] SciPy — [`scipy.linalg.expm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html)
- [S5] SciPy — [`scipy.stats.chi2`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html)
- [S6] CARLA — [Sensors reference](https://carla.readthedocs.io/en/latest/ref_sensors/)

The hardware budgets, trajectory, structure, experiment settings and targets are this project's own design choices, not copied course material.
