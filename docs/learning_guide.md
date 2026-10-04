# Learning guide

This guide walks through the project in the order the code was built. Each section answers one
question, points to the code, and, where useful, to a measured result.

Numbers quoted here come from the seed-7 demo run on a remote cloud machine. Your own
`results/demo/report.md` is the source of truth for your machine.

---

## Vocabulary

| Term | Meaning in this project |
|---|---|
| **IMU** | Inertial measurement unit: an accelerometer (specific force, m/s²) and a gyroscope (angular rate, rad/s), here at 100 Hz. |
| **GNSS** | Satellite positioning. Here a simulated antenna position at 5 Hz with about 1 m (horizontal) and 2 m (vertical) noise. |
| **LiDAR-localizer position** | The output of an external localization system, simulated here as truth plus 0.15–0.2 m noise at 10 Hz. Not raw LiDAR data. |
| **Nominal state** | The filter's best guess: `p, v, q, b_a, b_g` (16 numbers). |
| **Error state** | The small difference between the truth and the nominal state: `δp, δv, δθ, δb_a, δb_g` (15 numbers). |
| **Covariance `P`** | How uncertain the filter believes each error-state component is, and how the components are correlated. |
| **Innovation** | The measurement minus what the filter predicted it would be: `r = z − ẑ`. |
| **NIS** | Normalized innovation squared, `rᵀ S⁻¹ r`. About 3 on average for a 3-D measurement when the filter's models and uncertainties match the data. A much larger value signals an inconsistency (too-small measurement or process noise, unmodelled bias or calibration, timing, outliers) but doesn't say which. |
| **Jacobian** | The matrix of first derivatives (`A`, `H`, `J_r`) that linearizes a nonlinear function around the current estimate. |
| **Lever arm** | The fixed vector, in the body frame, from the body origin to a sensor's reference point. |
| **Observability** | Whether the measurements carry enough information to determine a state component. Some only become observable when the vehicle turns or accelerates. |

---

## 1. What the state is, and why 4 numbers become 3

Orientation is stored as a unit quaternion (4 numbers). But a rotation has only 3 degrees of
freedom, and the 4 numbers are tied together by `|q| = 1`.

A covariance over 4 constrained numbers would be singular and meaningless. So the filter keeps:

- the **nominal** quaternion, which can be large and is handled exactly;
- a **3-number error** `δθ`, which is small and lives in the body frame
  (`R_true = R_est Exp(δθ)`).

The covariance is over `δθ`, so `P` is 15×15, not 16×16.

Code: `state.py` (the slices `POS, VEL, ATT, BA, BG`) and `rotations.py`.

---

## 2. Why accelerometer readings need rotation and gravity

An accelerometer measures **specific force**, `f_B = R_WBᵀ (a_W − g_W)`. A phone lying still on a
table reads +9.81 m/s² upward, not zero.

To recover motion you must do two things:

1. Rotate `f_B` into the world frame.
2. Add gravity back: `a_W = R f_B + g_W`.

Get either step wrong (wrong frame, or gravity counted twice) and a stationary vehicle "falls" at
9.81 m/s².

- The E0 fixtures check this: a stationary body, level or tilted, stays put to within 1e-12 m over 10 s.
- Code: `sensors.specific_force` (simulator) and `eskf._predict_step` (filter). They are written
  separately, so a sign error can't cancel itself out.

---

## 3. What covariance means here

`P` is the filter's own estimate of how wrong it is:

- `√P[0,0]` is its 1σ guess of the x-position error in metres.
- Off-diagonal entries say how errors move together. For example, a heading error and a sideways
  position error grow together while the vehicle drives.

The figures plot the real error against `±3√P_ii`. If the filter is honest, the error stays mostly
inside the bounds.

In E2 the 15-state filter kept 99.9% of position samples inside the bounds. The 9-state filter,
which wrongly assumes zero bias, kept only 78% on its worst axis. Its covariance was too small
because it ignored the bias. See `fig03_position_axes_bounds.png`.

Code: `eskf._predict_step` (`P = Φ P Φᵀ + Qd`) and `evaluation._coverage`.

---

## 4. How a position measurement fixes velocity and attitude

GNSS and LiDAR measure only position, yet the filter also corrects velocity, attitude and biases.
This works through **cross-covariance**:

- During prediction, `A` couples the error blocks. Position error grows from velocity error
  (`A_pv = I`). Velocity error grows from attitude error, because a tilt error leaks gravity
  sideways (`A_vθ = −R[f]×`), and from accelerometer bias error (`A_vba = −R`).
- So `P` builds up correlations between position and those other states.
- The gain `K = P Hᵀ S⁻¹` then spreads a position innovation into every correlated state.

Roll and pitch become well determined this way: about 0.13° 1σ by the end of E2.

Code: `discretization.error_dynamics` and `eskf.update_position`.

---

## 5. Why quaternion correction needs a consistent error and a reset

Two things must match:

- **Same error definition everywhere.** `A`, `H` and the injection must all use
  `R_true = R Exp(δθ)` (right, body-local). If one Jacobian used the world-frame (left) error and
  another the body-frame error, the filter would be quietly wrong. The tests check `H` and the
  dynamics against finite differences built on this same definition.
- **A covariance reset.** After injecting `q ← q ⊗ quat(δθ̂)`, the error is measured from a
  different reference rotation. The covariance must be moved into the new coordinates with
  `P⁺ = G P Gᵀ`, where `G` contains `J_r(δθ̂)`. The Joseph update does not do this. A Monte Carlo
  test shows that skipping the reset gives the wrong covariance after a large correction.

Code: `eskf.update_position` (the lines after `# Reset`), `rotations.right_jacobian`, and
`tests/test_eskf_math.py`.

---

## 6. Why constant IMU biases cause growing errors

A constant accelerometer bias `b` integrates twice, so position error grows like `½ b t²`. A gyro
bias integrates into an attitude error that grows like `b_g t`. That attitude error tilts gravity
sideways, so position error grows even faster (about `g b_g t³ / 6`).

The E0 drift decomposition (IMU only, 120 s) shows the size of each effect:

| Case | Error after 120 s |
|---|---|
| Exact start, no noise | 0.0002 m |
| White noise only | about 436 m |
| Biases only | about 453 m |
| Perturbed initial prior only | about 2270 m (the sampled initial tilt error of about 1.8° leaks about 0.3 m/s² of gravity sideways) |

In E2 the 15-state filter estimated the biases and reached 0.096 m position RMSE, against
0.162 m for the 9-state filter on the same data. See `fig05_biases.png`.

---

## 7. Why losing GNSS matters differently with and without LiDAR

| Experiment | What is lost | Max position error during the 20 s outage |
|---|---|---|
| E3 | GNSS, which is the only aid | 15 m (pure inertial coasting) |
| E4 | GNSS, while LiDAR keeps arriving | 0.19 m (barely worse than normal) |
| E5 | Both GNSS and LiDAR | 23 m; position covariance trace grew from 0.011 to 389 m² |

In E5 the covariance grew along with the error, so when the measurements came back none were
rejected and the error dropped back to about 0.15 m within 5 s.

"GNSS dropout" therefore does **not** mean "no aiding" when another position source is still
running. See `fig02_position_error_dropouts.png`.

---

## 8. How lever arms, timing and assumed covariance affect results

### Lever arm (E7)

The filter was given a LiDAR lever arm 0.22 m off.

- Position RMSE rose from 0.096 to 0.249 m.
- LiDAR NIS barely moved (2.98 to 2.99), because the filter simply shifted its position by
  `R · Δℓ` to explain the data.
- In world axes the error averages out as the heading turns. In the body frame it is a constant
  `[0.20, 0.10, 0.00] m`, which is the lever-arm error itself. See `fig07`.

### Timing

A measurement applied at the wrong time is like a measurement with extra error proportional to the
vehicle's speed. That's why the scheduler propagates to the **exact** timestamp of every
measurement (see `tests/test_propagation_and_scheduler.py`).

### Assumed covariance (E6)

- Telling the GNSS-only filter that GNSS is 10× more accurate than it is made NIS jump to about 74.
  The gate then rejected 581 of 600 GNSS fixes, and the filter coasted and diverged (205 m RMSE).
- With LiDAR also running, the same mistake was harmless.
- Assuming 10× *less* accuracy only made the GNSS-only filter slightly worse (0.68 to 0.88 m).

---

### Degraded GNSS, and fusing vs switching (E9, E10)

Real GNSS rarely just switches off. Under a viaduct it gets **worse**, and the receiver often
doesn't know. E9 makes GNSS 5× noisier and adds a drifting bias for 20 s:

- **GNSS-only filter:** the gate correctly rejects the bad fixes, so the filter coasts on the IMU
  and drifts. When GNSS recovers, the good fixes disagree so much with the drifted estimate that
  they are rejected too ("gate lock-out"), and the error peaks at 93 m. Gating protects against
  short outliers, but with no second source it can make a long degradation worse.
- **GNSS + LiDAR fused:** the gate drops the bad GNSS while LiDAR keeps correcting. The error stays
  below 0.2 m.

E10 tests the design choice from Wang et al. (2024): use **one** source at a time instead of
fusing both. The switching filter uses GNSS while its NIS passes a chi-square test, and LiDAR
otherwise. It handles the bad section, but it ignores LiDAR the rest of the time, so its RMSE is
0.54–0.66 m against 0.096 m for fusing. With two independent sources, the Kalman filter already
weights each one by its uncertainty, so throwing one away costs accuracy.

The per-window **relative position error** (RPE) measures drift within each 10 s window,
ignoring any constant offset. It is the metric Wang et al. report per 150 s.

## 9. What the synthetic data leaves out

- GNSS multipath and outages caused by buildings or trees.
- Real point-cloud registration: its failure modes, degenerate geometry, and map errors.
- Time-correlated sensor errors. Real GNSS errors drift slowly rather than being white noise.
- Imperfect time synchronization between sensors.
- Temperature-dependent IMU biases and scale-factor errors.
- Correlation between the GNSS and LiDAR-localizer errors. Here they are independent by
  construction.
- Rich 3-D motion. The figure-eight is flat, with no hills, braking pitch or cornering roll.

Conclusions about **relative** behaviour (bias estimation helps, gating removes large outliers,
dropouts behave as expected) are supported by these experiments. The **absolute** numbers (for
example 0.1 m RMSE) only hold for these idealized simulated sensors and need recorded data to
confirm.

---

## 10. Reproduce one result and trace it to the code

Example: the E2 number "eskf15_all position RMSE 0.096 m".

1. Generate the biased dataset:

   ```bash
   python -m vehicle_localization generate --config configs/default.yaml --output data/generated/default
   ```

2. Run the 15-state filter:

   ```bash
   python -m vehicle_localization run --data data/generated/default --mode eskf15_all --output results/baseline
   ```

3. Trace where each part of the result comes from:

   | Step | Code | Equations |
   |---|---|---|
   | Load inputs (truth is never opened) | `dataset.load_dataset` | — |
   | Schedule IMU intervals and measurements | `runner.run_estimator` | `docs/math.md` §9 |
   | Each IMU step | `eskf.ErrorStateEKF.predict` | §4–6 |
   | Each measurement | `eskf.ErrorStateEKF.update_position` | §7–8 |
   | The RMSE | `evaluation.evaluate_run`, after the run, with truth | §10 |

The default config produces the same numbers on every run (the tests check this). Across machines,
expect agreement to many decimal places but not bit-for-bit.
