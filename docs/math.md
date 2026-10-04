# Mathematics and conventions

This page is the reference for every equation the code uses. The file and function that
implement each part are named at the end of its section.

---

## 1. Frames, units and symbols

| Symbol | Meaning |
|---|---|
| `W` | World frame: right-handed, `z` up. For real data this can be local ENU; synthetic `x/y` are just local axes. |
| `B` | Body/IMU frame: `x` forward, `y` left, `z` up. The IMU sits at the body origin with axes equal to `B`. |
| `R = R_WB` | Rotation taking body vectors to world vectors: `v_W = R v_B`. |
| `p, v` | Position and velocity of the body origin, in `W`. |
| `q` | Unit quaternion of `R_WB`, stored `[qx, qy, qz, qw]` (Hamilton product, scalar last, SciPy order). |
| `b_a, b_g` | Accelerometer and gyroscope biases. |
| `g_W` | Gravity, `[0, 0, -9.80665] m/s²`. |
| `[u]×` | Skew matrix with `[u]× v = u × v`. |
| `Exp(φ)`, `Log(R)` | SO(3) exponential of a rotation vector and its inverse (not element-wise). |
| `quat(φ)` | Unit quaternion of `Exp(φ)`. |

Units: metres, seconds, radians. CSV timestamps are integer nanoseconds.

Code: `rotations.py`, `state.py`.

---

## 2. Rotation conventions

The orientation error is **right-multiplicative and body-local**:

```
R_true = R_est · Exp(δθ)
```

Prediction and correction both compose on the right:

```
q_next      = q ⊗ quat(ω̂ Δt)
q_corrected = q ⊗ quat(δθ̂)
```

Checks used by the tests:

- `skew(u) @ v == np.cross(u, v)`.
- A +90° rotation about `z` maps `[1, 0, 0]` to `[0, 1, 0]`.
- `q` and `−q` are the same rotation.

### Right Jacobian of SO(3)

```
J_r(φ) = I − (1 − cos α)/α² [φ]× + (α − sin α)/α³ [φ]×²,      α = |φ|
```

Below `α = 1e-3` the code uses the series `I − (½ − α²/24)[φ]× + (⅙ − α²/120)[φ]×²`.

Key property: `Exp(φ + ε) ≈ Exp(φ) Exp(J_r(φ) ε)`, so `J_r(φ)` is the derivative of
`Log(Exp(−φ) Exp(φ + ε))` at `ε = 0`. A test checks this identity with central differences.

Code: `rotations.right_jacobian`.

---

## 3. Sensor models

### IMU

The accelerometer measures **specific force**, not world acceleration:

```
f_B = R_WBᵀ (a_W − g_W)
f_m = f_B + b_a + n_a
ω_m = ω_B + b_g + n_g
```

A level body at rest reads `f_B = [0, 0, +9.80665]`. Rotating that into the world frame and adding
`g_W` gives zero acceleration.

### Noise convention

White noise is continuous with `E[n(t) n(t')ᵀ] = σ² I δ(t − t')`. Over one IMU interval `Δt`:

| Quantity | Standard deviation per interval |
|---|---|
| Interval-average white noise | `σ / √Δt` |
| Bias random-walk step | `σ_rw · √Δt` |

The filter uses the **same** convention (`Qc = diag(σ_a², σ_g², σ_ba², σ_bg²)`), so nothing is
squared or multiplied by `Δt` twice.

Note: a real datasheet may quote a one-sided density or use different units. Check the
manufacturer's convention before reusing these numbers. The defaults here are teaching values,
not a particular IMU.

### External position sensors

```
z_s = p + R ℓ_s + n_s,        n_s ~ N(0, R_s)
```

Here `ℓ_s` is the **body-fixed** lever arm from the body origin to the sensor's reference point.
Its world-frame direction turns with the vehicle, so it is never a fixed world offset.

Code: `sensors.py`.

---

## 4. Nominal prediction (both filters, and IMU-only)

For each substep `h ≤ 0.01 s` with the held IMU reading:

```
f̂ = f_m − b̂_a,      ω̂ = ω_m − b̂_g
R_mid = R_k · Exp(ω̂ h / 2)                 midpoint attitude
a_W   = R_mid f̂ + g_W
p_{k+1} = p_k + v_k h + ½ a_W h²
v_{k+1} = v_k + a_W h
q_{k+1} = q_k ⊗ quat(ω̂ h)                  then normalize
```

- The biases stay constant during prediction; their uncertainty grows through the random walk.
- This is numerical integration of held readings, not an exact solution for arbitrary motion. The
  E0 fixtures show its error falls about 4× each time the step is halved (second order).
- IMU-only dead reckoning uses exactly this integrator, so any improvement from fusion can't come
  from a better integrator.

Code: `eskf.ErrorStateEKF._predict_step`.

---

## 5. Error state and its dynamics

Error state, 15 numbers (the 9-state filter keeps the first three blocks):

| Slice | Block |
|---|---|
| `0:3` | `δp` (world) |
| `3:6` | `δv` (world) |
| `6:9` | `δθ` (body-local, rad) |
| `9:12` | `δb_a` |
| `12:15` | `δb_g` |

The covariance `P` is 15×15 (or 9×9). There is never a 4×4 quaternion covariance.

### Deriving the velocity row

The true acceleration is `R Exp(δθ)(f_m − b_a − δb_a − n_a) + g`. To first order, and using
`[a]× b = −[b]× a`:

```
δv̇ = −R [f̂]× δθ − R δb_a − R n_a
```

### Deriving the attitude row

With `R_true = R Exp(δθ)` and `ω_true = ω̂ − δb_g − n_g`:

```
δθ̇ = −[ω̂]× δθ − δb_g − n_g
```

### Full continuous model

```
d(δx)/dt = A δx + L n

A:  A_pv = I,   A_vθ = −R[f̂]×,   A_vba = −R,   A_θθ = −[ω̂]×,   A_θbg = −I     (all other blocks 0)
L:  L_v,na = −R,   L_θ,ng = −I,   L_ba,nba = I,   L_bg,nbg = I

Qc = diag(σ_a² I, σ_g² I, σ_ba² I, σ_bg² I),      W = L Qc Lᵀ
```

`R` is the midpoint rotation of the substep. The 9-state model uses the top-left 9×9 block of
`A` and only the accelerometer and gyro channels.

Code: `discretization.error_dynamics`, `noise_mapping`, `continuous_noise`.

---

## 6. Discretization

### Fast method (the default)

```
Φ(h)  = I + A h + ½ A² h²
Qd   ≈ h/6 · [ W + 4 Φ(h/2) W Φ(h/2)ᵀ + Φ(h) W Φ(h)ᵀ ]      (Simpson's rule)
P⁻    = Φ P Φᵀ + Qd,      then P ← (P + Pᵀ)/2
```

`Qd` is a non-negatively weighted sum of PSD matrices, so it stays PSD. It is still an
**approximation**.

### Reference method (`van_loan`)

```
E  = expm( [[A, W], [0, −Aᵀ]] · h )
Φ  = E₁₁,      Qd = E₁₂ Φᵀ
```

This is exact for frozen `A` and `W`. Check: `E₁₂ Φᵀ = ∫₀ʰ e^{A(h−s)} W e^{Aᵀ(h−s)} ds`.

### Measured agreement

Relative Frobenius error at `h = 0.01 s`, for tilted-stationary, turning and 3-D rotating cases:

| Quantity | Measured error | Test limit |
|---|---|---|
| `Φ` | about 7e-7 | 2e-6 |
| `Qd` | about 6e-10 | 2e-9 |

Halving `h` shrinks both errors about 8× (third-order local error). A full 10 s run with each
method gives positions that agree to better than 1e-6 m.

Code: `discretization.fast_discretize`, `van_loan_discretize`; tests in `tests/test_eskf_math.py`.

---

## 7. Position update

### Measurement model and Jacobian

```
ẑ = p̂ + R̂ ℓ,          r = z − ẑ
```

With `R_true ℓ ≈ R̂ (I + [δθ]×) ℓ = R̂ ℓ − R̂ [ℓ]× δθ`:

```
H = [ I   0   −R̂ [ℓ]×   0   0 ]            (9-state: first three blocks)
```

With `ℓ = 0` this is a direct position observation.

### Gain and Joseph update

```
S = H P⁻ Hᵀ + R_s
K = P⁻ Hᵀ S⁻¹                       (Cholesky solve; no explicit inverse)
δx̂ = K r
P_J = (I − K H) P⁻ (I − K H)ᵀ + K R_s Kᵀ
```

### Injection

- Position, velocity and both biases are corrected additively.
- Attitude: `q ← q ⊗ quat(δθ̂)`, then normalize.

### Reset

After injection the error is measured from the **new** nominal attitude. Write the remaining true
error in the old coordinates as `δθ̂ + ε`. The new error is then:

```
δθ⁺ = Log( Exp(−δθ̂) Exp(δθ̂ + ε) ) ≈ J_r(δθ̂) ε
```

So the covariance is transformed with `G = blockdiag(I, I, J_r(δθ̂), I, I)`:

```
P⁺ = G P_J Gᵀ,      then symmetrize; error mean reset to 0
```

The Joseph form and the reset solve different problems; Joseph alone does not move the
covariance into the new coordinates. A Monte Carlo test checks that sampled errors re-expressed
after a large injection have covariance `G P_J Gᵀ` and not `P_J`.

Code: `eskf.ErrorStateEKF.update_position`.

---

## 8. Outlier gating

```
NIS = rᵀ S⁻¹ r
threshold = chi2.ppf(p, df=3)      p = estimator.gating.probability (default 0.9973 → 14.16; not 9)
```

- NIS is computed and logged **before** the state changes, for every measurement, including
  rejected ones.
- A rejected measurement changes neither the state nor `P`.
- Gating can be switched off for controlled comparisons.
- Repeated rejection is a diagnostic. An over-confident or diverged filter can reject good data:
  see experiment E6 in the report.

---

## 9. Timing

- The IMU row at `t_k` is held over `[t_k, t_{k+1})`. `N` intervals have `N + 1` rows, and the last
  row only marks the end of coverage.
- For each external measurement at time `t` inside an interval:
  1. Propagate to `t`.
  2. Update.
  3. Continue with the same raw reading and the **current** bias estimate.
- At a boundary:
  1. Finish the previous interval.
  2. Apply the measurements at that time: GNSS first, then LiDAR.
  3. Log the state.
  4. Start the next interval.
- All scheduling uses integer nanoseconds: `dt = (t₂ − t₁) · 1e-9`.

### Disclosed approximation

Holding one noisy IMU reading across several off-grid updates makes the noise in the pieces
correlated, which the per-substep `Qd` does not model. In the default experiments, updates fall on
IMU boundaries.

Code: `runner.run_estimator`.

---

## 10. Evaluation formulas

| Quantity | Formula |
|---|---|
| Additive errors | truth − estimate |
| Position RMSE | `√( (1/N) Σ |p_true − p̂|² )`; horizontal uses `x, y` only |
| Attitude error | `δθ = Log(R̂ᵀ R_true)` (same coordinates as `P_θθ`); angle `|δθ|`. Quaternion components are never subtracted. |
| Coverage | Fraction of samples with `|error_i| ≤ 3√P_ii`. A diagnostic only. |
| Mean NIS | Computed over **pre-gate** innovations. About 3 for a consistent filter; not a required equality for one run. A high value shows the innovations exceed the predicted covariance `S`. Possible causes include measurement or process noise set too small, unmodelled bias or calibration error, timing errors, outliers or divergence; NIS alone doesn't single out one. |

Code: `evaluation.py`.

---

## References

- J. Solà, *Quaternion kinematics for the error-state Kalman filter*, 2017, [arXiv:1711.02508](https://arxiv.org/abs/1711.02508). Used to check the local-error, injection and reset conventions. This project's ordering and fixed-gravity state are derived and tested here.
- SciPy: [`Rotation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) (scalar-last quaternions), [`linalg.expm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html), [`stats.chi2`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html).
- C. F. Van Loan, *Computing integrals involving the matrix exponential*, IEEE TAC, 1978.
