# Data format

A dataset is a folder of CSV and JSON files. The estimator reads only the **input** files; the
evaluator also reads the **evaluation-only** files.

Check any folder with:

```bash
python -m vehicle_localization validate --data <folder>
```

---

## 1. Files

| File | Who reads it | Required |
|---|---|---|
| `imu.csv` | estimator | yes |
| `gnss.csv` | estimator | yes |
| `lidar_position.csv` | estimator | no (IMU + GNSS runs work without it) |
| `initial_prior.json` | estimator | yes |
| `metadata.json` | estimator, validator | yes |
| `ground_truth.csv` | **evaluator only** | no (without it, accuracy metrics are turned off) |
| `truth_metadata.json` | **evaluator only** | no |

Inputs and truth are validated separately. `load_dataset` (used for estimation) never opens
`ground_truth.csv` or `truth_metadata.json`, so a missing or malformed truth file can't block or
change estimation. A malformed truth file only turns off the accuracy metrics, with a warning.
`validate` checks both and labels truth problems `[evaluation]`.

General rules:

- UTF-8 text, comma-separated, with exactly the header shown.
- Generated files use 17 significant digits, so they round-trip float64 exactly.

---

## 2. Units and frames

| Quantity | Unit / convention |
|---|---|
| Time `t_ns` | Integer nanoseconds from the start of the dataset |
| Position | metres, world frame `W` (right-handed, `z` up) |
| Specific force | m/s², body frame `B` (`x` forward, `y` left, `z` up) |
| Angular rate | rad/s, body frame |
| Quaternion | `R_WB`, stored **`qx, qy, qz, qw`** |
| Covariance | m² |

---

## 3. `imu.csv`

```
t_ns,fx,fy,fz,wx,wy,wz
0,0.20371227896797608,0.24359480780643419,9.8102703928779302,-0.0023971964439548801,0.010947685112605828,-0.0085189975473146089
10000000,0.2276782590299011,-0.10631885913345744,10.095873746483665,0.018427747678251959,-0.0052010933476413657,-0.0077612271652066564
```

`f` is **specific force**: a level body at rest reads about `[0, 0, +9.81]`.

### Timing semantics (important)

- Row `k` is an **interval measurement** held over `[t_k, t_{k+1})`.
- `N` supported intervals need `N + 1` rows. The **final row only marks the end of coverage**; its
  values must be finite but are never propagated. Tests check that the final interval is processed
  exactly once and that changing the final row's values changes nothing.
- The synthetic generator evaluates the true force and rate at each interval's midpoint and adds
  interval-average noise. That sample stands for the whole interval; it is not a claim that it was
  available before the interval ended.
- Timestamps must be strictly increasing. Shuffled rows are rejected unless `--normalize-imu` is
  given; duplicate rows are always rejected.
- A gap longer than `estimator.max_imu_gap_s` (default 0.05 s) stops the run with an error, so an
  old reading is never held across an outage.

---

## 4. `gnss.csv` and `lidar_position.csv`

```
t_ns,x,y,z,r_xx,r_xy,r_xz,r_yy,r_yz,r_zz
30000000,0.74306912428419736,0.18356663234941811,0.85955080237053094,1,0,0,1,0,4
```

| Column | Meaning |
|---|---|
| `x, y, z` | World position of the sensor's **own reference point** (GNSS antenna, or LiDAR-localizer reference point) — not the body origin. |
| `r_xx … r_zz` | Upper triangle of the reported symmetric 3×3 covariance. It must be positive definite. |

The filter's assumed lever arm (`estimator.lever_arms_m`) links the reference point to the body
origin. The estimator config can scale the reported covariance (`estimator.covariance_scale`).

Rules:

- A header-only file (no rows) is valid. The validator warns that the stream contains no
  measurements, and modes that use it get no aiding from it. The generator writes such a file when
  a sensor's first measurement time is after the end of the dataset.
- Two different sensors may share a timestamp; GNSS is applied first, then LiDAR.
- Within one stream, duplicate timestamps are rejected.
- An unsorted stream is sorted by measurement time, which assumes no delivery delay; the validator
  warns about it.
- Measurements outside the IMU time range are **reported and skipped**, never extrapolated.

In the synthetic data, `lidar_position.csv` holds **simulated LiDAR-localizer positions** (truth
plus noise), not real scan-matching output.

---

## 5. `initial_prior.json`

```json
{
  "t_ns": 0,
  "position_m": [-0.83, 0.62, 0.41],
  "velocity_mps": [4.03, 4.48, -0.12],
  "quaternion": {"order": "xyzw", "value": [0.012, -0.009, 0.383, 0.924]},
  "accel_bias_mps2": [0.0, 0.0, 0.0],
  "gyro_bias_radps": [0.0, 0.0, 0.0],
  "covariance": {
    "error_state_order": ["px","py","pz","vx","vy","vz","thx","thy","thz","bax","bay","baz","bgx","bgy","bgz"],
    "units": {"position": "m", "velocity": "m/s", "attitude": "rad (body-local, right-multiplicative)",
              "accel_bias": "m/s^2", "gyro_bias": "rad/s"},
    "matrix": [[1.0, 0, "..."], "..."]
  },
  "provenance": {"method": "ground_truth_assisted", "description": "..."}
}
```

Rules:

- The quaternion must say `"order": "xyzw"`. A bare 4-vector is rejected as ambiguous.
- `t_ns` must equal the first IMU timestamp.
- `covariance.error_state_order` and `covariance.units` must be **exactly** the values shown. They
  are checked before the matrix is read, so a matrix in another order, in degrees, or for a
  different attitude-error definition is rejected instead of being silently misread.
- The covariance is 15×15 for the error state above (the attitude block is the body-local rotation
  error in rad²). It must be symmetric and positive definite. The 9-state filter uses its top-left
  9×9 block.
- `provenance` must say where the prior came from.

Synthetic data use **ground-truth-assisted initialization**: the true state at `t = 0` minus a
sampled perturbation (default σ: 1 m, 0.3 m/s, 2°, 0.1 m/s², 0.005 rad/s), with a bias mean of zero.

---

## 6. `metadata.json`

These keys are required:

- `schema_version`
- `frames`
- `units`
- `quaternion_order`
- `timing`
- `seed`
- `duration_s`
- `sensors` (including each sensor's reference point)
- `generator_version`

### Declared conventions must match exactly

`frames`, `units` and `timing` are objects whose machine-readable keys must equal the conventions
this code implements. Any other value (another frame, another unit, another timing model, or an
unknown schema version) is **rejected**. Convert the data first instead of having it silently
misinterpreted.

| Key | Required value |
|---|---|
| `schema_version` | `"1.0"` |
| `quaternion_order` | `"xyzw"` |
| `frames.world` | `"right_handed_z_up"` |
| `frames.body` | `"x_forward_y_left_z_up"` |
| `frames.rotation` | `"R_WB_body_to_world"` |
| `units.time` | `"ns"` |
| `units.length` | `"m"` |
| `units.angle` | `"rad"` |
| `units.specific_force` | `"m/s^2"` |
| `units.angular_rate` | `"rad/s"` |
| `units.covariance` | `"m^2"` |
| `timing.imu` | `"interval_held_forward_last_row_end_marker"` |
| `timing.external` | `"measurement_time_no_delivery_delay"` |

Other keys, such as a `description` object, are allowed and ignored.

Example:

```json
"frames": {"world": "right_handed_z_up", "body": "x_forward_y_left_z_up",
           "rotation": "R_WB_body_to_world", "description": {"world": "local ENU around ..."}},
"units": {"time": "ns", "length": "m", "angle": "rad", "specific_force": "m/s^2",
          "angular_rate": "rad/s", "covariance": "m^2"},
"timing": {"imu": "interval_held_forward_last_row_end_marker",
           "external": "measurement_time_no_delivery_delay"}
```

Metadata never contains lever-arm values. The filter's lever arms come only from the estimator
config, so a calibration experiment can't leak the true value.

Datasets generated before this rule existed used free-text values and are now rejected. Regenerate
them with `python -m vehicle_localization generate ...`.

---

## 7. `ground_truth.csv` (evaluation only)

```
t_ns,px,py,pz,vx,vy,vz,qx,qy,qz,qw,bax,bay,baz,bgx,bgy,bgz
```

This is the body-origin truth at the IMU timestamps. Truth is checked only on the evaluation side:

- The header must match exactly. A header with `qw` before `qx` is rejected, with a hint that it
  looks like `wxyz` order.
- Values must be finite and quaternions unit length.
- Timestamps must be **strictly increasing**. Duplicate or decreasing timestamps are rejected
  before any interpolation.

## 8. `truth_metadata.json` (evaluation only)

Holds the true simulation parameters (noise, true lever arms, bias settings), the random-stream
seeds, the sampled prior error, and the corruption labels (for example, the indices and times of
injected outliers).

---

## 9. What the validator rejects

| Problem | Result |
|---|---|
| NaN or inf values | rejected |
| Wrong header or wrong number of columns | rejected |
| Non-integer `t_ns` | rejected |
| Covariance that is not positive definite | rejected |
| Ambiguous or `wxyz` quaternion order | rejected |
| Unsupported `schema_version`, frame, unit or timing convention | rejected |
| Prior covariance with a different `error_state_order` or `units` | rejected |
| Negative IMU time steps | rejected (unless `--normalize-imu`) |
| Duplicate IMU timestamps | rejected |
| Header-only `imu.csv` (fewer than 2 rows) | rejected |
| Missing metadata keys | rejected |
| Prior that doesn't match the IMU start | rejected |
| Header-only `gnss.csv` or `lidar_position.csv` | allowed, with a warning (no aiding from that stream) |
| Same timestamp across different sensors | allowed |
| Measurements outside IMU coverage | warning, then skipped |
| Missing `ground_truth.csv` | allowed (accuracy metrics off) |
| Malformed `ground_truth.csv`, or duplicate/decreasing truth timestamps | `[evaluation]` error; estimation still runs, accuracy metrics off |
