# Using recorded data

The estimator runs on any data in the canonical format (see [data_format.md](data_format.md)). It
doesn't need to be synthetic.

---

## 1. Steps

### 1. Convert your logs into the canonical files

Create `imu.csv`, `gnss.csv`, an optional `lidar_position.csv`, `initial_prior.json` and
`metadata.json`:

- **Times** must be integer nanoseconds from a common start time. Re-stamp all sensors on one clock
  first.
- **IMU:** specific force in m/s² and angular rate in rad/s, in the body frame (`x` forward,
  `y` left, `z` up), at the body origin. Add one final row to close the last interval.
- **Positions:** metres in one metric world frame with `z` up, plus their 3×3 covariance in m².
- **Metadata:** `metadata.json` must declare the supported conventions exactly (frames, units,
  timing and schema version; see [data_format.md §6](data_format.md#6-metadatajson)). If your data
  uses another convention, convert it. Relabelling it is not enough: the validator rejects anything
  else so a mismatch can't be silently misread.
- An empty position file (header only) is accepted with a warning. That stream then provides no
  aiding.

### 2. Write the prior

Write `initial_prior.json` with a covariance that honestly reflects how well you know the start.
The covariance must keep the documented `error_state_order` and per-block `units`
(attitude in radians, as a body-local right-multiplicative error); other layouts are rejected.
Record in `provenance` how it was obtained (for example: "averaged GNSS over the first 5 s while
stationary; heading from a surveyed alignment").

Automatic global heading initialization is **not** implemented. A stationary accelerometer cannot
determine yaw, and it cannot fully separate tilt from accelerometer bias.

### 3. Put your calibration in the estimator config

Set `estimator.lever_arms_m` and `estimator.covariance_scale` in a copy of `configs/default.yaml`.
They come from your calibration, never from the data files.

### 4. Validate, then run

```bash
python -m vehicle_localization validate --data path/to/recording
python -m vehicle_localization run --data path/to/recording --mode eskf15_all --config my_config.yaml --output results/recording
```

### 5. Without ground truth

If `ground_truth.csv` is absent or malformed (including duplicate or decreasing timestamps):

- Accuracy metrics are turned off.
- Estimation, the innovation logs (`innovations.csv`), NIS statistics and timing checks all still
  run.

If you do have a reference trajectory (for example from an RTK/INS system), save it as
`ground_truth.csv`. The evaluator interpolates it to the estimate times (linear for vectors, SLERP
for attitude) inside the overlap only. It never extrapolates and never does nearest-neighbour
matching.

---

## 2. GNSS latitude/longitude must be converted first

The state is in metres. **Never** put latitude and longitude in degrees into `gnss.csv`.

Convert to a local metric frame first, for example ENU around a fixed origin. Any converter must
state:

- the datum (for example WGS-84);
- the local origin (latitude, longitude, height);
- the altitude convention (ellipsoidal or orthometric height);
- how the receiver's covariance was converted into the local frame (rotated with the same
  transform).

### Different frames

If a LiDAR localizer reports positions in its own map frame, transform **both** the positions and
their covariances into the GNSS world frame with an independently obtained map-to-world
calibration before fusing.

### Correlated errors

Fusing two streams assumes independent errors. A real localizer that uses GNSS or IMU internally
violates this assumption.

---

## 3. A possible future CARLA adapter (not implemented)

CARLA is **not** installed or required. If a small, already-recorded CARLA sequence is converted
later, check the following first:

### Coordinates

CARLA documents `x` forward, `y` **right**, `z` up, which differs from this project's `y` left.
Convert positions, orientations, accelerations and angular rates carefully.

Angular rates are axial vectors: under a handedness change they do not simply copy a position-axis
sign flip.

### Accelerometer

Confirm with a stationary test in that exact CARLA version that the accelerometer reports specific
force (about +9.81 m/s² up when level) and not gravity-free acceleration.

### GNSS

CARLA GNSS outputs latitude and longitude. Convert them as described in section 2.

### LiDAR

The CARLA LiDAR produces detections (point clouds), not a vehicle pose. Use the output of a real
localization algorithm. If you use actor ground truth plus noise as a stand-in, label it as a
**surrogate**; never call it LiDAR odometry.

### Course files

Don't automatically download course pickle files, and don't load untrusted serialized objects. A
trusted dataset can be converted separately into the CSV/JSON format, with its provenance recorded.

### Hardware

A 4 GB GPU is a reason not to make CARLA a prerequisite. It doesn't prove that a given CARLA
release won't run.
