# vehicle-localization-eskf

Estimate a vehicle's motion from noisy sensors using an **error-state Kalman filter (ESKF)**.

> **Status: in development.** The code is not written yet. This README is the plan, laid out as steps. Results get added only after real runs.

---

## Step 1 — Understand the goal

- A vehicle moves along a known path.
- Its sensors are noisy.
- The filter estimates where the vehicle is from those sensors.
- We compare that estimate with the true path.

---

## Step 2 — Know the sensors

| Sensor | Rate | What it gives |
|---|---|---|
| IMU | 100 Hz | Acceleration and turn rate |
| GNSS | 5 Hz | Rough position (about 1 m error) |
| LiDAR localizer | 10 Hz | Better position (about 0.15 m error) |

- All sensor data is **simulated**.
- "LiDAR localizer" means a simulated position output, **not** real LiDAR scans.

---

## Step 3 — Know what we estimate

- Position
- Velocity
- Orientation (stored as a quaternion)
- IMU biases (only in the final filter)

---

## Step 4 — Learn the conventions

- World frame: `z` points up.
- Body frame: `x` forward, `y` left, `z` up.
- Quaternions are stored as `[qx, qy, qz, qw]`.
- Gravity is `[0, 0, -9.80665] m/s²`.
- Units are metres, seconds and radians.
- Timestamps in files are integer nanoseconds.

---

## Step 5 — Generate the data

- Make a 120-second figure-eight path.
- Create IMU, GNSS and LiDAR readings from it.
- Add noise using a fixed seed (default `7`).
- Save everything to CSV and JSON.

---

## Step 6 — Try IMU only

- Integrate the IMU readings with no corrections.
- Watch the error grow over time.
- This shows **why** we need a filter.

---

## Step 7 — Build the 9-state filter

- Estimates position, velocity and orientation.
- Assumes the IMU has no bias.
- Corrects itself with GNSS and LiDAR positions.

---

## Step 8 — Build the 15-state filter

- Adds accelerometer bias and gyroscope bias.
- Learns the biases while it runs.
- This is the final estimator.

---

## Step 9 — Compare five setups

| Name | What it uses |
|---|---|
| `imu_only` | IMU only |
| `eskf9_gnss` | 9-state + GNSS |
| `eskf9_all` | 9-state + GNSS + LiDAR |
| `eskf15_gnss` | 15-state + GNSS |
| `eskf15_all` | 15-state + GNSS + LiDAR |

All five use the same data, so the comparison is fair.

---

## Step 10 — Run the experiments

| ID | What we test |
|---|---|
| E0 | Simple noise-free cases |
| E1 | IMU only vs. 9-state filter |
| E2 | 9-state vs. 15-state with biased IMU |
| E3 | GNSS lost for 20 s, no LiDAR |
| E4 | GNSS lost for 20 s, LiDAR still on |
| E5 | Both sensors lost for 20 s |
| E6 | Filter trusts GNSS too much or too little |
| E7 | Wrong LiDAR mounting position |
| E8 | Fake GNSS outliers, rejection on vs. off |

---

## Step 11 — Check the results

- Position, velocity and orientation errors (RMSE).
- Uncertainty bounds compared with the actual error.
- How many bad measurements were rejected.
- Plots and a generated report.

Results will appear here after the runs.

---

## Step 12 — Install it (once the code exists)

You need Python 3.11 or 3.12. No GPU is needed.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

**Linux / WSL**

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

---

## Step 13 — Run it (once the code exists)

Run everything in one go:

```bash
python -m vehicle_localization demo
```

Or one step at a time:

```bash
python -m vehicle_localization doctor
python -m vehicle_localization generate --config configs/default.yaml --output data/generated/default
python -m vehicle_localization run --data data/generated/default --mode eskf15_all --output results/baseline
python -m vehicle_localization suite --config configs/default.yaml --output results/suite
python -m pytest -q
```

---

## Step 14 — Know the limits

- Synthetic data only.
- Localization only: no mapping.
- No neural networks.
- Offline processing, not real-time.
- The path is flat (2-D motion inside a 3-D filter).
- The starting point comes from the true state plus some noise.

---

## Background


- Main reference: J. Solà, [Quaternion kinematics for the error-state Kalman filter](https://arxiv.org/abs/1711.02508).

## License

MIT. See [LICENSE](LICENSE).
