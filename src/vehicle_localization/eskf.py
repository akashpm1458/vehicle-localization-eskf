"""Error-state Kalman filter shared by the 9-state and 15-state modes.

* 15-state: estimates position, velocity, attitude, accelerometer bias and gyro bias.
* 9-state: estimates position, velocity, attitude and treats both biases as KNOWN ZERO.
  In a biased experiment this assumption is deliberately wrong; that mismatch is what
  the 9-vs-15 comparison measures.

The filter only receives (a) the initial prior and (b) the estimator configuration.
It never sees ground truth, generator parameters or future measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.stats import chi2

from .config import MODES
from .discretization import DISCRETIZERS, continuous_noise, error_dynamics, noise_mapping
from .rotations import (
    quat_from_rotvec,
    quat_multiply,
    quat_normalize,
    quat_to_matrix,
    right_jacobian,
    skew,
    so3_exp,
)
from .state import ATT, BA, BG, POS, VEL, InitialPrior


class FilterNumericalError(RuntimeError):
    """Raised with an actionable message when the filter's numbers become invalid."""


@dataclass
class EstimatorConfig:
    """Everything the filter is allowed to know (built from the ``estimator`` group)."""

    mode: str
    n_states: int
    sensors: tuple[str, ...]
    gravity: np.ndarray
    accel_noise_density: float
    gyro_noise_density: float
    accel_bias_rw: float
    gyro_bias_rw: float
    lever_arms: dict[str, np.ndarray]
    covariance_scale: dict[str, float]
    gating_enabled: bool
    gating_probability: float
    discretization: str
    max_substep_s: float
    max_imu_gap_s: float
    covariance_check_every: int
    full_covariance_log_every: int
    # Measurement-selection policy, applied by the runner (the filter maths is unchanged).
    fusion_policy: str = "fuse"
    switch_judge_probability: float = 0.9973
    switch_recover_after: int = 10
    switch_gnss_timeout_s: float = 0.5
    extra: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(est: dict, mode: str | None = None) -> "EstimatorConfig":
        mode = mode or est["mode"]
        n, sensors = MODES[mode]
        return EstimatorConfig(
            mode=mode,
            n_states=n,
            sensors=sensors,
            gravity=np.asarray(est["gravity_mps2"], dtype=float),
            accel_noise_density=float(est["accel_noise_density"]),
            gyro_noise_density=float(est["gyro_noise_density"]),
            accel_bias_rw=float(est["accel_bias_rw"]),
            gyro_bias_rw=float(est["gyro_bias_rw"]),
            lever_arms={k: np.asarray(v, dtype=float) for k, v in est["lever_arms_m"].items()},
            covariance_scale={k: float(v) for k, v in est["covariance_scale"].items()},
            gating_enabled=bool(est["gating"]["enabled"]),
            gating_probability=float(est["gating"]["probability"]),
            discretization=est["discretization"],
            max_substep_s=float(est["max_substep_s"]),
            max_imu_gap_s=float(est["max_imu_gap_s"]),
            covariance_check_every=int(est["covariance_check_every"]),
            full_covariance_log_every=int(est["full_covariance_log_every"]),
            fusion_policy=est.get("fusion_policy", "fuse"),
            switch_judge_probability=float(est.get("switching", {}).get("judge_probability", 0.9973)),
            switch_recover_after=int(est.get("switching", {}).get("recover_after", 10)),
            switch_gnss_timeout_s=float(est.get("switching", {}).get("gnss_timeout_s", 0.5)),
        )


class ErrorStateEKF:
    """Nominal-state propagation plus error-state covariance."""

    def __init__(self, prior: InitialPrior, config: EstimatorConfig):
        self.cfg = config
        n = config.n_states
        self.n = n
        nom = prior.nominal()
        self.p, self.v, self.q = nom.p, nom.v, nom.q
        # The 9-state filter assumes known zero biases, whatever the prior says.
        self.ba = nom.ba.copy() if n == 15 else np.zeros(3)
        self.bg = nom.bg.copy() if n == 15 else np.zeros(3)
        self.P = prior.P[:n, :n].copy()
        self.g = config.gravity
        self.Qc = continuous_noise(config.accel_noise_density, config.gyro_noise_density,
                                   config.accel_bias_rw, config.gyro_bias_rw, n)
        self._discretize = DISCRETIZERS[config.discretization]
        # Chi-square gate for a 3-D innovation (not 9: the scalar 3-sigma rule does not
        # carry over to three dimensions).
        self.gate_threshold = float(chi2.ppf(config.gating_probability, df=3))
        self.t_ns: int = prior.t_ns
        self.n_predict_steps = 0
        self.cov_stats = {"checks": 0, "roundoff_negative_eigs": 0, "min_eig_seen": math.inf}

    # ------------------------------------------------------------------ prediction
    def predict(self, specific_force: np.ndarray, angular_rate: np.ndarray, dt: float) -> None:
        """Propagate nominal state and covariance over dt seconds with a held IMU reading.

        dt is split into equal substeps no longer than ``max_substep_s``.
        """
        if dt < 0:
            raise FilterNumericalError(f"negative propagation interval {dt} s")
        if dt == 0:
            return
        nsub = max(1, math.ceil(dt / self.cfg.max_substep_s - 1e-9))
        h = dt / nsub
        for _ in range(nsub):
            self._predict_step(specific_force, angular_rate, h)

    def _predict_step(self, f_m: np.ndarray, w_m: np.ndarray, h: float) -> None:
        f_hat = f_m - self.ba
        w_hat = w_m - self.bg
        R = quat_to_matrix(self.q)
        # Midpoint attitude for the acceleration over this substep.
        R_mid = R @ so3_exp(w_hat * (0.5 * h))
        a_w = R_mid @ f_hat + self.g
        self.p = self.p + self.v * h + 0.5 * a_w * h * h
        self.v = self.v + a_w * h
        self.q = quat_normalize(quat_multiply(self.q, quat_from_rotvec(w_hat * h)))

        A = error_dynamics(R_mid, f_hat, w_hat, self.n)
        L = noise_mapping(R_mid, self.n)
        W = L @ self.Qc @ L.T
        phi, Qd = self._discretize(A, W, h)
        P = phi @ self.P @ phi.T + Qd
        self.P = 0.5 * (P + P.T)
        self.n_predict_steps += 1
        if self.n_predict_steps % self.cfg.covariance_check_every == 0:
            self.check_covariance("prediction")

    # ------------------------------------------------------------------ correction
    def predict_measurement(self, lever_arm_body: np.ndarray) -> np.ndarray:
        """z_hat = p + R l: world position of a body-fixed sensor reference point."""
        return self.p + quat_to_matrix(self.q) @ lever_arm_body

    def measurement_jacobian(self, lever_arm_body: np.ndarray) -> np.ndarray:
        """H = [I, 0, -R [l]x, 0, 0] (first three blocks only for the 9-state filter).

        With R_true = R Exp(dtheta): R_true l ≈ R l + R [dtheta]x l = R l - R [l]x dtheta.
        """
        H = np.zeros((3, self.n))
        H[:, POS] = np.eye(3)
        H[:, ATT] = -quat_to_matrix(self.q) @ skew(lever_arm_body)
        return H

    def update_position(self, z_world: np.ndarray, covariance: np.ndarray, lever_arm_body: np.ndarray,
                        sensor: str) -> dict:
        """Gate, then correct with one 3-D position measurement. Returns an innovation record.

        Steps: innovation and NIS (logged before gating) -> gate -> Kalman gain (solves,
        no explicit inverse) -> Joseph covariance in the old tangent coordinates ->
        inject (additive; attitude by right quaternion multiplication) -> covariance reset
        with the SO(3) right Jacobian of the injected rotation.
        A rejected measurement leaves the state and covariance untouched.
        """
        n = self.n
        z_hat = self.predict_measurement(lever_arm_body)
        r = z_world - z_hat
        H = self.measurement_jacobian(lever_arm_body)
        PHt = self.P @ H.T
        S = H @ PHt + covariance
        S = 0.5 * (S + S.T)
        try:
            cho = cho_factor(S, lower=True)
        except LinAlgError as exc:
            raise FilterNumericalError(
                f"innovation covariance for sensor '{sensor}' at t_ns={self.t_ns} is not positive definite "
                f"(diag S = {np.diag(S)}); check the measurement covariance and filter state"
            ) from exc
        nis = float(r @ cho_solve(cho, r))
        record = {"sensor": sensor, "z": z_world.copy(), "zhat": z_hat, "r": r, "S": S, "nis": nis,
                  "threshold": self.gate_threshold, "gating_enabled": self.cfg.gating_enabled, "accepted": True}
        if self.cfg.gating_enabled and nis > self.gate_threshold:
            record["accepted"] = False
            return record

        K = cho_solve(cho, PHt.T).T  # K = P H^T S^-1
        dx = K @ r
        IKH = np.eye(n) - K @ H
        P_joseph = IKH @ self.P @ IKH.T + K @ covariance @ K.T

        # Inject the error estimate into the nominal state.
        self.p = self.p + dx[POS]
        self.v = self.v + dx[VEL]
        self.q = quat_normalize(quat_multiply(self.q, quat_from_rotvec(dx[ATT])))
        if n == 15:
            self.ba = self.ba + dx[BA]
            self.bg = self.bg + dx[BG]

        # Reset: express the covariance in the new tangent coordinates (error mean -> 0).
        G = np.eye(n)
        G[ATT, ATT] = right_jacobian(dx[ATT])
        P = G @ P_joseph @ G.T
        self.P = 0.5 * (P + P.T)
        self.check_covariance(f"{sensor} update")
        record["dx_norm_attitude_rad"] = float(np.linalg.norm(dx[ATT]))
        return record

    # ------------------------------------------------------------------ diagnostics
    def check_covariance(self, where: str) -> None:
        """Finite, symmetric and PSD within a scale-aware round-off tolerance.

        Eigenvalues are never clipped. Tiny negative eigenvalues at round-off level are
        counted (and reported); anything larger raises.
        """
        P = self.P
        if not np.all(np.isfinite(P)):
            raise FilterNumericalError(f"non-finite covariance after {where} at t_ns={self.t_ns}")
        eig = np.linalg.eigvalsh(P)
        tol = 64 * self.n * np.finfo(float).eps * max(abs(eig[-1]), 1e-300)
        self.cov_stats["checks"] += 1
        self.cov_stats["min_eig_seen"] = min(self.cov_stats["min_eig_seen"], float(eig[0]))
        if eig[0] < -tol:
            raise FilterNumericalError(
                f"covariance not positive semidefinite after {where} at t_ns={self.t_ns}: "
                f"min eigenvalue {eig[0]:.3e} < -{tol:.3e}"
            )
        if eig[0] < 0:
            self.cov_stats["roundoff_negative_eigs"] += 1

    def state_vector(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.p.copy(), self.v.copy(), self.q.copy(), self.ba.copy(), self.bg.copy()
