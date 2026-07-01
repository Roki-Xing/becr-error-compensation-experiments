from __future__ import annotations

import hashlib
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from moving_projection_state.core import MovingProjectionConfig, run_single_step


def safe_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float)))


def cosine(u: np.ndarray, v: np.ndarray) -> float | None:
    un = safe_norm(u)
    vn = safe_norm(v)
    if un <= 0.0 or vn <= 0.0:
        return None
    return float(np.dot(np.asarray(u, dtype=float), np.asarray(v, dtype=float)) / (un * vn))


def orthonormalize(P: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(np.asarray(P, dtype=float))
    return q


def theorem_condition_report(*, a: float, b: float, lr: float, eps_a: float, eps_s: float, x0: float) -> dict[str, Any]:
    R = float(a) * abs(float(x0))
    r_star = float(np.sqrt(float(eps_a) * float(eps_s)))
    r0 = float(min(R, r_star))
    M0 = float(r0 / ((r0 + float(eps_a)) * (r0 + float(eps_s))))
    eta_a = float(lr) * float(a)
    eta_b_M0 = float(lr) * float(b) * M0
    return {
        "eta_a": eta_a,
        "two_eps_a": 2.0 * float(eps_a),
        "ok_eta_a_lt_2epsa": bool(0.0 < eta_a < 2.0 * float(eps_a)),
        "R": R,
        "r_star": r_star,
        "r0": r0,
        "M0": M0,
        "eta_b_M0": eta_b_M0,
        "ok_eta_b_M0_le_half": bool(eta_b_M0 <= 0.5),
    }


@dataclass(frozen=True)
class CoordinateConfig:
    lr: float
    eps_a: float
    eps_s: float
    s_min: float
    s_max: float
    rho: float
    rho_min: float
    rho_max: float
    limiter_gamma: float | None = None
    dtype: Any = np.float64


@dataclass(frozen=True)
class AdamLimiterConfig:
    lr: float
    beta1: float
    beta2: float
    eps_a: float
    eps_s: float
    s_min: float
    s_max: float
    rho: float
    rho_min: float
    rho_max: float
    limiter_gamma: float | None = None
    dtype: Any = np.float64


class DiagQuadratic:
    def __init__(self, lam: np.ndarray):
        self.lam = np.asarray(lam, dtype=float)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.lam * np.asarray(x, dtype=float)

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(0.5 * np.dot(self.lam * x, x))


class ScheduleBase:
    projection_mode: str

    def __init__(self, *, d: int, r: int, projection_mode: str):
        self.d = int(d)
        self.r = int(r)
        self.projection_mode = projection_mode
        self.scheduler_factory_id = f"{projection_mode}:{uuid.uuid4()}"
        self.scheduler_object_id = str(uuid.uuid4())

    def basis(self, *, t: int, x: np.ndarray, g_batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def clone(self) -> "ScheduleBase":
        raise NotImplementedError


class FixedBasisSchedule(ScheduleBase):
    def __init__(self, basis: np.ndarray, *, projection_mode: str = "fixed_basis"):
        basis = np.asarray(basis, dtype=float)
        super().__init__(d=basis.shape[0], r=basis.shape[1], projection_mode=projection_mode)
        self._basis = orthonormalize(basis)

    def basis(self, *, t: int, x: np.ndarray, g_batch: np.ndarray) -> np.ndarray:
        return np.asarray(self._basis, dtype=float).copy()

    def clone(self) -> "FixedBasisSchedule":
        return FixedBasisSchedule(np.asarray(self._basis, dtype=float).copy(), projection_mode=self.projection_mode)


class TopGradientRefreshSchedule(ScheduleBase):
    def __init__(self, *, d: int, r: int, refresh_K: int | None):
        tag = "inf" if refresh_K is None else f"K{int(refresh_K)}"
        super().__init__(d=d, r=r, projection_mode=f"top_gradient_refresh_{tag}")
        self.refresh_K = None if refresh_K is None else int(refresh_K)
        self._basis: np.ndarray | None = None

    def basis(self, *, t: int, x: np.ndarray, g_batch: np.ndarray) -> np.ndarray:
        do_refresh = self._basis is None
        if self.refresh_K is not None and self.refresh_K > 0 and t % self.refresh_K == 0:
            do_refresh = True
        if do_refresh:
            g_mean = np.asarray(g_batch, dtype=float).mean(axis=1)
            idx = np.argsort(-np.abs(g_mean))[: self.r]
            P = np.zeros((self.d, len(idx)), dtype=float)
            for j, i in enumerate(idx.tolist()):
                P[int(i), j] = 1.0
            self._basis = P
        return np.asarray(self._basis, dtype=float).copy()

    def clone(self) -> "TopGradientRefreshSchedule":
        return TopGradientRefreshSchedule(d=self.d, r=self.r, refresh_K=self.refresh_K)


class SVDBatchRefreshSchedule(ScheduleBase):
    def __init__(self, *, d: int, r: int, refresh_K: int, random_projection: bool = False, rng_seed: int = 0):
        mode = "random_projection_refresh" if random_projection else "svd_batch_refresh"
        super().__init__(d=d, r=r, projection_mode=f"{mode}_K{int(refresh_K)}")
        self.refresh_K = int(refresh_K)
        self.random_projection = bool(random_projection)
        self.rng_seed = int(rng_seed)
        self._basis: np.ndarray | None = None
        self._rng = np.random.default_rng(self.rng_seed)

    def basis(self, *, t: int, x: np.ndarray, g_batch: np.ndarray) -> np.ndarray:
        if self._basis is None or (self.refresh_K > 0 and t % self.refresh_K == 0):
            if self.random_projection:
                P = self._rng.standard_normal((self.d, self.r))
                self._basis = orthonormalize(P)
            else:
                G = np.asarray(g_batch, dtype=float)
                U, S, _ = np.linalg.svd(G, full_matrices=False)
                r_eff = int(min(self.r, U.shape[1]))
                if len(S) == 0 or float(np.max(S)) <= 1e-12:
                    self._basis = np.eye(self.d, r_eff, dtype=float)
                else:
                    self._basis = orthonormalize(U[:, :r_eff])
        return np.asarray(self._basis, dtype=float).copy()

    def clone(self) -> "SVDBatchRefreshSchedule":
        return SVDBatchRefreshSchedule(
            d=self.d,
            r=self.r,
            refresh_K=self.refresh_K,
            random_projection=self.random_projection,
            rng_seed=self.rng_seed,
        )


def generate_crn_noise_bank(*, steps: int, batch_size: int, d: int, sigma: np.ndarray | float, rng_seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(rng_seed))
    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.ndim == 0:
        sigma_arr = np.full((d,), float(sigma_arr), dtype=float)
    sigma_arr = sigma_arr.reshape(1, 1, d)
    base = rng.standard_normal((int(steps), int(batch_size), int(d)))
    return base * sigma_arr


def hash_noise_bank(noise_bank: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(noise_bank, dtype=np.float64).tobytes()).hexdigest()


def _decompose(vec: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = np.asarray(P, dtype=float).T @ np.asarray(vec, dtype=float)
    parallel = np.asarray(P, dtype=float) @ R
    orth = np.asarray(vec, dtype=float) - parallel
    return R, parallel, orth


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(float(value), float(lo), float(hi)))


def _coordinate_psi(R: np.ndarray, eps_a: float) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    return R / (np.abs(R) + float(eps_a))


def _method_family(method: str) -> str:
    if method in {"fira_raw", "fira_clipped", "proj_baseline"}:
        return "fira_style_coordinate_diagnostic"
    if method in {"becr", "rho_no_lower_bound", "coupled_unbounded", "wrong_units_naive_ef"}:
        return "corrected_residual_coordinate_diagnostic"
    raise ValueError(f"unknown method={method}")


def _adam_limiter_method_family(method: str) -> str:
    if method in {
        "projected_adam_baseline",
        "fira_style_adam_limiter_no_residual",
        "clipping_only_limiter",
        "becr_effective_signal_residual",
        "wrong_pre_limiter_residual",
        "no_lower_bound_residual",
    }:
        return "adam_limiter_recovery_diagnostic"
    raise ValueError(f"unknown adam-limiter method={method}")


def run_adam_limiter_trace(
    *,
    method: str,
    quad: DiagQuadratic,
    x0: np.ndarray,
    steps: int,
    schedule: ScheduleBase,
    cfg: AdamLimiterConfig,
    noise_bank: np.ndarray | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    dtype = np.dtype(cfg.dtype)
    x = np.asarray(x0, dtype=dtype).copy()
    d = int(x.shape[0])
    residual = np.zeros((d,), dtype=dtype)
    prev_u_perp_norm = 0.0
    m = np.zeros((schedule.r,), dtype=dtype)
    v = np.zeros((schedule.r,), dtype=dtype)
    bases: list[np.ndarray] = []
    params = [x.copy()]
    f_series = [quad.value(x)]
    g0 = quad.grad(x)
    grad_norm = [safe_norm(g0)]
    grad_par_norm = [math.nan]
    grad_perp_norm = [math.nan]
    update_norm = [math.nan]
    update_cos = [math.nan]
    phi_raw_series = [math.nan]
    s_raw_series = [math.nan]
    s_applied_series = [math.nan]
    rho_series = [math.nan]
    tau_series = [math.nan]
    z_norm_series = [math.nan]
    z_eff_norm_series = [math.nan]
    lost_signal_series = [math.nan]
    residual_norm_series = [0.0 if method in {"becr_effective_signal_residual", "wrong_pre_limiter_residual", "no_lower_bound_residual"} else math.nan]
    residual_err_series = [0.0 if method in {"becr_effective_signal_residual", "wrong_pre_limiter_residual", "no_lower_bound_residual"} else math.nan]
    effective_raw_transmission = [math.nan]
    cumulative_effective_transmission = [0.0]
    x_component = [float(x[0])]
    y_component = [float(x[1]) if d >= 2 else 0.0]
    m_series = [math.nan]
    v_series = [math.nan]
    psi_series = [math.nan]
    steps_json: list[dict[str, Any]] = []
    used_noise = []
    limiter_count = 0

    for t in range(int(steps)):
        g_clean = np.asarray(quad.grad(x), dtype=dtype)
        if batch_size <= 1:
            if noise_bank is None:
                noise_batch = np.zeros((1, d), dtype=dtype)
            else:
                noise_batch = np.asarray(noise_bank[t : t + 1], dtype=dtype)
        else:
            noise_batch = np.zeros((batch_size, d), dtype=dtype) if noise_bank is None else np.asarray(noise_bank[t], dtype=dtype)
        g_batch = g_clean.reshape(1, d) + noise_batch
        g_batch_matrix = np.asarray(g_batch, dtype=dtype).T
        g_used = np.asarray(g_batch, dtype=dtype).mean(axis=0)
        used_noise.append(np.asarray(noise_batch, dtype=dtype).copy())

        P = np.asarray(schedule.basis(t=t, x=x.copy(), g_batch=g_batch_matrix.copy()), dtype=dtype)
        bases.append(P.copy())
        A = g_used + residual
        R, _, Q = _decompose(A, P)
        _, g_parallel_clean, g_perp_clean = _decompose(g_clean, P)

        m = float(cfg.beta1) * np.asarray(m, dtype=dtype) + (1.0 - float(cfg.beta1)) * np.asarray(R, dtype=dtype)
        v = float(cfg.beta2) * np.asarray(v, dtype=dtype) + (1.0 - float(cfg.beta2)) * np.square(np.asarray(R, dtype=dtype))
        psi_R = np.asarray(m, dtype=dtype) / (np.sqrt(np.asarray(v, dtype=dtype)) + float(cfg.eps_a))
        u_parallel = np.asarray(P, dtype=dtype) @ psi_R
        phi_raw = float(safe_norm(psi_R) / (safe_norm(R) + float(cfg.eps_s)))
        s_raw = float(phi_raw)

        if method == "projected_adam_baseline":
            s_applied = 0.0
            rho_t = math.nan
            tau_t = 1.0
            Z = np.zeros_like(Q)
            Z_eff = np.zeros_like(Q)
            u_perp = np.zeros_like(Q)
            residual_next = np.zeros_like(residual)
            residual_error = 0.0
        else:
            if method == "fira_style_adam_limiter_no_residual":
                s_applied = float(phi_raw)
                rho_t = math.nan
            else:
                s_applied = _clip(phi_raw, cfg.s_min, cfg.s_max)
                if method == "clipping_only_limiter":
                    rho_t = math.nan
                elif method == "becr_effective_signal_residual":
                    rho_t = _clip(cfg.rho, cfg.rho_min, cfg.rho_max)
                elif method == "wrong_pre_limiter_residual":
                    rho_t = _clip(cfg.rho, cfg.rho_min, cfg.rho_max)
                elif method == "no_lower_bound_residual":
                    rho_t = float(min(phi_raw, float(cfg.rho_max)))
                else:
                    raise ValueError(f"unsupported adam-limiter method={method}")

            Z = np.asarray(Q, dtype=dtype).copy() if not math.isfinite(float(rho_t)) else float(rho_t) * np.asarray(Q, dtype=dtype)
            tau_t = 1.0
            raw_u_perp = s_applied * Z
            raw_u_perp_norm = safe_norm(raw_u_perp)
            if cfg.limiter_gamma is not None and prev_u_perp_norm > 0.0 and raw_u_perp_norm > float(cfg.limiter_gamma) * prev_u_perp_norm:
                tau_t = float(float(cfg.limiter_gamma) * prev_u_perp_norm / raw_u_perp_norm)
                limiter_count += 1
            Z_eff = tau_t * Z
            u_perp = s_applied * Z_eff
            if method == "becr_effective_signal_residual":
                residual_next = np.asarray(Q, dtype=dtype) - Z_eff
                residual_error = safe_norm(residual_next - (np.asarray(Q, dtype=dtype) - Z_eff))
            elif method == "wrong_pre_limiter_residual":
                residual_next = np.asarray(Q, dtype=dtype) - Z
                residual_error = safe_norm(residual_next - (np.asarray(Q, dtype=dtype) - Z_eff))
            elif method == "no_lower_bound_residual":
                residual_next = np.asarray(Q, dtype=dtype) - Z_eff
                residual_error = safe_norm(residual_next - (np.asarray(Q, dtype=dtype) - Z_eff))
            else:
                residual_next = np.zeros_like(residual)
                residual_error = 0.0
        prev_u_perp_norm = safe_norm(u_perp)

        u = np.asarray(u_parallel, dtype=dtype) + np.asarray(u_perp, dtype=dtype)
        x_next = np.asarray(x, dtype=dtype) - float(cfg.lr) * u
        g_clean_next = np.asarray(quad.grad(x_next), dtype=dtype)
        lost_signal_norm = safe_norm(np.asarray(Z, dtype=dtype) - np.asarray(Z_eff, dtype=dtype))
        effective_transmission = (safe_norm(Z_eff) / safe_norm(Q)) if safe_norm(Q) > 0.0 else None
        cumulative_effective_transmission_value = cumulative_effective_transmission[-1] + (0.0 if effective_transmission is None or not math.isfinite(float(effective_transmission)) else float(effective_transmission))

        steps_json.append(
            {
                "step_index": int(t),
                "x_before": x.copy().tolist(),
                "x_after": x_next.copy().tolist(),
                "g_clean": g_clean.copy().tolist(),
                "g_used": g_used.copy().tolist(),
                "A": A.copy().tolist(),
                "P": P.copy().tolist(),
                "R": np.asarray(R, dtype=dtype).copy().tolist(),
                "Q": np.asarray(Q, dtype=dtype).copy().tolist(),
                "m": np.asarray(m, dtype=dtype).copy().tolist(),
                "v": np.asarray(v, dtype=dtype).copy().tolist(),
                "psi": np.asarray(psi_R, dtype=dtype).copy().tolist(),
                "u_parallel": np.asarray(u_parallel, dtype=dtype).copy().tolist(),
                "u_perp": np.asarray(u_perp, dtype=dtype).copy().tolist(),
                "phi_raw": float(phi_raw),
                "s_raw": float(s_raw),
                "s_applied": float(s_applied),
                "rho": None if not math.isfinite(float(rho_t)) else float(rho_t),
                "tau": float(tau_t),
                "Z_norm": safe_norm(Z),
                "Z_eff_norm": safe_norm(Z_eff),
                "lost_raw_signal_norm": lost_signal_norm,
                "effective_raw_transmission": effective_transmission,
                "residual_norm": safe_norm(residual_next),
                "residual_update_error": float(residual_error),
                "limiter_active": bool(abs(float(tau_t) - 1.0) > 1e-12),
                "g_parallel_clean_norm": safe_norm(g_parallel_clean),
                "g_perp_clean_norm": safe_norm(g_perp_clean),
                "cumulative_effective_transmission": float(cumulative_effective_transmission_value),
            }
        )

        x = np.asarray(x_next, dtype=dtype)
        residual = np.asarray(residual_next, dtype=dtype)
        params.append(x.copy())
        f_series.append(quad.value(x))
        grad_norm.append(safe_norm(g_clean_next))
        grad_par_norm.append(safe_norm(g_parallel_clean))
        grad_perp_norm.append(safe_norm(g_perp_clean))
        update_norm.append(safe_norm(u))
        update_cos.append(cosine(u, g_clean))
        phi_raw_series.append(float(phi_raw))
        s_raw_series.append(float(s_raw))
        s_applied_series.append(float(s_applied))
        rho_series.append(float(rho_t) if math.isfinite(float(rho_t)) else math.nan)
        tau_series.append(float(tau_t))
        z_norm_series.append(safe_norm(Z))
        z_eff_norm_series.append(safe_norm(Z_eff))
        lost_signal_series.append(lost_signal_norm)
        residual_norm_series.append(safe_norm(residual))
        residual_err_series.append(float(residual_error))
        effective_raw_transmission.append(math.nan if effective_transmission is None else float(effective_transmission))
        cumulative_effective_transmission.append(float(cumulative_effective_transmission_value))
        x_component.append(float(x[0]))
        y_component.append(float(x[1]) if d >= 2 else 0.0)
        m_series.append(float(np.asarray(m, dtype=float).reshape(-1)[0]) if np.asarray(m).size else math.nan)
        v_series.append(float(np.asarray(v, dtype=float).reshape(-1)[0]) if np.asarray(v).size else math.nan)
        psi_series.append(float(np.asarray(psi_R, dtype=float).reshape(-1)[0]) if np.asarray(psi_R).size else math.nan)

    noise_used = np.asarray(used_noise, dtype=dtype)
    basis_overlap = [None]
    for idx in range(1, len(bases)):
        prev = np.asarray(bases[idx - 1], dtype=float)
        cur = np.asarray(bases[idx], dtype=float)
        overlap = cur.T @ prev
        denom = math.sqrt(float(min(cur.shape[1], prev.shape[1])))
        basis_overlap.append(float(np.linalg.norm(overlap, ord="fro") / denom) if denom > 0 else None)

    return {
        "method": method,
        "method_family": _adam_limiter_method_family(method),
        "projection_mode": schedule.projection_mode,
        "scheduler_factory_id": schedule.scheduler_factory_id,
        "scheduler_object_id": schedule.scheduler_object_id,
        "noise_hash": hash_noise_bank(noise_used),
        "consumed_noise_hash": hash_noise_bank(noise_used),
        "steps": steps_json,
        "series": {
            "f": f_series,
            "grad_norm": grad_norm,
            "grad_par_norm": grad_par_norm,
            "grad_perp_norm": grad_perp_norm,
            "update_norm": update_norm,
            "update_cos": update_cos,
            "phi_raw": phi_raw_series,
            "s_raw": s_raw_series,
            "s_applied": s_applied_series,
            "rho": rho_series,
            "tau": tau_series,
            "Z_norm": z_norm_series,
            "Z_eff_norm": z_eff_norm_series,
            "lost_raw_signal_norm": lost_signal_series,
            "effective_raw_transmission": effective_raw_transmission,
            "cumulative_effective_transmission": cumulative_effective_transmission,
            "residual_norm": residual_norm_series,
            "residual_update_error": residual_err_series,
            "phi_cum": np.nancumsum(np.asarray(phi_raw_series, dtype=float)).tolist(),
            "s_applied_cum": np.nancumsum(np.asarray(s_applied_series, dtype=float)).tolist(),
            "x_component": x_component,
            "y_component": y_component,
            "basis_overlap": basis_overlap,
            "m": m_series,
            "v": v_series,
            "psi": psi_series,
        },
        "final_state": {
            "x": np.asarray(params[-1], dtype=float).tolist(),
            "f": float(f_series[-1]),
            "grad_norm": float(grad_norm[-1]),
            "grad_par_norm": float(grad_par_norm[-1]),
            "grad_perp_norm": float(grad_perp_norm[-1]),
            "phi_cum": float(np.nansum(np.asarray(phi_raw_series, dtype=float))),
            "s_applied_cum": float(np.nansum(np.asarray(s_applied_series, dtype=float))),
            "residual_norm": float(residual_norm_series[-1]) if math.isfinite(float(residual_norm_series[-1])) else None,
        },
        "limiter_count": int(limiter_count),
        "params": np.asarray(params, dtype=float),
        "bases": np.asarray(bases, dtype=float) if bases else np.zeros((0, d, schedule.r), dtype=float),
    }


def run_coordinate_trace(
    *,
    method: str,
    quad: DiagQuadratic,
    x0: np.ndarray,
    steps: int,
    schedule: ScheduleBase,
    cfg: CoordinateConfig,
    noise_bank: np.ndarray | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    dtype = np.dtype(cfg.dtype)
    x = np.asarray(x0, dtype=dtype).copy()
    d = int(x.shape[0])
    residual = np.zeros((d,), dtype=dtype)
    prev_u_perp_norm = 0.0
    bases: list[np.ndarray] = []
    params = [x.copy()]
    f_series = [quad.value(x)]
    g0 = quad.grad(x)
    grad_norm = [safe_norm(g0)]
    grad_par_norm = [math.nan]
    grad_perp_norm = [math.nan]
    update_norm = [math.nan]
    update_cos = [math.nan]
    phi_raw_series = [math.nan]
    s_raw_series = [math.nan]
    s_applied_series = [math.nan]
    rho_series = [math.nan]
    tau_series = [math.nan]
    z_norm_series = [math.nan]
    z_eff_norm_series = [math.nan]
    residual_norm_series = [0.0 if method in {"becr", "rho_no_lower_bound", "coupled_unbounded", "wrong_units_naive_ef"} else math.nan]
    residual_err_series = [0.0 if method in {"becr", "rho_no_lower_bound", "coupled_unbounded", "wrong_units_naive_ef"} else math.nan]
    effective_raw_transmission = [math.nan]
    x_component = [float(x[0])]
    y_component = [float(x[1]) if d >= 2 else 0.0]
    steps_json: list[dict[str, Any]] = []
    used_noise = []
    limiter_count = 0

    for t in range(int(steps)):
        g_clean = np.asarray(quad.grad(x), dtype=dtype)
        if batch_size <= 1:
            if noise_bank is None:
                noise_batch = np.zeros((1, d), dtype=dtype)
            else:
                noise_batch = np.asarray(noise_bank[t : t + 1], dtype=dtype)
        else:
            noise_batch = np.zeros((batch_size, d), dtype=dtype) if noise_bank is None else np.asarray(noise_bank[t], dtype=dtype)
        g_batch = g_clean.reshape(1, d) + noise_batch
        g_batch_matrix = np.asarray(g_batch, dtype=dtype).T
        g_used = np.asarray(g_batch, dtype=dtype).mean(axis=0)
        used_noise.append(np.asarray(noise_batch, dtype=dtype).copy())

        P = np.asarray(schedule.basis(t=t, x=x.copy(), g_batch=g_batch_matrix.copy()), dtype=dtype)
        bases.append(P.copy())
        if method in {"becr", "rho_no_lower_bound", "coupled_unbounded", "wrong_units_naive_ef"}:
            A = g_used + residual
        else:
            A = g_used.copy()

        R, A_parallel, Q = _decompose(A, P)
        _, g_parallel_clean, g_perp_clean = _decompose(g_clean, P)
        psi_R = _coordinate_psi(R, cfg.eps_a)
        u_parallel = np.asarray(P, dtype=dtype) @ psi_R
        phi_raw = float(safe_norm(psi_R) / (safe_norm(R) + float(cfg.eps_s)))
        s_raw = float(phi_raw)

        if method == "proj_baseline":
            s_applied = 0.0
            rho_t = math.nan
            tau_t = 1.0
            Z = np.zeros_like(Q)
            Z_eff = np.zeros_like(Q)
            u_perp = np.zeros_like(Q)
            residual_next = np.zeros_like(residual)
            residual_error = 0.0
        elif method == "fira_raw":
            s_applied = float(phi_raw)
            rho_t = math.nan
            tau_t = 1.0
            Z = np.asarray(Q, dtype=dtype).copy()
            Z_eff = Z.copy()
            u_perp = s_applied * Z_eff
            residual_next = np.zeros_like(residual)
            residual_error = 0.0
        elif method == "fira_clipped":
            s_applied = _clip(phi_raw, cfg.s_min, cfg.s_max)
            rho_t = math.nan
            tau_t = 1.0
            Z = np.asarray(Q, dtype=dtype).copy()
            Z_eff = Z.copy()
            u_perp = s_applied * Z_eff
            residual_next = np.zeros_like(residual)
            residual_error = 0.0
        else:
            s_applied = _clip(phi_raw, cfg.s_min, cfg.s_max)
            if method == "becr":
                rho_t = _clip(cfg.rho, cfg.rho_min, cfg.rho_max)
            elif method == "rho_no_lower_bound":
                rho_t = float(min(phi_raw, float(cfg.rho_max)))
            elif method == "coupled_unbounded":
                rho_t = float(phi_raw)
            elif method == "wrong_units_naive_ef":
                rho_t = _clip(cfg.rho, cfg.rho_min, cfg.rho_max)
            else:
                raise ValueError(f"unsupported method={method}")
            Z = float(rho_t) * np.asarray(Q, dtype=dtype)
            tau_t = 1.0
            raw_u_perp = s_applied * Z
            raw_u_perp_norm = safe_norm(raw_u_perp)
            if cfg.limiter_gamma is not None and prev_u_perp_norm > 0.0 and raw_u_perp_norm > float(cfg.limiter_gamma) * prev_u_perp_norm:
                tau_t = float(float(cfg.limiter_gamma) * prev_u_perp_norm / raw_u_perp_norm)
                limiter_count += 1
            Z_eff = tau_t * Z
            u_perp = s_applied * Z_eff
            if method == "wrong_units_naive_ef":
                residual_next = np.asarray(Q, dtype=dtype) + residual - u_perp
                residual_error = safe_norm(residual_next - (np.asarray(Q, dtype=dtype) - Z_eff))
            else:
                residual_next = np.asarray(Q, dtype=dtype) - Z_eff
                residual_error = safe_norm(residual_next - (np.asarray(Q, dtype=dtype) - Z_eff))
        prev_u_perp_norm = safe_norm(u_perp)

        u = np.asarray(u_parallel, dtype=dtype) + np.asarray(u_perp, dtype=dtype)
        x_next = np.asarray(x, dtype=dtype) - float(cfg.lr) * u
        g_clean_next = np.asarray(quad.grad(x_next), dtype=dtype)

        steps_json.append(
            {
                "step_index": int(t),
                "x_before": x.copy().tolist(),
                "x_after": x_next.copy().tolist(),
                "g_clean": g_clean.copy().tolist(),
                "g_used": g_used.copy().tolist(),
                "A": A.copy().tolist(),
                "P": P.copy().tolist(),
                "R": np.asarray(R, dtype=dtype).copy().tolist(),
                "Q": np.asarray(Q, dtype=dtype).copy().tolist(),
                "psi": np.asarray(psi_R, dtype=dtype).copy().tolist(),
                "u_parallel": np.asarray(u_parallel, dtype=dtype).copy().tolist(),
                "u_perp": np.asarray(u_perp, dtype=dtype).copy().tolist(),
                "phi_raw": float(phi_raw),
                "s_raw": float(s_raw),
                "s_applied": float(s_applied),
                "rho": None if not math.isfinite(float(rho_t)) else float(rho_t),
                "tau": float(tau_t),
                "Z_norm": safe_norm(Z),
                "Z_eff_norm": safe_norm(Z_eff),
                "effective_raw_transmission": (safe_norm(Z_eff) / safe_norm(Q)) if safe_norm(Q) > 0.0 else None,
                "residual_norm": safe_norm(residual_next),
                "residual_update_error": float(residual_error),
                "limiter_active": bool(abs(float(tau_t) - 1.0) > 1e-12),
                "g_parallel_clean_norm": safe_norm(g_parallel_clean),
                "g_perp_clean_norm": safe_norm(g_perp_clean),
            }
        )

        x = np.asarray(x_next, dtype=dtype)
        residual = np.asarray(residual_next, dtype=dtype)
        params.append(x.copy())
        f_series.append(quad.value(x))
        grad_norm.append(safe_norm(g_clean_next))
        grad_par_norm.append(safe_norm(g_parallel_clean))
        grad_perp_norm.append(safe_norm(g_perp_clean))
        update_norm.append(safe_norm(u))
        update_cos.append(cosine(u, g_clean))
        phi_raw_series.append(float(phi_raw))
        s_raw_series.append(float(s_raw))
        s_applied_series.append(float(s_applied))
        rho_series.append(float(rho_t) if math.isfinite(float(rho_t)) else math.nan)
        tau_series.append(float(tau_t))
        z_norm_series.append(safe_norm(Z))
        z_eff_norm_series.append(safe_norm(Z_eff))
        residual_norm_series.append(safe_norm(residual))
        residual_err_series.append(float(residual_error))
        effective_raw_transmission.append((safe_norm(Z_eff) / safe_norm(Q)) if safe_norm(Q) > 0.0 else math.nan)
        x_component.append(float(x[0]))
        y_component.append(float(x[1]) if d >= 2 else 0.0)

    noise_used = np.asarray(used_noise, dtype=dtype)
    basis_overlap = [None]
    for idx in range(1, len(bases)):
        prev = np.asarray(bases[idx - 1], dtype=float)
        cur = np.asarray(bases[idx], dtype=float)
        overlap = cur.T @ prev
        denom = math.sqrt(float(min(cur.shape[1], prev.shape[1])))
        basis_overlap.append(float(np.linalg.norm(overlap, ord="fro") / denom) if denom > 0 else None)

    return {
        "method": method,
        "method_family": _method_family(method),
        "projection_mode": schedule.projection_mode,
        "scheduler_factory_id": schedule.scheduler_factory_id,
        "scheduler_object_id": schedule.scheduler_object_id,
        "noise_hash": hash_noise_bank(noise_used),
        "consumed_noise_hash": hash_noise_bank(noise_used),
        "steps": steps_json,
        "series": {
            "f": f_series,
            "grad_norm": grad_norm,
            "grad_par_norm": grad_par_norm,
            "grad_perp_norm": grad_perp_norm,
            "update_norm": update_norm,
            "update_cos": update_cos,
            "phi_raw": phi_raw_series,
            "s_raw": s_raw_series,
            "s_applied": s_applied_series,
            "rho": rho_series,
            "tau": tau_series,
            "Z_norm": z_norm_series,
            "Z_eff_norm": z_eff_norm_series,
            "effective_raw_transmission": effective_raw_transmission,
            "residual_norm": residual_norm_series,
            "residual_update_error": residual_err_series,
            "phi_cum": np.nancumsum(np.asarray(phi_raw_series, dtype=float)).tolist(),
            "s_applied_cum": np.nancumsum(np.asarray(s_applied_series, dtype=float)).tolist(),
            "x_component": x_component,
            "y_component": y_component,
            "basis_overlap": basis_overlap,
        },
        "final_state": {
            "x": np.asarray(params[-1], dtype=float).tolist(),
            "f": float(f_series[-1]),
            "grad_norm": float(grad_norm[-1]),
            "grad_par_norm": float(grad_par_norm[-1]),
            "grad_perp_norm": float(grad_perp_norm[-1]),
            "phi_cum": float(np.nansum(np.asarray(phi_raw_series, dtype=float))),
            "s_applied_cum": float(np.nansum(np.asarray(s_applied_series, dtype=float))),
            "residual_norm": float(residual_norm_series[-1]) if math.isfinite(float(residual_norm_series[-1])) else None,
        },
        "limiter_count": int(limiter_count),
        "params": np.asarray(params, dtype=float),
        "bases": np.asarray(bases, dtype=float) if bases else np.zeros((0, d, schedule.r), dtype=float),
    }


def run_moving_projection_dynamic_trace(
    *,
    mode: str,
    quad: DiagQuadratic,
    x0: np.ndarray,
    steps: int,
    schedule: ScheduleBase,
    cfg: MovingProjectionConfig,
    noise_bank: np.ndarray | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    dtype = np.dtype(cfg.dtype)
    x = np.asarray(x0, dtype=dtype).copy()
    state: dict[str, Any] | None = None
    prev_P: np.ndarray | None = None
    used_noise = []
    params = [x.copy()]
    f_series = [quad.value(x)]
    grad_norm = [safe_norm(quad.grad(x))]
    grad_par_norm = [math.nan]
    grad_perp_norm = [math.nan]
    residual_norm = [0.0]
    residual_update_error = [0.0]
    basis_overlap = [None]
    reset_events = []
    transport_error_first = [0.0]
    transport_error_second = [0.0]
    steps_json = []

    for t in range(int(steps)):
        g_clean = np.asarray(quad.grad(x), dtype=dtype)
        if batch_size <= 1:
            noise_batch = np.zeros((1, len(x)), dtype=dtype) if noise_bank is None else np.asarray(noise_bank[t : t + 1], dtype=dtype)
        else:
            noise_batch = np.zeros((batch_size, len(x)), dtype=dtype) if noise_bank is None else np.asarray(noise_bank[t], dtype=dtype)
        g_batch = g_clean.reshape(1, len(x)) + noise_batch
        g_batch_matrix = np.asarray(g_batch, dtype=dtype).T
        g_used = np.asarray(g_batch, dtype=dtype).mean(axis=0)
        used_noise.append(np.asarray(noise_batch, dtype=dtype).copy())

        P = np.asarray(schedule.basis(t=t, x=x.copy(), g_batch=g_batch_matrix.copy()), dtype=dtype)
        result = run_single_step(
            mode=mode,
            cfg=cfg,
            g=g_used,
            P=P,
            prev_P=prev_P,
            initial_state=state,
            step_index=t,
        )
        u = np.asarray(result["u"], dtype=dtype)
        x_next = np.asarray(x, dtype=dtype) - float(cfg.lr) * u
        g_clean_next = np.asarray(quad.grad(x_next), dtype=dtype)
        _, g_par_clean, g_perp_clean = _decompose(g_clean_next, P)

        step_row = {
            "step_index": int(t),
            "x_before": x.copy().tolist(),
            "x_after": x_next.copy().tolist(),
            "g_clean": g_clean.copy().tolist(),
            "g_used": g_used.copy().tolist(),
            "basis_overlap": result["basis_overlap"],
            "reset_event": bool(result["reset_event"]),
            "refresh_happened": bool(result["refresh_happened"]),
            "first_moment_transport_error": float(result["first_moment_transport_error"]),
            "second_moment_transport_error": float(result["second_moment_transport_error"]),
            "decomposition_error": float(result["decomposition_error"]),
            "residual_update_error": float(result["residual_error"]),
            "phi_raw": float(result["phi_raw"]),
            "s_raw": float(result["phi_raw"]),
            "s_applied": float(result["s"]),
            "rho": None if not math.isfinite(float(result["rho"])) else float(result["rho"]),
            "tau": float(result["tau"]),
            "residual_norm": float(result["state_after_step"]["e_norm"]),
            "g_parallel_clean_norm": safe_norm(g_par_clean),
            "g_perp_clean_norm": safe_norm(g_perp_clean),
        }
        steps_json.append(step_row)

        if bool(result["reset_event"]):
            reset_events.append(int(t))
        basis_overlap.append(result["basis_overlap"] if result["basis_overlap"] == result["basis_overlap"] else None)
        transport_error_first.append(float(result["first_moment_transport_error"]))
        transport_error_second.append(float(result["second_moment_transport_error"]))

        x = np.asarray(x_next, dtype=dtype)
        state = result["state_after_step"]
        prev_P = np.asarray(P, dtype=dtype)
        params.append(x.copy())
        f_series.append(quad.value(x))
        grad_norm.append(safe_norm(g_clean_next))
        grad_par_norm.append(safe_norm(g_par_clean))
        grad_perp_norm.append(safe_norm(g_perp_clean))
        residual_norm.append(float(result["state_after_step"]["e_norm"]))
        residual_update_error.append(float(result["residual_error"]))

    return {
        "mode": mode,
        "projection_mode": schedule.projection_mode,
        "scheduler_factory_id": schedule.scheduler_factory_id,
        "scheduler_object_id": schedule.scheduler_object_id,
        "noise_hash": hash_noise_bank(np.asarray(used_noise, dtype=dtype)),
        "consumed_noise_hash": hash_noise_bank(np.asarray(used_noise, dtype=dtype)),
        "steps": steps_json,
        "series": {
            "f": f_series,
            "grad_norm": grad_norm,
            "grad_par_norm": grad_par_norm,
            "grad_perp_norm": grad_perp_norm,
            "residual_norm": residual_norm,
            "residual_update_error": residual_update_error,
            "basis_overlap": basis_overlap,
            "first_moment_transport_error": transport_error_first,
            "second_moment_transport_error": transport_error_second,
        },
        "final_state": {
            "x": np.asarray(params[-1], dtype=float).tolist(),
            "f": float(f_series[-1]),
            "grad_norm": float(grad_norm[-1]),
            "grad_par_norm": float(grad_par_norm[-1]),
            "grad_perp_norm": float(grad_perp_norm[-1]),
            "residual_norm": float(residual_norm[-1]),
        },
        "reset_events": reset_events,
    }


def confidence_interval(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, mean, mean
    std = float(arr.std(ddof=1))
    half = 1.96 * std / math.sqrt(float(arr.size))
    return mean, mean - half, mean + half


def wall_clock_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
