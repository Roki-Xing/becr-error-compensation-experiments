from __future__ import annotations

import hashlib
import math
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np


MODE_SPECS = {
    "state_reset_explicit": {
        "projection_mode": "provided_schedule",
        "state_transport_mode": "reset_on_refresh",
        "residual_mode": "current_projection_compensated",
        "explicit_reset": True,
        "use_residual": True,
    },
    "official_fira_carry": {
        "projection_mode": "provided_schedule",
        "state_transport_mode": "carry_unchanged",
        "residual_mode": "none",
        "explicit_reset": False,
        "use_residual": False,
    },
    "projection_aware_transport": {
        "projection_mode": "provided_schedule",
        "state_transport_mode": "basis_overlap_transport",
        "residual_mode": "none",
        "explicit_reset": False,
        "use_residual": False,
    },
    "full_residual_current_projection": {
        "projection_mode": "provided_schedule",
        "state_transport_mode": "basis_overlap_transport",
        "residual_mode": "current_projection_compensated",
        "explicit_reset": False,
        "use_residual": True,
    },
}

REQUIRED_MANIFEST_FIELDS = {
    "mode",
    "projection_mode",
    "state_transport_mode",
    "residual_mode",
    "reset_events",
    "basis_overlap",
    "rng_seed",
    "noise_hash",
    "scheduler_factory_id",
    "code_commit",
    "parent_task_id",
}


@dataclass(frozen=True)
class MovingProjectionConfig:
    lr: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps_adam: float = 1e-8
    eps_scale: float = 1e-8
    s_min: float = 0.1
    s_max: float = 10.0
    rho: float = 0.5
    rho_min: float = 1e-6
    rho_max: float = 1.0
    limiter_gamma: float | None = None
    dtype: Any = np.float64


def make_basis(d: int, indices: list[int], *, dtype=np.float64) -> np.ndarray:
    basis = np.zeros((int(d), len(indices)), dtype=dtype)
    for j, idx in enumerate(indices):
        basis[int(idx), j] = 1.0
    return basis


def _as_dtype(x: np.ndarray | list[float], dtype) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _basis_overlap(P: np.ndarray | None, prev_P: np.ndarray | None) -> float:
    if P is None or prev_P is None:
        return float("nan")
    overlap = np.asarray(P, dtype=float).T @ np.asarray(prev_P, dtype=float)
    denom = math.sqrt(float(min(P.shape[1], prev_P.shape[1])))
    if denom <= 0.0:
        return float("nan")
    return float(np.linalg.norm(overlap, ord="fro") / denom)


def _clip_scale(phi_raw: float, cfg: MovingProjectionConfig) -> float:
    return float(np.clip(phi_raw, float(cfg.s_min), float(cfg.s_max)))


def _pad_or_trim(vec: np.ndarray, size: int, *, fill: float = 0.0) -> np.ndarray:
    out = np.full((int(size),), fill, dtype=vec.dtype)
    n = min(len(vec), int(size))
    if n > 0:
        out[:n] = vec[:n]
    return out


def _transport_state(
    *,
    mode: str,
    cfg: MovingProjectionConfig,
    P: np.ndarray,
    prev_P: np.ndarray | None,
    state: dict[str, Any],
    refresh_happened: bool,
) -> tuple[dict[str, Any], bool]:
    spec = MODE_SPECS[mode]
    k = int(P.shape[1])
    d = int(P.shape[0])
    dtype = np.dtype(cfg.dtype)
    state = {
        "m": _as_dtype(state.get("m", np.zeros((k,), dtype=dtype)), dtype),
        "v": _as_dtype(state.get("v", np.zeros((k,), dtype=dtype)), dtype),
        "t": int(state.get("t", 0)),
        "e": _as_dtype(state.get("e", np.zeros((d,), dtype=dtype)), dtype),
        "prev_u_perp_norm": float(state.get("prev_u_perp_norm", 0.0)),
    }
    if state["m"].shape[0] != k:
        state["m"] = _pad_or_trim(state["m"], k)
    if state["v"].shape[0] != k:
        state["v"] = _pad_or_trim(state["v"], k)
    if state["e"].shape[0] != d:
        state["e"] = _pad_or_trim(state["e"], d)
    if prev_P is None:
        return state, False
    if not refresh_happened:
        return state, False
    if spec["explicit_reset"]:
        return {
            "m": np.zeros((k,), dtype=dtype),
            "v": np.zeros((k,), dtype=dtype),
            "t": 0,
            "e": np.zeros((d,), dtype=dtype),
            "prev_u_perp_norm": 0.0,
        }, True
    transport_mode = spec["state_transport_mode"]
    if transport_mode == "carry_unchanged":
        return {
            "m": _pad_or_trim(state["m"], k),
            "v": _pad_or_trim(state["v"], k),
            "t": int(state["t"]),
            "e": np.zeros((d,), dtype=dtype),
            "prev_u_perp_norm": float(state["prev_u_perp_norm"]),
        }, False
    if transport_mode == "basis_overlap_transport":
        overlap = np.asarray(P, dtype=dtype).T @ np.asarray(prev_P, dtype=dtype)
        m_new = overlap @ state["m"]
        v_new = (overlap * overlap) @ state["v"]
        return {
            "m": np.asarray(m_new, dtype=dtype),
            "v": np.asarray(v_new, dtype=dtype),
            "t": int(state["t"]),
            "e": np.asarray(state["e"], dtype=dtype),
            "prev_u_perp_norm": float(state["prev_u_perp_norm"]),
        }, False
    raise ValueError(f"unsupported state transport mode: {transport_mode}")


def run_single_step(
    *,
    mode: str,
    cfg: MovingProjectionConfig,
    g: np.ndarray,
    P: np.ndarray,
    prev_P: np.ndarray | None,
    initial_state: dict[str, Any] | None,
    step_index: int,
) -> dict[str, Any]:
    if mode not in MODE_SPECS:
        raise ValueError(f"unknown mode={mode}")
    dtype = np.dtype(cfg.dtype)
    g = _as_dtype(g, dtype)
    P = _as_dtype(P, dtype)
    init = initial_state or {}
    refresh_happened = prev_P is None or not np.allclose(P, prev_P)
    state, reset_event = _transport_state(
        mode=mode,
        cfg=cfg,
        P=P,
        prev_P=prev_P,
        state=init,
        refresh_happened=refresh_happened,
    )
    state_before = {
        "m": np.asarray(state["m"], dtype=dtype).copy(),
        "v": np.asarray(state["v"], dtype=dtype).copy(),
        "t": int(state["t"]),
        "e": np.asarray(state["e"], dtype=dtype).copy(),
        "prev_u_perp_norm": float(state["prev_u_perp_norm"]),
    }

    if MODE_SPECS[mode]["residual_mode"] == "current_projection_compensated":
        A = g + state_before["e"]
    else:
        A = g.copy()
    R = P.T @ A
    reconstructed = P @ R
    Q = A - reconstructed

    m = np.asarray(state_before["m"], dtype=dtype).copy()
    v = np.asarray(state_before["v"], dtype=dtype).copy()
    beta1 = float(cfg.beta1)
    beta2 = float(cfg.beta2)
    m = beta1 * m + (1.0 - beta1) * R
    v = beta2 * v + (1.0 - beta2) * (R * R)
    t = int(state_before["t"]) + 1
    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)
    psi = m_hat / (np.sqrt(v_hat) + float(cfg.eps_adam))
    u_parallel = P @ psi
    phi_raw = float(np.linalg.norm(psi) / (np.linalg.norm(R) + float(cfg.eps_scale)))
    s_t = _clip_scale(phi_raw, cfg)

    if MODE_SPECS[mode]["residual_mode"] == "current_projection_compensated":
        rho_t = float(np.clip(float(cfg.rho), float(cfg.rho_min), float(cfg.rho_max)))
        Z = rho_t * Q
        tau = 1.0
        raw_u_perp = s_t * Z
        raw_u_perp_norm = float(np.linalg.norm(raw_u_perp))
        prev_u_norm = float(state_before["prev_u_perp_norm"])
        if (
            cfg.limiter_gamma is not None
            and prev_u_norm > 0.0
            and raw_u_perp_norm > float(cfg.limiter_gamma) * prev_u_norm
            and raw_u_perp_norm > 0.0
        ):
            tau = float(float(cfg.limiter_gamma) * prev_u_norm / raw_u_perp_norm)
        Z_eff = tau * Z
        u_perp = s_t * Z_eff
        E_next = Q - Z_eff
        residual_error = float(np.linalg.norm(E_next - (Q - Z_eff)))
        next_prev_u_perp_norm = float(np.linalg.norm(u_perp))
    else:
        rho_t = float("nan")
        Z = Q.copy()
        tau = 1.0
        Z_eff = Z.copy()
        u_perp = s_t * Z_eff
        E_next = np.zeros_like(g)
        residual_error = 0.0
        next_prev_u_perp_norm = float(np.linalg.norm(u_perp))

    u = u_parallel + u_perp
    state_after = {
        "m": m.copy(),
        "v": v.copy(),
        "t": int(t),
        "e": E_next.copy(),
        "prev_u_perp_norm": float(next_prev_u_perp_norm),
    }
    return {
        "mode": mode,
        "step_index": int(step_index),
        "refresh_happened": bool(refresh_happened),
        "reset_event": bool(reset_event),
        "basis_overlap": _basis_overlap(P, prev_P),
        "g": g,
        "A": A,
        "R": R,
        "Q": Q,
        "P": P,
        "reconstructed": reconstructed,
        "m": m,
        "v": v,
        "psi": psi,
        "phi_raw": float(phi_raw),
        "s": float(s_t),
        "rho": float(rho_t),
        "Z": Z,
        "tau": float(tau),
        "Z_eff": Z_eff,
        "E_next": E_next,
        "u_parallel": u_parallel,
        "u_perp": u_perp,
        "u": u,
        "decomposition_error": float(np.linalg.norm(A - reconstructed - Q)),
        "residual_error": float(residual_error),
        "state_before_step": {
            "m": state_before["m"].copy(),
            "v": state_before["v"].copy(),
            "t": int(state_before["t"]),
            "e": state_before["e"].copy(),
            "e_norm": float(np.linalg.norm(state_before["e"])),
            "prev_u_perp_norm": float(state_before["prev_u_perp_norm"]),
        },
        "state_after_step": {
            "m": state_after["m"].copy(),
            "v": state_after["v"].copy(),
            "t": int(state_after["t"]),
            "e": state_after["e"].copy(),
            "e_norm": float(np.linalg.norm(state_after["e"])),
            "prev_u_perp_norm": float(state_after["prev_u_perp_norm"]),
        },
    }


class StaticSchedule:
    def __init__(self, bases: list[np.ndarray], *, scheduler_factory_id: str):
        self._bases = [np.asarray(b, dtype=float).copy() for b in bases]
        self.scheduler_factory_id = str(scheduler_factory_id)
        self.steps = 0

    def basis(self, step_index: int) -> np.ndarray:
        self.steps += 1
        return np.asarray(self._bases[step_index], dtype=float).copy()


def _serialize_step(step: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in step.items():
        if isinstance(value, np.ndarray):
            out[key] = np.asarray(value).tolist()
        elif isinstance(value, dict):
            out[key] = _serialize_step(value)
        else:
            out[key] = value
    return out


def run_trace(
    *,
    mode: str,
    cfg: MovingProjectionConfig,
    gradients: list[np.ndarray],
    bases: list[np.ndarray],
    noise_bank: np.ndarray | None = None,
    rng_seed: int = 0,
    scheduler_factory_id: str | None = None,
    parent_task_id: str = "P0-MOVING-PROJECTION-STATE-002",
) -> dict[str, Any]:
    scheduler_factory_id = scheduler_factory_id or str(uuid.uuid4())
    scheduler = StaticSchedule(bases, scheduler_factory_id=scheduler_factory_id)
    dtype = np.dtype(cfg.dtype)
    steps = []
    state = None
    prev_P = None
    reset_events = []
    basis_overlap = []
    max_decomp = 0.0
    max_residual = 0.0
    used_noise = []
    for step_index, g in enumerate(gradients):
        noise = np.zeros_like(g, dtype=dtype)
        if noise_bank is not None:
            noise = _as_dtype(noise_bank[step_index], dtype)
        used_noise.append(noise.copy())
        P = _as_dtype(scheduler.basis(step_index), dtype)
        result = run_single_step(
            mode=mode,
            cfg=cfg,
            g=_as_dtype(g, dtype) + noise,
            P=P,
            prev_P=prev_P,
            initial_state=state,
            step_index=step_index,
        )
        steps.append(result)
        if result["reset_event"]:
            reset_events.append(int(step_index))
        if math.isfinite(float(result["basis_overlap"])):
            basis_overlap.append(float(result["basis_overlap"]))
        max_decomp = max(max_decomp, float(result["decomposition_error"]))
        max_residual = max(max_residual, float(result["residual_error"]))
        state = result["state_after_step"]
        prev_P = P
    noise_hash = None
    if noise_bank is not None:
        noise_hash = hashlib.sha256(np.asarray(noise_bank, dtype=dtype).tobytes()).hexdigest()
    manifest = {
        "mode": mode,
        "projection_mode": MODE_SPECS[mode]["projection_mode"],
        "state_transport_mode": MODE_SPECS[mode]["state_transport_mode"],
        "residual_mode": MODE_SPECS[mode]["residual_mode"],
        "reset_events": reset_events,
        "basis_overlap": basis_overlap,
        "rng_seed": int(rng_seed),
        "noise_hash": noise_hash,
        "scheduler_factory_id": str(scheduler_factory_id),
        "scheduler_object_id": str(id(scheduler)),
        "code_commit": _git_commit(),
        "parent_task_id": parent_task_id,
    }
    return {
        "mode": mode,
        "steps": steps,
        "manifest": manifest,
        "noise_bank": np.asarray(used_noise, dtype=dtype),
        "max_decomposition_error": float(max_decomp),
        "max_residual_error": float(max_residual),
    }


def run_method_suite(
    *,
    modes: list[str],
    cfg: MovingProjectionConfig,
    gradients: list[np.ndarray],
    bases: list[np.ndarray],
    stochastic_noise_std: float | None = None,
    rng_seed: int = 0,
    parent_task_id: str = "P0-MOVING-PROJECTION-STATE-002",
) -> dict[str, Any]:
    dtype = np.dtype(cfg.dtype)
    noise_bank = None
    if stochastic_noise_std is not None:
        rng = np.random.default_rng(int(rng_seed))
        stacked = np.stack([_as_dtype(g, dtype) for g in gradients], axis=0)
        noise_bank = rng.standard_normal(stacked.shape).astype(dtype) * float(stochastic_noise_std)
    suite = {}
    for mode in modes:
        suite[mode] = run_trace(
            mode=mode,
            cfg=cfg,
            gradients=gradients,
            bases=bases,
            noise_bank=noise_bank,
            rng_seed=rng_seed,
            scheduler_factory_id=f"{mode}-{uuid.uuid4()}",
            parent_task_id=parent_task_id,
        )
    return suite
