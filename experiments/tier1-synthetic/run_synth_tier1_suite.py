from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Write outputs next to this script (portable across Windows Python / WSL).
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
RES_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)


def _safe_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float)))


def _cosine(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= eps or nb <= eps:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _orthonormalize(P: np.ndarray) -> np.ndarray:
    # QR orthonormalization (columns).
    Q, _ = np.linalg.qr(np.asarray(P, dtype=float))
    return np.asarray(Q, dtype=float)


@dataclass(frozen=True)
class DiagQuad:
    lam: np.ndarray  # shape (d,)

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return 0.5 * float(np.dot(self.lam * x, x))

    def grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.lam * x


class PScheduler:
    def step(self, *, t: int, x: np.ndarray, g_batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class FixedP(PScheduler):
    def __init__(self, P: np.ndarray):
        self.P = np.asarray(P, dtype=float)

    def step(self, *, t: int, x: np.ndarray, g_batch: np.ndarray) -> np.ndarray:
        return self.P


class TopGradStaleRefreshP(PScheduler):
    def __init__(self, *, d: int, r: int, refresh_K: int | None):
        self.d = int(d)
        self.r = int(r)
        self.refresh_K = None if refresh_K is None else int(refresh_K)
        self._P = None

    def _make_basis(self, idx: np.ndarray) -> np.ndarray:
        P = np.zeros((self.d, len(idx)), dtype=float)
        for j, i in enumerate(idx.tolist()):
            P[int(i), j] = 1.0
        return P

    def step(self, *, t: int, x: np.ndarray, g_batch: np.ndarray) -> np.ndarray:
        # Use the batch-mean gradient to decide which coordinates to keep.
        g = np.asarray(g_batch, dtype=float).mean(axis=1)
        do_refresh = self._P is None
        if self.refresh_K is not None and self.refresh_K > 0 and (t % self.refresh_K == 0):
            do_refresh = True
        if do_refresh:
            mags = np.abs(g)
            idx = np.argsort(-mags)[: self.r]
            self._P = self._make_basis(idx)
        return self._P


class SVDBiasedP(PScheduler):
    def __init__(self, *, r: int, update_K: int, eps: float = 1e-12):
        self.r = int(r)
        self.update_K = int(update_K)
        self.eps = float(eps)
        self._P = None

    def step(self, *, t: int, x: np.ndarray, g_batch: np.ndarray) -> np.ndarray:
        # Update every K steps using top left singular vectors of the batch gradients.
        if self._P is None or (self.update_K > 0 and t % self.update_K == 0):
            G = np.asarray(g_batch, dtype=float)
            # columns = samples, but we store as (d, B) already.
            # Tall-skinny SVD; r <= B is recommended.
            U, S, _ = np.linalg.svd(G, full_matrices=False)
            r_eff = int(min(self.r, U.shape[1]))
            P = U[:, :r_eff]
            # Guard: if batch is nearly zero, fall back to identity columns.
            if float(np.max(S)) <= self.eps:
                P = np.eye(U.shape[0], r_eff, dtype=float)
            self._P = _orthonormalize(P)
        return self._P


class Optim:
    name: str

    def step(self, *, x: np.ndarray, g: np.ndarray, P: np.ndarray, t: int) -> tuple[np.ndarray, dict]:
        raise NotImplementedError


def _decompose(g: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Assume P has orthonormal columns.
    R = np.asarray(P, dtype=float).T @ np.asarray(g, dtype=float)
    g_parallel = np.asarray(P, dtype=float) @ R
    g_perp = np.asarray(g, dtype=float) - g_parallel
    return R, g_parallel, g_perp


class ProjSGDHD(Optim):
    def __init__(self, *, lr: float):
        self.name = "proj_sgd"
        self.lr = float(lr)

    def step(self, *, x: np.ndarray, g: np.ndarray, P: np.ndarray, t: int) -> tuple[np.ndarray, dict]:
        _, g_parallel, g_perp = _decompose(g, P)
        u = g_parallel
        x_next = np.asarray(x, dtype=float) - self.lr * u
        return x_next, {
            "phi_raw": float("nan"),
            "s": float("nan"),
            "phi_eff": float("nan"),
            "rho": float("nan"),
            "g_parallel_norm": _safe_norm(g_parallel),
            "g_perp_norm": _safe_norm(g_perp),
            "update_norm": _safe_norm(u),
            "e_norm": float("nan"),
        }


class FiraScalarHD(Optim):
    def __init__(
        self,
        *,
        lr: float,
        eps_a: float,
        eps_s: float,
        s_clip: tuple[float, float] | None = None,
        limiter_gamma: float | None = None,
    ):
        self.lr = float(lr)
        self.eps_a = float(eps_a)
        self.eps_s = float(eps_s)
        self.s_clip = s_clip
        self.limiter_gamma = limiter_gamma

        if limiter_gamma is not None:
            self.name = "fira_limiter"
        else:
            self.name = "fira_raw" if s_clip is None else "fira_clipped"

        self._prev_u_perp_norm = None

    def _psi(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        return R / (np.abs(R) + self.eps_a)

    def step(self, *, x: np.ndarray, g: np.ndarray, P: np.ndarray, t: int) -> tuple[np.ndarray, dict]:
        R, g_parallel, g_perp = _decompose(g, P)
        psi_R = self._psi(R)
        u_parallel = np.asarray(P, dtype=float) @ psi_R

        s_raw = float(_safe_norm(psi_R) / (_safe_norm(R) + self.eps_s))
        if self.s_clip is None:
            s_t = float(s_raw)
        else:
            s_min, s_max = self.s_clip
            s_t = float(np.clip(float(s_raw), float(s_min), float(s_max)))

        u_perp = s_t * g_perp
        u_perp_eff = np.asarray(u_perp, dtype=float)
        if self.limiter_gamma is not None:
            cur = float(np.linalg.norm(u_perp_eff))
            if (
                self._prev_u_perp_norm is not None
                and self._prev_u_perp_norm > 0
                and cur > self.limiter_gamma * self._prev_u_perp_norm
            ):
                u_perp_eff = u_perp_eff * (self.limiter_gamma * self._prev_u_perp_norm / cur)
                cur = float(np.linalg.norm(u_perp_eff))
            self._prev_u_perp_norm = cur

        u = u_parallel + u_perp_eff
        x_next = np.asarray(x, dtype=float) - self.lr * u

        gperp_norm = float(np.linalg.norm(g_perp))
        phi_eff = float(np.linalg.norm(u_perp_eff) / (gperp_norm + 1e-12))

        return x_next, {
            "phi_raw": float(s_raw),
            "s": float(s_t),
            "phi_eff": float(phi_eff),
            "rho": float("nan"),
            "g_parallel_norm": _safe_norm(g_parallel),
            "g_perp_norm": _safe_norm(g_perp),
            "update_norm": _safe_norm(u),
            "e_norm": float("nan"),
        }


class BECRFiraHD(Optim):
    def __init__(
        self,
        *,
        lr: float,
        eps_a: float,
        eps_s: float,
        rho: float,
        s_clip: tuple[float, float],
        limiter_gamma: float | None = None,
    ):
        self.name = "becr_fira"
        self.lr = float(lr)
        self.eps_a = float(eps_a)
        self.eps_s = float(eps_s)
        self.rho = float(rho)
        self.s_clip = s_clip
        self.limiter_gamma = limiter_gamma

        self.e = None  # raw-gradient units, shape (d,)
        self._prev_u_perp_norm = None

    def _psi(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        return R / (np.abs(R) + self.eps_a)

    def step(self, *, x: np.ndarray, g: np.ndarray, P: np.ndarray, t: int) -> tuple[np.ndarray, dict]:
        R, g_parallel, g_perp = _decompose(g, P)
        psi_R = self._psi(R)
        u_parallel = np.asarray(P, dtype=float) @ psi_R

        s_raw = float(_safe_norm(psi_R) / (_safe_norm(R) + self.eps_s))
        s_min, s_max = self.s_clip
        s_t = float(np.clip(float(s_raw), float(s_min), float(s_max)))

        if self.e is None:
            self.e = np.zeros_like(g_perp, dtype=float)

        h = g_perp + self.e
        z = self.rho * h
        z_eff = np.asarray(z, dtype=float)

        u_perp = s_t * z_eff
        if self.limiter_gamma is not None:
            cur = float(np.linalg.norm(u_perp))
            if (
                self._prev_u_perp_norm is not None
                and self._prev_u_perp_norm > 0
                and cur > self.limiter_gamma * self._prev_u_perp_norm
            ):
                tau = self.limiter_gamma * self._prev_u_perp_norm / cur
                z_eff = tau * z_eff
                u_perp = s_t * z_eff
                cur = float(np.linalg.norm(u_perp))
            self._prev_u_perp_norm = cur

        self.e = h - z_eff

        u = u_parallel + u_perp
        x_next = np.asarray(x, dtype=float) - self.lr * u

        gperp_norm = float(np.linalg.norm(g_perp))
        phi_eff = float(np.linalg.norm(u_perp) / (gperp_norm + 1e-12))

        return x_next, {
            "phi_raw": float(s_raw),
            "s": float(s_t),
            "phi_eff": float(phi_eff),
            "rho": float(self.rho),
            "g_parallel_norm": _safe_norm(g_parallel),
            "g_perp_norm": _safe_norm(g_perp),
            "update_norm": _safe_norm(u),
            "e_norm": float(np.linalg.norm(self.e)),
        }


class ResidualRhoNoLowerBoundCappedHD(Optim):
    # Ablation A: no lower bound, but upper bounded transmission.
    #   rho_t = min(phi_raw, rho_max), with rho_max <= 1.
    def __init__(
        self,
        *,
        lr: float,
        eps_a: float,
        eps_s: float,
        s_clip: tuple[float, float],
        rho_max: float,
    ):
        self.name = "rho_no_lower_bound"
        self.lr = float(lr)
        self.eps_a = float(eps_a)
        self.eps_s = float(eps_s)
        self.s_clip = s_clip
        self.rho_max = float(rho_max)
        self.e = None  # raw-gradient units

    def _psi(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        return R / (np.abs(R) + self.eps_a)

    def step(self, *, x: np.ndarray, g: np.ndarray, P: np.ndarray, t: int) -> tuple[np.ndarray, dict]:
        R, g_parallel, g_perp = _decompose(g, P)
        psi_R = self._psi(R)
        u_parallel = np.asarray(P, dtype=float) @ psi_R

        s_raw = float(_safe_norm(psi_R) / (_safe_norm(R) + self.eps_s))
        s_min, s_max = self.s_clip
        s_t = float(np.clip(float(s_raw), float(s_min), float(s_max)))

        if self.e is None:
            self.e = np.zeros_like(g_perp, dtype=float)

        h = g_perp + self.e
        rho_t = float(min(self.rho_max, s_raw))  # no lower bound, but upper bounded
        z = rho_t * h
        self.e = h - z
        u_perp = s_t * z

        u = u_parallel + u_perp
        x_next = np.asarray(x, dtype=float) - self.lr * u

        gperp_norm = float(np.linalg.norm(g_perp))
        phi_eff = float(np.linalg.norm(u_perp) / (gperp_norm + 1e-12))

        return x_next, {
            "phi_raw": float(s_raw),
            "s": float(s_t),
            "phi_eff": float(phi_eff),
            "rho": float(rho_t),
            "g_parallel_norm": _safe_norm(g_parallel),
            "g_perp_norm": _safe_norm(g_perp),
            "update_norm": _safe_norm(u),
            "e_norm": float(np.linalg.norm(self.e)),
        }


class ResidualCoupledUnboundedTransmissionHD(Optim):
    # Ablation B: residual transmission coupled to phi_raw, without bounds.
    #   rho_t = phi_raw. This mixes no-lower-bound + no-upper-bound effects.
    def __init__(
        self,
        *,
        lr: float,
        eps_a: float,
        eps_s: float,
        s_clip: tuple[float, float],
    ):
        self.name = "coupled_unbounded"
        self.lr = float(lr)
        self.eps_a = float(eps_a)
        self.eps_s = float(eps_s)
        self.s_clip = s_clip
        self.e = None  # raw-gradient units

    def _psi(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        return R / (np.abs(R) + self.eps_a)

    def step(self, *, x: np.ndarray, g: np.ndarray, P: np.ndarray, t: int) -> tuple[np.ndarray, dict]:
        R, g_parallel, g_perp = _decompose(g, P)
        psi_R = self._psi(R)
        u_parallel = np.asarray(P, dtype=float) @ psi_R

        s_raw = float(_safe_norm(psi_R) / (_safe_norm(R) + self.eps_s))
        s_min, s_max = self.s_clip
        s_t = float(np.clip(float(s_raw), float(s_min), float(s_max)))

        if self.e is None:
            self.e = np.zeros_like(g_perp, dtype=float)

        h = g_perp + self.e
        rho_t = float(s_raw)  # coupled/unbounded
        z = rho_t * h
        self.e = h - z
        u_perp = s_t * z

        u = u_parallel + u_perp
        x_next = np.asarray(x, dtype=float) - self.lr * u

        gperp_norm = float(np.linalg.norm(g_perp))
        phi_eff = float(np.linalg.norm(u_perp) / (gperp_norm + 1e-12))

        return x_next, {
            "phi_raw": float(s_raw),
            "s": float(s_t),
            "phi_eff": float(phi_eff),
            "rho": float(rho_t),
            "g_parallel_norm": _safe_norm(g_parallel),
            "g_perp_norm": _safe_norm(g_perp),
            "update_norm": _safe_norm(u),
            "e_norm": float(np.linalg.norm(self.e)),
        }


def run_hd_experiment(
    *,
    exp_name: str,
    quad: DiagQuad,
    P_sched: PScheduler,
    optimizers: list[Optim],
    x0: np.ndarray,
    steps: int,
    batch_size: int = 1,
    noise_sigma: np.ndarray | None = None,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(int(seed))
    d = int(x0.shape[0])

    traces: dict[str, dict[str, np.ndarray]] = {}
    for opt in optimizers:
        x = np.asarray(x0, dtype=float).copy()

        # time series
        f = np.zeros((steps + 1,), dtype=float)
        grad_norm = np.zeros((steps + 1,), dtype=float)
        grad_par_norm = np.zeros((steps + 1,), dtype=float)
        grad_perp_norm = np.zeros((steps + 1,), dtype=float)
        update_norm = np.zeros((steps + 1,), dtype=float)
        update_cos = np.zeros((steps + 1,), dtype=float)

        phi_raw = np.zeros((steps + 1,), dtype=float)
        s = np.zeros((steps + 1,), dtype=float)
        phi_eff = np.zeros((steps + 1,), dtype=float)
        rho = np.zeros((steps + 1,), dtype=float)
        e_norm = np.zeros((steps + 1,), dtype=float)

        f[0] = quad.value(x)
        g0 = quad.grad(x)
        grad_norm[0] = _safe_norm(g0)
        phi_raw[0] = float("nan")
        s[0] = float("nan")
        phi_eff[0] = float("nan")
        rho[0] = float("nan")
        e_norm[0] = float("nan")
        update_norm[0] = float("nan")
        update_cos[0] = float("nan")

        # NOTE: P may be time-varying; compute P and decompositions per step.
        for t in range(steps):
            g_clean = quad.grad(x)

            if batch_size <= 1:
                g_batch = g_clean.reshape(d, 1)
                g_used = g_clean
            else:
                if noise_sigma is None:
                    noise = rng.standard_normal((d, batch_size))
                    g_batch = g_clean.reshape(d, 1) + noise
                else:
                    sig = np.asarray(noise_sigma, dtype=float).reshape(d, 1)
                    noise = rng.standard_normal((d, batch_size)) * sig
                    g_batch = g_clean.reshape(d, 1) + noise
                g_used = g_batch.mean(axis=1)

            P_t = P_sched.step(t=t, x=x, g_batch=g_batch)
            x_next, info = opt.step(x=x, g=g_used, P=P_t, t=t)

            # Metrics for time t+1 computed at x_next (clean gradient), using the P_t that drove the update.
            g_clean_next = quad.grad(x_next)
            _, g_par_clean, g_perp_clean = _decompose(g_clean_next, P_t)
            u_vec = (np.asarray(x, dtype=float) - np.asarray(x_next, dtype=float)) / float(getattr(opt, "lr", 1.0))

            f[t + 1] = quad.value(x_next)
            grad_norm[t + 1] = _safe_norm(g_clean_next)
            grad_par_norm[t + 1] = _safe_norm(g_par_clean)
            grad_perp_norm[t + 1] = _safe_norm(g_perp_clean)
            update_norm[t + 1] = float(info.get("update_norm", _safe_norm(u_vec)))
            update_cos[t + 1] = _cosine(u_vec, g_clean)

            phi_raw[t + 1] = float(info.get("phi_raw", float("nan")))
            s[t + 1] = float(info.get("s", float("nan")))
            phi_eff[t + 1] = float(info.get("phi_eff", float("nan")))
            rho[t + 1] = float(info.get("rho", float("nan")))
            e_norm[t + 1] = float(info.get("e_norm", float("nan")))

            x = np.asarray(x_next, dtype=float)

        traces[opt.name] = {
            "f": f,
            "grad_norm": grad_norm,
            "grad_par_norm": grad_par_norm,
            "grad_perp_norm": grad_perp_norm,
            "update_norm": update_norm,
            "update_cos": update_cos,
            "phi_raw": phi_raw,
            "s": s,
            "phi_eff": phi_eff,
            "rho": rho,
            "e_norm": e_norm,
            "phi_cum": np.nancumsum(phi_raw),
            "s_cum": np.nancumsum(s),
        }

    # Save traces
    npz_payload: dict[str, np.ndarray] = {}
    for name, tr in traces.items():
        for k, v in tr.items():
            npz_payload[f"{name}__{k}"] = np.asarray(v, dtype=float)
    np.savez_compressed(RES_DIR / f"{exp_name}__seed{seed}__traces.npz", **npz_payload)

    # Per-key plots (quick debug)
    def _plot_key(key: str, ylabel: str, *, yscale: str | None = None):
        fig, ax = plt.subplots(figsize=(6.4, 4.1))
        for name, tr in traces.items():
            ax.plot(tr[key], linewidth=1.5, label=name)
        ax.set_title(f"{exp_name}: {key}")
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        if yscale is not None:
            ax.set_yscale(yscale)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{exp_name}__seed{seed}__{key}.png", dpi=180)
        plt.close(fig)

    _plot_key("grad_norm", "||∇f|| (clean)", yscale="log")
    _plot_key("grad_par_norm", "||P P^T ∇f|| (clean)", yscale="log")
    _plot_key("grad_perp_norm", "||(I-PP^T) ∇f|| (clean)", yscale="log")
    _plot_key("phi_raw", "phi_raw")
    _plot_key("s", "s_t (applied)")
    _plot_key("phi_cum", "∑ phi_raw", yscale="log")
    _plot_key("s_cum", "∑ s_t", yscale="log")
    _plot_key("e_norm", "||e_t||", yscale="log")
    _plot_key("f", "f(x_t)", yscale="log")

    # Composite figure (2x3)
    fig, axs = plt.subplots(2, 3, figsize=(15.2, 7.6))
    ax_gn, ax_gp, ax_go = axs[0, 0], axs[0, 1], axs[0, 2]
    ax_scale, ax_e, ax_obj = axs[1, 0], axs[1, 1], axs[1, 2]

    for name, tr in traces.items():
        ax_gn.plot(tr["grad_norm"], linewidth=1.6, label=name)
        ax_gp.plot(tr["grad_par_norm"], linewidth=1.6, label=name)
        ax_go.plot(tr["grad_perp_norm"], linewidth=1.6, label=name)

        s_used = np.asarray(tr["s"], dtype=float)
        s_pos = np.where(np.isfinite(s_used) & (s_used > 0), s_used, np.nan)
        if np.isfinite(s_pos).any():
            ax_scale.plot(s_pos, linewidth=1.4, label=name)

        e_series = np.asarray(tr["e_norm"], dtype=float)
        e_pos = np.where(np.isfinite(e_series) & (e_series > 0), e_series, np.nan)
        if np.isfinite(e_pos).any():
            ax_e.plot(e_pos, linewidth=1.4, label=name)

        f_series = np.maximum(np.asarray(tr["f"], dtype=float), 1e-300)
        ax_obj.plot(f_series, linewidth=1.5, label=name)

    ax_gn.set_title("(a) ||∇f|| (clean)")
    ax_gn.set_yscale("log")
    ax_gn.grid(True, alpha=0.3)
    ax_gn.legend(loc="best", fontsize=9)

    ax_gp.set_title("(b) ||P P^T ∇f|| (clean)")
    ax_gp.set_yscale("log")
    ax_gp.grid(True, alpha=0.3)

    ax_go.set_title("(c) ||(I-PP^T) ∇f|| (clean)")
    ax_go.set_yscale("log")
    ax_go.grid(True, alpha=0.3)

    ax_scale.set_title("(d) s_t (applied)")
    ax_scale.set_yscale("log")
    ax_scale.grid(True, alpha=0.3)

    ax_e.set_title("(e) ||e_t||")
    ax_e.set_yscale("log")
    ax_e.grid(True, alpha=0.3)

    ax_obj.set_title("(f) f(x_t)")
    ax_obj.set_yscale("log")
    ax_obj.grid(True, alpha=0.3)

    fig.suptitle(f"{exp_name} (seed={seed})")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / f"{exp_name}__seed{seed}__SYNTH_MAIN.png", dpi=240)
    fig.savefig(FIG_DIR / f"{exp_name}__seed{seed}__SYNTH_MAIN.pdf")
    plt.close(fig)

    summary = {
        "exp_name": exp_name,
        "d": int(d),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "optimizers": {},
    }
    for name, tr in traces.items():
        summary["optimizers"][name] = {
            "grad_norm_final": float(tr["grad_norm"][-1]),
            "grad_par_norm_final": float(tr["grad_par_norm"][-1]),
            "grad_perp_norm_final": float(tr["grad_perp_norm"][-1]),
            "residual_norm_final": float(tr["e_norm"][-1]) if np.isfinite(tr["e_norm"][-1]) else None,
            "sum_phi_raw": float(tr["phi_cum"][-1]) if np.isfinite(tr["phi_cum"][-1]) else None,
            "sum_s": float(tr["s_cum"][-1]) if np.isfinite(tr["s_cum"][-1]) else None,
            "f_final": float(tr["f"][-1]),
        }
    (RES_DIR / f"{exp_name}__seed{seed}__summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def exp1_diag_fixed_stale() -> dict:
    # Diagonal quadratic + fixed stale subspace: reproduces vanishing-scale failure beyond 2D.
    d = 500
    r = 20
    lam = np.ones((d,), dtype=float)
    quad = DiagQuad(lam=lam)

    # Fixed P = [e1,...,er].
    P0 = np.eye(d, r, dtype=float)
    P_sched = FixedP(P0)

    lr = 0.5
    eps_a = 1.0
    eps_s = 1.0
    steps = 3000
    x0 = np.ones((d,), dtype=float)

    opts: list[Optim] = [
        ProjSGDHD(lr=lr),
        FiraScalarHD(lr=lr, eps_a=eps_a, eps_s=eps_s, s_clip=None),
        FiraScalarHD(lr=lr, eps_a=eps_a, eps_s=eps_s, s_clip=(0.2, 0.5)),
        BECRFiraHD(lr=lr, eps_a=eps_a, eps_s=eps_s, rho=0.5, s_clip=(0.2, 0.5)),
        ResidualRhoNoLowerBoundCappedHD(lr=lr, eps_a=eps_a, eps_s=eps_s, s_clip=(0.2, 0.5), rho_max=1.0),
        ResidualCoupledUnboundedTransmissionHD(lr=lr, eps_a=eps_a, eps_s=eps_s, s_clip=(0.2, 0.5)),
    ]

    return run_hd_experiment(
        exp_name="exp1_diag_fixed_stale",
        quad=quad,
        P_sched=P_sched,
        optimizers=opts,
        x0=x0,
        steps=steps,
        batch_size=1,
        noise_sigma=None,
        seed=0,
    )


def exp2_stale_refresh_sweep() -> dict:
    # Stale-then-refresh: sweep refresh interval K to address "fixed subspace is unrealistic".
    d = 200
    r = 10
    # Mildly decaying spectrum so coordinate importances drift over time.
    lam = np.asarray([10 ** (-2.0 * i / (d - 1)) for i in range(d)], dtype=float)
    quad = DiagQuad(lam=lam)

    lr = 0.5
    eps_a = 1.0
    eps_s = 1.0
    steps = 2000
    x0 = np.ones((d,), dtype=float)

    K_list = [None, 200, 50, 10, 1]  # None = fixed forever
    sweep_rows = []

    for K in K_list:
        tag = "inf" if K is None else str(int(K))
        P_sched = TopGradStaleRefreshP(d=d, r=r, refresh_K=K)
        opts: list[Optim] = [
            ProjSGDHD(lr=lr),
            FiraScalarHD(lr=lr, eps_a=eps_a, eps_s=eps_s, s_clip=None),
            FiraScalarHD(lr=lr, eps_a=eps_a, eps_s=eps_s, s_clip=(0.2, 0.5)),
            BECRFiraHD(lr=lr, eps_a=eps_a, eps_s=eps_s, rho=0.5, s_clip=(0.2, 0.5)),
            ResidualRhoNoLowerBoundCappedHD(lr=lr, eps_a=eps_a, eps_s=eps_s, s_clip=(0.2, 0.5), rho_max=1.0),
            ResidualCoupledUnboundedTransmissionHD(lr=lr, eps_a=eps_a, eps_s=eps_s, s_clip=(0.2, 0.5)),
        ]
        summary = run_hd_experiment(
            exp_name=f"exp2_stale_refresh_K{tag}",
            quad=quad,
            P_sched=P_sched,
            optimizers=opts,
            x0=x0,
            steps=steps,
            batch_size=1,
            noise_sigma=None,
            seed=0,
        )
        row = {"K": ("inf" if K is None else int(K))}
        for m, s in summary["optimizers"].items():
            row[f"{m}__grad_norm_final"] = float(s["grad_norm_final"])
        sweep_rows.append(row)

    _write_json(RES_DIR / "exp2_stale_refresh__sweep.json", sweep_rows)

    # Plot: final grad norm vs K (log x for K)
    methods = ["proj_sgd", "fira_raw", "fira_clipped", "becr_fira", "rho_no_lower_bound", "coupled_unbounded"]
    K_numeric = [1e6 if r["K"] == "inf" else float(r["K"]) for r in sweep_rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for m in methods:
        y = [float(r.get(f"{m}__grad_norm_final", float("nan"))) for r in sweep_rows]
        ax.plot(K_numeric, y, marker="o", linewidth=1.6, label=m)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("refresh interval K (inf shown as 1e6)")
    ax.set_ylabel("final ||∇f|| (clean)")
    ax.set_title("Exp2: stale refresh sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp2_stale_refresh__final_grad_vs_K.png", dpi=220)
    fig.savefig(FIG_DIR / "exp2_stale_refresh__final_grad_vs_K.pdf")
    plt.close(fig)

    return {"exp_name": "exp2_stale_refresh_sweep", "rows": sweep_rows}


def exp3_anisotropic_noise_svd_projection() -> dict:
    # Anisotropic noise + SVD-biased projection (diagnostic stress).
    d = 120
    r = 5
    # Block spectrum: a few stiff directions + many mild.
    lam = np.ones((d,), dtype=float)
    lam[:10] = 5.0
    quad = DiagQuad(lam=lam)

    lr = 0.05
    eps_a = 1e-3
    eps_s = 1e-3
    steps = 2500
    x0 = np.ones((d,), dtype=float)

    # Noise: high variance on a subset that is NOT aligned with initial top-curvature block.
    noise_sigma = np.full((d,), 0.02, dtype=float)
    noise_sigma[40:60] = 0.2

    batch_size = 20
    update_K = 20
    seeds = list(range(5))

    all_summaries = []
    for seed in seeds:
        P_sched = SVDBiasedP(r=r, update_K=update_K)
        opts: list[Optim] = [
            ProjSGDHD(lr=lr),
            FiraScalarHD(lr=lr, eps_a=eps_a, eps_s=eps_s, s_clip=None),
            FiraScalarHD(lr=lr, eps_a=eps_a, eps_s=eps_s, s_clip=(0.1, 10.0)),
            BECRFiraHD(lr=lr, eps_a=eps_a, eps_s=eps_s, rho=0.5, s_clip=(0.1, 10.0)),
            ResidualRhoNoLowerBoundCappedHD(lr=lr, eps_a=eps_a, eps_s=eps_s, s_clip=(0.1, 10.0), rho_max=1.0),
            ResidualCoupledUnboundedTransmissionHD(lr=lr, eps_a=eps_a, eps_s=eps_s, s_clip=(0.1, 10.0)),
        ]
        all_summaries.append(
            run_hd_experiment(
                exp_name="exp3_aniso_noise_svdP",
                quad=quad,
                P_sched=P_sched,
                optimizers=opts,
                x0=x0,
                steps=steps,
                batch_size=batch_size,
                noise_sigma=noise_sigma,
                seed=seed,
            )
        )

    # Aggregate stats (final clean grad norm).
    methods = ["proj_sgd", "fira_raw", "fira_clipped", "becr_fira", "rho_no_lower_bound", "coupled_unbounded"]
    agg = {"exp_name": "exp3_anisotropic_noise_svd_projection", "seeds": seeds, "methods": {}}
    for m in methods:
        vals = []
        for s in all_summaries:
            vals.append(float(s["optimizers"][m]["grad_norm_final"]))
        vals = np.asarray(vals, dtype=float)
        agg["methods"][m] = {
            "grad_norm_final_mean": float(vals.mean()),
            "grad_norm_final_std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
        }
    _write_json(RES_DIR / "exp3_aniso_noise_svdP__aggregate.json", agg)

    # Bar plot with error bars
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    xs = np.arange(len(methods))
    means = [float(agg["methods"][m]["grad_norm_final_mean"]) for m in methods]
    stds = [float(agg["methods"][m]["grad_norm_final_std"]) for m in methods]
    ax.bar(xs, means, yerr=stds, capsize=4, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(xs, methods, rotation=25, ha="right")
    ax.set_ylabel("final ||∇f|| (clean), mean±std over seeds")
    ax.set_title("Exp3: anisotropic-noise + SVD-biased projection")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp3_aniso_noise_svdP__final_grad_bar.png", dpi=220)
    fig.savefig(FIG_DIR / "exp3_aniso_noise_svdP__final_grad_bar.pdf")
    plt.close(fig)

    return agg


def main() -> None:
    out = []
    out.append(exp1_diag_fixed_stale())
    out.append(exp2_stale_refresh_sweep())
    out.append(exp3_anisotropic_noise_svd_projection())
    _write_json(RES_DIR / "TIER1_ALL_SUMMARIES.json", out)
    print("Done. BASE_DIR =", str(BASE_DIR))
    print("Figures:", str(FIG_DIR))
    print("Results:", str(RES_DIR))


if __name__ == "__main__":
    main()
