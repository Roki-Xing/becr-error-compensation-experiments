from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import MovingProjectionConfig, make_basis, run_method_suite


def _serialize(obj):
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if not np.isfinite(val):
            return None
        return val
    if isinstance(obj, np.ndarray):
        return _serialize(obj.tolist())
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_serialize(obj), indent=2, allow_nan=False), encoding="utf-8")


def _build_demo_suite(parent_task_id: str, parent_pr: str | None) -> dict[str, dict]:
    bases = [
        make_basis(2, [0], dtype=np.float64),
        make_basis(2, [0], dtype=np.float64),
        make_basis(2, [1], dtype=np.float64),
        make_basis(2, [1], dtype=np.float64),
        make_basis(2, [0], dtype=np.float64),
        make_basis(2, [0], dtype=np.float64),
    ]
    gradients = [
        np.asarray([2.0, 1.0], dtype=np.float64),
        np.asarray([1.5, 1.1], dtype=np.float64),
        np.asarray([1.0, -2.0], dtype=np.float64),
        np.asarray([0.8, -1.5], dtype=np.float64),
        np.asarray([-0.6, 1.2], dtype=np.float64),
        np.asarray([-0.4, 0.8], dtype=np.float64),
    ]
    modes = [
        "state_reset_explicit",
        "official_fira_carry",
        "projection_aware_transport",
        "full_residual_current_projection",
    ]
    cfg = MovingProjectionConfig(dtype=np.float64, rho=0.5, limiter_gamma=1.25)
    return run_method_suite(
        modes=modes,
        cfg=cfg,
        gradients=gradients,
        bases=bases,
        stochastic_noise_std=0.15,
        rng_seed=11,
        parent_task_id=parent_task_id,
        parent_pr=parent_pr,
    )


def _plot_refresh_alignment(path_base: Path, suite: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.2), sharex=True)
    ax_q, ax_e = axes
    for mode, result in suite.items():
        q_norm = [float(np.linalg.norm(np.asarray(step["Q"], dtype=float))) for step in result["steps"]]
        e_norm = [float(step["state_after_step"]["e_norm"]) for step in result["steps"]]
        xs = list(range(len(q_norm)))
        ax_q.plot(xs, q_norm, marker="o", linewidth=1.5, label=mode)
        ax_e.plot(xs, e_norm, marker="o", linewidth=1.5, label=mode)
        for step in result["steps"]:
            if step["refresh_happened"]:
                ax_q.axvline(int(step["step_index"]), color="k", alpha=0.08)
                ax_e.axvline(int(step["step_index"]), color="k", alpha=0.08)
    ax_q.set_title("Refresh-aligned orthogonal remainder norm")
    ax_q.set_ylabel("||Q_t||")
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(loc="best", fontsize=8)
    ax_e.set_title("Refresh-aligned residual norm after update")
    ax_e.set_xlabel("step")
    ax_e.set_ylabel("||E_{t+1}||")
    ax_e.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def generate_artifacts(
    *,
    output_root: Path,
    parent_task_id: str = "P0-MOVING-PROJECTION-STATE-002",
    parent_pr: str | None = None,
    run_id: str | None = None,
) -> Path:
    output_root = Path(output_root)
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_moving_projection_state")
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    suite = _build_demo_suite(parent_task_id=parent_task_id, parent_pr=parent_pr)
    sample = suite["full_residual_current_projection"]

    _write_json(out_dir / "manifest.json", sample["manifest"])
    _write_json(out_dir / "sample_trace.json", sample)
    _write_json(out_dir / "suite_summary.json", suite)
    _plot_refresh_alignment(out_dir / "refresh_alignment", suite)

    observed_lines = []
    full_resid = suite["full_residual_current_projection"]
    official = suite["official_fira_carry"]
    transported = suite["projection_aware_transport"]
    reset_mode = suite["state_reset_explicit"]
    observed_lines.append(
        f"- `full_residual_current_projection` keeps `reset_events={full_resid['manifest']['reset_events']}` and reaches "
        f"`max_decomposition_error={full_resid['max_decomposition_error']:.3e}`, "
        f"`max_residual_error={full_resid['max_residual_error']:.3e}`."
    )
    observed_lines.append(
        f"- `state_reset_explicit` records reset events at steps {reset_mode['manifest']['reset_events']}, making refresh-induced resets explicit instead of silent."
    )
    observed_lines.append(
        f"- `official_fira_carry` and `projection_aware_transport` differ in final update norm "
        f"({np.linalg.norm(np.asarray(official['steps'][-1]['u'], dtype=float)):.3e} vs "
        f"{np.linalg.norm(np.asarray(transported['steps'][-1]['u'], dtype=float)):.3e}), isolating basis-coordinate mismatch from residual compensation."
    )

    summary_lines = [
        "# Moving Projection State Diagnostic",
        "",
        "- `state_reset_explicit` resets state by design and therefore removes carried residual/history at refresh.",
        "- `official_fira_carry` carries low-rank state without basis transport and uses no residual compensation.",
        "- `projection_aware_transport` keeps low-rank history but transports it with the new basis overlap.",
        "- `full_residual_current_projection` applies compensated decomposition `A_t = G_t + E_t` and updates residual with `Q_t - Z_eff_t`.",
        "",
        "## Observed Result",
        "",
        *observed_lines,
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return out_dir
