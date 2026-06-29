from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiment_provenance.aggregate import write_explicit_aggregate
from experiment_provenance.ids import build_run_id, config_hash, utc_timestamp
from experiment_provenance.json_utils import sanitize_for_json, write_json_strict, write_jsonl_strict
from experiment_provenance.metadata import get_git_metadata
from experiment_provenance.schema import (
    MANIFEST_SCHEMA_VERSION,
    MEMORY_SCHEMA_VERSION,
    SCALE_LOGGING_SCHEMA_VERSION,
    build_memory_runtime_schema,
    validate_manifest,
)
from fira_parity.runner import run_fixture
from moving_projection_state.core import MovingProjectionConfig

from .core import (
    CoordinateConfig,
    DiagQuadratic,
    FixedBasisSchedule,
    SVDBatchRefreshSchedule,
    TopGradientRefreshSchedule,
    confidence_interval,
    generate_crn_noise_bank,
    run_coordinate_trace,
    run_moving_projection_dynamic_trace,
    theorem_condition_report,
    wall_clock_ms,
)
from .plots import (
    plot_anisotropic_noise,
    plot_high_dimensional,
    plot_refresh_sweep,
    plot_theorem_regime,
)


TASK_ID = "P1-CORRECTED-SYNTHETIC-RERUNS"
DEFAULT_COORD_METHODS = [
    "proj_baseline",
    "fira_raw",
    "fira_clipped",
    "becr",
    "rho_no_lower_bound",
    "coupled_unbounded",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
STATE_MODES = [
    "state_reset_explicit",
    "official_fira_carry",
    "projection_aware_transport",
    "full_residual_current_projection",
]


def _memory_schema_for_coordinate(*, trace: dict[str, Any], d: int, rank: int, cfg: CoordinateConfig, elapsed_ms: float) -> dict[str, Any]:
    itemsize = np.dtype(cfg.dtype).itemsize
    residual_bytes = d * itemsize if trace["method"] in {"becr", "rho_no_lower_bound", "coupled_unbounded", "wrong_units_naive_ef"} else 0
    return build_memory_runtime_schema(
        device="cpu",
        parameter_bytes=d * itemsize,
        gradient_buffer_bytes=d * itemsize,
        first_moment_bytes=rank * itemsize,
        second_moment_bytes=0,
        projection_bytes=d * rank * itemsize,
        residual_bytes=residual_bytes,
        temporary_buffer_bytes_estimate=4 * d * itemsize,
        peak_memory_allocated=None,
        peak_memory_reserved=None,
        optimizer_step_time_ms=elapsed_ms / max(len(trace["steps"]), 1),
        wall_clock_time_ms=elapsed_ms,
    )


def _memory_schema_for_state(*, d: int, rank: int, cfg: MovingProjectionConfig, elapsed_ms: float) -> dict[str, Any]:
    itemsize = np.dtype(cfg.dtype).itemsize
    return build_memory_runtime_schema(
        device="cpu",
        parameter_bytes=d * itemsize,
        gradient_buffer_bytes=d * itemsize,
        first_moment_bytes=rank * itemsize,
        second_moment_bytes=rank * itemsize,
        projection_bytes=d * rank * itemsize,
        residual_bytes=d * itemsize,
        temporary_buffer_bytes_estimate=6 * d * itemsize,
        peak_memory_allocated=None,
        peak_memory_reserved=None,
        optimizer_step_time_ms=elapsed_ms / max(int(elapsed_ms > 0), 1),
        wall_clock_time_ms=elapsed_ms,
    )


def _scale_rows_from_coordinate(trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in trace["steps"]:
        rows.append(
            {
                "step_index": row["step_index"],
                "phi_raw": row["phi_raw"],
                "s_raw": row["s_raw"],
                "s_applied": row["s_applied"],
                "rho": row["rho"],
                "tau": row["tau"],
                "Z_norm": row["Z_norm"],
                "Z_eff_norm": row["Z_eff_norm"],
                "effective_raw_transmission": row["effective_raw_transmission"],
                "residual_update_error": row["residual_update_error"],
                "limiter_active": row["limiter_active"],
                "raw_recovery_scale": row["phi_raw"],
                "applied_recovery_scale": row["s_applied"],
                "effective_recovery_scale": row["s_applied"] * row["tau"],
            }
        )
    return rows


def _scale_rows_from_state(trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in trace["steps"]:
        rho = row["rho"]
        rows.append(
            {
                "step_index": row["step_index"],
                "phi_raw": row["phi_raw"],
                "s_raw": row["s_raw"],
                "s_applied": row["s_applied"],
                "rho": rho,
                "tau": row["tau"],
                "Z_norm": None,
                "Z_eff_norm": None,
                "effective_raw_transmission": None,
                "residual_update_error": row["residual_update_error"],
                "limiter_active": bool(abs(float(row["tau"]) - 1.0) > 1e-12),
                "raw_recovery_scale": row["phi_raw"],
                "applied_recovery_scale": row["s_applied"],
                "effective_recovery_scale": row["s_applied"] * row["tau"],
            }
        )
    return rows


def _metric_summary_from_series(method: str, family: str, trace: dict[str, Any], noise_hash: str, seed: int) -> dict[str, Any]:
    series = trace["series"]
    phi = np.asarray(series["phi_raw"], dtype=float)
    phi_finite = phi[np.isfinite(phi)]
    return {
        "method": method,
        "family": family,
        "seed": int(seed),
        "noise_hash": noise_hash,
        "consumed_noise_hash": trace["consumed_noise_hash"],
        "grad_norm_final": float(series["grad_norm"][-1]),
        "grad_par_norm_final": float(series["grad_par_norm"][-1]),
        "grad_perp_norm_final": float(series["grad_perp_norm"][-1]),
        "f_final": float(series["f"][-1]),
        "phi_raw_cum_final": float(series["phi_cum"][-1]),
        "s_applied_cum_final": float(series["s_applied_cum"][-1]),
        "residual_norm_final": float(series["residual_norm"][-1]) if math.isfinite(float(series["residual_norm"][-1])) else None,
        "residual_update_error_max": float(np.nanmax(np.asarray(series["residual_update_error"], dtype=float))),
        "phi_raw_p95": float(np.nanpercentile(phi_finite, 95)) if phi_finite.size else None,
        "phi_raw_p99": float(np.nanpercentile(phi_finite, 99)) if phi_finite.size else None,
        "grad_norm_auc": float(np.trapezoid(np.asarray(series["grad_norm"], dtype=float))),
        "limiter_count": int(trace.get("limiter_count", 0)),
    }


def _state_metric_summary(mode: str, trace: dict[str, Any], noise_hash: str, seed: int) -> dict[str, Any]:
    series = trace["series"]
    phi = np.asarray([row["phi_raw"] for row in trace["steps"]], dtype=float)
    phi_finite = phi[np.isfinite(phi)]
    return {
        "method": mode,
        "family": "moving_projection_state_mode",
        "seed": int(seed),
        "noise_hash": noise_hash,
        "consumed_noise_hash": trace["consumed_noise_hash"],
        "grad_norm_final": float(series["grad_norm"][-1]),
        "grad_par_norm_final": float(series["grad_par_norm"][-1]),
        "grad_perp_norm_final": float(series["grad_perp_norm"][-1]),
        "f_final": float(series["f"][-1]),
        "residual_norm_final": float(series["residual_norm"][-1]),
        "residual_update_error_max": float(np.nanmax(np.asarray(series["residual_update_error"], dtype=float))),
        "basis_overlap_mean": float(np.nanmean(np.asarray([v if v is not None else math.nan for v in series["basis_overlap"]], dtype=float))),
        "phi_raw_p95": float(np.nanpercentile(phi_finite, 95)) if phi_finite.size else None,
        "phi_raw_p99": float(np.nanpercentile(phi_finite, 99)) if phi_finite.size else None,
        "reset_events": trace["reset_events"],
    }


def _write_run_artifacts(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    memory: dict[str, Any],
    scale_rows: list[dict[str, Any]],
    trace: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    memory_path = run_dir / "memory_runtime.json"
    scale_log_path = run_dir / "scale_log.jsonl"
    trace_path = run_dir / "trace.json"
    validate_manifest(manifest)
    write_json_strict(manifest_path, manifest, sort_keys=True)
    write_json_strict(metrics_path, metrics, sort_keys=True)
    write_json_strict(memory_path, memory, sort_keys=True)
    write_jsonl_strict(scale_log_path, scale_rows)
    write_json_strict(trace_path, sanitize_for_json(trace), sort_keys=True)
    return manifest_path, metrics_path, memory_path, scale_log_path


def _base_manifest(
    *,
    metadata: dict[str, Any],
    config: dict[str, Any],
    cfg_hash: str,
    method: str,
    seed: int,
    noise_hash: str | None,
    scheduler_factory_id: str,
    scheduler_object_id: str,
    started_at: str,
    completed_at: str,
    optimizer_semantics: str,
    projection_mode: str,
    state_transport_mode: str,
    residual_mode: str,
    second_moment_mode: str,
    command: str,
    parent_pr: str | None,
    outputs: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": outputs["run_id"],
        "parent_task_id": TASK_ID,
        "parent_pr": parent_pr,
        "code_commit": metadata["code_commit"],
        "dirty": bool(metadata["dirty"]),
        "branch": metadata["branch"],
        "command": command,
        "config": config,
        "config_hash": cfg_hash,
        "method": method,
        "mode": method,
        "seed": int(seed),
        "rng_seed": int(seed),
        "noise_hash": noise_hash,
        "dataset_hash": None,
        "scheduler_factory_id": scheduler_factory_id,
        "scheduler_object_id": scheduler_object_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "old_results_used": False,
        "legacy_results_used": False,
        "optimizer_semantics": optimizer_semantics,
        "projection_mode": projection_mode,
        "state_transport_mode": state_transport_mode,
        "residual_mode": residual_mode,
        "second_moment_mode": second_moment_mode,
        "weight_decay_semantics": "decoupled_none",
        "scale_logging_schema": SCALE_LOGGING_SCHEMA_VERSION,
        "memory_schema_version": MEMORY_SCHEMA_VERSION,
        "outputs": outputs,
    }


def _experiment_dir(group_dir: Path, experiment_name: str) -> Path:
    out = group_dir / experiment_name
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "aggregates").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    return out


def _aggregate_metrics(metric_rows: list[dict[str, Any]], methods: list[str], label_map: dict[str, str]) -> dict[str, Any]:
    summary = {"methods": {}}
    for method in methods:
        rows = [row for row in metric_rows if row["method"] == method]
        if not rows:
            continue
        grad = [row["grad_norm_final"] for row in rows]
        gpar = [row["grad_par_norm_final"] for row in rows]
        gperp = [row["grad_perp_norm_final"] for row in rows]
        residual = [row["residual_norm_final"] for row in rows if row["residual_norm_final"] is not None]
        summary["methods"][method] = {
            "label": label_map.get(method, method),
            "count": len(rows),
            "grad_norm_mean": float(np.mean(grad)),
            "grad_norm_ci_low": confidence_interval(grad)[1],
            "grad_norm_ci_high": confidence_interval(grad)[2],
            "grad_par_norm_mean": float(np.mean(gpar)),
            "grad_perp_norm_mean": float(np.mean(gperp)),
            "residual_norm_mean": (float(np.mean(residual)) if residual else None),
            "phi_raw_p95_values": [row["phi_raw_p95"] for row in rows if row["phi_raw_p95"] is not None],
        }
    return summary


def _command_for_run(*, output_root: Path, run_id: str, experiment_name: str | None = None, tiny: bool, paper_quality: bool, parent_pr: str | None) -> str:
    parts = [
        "python",
        "experiments/tier1-synthetic/run_corrected_synthetic_suite.py",
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
    ]
    if paper_quality:
        parts.append("--paper-quality")
    if tiny:
        parts.append("--tiny")
    if parent_pr is not None:
        parts.extend(["--parent-pr", parent_pr])
    if experiment_name is not None:
        parts.extend(["--experiments", experiment_name])
    return " ".join(parts)


def _claim_summary_for_theorem(summary: dict[str, Any]) -> dict[str, Any]:
    raw = summary["methods"]["fira_raw"]
    clipped = summary["methods"]["fira_clipped"]
    becr = summary["methods"]["becr"]
    no_lower = summary["methods"]["rho_no_lower_bound"]
    coupled = summary["methods"]["coupled_unbounded"]
    return {
        "raw_nonzero_stationary": bool(raw["grad_norm_final"] > 1e-2 and abs(raw["final_y"]) > 1e-2),
        "raw_cumulative_phi_finite": bool(raw["phi_raw_cum_final"] < 10.0),
        "clipped_repairs_relative_to_raw": bool(clipped["grad_norm_final"] < raw["grad_norm_final"] * 1e-2),
        "becr_repairs_relative_to_raw": bool(becr["grad_norm_final"] < raw["grad_norm_final"] * 1e-2),
        "becr_residual_small": bool((becr["residual_norm_final"] or 0.0) < 1e-2),
        "no_lower_worse_than_becr": bool(no_lower["grad_norm_final"] > becr["grad_norm_final"] * 100.0),
        "coupled_worse_than_becr": bool(coupled["grad_norm_final"] > becr["grad_norm_final"] * 100.0),
        "statement": "Raw Fira-style recovery leaves a nonzero orthogonal gradient; clipped recovery and BECR repair it, while residual without bounded transmission remains worse than BECR.",
    }


def _claim_summary_for_high_dimensional(summary: dict[str, Any]) -> dict[str, Any]:
    raw = summary["methods"]["fira_raw"]
    clipped = summary["methods"]["fira_clipped"]
    becr = summary["methods"]["becr"]
    return {
        "raw_orthogonal_gradient_mean": raw["grad_perp_norm_mean"],
        "clipped_orthogonal_gradient_mean": clipped["grad_perp_norm_mean"],
        "becr_orthogonal_gradient_mean": becr["grad_perp_norm_mean"],
        "clipped_reduces_orthogonal_stale_channel": bool(clipped["grad_perp_norm_mean"] < raw["grad_perp_norm_mean"] * 1e-3),
        "becr_reduces_orthogonal_stale_channel": bool(becr["grad_perp_norm_mean"] < raw["grad_perp_norm_mean"] * 1e-3),
        "full_stationarity_claim_supported": False,
        "statement": "In the corrected fixed-stale synthetic, clipping and BECR fix the orthogonal stale channel relative to raw recovery. The projected branch can still leave nonzero parallel gradient, so this summary does not claim universal full stationarity.",
    }


def _claim_summary_for_anisotropic(summary: dict[str, Any]) -> dict[str, Any]:
    methods = summary["methods"]
    becr = methods["becr"]["grad_norm_mean"]
    clipped = methods["fira_clipped"]["grad_norm_mean"]
    raw = methods["fira_raw"]["grad_norm_mean"]
    proj = methods["proj_baseline"]["grad_norm_mean"]
    becr_beats_clipped = bool(becr < clipped)
    becr_beats_raw = bool(becr < raw)
    becr_beats_proj = bool(becr < proj)
    if becr_beats_clipped and becr_beats_raw:
        if becr_beats_proj:
            statement = (
                "In this anisotropic synthetic diagnostic, BECR improves over clipped/raw on mean final gradient norm "
                "and also beats the projected baseline in this setting. This is a narrow synthetic diagnostic, not a broad optimizer superiority claim."
            )
        else:
            statement = (
                "In this anisotropic synthetic diagnostic, BECR improves over clipped/raw on mean final gradient norm "
                "under paired noisy projection, but does not beat the projected baseline in this setting."
            )
    elif becr_beats_clipped or becr_beats_raw:
        pairwise = "clipped" if becr_beats_clipped else "raw"
        missing = "raw" if becr_beats_clipped else "clipped"
        if becr_beats_proj:
            statement = (
                f"In this anisotropic synthetic diagnostic, BECR beats {pairwise} and the projected baseline on mean final gradient norm, "
                f"but it does not improve over {missing}. This remains a narrow setting-specific result."
            )
        else:
            statement = (
                f"In this anisotropic synthetic diagnostic, BECR beats {pairwise} on mean final gradient norm, "
                f"but it does not improve over {missing} or the projected baseline in this setting."
            )
    elif becr_beats_proj:
        statement = (
            "In this anisotropic synthetic diagnostic, BECR beats the projected baseline on mean final gradient norm in this setting, "
            "but it does not improve over clipped/raw."
        )
    else:
        statement = (
            "In this anisotropic synthetic diagnostic, no supported improvement is recorded for BECR over clipped/raw or the projected baseline "
            "on mean final gradient norm."
        )
    return {
        "becr_beats_clipped_on_grad_norm_mean": becr_beats_clipped,
        "becr_beats_raw_on_grad_norm_mean": becr_beats_raw,
        "becr_beats_projected_baseline_on_grad_norm_mean": becr_beats_proj,
        "statement": statement,
    }


def _write_memory_runtime_summary(*, group_dir: Path, manifest_paths: list[Path], memory_paths: list[Path]) -> Path:
    rows = []
    for manifest_path, memory_path in zip(manifest_paths, memory_paths, strict=False):
        manifest = _read_json(manifest_path)
        memory = _read_json(memory_path)
        rows.append(
            {
                "run_id": manifest["run_id"],
                "experiment_name": manifest["config"]["experiment_name"],
                "method": manifest["method"],
                "memory_path": str(memory_path),
                "parameter_bytes": memory["parameter_bytes"],
                "first_moment_bytes": memory["first_moment_bytes"],
                "second_moment_bytes": memory["second_moment_bytes"],
                "projection_bytes": memory["projection_bytes"],
                "residual_bytes": memory["residual_bytes"],
                "optimizer_step_time_ms": memory["optimizer_step_time_ms"],
                "wall_clock_time_ms": memory["wall_clock_time_ms"],
            }
        )
    by_experiment: dict[str, dict[str, Any]] = {}
    for row in rows:
        exp = row["experiment_name"]
        bucket = by_experiment.setdefault(exp, {"count": 0, "representative_files": [], "residual_bytes": [], "projection_bytes": []})
        bucket["count"] += 1
        if len(bucket["representative_files"]) < 4:
            bucket["representative_files"].append(row["memory_path"])
        if row["residual_bytes"] is not None:
            bucket["residual_bytes"].append(row["residual_bytes"])
        if row["projection_bytes"] is not None:
            bucket["projection_bytes"].append(row["projection_bytes"])
    out = group_dir / "memory_runtime_summary.json"
    write_json_strict(
        out,
        {
            "run_id": group_dir.name,
            "row_count": len(rows),
            "experiments": by_experiment,
            "representative_rows": rows[:12],
        },
        sort_keys=True,
    )
    return out


def _write_scale_log_summary(*, group_dir: Path, manifest_paths: list[Path], scale_log_paths: list[Path]) -> Path:
    rows = []
    for manifest_path, scale_log_path in zip(manifest_paths, scale_log_paths, strict=False):
        manifest = _read_json(manifest_path)
        rows.append(
            {
                "run_id": manifest["run_id"],
                "experiment_name": manifest["config"]["experiment_name"],
                "method": manifest["method"],
                "scale_log_path": str(scale_log_path),
                "scale_logging_schema": manifest["scale_logging_schema"],
            }
        )
    by_experiment: dict[str, dict[str, Any]] = {}
    for row in rows:
        exp = row["experiment_name"]
        bucket = by_experiment.setdefault(exp, {"count": 0, "representative_scale_logs": [], "methods": []})
        bucket["count"] += 1
        if len(bucket["representative_scale_logs"]) < 6:
            bucket["representative_scale_logs"].append(row["scale_log_path"])
        if row["method"] not in bucket["methods"]:
            bucket["methods"].append(row["method"])
    out = group_dir / "scale_log_summary.json"
    write_json_strict(
        out,
        {
            "run_id": group_dir.name,
            "row_count": len(rows),
            "experiments": by_experiment,
            "representative_rows": rows[:12],
        },
        sort_keys=True,
    )
    return out


def _write_artifact_index(
    *,
    group_dir: Path,
    output_root: Path,
    metadata: dict[str, Any],
    run_id: str,
    paper_quality: bool,
    tiny: bool,
    parent_pr: str | None,
    command: str,
    results: dict[str, Any],
    memory_runtime_summary_path: Path,
    scale_log_summary_path: Path,
) -> Path:
    artifact_index = {
        "run_id": run_id,
        "code_commit": metadata["code_commit"],
        "dirty": bool(metadata["dirty"]),
        "branch": metadata["branch"],
        "command": command,
        "paper_quality": paper_quality,
        "tiny": tiny,
        "parent_pr": parent_pr,
        "output_root": str(output_root),
        "group_dir": str(group_dir),
        "aggregate_manifest_paths": [str(path) for path in results["aggregate_manifest_paths"]],
        "aggregate_summary_paths": [str(path) for path in results["aggregate_summary_paths"]],
        "experiment_summary_paths": [str(path) for path in results["experiment_summary_paths"]],
        "figure_paths": [str(path) for path in results["figure_paths"]],
        "figure_metadata_paths": [str(path) for path in results["plot_metadata_paths"]],
        "table_paths": [str(path) for path in results["table_paths"]],
        "memory_paths": [str(path) for path in results["memory_paths"]],
        "scale_log_paths": [str(path) for path in results["scale_log_paths"]],
        "memory_runtime_summary_path": str(memory_runtime_summary_path),
        "scale_log_summary_path": str(scale_log_summary_path),
        "run_manifest_index_path": str(group_dir / "run_manifest_index.json"),
        "exact_fira_fixture_summary_path": str(results["exact_fira_summary_path"]),
        "ldadam_status_path": str(results["ldadam_status_path"]),
        "old_results_used": False,
        "legacy_results_used": False,
    }
    out = group_dir / "artifact_index.json"
    write_json_strict(out, artifact_index, sort_keys=True)
    return out


def _write_review_snapshot(*, group_dir: Path, artifact_index_path: Path, review_snapshot_dir: Path) -> Path:
    artifact_index = _read_json(artifact_index_path)
    if review_snapshot_dir.exists():
        shutil.rmtree(review_snapshot_dir)
    review_snapshot_dir.mkdir(parents=True, exist_ok=False)
    keep = [
        Path(artifact_index_path),
        Path(artifact_index["run_manifest_index_path"]),
        Path(artifact_index["exact_fira_fixture_summary_path"]),
        Path(artifact_index["ldadam_status_path"]),
        Path(artifact_index["memory_runtime_summary_path"]),
        Path(artifact_index["scale_log_summary_path"]),
    ]
    for key in (
        "aggregate_manifest_paths",
        "aggregate_summary_paths",
        "experiment_summary_paths",
        "figure_paths",
        "figure_metadata_paths",
        "table_paths",
    ):
        keep.extend(Path(path) for path in artifact_index[key])
    for source in keep:
        rel = source.relative_to(group_dir)
        dest = review_snapshot_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    readme = review_snapshot_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Corrected Synthetic Review Snapshot",
                "",
                "- This directory is a reviewer-convenience snapshot, not the sole source of truth.",
                f"- Semantic source commit: `{artifact_index['code_commit']}`.",
                "- Source of truth remains reproducible generation from current code plus strict JSON validation and CI regeneration.",
                "- Regenerate the full local paper-quality suite with:",
                f"  `{artifact_index['command']}`",
                "- CI generates a tiny paper-quality corrected synthetic artifact separately for reproducibility checks.",
                "- No paper claim should rely on stale or unchecked artifacts alone.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return review_snapshot_dir


def _config_dict(experiment_name: str, extra: dict[str, Any]) -> dict[str, Any]:
    out = {"experiment_name": experiment_name}
    out.update(extra)
    return out


def _run_theorem_regime(
    *,
    group_dir: Path,
    output_root: Path,
    metadata: dict[str, Any],
    tiny: bool,
    paper_quality: bool,
    parent_pr: str | None,
) -> dict[str, Any]:
    experiment_name = "theorem_regime"
    exp_dir = _experiment_dir(group_dir, experiment_name)
    a = 1.0
    b = 1.0
    lr = 0.5
    eps_a = 1.0
    eps_s = 1.0
    s_min = 0.2
    s_max = 0.5
    rho = 0.5
    steps = 80 if tiny else 600
    x0 = np.asarray([1.0, 1.0], dtype=np.float64)
    quad = DiagQuadratic(np.asarray([a, b], dtype=float))
    basis = np.asarray([[1.0], [0.0]], dtype=float)
    methods = list(DEFAULT_COORD_METHODS)
    theorem = theorem_condition_report(a=a, b=b, lr=lr, eps_a=eps_a, eps_s=eps_s, x0=float(x0[0]))
    cfg = CoordinateConfig(lr=lr, eps_a=eps_a, eps_s=eps_s, s_min=s_min, s_max=s_max, rho=rho, rho_min=1e-6, rho_max=1.0, limiter_gamma=None, dtype=np.float64)
    config = _config_dict(
        experiment_name,
        {
            "a": a,
            "b": b,
            "lr": lr,
            "eps_a": eps_a,
            "eps_s": eps_s,
            "s_min": s_min,
            "s_max": s_max,
            "rho": rho,
            "steps": steps,
            "methods": methods,
            "x0": x0.tolist(),
        },
    )
    cfg_hash = config_hash(config)
    run_command = _command_for_run(output_root=output_root, run_id=group_dir.name, experiment_name=experiment_name, tiny=tiny, paper_quality=paper_quality, parent_pr=parent_pr)
    method_payloads: dict[str, dict[str, Any]] = {}
    metric_rows = []
    manifest_paths = []
    memory_paths = []
    scale_log_paths = []
    for method in methods:
        started = utc_timestamp()
        schedule = FixedBasisSchedule(basis.copy(), projection_mode="fixed_e1")
        start_perf = time.perf_counter()
        trace = run_coordinate_trace(method=method, quad=quad, x0=x0, steps=steps, schedule=schedule, cfg=cfg)
        elapsed_ms = wall_clock_ms(start_perf)
        run_id = build_run_id(
            task=TASK_ID,
            experiment=experiment_name,
            method=method,
            seed=0,
            short_commit=str(metadata["short_commit"]),
            config_hash_value=cfg_hash,
            timestamp=started,
        )
        run_dir = exp_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        metrics = _metric_summary_from_series(method, trace["method_family"], trace, trace["noise_hash"], 0)
        metrics["final_x"] = float(trace["series"]["x_component"][-1])
        metrics["final_y"] = float(trace["series"]["y_component"][-1])
        memory = _memory_schema_for_coordinate(trace=trace, d=2, rank=1, cfg=cfg, elapsed_ms=elapsed_ms)
        outputs = {
            "run_id": run_id,
            "manifest": str(run_dir / "manifest.json"),
            "metrics": str(run_dir / "metrics.json"),
            "memory_runtime": str(run_dir / "memory_runtime.json"),
            "scale_log": str(run_dir / "scale_log.jsonl"),
            "trace": str(run_dir / "trace.json"),
        }
        if method == "proj_baseline":
            residual_mode = "none"
        elif method in {"fira_raw", "fira_clipped"}:
            residual_mode = "none"
        elif method == "becr":
            residual_mode = "current_projection_compensated"
        elif method == "rho_no_lower_bound":
            residual_mode = "rho_no_lower_bound"
        else:
            residual_mode = "coupled_rho_phi_raw"
        manifest = _base_manifest(
            metadata=metadata,
            config=config,
            cfg_hash=cfg_hash,
            method=method,
            seed=0,
            noise_hash=None,
            scheduler_factory_id=trace["scheduler_factory_id"],
            scheduler_object_id=trace["scheduler_object_id"],
            started_at=started,
            completed_at=utc_timestamp(),
            optimizer_semantics=trace["method_family"],
            projection_mode=trace["projection_mode"],
            state_transport_mode="not_applicable",
            residual_mode=residual_mode,
            second_moment_mode="not_applicable",
            command=run_command,
            parent_pr=parent_pr,
            outputs=outputs,
        )
        manifest_path, _, memory_path, scale_log_path = _write_run_artifacts(
            run_dir=run_dir,
            manifest=manifest,
            metrics=metrics,
            memory=memory,
            scale_rows=_scale_rows_from_coordinate(trace),
            trace=trace,
        )
        method_payloads[method] = trace
        metric_rows.append(metrics)
        manifest_paths.append(manifest_path)
        memory_paths.append(memory_path)
        scale_log_paths.append(scale_log_path)

    aggregate_id = f"{group_dir.name}_{experiment_name}_aggregate"
    aggregate_info = write_explicit_aggregate(
        output_dir=exp_dir / "aggregates" / aggregate_id,
        run_manifest_paths=manifest_paths,
        parent_task_id=TASK_ID,
        aggregate_id=aggregate_id,
    )
    aggregate_manifest = aggregate_info["aggregate_manifest"]
    summary = {
        "experiment_name": experiment_name,
        "aggregate_id": aggregate_id,
        "seed_count": 1,
        "source_run_ids": aggregate_manifest["source_run_ids"],
        "source_manifest_paths": aggregate_manifest["source_manifest_paths"],
        "theorem_conditions": theorem,
        "methods": {row["method"]: row for row in metric_rows},
        "claim_summary": {},
        "paper_quality": paper_quality,
    }
    summary["claim_summary"] = _claim_summary_for_theorem(summary)
    summary_path = exp_dir / "aggregates" / aggregate_id / "theorem_summary.json"
    write_json_strict(summary_path, summary, sort_keys=True)
    table_path = exp_dir / "tables" / "theorem_stationarity_table.md"
    lines = ["# Theorem-Regime Final Stationarity", "", "| Method | final x | final y | final ||grad|| | cum phi_raw | cum s_applied | residual |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in methods:
        row = summary["methods"][method]
        lines.append(
            f"| {method} | {row['final_x']:.6e} | {row['final_y']:.6e} | {row['grad_norm_final']:.6e} | {row['phi_raw_cum_final']:.6e} | {row['s_applied_cum_final']:.6e} | {0.0 if row['residual_norm_final'] is None else row['residual_norm_final']:.6e} |"
        )
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig = plot_theorem_regime(
        out_dir=exp_dir / "figures",
        aggregate_manifest_path=aggregate_info["aggregate_manifest_path"],
        aggregate_manifest=aggregate_manifest,
        method_payloads=method_payloads,
    )
    return {
        "manifest_paths": manifest_paths,
        "memory_paths": memory_paths,
        "scale_log_paths": scale_log_paths,
        "aggregate_manifest_path": aggregate_info["aggregate_manifest_path"],
        "aggregate_summary_path": aggregate_info["summary_path"],
        "experiment_summary_path": summary_path,
        "plot_metadata_paths": [fig["metadata"]],
        "figure_paths": [fig["png"], fig["pdf"]],
        "table_paths": [table_path],
        "method_order_report": [{k: v for k, v in row.items() if k in {"method", "noise_hash", "consumed_noise_hash"}} for row in metric_rows],
    }


def _run_high_dimensional_fixed(
    *,
    group_dir: Path,
    output_root: Path,
    metadata: dict[str, Any],
    tiny: bool,
    paper_quality: bool,
    parent_pr: str | None,
) -> dict[str, Any]:
    experiment_name = "high_dimensional_fixed"
    exp_dir = _experiment_dir(group_dir, experiment_name)
    methods = list(DEFAULT_COORD_METHODS)
    steps = 120 if tiny else 600
    settings = [
        {"setting": "flat", "d": 40 if tiny else 100, "r": 4 if tiny else 5, "lam": "flat"},
        {"setting": "anisotropic", "d": 40 if tiny else 100, "r": 4 if tiny else 5, "lam": "block"},
    ]
    seeds = [0] if tiny else [0, 1, 2]
    cfg = CoordinateConfig(lr=0.5, eps_a=1.0, eps_s=1.0, s_min=0.2, s_max=0.5, rho=0.5, rho_min=1e-6, rho_max=1.0, limiter_gamma=None, dtype=np.float64)
    manifest_paths = []
    memory_paths = []
    scale_log_paths = []
    metric_rows = []
    label_map = {m: m for m in methods}
    run_command = _command_for_run(output_root=output_root, run_id=group_dir.name, experiment_name=experiment_name, tiny=tiny, paper_quality=paper_quality, parent_pr=parent_pr)
    for setting in settings:
        d = int(setting["d"])
        r = int(setting["r"])
        lam = np.ones((d,), dtype=float)
        if setting["lam"] == "block":
            lam[: max(5, r)] = 5.0
        quad = DiagQuadratic(lam)
        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            x0 = np.ones((d,), dtype=float) + 0.05 * rng.standard_normal((d,))
            basis = np.eye(d, r, dtype=float)
            config = _config_dict(
                experiment_name,
                {"setting": setting["setting"], "d": d, "r": r, "steps": steps, "seed": seed, "methods": methods, "lam_kind": setting["lam"]},
            )
            cfg_hash = config_hash(config)
            for method in methods:
                started = utc_timestamp()
                schedule = FixedBasisSchedule(basis.copy(), projection_mode=f"fixed_first_{r}")
                start_perf = time.perf_counter()
                trace = run_coordinate_trace(method=method, quad=quad, x0=x0, steps=steps, schedule=schedule, cfg=cfg)
                elapsed_ms = wall_clock_ms(start_perf)
                run_id = build_run_id(
                    task=TASK_ID,
                    experiment=f"{experiment_name}_{setting['setting']}",
                    method=method,
                    seed=seed,
                    short_commit=str(metadata["short_commit"]),
                    config_hash_value=cfg_hash,
                    timestamp=started,
                )
                run_dir = exp_dir / "runs" / run_id
                run_dir.mkdir(parents=True, exist_ok=False)
                metrics = _metric_summary_from_series(method, trace["method_family"], trace, trace["noise_hash"], seed)
                metrics["setting"] = setting["setting"]
                metrics["d"] = d
                metrics["r"] = r
                memory = _memory_schema_for_coordinate(trace=trace, d=d, rank=r, cfg=cfg, elapsed_ms=elapsed_ms)
                outputs = {
                    "run_id": run_id,
                    "manifest": str(run_dir / "manifest.json"),
                    "metrics": str(run_dir / "metrics.json"),
                    "memory_runtime": str(run_dir / "memory_runtime.json"),
                    "scale_log": str(run_dir / "scale_log.jsonl"),
                    "trace": str(run_dir / "trace.json"),
                }
                manifest = _base_manifest(
                    metadata=metadata,
                    config=config,
                    cfg_hash=cfg_hash,
                    method=method,
                    seed=seed,
                    noise_hash=None,
                    scheduler_factory_id=trace["scheduler_factory_id"],
                    scheduler_object_id=trace["scheduler_object_id"],
                    started_at=started,
                    completed_at=utc_timestamp(),
                    optimizer_semantics=trace["method_family"],
                    projection_mode=trace["projection_mode"],
                    state_transport_mode="not_applicable",
                    residual_mode=("current_projection_compensated" if method == "becr" else ("rho_no_lower_bound" if method == "rho_no_lower_bound" else ("coupled_rho_phi_raw" if method == "coupled_unbounded" else "none"))),
                    second_moment_mode="not_applicable",
                    command=run_command,
                    parent_pr=parent_pr,
                    outputs=outputs,
                )
                manifest_path, _, memory_path, scale_log_path = _write_run_artifacts(
                    run_dir=run_dir,
                    manifest=manifest,
                    metrics=metrics,
                    memory=memory,
                    scale_rows=_scale_rows_from_coordinate(trace),
                    trace=trace,
                )
                manifest_paths.append(manifest_path)
                memory_paths.append(memory_path)
                scale_log_paths.append(scale_log_path)
                metric_rows.append(metrics)

    aggregate_id = f"{group_dir.name}_{experiment_name}_aggregate"
    aggregate_info = write_explicit_aggregate(
        output_dir=exp_dir / "aggregates" / aggregate_id,
        run_manifest_paths=manifest_paths,
        parent_task_id=TASK_ID,
        aggregate_id=aggregate_id,
    )
    aggregate_manifest = aggregate_info["aggregate_manifest"]
    summary = _aggregate_metrics(metric_rows, methods, label_map)
    summary.update(
        {
            "experiment_name": experiment_name,
            "aggregate_id": aggregate_id,
            "seed_count": len(seeds),
            "settings": settings,
            "source_run_ids": aggregate_manifest["source_run_ids"],
            "source_manifest_paths": aggregate_manifest["source_manifest_paths"],
            "claim_summary": {},
            "paper_quality": paper_quality,
        }
    )
    summary["claim_summary"] = _claim_summary_for_high_dimensional(summary)
    summary_path = exp_dir / "aggregates" / aggregate_id / "high_dimensional_summary.json"
    write_json_strict(summary_path, summary, sort_keys=True)
    table_path = exp_dir / "tables" / "high_dimensional_stationarity_table.md"
    lines = ["# High-Dimensional Final Metrics", "", "| Method | mean ||grad|| | mean ||grad_parallel|| | mean ||grad_perp|| | mean residual |", "|---|---:|---:|---:|---:|"]
    for method, row in summary["methods"].items():
        resid = row["residual_norm_mean"]
        lines.append(f"| {method} | {row['grad_norm_mean']:.6e} | {row['grad_par_norm_mean']:.6e} | {row['grad_perp_norm_mean']:.6e} | {0.0 if resid is None else resid:.6e} |")
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig = plot_high_dimensional(
        out_dir=exp_dir / "figures",
        aggregate_manifest_path=aggregate_info["aggregate_manifest_path"],
        aggregate_manifest=aggregate_manifest,
        summary=summary,
    )
    return {
        "manifest_paths": manifest_paths,
        "memory_paths": memory_paths,
        "scale_log_paths": scale_log_paths,
        "aggregate_manifest_path": aggregate_info["aggregate_manifest_path"],
        "aggregate_summary_path": aggregate_info["summary_path"],
        "experiment_summary_path": summary_path,
        "plot_metadata_paths": [fig["metadata"]],
        "figure_paths": [fig["png"], fig["pdf"]],
        "table_paths": [table_path],
        "method_order_report": [{k: row[k] for k in ("method", "noise_hash", "consumed_noise_hash")} for row in metric_rows],
    }


def _run_refresh_sweep(
    *,
    group_dir: Path,
    output_root: Path,
    metadata: dict[str, Any],
    tiny: bool,
    paper_quality: bool,
    parent_pr: str | None,
) -> dict[str, Any]:
    experiment_name = "refresh_sweep"
    exp_dir = _experiment_dir(group_dir, experiment_name)
    K_list = [1, None] if tiny else [1, 10, 50, 200, None]
    K_numeric = [1e6 if K is None else float(K) for K in K_list]
    d = 20 if tiny else 40
    r = 4 if tiny else 5
    steps = 80 if tiny else 300
    lam = np.asarray([10 ** (-2.0 * i / (d - 1)) for i in range(d)], dtype=float)
    quad = DiagQuadratic(lam)
    x0 = np.ones((d,), dtype=float)
    coord_methods = ["proj_baseline", "fira_raw", "fira_clipped", "becr"]
    cfg_coord = CoordinateConfig(lr=0.5, eps_a=1.0, eps_s=1.0, s_min=0.2, s_max=0.5, rho=0.5, rho_min=1e-6, rho_max=1.0, limiter_gamma=None, dtype=np.float64)
    cfg_state = MovingProjectionConfig(lr=0.5, rho=0.5, s_min=0.2, s_max=0.5, dtype=np.float64)
    manifest_paths = []
    memory_paths = []
    scale_log_paths = []
    metric_rows = []
    state_metric_rows = []
    run_command = _command_for_run(output_root=output_root, run_id=group_dir.name, experiment_name=experiment_name, tiny=tiny, paper_quality=paper_quality, parent_pr=parent_pr)
    for K in K_list:
        tag = "inf" if K is None else str(int(K))
        config_base = _config_dict(experiment_name, {"K": tag, "d": d, "r": r, "steps": steps})
        cfg_hash = config_hash(config_base)
        for method in coord_methods:
            started = utc_timestamp()
            schedule = TopGradientRefreshSchedule(d=d, r=r, refresh_K=K)
            start_perf = time.perf_counter()
            trace = run_coordinate_trace(method=method, quad=quad, x0=x0, steps=steps, schedule=schedule, cfg=cfg_coord)
            elapsed_ms = wall_clock_ms(start_perf)
            run_id = build_run_id(task=TASK_ID, experiment=f"{experiment_name}_coord_K{tag}", method=method, seed=0, short_commit=str(metadata["short_commit"]), config_hash_value=cfg_hash, timestamp=started)
            run_dir = exp_dir / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            metrics = _metric_summary_from_series(method, trace["method_family"], trace, trace["noise_hash"], 0)
            metrics["K"] = tag
            memory = _memory_schema_for_coordinate(trace=trace, d=d, rank=r, cfg=cfg_coord, elapsed_ms=elapsed_ms)
            outputs = {
                "run_id": run_id,
                "manifest": str(run_dir / "manifest.json"),
                "metrics": str(run_dir / "metrics.json"),
                "memory_runtime": str(run_dir / "memory_runtime.json"),
                "scale_log": str(run_dir / "scale_log.jsonl"),
                "trace": str(run_dir / "trace.json"),
            }
            manifest = _base_manifest(
                metadata=metadata,
                config=config_base | {"family": "coordinate", "method": method},
                cfg_hash=cfg_hash,
                method=method,
                seed=0,
                noise_hash=None,
                scheduler_factory_id=trace["scheduler_factory_id"],
                scheduler_object_id=trace["scheduler_object_id"],
                started_at=started,
                completed_at=utc_timestamp(),
                optimizer_semantics=trace["method_family"],
                projection_mode=trace["projection_mode"],
                state_transport_mode="not_applicable",
                residual_mode=("current_projection_compensated" if method == "becr" else "none"),
                second_moment_mode="not_applicable",
                command=run_command,
                parent_pr=parent_pr,
                outputs=outputs,
            )
            manifest_path, _, memory_path, scale_log_path = _write_run_artifacts(run_dir=run_dir, manifest=manifest, metrics=metrics, memory=memory, scale_rows=_scale_rows_from_coordinate(trace), trace=trace)
            manifest_paths.append(manifest_path)
            memory_paths.append(memory_path)
            scale_log_paths.append(scale_log_path)
            metric_rows.append(metrics)
        for mode in STATE_MODES:
            started = utc_timestamp()
            schedule = TopGradientRefreshSchedule(d=d, r=r, refresh_K=K)
            start_perf = time.perf_counter()
            trace = run_moving_projection_dynamic_trace(mode=mode, quad=quad, x0=x0, steps=steps, schedule=schedule, cfg=cfg_state)
            elapsed_ms = wall_clock_ms(start_perf)
            run_id = build_run_id(task=TASK_ID, experiment=f"{experiment_name}_state_K{tag}", method=mode, seed=0, short_commit=str(metadata["short_commit"]), config_hash_value=cfg_hash, timestamp=started)
            run_dir = exp_dir / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            metrics = _state_metric_summary(mode, trace, trace["noise_hash"], 0)
            metrics["K"] = tag
            memory = _memory_schema_for_state(d=d, rank=r, cfg=cfg_state, elapsed_ms=elapsed_ms)
            outputs = {
                "run_id": run_id,
                "manifest": str(run_dir / "manifest.json"),
                "metrics": str(run_dir / "metrics.json"),
                "memory_runtime": str(run_dir / "memory_runtime.json"),
                "scale_log": str(run_dir / "scale_log.jsonl"),
                "trace": str(run_dir / "trace.json"),
            }
            manifest = _base_manifest(
                metadata=metadata,
                config=config_base | {"family": "state_modes", "mode": mode},
                cfg_hash=cfg_hash,
                method=mode,
                seed=0,
                noise_hash=None,
                scheduler_factory_id=trace["scheduler_factory_id"],
                scheduler_object_id=trace["scheduler_object_id"],
                started_at=started,
                completed_at=utc_timestamp(),
                optimizer_semantics="corrected_moving_projection_semantics_v1",
                projection_mode=trace["projection_mode"],
                state_transport_mode=mode if mode != "full_residual_current_projection" else "basis_overlap_transport",
                residual_mode=("current_projection_compensated" if mode == "full_residual_current_projection" else "none"),
                second_moment_mode=("squared_basis_overlap_transport" if mode in {"projection_aware_transport", "full_residual_current_projection"} else ("carry_unchanged" if mode == "official_fira_carry" else "reset_on_refresh")),
                command=run_command,
                parent_pr=parent_pr,
                outputs=outputs,
            )
            manifest_path, _, memory_path, scale_log_path = _write_run_artifacts(run_dir=run_dir, manifest=manifest, metrics=metrics, memory=memory, scale_rows=_scale_rows_from_state(trace), trace=trace)
            manifest_paths.append(manifest_path)
            memory_paths.append(memory_path)
            scale_log_paths.append(scale_log_path)
            state_metric_rows.append(metrics)
    aggregate_id = f"{group_dir.name}_{experiment_name}_aggregate"
    aggregate_info = write_explicit_aggregate(output_dir=exp_dir / "aggregates" / aggregate_id, run_manifest_paths=manifest_paths, parent_task_id=TASK_ID, aggregate_id=aggregate_id)
    aggregate_manifest = aggregate_info["aggregate_manifest"]
    coord_summary = {}
    for method in coord_methods:
        rows = [row for row in metric_rows if row["method"] == method]
        coord_summary[method] = {
            "label": method,
            "grad_norm_final": [row["grad_norm_final"] for row in rows],
            "residual_norm_final": [0.0 if row["residual_norm_final"] is None else row["residual_norm_final"] for row in rows],
        }
    state_summary = {}
    for mode in STATE_MODES:
        rows = [row for row in state_metric_rows if row["method"] == mode]
        state_summary[mode] = {
            "label": mode,
            "grad_norm_final": [row["grad_norm_final"] for row in rows],
            "residual_norm_final": [row["residual_norm_final"] for row in rows],
        }
    summary = {
        "experiment_name": experiment_name,
        "aggregate_id": aggregate_id,
        "seed_count": 1,
        "K_values": ["inf" if K is None else int(K) for K in K_list],
        "K_values_numeric": K_numeric,
        "source_run_ids": aggregate_manifest["source_run_ids"],
        "source_manifest_paths": aggregate_manifest["source_manifest_paths"],
        "coordinate_family": coord_summary,
        "state_modes": state_summary,
        "paper_quality": paper_quality,
    }
    summary_path = exp_dir / "aggregates" / aggregate_id / "refresh_sweep_summary.json"
    write_json_strict(summary_path, summary, sort_keys=True)
    table_path = exp_dir / "tables" / "refresh_sweep_table.md"
    lines = ["# Refresh Sweep Summary", "", f"K values: {summary['K_values']}", ""]
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig = plot_refresh_sweep(out_dir=exp_dir / "figures", aggregate_manifest_path=aggregate_info["aggregate_manifest_path"], aggregate_manifest=aggregate_manifest, summary=summary)
    all_rows = metric_rows + state_metric_rows
    return {
        "manifest_paths": manifest_paths,
        "memory_paths": memory_paths,
        "scale_log_paths": scale_log_paths,
        "aggregate_manifest_path": aggregate_info["aggregate_manifest_path"],
        "aggregate_summary_path": aggregate_info["summary_path"],
        "experiment_summary_path": summary_path,
        "plot_metadata_paths": [fig["metadata"]],
        "figure_paths": [fig["png"], fig["pdf"]],
        "table_paths": [table_path],
        "method_order_report": [{k: row[k] for k in ("method", "noise_hash", "consumed_noise_hash")} for row in all_rows],
    }


def _run_anisotropic_noise(
    *,
    group_dir: Path,
    output_root: Path,
    metadata: dict[str, Any],
    tiny: bool,
    paper_quality: bool,
    parent_pr: str | None,
    method_order: list[str] | None = None,
) -> dict[str, Any]:
    experiment_name = "anisotropic_noise"
    exp_dir = _experiment_dir(group_dir, experiment_name)
    methods = list(method_order or DEFAULT_COORD_METHODS)
    d = 30 if tiny else 60
    r = 4 if tiny else 5
    steps = 80 if tiny else 240
    seeds = [0, 1] if tiny else [0, 1, 2, 3, 4]
    batch_size = 4 if tiny else 8
    schedule_update_K = 10 if tiny else 20
    lam = np.ones((d,), dtype=float)
    lam[: min(10, d)] = 5.0
    quad = DiagQuadratic(lam)
    x0 = np.ones((d,), dtype=float)
    sigma = np.full((d,), 0.02, dtype=float)
    sigma[min(10, d // 2) : min(20, d)] = 0.2
    cfg = CoordinateConfig(lr=0.05, eps_a=1e-3, eps_s=1e-3, s_min=0.1, s_max=10.0, rho=0.5, rho_min=1e-6, rho_max=1.0, limiter_gamma=None, dtype=np.float64)
    manifest_paths = []
    memory_paths = []
    scale_log_paths = []
    metric_rows = []
    label_map = {m: m for m in methods}
    run_command = _command_for_run(output_root=output_root, run_id=group_dir.name, experiment_name=experiment_name, tiny=tiny, paper_quality=paper_quality, parent_pr=parent_pr)
    for seed in seeds:
        noise_bank = generate_crn_noise_bank(steps=steps, batch_size=batch_size, d=d, sigma=sigma, rng_seed=seed)
        config = _config_dict(experiment_name, {"d": d, "r": r, "steps": steps, "seed": seed, "batch_size": batch_size, "methods": methods, "schedule": "svd_batch_refresh"})
        cfg_hash = config_hash(config)
        for method in methods:
            started = utc_timestamp()
            schedule = SVDBatchRefreshSchedule(d=d, r=r, refresh_K=schedule_update_K, random_projection=False, rng_seed=seed)
            start_perf = time.perf_counter()
            trace = run_coordinate_trace(method=method, quad=quad, x0=x0, steps=steps, schedule=schedule, cfg=cfg, noise_bank=noise_bank, batch_size=batch_size)
            elapsed_ms = wall_clock_ms(start_perf)
            run_id = build_run_id(task=TASK_ID, experiment=experiment_name, method=method, seed=seed, short_commit=str(metadata["short_commit"]), config_hash_value=cfg_hash, timestamp=started)
            run_dir = exp_dir / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            metrics = _metric_summary_from_series(method, trace["method_family"], trace, trace["noise_hash"], seed)
            memory = _memory_schema_for_coordinate(trace=trace, d=d, rank=r, cfg=cfg, elapsed_ms=elapsed_ms)
            outputs = {
                "run_id": run_id,
                "manifest": str(run_dir / "manifest.json"),
                "metrics": str(run_dir / "metrics.json"),
                "memory_runtime": str(run_dir / "memory_runtime.json"),
                "scale_log": str(run_dir / "scale_log.jsonl"),
                "trace": str(run_dir / "trace.json"),
            }
            manifest = _base_manifest(
                metadata=metadata,
                config=config,
                cfg_hash=cfg_hash,
                method=method,
                seed=seed,
                noise_hash=trace["noise_hash"],
                scheduler_factory_id=trace["scheduler_factory_id"],
                scheduler_object_id=trace["scheduler_object_id"],
                started_at=started,
                completed_at=utc_timestamp(),
                optimizer_semantics=trace["method_family"],
                projection_mode=trace["projection_mode"],
                state_transport_mode="not_applicable",
                residual_mode=("current_projection_compensated" if method == "becr" else ("rho_no_lower_bound" if method == "rho_no_lower_bound" else ("coupled_rho_phi_raw" if method == "coupled_unbounded" else "none"))),
                second_moment_mode="not_applicable",
                command=run_command,
                parent_pr=parent_pr,
                outputs=outputs,
            )
            manifest_path, _, memory_path, scale_log_path = _write_run_artifacts(run_dir=run_dir, manifest=manifest, metrics=metrics, memory=memory, scale_rows=_scale_rows_from_coordinate(trace), trace=trace)
            manifest_paths.append(manifest_path)
            memory_paths.append(memory_path)
            scale_log_paths.append(scale_log_path)
            metric_rows.append(metrics)
    aggregate_id = f"{group_dir.name}_{experiment_name}_aggregate"
    aggregate_info = write_explicit_aggregate(output_dir=exp_dir / "aggregates" / aggregate_id, run_manifest_paths=manifest_paths, parent_task_id=TASK_ID, aggregate_id=aggregate_id)
    aggregate_manifest = aggregate_info["aggregate_manifest"]
    summary = _aggregate_metrics(metric_rows, methods, label_map)
    by_method = summary["methods"]
    clipped_rows = [row for row in metric_rows if row["method"] == "fira_clipped"]
    becr_rows = [row for row in metric_rows if row["method"] == "becr"]
    raw_rows = [row for row in metric_rows if row["method"] == "fira_raw"]
    proj_rows = [row for row in metric_rows if row["method"] == "proj_baseline"]
    clipped_by_seed = {row["seed"]: row for row in clipped_rows}
    becr_by_seed = {row["seed"]: row for row in becr_rows}
    raw_by_seed = {row["seed"]: row for row in raw_rows}
    proj_by_seed = {row["seed"]: row for row in proj_rows}
    paired_clipped = []
    paired_raw = []
    paired_proj = []
    for seed in sorted(set(clipped_by_seed) & set(becr_by_seed)):
        paired_clipped.append(becr_by_seed[seed]["grad_norm_final"] - clipped_by_seed[seed]["grad_norm_final"])
    for seed in sorted(set(raw_by_seed) & set(becr_by_seed)):
        paired_raw.append(becr_by_seed[seed]["grad_norm_final"] - raw_by_seed[seed]["grad_norm_final"])
    for seed in sorted(set(proj_by_seed) & set(becr_by_seed)):
        paired_proj.append(becr_by_seed[seed]["grad_norm_final"] - proj_by_seed[seed]["grad_norm_final"])
    summary.update(
        {
            "experiment_name": experiment_name,
            "aggregate_id": aggregate_id,
            "seed_count": len(seeds),
            "source_run_ids": aggregate_manifest["source_run_ids"],
            "source_manifest_paths": aggregate_manifest["source_manifest_paths"],
            "paired_differences": {
                "becr_minus_clipped": paired_clipped,
                "becr_minus_raw": paired_raw,
                "becr_minus_proj_baseline": paired_proj,
            },
            "claim_summary": {},
            "paper_quality": paper_quality,
        }
    )
    summary["claim_summary"] = _claim_summary_for_anisotropic(summary)
    summary_path = exp_dir / "aggregates" / aggregate_id / "anisotropic_noise_summary.json"
    write_json_strict(summary_path, summary, sort_keys=True)
    table_path = exp_dir / "tables" / "anisotropic_noise_table.md"
    lines = ["# Anisotropic Noise Summary", "", "| Method | mean ||grad|| | CI low | CI high |", "|---|---:|---:|---:|"]
    for method, row in by_method.items():
        lines.append(f"| {method} | {row['grad_norm_mean']:.6e} | {row['grad_norm_ci_low']:.6e} | {row['grad_norm_ci_high']:.6e} |")
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig = plot_anisotropic_noise(out_dir=exp_dir / "figures", aggregate_manifest_path=aggregate_info["aggregate_manifest_path"], aggregate_manifest=aggregate_manifest, summary=summary)
    return {
        "manifest_paths": manifest_paths,
        "memory_paths": memory_paths,
        "scale_log_paths": scale_log_paths,
        "aggregate_manifest_path": aggregate_info["aggregate_manifest_path"],
        "aggregate_summary_path": aggregate_info["summary_path"],
        "experiment_summary_path": summary_path,
        "plot_metadata_paths": [fig["metadata"]],
        "figure_paths": [fig["png"], fig["pdf"]],
        "table_paths": [table_path],
        "method_order_report": [{k: row[k] for k in ("method", "noise_hash", "consumed_noise_hash")} for row in metric_rows],
    }


def _run_exact_fira_fixture(*, group_dir: Path) -> dict[str, Any]:
    exp_dir = _experiment_dir(group_dir, "exact_fira_fixture")
    fixture = run_fixture("shape_2x1_rank1_gap1_fp64")
    out_path = exp_dir / "aggregates" / "exact_fira_fixture_summary.json"
    summary = {
        "fixture_id": fixture["config"].fixture_id,
        "oracle_final_grad_norm": float(np.linalg.norm(np.asarray(fixture["oracle_traces"][-1]["raw_gradient"], dtype=float))),
        "candidate_final_grad_norm": float(np.linalg.norm(np.asarray(fixture["candidate_traces"][-1]["raw_gradient"], dtype=float))),
        "first_mismatch": fixture["comparison"]["first_mismatch"] or "none",
        "max_errors": fixture["comparison"]["max_errors"],
        "method_class": "official_fira_oracle_fixture",
    }
    write_json_strict(out_path, summary, sort_keys=True)
    return {"summary_path": out_path}


def _write_ldadam_status(group_dir: Path) -> Path:
    out = group_dir / "ldadam_status.json"
    write_json_strict(
        out,
        {
            "included": False,
            "reason": "Not included in P1 because matched synthetic state-compensation semantics and memory assumptions would require a separate audited baseline.",
            "claim_implication": "No LDAdam comparison claim is made in corrected synthetic evidence.",
        },
        sort_keys=True,
    )
    return out


def run_corrected_synthetic_suite(
    *,
    output_root: Path,
    run_id: str,
    paper_quality: bool = False,
    allow_dirty: bool = False,
    overwrite_debug: bool = False,
    experiments: list[str] | None = None,
    tiny: bool = False,
    method_order: list[str] | None = None,
    parent_pr: str | None = None,
    command: str | None = None,
    review_snapshot_dir: Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    group_dir = output_root / run_id
    if group_dir.exists():
        if not overwrite_debug:
            raise FileExistsError(f"run output already exists: {group_dir}")
        import shutil

        shutil.rmtree(group_dir)
    metadata = get_git_metadata(Path.cwd())
    if paper_quality and metadata["dirty"] and not allow_dirty:
        raise RuntimeError("paper-quality run requires a clean worktree unless allow_dirty is set")
    group_dir.mkdir(parents=True, exist_ok=False)
    selected = list(experiments or ["theorem_regime", "high_dimensional_fixed", "refresh_sweep", "anisotropic_noise"])
    command = command or _command_for_run(output_root=output_root, run_id=run_id, experiment_name=None, tiny=tiny, paper_quality=paper_quality, parent_pr=parent_pr)
    results = {
        "group_dir": group_dir,
        "manifest_paths": [],
        "aggregate_manifest_paths": [],
        "aggregate_summary_paths": [],
        "experiment_summary_paths": [],
        "plot_metadata_paths": [],
        "figure_paths": [],
        "table_paths": [],
        "memory_paths": [],
        "scale_log_paths": [],
        "method_order_report": {},
    }
    if "theorem_regime" in selected:
        theorem_out = _run_theorem_regime(group_dir=group_dir, output_root=output_root, metadata=metadata, tiny=tiny, paper_quality=paper_quality, parent_pr=parent_pr)
        for key in ("manifest_paths", "memory_paths", "scale_log_paths", "figure_paths", "plot_metadata_paths", "table_paths"):
            results[key].extend(theorem_out[key])
        results["aggregate_manifest_paths"].append(theorem_out["aggregate_manifest_path"])
        results["aggregate_summary_paths"].append(theorem_out["aggregate_summary_path"])
        results["experiment_summary_paths"].append(theorem_out["experiment_summary_path"])
        results["method_order_report"]["theorem_regime"] = theorem_out["method_order_report"]
    if "high_dimensional_fixed" in selected:
        hd_out = _run_high_dimensional_fixed(group_dir=group_dir, output_root=output_root, metadata=metadata, tiny=tiny, paper_quality=paper_quality, parent_pr=parent_pr)
        for key in ("manifest_paths", "memory_paths", "scale_log_paths", "figure_paths", "plot_metadata_paths", "table_paths"):
            results[key].extend(hd_out[key])
        results["aggregate_manifest_paths"].append(hd_out["aggregate_manifest_path"])
        results["aggregate_summary_paths"].append(hd_out["aggregate_summary_path"])
        results["experiment_summary_paths"].append(hd_out["experiment_summary_path"])
        results["method_order_report"]["high_dimensional_fixed"] = hd_out["method_order_report"]
    if "refresh_sweep" in selected:
        refresh_out = _run_refresh_sweep(group_dir=group_dir, output_root=output_root, metadata=metadata, tiny=tiny, paper_quality=paper_quality, parent_pr=parent_pr)
        for key in ("manifest_paths", "memory_paths", "scale_log_paths", "figure_paths", "plot_metadata_paths", "table_paths"):
            results[key].extend(refresh_out[key])
        results["aggregate_manifest_paths"].append(refresh_out["aggregate_manifest_path"])
        results["aggregate_summary_paths"].append(refresh_out["aggregate_summary_path"])
        results["experiment_summary_paths"].append(refresh_out["experiment_summary_path"])
        results["method_order_report"]["refresh_sweep"] = refresh_out["method_order_report"]
    if "anisotropic_noise" in selected:
        noise_out = _run_anisotropic_noise(group_dir=group_dir, output_root=output_root, metadata=metadata, tiny=tiny, paper_quality=paper_quality, parent_pr=parent_pr, method_order=method_order)
        for key in ("manifest_paths", "memory_paths", "scale_log_paths", "figure_paths", "plot_metadata_paths", "table_paths"):
            results[key].extend(noise_out[key])
        results["aggregate_manifest_paths"].append(noise_out["aggregate_manifest_path"])
        results["aggregate_summary_paths"].append(noise_out["aggregate_summary_path"])
        results["experiment_summary_paths"].append(noise_out["experiment_summary_path"])
        results["method_order_report"]["anisotropic_noise"] = noise_out["method_order_report"]
    exact_out = _run_exact_fira_fixture(group_dir=group_dir)
    results["exact_fira_summary_path"] = exact_out["summary_path"]
    results["ldadam_status_path"] = _write_ldadam_status(group_dir)
    write_json_strict(group_dir / "run_manifest_index.json", {"run_id": run_id, "manifest_paths": [str(path) for path in results["manifest_paths"]]}, sort_keys=True)
    memory_runtime_summary_path = _write_memory_runtime_summary(group_dir=group_dir, manifest_paths=results["manifest_paths"], memory_paths=results["memory_paths"])
    scale_log_summary_path = _write_scale_log_summary(group_dir=group_dir, manifest_paths=results["manifest_paths"], scale_log_paths=results["scale_log_paths"])
    results["memory_runtime_summary_path"] = memory_runtime_summary_path
    results["scale_log_summary_path"] = scale_log_summary_path
    artifact_index_path = _write_artifact_index(
        group_dir=group_dir,
        output_root=output_root,
        metadata=metadata,
        run_id=run_id,
        paper_quality=paper_quality,
        tiny=tiny,
        parent_pr=parent_pr,
        command=command,
        results=results,
        memory_runtime_summary_path=memory_runtime_summary_path,
        scale_log_summary_path=scale_log_summary_path,
    )
    results["artifact_index_path"] = artifact_index_path
    if review_snapshot_dir is not None:
        results["review_snapshot_dir"] = _write_review_snapshot(group_dir=group_dir, artifact_index_path=artifact_index_path, review_snapshot_dir=Path(review_snapshot_dir))
    return results
