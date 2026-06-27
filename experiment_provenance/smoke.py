from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from moving_projection_state.core import MovingProjectionConfig, make_basis, run_trace

from .aggregate import write_explicit_aggregate
from .ids import build_run_id, config_hash, utc_timestamp
from .json_utils import sanitize_for_json, write_json_strict, write_jsonl_strict
from .metadata import get_git_metadata
from .noise import generate_noise_bank, hash_array
from .schema import (
    MANIFEST_SCHEMA_VERSION,
    MEMORY_SCHEMA_VERSION,
    SCALE_LOGGING_SCHEMA_VERSION,
    build_memory_runtime_schema,
    validate_manifest,
)


TASK_ID = "P0-CRN-MANIFEST-003"
EXPERIMENT_NAME = "tier1synthetic_provenance_smoke"
DEFAULT_METHODS = [
    "official_fira_carry",
    "projection_aware_transport",
    "full_residual_current_projection",
]


def _smoke_gradients(dtype) -> list[np.ndarray]:
    return [
        np.asarray([2.0, 0.5], dtype=dtype),
        np.asarray([1.5, 0.7], dtype=dtype),
        np.asarray([0.8, -1.2], dtype=dtype),
        np.asarray([0.6, -1.0], dtype=dtype),
        np.asarray([-0.5, 0.9], dtype=dtype),
        np.asarray([-0.4, 0.7], dtype=dtype),
    ]


def _smoke_bases(dtype) -> list[np.ndarray]:
    return [
        make_basis(2, [0], dtype=dtype),
        make_basis(2, [0], dtype=dtype),
        make_basis(2, [1], dtype=dtype),
        make_basis(2, [1], dtype=dtype),
        make_basis(2, [0], dtype=dtype),
        make_basis(2, [0], dtype=dtype),
    ]


def _config_dict(*, methods: list[str], seed: int, noise_std: float, cfg: MovingProjectionConfig) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "methods": methods,
        "seed": int(seed),
        "noise_std": float(noise_std),
        "lr": float(cfg.lr),
        "beta1": float(cfg.beta1),
        "beta2": float(cfg.beta2),
        "eps_adam": float(cfg.eps_adam),
        "eps_scale": float(cfg.eps_scale),
        "s_min": float(cfg.s_min),
        "s_max": float(cfg.s_max),
        "rho": float(cfg.rho),
        "rho_min": float(cfg.rho_min),
        "rho_max": float(cfg.rho_max),
        "limiter_gamma": None if cfg.limiter_gamma is None else float(cfg.limiter_gamma),
        "dtype": str(np.dtype(cfg.dtype)),
    }


def _memory_schema(*, method: str, cfg: MovingProjectionConfig, bases: list[np.ndarray], gradients: list[np.ndarray]) -> dict[str, Any]:
    dtype = np.dtype(cfg.dtype)
    dim = int(gradients[0].shape[0])
    rank = int(bases[0].shape[1])
    itemsize = dtype.itemsize
    uses_residual = method == "full_residual_current_projection"
    return build_memory_runtime_schema(
        device="cpu",
        parameter_bytes=dim * itemsize,
        gradient_buffer_bytes=dim * itemsize,
        first_moment_bytes=rank * itemsize,
        second_moment_bytes=rank * itemsize,
        projection_bytes=max(int(np.asarray(P).nbytes) for P in bases),
        residual_bytes=(dim * itemsize if uses_residual else 0),
        temporary_buffer_bytes_estimate=dim * itemsize,
    )


def _scale_log_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in result["steps"]:
        q_norm = float(np.linalg.norm(np.asarray(step["Q"], dtype=float)))
        z_norm = float(np.linalg.norm(np.asarray(step["Z"], dtype=float)))
        z_eff_norm = float(np.linalg.norm(np.asarray(step["Z_eff"], dtype=float)))
        rows.append(
            {
                "step_index": int(step["step_index"]),
                "phi_raw": step["phi_raw"],
                "s_raw": step["phi_raw"],
                "s_applied": step["s"],
                "rho": step["rho"],
                "tau": step["tau"],
                "Z_norm": z_norm,
                "Z_eff_norm": z_eff_norm,
                "effective_raw_transmission": (z_eff_norm / q_norm) if q_norm > 0.0 else None,
                "residual_update_error": step["residual_error"],
                "limiter_active": bool(abs(float(step["tau"]) - 1.0) > 1e-12),
                "raw_recovery_scale": step["phi_raw"],
                "applied_recovery_scale": step["s"],
                "effective_recovery_scale": float(step["s"]) * float(step["tau"]),
            }
        )
    return rows


def run_provenance_smoke(
    *,
    output_root: Path,
    run_id: str,
    methods: list[str] | None = None,
    seed: int = 0,
    noise_std: float = 0.15,
    paper_quality: bool = False,
    allow_dirty: bool = False,
    overwrite_debug: bool = False,
    parent_pr: str | None = None,
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
    methods = list(methods or DEFAULT_METHODS)
    cfg = MovingProjectionConfig(dtype=np.float64, rho=0.5, limiter_gamma=1.25)
    gradients = _smoke_gradients(np.float64)
    bases = _smoke_bases(np.float64)
    config = _config_dict(methods=methods, seed=seed, noise_std=noise_std, cfg=cfg)
    cfg_hash = config_hash(config)
    noise_bank = generate_noise_bank(gradients, std=noise_std, rng_seed=seed, dtype=np.float64)
    noise_hash = hash_array(noise_bank)

    runs_dir = group_dir / "runs"
    aggregates_dir = group_dir / "aggregates"
    runs_dir.mkdir(parents=True, exist_ok=True)
    aggregates_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths: list[Path] = []
    scale_log_paths: list[Path] = []
    memory_paths: list[Path] = []
    metrics_paths: list[Path] = []
    trace_paths: list[Path] = []
    manifests: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []

    for method in methods:
        started_at = utc_timestamp()
        scheduler_factory_id = f"{run_id}:{method}:scheduler"
        t0 = time.perf_counter()
        result = run_trace(
            mode=method,
            cfg=cfg,
            gradients=gradients,
            bases=bases,
            noise_bank=noise_bank,
            rng_seed=seed,
            scheduler_factory_id=scheduler_factory_id,
            parent_task_id=TASK_ID,
            parent_pr=parent_pr,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        method_run_id = build_run_id(
            task=TASK_ID,
            experiment=EXPERIMENT_NAME,
            method=method,
            seed=seed,
            short_commit=str(metadata["short_commit"]),
            config_hash_value=cfg_hash,
            timestamp=started_at,
        )
        run_dir = runs_dir / method_run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        memory = _memory_schema(method=method, cfg=cfg, bases=bases, gradients=gradients)
        memory["wall_clock_time_ms"] = elapsed_ms
        memory["optimizer_step_time_ms"] = elapsed_ms / max(len(result["steps"]), 1)

        scale_rows = _scale_log_rows(result)
        scale_log_path = run_dir / "scale_log.jsonl"
        metrics_path = run_dir / "metrics.json"
        memory_path = run_dir / "memory_runtime.json"
        trace_path = run_dir / "trace.json"
        manifest_path = run_dir / "manifest.json"

        metrics = {
            "method": method,
            "run_id": method_run_id,
            "noise_hash": noise_hash,
            "consumed_noise_hash": hash_array(np.asarray(result["noise_bank"], dtype=np.float64)),
            "max_decomposition_error": result["max_decomposition_error"],
            "max_residual_error": result["max_residual_error"],
            "max_first_moment_transport_error": result["max_first_moment_transport_error"],
            "max_second_moment_transport_error": result["max_second_moment_transport_error"],
            "old_results_used": False,
            "legacy_results_used": False,
        }

        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": method_run_id,
            "parent_task_id": TASK_ID,
            "parent_pr": parent_pr,
            "code_commit": metadata["code_commit"],
            "dirty": bool(metadata["dirty"]),
            "branch": metadata["branch"],
            "command": f"python experiments/tier1-synthetic/run_provenance_smoke.py --output-root {output_root} --run-id {run_id}",
            "config": config,
            "config_hash": cfg_hash,
            "method": method,
            "mode": method,
            "seed": int(seed),
            "rng_seed": int(seed),
            "noise_hash": noise_hash,
            "dataset_hash": None,
            "scheduler_factory_id": result["manifest"]["scheduler_factory_id"],
            "scheduler_object_id": result["manifest"]["scheduler_object_id"],
            "started_at": started_at,
            "completed_at": utc_timestamp(),
            "old_results_used": False,
            "legacy_results_used": False,
            "optimizer_semantics": "corrected_moving_projection_semantics_v1",
            "projection_mode": result["manifest"]["projection_mode"],
            "state_transport_mode": result["manifest"]["state_transport_mode"],
            "residual_mode": result["manifest"]["residual_mode"],
            "second_moment_mode": result["manifest"]["second_moment_mode"],
            "weight_decay_semantics": "decoupled_none",
            "scale_logging_schema": SCALE_LOGGING_SCHEMA_VERSION,
            "memory_schema_version": MEMORY_SCHEMA_VERSION,
            "outputs": {
                "manifest": str(manifest_path),
                "metrics": str(metrics_path),
                "scale_log": str(scale_log_path),
                "memory_runtime": str(memory_path),
                "trace": str(trace_path),
            },
        }
        validate_manifest(manifest)

        write_json_strict(metrics_path, metrics, sort_keys=True)
        write_jsonl_strict(scale_log_path, scale_rows)
        write_json_strict(memory_path, memory, sort_keys=True)
        write_json_strict(trace_path, sanitize_for_json(result), sort_keys=True)
        write_json_strict(manifest_path, manifest, sort_keys=True)

        manifest_paths.append(manifest_path)
        scale_log_paths.append(scale_log_path)
        memory_paths.append(memory_path)
        metrics_paths.append(metrics_path)
        trace_paths.append(trace_path)
        manifests.append(manifest)
        metrics_rows.append(metrics)

    aggregate_id = f"{run_id}_aggregate"
    aggregate_dir = aggregates_dir / aggregate_id
    aggregate = write_explicit_aggregate(
        output_dir=aggregate_dir,
        run_manifest_paths=manifest_paths,
        parent_task_id=TASK_ID,
        aggregate_id=aggregate_id,
    )
    return {
        "paper_quality": bool(paper_quality and not overwrite_debug),
        "group_dir": group_dir,
        "run_ids": [manifest["run_id"] for manifest in manifests],
        "manifest_paths": manifest_paths,
        "scale_log_paths": scale_log_paths,
        "memory_paths": memory_paths,
        "metrics_paths": metrics_paths,
        "trace_paths": trace_paths,
        "aggregate_manifest_path": aggregate["aggregate_manifest_path"],
        "summary_path": aggregate["summary_path"],
        "manifests": manifests,
        "metrics": metrics_rows,
    }
