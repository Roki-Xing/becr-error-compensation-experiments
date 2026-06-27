from __future__ import annotations

from typing import Any


MANIFEST_SCHEMA_VERSION = "p0-crn-manifest-v1"
AGGREGATE_SCHEMA_VERSION = "aggregate-v1"
SCALE_LOGGING_SCHEMA_VERSION = "raw_applied_effective_v1"
MEMORY_SCHEMA_VERSION = "memory_v1"

REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "run_id",
    "parent_task_id",
    "parent_pr",
    "code_commit",
    "dirty",
    "branch",
    "command",
    "config",
    "config_hash",
    "method",
    "mode",
    "seed",
    "rng_seed",
    "noise_hash",
    "dataset_hash",
    "scheduler_factory_id",
    "scheduler_object_id",
    "started_at",
    "completed_at",
    "old_results_used",
    "legacy_results_used",
    "optimizer_semantics",
    "projection_mode",
    "state_transport_mode",
    "residual_mode",
    "second_moment_mode",
    "weight_decay_semantics",
    "scale_logging_schema",
    "memory_schema_version",
    "outputs",
}


def validate_manifest(manifest: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest.keys()))
    if missing:
        raise ValueError(f"manifest missing required fields: {missing}")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {manifest['schema_version']}")


def build_memory_runtime_schema(
    *,
    device: str,
    parameter_bytes: int | None,
    gradient_buffer_bytes: int | None,
    first_moment_bytes: int | None,
    second_moment_bytes: int | None,
    projection_bytes: int | None,
    residual_bytes: int | None,
    temporary_buffer_bytes_estimate: int | None,
    peak_memory_allocated: int | None = None,
    peak_memory_reserved: int | None = None,
    optimizer_step_time_ms: float | None = None,
    wall_clock_time_ms: float | None = None,
) -> dict[str, Any]:
    return {
        "parameter_bytes": parameter_bytes,
        "gradient_buffer_bytes": gradient_buffer_bytes,
        "first_moment_bytes": first_moment_bytes,
        "second_moment_bytes": second_moment_bytes,
        "projection_bytes": projection_bytes,
        "residual_bytes": residual_bytes,
        "temporary_buffer_bytes_estimate": temporary_buffer_bytes_estimate,
        "peak_memory_allocated": peak_memory_allocated,
        "peak_memory_reserved": peak_memory_reserved,
        "optimizer_step_time_ms": optimizer_step_time_ms,
        "wall_clock_time_ms": wall_clock_time_ms,
        "device": device,
    }
