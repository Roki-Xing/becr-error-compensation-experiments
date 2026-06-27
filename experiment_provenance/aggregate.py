from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .json_utils import write_json_strict
from .schema import AGGREGATE_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported run manifest schema at {path}")
    if "legacy_diagnostic" in path.as_posix() or path.name == "ALL_RUNS.json":
        raise ValueError(f"legacy result path not allowed: {path}")
    return manifest


def write_explicit_aggregate(
    *,
    output_dir: Path,
    run_manifest_paths: list[Path],
    parent_task_id: str = "P0-CRN-MANIFEST-003",
    aggregate_id: str | None = None,
) -> dict[str, Any]:
    if not run_manifest_paths:
        raise ValueError("explicit run manifest paths are required")
    manifests = [_load_manifest(Path(path)) for path in run_manifest_paths]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_id = aggregate_id or output_dir.name
    config_hashes = sorted({manifest["config_hash"] for manifest in manifests})
    noise_hashes = sorted({manifest["noise_hash"] for manifest in manifests})
    weight_decay_match = len({manifest["weight_decay_semantics"] for manifest in manifests}) == 1
    aggregate_manifest = {
        "aggregate_id": aggregate_id,
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "parent_task_id": parent_task_id,
        "code_commit": manifests[0]["code_commit"],
        "source_run_ids": [manifest["run_id"] for manifest in manifests],
        "source_manifest_paths": [str(Path(path)) for path in run_manifest_paths],
        "old_results_used": False,
        "legacy_results_used": False,
        "created_at": manifests[-1]["completed_at"],
        "config_hashes": config_hashes,
        "noise_hashes": noise_hashes,
    }
    summary = {
        "aggregate_id": aggregate_id,
        "methods": [manifest["method"] for manifest in manifests],
        "fairness": {
            "weight_decay_semantics_match": weight_decay_match,
        },
        "old_results_used": False,
        "legacy_results_used": False,
    }
    write_json_strict(output_dir / "aggregate_manifest.json", aggregate_manifest, sort_keys=True)
    write_json_strict(output_dir / "summary.json", summary, sort_keys=True)
    return {
        "aggregate_manifest": aggregate_manifest,
        "summary": summary,
        "aggregate_manifest_path": output_dir / "aggregate_manifest.json",
        "summary_path": output_dir / "summary.json",
    }
