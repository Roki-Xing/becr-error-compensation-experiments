from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from experiment_provenance.aggregate import write_explicit_aggregate
from experiment_provenance.ids import config_hash
from experiment_provenance.noise import generate_noise_bank, hash_array
from experiment_provenance.schema import MANIFEST_SCHEMA_VERSION
from experiment_provenance.smoke import run_provenance_smoke


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_json_strict(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


def test_noise_bank_same_across_methods(tmp_path: Path):
    result = run_provenance_smoke(output_root=tmp_path, run_id="noise_same")
    manifests = result["manifests"]
    noise_hashes = {manifest["method"]: manifest["noise_hash"] for manifest in manifests}
    assert len(set(noise_hashes.values())) == 1
    consumed = {metrics["method"]: metrics["consumed_noise_hash"] for metrics in result["metrics"]}
    assert len(set(consumed.values())) == 1


def test_method_order_invariance(tmp_path: Path):
    order_ab = ["official_fira_carry", "full_residual_current_projection"]
    order_ba = list(reversed(order_ab))
    lhs = run_provenance_smoke(output_root=tmp_path, run_id="order_ab", methods=order_ab)
    rhs = run_provenance_smoke(output_root=tmp_path, run_id="order_ba", methods=order_ba)
    lhs_metrics = {row["method"]: row for row in lhs["metrics"]}
    rhs_metrics = {row["method"]: row for row in rhs["metrics"]}
    for method in order_ab:
        assert lhs_metrics[method]["consumed_noise_hash"] == rhs_metrics[method]["consumed_noise_hash"]
        assert lhs_metrics[method]["noise_hash"] == rhs_metrics[method]["noise_hash"]


def test_scheduler_factory_isolation(tmp_path: Path):
    result = run_provenance_smoke(output_root=tmp_path, run_id="scheduler_iso")
    manifests = result["manifests"]
    factory_ids = [manifest["scheduler_factory_id"] for manifest in manifests]
    object_ids = [manifest["scheduler_object_id"] for manifest in manifests]
    assert len(factory_ids) == len(set(factory_ids))
    assert len(object_ids) == len(set(object_ids))


def test_manifest_required_fields(tmp_path: Path):
    result = run_provenance_smoke(output_root=tmp_path, run_id="manifest_fields")
    manifest = result["manifests"][0]
    required = {
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
    assert required.issubset(manifest.keys())
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION


def test_config_hash_deterministic():
    cfg = {"seed": 0, "noise_std": 0.15, "methods": ["a", "b"], "rho": 0.5}
    lhs = config_hash(cfg)
    rhs = config_hash({"rho": 0.5, "methods": ["a", "b"], "noise_std": 0.15, "seed": 0})
    assert lhs == rhs


def test_noise_hash_deterministic():
    grads = [np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0])]
    lhs = generate_noise_bank(grads, std=0.15, rng_seed=7, dtype=float)
    rhs = generate_noise_bank(grads, std=0.15, rng_seed=7, dtype=float)
    assert hash_array(lhs) == hash_array(rhs)


def test_manifest_strict_json(tmp_path: Path):
    result = run_provenance_smoke(output_root=tmp_path, run_id="strict_json")
    for manifest_path in result["manifest_paths"]:
        _parse_json_strict(manifest_path)


def test_manifest_code_commit_matches_head_for_generated_runs(tmp_path: Path):
    result = run_provenance_smoke(output_root=tmp_path, run_id="head_match")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for manifest in result["manifests"]:
        assert manifest["code_commit"] == head


def test_dirty_worktree_rejected_for_paper_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from experiment_provenance import smoke as smoke_mod
    original = smoke_mod.get_git_metadata

    def _dirty_metadata(*args, **kwargs):
        metadata = original(*args, **kwargs)
        metadata["dirty"] = True
        return metadata

    monkeypatch.setattr(smoke_mod, "get_git_metadata", _dirty_metadata)
    with pytest.raises(RuntimeError):
        run_provenance_smoke(output_root=tmp_path, run_id="dirty_blocked", paper_quality=True)
    debug = run_provenance_smoke(
        output_root=tmp_path,
        run_id="dirty_debug",
        allow_dirty=True,
        overwrite_debug=True,
    )
    assert all(manifest["dirty"] is True for manifest in debug["manifests"])


def test_run_id_immutable_no_overwrite(tmp_path: Path):
    run_provenance_smoke(output_root=tmp_path, run_id="immutable")
    with pytest.raises(FileExistsError):
        run_provenance_smoke(output_root=tmp_path, run_id="immutable")
    rerun = run_provenance_smoke(output_root=tmp_path, run_id="immutable", overwrite_debug=True)
    assert rerun["paper_quality"] is False


def test_aggregate_requires_explicit_run_ids(tmp_path: Path):
    with pytest.raises(ValueError):
        write_explicit_aggregate(output_dir=tmp_path / "agg", run_manifest_paths=[])


def test_aggregate_rejects_old_schema(tmp_path: Path):
    legacy_dir = tmp_path / "legacy_diagnostic"
    legacy_dir.mkdir(parents=True)
    manifest_path = legacy_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": "legacy-v0"}), encoding="utf-8")
    with pytest.raises(ValueError):
        write_explicit_aggregate(output_dir=tmp_path / "agg", run_manifest_paths=[manifest_path])


def test_old_results_quarantined_by_default(tmp_path: Path):
    result = run_provenance_smoke(output_root=tmp_path, run_id="quarantine")
    aggregate = _read_json(result["aggregate_manifest_path"])
    assert aggregate["old_results_used"] is False
    assert aggregate["legacy_results_used"] is False
    assert all("legacy_diagnostic" not in path for path in aggregate["source_manifest_paths"])


def test_weight_decay_semantics_recorded(tmp_path: Path):
    result = run_provenance_smoke(output_root=tmp_path, run_id="weight_decay")
    aggregate = _read_json(result["summary_path"])
    for manifest in result["manifests"]:
        assert manifest["weight_decay_semantics"] == "decoupled_none"
    assert aggregate["fairness"]["weight_decay_semantics_match"] is True


def test_scale_logging_has_raw_applied_effective(tmp_path: Path):
    result = run_provenance_smoke(output_root=tmp_path, run_id="scale_schema")
    scale_path = result["scale_log_paths"][0]
    first = json.loads(scale_path.read_text(encoding="utf-8").splitlines()[0])
    required = {
        "phi_raw",
        "s_raw",
        "s_applied",
        "rho",
        "tau",
        "Z_norm",
        "Z_eff_norm",
        "effective_raw_transmission",
        "residual_update_error",
        "limiter_active",
        "raw_recovery_scale",
        "applied_recovery_scale",
        "effective_recovery_scale",
    }
    assert required.issubset(first.keys())


def test_memory_schema_records_residual_bytes(tmp_path: Path):
    result = run_provenance_smoke(output_root=tmp_path, run_id="memory_schema")
    memory = _read_json(result["memory_paths"][0])
    assert "residual_bytes" in memory
    assert "first_moment_bytes" in memory
    assert "second_moment_bytes" in memory
    assert "projection_bytes" in memory
    assert "peak_memory_allocated" in memory


def test_no_neural_training_in_ci():
    workflow = Path(".github/workflows/python-tests.yml").read_text(encoding="utf-8")
    assert "run_provenance_smoke.py" in workflow
    forbidden = [
        "run_cifar10_coordproj_mechanism.py",
        "run_mnist_mlp_coordproj_mechanism.py",
        "run_wikitext2_transformer_coordproj_mechanism.py",
    ]
    for token in forbidden:
        assert token not in workflow


def test_all_pr1_pr2_gates_still_present():
    required = [
        Path("fira_parity"),
        Path("third_party/fira_oracle"),
        Path("tests/test_fira_parity.py"),
        Path("tests/test_labeling.py"),
        Path("tests/test_moving_projection_state.py"),
        Path(".github/workflows/fira-parity.yml"),
        Path(".github/workflows/python-tests.yml"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    assert not missing, f"missing prerequisite gates: {missing}"
