from __future__ import annotations

import json
from pathlib import Path

import pytest

from corrected_synthetic.runner import run_corrected_synthetic_suite, theorem_condition_report


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_strict(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


def _summary_by_name(result: dict, filename: str) -> dict:
    for path in result["experiment_summary_paths"]:
        if Path(path).name == filename:
            return _read_json(Path(path))
    raise AssertionError(f"missing summary {filename}")


@pytest.fixture(scope="module")
def full_tiny_suite(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("p1_corrected_synthetic_full_tiny")
    return run_corrected_synthetic_suite(
        output_root=root,
        run_id="full_tiny_suite",
        paper_quality=False,
        tiny=True,
    )


def test_theorem_regime_conditions_logged():
    report = theorem_condition_report(a=1.0, b=1.0, lr=0.5, eps_a=1.0, eps_s=1.0, x0=1.0)
    assert report["eta_a"] == 0.5
    assert report["two_eps_a"] == 2.0
    assert report["M0"] == 0.25
    assert report["eta_b_M0"] == 0.125
    assert report["ok_eta_a_lt_2epsa"] is True
    assert report["ok_eta_b_M0_le_half"] is True


def test_corrected_synthetic_suite_writes_strict_json_and_explicit_aggregates(full_tiny_suite: dict):
    for manifest_path in full_tiny_suite["manifest_paths"]:
        _parse_strict(Path(manifest_path))
    for aggregate_path in full_tiny_suite["aggregate_manifest_paths"]:
        aggregate = _read_json(Path(aggregate_path))
        assert aggregate["source_run_ids"]
        assert aggregate["source_manifest_paths"]
        assert aggregate["old_results_used"] is False
        assert aggregate["legacy_results_used"] is False


def test_theorem_regime_numerical_outcomes_tiny(full_tiny_suite: dict):
    summary = _summary_by_name(full_tiny_suite, "theorem_summary.json")
    theorem = summary["theorem_conditions"]
    raw = summary["methods"]["fira_raw"]
    clipped = summary["methods"]["fira_clipped"]
    becr = summary["methods"]["becr"]
    no_lower = summary["methods"]["rho_no_lower_bound"]
    coupled = summary["methods"]["coupled_unbounded"]
    claims = summary["claim_summary"]

    assert theorem["ok_eta_a_lt_2epsa"] is True
    assert theorem["ok_eta_b_M0_le_half"] is True
    assert raw["grad_norm_final"] > 0.1
    assert abs(raw["final_y"]) > 0.1
    assert 0.0 < raw["phi_raw_cum_final"] < 10.0
    assert clipped["grad_norm_final"] < raw["grad_norm_final"] * 1e-2
    assert becr["grad_norm_final"] < raw["grad_norm_final"] * 1e-2
    assert (becr["residual_norm_final"] or 0.0) < 1e-2
    assert no_lower["grad_norm_final"] > becr["grad_norm_final"] * 100.0
    assert coupled["grad_norm_final"] > becr["grad_norm_final"] * 100.0
    assert no_lower["residual_norm_final"] > 1.0
    assert coupled["residual_norm_final"] > 1.0
    assert claims["raw_nonzero_stationary"] is True
    assert claims["clipped_repairs_relative_to_raw"] is True
    assert claims["becr_repairs_relative_to_raw"] is True
    assert claims["no_lower_worse_than_becr"] is True
    assert claims["coupled_worse_than_becr"] is True


def test_fixed_stale_tiny_repair_ordering(full_tiny_suite: dict):
    summary = _summary_by_name(full_tiny_suite, "high_dimensional_summary.json")
    raw = summary["methods"]["fira_raw"]
    clipped = summary["methods"]["fira_clipped"]
    becr = summary["methods"]["becr"]
    claims = summary["claim_summary"]

    assert raw["grad_perp_norm_mean"] > clipped["grad_perp_norm_mean"]
    assert raw["grad_perp_norm_mean"] > becr["grad_perp_norm_mean"]
    assert clipped["grad_perp_norm_mean"] < raw["grad_perp_norm_mean"] * 1e-3
    assert becr["grad_perp_norm_mean"] < raw["grad_perp_norm_mean"] * 1e-3
    assert claims["clipped_reduces_orthogonal_stale_channel"] is True
    assert claims["becr_reduces_orthogonal_stale_channel"] is True
    assert claims["full_stationarity_claim_supported"] is False
    assert "orthogonal stale channel" in claims["statement"]


def test_noisy_coordinate_method_order_invariance(tmp_path: Path):
    lhs = run_corrected_synthetic_suite(
        output_root=tmp_path,
        run_id="order_ab",
        paper_quality=False,
        experiments=["anisotropic_noise"],
        tiny=True,
        method_order=["proj_baseline", "fira_raw", "fira_clipped", "becr"],
    )
    rhs = run_corrected_synthetic_suite(
        output_root=tmp_path,
        run_id="order_ba",
        paper_quality=False,
        experiments=["anisotropic_noise"],
        tiny=True,
        method_order=["becr", "fira_clipped", "fira_raw", "proj_baseline"],
    )
    lhs_map = {row["method"]: row for row in lhs["method_order_report"]["anisotropic_noise"]}
    rhs_map = {row["method"]: row for row in rhs["method_order_report"]["anisotropic_noise"]}
    for method in lhs_map:
        assert lhs_map[method]["noise_hash"] == rhs_map[method]["noise_hash"]
        assert lhs_map[method]["consumed_noise_hash"] == rhs_map[method]["consumed_noise_hash"]


def test_anisotropic_summary_records_paired_differences_and_baseline_nuance(full_tiny_suite: dict):
    summary = _summary_by_name(full_tiny_suite, "anisotropic_noise_summary.json")
    claims = summary["claim_summary"]
    assert summary["seed_count"] >= 2
    assert "becr_minus_clipped" in summary["paired_differences"]
    assert "becr_minus_raw" in summary["paired_differences"]
    assert "becr_minus_proj_baseline" in summary["paired_differences"]
    assert isinstance(claims["becr_beats_clipped_on_grad_norm_mean"], bool)
    assert isinstance(claims["becr_beats_raw_on_grad_norm_mean"], bool)
    assert isinstance(claims["becr_beats_projected_baseline_on_grad_norm_mean"], bool)
    assert "projected baseline" in claims["statement"]


def test_artifact_index_lists_all_review_files(full_tiny_suite: dict):
    artifact_index = _read_json(Path(full_tiny_suite["artifact_index_path"]))
    required = [
        "aggregate_manifest_paths",
        "aggregate_summary_paths",
        "experiment_summary_paths",
        "figure_paths",
        "figure_metadata_paths",
        "table_paths",
        "memory_runtime_summary_path",
        "scale_log_summary_path",
        "run_manifest_index_path",
        "exact_fira_fixture_summary_path",
        "ldadam_status_path",
    ]
    for key in required:
        assert key in artifact_index
        paths = artifact_index[key] if isinstance(artifact_index[key], list) else [artifact_index[key]]
        assert paths
        for path in paths:
            assert Path(path).exists()
    assert artifact_index["old_results_used"] is False
    assert artifact_index["legacy_results_used"] is False
    _parse_strict(Path(full_tiny_suite["artifact_index_path"]))


def test_all_figure_metadata_sidecars_have_sources(full_tiny_suite: dict):
    for meta_path in full_tiny_suite["plot_metadata_paths"]:
        meta = _read_json(Path(meta_path))
        assert meta["aggregate_id"]
        assert meta["source_run_ids"]
        assert meta["source_manifest_paths"]
        assert meta["aggregate_manifest_path"]
        assert meta["code_commit"]
        assert meta["config_hashes"] is not None
        assert meta["noise_hashes"] is not None
        assert meta["candidate_level"]
        assert meta["figure_candidate_level"]
        assert meta["main_or_appendix_recommendation"]


def test_plots_record_source_run_ids(full_tiny_suite: dict):
    plot_meta = _read_json(Path(full_tiny_suite["plot_metadata_paths"][0]))
    assert plot_meta["source_run_ids"]
    assert plot_meta["source_manifest_paths"]
    assert plot_meta["aggregate_manifest_path"]


def test_seed_count_recorded_in_aggregate_summary(full_tiny_suite: dict):
    summary = _summary_by_name(full_tiny_suite, "anisotropic_noise_summary.json")
    assert summary["seed_count"] >= 2
    assert summary["aggregate_id"]


def test_memory_and_scale_schema_present(full_tiny_suite: dict):
    memory = _read_json(Path(full_tiny_suite["memory_paths"][0]))
    assert "residual_bytes" in memory
    scale_first = json.loads(Path(full_tiny_suite["scale_log_paths"][0]).read_text(encoding="utf-8").splitlines()[0])
    for key in [
        "phi_raw",
        "s_raw",
        "s_applied",
        "rho",
        "tau",
        "effective_recovery_scale",
    ]:
        assert key in scale_first


def test_no_legacy_all_runs_json_reference_in_p1_runner():
    paths = [
        Path("experiments/tier1-synthetic/run_corrected_synthetic_suite.py"),
        Path("corrected_synthetic/runner.py"),
        Path("corrected_synthetic/plots.py"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "ALL_RUNS.json" not in text


def test_ci_synthetic_tiny_command_has_no_neural_entrypoints():
    text = Path(".github/workflows/python-tests.yml").read_text(encoding="utf-8")
    assert "run_corrected_synthetic_suite.py" in text
    assert "--tiny" in text
    assert "--paper-quality" in text
    forbidden = [
        "run_cifar10_coordproj_mechanism.py",
        "run_mnist_mlp_coordproj_mechanism.py",
        "run_wikitext2_transformer_coordproj_mechanism.py",
        "TinyGPT",
    ]
    for token in forbidden:
        assert token not in text


def test_no_neural_entrypoints_called_in_p1_runner():
    text = Path("experiments/tier1-synthetic/run_corrected_synthetic_suite.py").read_text(encoding="utf-8")
    forbidden = [
        "run_cifar10_coordproj_mechanism.py",
        "run_mnist_mlp_coordproj_mechanism.py",
        "run_wikitext2_transformer_coordproj_mechanism.py",
        "TinyGPT",
    ]
    for token in forbidden:
        assert token not in text


def test_artifact_snapshot_or_ci_upload_policy_documented():
    workflow = Path(".github/workflows/python-tests.yml").read_text(encoding="utf-8")
    readme_path = Path("experiments/tier1-synthetic/corrected_synthetic_artifacts/README.md")
    readme = readme_path.read_text(encoding="utf-8")
    assert "upload-artifact@v4" in workflow
    assert "corrected-synthetic-tiny-review" in workflow
    assert readme_path.exists()
    assert "reviewer-convenience snapshots" in readme
    assert "source of truth" in readme
