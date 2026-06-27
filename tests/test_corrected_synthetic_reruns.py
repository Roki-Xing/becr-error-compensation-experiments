from __future__ import annotations

import json
from pathlib import Path

from corrected_synthetic.runner import (
    theorem_condition_report,
    run_corrected_synthetic_suite,
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_strict(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


def test_theorem_regime_conditions_logged():
    report = theorem_condition_report(a=1.0, b=1.0, lr=0.5, eps_a=1.0, eps_s=1.0, x0=1.0)
    assert report["eta_a"] == 0.5
    assert report["two_eps_a"] == 2.0
    assert report["M0"] == 0.25
    assert report["eta_b_M0"] == 0.125
    assert report["ok_eta_a_lt_2epsa"] is True
    assert report["ok_eta_b_M0_le_half"] is True


def test_corrected_synthetic_suite_writes_strict_json_and_explicit_aggregates(tmp_path: Path):
    result = run_corrected_synthetic_suite(
        output_root=tmp_path,
        run_id="tiny_suite",
        paper_quality=False,
        experiments=["theorem_regime", "high_dimensional_fixed"],
        tiny=True,
    )
    for manifest_path in result["manifest_paths"]:
        _parse_strict(manifest_path)
    for aggregate_path in result["aggregate_manifest_paths"]:
        aggregate = _read_json(aggregate_path)
        assert aggregate["source_run_ids"]
        assert aggregate["source_manifest_paths"]
        assert aggregate["old_results_used"] is False
        assert aggregate["legacy_results_used"] is False


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


def test_plots_record_source_run_ids(tmp_path: Path):
    result = run_corrected_synthetic_suite(
        output_root=tmp_path,
        run_id="plot_meta",
        paper_quality=False,
        experiments=["theorem_regime"],
        tiny=True,
    )
    plot_meta = _read_json(result["plot_metadata_paths"][0])
    assert plot_meta["source_run_ids"]
    assert plot_meta["source_manifest_paths"]
    assert plot_meta["aggregate_manifest_path"]


def test_seed_count_recorded_in_aggregate_summary(tmp_path: Path):
    result = run_corrected_synthetic_suite(
        output_root=tmp_path,
        run_id="seed_count",
        paper_quality=False,
        experiments=["anisotropic_noise"],
        tiny=True,
    )
    summary = _read_json(result["experiment_summary_paths"][0])
    assert summary["seed_count"] >= 2
    assert summary["aggregate_id"]


def test_memory_and_scale_schema_present(tmp_path: Path):
    result = run_corrected_synthetic_suite(
        output_root=tmp_path,
        run_id="schema_check",
        paper_quality=False,
        experiments=["theorem_regime"],
        tiny=True,
    )
    memory = _read_json(result["memory_paths"][0])
    assert "residual_bytes" in memory
    scale_first = json.loads(result["scale_log_paths"][0].read_text(encoding="utf-8").splitlines()[0])
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
