from __future__ import annotations

import json
from pathlib import Path

import pytest

from corrected_synthetic.runner import run_corrected_synthetic_suite


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
def adam_limiter_tiny(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("p1_adam_limiter_tiny")
    return run_corrected_synthetic_suite(
        output_root=root,
        run_id="adam_limiter_tiny",
        paper_quality=False,
        tiny=True,
        experiments=["adam_limiter_loss_diagnostic"],
    )


def test_adam_limiter_activates_in_deterministic_case(adam_limiter_tiny: dict):
    summary = _summary_by_name(adam_limiter_tiny, "adam_limiter_loss_summary.json")
    assert summary["methods"]["becr_effective_signal_residual"]["limiter_activation_count"] > 0


def test_effective_signal_residual_conservation_fp64(adam_limiter_tiny: dict):
    summary = _summary_by_name(adam_limiter_tiny, "adam_limiter_loss_summary.json")
    assert summary["methods"]["becr_effective_signal_residual"]["residual_update_error_max"] <= 1e-10


def test_wrong_residual_differs_when_limiter_active(adam_limiter_tiny: dict):
    summary = _summary_by_name(adam_limiter_tiny, "adam_limiter_loss_summary.json")
    wrong = summary["methods"]["wrong_pre_limiter_residual"]
    correct = summary["methods"]["becr_effective_signal_residual"]
    assert wrong["wrong_vs_correct_residual_gap"] > 1e-8
    assert wrong["final_residual_norm"] != pytest.approx(correct["final_residual_norm"])


def test_appendix_recommendation_and_primary_ordering(adam_limiter_tiny: dict):
    summary = _summary_by_name(adam_limiter_tiny, "adam_limiter_loss_summary.json")
    claim_summary = summary["claim_summary"]
    methods = summary["methods"]
    becr = methods["becr_effective_signal_residual"]
    no_residual = methods["fira_style_adam_limiter_no_residual"]
    clipping = methods["clipping_only_limiter"]
    wrong = methods["wrong_pre_limiter_residual"]

    assert claim_summary["include_recommendation"] == "appendix"
    assert claim_summary["primary_setting_id"] == "beta1_0.9_beta2_0.99_gamma_active"
    assert becr["grad_norm_final"] < no_residual["grad_norm_final"]
    assert becr["grad_norm_final"] < clipping["grad_norm_final"]
    assert becr["grad_norm_final"] < wrong["grad_norm_final"]


def test_no_neural_entrypoints_in_adam_limiter_diagnostic():
    runner_text = Path("corrected_synthetic/runner.py").read_text(encoding="utf-8")
    workflow_text = Path(".github/workflows/python-tests.yml").read_text(encoding="utf-8")
    forbidden = [
        "run_cifar10_coordproj_mechanism.py",
        "run_mnist_mlp_coordproj_mechanism.py",
        "run_wikitext2_transformer_coordproj_mechanism.py",
        "TinyGPT",
    ]
    for token in forbidden:
        assert token not in runner_text
        assert token not in workflow_text


def test_no_old_all_runs_json_used():
    for path in [
        Path("corrected_synthetic/runner.py"),
        Path("corrected_synthetic/core.py"),
        Path("experiments/tier1-synthetic/run_corrected_synthetic_suite.py"),
    ]:
        assert "ALL_RUNS.json" not in path.read_text(encoding="utf-8")


def test_adam_limiter_artifact_index_or_summary_exists(adam_limiter_tiny: dict):
    artifact_index = _read_json(Path(adam_limiter_tiny["artifact_index_path"]))
    assert Path(adam_limiter_tiny["artifact_index_path"]).exists()
    assert any("adam_limiter_loss_summary.json" in path for path in artifact_index["experiment_summary_paths"])
    _parse_strict(Path(adam_limiter_tiny["artifact_index_path"]))


def test_claims_remain_diagnostic_only():
    text = Path("paper/sections/07_corrected_synthetic_diagnostics.tex").read_text(encoding="utf-8")
    assert "Appendix-level limiter-active diagnostics" in text
    forbidden = [
        "Adam convergence",
        "optimizer superiority",
        "beats AdamW",
        "beats LDAdam",
    ]
    for token in forbidden:
        assert token not in text
