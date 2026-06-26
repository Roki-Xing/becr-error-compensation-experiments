from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import torch

from fira_parity.compare import compare_traces
from fira_parity.fixtures import parity_fixtures
from fira_parity.runner import run_fixture
from fira_parity.upstream_metadata import UPSTREAM_COMMIT, UPSTREAM_LICENSE, UPSTREAM_REPOSITORY


REPO_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=None)
def _result(fixture_id: str):
    return run_fixture(fixture_id)


def _tol(dtype: str) -> tuple[float, float]:
    return (1e-10, 1e-10) if dtype == "float64" else (1e-5, 1e-5)


def test_official_fira_oracle_is_pinned():
    upstream = REPO_ROOT / "third_party" / "fira_oracle" / "UPSTREAM.md"
    license_path = REPO_ROOT / "third_party" / "fira_oracle" / "LICENSE"
    assert upstream.exists()
    assert license_path.exists()
    text = upstream.read_text(encoding="utf-8")
    assert UPSTREAM_REPOSITORY in text
    assert UPSTREAM_COMMIT in text
    assert UPSTREAM_LICENSE in text
    assert "Apache License" in license_path.read_text(encoding="utf-8")


def test_exact_fira_refresh_schedule_parity():
    for fixture_id in (
        "shape_2x1_rank1_gap1_fp64",
        "shape_2x2_rank1_gap2_fp64",
        "shape_4x3_rank2_gap5_fp64_100step",
    ):
        result = _result(fixture_id)
        config = result["config"]
        refresh_steps = [t["step_index"] for t in result["oracle_traces"] if t["refresh_happened"]]
        expected = list(range(1, config.steps + 1, config.update_proj_gap))
        assert refresh_steps == expected
        assert refresh_steps == [t["step_index"] for t in result["candidate_traces"] if t["refresh_happened"]]


def test_exact_fira_projection_parity():
    for fixture_id in ("shape_2x2_rank1_gap2_fp64", "shape_3x4_rank2_gap2_fp64_left_orientation"):
        result = _result(fixture_id)
        errors = result["comparison"]["max_errors"]
        atol, rtol = _tol(result["config"].dtype)
        assert errors["invariant_projector_after"]["abs"] <= atol
        assert errors["reconstructed_projected_gradient"]["abs"] <= atol
        assert errors["projected_gradient"]["abs"] <= atol
        assert errors["projected_gradient"]["rel"] <= rtol


def test_exact_fira_moment_parity():
    for fixture_id in ("shape_2x2_rank1_gap2_fp64", "shape_2x2_rank1_gap2_beta0"):
        result = _result(fixture_id)
        errors = result["comparison"]["max_errors"]
        atol, rtol = _tol(result["config"].dtype)
        for field in ("exp_avg_after", "exp_avg_sq_after", "bias_correction1", "bias_correction2", "normalized_low_rank_update"):
            assert errors[field]["abs"] <= atol
            assert errors[field]["rel"] <= rtol


def test_exact_fira_column_scale_parity():
    result = _result("shape_3x4_rank2_gap2_fp64_left_orientation")
    errors = result["comparison"]["max_errors"]
    atol, rtol = _tol(result["config"].dtype)
    for field in ("raw_recovery_scale", "applied_recovery_scale", "candidate_recovery_update"):
        assert errors[field]["abs"] <= atol
        assert errors[field]["rel"] <= rtol


def test_exact_fira_limiter_inactive_parity():
    result = _result("shape_2x1_rank1_gap1_fp64")
    assert not any(t["limiter_triggered"] for t in result["oracle_traces"])
    assert not any(t["limiter_triggered"] for t in result["candidate_traces"])
    assert max(t["limiter_factor"] for t in result["oracle_traces"]) == 1.0


def test_exact_fira_limiter_active_parity():
    result = _result("shape_4x3_rank1_gap2_fp64_limiter_active")
    errors = result["comparison"]["max_errors"]
    atol, rtol = _tol(result["config"].dtype)
    assert any(t["limiter_triggered"] for t in result["oracle_traces"])
    assert any(t["limiter_triggered"] for t in result["candidate_traces"])
    assert result["comparison"]["first_mismatch"] is None
    assert errors["limiter_factor"]["abs"] <= atol
    assert errors["limiter_factor"]["rel"] <= rtol


def test_exact_fira_weight_decay_parity():
    result = _result("shape_2x2_rank2_gap5_fp32_wd")
    errors = result["comparison"]["max_errors"]
    atol, rtol = _tol(result["config"].dtype)
    assert result["comparison"]["first_mismatch"] is None
    assert any(float(torch.norm(t["weight_decay_contribution"]).item()) > 0 for t in result["oracle_traces"])
    assert errors["weight_decay_contribution"]["abs"] <= atol
    assert errors["weight_decay_contribution"]["rel"] <= rtol


def test_exact_fira_fp32_active_fixture_has_no_semantic_mismatch():
    result = _result("shape_4x3_rank2_gap5_fp32_wd_limiter_active")
    assert result["comparison"]["first_mismatch"] is None


def test_exact_fira_state_survives_refresh():
    result = _result("shape_2x2_rank1_gap2_fp64")
    traces = result["oracle_traces"]
    assert traces[1]["exp_avg_after"] is not None
    assert traces[2]["refresh_happened"] is True
    assert torch.allclose(traces[2]["exp_avg_before"], traces[1]["exp_avg_after"])
    assert torch.allclose(traces[2]["exp_avg_sq_before"], traces[1]["exp_avg_sq_after"])
    assert traces[2]["step_index"] == 3


def test_exact_fira_100_step_trace_parity():
    result = _result("shape_4x3_rank2_gap5_fp64_100step")
    errors = result["comparison"]["max_errors"]
    atol, rtol = _tol(result["config"].dtype)
    assert result["comparison"]["first_mismatch"] is None
    assert sum(t["refresh_happened"] for t in result["oracle_traces"]) >= 2
    for field, stats in errors.items():
        assert stats["abs"] <= atol, field
        assert stats["rel"] <= rtol, field


def test_parity_trace_contains_no_nan_or_inf():
    result = _result("shape_4x3_rank2_gap5_fp64_100step")
    for trace in result["oracle_traces"] + result["candidate_traces"]:
        assert trace["finite"] is True


def test_svd_sign_alignment_does_not_create_false_failure():
    result = _result("shape_2x2_rank1_gap2_fp64")
    oracle = result["oracle_traces"]
    candidate = deepcopy(result["candidate_traces"])
    for trace in candidate:
        if trace["projection_basis_after"] is not None:
            trace["projection_basis_after"] = -trace["projection_basis_after"]
        if trace["projection_basis_before"] is not None:
            trace["projection_basis_before"] = -trace["projection_basis_before"]
        for field in (
            "projected_gradient",
            "exp_avg_before",
            "exp_avg_after",
            "exp_avg_sq_before",
            "exp_avg_sq_after",
            "normalized_low_rank_update",
        ):
            if trace[field] is not None:
                trace[field] = -trace[field]
    atol, rtol = _tol(result["config"].dtype)
    comparison = compare_traces(oracle, candidate, atol=atol, rtol=rtol)
    assert comparison["first_mismatch"] is None
