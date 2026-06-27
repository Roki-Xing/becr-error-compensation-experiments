from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from moving_projection_state.core import (
    REQUIRED_MANIFEST_FIELDS,
    MovingProjectionConfig,
    make_basis,
    run_method_suite,
    run_single_step,
    run_trace,
)
from moving_projection_state.runner import generate_artifacts


def test_residual_enters_new_subspace():
    cfg = MovingProjectionConfig(dtype=np.float64)
    prev_basis = make_basis(2, [0], dtype=np.float64)
    new_basis = make_basis(2, [1], dtype=np.float64)
    result = run_single_step(
        mode="full_residual_current_projection",
        cfg=cfg,
        g=np.zeros((2,), dtype=np.float64),
        P=new_basis,
        prev_P=prev_basis,
        initial_state={
            "e": np.asarray([0.0, 3.0], dtype=np.float64),
            "m": np.zeros((1,), dtype=np.float64),
            "v": np.zeros((1,), dtype=np.float64),
            "t": 0,
        },
        step_index=0,
    )
    assert np.allclose(result["R"], np.asarray([3.0], dtype=np.float64))
    assert np.allclose(result["Q"], np.zeros((2,), dtype=np.float64))
    assert np.allclose(result["A"], np.asarray([0.0, 3.0], dtype=np.float64))


def test_residual_remains_orthogonal():
    cfg = MovingProjectionConfig(dtype=np.float64)
    prev_basis = make_basis(2, [0], dtype=np.float64)
    new_basis = make_basis(2, [1], dtype=np.float64)
    result = run_single_step(
        mode="full_residual_current_projection",
        cfg=cfg,
        g=np.zeros((2,), dtype=np.float64),
        P=new_basis,
        prev_P=prev_basis,
        initial_state={
            "e": np.asarray([5.0, 0.0], dtype=np.float64),
            "m": np.zeros((1,), dtype=np.float64),
            "v": np.zeros((1,), dtype=np.float64),
            "t": 0,
        },
        step_index=0,
    )
    assert np.allclose(result["R"], np.asarray([0.0], dtype=np.float64))
    assert np.allclose(result["Q"], np.asarray([5.0, 0.0], dtype=np.float64))


def test_90_degree_rotation_preserves_decomposition_invariant():
    cfg = MovingProjectionConfig(dtype=np.float64)
    prev_basis = make_basis(2, [0], dtype=np.float64)
    new_basis = make_basis(2, [1], dtype=np.float64)
    result = run_single_step(
        mode="full_residual_current_projection",
        cfg=cfg,
        g=np.asarray([2.0, -1.0], dtype=np.float64),
        P=new_basis,
        prev_P=prev_basis,
        initial_state={
            "e": np.asarray([0.75, 1.25], dtype=np.float64),
            "m": np.zeros((1,), dtype=np.float64),
            "v": np.zeros((1,), dtype=np.float64),
            "t": 0,
        },
        step_index=0,
    )
    assert result["decomposition_error"] <= 1e-10


def test_no_silent_reset_outside_explicit_reset_mode():
    bases = [
        make_basis(2, [0], dtype=np.float64),
        make_basis(2, [1], dtype=np.float64),
    ]
    grads = [
        np.asarray([2.0, 1.0], dtype=np.float64),
        np.asarray([1.0, -3.0], dtype=np.float64),
    ]
    cfg = MovingProjectionConfig(dtype=np.float64, rho=0.5)
    trace = run_trace(mode="full_residual_current_projection", cfg=cfg, gradients=grads, bases=bases)
    refresh_trace = trace["steps"][1]
    assert refresh_trace["refresh_happened"] is True
    assert refresh_trace["reset_event"] is False
    assert refresh_trace["state_before_step"]["t"] == 1
    assert trace["steps"][0]["state_after_step"]["e_norm"] > 0.0
    assert refresh_trace["state_before_step"]["e_norm"] == trace["steps"][0]["state_after_step"]["e_norm"]


def test_limiter_uses_effective_transmitted_signal_for_residual_update():
    cfg = MovingProjectionConfig(dtype=np.float64, rho=1.0, limiter_gamma=0.5)
    P = make_basis(2, [0], dtype=np.float64)
    result = run_single_step(
        mode="full_residual_current_projection",
        cfg=cfg,
        g=np.asarray([0.0, 4.0], dtype=np.float64),
        P=P,
        prev_P=P,
        initial_state={
            "e": np.zeros((2,), dtype=np.float64),
            "m": np.zeros((1,), dtype=np.float64),
            "v": np.zeros((1,), dtype=np.float64),
            "t": 0,
            "prev_u_perp_norm": 1.0,
        },
        step_index=0,
    )
    expected = result["Q"] - result["tau"] * result["Z"]
    assert np.allclose(result["E_next"], expected)
    assert result["residual_error"] <= 1e-10


def test_scheduler_isolation():
    bases = [
        make_basis(4, [0, 1], dtype=np.float64),
        make_basis(4, [2, 3], dtype=np.float64),
        make_basis(4, [0, 1], dtype=np.float64),
    ]
    grads = [np.ones((4,), dtype=np.float64) for _ in range(3)]
    suite = run_method_suite(
        modes=["state_reset_explicit", "official_fira_carry"],
        cfg=MovingProjectionConfig(dtype=np.float64),
        gradients=grads,
        bases=bases,
    )
    assert suite["state_reset_explicit"]["manifest"]["scheduler_factory_id"] != suite["official_fira_carry"]["manifest"]["scheduler_factory_id"]
    assert suite["state_reset_explicit"]["manifest"]["scheduler_object_id"] != suite["official_fira_carry"]["manifest"]["scheduler_object_id"]


def test_crn_smoke_uses_identical_pre_generated_noise():
    bases = [make_basis(3, [0], dtype=np.float64) for _ in range(4)]
    grads = [np.asarray([1.0, 0.0, 0.0], dtype=np.float64) for _ in range(4)]
    suite = run_method_suite(
        modes=["official_fira_carry", "full_residual_current_projection"],
        cfg=MovingProjectionConfig(dtype=np.float64),
        gradients=grads,
        bases=bases,
        stochastic_noise_std=0.2,
        rng_seed=7,
    )
    lhs = suite["official_fira_carry"]["manifest"]
    rhs = suite["full_residual_current_projection"]["manifest"]
    assert lhs["noise_hash"] == rhs["noise_hash"]
    assert lhs["rng_seed"] == rhs["rng_seed"] == 7
    assert np.allclose(
        np.asarray(suite["official_fira_carry"]["noise_bank"], dtype=np.float64),
        np.asarray(suite["full_residual_current_projection"]["noise_bank"], dtype=np.float64),
    )


def test_manifest_schema_and_artifact_generation(tmp_path: Path):
    out_dir = tmp_path / "artifacts"
    artifact_dir = generate_artifacts(output_root=out_dir, parent_task_id="P0-MOVING-PROJECTION-STATE-002")
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert REQUIRED_MANIFEST_FIELDS.issubset(manifest.keys())
    assert (artifact_dir / "sample_trace.json").exists()
    assert (artifact_dir / "refresh_alignment.png").exists()


def test_fp64_trace_invariants():
    bases = [
        make_basis(2, [0], dtype=np.float64),
        make_basis(2, [1], dtype=np.float64),
        make_basis(2, [0], dtype=np.float64),
    ]
    grads = [
        np.asarray([2.0, 1.0], dtype=np.float64),
        np.asarray([-1.0, 3.0], dtype=np.float64),
        np.asarray([0.5, -0.5], dtype=np.float64),
    ]
    trace = run_trace(
        mode="full_residual_current_projection",
        cfg=MovingProjectionConfig(dtype=np.float64),
        gradients=grads,
        bases=bases,
    )
    assert trace["max_decomposition_error"] <= 1e-10
    assert trace["max_residual_error"] <= 1e-10


def test_first_moment_transport_formula_exact():
    cfg = MovingProjectionConfig(dtype=np.float64)

    P_old = make_basis(2, [0], dtype=np.float64)
    P_new = make_basis(2, [1], dtype=np.float64)
    result = run_single_step(
        mode="projection_aware_transport",
        cfg=cfg,
        g=np.zeros((2,), dtype=np.float64),
        P=P_new,
        prev_P=P_old,
        initial_state={
            "m": np.asarray([3.5], dtype=np.float64),
            "v": np.asarray([2.0], dtype=np.float64),
            "t": 4,
            "e": np.zeros((2,), dtype=np.float64),
        },
        step_index=0,
    )
    assert np.linalg.norm(np.asarray(result["state_before_step"]["m"], dtype=np.float64)) <= 1e-10

    P_old = make_basis(2, [0], dtype=np.float64)
    P_new = np.asarray([[1.0], [1.0]], dtype=np.float64) / np.sqrt(2.0)
    m_old = np.asarray([2.25], dtype=np.float64)
    expected = P_new.T @ P_old @ m_old
    result = run_single_step(
        mode="projection_aware_transport",
        cfg=cfg,
        g=np.zeros((2,), dtype=np.float64),
        P=P_new,
        prev_P=P_old,
        initial_state={
            "m": m_old,
            "v": np.asarray([1.0], dtype=np.float64),
            "t": 2,
            "e": np.zeros((2,), dtype=np.float64),
        },
        step_index=0,
    )
    transported = np.asarray(result["state_before_step"]["m"], dtype=np.float64)
    assert np.linalg.norm(transported - expected) <= 1e-10


def test_second_moment_transport_mode_explicit():
    bases = [make_basis(2, [0], dtype=np.float64), make_basis(2, [1], dtype=np.float64)]
    grads = [np.ones((2,), dtype=np.float64) for _ in range(2)]
    suite = run_method_suite(
        modes=["state_reset_explicit", "official_fira_carry", "projection_aware_transport", "full_residual_current_projection"],
        cfg=MovingProjectionConfig(dtype=np.float64),
        gradients=grads,
        bases=bases,
        parent_pr="#2",
    )
    assert suite["state_reset_explicit"]["manifest"]["second_moment_mode"] == "reset_on_refresh"
    assert suite["official_fira_carry"]["manifest"]["second_moment_mode"] == "carry_unchanged"
    assert suite["projection_aware_transport"]["manifest"]["second_moment_mode"] == "squared_basis_overlap_transport"
    assert suite["full_residual_current_projection"]["manifest"]["second_moment_mode"] == "squared_basis_overlap_transport"


def test_second_moment_transport_formula_exact():
    cfg = MovingProjectionConfig(dtype=np.float64)
    P_old = make_basis(2, [0], dtype=np.float64)
    P_new = np.asarray([[1.0], [1.0]], dtype=np.float64) / np.sqrt(2.0)
    v_old = np.asarray([5.0], dtype=np.float64)
    overlap = P_new.T @ P_old
    expected = (overlap * overlap) @ v_old
    result = run_single_step(
        mode="projection_aware_transport",
        cfg=cfg,
        g=np.zeros((2,), dtype=np.float64),
        P=P_new,
        prev_P=P_old,
        initial_state={
            "m": np.asarray([1.0], dtype=np.float64),
            "v": v_old,
            "t": 3,
            "e": np.zeros((2,), dtype=np.float64),
        },
        step_index=0,
    )
    transported = np.asarray(result["state_before_step"]["v"], dtype=np.float64)
    assert np.linalg.norm(transported - expected) <= 1e-10


def test_manifest_contains_parent_pr_and_second_moment_mode(tmp_path: Path):
    artifact_dir = generate_artifacts(output_root=tmp_path, parent_task_id="P0-MOVING-PROJECTION-STATE-002", parent_pr="#2")
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parent_pr"] == "#2"
    assert manifest["second_moment_mode"] == "squared_basis_overlap_transport"


def test_sample_trace_is_strict_json():
    path = Path("experiments/tier1-synthetic/moving_projection_artifacts/20260627T000000Z_p0_moving_projection_state/sample_trace.json")
    text = path.read_text(encoding="utf-8")
    json.loads(text, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    assert "NaN" not in text
    assert "Infinity" not in text


def test_generated_manifest_commit_matches_head(tmp_path: Path):
    artifact_dir = generate_artifacts(output_root=tmp_path, parent_task_id="P0-MOVING-PROJECTION-STATE-002", parent_pr="#2")
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert manifest["code_commit"] == head


def test_all_pr1_gates_present_after_rebase():
    required = [
        Path("fira_parity"),
        Path("third_party/fira_oracle"),
        Path("tests/test_fira_parity.py"),
        Path("tests/test_labeling.py"),
        Path("docs/FIRA_PARITY_SPEC.md"),
        Path("docs/FIRA_LABEL_AUDIT.md"),
        Path(".github/workflows/fira-parity.yml"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    assert not missing, f"missing PR #1 gates: {missing}"
