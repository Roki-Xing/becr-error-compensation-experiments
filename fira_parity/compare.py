from __future__ import annotations

from typing import Any

import torch

from .common import align_low_rank_tensor, orthogonal_procrustes, tensor_max_abs_rel


INVARIANT_FIELDS = (
    "raw_gradient",
    "invariant_projector_after",
    "reconstructed_projected_gradient",
    "remainder_gradient",
    "raw_recovery_scale",
    "applied_recovery_scale",
    "candidate_recovery_update",
    "effective_recovery_update",
    "complete_update_before_weight_decay",
    "weight_decay_contribution",
    "complete_applied_update",
    "parameter_before",
    "parameter_after",
)

ALIGNED_LOW_RANK_FIELDS = (
    "projected_gradient",
    "exp_avg_before",
    "exp_avg_after",
    "exp_avg_sq_before",
    "exp_avg_sq_after",
    "normalized_low_rank_update",
)

SCALAR_FIELDS = (
    "bias_correction1",
    "bias_correction2",
    "limiter_factor",
    "step_size",
)

BOOL_FIELDS = (
    "refresh_happened",
    "limiter_triggered",
    "finite",
)

META_FIELDS = (
    "step_index",
    "projection_orientation",
    "projection_rank",
    "dtype",
)


def _tensor_from(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    raise TypeError(type(value))


def _tensor_error(a: Any, b: Any) -> tuple[float, float]:
    if a is None and b is None:
        return 0.0, 0.0
    return tensor_max_abs_rel(_tensor_from(a), _tensor_from(b))


def _passes_tol(abs_err: float, oracle_value: Any, candidate_value: Any, atol: float, rtol: float) -> bool:
    if isinstance(oracle_value, torch.Tensor):
        ref = torch.maximum(oracle_value.abs(), candidate_value.abs())
        bound = float((atol + rtol * ref).max().item()) if ref.numel() else atol
        return abs_err <= bound
    ref = max(abs(float(oracle_value)), abs(float(candidate_value)))
    return abs_err <= atol + rtol * ref


def compare_traces(
    oracle_traces: list[dict[str, Any]],
    candidate_traces: list[dict[str, Any]],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if len(oracle_traces) != len(candidate_traces):
        raise AssertionError("trace lengths differ")

    max_errors: dict[str, dict[str, float]] = {}
    first_mismatch: dict[str, Any] | None = None
    alignments: list[dict[str, Any]] = []

    for step_idx, (oracle, candidate) in enumerate(zip(oracle_traces, candidate_traces), start=1):
        for field in META_FIELDS + BOOL_FIELDS:
            if oracle[field] != candidate[field]:
                if first_mismatch is None:
                    first_mismatch = {
                        "step": step_idx,
                        "field": field,
                        "oracle": oracle[field],
                        "candidate": candidate[field],
                        "abs_err": None,
                        "rel_err": None,
                    }
                break

        orientation = oracle["projection_orientation"]
        oracle_basis = oracle["projection_basis_after"]
        candidate_basis = candidate["projection_basis_after"]
        alignment = None
        alignment_residual = 0.0
        if orientation in {"left", "right"} and oracle_basis is not None and candidate_basis is not None:
            alignment = orthogonal_procrustes(oracle_basis, candidate_basis)
            aligned_basis = candidate_basis @ alignment
            alignment_residual = float(torch.max((oracle_basis - aligned_basis).abs()).item())
        alignments.append(
            {
                "step": step_idx,
                "orientation": orientation,
                "alignment": None if alignment is None else alignment.detach().cpu().tolist(),
                "alignment_residual": alignment_residual,
            }
        )

        for field in INVARIANT_FIELDS:
            abs_err, rel_err = _tensor_error(oracle[field], candidate[field])
            current = max_errors.setdefault(field, {"abs": 0.0, "rel": 0.0})
            current["abs"] = max(current["abs"], abs_err)
            current["rel"] = max(current["rel"], rel_err)
            if first_mismatch is None and not _passes_tol(abs_err, oracle[field], candidate[field], atol=atol, rtol=rtol):
                first_mismatch = {
                    "step": step_idx,
                    "field": field,
                    "oracle": oracle[field].detach().cpu().tolist(),
                    "candidate": candidate[field].detach().cpu().tolist(),
                    "abs_err": abs_err,
                    "rel_err": rel_err,
                }

        for field in ALIGNED_LOW_RANK_FIELDS:
            oracle_value = oracle[field]
            candidate_value = candidate[field]
            if oracle_value is None and candidate_value is None:
                continue
            aligned_candidate = candidate_value
            if alignment is not None:
                aligned_candidate = align_low_rank_tensor(candidate_value, orientation, alignment)
            abs_err, rel_err = _tensor_error(oracle_value, aligned_candidate)
            current = max_errors.setdefault(field, {"abs": 0.0, "rel": 0.0})
            current["abs"] = max(current["abs"], abs_err)
            current["rel"] = max(current["rel"], rel_err)
            if first_mismatch is None and not _passes_tol(abs_err, oracle_value, aligned_candidate, atol=atol, rtol=rtol):
                first_mismatch = {
                    "step": step_idx,
                    "field": field,
                    "oracle": oracle_value.detach().cpu().tolist(),
                    "candidate": aligned_candidate.detach().cpu().tolist(),
                    "abs_err": abs_err,
                    "rel_err": rel_err,
                }

        for field in SCALAR_FIELDS:
            abs_err = abs(float(oracle[field]) - float(candidate[field]))
            rel_base = max(abs(float(oracle[field])), 1e-30)
            rel_err = abs_err / rel_base
            current = max_errors.setdefault(field, {"abs": 0.0, "rel": 0.0})
            current["abs"] = max(current["abs"], abs_err)
            current["rel"] = max(current["rel"], rel_err)
            if first_mismatch is None and not _passes_tol(abs_err, oracle[field], candidate[field], atol=atol, rtol=rtol):
                first_mismatch = {
                    "step": step_idx,
                    "field": field,
                    "oracle": oracle[field],
                    "candidate": candidate[field],
                    "abs_err": abs_err,
                    "rel_err": rel_err,
                }

    return {
        "max_errors": max_errors,
        "first_mismatch": first_mismatch,
        "alignments": alignments,
    }
