from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_VENDORED_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "fira_oracle"
if str(_VENDORED_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDORED_ROOT))

from optimizer_torch.fira_adamw import AdamW as FiraAdamW
from optimizer_torch.gradient_projection import GradientProjector as UpstreamGradientProjector

from .common import (
    FiraParityConfig,
    canonical_basis,
    clone_ortho_matrix,
    clone_tensor,
    effective_orientation,
    finite_tensor_tree,
    invariant_projector,
    tensor_max_abs_rel,
)


def _clone_upstream_projector(projector: UpstreamGradientProjector | None, config: FiraParityConfig) -> UpstreamGradientProjector:
    cloned = UpstreamGradientProjector(
        config.rank,
        update_proj_gap=config.update_proj_gap,
        alpha=config.alpha,
        proj_type=config.proj_type,
    )
    if projector is not None:
        cloned.ortho_matrix = clone_ortho_matrix(projector.ortho_matrix)
    return cloned


@dataclass
class OracleVerification:
    param_abs_err: float
    param_rel_err: float
    exp_avg_abs_err: float
    exp_avg_sq_abs_err: float


class OfficialFiraOracle:
    """Read-only oracle backed by the pinned upstream optimizer implementation."""

    def __init__(self, param_init: torch.Tensor, config: FiraParityConfig):
        self.config = config
        self.param = torch.nn.Parameter(param_init.detach().clone().to(config.torch_dtype()))
        group = {
            "params": [self.param],
            "rank": config.rank,
            "update_proj_gap": config.update_proj_gap,
            "alpha": config.alpha,
            "proj_type": config.proj_type,
        }
        self.optimizer = FiraAdamW(
            [group],
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            correct_bias=config.correct_bias,
            no_deprecation_warning=True,
        )

    def step_and_trace(self, grad: torch.Tensor) -> dict[str, Any]:
        if grad.is_sparse:
            raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

        grad = grad.detach().clone().to(self.param.dtype)
        param_before = self.param.detach().clone()
        state = self.optimizer.state[self.param]
        trace = self._simulate_from_snapshot(param_before, grad, state)

        self.param.grad = grad.detach().clone()
        self.optimizer.step()
        actual_state = self.optimizer.state[self.param]

        param_abs_err, param_rel_err = tensor_max_abs_rel(self.param.detach(), trace["parameter_after"])
        exp_avg_abs_err, _ = tensor_max_abs_rel(actual_state["exp_avg"], trace["exp_avg_after"])
        exp_avg_sq_abs_err, _ = tensor_max_abs_rel(actual_state["exp_avg_sq"], trace["exp_avg_sq_after"])
        adapter_tol = 1e-12 if self.param.dtype == torch.float64 else 1e-6
        if param_abs_err > adapter_tol or exp_avg_abs_err > adapter_tol or exp_avg_sq_abs_err > adapter_tol:
            raise AssertionError(
                "Official oracle adapter diverged from upstream execution: "
                f"param_abs_err={param_abs_err} exp_avg_abs_err={exp_avg_abs_err} "
                f"exp_avg_sq_abs_err={exp_avg_sq_abs_err}"
            )
        return trace

    def _simulate_from_snapshot(self, param_before: torch.Tensor, grad: torch.Tensor, state: dict[str, Any]) -> dict[str, Any]:
        step_before = int(state.get("step", 0))
        orientation = effective_orientation(tuple(grad.shape), self.config.proj_type)
        projector_before = _clone_upstream_projector(state.get("projector"), self.config)
        projector_after = _clone_upstream_projector(state.get("projector"), self.config)
        low_rank_grad = projector_after.project(grad, step_before)
        refresh = projector_before.ortho_matrix is None or step_before % self.config.update_proj_gap == 0

        exp_avg_prev = clone_tensor(state.get("exp_avg"))
        exp_avg_sq_prev = clone_tensor(state.get("exp_avg_sq"))
        scaling_norm_prev = clone_tensor(state.get("scaling_grad"))

        if exp_avg_prev is None:
            exp_avg_prev = torch.zeros_like(low_rank_grad)
            exp_avg_sq_prev = torch.zeros_like(low_rank_grad)

        beta1, beta2 = self.config.betas
        step_after = step_before + 1
        exp_avg_after = exp_avg_prev * beta1 + low_rank_grad * (1.0 - beta1)
        exp_avg_sq_after = exp_avg_sq_prev * beta2 + low_rank_grad * low_rank_grad * (1.0 - beta2)
        denom = exp_avg_sq_after.sqrt().add_(self.config.eps)

        step_size = self.config.lr
        bias_correction1 = 1.0
        bias_correction2 = 1.0
        if self.config.correct_bias:
            bias_correction1 = 1.0 - beta1 ** step_after
            bias_correction2 = 1.0 - beta2 ** step_after
            step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

        normalized_low_rank = exp_avg_after / denom
        reconstructed_low_rank = projector_after.project_back(low_rank_grad)
        remainder = grad - reconstructed_low_rank

        norm_dim = 0 if normalized_low_rank.shape[0] < normalized_low_rank.shape[1] else 1
        raw_scale = torch.norm(normalized_low_rank, dim=norm_dim) / (torch.norm(low_rank_grad, dim=norm_dim) + 1e-8)
        raw_scale_broadcast = raw_scale.unsqueeze(1) if norm_dim == 1 else raw_scale
        candidate_recovery_update = remainder * raw_scale_broadcast

        limiter_factor = 1.0
        limiter_triggered = False
        if scaling_norm_prev is not None:
            candidate_recovery_norm = torch.norm(candidate_recovery_update)
            limiter_factor = max(
                float(candidate_recovery_norm / (scaling_norm_prev + 1e-8)),
                1.01,
            ) / 1.01
            limiter_triggered = limiter_factor > 1.0
            effective_recovery_update = candidate_recovery_update / limiter_factor
            scaling_norm_after = candidate_recovery_norm / limiter_factor
        else:
            effective_recovery_update = candidate_recovery_update
            scaling_norm_after = torch.norm(candidate_recovery_update)

        full_update_before_wd = projector_after.project_back(normalized_low_rank) + effective_recovery_update
        param_after_adam = param_before - step_size * full_update_before_wd
        if self.config.weight_decay > 0.0:
            weight_decay_contribution = -self.config.lr * self.config.weight_decay * param_after_adam
        else:
            weight_decay_contribution = torch.zeros_like(param_before)
        param_after = param_after_adam + weight_decay_contribution

        trace = {
            "fixture_id": self.config.fixture_id,
            "step_index": step_after,
            "refresh_happened": refresh,
            "projection_orientation": orientation,
            "projection_rank": int(min(self.config.rank, min(grad.shape))),
            "raw_gradient": grad,
            "projection_basis_before": canonical_basis(projector_before.ortho_matrix, orientation) if projector_before.ortho_matrix is not None else None,
            "projection_basis_after": canonical_basis(projector_after.ortho_matrix, orientation),
            "invariant_projector_after": invariant_projector(projector_after.ortho_matrix, orientation),
            "projected_gradient": low_rank_grad,
            "reconstructed_projected_gradient": reconstructed_low_rank,
            "remainder_gradient": remainder,
            "exp_avg_before": exp_avg_prev,
            "exp_avg_after": exp_avg_after,
            "exp_avg_sq_before": exp_avg_sq_prev,
            "exp_avg_sq_after": exp_avg_sq_after,
            "bias_correction1": float(bias_correction1),
            "bias_correction2": float(bias_correction2),
            "normalized_low_rank_update": normalized_low_rank,
            "raw_recovery_scale": raw_scale_broadcast,
            "applied_recovery_scale": raw_scale_broadcast / limiter_factor,
            "candidate_recovery_update": candidate_recovery_update,
            "limiter_triggered": limiter_triggered,
            "limiter_factor": float(limiter_factor),
            "effective_recovery_update": effective_recovery_update,
            "complete_update_before_weight_decay": full_update_before_wd,
            "step_size": float(step_size),
            "adam_parameter_delta": -step_size * full_update_before_wd,
            "weight_decay_contribution": weight_decay_contribution,
            "complete_applied_update": param_after - param_before,
            "parameter_before": param_before,
            "parameter_after": param_after,
            "previous_recovery_norm": scaling_norm_prev,
            "recovery_norm_after": scaling_norm_after,
            "dtype": str(param_before.dtype),
        }
        trace["finite"] = finite_tensor_tree(trace)
        return trace
