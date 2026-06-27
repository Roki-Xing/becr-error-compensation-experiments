from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .common import (
    FiraParityConfig,
    canonical_basis,
    clone_ortho_matrix,
    clone_tensor,
    effective_orientation,
    finite_tensor_tree,
    invariant_projector,
)


class CandidateGradientProjector:
    def __init__(self, rank: int, update_proj_gap: int = 200, alpha: float = 1.0, proj_type: str = "std"):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.alpha = alpha
        self.proj_type = proj_type
        self.ortho_matrix = None

    def clone(self) -> "CandidateGradientProjector":
        other = CandidateGradientProjector(
            rank=self.rank,
            update_proj_gap=self.update_proj_gap,
            alpha=self.alpha,
            proj_type=self.proj_type,
        )
        other.ortho_matrix = clone_ortho_matrix(self.ortho_matrix)
        return other

    def project(self, full_rank_grad: torch.Tensor, iteration: int) -> tuple[torch.Tensor, bool]:
        refresh = False
        if self.proj_type == "std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                refresh = self._maybe_refresh(full_rank_grad, iteration, "right")
                low_rank_grad = full_rank_grad @ self.ortho_matrix.t()
            else:
                refresh = self._maybe_refresh(full_rank_grad, iteration, "left")
                low_rank_grad = self.ortho_matrix.t() @ full_rank_grad
        elif self.proj_type == "reverse_std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                refresh = self._maybe_refresh(full_rank_grad, iteration, "left")
                low_rank_grad = self.ortho_matrix.t() @ full_rank_grad
            else:
                refresh = self._maybe_refresh(full_rank_grad, iteration, "right")
                low_rank_grad = full_rank_grad @ self.ortho_matrix.t()
        elif self.proj_type == "right":
            refresh = self._maybe_refresh(full_rank_grad, iteration, "right")
            low_rank_grad = full_rank_grad @ self.ortho_matrix.t()
        elif self.proj_type == "left":
            refresh = self._maybe_refresh(full_rank_grad, iteration, "left")
            low_rank_grad = self.ortho_matrix.t() @ full_rank_grad
        elif self.proj_type == "full":
            refresh = self._maybe_refresh(full_rank_grad, iteration, "full")
            low_rank_grad = self.ortho_matrix[0].t() @ full_rank_grad @ self.ortho_matrix[1].t()
        else:
            raise ValueError(f"unsupported proj_type={self.proj_type}")
        return low_rank_grad, refresh

    def project_back(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        if self.proj_type == "std":
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = low_rank_grad @ self.ortho_matrix
            else:
                full_rank_grad = self.ortho_matrix @ low_rank_grad
        elif self.proj_type == "reverse_std":
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:
                full_rank_grad = self.ortho_matrix @ low_rank_grad
            else:
                full_rank_grad = low_rank_grad @ self.ortho_matrix
        elif self.proj_type == "right":
            full_rank_grad = low_rank_grad @ self.ortho_matrix
        elif self.proj_type == "left":
            full_rank_grad = self.ortho_matrix @ low_rank_grad
        elif self.proj_type == "full":
            full_rank_grad = self.ortho_matrix[0] @ low_rank_grad @ self.ortho_matrix[1]
        else:
            raise ValueError(f"unsupported proj_type={self.proj_type}")
        return full_rank_grad * self.alpha

    def _maybe_refresh(self, full_rank_grad: torch.Tensor, iteration: int, basis_type: str) -> bool:
        if self.ortho_matrix is None or iteration % self.update_proj_gap == 0:
            self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, basis_type)
            return True
        return False

    @staticmethod
    def get_orthogonal_matrix(weights: torch.Tensor, rank: int, basis_type: str) -> Any:
        if weights.dtype != torch.float32:
            matrix = weights.float()
            cast_back = True
            original_dtype = weights.dtype
            original_device = weights.device
        else:
            matrix = weights
            cast_back = False
            original_dtype = weights.dtype
            original_device = weights.device

        u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        if basis_type == "right":
            basis = vh[:rank, :]
            if cast_back:
                basis = basis.to(original_device).to(original_dtype)
            return basis
        if basis_type == "left":
            basis = u[:, :rank]
            if cast_back:
                basis = basis.to(original_device).to(original_dtype)
            return basis
        if basis_type == "full":
            left = u[:, :rank]
            right = vh[:rank, :]
            if cast_back:
                left = left.to(original_device).to(original_dtype)
                right = right.to(original_device).to(original_dtype)
            return [left, right]
        raise ValueError(f"unsupported basis_type={basis_type}")


@dataclass
class CandidateState:
    step: int = 0
    projector: CandidateGradientProjector | None = None
    exp_avg: torch.Tensor | None = None
    exp_avg_sq: torch.Tensor | None = None
    scaling_grad: torch.Tensor | None = None


class ExactFiraReference:
    """Project-side exact candidate with explicit tracing and no shared step code."""

    def __init__(self, param_init: torch.Tensor, config: FiraParityConfig):
        self.config = config
        self.param = param_init.detach().clone().to(config.torch_dtype())
        self.state = CandidateState(
            projector=CandidateGradientProjector(
                rank=config.rank,
                update_proj_gap=config.update_proj_gap,
                alpha=config.alpha,
                proj_type=config.proj_type,
            )
        )

    def step_and_trace(self, grad: torch.Tensor) -> dict[str, Any]:
        if grad.is_sparse:
            raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

        grad = grad.detach().clone().to(self.param.dtype)
        param_before = self.param.detach().clone()
        step_before = self.state.step
        projector_before = self.state.projector.clone()
        orientation = effective_orientation(tuple(grad.shape), self.config.proj_type)

        low_rank_grad, refresh = self.state.projector.project(grad, step_before)
        if self.state.exp_avg is None:
            self.state.exp_avg = torch.zeros_like(low_rank_grad)
            self.state.exp_avg_sq = torch.zeros_like(low_rank_grad)

        exp_avg_prev = clone_tensor(self.state.exp_avg)
        exp_avg_sq_prev = clone_tensor(self.state.exp_avg_sq)
        scaling_norm_prev = clone_tensor(self.state.scaling_grad)

        self.state.step += 1
        beta1, beta2 = self.config.betas
        self.state.exp_avg.mul_(beta1).add_(low_rank_grad, alpha=1.0 - beta1)
        self.state.exp_avg_sq.mul_(beta2).addcmul_(low_rank_grad, low_rank_grad, value=1.0 - beta2)

        denom = self.state.exp_avg_sq.sqrt().add_(self.config.eps)
        step_size = self.config.lr
        bias_correction1 = 1.0
        bias_correction2 = 1.0
        if self.config.correct_bias:
            bias_correction1 = 1.0 - beta1 ** self.state.step
            bias_correction2 = 1.0 - beta2 ** self.state.step
            step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

        normalized_low_rank = self.state.exp_avg / denom
        reconstructed_low_rank = self.state.projector.project_back(low_rank_grad)
        remainder = grad - reconstructed_low_rank

        norm_dim = 0 if normalized_low_rank.shape[0] < normalized_low_rank.shape[1] else 1
        raw_scale = torch.norm(normalized_low_rank, dim=norm_dim) / (torch.norm(low_rank_grad, dim=norm_dim) + 1e-8)
        raw_scale_broadcast = raw_scale.unsqueeze(1) if norm_dim == 1 else raw_scale
        candidate_recovery_update = remainder * raw_scale_broadcast

        limiter_factor = 1.0
        limiter_triggered = False
        if self.state.scaling_grad is not None:
            candidate_recovery_norm = torch.norm(candidate_recovery_update)
            limiter_factor = max(
                float(candidate_recovery_norm / (self.state.scaling_grad + 1e-8)),
                1.01,
            ) / 1.01
            limiter_triggered = limiter_factor > 1.0
            effective_recovery_update = candidate_recovery_update / limiter_factor
            self.state.scaling_grad = candidate_recovery_norm / limiter_factor
        else:
            effective_recovery_update = candidate_recovery_update
            self.state.scaling_grad = torch.norm(candidate_recovery_update)

        full_update_before_wd = self.state.projector.project_back(normalized_low_rank) + effective_recovery_update
        param_after_adam = param_before - step_size * full_update_before_wd
        if self.config.weight_decay > 0.0:
            weight_decay_contribution = -self.config.lr * self.config.weight_decay * param_after_adam
        else:
            weight_decay_contribution = torch.zeros_like(param_before)
        self.param = param_after_adam + weight_decay_contribution

        trace = {
            "fixture_id": self.config.fixture_id,
            "step_index": self.state.step,
            "refresh_happened": refresh,
            "projection_orientation": orientation,
            "projection_rank": int(min(self.config.rank, min(grad.shape))),
            "raw_gradient": grad,
            "projection_basis_before": canonical_basis(projector_before.ortho_matrix, orientation) if projector_before.ortho_matrix is not None else None,
            "projection_basis_after": canonical_basis(self.state.projector.ortho_matrix, orientation),
            "invariant_projector_after": invariant_projector(self.state.projector.ortho_matrix, orientation),
            "projected_gradient": low_rank_grad,
            "reconstructed_projected_gradient": reconstructed_low_rank,
            "remainder_gradient": remainder,
            "exp_avg_before": exp_avg_prev,
            "exp_avg_after": clone_tensor(self.state.exp_avg),
            "exp_avg_sq_before": exp_avg_sq_prev,
            "exp_avg_sq_after": clone_tensor(self.state.exp_avg_sq),
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
            "complete_applied_update": self.param - param_before,
            "parameter_before": param_before,
            "parameter_after": self.param.detach().clone(),
            "previous_recovery_norm": scaling_norm_prev,
            "recovery_norm_after": clone_tensor(self.state.scaling_grad),
            "dtype": str(self.param.dtype),
        }
        trace["finite"] = finite_tensor_tree(trace)
        return trace
