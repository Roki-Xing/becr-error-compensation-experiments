from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class FiraParityConfig:
    fixture_id: str
    shape: tuple[int, int]
    rank: int
    update_proj_gap: int
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    correct_bias: bool
    alpha: float
    proj_type: str
    dtype: str
    steps: int
    seed: int
    expect_limiter: bool

    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)


def clone_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().clone()


def clone_ortho_matrix(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [clone_tensor(item) for item in value]
    return clone_tensor(value)


def effective_orientation(shape: tuple[int, int], proj_type: str) -> str:
    rows, cols = shape
    if proj_type == "std":
        return "right" if rows >= cols else "left"
    if proj_type == "reverse_std":
        return "left" if rows >= cols else "right"
    if proj_type in {"left", "right", "full"}:
        return proj_type
    raise ValueError(f"unsupported proj_type={proj_type}")


def effective_rank(shape: tuple[int, int], rank: int) -> int:
    return int(min(rank, min(shape)))


def canonical_basis(ortho_matrix: Any, orientation: str) -> Any:
    if ortho_matrix is None:
        return None
    if orientation == "right":
        return ortho_matrix.t()
    if orientation == "left":
        return ortho_matrix
    if orientation == "full":
        left, right = ortho_matrix
        return {"left": left, "right": right.t()}
    raise ValueError(f"unsupported orientation={orientation}")


def invariant_projector(ortho_matrix: Any, orientation: str) -> Any:
    if ortho_matrix is None:
        return None
    if orientation == "right":
        return ortho_matrix.t() @ ortho_matrix
    if orientation == "left":
        return ortho_matrix @ ortho_matrix.t()
    if orientation == "full":
        left, right = ortho_matrix
        return {
            "left": left @ left.t(),
            "right": right.t() @ right,
        }
    raise ValueError(f"unsupported orientation={orientation}")


def tensor_to_builtin(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: tensor_to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_builtin(v) for v in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def trace_to_jsonable(trace: dict[str, Any]) -> dict[str, Any]:
    return {key: tensor_to_builtin(value) for key, value in trace.items()}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def repo_command(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def repo_manifest(repo_root: Path) -> dict[str, Any]:
    return {
        "project_commit": repo_command(["git", "rev-parse", "HEAD"], repo_root),
        "project_branch": repo_command(["git", "branch", "--show-current"], repo_root),
        "project_status": repo_command(["git", "status", "--short", "--branch"], repo_root),
    }


def tensor_max_abs_rel(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff = (a - b).abs()
    abs_err = float(diff.max().item()) if diff.numel() else 0.0
    denom = torch.maximum(a.abs(), torch.full_like(a, 1e-30))
    rel = diff / denom
    rel_err = float(rel.max().item()) if rel.numel() else 0.0
    return abs_err, rel_err


def orthogonal_procrustes(oracle_basis: torch.Tensor, candidate_basis: torch.Tensor) -> torch.Tensor:
    cross = candidate_basis.transpose(0, 1) @ oracle_basis
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    return u @ vh


def align_low_rank_tensor(
    tensor: torch.Tensor,
    orientation: str,
    alignment: torch.Tensor,
) -> torch.Tensor:
    if orientation == "right":
        return tensor @ alignment
    if orientation == "left":
        return alignment.transpose(0, 1) @ tensor
    raise ValueError(f"alignment not implemented for orientation={orientation}")


def finite_tensor_tree(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(finite_tensor_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensor_tree(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def config_to_dict(config: FiraParityConfig) -> dict[str, Any]:
    return asdict(config)
