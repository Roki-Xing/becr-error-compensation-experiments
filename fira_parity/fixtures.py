from __future__ import annotations

import math
from dataclasses import replace

import torch

from .common import FiraParityConfig


def _base_random_matrix(shape: tuple[int, int], seed: int, step: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 97 * step)
    rows, cols = shape
    matrix = torch.randn(rows, cols, generator=generator, dtype=torch.float64)
    diag = torch.zeros(rows, cols, dtype=torch.float64)
    for idx in range(min(rows, cols)):
        diag[idx, idx] = 1.5 + 0.25 * idx + 0.05 * step
    return (0.7 * matrix + diag).to(dtype)


def _gradient_sequence(config: FiraParityConfig) -> list[torch.Tensor]:
    dtype = config.torch_dtype()
    rows, cols = config.shape
    seq: list[torch.Tensor] = []
    for step in range(config.steps):
        base = _base_random_matrix(config.shape, config.seed, step, dtype)
        if config.expect_limiter:
            amp = 0.02 if step % 7 == 0 else 5.0 if step % 7 == 1 else 1.0 + 0.15 * math.sin(step)
        else:
            amp = 0.3 + 0.02 * step
        row_bias = torch.linspace(1.0, 1.0 + 0.2 * max(rows - 1, 0), rows, dtype=dtype).unsqueeze(1)
        col_bias = torch.linspace(1.0, 1.0 + 0.1 * max(cols - 1, 0), cols, dtype=dtype).unsqueeze(0)
        seq.append(base * amp + 0.05 * row_bias @ col_bias)
    return seq


def _initial_param(config: FiraParityConfig) -> torch.Tensor:
    dtype = config.torch_dtype()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 1000)
    param = torch.randn(config.shape, generator=generator, dtype=torch.float64) * 0.2
    return param.to(dtype)


def parity_fixtures() -> dict[str, tuple[FiraParityConfig, torch.Tensor, list[torch.Tensor]]]:
    base = [
        FiraParityConfig(
            fixture_id="shape_2x1_rank1_gap1_fp64",
            shape=(2, 1),
            rank=1,
            update_proj_gap=1,
            lr=1e-2,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.0,
            correct_bias=True,
            alpha=1.0,
            proj_type="std",
            dtype="float64",
            steps=12,
            seed=11,
            expect_limiter=False,
        ),
        FiraParityConfig(
            fixture_id="shape_2x2_rank1_gap2_fp64",
            shape=(2, 2),
            rank=1,
            update_proj_gap=2,
            lr=5e-3,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.0,
            correct_bias=True,
            alpha=1.0,
            proj_type="std",
            dtype="float64",
            steps=18,
            seed=17,
            expect_limiter=False,
        ),
        FiraParityConfig(
            fixture_id="shape_2x2_rank2_gap5_fp32_wd",
            shape=(2, 2),
            rank=2,
            update_proj_gap=5,
            lr=8e-3,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=1e-2,
            correct_bias=True,
            alpha=1.0,
            proj_type="std",
            dtype="float32",
            steps=20,
            seed=23,
            expect_limiter=False,
        ),
        FiraParityConfig(
            fixture_id="shape_4x3_rank1_gap2_fp64_limiter_active",
            shape=(4, 3),
            rank=1,
            update_proj_gap=2,
            lr=1e-2,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.0,
            correct_bias=True,
            alpha=1.0,
            proj_type="std",
            dtype="float64",
            steps=24,
            seed=31,
            expect_limiter=True,
        ),
        FiraParityConfig(
            fixture_id="shape_4x3_rank2_gap5_fp32_wd_limiter_active",
            shape=(4, 3),
            rank=2,
            update_proj_gap=5,
            lr=7e-3,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=1e-2,
            correct_bias=True,
            alpha=1.0,
            proj_type="std",
            dtype="float32",
            steps=30,
            seed=37,
            expect_limiter=True,
        ),
        FiraParityConfig(
            fixture_id="shape_3x4_rank2_gap2_fp64_left_orientation",
            shape=(3, 4),
            rank=2,
            update_proj_gap=2,
            lr=6e-3,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.0,
            correct_bias=True,
            alpha=1.0,
            proj_type="std",
            dtype="float64",
            steps=20,
            seed=41,
            expect_limiter=False,
        ),
        FiraParityConfig(
            fixture_id="shape_4x3_rank2_gap5_fp64_100step",
            shape=(4, 3),
            rank=2,
            update_proj_gap=5,
            lr=7e-3,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=1e-2,
            correct_bias=True,
            alpha=1.0,
            proj_type="std",
            dtype="float64",
            steps=100,
            seed=53,
            expect_limiter=True,
        ),
        FiraParityConfig(
            fixture_id="shape_2x2_rank1_gap2_beta0",
            shape=(2, 2),
            rank=1,
            update_proj_gap=2,
            lr=1e-2,
            betas=(0.0, 0.0),
            eps=1e-6,
            weight_decay=0.0,
            correct_bias=True,
            alpha=1.0,
            proj_type="std",
            dtype="float64",
            steps=10,
            seed=61,
            expect_limiter=False,
        ),
    ]
    return {
        cfg.fixture_id: (cfg, _initial_param(cfg), _gradient_sequence(cfg))
        for cfg in base
    }
