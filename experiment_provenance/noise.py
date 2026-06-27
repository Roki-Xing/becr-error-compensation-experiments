from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def generate_noise_bank(
    gradients: Iterable[np.ndarray],
    *,
    std: float,
    rng_seed: int,
    dtype,
) -> np.ndarray:
    stacked = np.stack([np.asarray(g, dtype=dtype) for g in gradients], axis=0)
    rng = np.random.default_rng(int(rng_seed))
    noise = rng.standard_normal(stacked.shape).astype(dtype) * float(std)
    return noise


def hash_array(array: np.ndarray) -> str:
    arr = np.asarray(array)
    return hashlib.sha256(arr.tobytes()).hexdigest()
