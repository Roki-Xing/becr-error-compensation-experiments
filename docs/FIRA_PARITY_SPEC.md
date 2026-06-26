# Fira Parity Spec

Task: `P0-EXACT-FIRA-PARITY-001`  
Upstream repository: `https://github.com/xichen-fy/Fira`  
Pinned commit: `5af6a9860b633138f12837ad25e528b7d54217eb`  
Pinned files:
- `optimizer_torch/fira_adamw.py`
- `optimizer_torch/gradient_projection.py`
- `optimizer_torch/__init__.py`
- `LICENSE`

This document is based on the pinned upstream code, not the paper summary.

## Exact Step Semantics

1. Projection orientation
   [Official] `proj_type="std"` uses right projection when `rows >= cols`, and left projection when `rows < cols`.
   Source: `third_party/fira_oracle/optimizer_torch/gradient_projection.py`

2. Projection initialization
   [Official] The projector is created lazily on the first optimizer step when `state["projector"]` is absent.
   Source: `third_party/fira_oracle/optimizer_torch/fira_adamw.py`

3. Refresh schedule
   [Official] Projection refresh is tested before incrementing `state["step"]`. The refresh condition is
   `ortho_matrix is None or iter % update_proj_gap == 0`, where `iter` is the pre-increment step counter.
   [Derived] With 1-indexed user-facing steps, refresh happens at `1, 1+gap, 1+2*gap, ...`.

4. SVD input
   [Official] SVD is computed on the current raw full-rank gradient passed to `project(...)`.
   Source: `third_party/fira_oracle/optimizer_torch/gradient_projection.py`

5. Rank truncation
   [Official] There is no explicit rank validation. Slicing `U[:, :rank]` or `Vh[:rank, :]` silently clamps
   to `min(rank, min(shape))`.

6. Projected gradient
   [Official] For right projection, `R = G @ B^T`. For left projection, `R = A^T @ G`.
   Here `B = Vh[:rank, :]` and `A = U[:, :rank]`.

7. Reconstructed projected gradient
   [Official] For right projection, `G_parallel = R @ B`. For left projection, `G_parallel = A @ R`.
   The reconstruction is multiplied by `alpha` inside `project_back(...)`.

8. Orthogonal remainder
   [Official] The recovered branch uses `G_perp = p.grad - project_back(R)`.
   [Derived] The official code uses the original full-rank gradient tensor `p.grad`, not the projected `grad`.

9. First moment update
   [Official] `exp_avg = beta1 * exp_avg + (1-beta1) * R`.

10. Second moment update
    [Official] `exp_avg_sq = beta2 * exp_avg_sq + (1-beta2) * R^2`.

11. Bias correction
    [Official] `state["step"]` is incremented before bias correction.
    `step_size = lr * sqrt(1-beta2^t) / (1-beta1^t)` when `correct_bias=True`.

12. Adam epsilon
    [Official] The Adam epsilon is added to `sqrt(exp_avg_sq)` in low-rank coordinates before division.

13. Recovery scale formula
    [Official] The recovery scale is not a scalar. It is a vector produced by
    `norm(norm_grad, dim=norm_dim) / (norm(grad, dim=norm_dim) + 1e-8)`.

14. Broadcast dimension
    [Official] `norm_dim = 0 if norm_grad.shape[0] < norm_grad.shape[1] else 1`.
    [Derived] For right projection on tall-or-square matrices, this gives row-wise scaling broadcast across columns.
    [Derived] For left projection on wide matrices, this gives column-wise scaling broadcast across rows.

15. Recovery denominator epsilon
    [Official] The recovery denominator uses a hard-coded `1e-8`, not the Adam epsilon and not a separate config.

16. Candidate recovery update
    [Official] `candidate = (p.grad - project_back(R)) * scaling_factor`.

17. Norm-growth limiter trigger
    [Official] The limiter is active only after `state["scaling_grad"]` has been initialized on a previous step.
    The limiter factor is `max(candidate_norm / (prev_norm + 1e-8), 1.01) / 1.01`.

18. Limiter first-step behavior
    [Official] There is no limiter on the first recovered step. The state is initialized as
    `state["scaling_grad"] = ||candidate||`.

19. Effective recovery update
    [Official] `candidate` is divided by the limiter factor when the limiter state exists.
    [Derived] The effective recovery scale is `scaling_factor / limiter_factor`.

20. Final adaptive update before weight decay
    [Official] `full_update = project_back(norm_grad) + effective_recovery_update`.

21. Decoupled weight decay
    [Official] Weight decay is applied after the adaptive update, using the post-update parameter and the raw group
    learning rate `lr`, not the bias-corrected `step_size`.

22. Final parameter update order
    [Official]
    1. low-rank project
    2. update `exp_avg`, `exp_avg_sq`, `step`
    3. compute low-rank adaptive direction
    4. reconstruct projected branch
    5. build recovered branch
    6. apply limiter
    7. apply adaptive parameter step
    8. apply decoupled weight decay

23. Refresh-state survival
    [Official] Refresh does not reset `exp_avg`, `exp_avg_sq`, `step`, or `scaling_grad`.

24. Dtype behavior
    [Official] `get_orthogonal_matrix(...)` casts any non-`torch.float` input, including `float64`, to `float32`
    for SVD, then casts the basis back to the original dtype/device.
    [Derived] Exact parity therefore requires reproducing the internal float32 SVD path for `float64` fixtures.

25. Sparse and zero-gradient behavior
    [Official] Sparse gradients raise `RuntimeError`.
    [Official] Zero gradients have no special branch; the standard SVD and scaling formula are used.

## Project Mismatch Audit

1. Projection family
   [Project mismatch] The current neural diagnostic code uses top-`|g|` coordinate selection instead of matrix SVD.

2. Refresh-state handling
   [Project mismatch] The current neural diagnostic code reinitializes `m`, `v`, `t`, and residual state on refresh.
   Official Fira preserves optimizer state across refresh.

3. Recovery scale
   [Project mismatch] The current neural diagnostic code uses a single scalar
   `phi_raw = ||psi|| / (||R|| + eps_scale)`.
   Official Fira uses row-wise or column-wise vector scaling, depending on orientation.

4. Recovery epsilon
   [Project mismatch] The current neural diagnostic code uses configurable `eps_scale`.
   Official Fira uses a hard-coded `1e-8` in the recovery denominator.

5. Norm-growth limiter
   [Project mismatch] The current neural diagnostic code has no exact upstream limiter in the `fira_raw` path.
   Official Fira always runs the limiter once the recovery state exists.

6. Weight decay
   [Project mismatch] The current neural diagnostic code is not an exact reproduction of upstream decoupled weight decay
   ordering or scaling because it does not implement the upstream matrix-recovery path.

7. Step counter timing
   [Project mismatch] The current neural diagnostic scripts maintain a separate `_step` for stale coordinate refresh.
   Official Fira refresh is keyed off the optimizer state's pre-increment `state["step"]`.

8. Remainder definition
   [Project mismatch] The current neural diagnostic code uses coordinate-masked remainder.
   Official Fira uses matrix reconstruction from an SVD basis.
