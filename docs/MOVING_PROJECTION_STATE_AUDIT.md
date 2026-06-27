# Moving Projection State Audit

This note records why `P0-MOVING-PROJECTION-STATE-002` adds a separate
moving-projection correctness harness instead of reusing the older synthetic or
neural diagnostic loops.

## Legacy synthetic confounds

- `experiments/tier1-synthetic/run_synth_tier1_suite.py:250-280`
  decomposes `g` first and only then forms `h = g_perp + e`. Under changing
  `P_t`, this is not the required compensated decomposition
  `A_t = G_t + E_t`, `R_t = P_t^T A_t`, `Q_t = A_t - P_t R_t`.
- `experiments/tier1-synthetic/run_synth_tier1_suite.py:433-480`
  iterates optimizers sequentially while reusing the same stateful `P_sched`
  object and the same advancing RNG stream. That couples methods through
  scheduler state and noise consumption order.

## Legacy neural confounds

- `experiments/cifar10-small-cnn/run_cifar10_coordproj_mechanism.py:103-126`
  resets `idx`, `m`, `v`, `t`, and residual `e` whenever projection refresh
  happens.
- `experiments/mnist-mlp/run_mnist_mlp_coordproj_mechanism.py:87-108`
  has the same refresh-reset pattern.
- `experiments/wikitext2-tinygpt/run_wikitext2_transformer_coordproj_mechanism.py:255-274`
  has the same refresh-reset pattern.

These scripts remain legacy coordinate diagnostics. They are not used as the
corrected moving-projection source of truth in this task.

## Corrected harness scope

The new harness under `moving_projection_state/` and
`experiments/tier1-synthetic/run_moving_projection_state_suite.py` isolates:

- explicit reset vs non-reset behavior
- carry-without-transport vs basis-overlap transport
- compensated residual decomposition in current projection
- scheduler isolation per method
- shared pre-generated noise for CRN-style paired comparisons

Its outputs live under
`experiments/tier1-synthetic/moving_projection_artifacts/20260627T000000Z_p0_moving_projection_state/`
and should be used for moving-projection state claims in place of the older
confounded loops.
