# BECR Error Compensation Experiments

This private repository contains the code, notebooks, figures, and result files
for the BECR / Fira-style recovery mechanism diagnostics.

The experimental scope is intentionally mechanism-oriented. The repository
does not claim optimizer superiority over full AdamW. It tests:

1. Whether raw Fira-style scalar recovery can leave a nonzero orthogonal
   gradient when the cumulative recovery strength is finite.
2. Whether clipping and BECR repair the failure in the theorem regime.
3. Whether bounded residual transmission is necessary for stable compensation.
4. Whether the same scale and residual diagnostics appear in small neural
   training loops.

Current `fira_raw` / `fira_clipped` entries in this repository are
Fira-style coordinate diagnostics, not parity-validated official Fira
baselines.

## Repository Layout

```text
experiments/
  2d-theorem-regime/       # Main Figure 1 and theorem-regime notebooks
  tier1-synthetic/         # High-dimensional synthetic suite and Figure 2
  cifar10-small-cnn/       # CIFAR-10 neural mechanism diagnostic
  mnist-mlp/               # MNIST appendix diagnostic
  wikitext2-tinygpt/       # Preliminary TinyGPT diagnostic
docs/
  source-notes/            # Original project notes
```

## Paper-Ready Figures

- `experiments/2d-theorem-regime/figures/fig1_theorem_regime.pdf`
- `experiments/tier1-synthetic/figures/fig2_tier1_synthetic.pdf`
- `experiments/cifar10-small-cnn/figures/cifar10_coordproj__MAIN.pdf`
- `experiments/cifar10-small-cnn/figures/cifar10_coordproj__PER_LAYER.pdf`
- `experiments/mnist-mlp/figures/mnist_coordproj__MAIN.pdf`
- `experiments/mnist-mlp/figures/mnist_coordproj__PER_LAYER.pdf`

## Main Scripts

```bash
python experiments/tier1-synthetic/run_synth_tier1_suite.py
python experiments/tier1-synthetic/run_moving_projection_state_suite.py
python experiments/tier1-synthetic/plot_tier1_paper_fig2.py
python experiments/cifar10-small-cnn/run_cifar10_coordproj_mechanism.py --seeds 0 1 2
python experiments/mnist-mlp/run_mnist_mlp_coordproj_mechanism.py --seeds 0 1 2
```

The CIFAR-10 and MNIST scripts include a first-step projection initialization
sanity check and the residual transmission ablations:

- `rho_no_lower_bound`: `rho_t = min(phi_raw, rho_max)`
- `coupled_unbounded`: `rho_t = phi_raw`

## Moving-Projection State Harness

`experiments/tier1-synthetic/run_moving_projection_state_suite.py` is the
P0 moving-projection correctness harness. It generates small deterministic
and CRN-controlled traces for the explicit modes:

- `state_reset_explicit`
- `official_fira_carry`
- `projection_aware_transport`
- `full_residual_current_projection`

These artifacts are separate from the older tier1/neural result aggregates and
should be used when auditing moving-projection state transport, residual
decomposition, scheduler isolation, and CRN provenance. See
`docs/MOVING_PROJECTION_STATE_AUDIT.md` for the legacy-confound audit that
motivated this harness. The committed files under
`experiments/tier1-synthetic/moving_projection_artifacts/` are review
snapshots; CI regeneration is the executable provenance check.

## Data

Datasets are intentionally excluded from Git. The neural scripts download or
prepare their required datasets under their local `data/` directories.

For neural diagnostics, `results/ALL_RUNS.json` is the authoritative aggregate.
Individual run JSON copies are intentionally omitted from this snapshot to avoid
mixing earlier and corrected reruns.

## Environment

The synthetic suite requires NumPy and Matplotlib. Neural diagnostics require
PyTorch and Torchvision. See `requirements.txt`.
