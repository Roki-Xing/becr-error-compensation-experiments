# Fira Label Audit

This repository contains historical diagnostic artifacts generated before official Fira parity validation.

## Safe Current Labels

The following source files now label the coordinate implementation as a diagnostic, not official Fira:
- `README.md`
- `experiments/cifar10-small-cnn/plot_cifar10_results.py`
- `experiments/mnist-mlp/plot_mnist_results.py`
- `experiments/wikitext2-tinygpt/plot_wikitext2_results.py`
- `experiments/tier1-synthetic/plot_tier1_paper_fig2.py`
- `experiments/cifar10-small-cnn/run_cifar10_coordproj_mechanism.py`
- `experiments/mnist-mlp/run_mnist_mlp_coordproj_mechanism.py`
- `experiments/wikitext2-tinygpt/run_wikitext2_transformer_coordproj_mechanism.py`

## Historical Artifacts That Remain Diagnostic-Only

These files still contain historical `fira_raw` / `fira_clipped` result keys or rendered labels and must be interpreted as
Fira-style coordinate diagnostics, not official Fira baselines:

- `experiments/cifar10-small-cnn/results/ALL_RUNS.json`
- `experiments/mnist-mlp/results/ALL_RUNS.json`
- `experiments/wikitext2-tinygpt/results/ALL_RUNS.json`
- `experiments/tier1-synthetic/results/*.json`
- `experiments/2d-theorem-regime/results/*.json`
- Existing rendered PDFs/PNGs under `experiments/*/figures/`

## Scope Rule

This task does not regenerate old figures or mutate archived result JSON keys. The parity gate adds:
- explicit source-code labeling,
- parity tests preventing accidental `exact Fira` aliasing,
- a separate official oracle and candidate implementation under `fira_parity/`.
