from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_coordinate_fira_is_not_labeled_exact():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Fira-style coordinate diagnostics" in readme
    assert "exact Fira baseline" not in readme

    plot_files = [
        REPO_ROOT / "experiments/cifar10-small-cnn/plot_cifar10_results.py",
        REPO_ROOT / "experiments/mnist-mlp/plot_mnist_results.py",
        REPO_ROOT / "experiments/wikitext2-tinygpt/plot_wikitext2_results.py",
        REPO_ROOT / "experiments/tier1-synthetic/plot_tier1_paper_fig2.py",
    ]
    for path in plot_files:
        text = path.read_text(encoding="utf-8")
        assert "Fira-style coord (raw)" in text
        assert "Fira (raw)" not in text
        assert "Fira (clipped)" not in text

    run_files = [
        REPO_ROOT / "experiments/cifar10-small-cnn/run_cifar10_coordproj_mechanism.py",
        REPO_ROOT / "experiments/mnist-mlp/run_mnist_mlp_coordproj_mechanism.py",
        REPO_ROOT / "experiments/wikitext2-tinygpt/run_wikitext2_transformer_coordproj_mechanism.py",
    ]
    for path in run_files:
        text = path.read_text(encoding="utf-8")
        assert "official Fira" in text
