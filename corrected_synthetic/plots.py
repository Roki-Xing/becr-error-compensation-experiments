from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from experiment_provenance.json_utils import write_json_strict

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _write_plot_metadata(
    *,
    path_base: Path,
    title: str,
    aggregate_manifest_path: Path,
    aggregate_manifest: dict[str, Any],
    candidate_level: str,
    main_or_appendix_recommendation: str,
) -> Path:
    meta = {
        "title": title,
        "aggregate_id": aggregate_manifest["aggregate_id"],
        "aggregate_manifest_path": str(aggregate_manifest_path),
        "source_run_ids": aggregate_manifest["source_run_ids"],
        "source_manifest_paths": aggregate_manifest["source_manifest_paths"],
        "code_commit": aggregate_manifest["code_commit"],
        "config_hashes": aggregate_manifest["config_hashes"],
        "noise_hashes": aggregate_manifest["noise_hashes"],
        "candidate_level": candidate_level,
        "figure_candidate_level": candidate_level,
        "main_or_appendix_recommendation": main_or_appendix_recommendation,
    }
    meta_path = path_base.with_suffix(".metadata.json")
    write_json_strict(meta_path, meta, sort_keys=True)
    return meta_path


def plot_theorem_regime(
    *,
    out_dir: Path,
    aggregate_manifest_path: Path,
    aggregate_manifest: dict[str, Any],
    method_payloads: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    order = ["proj_baseline", "fira_raw", "fira_clipped", "becr", "rho_no_lower_bound", "coupled_unbounded"]
    label = {
        "proj_baseline": "Projected",
        "fira_raw": "Raw Fira-style",
        "fira_clipped": "Lower-clipped",
        "becr": "BECR",
        "rho_no_lower_bound": "No-lower rho",
        "coupled_unbounded": "Coupled rho=phi",
    }
    color = {
        "proj_baseline": "#666666",
        "fira_raw": "#d62728",
        "fira_clipped": "#1f77b4",
        "becr": "#2ca02c",
        "rho_no_lower_bound": "#9467bd",
        "coupled_unbounded": "#ff7f0e",
    }
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 7.8))
    ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()
    for method in order:
        if method not in method_payloads:
            continue
        series = method_payloads[method]["series"]
        xs = list(range(len(series["grad_norm"])))
        ax1.plot(xs, np.maximum(series["grad_norm"], 1e-300), label=label[method], color=color[method], linewidth=1.6)
        ax2.plot(xs, np.maximum(np.abs(series["y_component"]), 1e-300), label=label[method], color=color[method], linewidth=1.6)
        ax3.plot(xs, np.maximum(series["phi_raw"], 1e-300), label=f"{label[method]} raw", color=color[method], linewidth=1.2)
        ax3.plot(xs, np.maximum(series["s_applied"], 1e-300), linestyle="--", color=color[method], linewidth=1.0)
        ax4.plot(xs, np.maximum(series["phi_cum"], 1e-300), label=f"{label[method]} raw", color=color[method], linewidth=1.2)
        ax4.plot(xs, np.maximum(series["s_applied_cum"], 1e-300), linestyle="--", color=color[method], linewidth=1.0)
        if any(math.isfinite(float(v)) for v in series["residual_norm"]):
            ax5.plot(xs, np.maximum(series["residual_norm"], 1e-300), label=label[method], color=color[method], linewidth=1.6)
        ax6.plot(xs, np.maximum(series["f"], 1e-300), label=label[method], color=color[method], linewidth=1.6)

    ax1.set_title("(a) ||grad||")
    ax2.set_title("(b) |y_t|")
    ax3.set_title("(c) raw phi vs applied s")
    ax4.set_title("(d) cumulative raw vs applied")
    ax5.set_title("(e) residual norm")
    ax6.set_title("(f) objective")
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlabel("step")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path_base = out_dir / "figure_candidate_1_theorem_regime"
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)
    meta = _write_plot_metadata(
        path_base=path_base,
        title="Corrected theorem-regime 2D validation",
        aggregate_manifest_path=aggregate_manifest_path,
        aggregate_manifest=aggregate_manifest,
        candidate_level="main-candidate",
        main_or_appendix_recommendation="main-candidate",
    )
    return {"png": path_base.with_suffix(".png"), "pdf": path_base.with_suffix(".pdf"), "metadata": meta}


def plot_high_dimensional(
    *,
    out_dir: Path,
    aggregate_manifest_path: Path,
    aggregate_manifest: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Path]:
    methods = list(summary["methods"].keys())
    labels = [summary["methods"][m]["label"] for m in methods]
    grad = [summary["methods"][m]["grad_norm_mean"] for m in methods]
    gpar = [summary["methods"][m]["grad_par_norm_mean"] for m in methods]
    gperp = [summary["methods"][m]["grad_perp_norm_mean"] for m in methods]
    residual = [summary["methods"][m]["residual_norm_mean"] if summary["methods"][m]["residual_norm_mean"] is not None else math.nan for m in methods]
    fig, axs = plt.subplots(2, 2, figsize=(13.8, 8.2))
    xs = np.arange(len(methods))
    axs[0, 0].bar(xs, grad, color="#4c72b0")
    axs[0, 0].set_title("Final ||grad|| mean")
    axs[0, 1].bar(xs, gperp, color="#dd8452")
    axs[0, 1].set_title("Final ||grad_perp|| mean")
    axs[1, 0].bar(xs, gpar, color="#55a868")
    axs[1, 0].set_title("Final ||grad_parallel|| mean")
    axs[1, 1].bar(xs, [r if math.isfinite(float(r)) else 1e-300 for r in residual], color="#c44e52")
    axs[1, 1].set_title("Final residual norm mean")
    for ax in axs.flatten():
        ax.set_xticks(xs, labels, rotation=25, ha="right")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path_base = out_dir / "figure_candidate_2_high_dimensional"
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)
    meta = _write_plot_metadata(
        path_base=path_base,
        title="Corrected high-dimensional stale synthetic",
        aggregate_manifest_path=aggregate_manifest_path,
        aggregate_manifest=aggregate_manifest,
        candidate_level="main-or-appendix",
        main_or_appendix_recommendation="main-or-appendix",
    )
    return {"png": path_base.with_suffix(".png"), "pdf": path_base.with_suffix(".pdf"), "metadata": meta}


def plot_refresh_sweep(
    *,
    out_dir: Path,
    aggregate_manifest_path: Path,
    aggregate_manifest: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Path]:
    fig, axs = plt.subplots(2, 1, figsize=(10.8, 8.4), sharex=True)
    K_values = summary["K_values_numeric"]
    for family_key, title, ax in [
        ("coordinate_family", "Coordinate diagnostics", axs[0]),
        ("state_modes", "State / transport modes", axs[1]),
    ]:
        for method, row in summary[family_key].items():
            ax.plot(K_values, row["grad_norm_final"], marker="o", linewidth=1.6, label=row["label"])
            ax.plot(K_values, row["residual_norm_final"], marker="x", linestyle="--", linewidth=1.1, color=ax.lines[-1].get_color())
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
    axs[1].set_xlabel("refresh interval K (inf shown as 1e6)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path_base = out_dir / "figure_candidate_3_refresh_sweep"
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)
    meta = _write_plot_metadata(
        path_base=path_base,
        title="Corrected refresh sweep under explicit provenance",
        aggregate_manifest_path=aggregate_manifest_path,
        aggregate_manifest=aggregate_manifest,
        candidate_level="main-or-appendix",
        main_or_appendix_recommendation="main-or-appendix",
    )
    return {"png": path_base.with_suffix(".png"), "pdf": path_base.with_suffix(".pdf"), "metadata": meta}


def plot_anisotropic_noise(
    *,
    out_dir: Path,
    aggregate_manifest_path: Path,
    aggregate_manifest: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Path]:
    methods = list(summary["methods"].keys())
    labels = [summary["methods"][m]["label"] for m in methods]
    means = [summary["methods"][m]["grad_norm_mean"] for m in methods]
    ci_low = [summary["methods"][m]["grad_norm_ci_low"] for m in methods]
    ci_high = [summary["methods"][m]["grad_norm_ci_high"] for m in methods]
    errs = np.vstack([np.asarray(means) - np.asarray(ci_low), np.asarray(ci_high) - np.asarray(means)])
    fig, axs = plt.subplots(1, 3, figsize=(16.2, 4.8))
    xs = np.arange(len(methods))
    axs[0].bar(xs, means, yerr=errs, capsize=4, color="#4c72b0")
    axs[0].set_xticks(xs, labels, rotation=25, ha="right")
    axs[0].set_yscale("log")
    axs[0].set_title("Final ||grad|| mean ± CI")
    axs[0].grid(True, axis="y", alpha=0.3)

    paired = summary["paired_differences"]["becr_minus_clipped"]
    axs[1].axhline(0.0, color="k", linewidth=1.0, alpha=0.4)
    axs[1].plot(range(len(paired)), paired, marker="o", linewidth=1.4)
    axs[1].set_title("Paired diff: BECR - clipped")
    axs[1].set_xlabel("seed")
    axs[1].grid(True, alpha=0.3)

    spike_data = [summary["methods"][m]["phi_raw_p95_values"] for m in methods]
    axs[2].boxplot(spike_data, tick_labels=labels, showfliers=False)
    axs[2].set_yscale("log")
    axs[2].tick_params(axis="x", rotation=25)
    axs[2].set_title("phi_raw p95 by method")
    axs[2].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path_base = out_dir / "figure_candidate_4_anisotropic_noise"
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)
    meta = _write_plot_metadata(
        path_base=path_base,
        title="Corrected anisotropic noise with CRN",
        aggregate_manifest_path=aggregate_manifest_path,
        aggregate_manifest=aggregate_manifest,
        candidate_level="appendix-or-main-panel",
        main_or_appendix_recommendation="appendix-or-main-panel",
    )
    return {"png": path_base.with_suffix(".png"), "pdf": path_base.with_suffix(".pdf"), "metadata": meta}


def plot_adam_limiter_loss(
    *,
    out_dir: Path,
    aggregate_manifest_path: Path,
    aggregate_manifest: dict[str, Any],
    setting_id: str,
    method_payloads: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    order = [
        "projected_adam_baseline",
        "fira_style_adam_limiter_no_residual",
        "clipping_only_limiter",
        "becr_effective_signal_residual",
        "wrong_pre_limiter_residual",
        "no_lower_bound_residual",
    ]
    labels = {
        "projected_adam_baseline": "Projected Adam",
        "fira_style_adam_limiter_no_residual": "No residual",
        "clipping_only_limiter": "Clipping only",
        "becr_effective_signal_residual": "BECR effective",
        "wrong_pre_limiter_residual": "Wrong residual",
        "no_lower_bound_residual": "No-lower rho",
    }
    colors = {
        "projected_adam_baseline": "#666666",
        "fira_style_adam_limiter_no_residual": "#d62728",
        "clipping_only_limiter": "#1f77b4",
        "becr_effective_signal_residual": "#2ca02c",
        "wrong_pre_limiter_residual": "#9467bd",
        "no_lower_bound_residual": "#ff7f0e",
    }
    fig, axs = plt.subplots(2, 2, figsize=(13.6, 8.0))
    ax_tau, ax_z, ax_resid, ax_grad = axs.flatten()
    for method in order:
        if method not in method_payloads:
            continue
        series = method_payloads[method]["series"]
        xs = list(range(len(series["tau"])))
        ax_tau.plot(xs, series["tau"], label=labels[method], color=colors[method], linewidth=1.5)
        ax_resid.plot(xs, np.maximum(series["residual_norm"], 1e-300), label=labels[method], color=colors[method], linewidth=1.5)
        ax_grad.plot(xs, np.maximum(series["grad_norm"], 1e-300), label=labels[method], color=colors[method], linewidth=1.5)

    if "becr_effective_signal_residual" in method_payloads:
        becr = method_payloads["becr_effective_signal_residual"]["series"]
        xs = list(range(len(becr["Z_norm"])))
        ax_z.plot(xs, np.maximum(becr["Z_norm"], 1e-300), label="|Z_t|", color="#4c72b0", linewidth=1.6)
        ax_z.plot(xs, np.maximum(becr["Z_eff_norm"], 1e-300), label="|Z_t_eff|", color="#dd8452", linewidth=1.6)
        ax_z.plot(xs, np.maximum(becr["lost_raw_signal_norm"], 1e-300), label="|Z_t-Z_t_eff|", color="#55a868", linewidth=1.4, linestyle="--")

    ax_tau.set_title(f"(a) limiter tau, {setting_id}")
    ax_z.set_title("(b) raw vs effective transmitted signal")
    ax_resid.set_title("(c) residual norm")
    ax_grad.set_title("(d) gradient norm")
    ax_tau.set_xlabel("step")
    ax_z.set_xlabel("step")
    ax_resid.set_xlabel("step")
    ax_grad.set_xlabel("step")
    ax_tau.grid(True, alpha=0.3)
    for ax in [ax_z, ax_resid, ax_grad]:
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    ax_tau.legend(loc="best", fontsize=8)
    ax_z.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path_base = out_dir / "figure_candidate_5_adam_limiter_loss"
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)
    meta = _write_plot_metadata(
        path_base=path_base,
        title="Adam limiter-loss recovery diagnostic",
        aggregate_manifest_path=aggregate_manifest_path,
        aggregate_manifest=aggregate_manifest,
        candidate_level="appendix-candidate",
        main_or_appendix_recommendation="appendix-candidate",
    )
    return {"png": path_base.with_suffix(".png"), "pdf": path_base.with_suffix(".pdf"), "metadata": meta}
