from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
RES_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _arr(traces: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(t.get(key, float("nan"))) for t in traces], dtype=float)


def _last_per_layer(traces: list[dict]) -> dict | None:
    for t in reversed(traces):
        pl = t.get("per_layer", None)
        if isinstance(pl, dict) and pl:
            return pl
    return None


def main() -> None:
    runs = json.loads((RES_DIR / "ALL_RUNS.json").read_text(encoding="utf-8"))
    for r in runs:
        if r.get("mode") == "residual_no_bound":
            r["mode"] = "coupled_unbounded"
    order = ["adamw_full", "proj_only", "fira_raw", "fira_clipped", "becr", "rho_no_lower_bound", "coupled_unbounded"]

    label = {
        "adamw_full": "AdamW (full)",
        "proj_only": "Proj-only",
        "fira_raw": "Fira (raw)",
        "fira_clipped": "Fira (clipped)",
        "becr": "BECR",
        "rho_no_lower_bound": "Residual (no-lower, capped)",
        "coupled_unbounded": "Coupled EF (rho=phi_raw)",
    }
    color = {
        "adamw_full": "#111111",
        "proj_only": "#666666",
        "fira_raw": "#d62728",
        "fira_clipped": "#1f77b4",
        "becr": "#2ca02c",
        "rho_no_lower_bound": "#9467bd",
        "coupled_unbounded": "#ff7f0e",
    }

    fig, axs = plt.subplots(2, 3, figsize=(15.5, 7.8))
    ax_loss, ax_gn, ax_gperp = axs[0, 0], axs[0, 1], axs[0, 2]
    ax_scale, ax_e, ax_acc = axs[1, 0], axs[1, 1], axs[1, 2]

    # Plot only seed=0 curves for readability.
    for mode in order:
        run0 = next((r for r in runs if r["mode"] == mode and int(r["seed"]) == 0), None)
        if run0 is None:
            continue
        tr = run0["traces"]
        x = _arr(tr, "step")
        ax_loss.plot(x, _arr(tr, "loss"), linewidth=1.6, color=color.get(mode), label=label.get(mode, mode))

        gn = _arr(tr, "grad_norm")
        if np.isfinite(gn).any():
            ax_gn.plot(x, gn, linewidth=1.5, color=color.get(mode))

        gp = _arr(tr, "grad_perp_norm")
        if np.isfinite(gp).any():
            ax_gperp.plot(x, gp, linewidth=1.5, color=color.get(mode))

        phi_p95 = _arr(tr, "phi_raw_p95")
        s_p95 = _arr(tr, "s_p95")
        if np.isfinite(phi_p95).any():
            ax_scale.plot(x, phi_p95, linewidth=1.3, color=color.get(mode), linestyle="--")
        if np.isfinite(s_p95).any():
            ax_scale.plot(x, s_p95, linewidth=1.6, color=color.get(mode))

        en = _arr(tr, "residual_norm")
        if np.isfinite(en).any():
            ax_e.plot(x, en, linewidth=1.5, color=color.get(mode))

    ax_loss.set_title("(a) Train loss (seed=0)")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("CE loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="best", fontsize=9)

    ax_gn.set_title("(b) ||grad|| (logged, diagnostic)")
    ax_gn.set_xlabel("step")
    ax_gn.set_ylabel("||g||")
    ax_gn.set_yscale("log")
    ax_gn.grid(True, alpha=0.3)

    ax_gperp.set_title("(c) ||g_perp|| (logged, diagnostic)")
    ax_gperp.set_xlabel("step")
    ax_gperp.set_ylabel("||g_perp||")
    ax_gperp.set_yscale("log")
    ax_gperp.grid(True, alpha=0.3)

    ax_scale.set_title("(d) Scale p95: s_t (solid) vs phi_raw (dashed)")
    ax_scale.set_xlabel("step")
    ax_scale.set_ylabel("scale")
    ax_scale.set_yscale("log")
    ax_scale.grid(True, alpha=0.3)

    ax_e.set_title("(e) Residual norm ||e|| (BECR variants)")
    ax_e.set_xlabel("step")
    ax_e.set_ylabel("||e||")
    ax_e.set_yscale("log")
    ax_e.grid(True, alpha=0.3)

    # Final test acc aggregated over seeds.
    by_mode: dict[str, list[float]] = {}
    for r in runs:
        by_mode.setdefault(r["mode"], []).append(float(r["test_acc"]))
    modes = [m for m in order if m in by_mode]
    means = [float(np.mean(by_mode[m])) for m in modes]
    stds = [float(np.std(by_mode[m], ddof=1)) if len(by_mode[m]) > 1 else 0.0 for m in modes]
    xs = np.arange(len(modes))
    ax_acc.bar(xs, means, yerr=stds, capsize=4, color=[color.get(m, "#999999") for m in modes], alpha=0.9)
    ax_acc.set_xticks(xs, [label.get(m, m) for m in modes], rotation=25, ha="right")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.set_title("(f) Final test accuracy (mean±std over seeds)")
    ax_acc.grid(True, axis="y", alpha=0.3)

    fig.suptitle("MNIST MLP (mechanism diagnostic, 1000 steps)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "mnist_coordproj__MAIN.png", dpi=240)
    fig.savefig(FIG_DIR / "mnist_coordproj__MAIN.pdf")
    plt.close(fig)

    print("Wrote:", str(FIG_DIR / "mnist_coordproj__MAIN.png"))

    # Per-layer diagnostic heatmaps (requires per_layer logs from the training script).
    modes_pl = ["proj_only", "fira_raw", "fira_clipped", "becr", "rho_no_lower_bound", "coupled_unbounded"]
    runs0 = {m: next((r for r in runs if r["mode"] == m and int(r.get("seed", 0)) == 0), None) for m in modes_pl}
    per0 = {m: (_last_per_layer(r["traces"]) if r is not None else None) for m, r in runs0.items()}
    per0 = {m: v for m, v in per0.items() if isinstance(v, dict) and v}
    if not per0:
        return

    layer_names = sorted({ln for d in per0.values() for ln in d.keys()})
    mode_names = [m for m in modes_pl if m in per0]

    def _mat(metric: str) -> np.ndarray:
        a = np.full((len(layer_names), len(mode_names)), np.nan, dtype=float)
        for j, m in enumerate(mode_names):
            d = per0[m]
            for i, ln in enumerate(layer_names):
                v = d.get(ln, {}).get(metric, float("nan"))
                try:
                    a[i, j] = float(v)
                except Exception:
                    a[i, j] = np.nan
        return a

    gperp_ratio = _mat("g_perp_ratio")
    s_p95 = _mat("s_p95")
    res_norm = _mat("residual_norm")

    fig2, axs2 = plt.subplots(1, 3, figsize=(15.8, 5.2))
    ax0, ax1, ax2 = axs2[0], axs2[1], axs2[2]

    im0 = ax0.imshow(np.ma.masked_invalid(gperp_ratio), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax0.set_title("(a) g_perp_ratio (final trace, seed=0)")
    ax0.set_xticks(np.arange(len(mode_names)), [label.get(m, m) for m in mode_names], rotation=25, ha="right")
    ax0.set_yticks(np.arange(len(layer_names)), layer_names)
    fig2.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    im1 = ax1.imshow(np.ma.masked_invalid(np.log10(np.maximum(s_p95, 1e-12))), aspect="auto", cmap="coolwarm")
    ax1.set_title("(b) log10(s_p95) (final trace, seed=0)")
    ax1.set_xticks(np.arange(len(mode_names)), [label.get(m, m) for m in mode_names], rotation=25, ha="right")
    ax1.set_yticks(np.arange(len(layer_names)), [""] * len(layer_names))
    fig2.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(np.ma.masked_invalid(np.log10(np.maximum(res_norm, 1e-30))), aspect="auto", cmap="magma")
    ax2.set_title("(c) log10(residual_norm) (final trace, seed=0)")
    ax2.set_xticks(np.arange(len(mode_names)), [label.get(m, m) for m in mode_names], rotation=25, ha="right")
    ax2.set_yticks(np.arange(len(layer_names)), [""] * len(layer_names))
    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig2.suptitle("MNIST per-layer mechanism diagnostics (weights only)")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(FIG_DIR / "mnist_coordproj__PER_LAYER.png", dpi=240)
    fig2.savefig(FIG_DIR / "mnist_coordproj__PER_LAYER.pdf")
    plt.close(fig2)

    print("Wrote:", str(FIG_DIR / "mnist_coordproj__PER_LAYER.png"))


if __name__ == "__main__":
    main()
