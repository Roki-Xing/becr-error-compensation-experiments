from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
RES_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    exp1 = _load_json(RES_DIR / "exp1_diag_fixed_stale__seed0__summary.json")
    exp2 = _load_json(RES_DIR / "exp2_stale_refresh__sweep.json")
    exp3 = _load_json(RES_DIR / "exp3_aniso_noise_svdP__aggregate.json")

    order = ["proj_sgd", "fira_raw", "fira_clipped", "becr_fira", "rho_no_lower_bound", "coupled_unbounded"]
    label = {
        "proj_sgd": "ProjSGD",
        "fira_raw": "Fira-style coord (raw)",
        "fira_clipped": "Fira-style coord (clipped)",
        "becr_fira": "BECR",
        "rho_no_lower_bound": "No-lower (capped)",
        "coupled_unbounded": "Coupled (unbounded)",
    }
    color = {
        "proj_sgd": "#666666",
        "fira_raw": "#d62728",
        "fira_clipped": "#1f77b4",
        "becr_fira": "#2ca02c",
        "rho_no_lower_bound": "#9467bd",
        "coupled_unbounded": "#ff7f0e",
    }

    fig, axs = plt.subplots(1, 3, figsize=(15.8, 4.6))
    axA, axB, axC = axs[0], axs[1], axs[2]

    # Panel A: fixed stale subspace (seed=0).
    x = np.arange(len(order))
    gn = np.array([float(exp1["optimizers"][m]["grad_norm_final"]) for m in order], dtype=float)
    en = np.array(
        [
            float(exp1["optimizers"][m]["residual_norm_final"])
            if exp1["optimizers"][m]["residual_norm_final"] is not None
            else float("nan")
            for m in order
        ],
        dtype=float,
    )

    axA.bar(x, gn, color=[color[m] for m in order], alpha=0.9, label="final ||grad||")
    # Overlay residual norms (where available) as hatched bars (offset slightly).
    w = 0.35
    for i, m in enumerate(order):
        if not math.isfinite(float(en[i])):
            continue
        axA.bar(float(x[i]) + w * 0.55, float(en[i]), width=w, color="#999999", hatch="///", alpha=0.8, label=None)

    axA.set_yscale("log")
    axA.set_xticks(x, [label[m] for m in order], rotation=28, ha="right")
    axA.set_ylabel("final magnitude (log)")
    axA.set_title("(A) Fixed stale subspace")
    axA.grid(True, axis="y", alpha=0.3)
    axA.text(0.02, 0.98, "bars: ||grad||\nhatched: ||e|| (if any)", transform=axA.transAxes, va="top", ha="left", fontsize=8)

    # Panel B: stale refresh sweep (K).
    K = [1e6 if r["K"] == "inf" else float(r["K"]) for r in exp2]
    methods_B = ["proj_sgd", "fira_raw", "fira_clipped", "becr_fira", "rho_no_lower_bound"]
    for m in methods_B:
        y = [float(r.get(f"{m}__grad_norm_final", float("nan"))) for r in exp2]
        axB.plot(K, y, marker="o", linewidth=1.6, color=color.get(m, "#333333"), label=label.get(m, m))
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlabel("refresh interval K (inf shown as 1e6)")
    axB.set_ylabel("final ||grad|| (log)")
    axB.set_title("(B) Stale refresh sweep")
    axB.grid(True, alpha=0.3)
    axB.legend(loc="best", fontsize=8)

    # Panel C: anisotropic noise / SVD-biased projection (mean±std over 5 seeds).
    methods_C = order
    means = [float(exp3["methods"][m]["grad_norm_final_mean"]) for m in methods_C]
    stds = [float(exp3["methods"][m]["grad_norm_final_std"]) for m in methods_C]
    xs = np.arange(len(methods_C))
    axC.bar(xs, means, yerr=stds, capsize=3, color=[color[m] for m in methods_C], alpha=0.9)
    axC.set_yscale("log")
    axC.set_xticks(xs, [label[m] for m in methods_C], rotation=28, ha="right")
    axC.set_ylabel("final ||grad|| (log)")
    axC.set_title("(C) Anisotropic noise (mean±std)")
    axC.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Tier-1 high-dimensional synthetic diagnostics")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG_DIR / "fig2_tier1_synthetic.png", dpi=240)
    fig.savefig(FIG_DIR / "fig2_tier1_synthetic.pdf")
    plt.close(fig)

    print("Wrote:", str(FIG_DIR / "fig2_tier1_synthetic.png"))


if __name__ == "__main__":
    main()
