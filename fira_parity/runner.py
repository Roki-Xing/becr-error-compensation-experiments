from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .common import config_to_dict, repo_manifest, trace_to_jsonable, write_json, write_jsonl
from .compare import compare_traces
from .exact_candidate import ExactFiraReference
from .fixtures import parity_fixtures
from .official_oracle import OfficialFiraOracle
from .upstream_metadata import UPSTREAM_COMMIT, UPSTREAM_LICENSE, UPSTREAM_REPOSITORY


def run_fixture(fixture_id: str) -> dict[str, Any]:
    fixtures = parity_fixtures()
    config, param_init, grad_seq = fixtures[fixture_id]
    oracle = OfficialFiraOracle(param_init, config)
    candidate = ExactFiraReference(param_init, config)
    oracle_traces = []
    candidate_traces = []
    for grad in grad_seq:
        oracle_traces.append(oracle.step_and_trace(grad))
        candidate_traces.append(candidate.step_and_trace(grad))
    tol = (1e-10, 1e-10) if config.dtype == "float64" else (1e-5, 1e-5)
    comparison = compare_traces(oracle_traces, candidate_traces, atol=tol[0], rtol=tol[1])
    return {
        "config": config,
        "oracle_traces": oracle_traces,
        "candidate_traces": candidate_traces,
        "comparison": comparison,
    }


def _save_escape_diagnostic(out_dir: Path, traces: list[dict[str, Any]]) -> None:
    steps = [row["step_index"] for row in traces]
    grad_norm = [float(torch.norm(row["raw_gradient"]).item()) for row in traces]
    remainder_norm = [float(torch.norm(row["remainder_gradient"]).item()) for row in traces]
    refresh = [row["refresh_happened"] for row in traces]
    alignment = []
    for row in traces:
        grad = row["raw_gradient"]
        projector = row["invariant_projector_after"]
        if row["projection_orientation"] == "right":
            aligned = grad @ projector
        else:
            aligned = projector @ grad
        denom = max(float(torch.norm(grad).item()), 1e-30)
        alignment.append(float(torch.norm(aligned).item()) / denom)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(steps, grad_norm, label="||G_t||")
    axes[0].plot(steps, remainder_norm, label="||G_t^perp||")
    axes[0].set_title("Gradient Norms")
    axes[0].legend()

    axes[1].plot(steps, alignment, label="projector-grad alignment")
    axes[1].set_title("Projector Alignment")
    axes[1].set_ylim(0.0, 1.05)

    axes[2].plot(steps, [1.0 if flag else 0.0 for flag in refresh], label="refresh")
    axes[2].set_title("Refresh Events")
    axes[2].set_ylim(-0.05, 1.05)

    for ax in axes:
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_dir / "exact_svd_escape.png", dpi=200)
    fig.savefig(out_dir / "exact_svd_escape.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="shape_4x3_rank2_gap5_fp64_100step")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_fixture(args.fixture)
    config = result["config"]
    oracle_traces = result["oracle_traces"]
    candidate_traces = result["candidate_traces"]
    comparison = result["comparison"]

    manifest = repo_manifest(repo_root)
    manifest.update(
        {
            "task_id": "P0-EXACT-FIRA-PARITY-001",
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "fixture": config_to_dict(config),
            "official_fira_repository": UPSTREAM_REPOSITORY,
            "official_fira_commit": UPSTREAM_COMMIT,
            "official_fira_license": UPSTREAM_LICENSE,
            "command": f"python -m fira_parity.runner --fixture {args.fixture} --out {args.out}",
        }
    )

    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "oracle_trace.jsonl", [trace_to_jsonable(row) for row in oracle_traces])
    write_jsonl(out_dir / "candidate_trace.jsonl", [trace_to_jsonable(row) for row in candidate_traces])
    write_json(out_dir / "mismatch_report.json", comparison)

    summary = {
        "fixture_id": config.fixture_id,
        "dtype": config.dtype,
        "steps": config.steps,
        "refresh_steps": [row["step_index"] for row in oracle_traces if row["refresh_happened"]],
        "first_mismatch": comparison["first_mismatch"] or "none",
        "max_errors": comparison["max_errors"],
    }
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# Fira Parity Summary: {config.fixture_id}",
                "",
                f"- official commit: `{UPSTREAM_COMMIT}`",
                f"- dtype: `{config.dtype}`",
                f"- steps: `{config.steps}`",
                f"- refresh steps: `{summary['refresh_steps']}`",
                f"- first mismatch: `{summary['first_mismatch']}`",
            ]
        ),
        encoding="utf-8",
    )

    # Low-cost diagnostic: exact rank-one SVD on a one-column matrix.
    escape_result = run_fixture("shape_2x1_rank1_gap1_fp64")
    _save_escape_diagnostic(out_dir, escape_result["oracle_traces"])


if __name__ == "__main__":
    main()
