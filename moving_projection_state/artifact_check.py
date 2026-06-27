from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


DEFAULT_ARTIFACT_DIR = Path(
    "experiments/tier1-synthetic/moving_projection_artifacts/20260627T000000Z_p0_moving_projection_state"
)
REQUIRED_FILES = (
    "manifest.json",
    "sample_trace.json",
    "suite_summary.json",
    "summary.md",
    "refresh_alignment.png",
    "refresh_alignment.pdf",
)


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _load_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(text, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))


def check_artifact_dir(path: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (path / name).exists()]
    if missing:
        raise SystemExit(f"artifact dir missing files: {missing}")
    _load_json(path / "manifest.json")
    _load_json(path / "sample_trace.json")
    _load_json(path / "suite_summary.json")


def smoke_generate(repo_root: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="moving_projection_smoke_") as tmp:
        out_root = Path(tmp)
        subprocess.check_call(
            [
                "python3",
                "experiments/tier1-synthetic/run_moving_projection_state_suite.py",
                "--output-root",
                str(out_root),
                "--run-id",
                "artifact_check_smoke",
                "--parent-pr",
                "#2",
            ],
            cwd=repo_root,
        )
        artifact_dir = out_root / "artifact_check_smoke"
        check_artifact_dir(artifact_dir)
        return _load_json(artifact_dir / "manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--smoke-generate", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    artifact_dir = (repo_root / args.artifact_dir).resolve()
    check_artifact_dir(artifact_dir)

    committed_manifest = _load_json(artifact_dir / "manifest.json")
    head = _git(["rev-parse", "HEAD"], repo_root)
    result = {
        "head": head,
        "committed_manifest_code_commit": committed_manifest["code_commit"],
        "committed_matches_head": committed_manifest["code_commit"] == head,
        "artifact_policy": "review_snapshot_with_executable_regeneration",
    }
    if args.smoke_generate:
        generated_manifest = smoke_generate(repo_root)
        result["generated_manifest_code_commit"] = generated_manifest["code_commit"]
        if generated_manifest["code_commit"] != head:
            raise SystemExit(
                "smoke-generated moving-projection manifest does not match checkout HEAD: "
                f"generated={generated_manifest['code_commit']} head={head}"
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
