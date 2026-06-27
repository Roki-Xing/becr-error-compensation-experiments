from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


DEFAULT_ARTIFACT_DIR = Path("artifacts/fira_parity/20260626T000000Z_p0_exact_fira_parity")
REQUIRED_FILES = (
    "manifest.json",
    "oracle_trace.jsonl",
    "candidate_trace.jsonl",
    "mismatch_report.json",
    "summary.json",
    "summary.md",
    "exact_svd_escape.png",
    "exact_svd_escape.pdf",
)


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_artifact_dir(path: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (path / name).exists()]
    if missing:
        raise SystemExit(f"artifact dir missing files: {missing}")


def smoke_generate(repo_root: Path, artifact_dir: Path) -> dict:
    fixture = _load_json(artifact_dir / "manifest.json")["fixture"]["fixture_id"]
    with tempfile.TemporaryDirectory(prefix="fira_parity_smoke_") as tmp:
        out_dir = Path(tmp) / "artifact"
        subprocess.check_call(
            [
                "python3",
                "-m",
                "fira_parity.runner",
                "--fixture",
                fixture,
                "--out",
                str(out_dir),
            ],
            cwd=repo_root,
        )
        check_artifact_dir(out_dir)
        generated_manifest = _load_json(out_dir / "manifest.json")
    return generated_manifest


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
        "committed_manifest_project_commit": committed_manifest["project_commit"],
        "committed_matches_head": committed_manifest["project_commit"] == head,
    }

    if args.smoke_generate:
        generated_manifest = smoke_generate(repo_root, artifact_dir)
        result["generated_manifest_project_commit"] = generated_manifest["project_commit"]
        if generated_manifest["project_commit"] != head:
            raise SystemExit(
                "smoke-generated manifest does not match checkout HEAD: "
                f"generated={generated_manifest['project_commit']} head={head}"
            )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
