from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corrected_synthetic.runner import run_corrected_synthetic_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run corrected synthetic diagnostics under CRN provenance.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--paper-quality", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--overwrite-debug", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--parent-pr", type=str, default=None)
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--review-snapshot-dir", type=Path, default=None)
    args = parser.parse_args()
    command = "python " + " ".join([str(Path(__file__).as_posix()), *sys.argv[1:]])
    result = run_corrected_synthetic_suite(
        output_root=args.output_root,
        run_id=args.run_id,
        paper_quality=args.paper_quality,
        allow_dirty=args.allow_dirty,
        overwrite_debug=args.overwrite_debug,
        tiny=args.tiny,
        parent_pr=args.parent_pr,
        experiments=args.experiments,
        command=command,
        review_snapshot_dir=args.review_snapshot_dir,
    )
    print(f"group_dir={result['group_dir']}")
    print(f"artifact_index={result['artifact_index_path']}")
    for path in result["aggregate_manifest_paths"]:
        print(f"aggregate_manifest={path}")
    if "review_snapshot_dir" in result:
        print(f"review_snapshot_dir={result['review_snapshot_dir']}")


if __name__ == "__main__":
    main()
