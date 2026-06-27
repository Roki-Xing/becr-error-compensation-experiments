from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_provenance.smoke import run_provenance_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiny stochastic provenance smoke diagnostics.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--parent-pr", type=str, default=None)
    parser.add_argument("--paper-quality", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--overwrite-debug", action="store_true")
    args = parser.parse_args()

    result = run_provenance_smoke(
        output_root=args.output_root,
        run_id=args.run_id,
        parent_pr=args.parent_pr,
        paper_quality=args.paper_quality,
        allow_dirty=args.allow_dirty,
        overwrite_debug=args.overwrite_debug,
    )
    print(f"group_dir={result['group_dir']}")
    print(f"aggregate_manifest={result['aggregate_manifest_path']}")


if __name__ == "__main__":
    main()
