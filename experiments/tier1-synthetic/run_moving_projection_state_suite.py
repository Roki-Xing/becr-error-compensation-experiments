from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moving_projection_state.runner import generate_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "moving_projection_artifacts",
        help="Directory where artifact runs are written.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="20260627T000000Z_p0_moving_projection_state",
        help="Stable run identifier for committed review artifacts.",
    )
    parser.add_argument(
        "--parent-task-id",
        type=str,
        default="P0-MOVING-PROJECTION-STATE-002",
        help="Parent task id recorded in manifests.",
    )
    args = parser.parse_args()
    out_dir = generate_artifacts(
        output_root=args.output_root,
        run_id=args.run_id,
        parent_task_id=args.parent_task_id,
    )
    print(f"Wrote moving-projection artifacts to {out_dir}")


if __name__ == "__main__":
    main()
