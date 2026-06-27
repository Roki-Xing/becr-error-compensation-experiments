from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moving_projection_state.runner import generate_artifacts


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    out_dir = generate_artifacts(
        output_root=base_dir / "moving_projection_artifacts",
        run_id="20260627T000000Z_p0_moving_projection_state",
    )
    print(f"Wrote moving-projection artifacts to {out_dir}")


if __name__ == "__main__":
    main()
