# Corrected Synthetic Review Snapshot

- This directory is a reviewer-convenience snapshot, not the sole source of truth.
- Semantic source commit: `ac56a7c18afd4d91e2b63feeac9de4a14571cc82`.
- Source of truth remains reproducible generation from current code plus strict JSON validation and CI regeneration.
- Regenerate the full local paper-quality suite with:
  `python /mnt/c/Users/Xing/Desktop/文件夹/becr-error-compensation-experiments/experiments/tier1-synthetic/run_corrected_synthetic_suite.py --output-root /tmp/becr_p1_corrected_synthetic_fix --run-id p1_corrected_synthetic_paper_fix --paper-quality --parent-pr #4 --review-snapshot-dir experiments/tier1-synthetic/corrected_synthetic_artifacts/review_snapshot/p1_corrected_synthetic_paper_ac56a7c`
- CI generates a tiny paper-quality corrected synthetic artifact separately for reproducibility checks.
- No paper claim should rely on stale or unchecked artifacts alone.
