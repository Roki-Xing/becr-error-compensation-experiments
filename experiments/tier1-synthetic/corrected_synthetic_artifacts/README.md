# Corrected Synthetic Artifacts

This directory holds reviewer-facing artifacts for `P1-CORRECTED-SYNTHETIC-RERUNS`.

Policy:

- Committed artifacts under `review_snapshot/` are reviewer-convenience snapshots.
- The source of truth remains reproducible generation from current code, strict JSON validation, and CI regeneration.
- CI runs a tiny paper-quality corrected synthetic suite and uploads it as the `corrected-synthetic-tiny-review` GitHub Actions artifact.
- Local full paper-quality runs may be exported into `review_snapshot/` for inspection, but paper claims should not rely on stale unchecked snapshots alone.

Regenerate the full local paper-quality suite:

```bash
python experiments/tier1-synthetic/run_corrected_synthetic_suite.py \
  --output-root /tmp/becr_p1_corrected_synthetic \
  --run-id p1_corrected_synthetic_paper \
  --paper-quality \
  --parent-pr "#4"
```

Export a review snapshot from a generated run:

```bash
python experiments/tier1-synthetic/run_corrected_synthetic_suite.py \
  --output-root /tmp/becr_p1_corrected_synthetic \
  --run-id p1_corrected_synthetic_paper \
  --paper-quality \
  --parent-pr "#4" \
  --review-snapshot-dir experiments/tier1-synthetic/corrected_synthetic_artifacts/review_snapshot/p1_corrected_synthetic_paper
```
