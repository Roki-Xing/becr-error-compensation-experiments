This directory contains committed review snapshots for
`P0-MOVING-PROJECTION-STATE-002`.

Policy:

- Committed files under `20260627T000000Z_p0_moving_projection_state/` are
  review snapshots.
- They are not the sole source of truth for scientific claims.
- The source of truth is:
  `tests + CI regeneration + strict JSON trace inspection`.
- CI smoke generation must be able to regenerate a manifest whose
  `code_commit` equals the current checkout `HEAD`.
- The committed snapshot may record the semantic source commit used to generate
  it, which can differ from the final PR head because committing generated
  artifacts changes `HEAD`.

Checker:

```bash
python -m moving_projection_state.artifact_check \
  --artifact-dir experiments/tier1-synthetic/moving_projection_artifacts/20260627T000000Z_p0_moving_projection_state \
  --smoke-generate
```
