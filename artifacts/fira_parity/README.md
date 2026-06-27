# Fira Parity Artifacts

This directory contains committed parity-review artifacts for
`P0-EXACT-FIRA-PARITY-001`.

## Why keep `exact_svd_escape.{png,pdf}` committed?

They are retained for review hygiene:
- the PR includes a human-readable diagnostic without requiring local reruns
- the figure is lightweight and deterministic
- it is produced by the same committed parity runner used for the JSON artifacts

## Commit semantics

`manifest.json.project_commit` records the source commit used to generate the
artifact snapshot.

Because Git commits are content-addressed, a committed file cannot practically
self-reference the final commit hash that contains that file. The enforceable
hygiene rule is therefore:

1. committed artifacts are refreshed from the latest semantic source commit
2. CI smoke-generates fresh artifacts on the checkout HEAD
3. the generated smoke manifest must report the checkout HEAD commit

This keeps the committed review artifacts traceable while still validating that
the current PR head can regenerate them.
