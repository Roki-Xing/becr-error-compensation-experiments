# Provenance and CRN Spec

This document defines the corrected stochastic synthetic provenance policy for
`P0-CRN-MANIFEST-003`.

## Scope

This infrastructure supports tiny corrected synthetic diagnostics only. It does
not validate:

- BECR optimizer superiority
- neural performance
- Adam convergence
- LDAdam-equivalent state compensation
- legacy result validity

## Why `ALL_RUNS.json` Is Not Source of Truth

The older neural scripts write `results/ALL_RUNS.json` by overwriting a single
aggregate file in-place. That pattern is not immutable, is easy to contaminate
with mixed reruns, and cannot support corrected paper claims.

Corrected stochastic aggregation therefore uses:

1. immutable per-run manifests
2. explicit run IDs
3. aggregate manifests built only from explicit run manifest paths

The corrected aggregate pipeline does not glob directories, does not read
`ALL_RUNS.json`, and rejects legacy schemas by default.

## Immutable Run ID Policy

Each corrected run uses a run ID derived from:

- task
- experiment
- method
- seed
- UTC timestamp
- short git commit
- deterministic config hash

The same output path cannot be reused unless `--overwrite-debug` is set. Debug
overwrites are not paper-quality runs.

## Strict JSON Policy

All corrected JSON and JSONL writers use strict serialization with
`allow_nan=False`. Non-finite values are sanitized to `null` before writing.

This applies to:

- `manifest.json`
- `metrics.json`
- `memory_runtime.json`
- `trace.json`
- `scale_log.jsonl`
- aggregate manifests and summaries

## Common Random Numbers Policy

Corrected stochastic synthetic comparisons use a pre-generated noise bank. For a
fixed seed, step schedule, and configuration:

- every method receives the same noise tensor
- method order must not change consumed noise
- the noise bank hash is recorded in every manifest

The provenance smoke runner uses the corrected moving-projection semantics from
PR #2 and records both `noise_hash` and method-specific `consumed_noise_hash`.

## Scheduler Factory Policy

Schedulers must not be shared across methods. Each method run gets:

- a distinct `scheduler_factory_id`
- a distinct `scheduler_object_id`

This prevents mutable scheduler state from leaking across paired stochastic
comparisons.

## Aggregation Policy

Corrected aggregates are constructed only from explicit run manifest paths. The
aggregate manifest records:

- source run IDs
- source manifest paths
- config hashes
- noise hashes
- old/legacy usage flags

Directory globbing and “latest file wins” aggregation are intentionally
disallowed.

## Old-Result Quarantine

Legacy outputs are not deleted, but they remain:

- deprecated
- diagnostic-only
- excluded from corrected aggregation

The corrected aggregate path rejects legacy schemas by default and does not use
legacy `results/` folders as data sources.

## Scale Logging Semantics

Corrected scale logs must distinguish:

- raw recovery scale
- applied recovery scale
- effective recovery scale
- raw transmitted residual norm
- effective transmitted residual norm

In the current corrected smoke harness:

- `phi_raw` / `s_raw` are the unclipped recovery-scale candidate
- `s_applied` is the post-clip scale
- `effective_recovery_scale` is `s_applied * tau`
- `effective_raw_transmission` is `||Z_eff|| / ||Q||` when defined

Even when numeric values coincide, the fields remain semantically distinct and
must not be conflated in downstream analysis.

## Memory and Runtime Schema

Every corrected run writes `memory_runtime.json` with a stable schema including:

- parameter bytes
- gradient buffer bytes
- first moment bytes
- second moment bytes
- projection bytes
- residual bytes
- temporary buffer estimate
- peak memory placeholders
- optimizer step time
- wall clock time
- device

For CPU smoke runs some peak-memory fields may remain `null`, but the fields
must exist. Residual bytes must always be present.

## Neural Rerun Policy

CIFAR / MNIST / WikiText / TinyGPT reruns must not begin until:

1. CRN and explicit aggregation pass CI
2. old-result quarantine is enforced
3. corrected stochastic synthetic reruns are attributable

This PR does not run any neural training and does not validate neural claims.

## Claim Discipline

What this infrastructure supports:

- future corrected synthetic reruns can be attributable and reproducible
- stochastic comparisons can use paired noise
- aggregates can be traced to explicit run IDs
- future memory claims can begin to be audited once residual bytes are
  populated by larger runs

What this infrastructure does not support:

- BECR superiority claims
- corrected neural performance claims
- optimizer convergence claims
- reuse of legacy results as corrected evidence
