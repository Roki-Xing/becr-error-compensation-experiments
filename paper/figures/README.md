# Paper Figure Provenance

This directory contains paper-packaged figures copied from the accepted PR #4
corrected synthetic review snapshot:

`experiments/tier1-synthetic/corrected_synthetic_artifacts/review_snapshot/p1_corrected_synthetic_paper_ac56a7c/`

## figure1_theorem_regime.png

- Source artifact path:
  `.../theorem_regime/figures/figure_candidate_1_theorem_regime.png`
- Source aggregate manifest:
  `.../theorem_regime/aggregates/p1_corrected_synthetic_paper_fix_theorem_regime_aggregate/aggregate_manifest.json`
- Source metadata sidecar:
  `.../theorem_regime/figures/figure_candidate_1_theorem_regime.metadata.json`
- Source run IDs:
  six theorem-regime corrected runs covering projected baseline, raw recovery,
  clipped recovery, BECR, no-lower-bound rho, and coupled rho=phi.
- Claim supported:
  positive-epsilon stale-subspace failure with finite cumulative raw recovery
  mass; clipping and BECR repair the theorem regime.
- Claim not supported:
  broad official-Fira nonconvergence; neural performance; optimizer superiority.

## figure2_highdim_refresh.png

- Packaged from two accepted snapshot figures:
  `.../high_dimensional_fixed/figures/figure_candidate_2_high_dimensional.png`
  and
  `.../refresh_sweep/figures/figure_candidate_3_refresh_sweep.png`
- Source aggregate manifests:
  `.../high_dimensional_fixed/aggregates/p1_corrected_synthetic_paper_fix_high_dimensional_fixed_aggregate/aggregate_manifest.json`
  and
  `.../refresh_sweep/aggregates/p1_corrected_synthetic_paper_fix_refresh_sweep_aggregate/aggregate_manifest.json`
- Source metadata sidecars:
  `.../high_dimensional_fixed/figures/figure_candidate_2_high_dimensional.metadata.json`
  and
  `.../refresh_sweep/figures/figure_candidate_3_refresh_sweep.metadata.json`
- Source run IDs:
  explicit run IDs are listed in the two metadata sidecars and aggregate
  manifests.
- Claim supported:
  the orthogonal stale channel persists beyond 2D; clipping and BECR repair
  that orthogonal channel; refresh frequency is a boundary condition.
- Claim not supported:
  universal full stationarity; optimizer superiority; neural transfer.

## figure3_anisotropic_noise.png

- Source artifact path:
  `.../anisotropic_noise/figures/figure_candidate_4_anisotropic_noise.png`
- Source aggregate manifest:
  `.../anisotropic_noise/aggregates/p1_corrected_synthetic_paper_fix_anisotropic_noise_aggregate/aggregate_manifest.json`
- Source metadata sidecar:
  `.../anisotropic_noise/figures/figure_candidate_4_anisotropic_noise.metadata.json`
- Source run IDs:
  five-seed CRN corrected synthetic runs listed in the metadata sidecar.
- Claim supported:
  paired noisy-projection diagnostics are reproducible under CRN and must use
  claim text consistent with aggregate booleans.
- Claim not supported:
  broad projected-baseline superiority, neural robustness, or optimizer
  superiority.
