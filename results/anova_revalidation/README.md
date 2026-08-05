# Downstream ANOVA Revalidation

This directory contains the CPU execution family used for the downstream
plots and result tables. All outputs were generated from the canonical v04.2
Gold-enriched input, the fixed ID split, the train-only ANOVA selection, and
the v04.2 train-scope lexicon.

## Execution

- Models: LR, RF, XGB, and CNN1D
- Conditions: BSL and ENR
- Seeds: 42, 123, and 2024
- Test support: 48 negative, 119 neutral, 156 positive
- CNN1D device: CPU for all six runs
- CNN1D head: pooled text plus optional semantic inputs, hidden 128, output 3
- Semantic profiles: LR and CNN1D train-standardized; RF and XGB unscaled
- XGB stability setting: `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`
- Execution device: CPU for the authoritative released runs

## Complete output check

- 24 metric JSON files: 8 model-condition groups x 3 seeds
- 24 test-prediction CSV files
- 24 rows in `metrics/multiseed_raw_v2.csv`
- 3 seeds in every model-condition group
- One dataset SHA-256 across all metric files:
  `62c6394d4ef68191b713d80c9b54849ecf00176822719bf515b8f0ee65b31162`
- One lexicon-manifest SHA-256 across all metric files:
  `313af4e18f4d7b58f87e87a7ac5bbbb19c32d5e7c829816bdad9fa6515e51e77`

## Summary-file hashes

- `metrics/multiseed_summary_v2.csv`:
  `f8a27e7ae736f92e044d82a0a1e3507e83ca292150d7f7576f2f806d1da9b351`
- `metrics/multiseed_raw_v2.csv`:
  `3ac46c26474a67e1ed478369f31e141b27fbeb24585ce90466e319fd691b41fc`

These files are the canonical released downstream outputs.
