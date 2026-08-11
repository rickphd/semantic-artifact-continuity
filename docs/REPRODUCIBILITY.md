# Reproducibility

## Levels

The package supports three distinct levels of verification.

1. **Stored-result verification** checks file integrity, counts, split support,
   evidence relations, and figure provenance without retraining.
2. **KE artifact regeneration** rebuilds RDF, SHACL reports, coverage, and
   traceability from the released dataset.
3. **Model reruns** retrain the downstream and ablation models. These are
   computationally heavier and may show small platform-dependent differences.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For LR/RF/XGB/CNN1D training:

```bash
python -m pip install -r requirements-training.txt
```

## Verify The Release

```bash
python scripts/verify_release.py
```

This is the required first command. It validates the active v04.2 chain and
checks every entry in `MANIFEST.sha256`.

## Regenerate KE Artifacts

Run from the repository root:

```bash
python scripts/experiments/materialize_reddit_rdf.py
python scripts/experiments/run_shacl_conformance.py
python scripts/experiments/analyze_ontology_coverage.py
python scripts/experiments/audit_traceability.py
python scripts/experiments/audit_model_inputs.py
python scripts/experiments/audit_model_input_cells.py
python scripts/experiments/reconcile_artifact_chain.py
```

Rebuild the result figures and their provenance manifest:

```bash
python scripts/figures/generate_results_plots.py
```

After intentional regeneration, refresh and verify the release manifest:

```bash
python scripts/release/generate_manifest.py
python scripts/verify_release.py
```

## Rerun Downstream Models

```bash
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/experiments/train_canonical_anova_revalidation.py --skip-cnn
python scripts/experiments/train_canonical_anova_revalidation.py --only-cnn --device cpu
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/experiments/run_module_ablation.py
python scripts/experiments/run_cnn1d_module_ablation.py --device cpu
```

Use `--quick` on the training and CNN1D scripts for smoke tests. Full CNN1D
reruns are substantially slower than stored-result verification.

## Rerun Lexical Sensitivity

```bash
python scripts/experiments/run_vader_ke_sensitivity.py
```

The complete 125-condition intermediate grid is intentionally not stored in
Git. The six aggregate outputs under
`results/sensitivity/` are distributed.

## Determinism And Scope

- The split is fixed by post ID.
- Lexical induction and ANOVA feature selection use training data only.
- Model seeds are 42, 123, and 2024.
- LR and CNN1D standardize semantic inputs with training-set statistics; RF and
  XGB use unscaled semantic inputs.
- BSL and ENR CNN1D runs use the same pooled-to-hidden-128-to-output
  classification head.
- Stored predictions and metrics are the authoritative released outputs.
- The package contains the active v04.2 lineage and its authoritative outputs.
- Model binaries are not distributed, except for the small LR semantic-input
  scaler used for matrix reconciliation.
