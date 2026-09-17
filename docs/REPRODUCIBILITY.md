# Reproducibility

## Verify Before Rerunning

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python -m unittest discover -s tests
```

The computational verifier is read-only. It recalculates metrics from
62,016 predictions, checks fixed IDs/labels/order, the six-variable interface,
27 captured runtime profiles, canonical RDF cardinality and the 125-setting
sensitivity grid. File integrity alone does not validate scientific meaning.

Frozen environment records are under `results/provenance/canonical/` and
`results/provenance/campaign/`. They record Python 3.13 and the original package
versions. The root requirements give installation ranges; they do not promise
cross-version bitwise identity. Training additionally requires
`requirements-training.txt`. Joblib scalers should be loaded only from a trusted,
hash-verified checkout with compatible scikit-learn versions.

## Reproduce In A Separate Checkout

For a bounded six-stage canonical replay with exact numeric comparisons,
RDF isomorphism and all nine prepared matrices checked, use:

```bash
python scripts/replay_canonical.py --output-dir /tmp/semantic-canonical-replay
```

This output directory must not already exist and must be outside the checkout.

The original producer scripts write repository-relative outputs. Do not run
them over the preserved evidence. Create a disposable clone first:

```bash
git clone --branch v1.1.1 https://github.com/rickphd/semantic-artifact-continuity.git /tmp/semantic-artifact-replay
cd /tmp/semantic-artifact-replay
python scripts/experiments/build_train_only_ontology_dataset.py
python scripts/experiments/select_ont_features_anova_train_only.py
python scripts/experiments/materialize_reddit_rdf.py
python scripts/experiments/run_shacl_conformance.py
python scripts/experiments/analyze_ontology_coverage.py
python scripts/experiments/audit_traceability.py
python scripts/experiments/audit_model_inputs.py
python scripts/experiments/audit_model_input_cells.py
python scripts/experiments/reconcile_artifact_chain.py
```

The minimized input is `inputs/original_gold.parquet`. Induction uses only the
968 training texts, without their labels; ANOVA uses training labels. A fresh
run can differ in timestamps, RDF serialization, library metadata or model
optimization while preserving substantive values. Compare values and graph
isomorphism as appropriate. Do not regenerate `MANIFEST.sha256` merely to turn
a failed reproduction into a passing check: it authenticates released bytes.

## Controlled Tests And Human Assessment

The READMEs in `experiments/controlled_discontinuities/` and
`experiments/human_assessment/` provide independent portable replay commands
with explicit scratch output directories. E1 distinguishes original from
supplementary checks and operational errors from expected detections. E2
recomputes the reference and extraction comparison from anonymous structured
decisions; omitted private prose cannot be reconstructed from this release.

## Training And Sensitivity

Run these commands only in the disposable checkout after installing training
dependencies. They were not rerun as part of publishing v1.1.0:

```bash
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/experiments/train_canonical_anova_revalidation.py --skip-cnn
python scripts/experiments/train_canonical_anova_revalidation.py --only-cnn --device cpu
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/experiments/run_module_ablation.py
python scripts/experiments/run_cnn1d_module_ablation.py --device cpu
python scripts/experiments/run_vader_ke_sensitivity.py --grid expanded
```

The fixed seeds are 42, 123 and 2024. LR/CNN1D use train-standardized numeric
inputs; RF/XGB use raw values. The CNN1D head is matched between BSL and ENR.
Stored sensitivity results comprise 97 admissible configurations and 28 loader
rejections. A rejection means no evaluated numeric/RDF/SHACL output, even if a
historical summary contains a default `False` in its conformance column.

Trained classifier weights are not released. The LR input scaler and the
prepared train-only scaler are included for input reconciliation. Predictions,
metrics, configurations and training code are distributed; this is not a
weight-complete model archive.

## Figures

```bash
python scripts/figures/regenerate_release_figures.py --output-dir /tmp/semantic-figures
```

This replays Figures 5-11 without modifying released images. Figure 6 displays
only admissible settings. Figure 4 is exported separately from its Draw.io
source. Historical figure hashes and source-to-public mappings are retained;
font and renderer versions can affect output bytes.

## Interpretation Limits

SHACL tests structural constraints; E1 tests declared correspondences under
specified mutations; E2 tests concept presence/local polarity. E2 does not
validate the global sentiment labels, whose detailed original annotation
protocol has not been recovered. Adjudications are author-assisted decisions,
not a third independent annotator. Three-seed predictive comparisons use one
fixed corpus/split and do not establish cross-corpus generalization.
